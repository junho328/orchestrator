import inspect
import os
import re
import textwrap
from collections import defaultdict, deque
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Union, List
from contextlib import nullcontext

import transformers
from accelerate import logging
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    is_wandb_available,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_flash_attn_2_available, is_peft_available, is_rich_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template, prepare_multimodal_messages
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.extras.vllm_client import VLLMClient
# Liger torch.compile guard crashes can happen if not disabled before import.

try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

import torch

from trl.import_utils import is_liger_kernel_available, is_vllm_available
from trl.models import prepare_deepspeed, prepare_fsdp, prepare_peft_model, unwrap_model_for_generation
from trl.models.utils import _ForwardRedirection
from trl.trainer.base_trainer import BaseTrainer
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import (
    RepeatSampler,
    disable_dropout_in_model,
    ensure_master_addr_port,
    entropy_from_logits,
    identity,
    nanmax,
    nanmin,
    nanstd,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
    shuffle_sequence_dict,
    split_pixel_values_by_grid,
    split_tensor_dict,
    unsplit_pixel_values_by_grid,
)

# User defined prompts and rewards
from davids.train.utils.pubmdp_prompt import PUBLIC_PROMPT, PRIVATE_PROMPT, PUBLIC_SYSTEM_PROMPT, PRIVATE_SYSTEM_PROMPT
from davids.reward_utils.math_grader import answer_tag_reward_fn
from davids.reward_utils.think_answer_format_reward import think_answer_format_reward

from peft import PeftConfig, PeftModel

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_wandb_available():
    import wandb

logger = logging.get_logger(__name__)

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

# Distributed + repeat sampler: sharded indices per rank, then repeat like RepeatSampler.
class DistributedRepeatSampler(DistributedSampler):
    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        mini_repeat_count: int = 1,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count

    def __iter__(self):
        # base_indices are already sharded to this rank by DistributedSampler
        base_indices = list(super().__iter__())
        batches = [base_indices[i : i + self.batch_size] for i in range(0, len(base_indices), self.batch_size)]
        batches = [b for b in batches if len(b) == self.batch_size]  # drop incomplete to keep shapes consistent

        for batch in batches:
            for _ in range(self.repeat_count):
                for idx in batch:
                    for _ in range(self.mini_repeat_count):
                        yield idx

    def __len__(self) -> int:
        base_len = super().__len__()  # number of items for this rank (after padding)
        full_batches = base_len // self.batch_size
        return full_batches * self.batch_size * self.mini_repeat_count * self.repeat_count

logger = logging.get_logger(__name__)

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class PUBPRIGRPOTrainer(BaseTrainer):
    """
    Trainer for the Public-Private Multi-Agent GRPO method.
    Optimized for DDP/Accelerate (avoiding DeepSpeed for multi-adapter switching in one step).
    """

    _tag_names = ["trl", "grpo"]
    _name = "GRPO"

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args initialization (Same as original GRPOTrainer)
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            dtype = model_init_kwargs.get("dtype")
            if isinstance(dtype, torch.dtype) or dtype == "auto" or dtype is None:
                pass  # dtype is already a torch.dtype or "auto" or None
            elif isinstance(dtype, str):  # it's a str, but not "auto"
                dtype = getattr(torch, dtype)
                model_init_kwargs["dtype"] = dtype
            else:
                raise ValueError(
                    "Invalid `dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            config = AutoConfig.from_pretrained(model_id)
            if getattr(args, "gradient_checkpointing", False):
                config.use_cache = False
            architecture = getattr(transformers, config.architectures[0])
            model = architecture.from_pretrained(model_id, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if getattr(args, "gradient_checkpointing", False):
                model.config.use_cache = False
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )

        # Some models (SmolVLM/Idefics3) don't support `logits_to_keep` argument and error out if we pass it
        # Inspect the forward method before we wrap the model with PEFT
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        if peft_config is not None or (is_peft_available() and isinstance(model, PeftModel)):
            model = prepare_peft_model(model, peft_config, args)

        # Ensure both public/private adapters exist on every rank (DDP requires identical param sets)
        if is_peft_available() and isinstance(model, PeftModel):
            required_adapters = ("public", "private")
            missing = [name for name in required_adapters if name not in getattr(model, "peft_config", {})]
            if missing:
                # Reuse the first available adapter config to materialize the missing ones
                base_cfg = next(iter(model.peft_config.values()))
                for name in missing:
                    model.add_adapter(name, base_cfg)
                    logger.warning(f"Missing '{name}' adapter detected; added to keep ranks in sync for DDP.")

        # Enable gradient checkpointing (and required grads for PEFT) if requested
        if getattr(args, "gradient_checkpointing", False):
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=getattr(args, "gradient_checkpointing_kwargs", None)
                )
            if is_peft_available() and isinstance(model, PeftModel):
                # Ensure LoRA params require grads when using checkpointing
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model.config._name_or_path, truncation_side="left")

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        
        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            if isinstance(reward_funcs[i], nn.Module):  # Use Module over PretrainedModel for compat w/ compiled models
                self.reward_func_names.append(reward_funcs[i].config._name_or_path.split("/")[-1])
            else:
                self.reward_func_names.append(reward_funcs[i].__name__)
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        if len(reward_processing_classes) != len(reward_funcs):
            raise ValueError(
                f"The number of reward processing classes ({len(reward_processing_classes)}) must match the number of "
                f"reward functions ({len(reward_funcs)})."
            )

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class

        self.reward_processing_classes = reward_processing_classes

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_transformers_paged = args.use_transformers_paged
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        self.vllm_importance_sampling_correction = args.vllm_importance_sampling_correction
        self.vllm_importance_sampling_cap = args.vllm_importance_sampling_cap
        self.use_liger_loss = args.use_liger_loss
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.importance_sampling_level = args.importance_sampling_level
        self.mask_truncated_completions = args.mask_truncated_completions
        self.top_entropy_quantile = args.top_entropy_quantile
        if self.use_liger_loss and self.top_entropy_quantile < 1.0:
            raise NotImplementedError(
                "Liger Kernels don't currently support masking token positions based on entropy."
            )
        if self.use_liger_loss and not self.importance_sampling_level == "token":
            raise NotImplementedError(
                "Liger Kernels currently only support token-level importance sampling. Please set"
                "`importance_sampling_level` to 'token'."
            )

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead."
            )

        # Multi-step
        self.num_iterations = args.num_iterations  
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None
        
        # Public-Private agent configuration
        self.num_agents = getattr(args, 'num_agents', 2)  # Number of private agents
        self.num_turns = 2 * self.num_agents  # public-private-public-private...
        
        self.public_agent_max_completion_length = getattr(args, 'public_agent_max_completion_length', args.max_completion_length)
        self.private_agent_max_completion_length = getattr(args, 'private_agent_max_completion_length', args.max_completion_length)
        
        # For Liger loss, use the maximum of all completion lengths to ensure it can handle all cases
        self.liger_max_completion_length = max(
            self.max_completion_length,
            self.public_agent_max_completion_length,
            self.private_agent_max_completion_length
        )

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Log before calling super().__init__ to track initialization
        try:
            process_index = getattr(args, 'process_index', 0)
            if process_index == 0:
                logger.info("Calling super().__init__() to initialize BaseTrainer...")
        except:
            pass
        
        super().__init__(
            model=model,
            args=args,
            data_collator=identity,  # No data collation is needed in GRPO
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            # In Trainer, `training_step` scales the loss by `gradient_accumulation_steps` only if `compute_loss_func`
            # is None. For DAPO, loss scaling instead depends on the total number of completions tokens across the
            # global accumulated batch. To control scaling ourselves, we must disable Trainer's built-in scaling. The
            # simplest (though a bit hacky) way is to set `compute_loss_func` to any non-None value, which bypasses
            # that behavior without rewriting `training_step`.
            compute_loss_func="non-None value to disable scaling",
        )
        
        if hasattr(args, 'process_index') and args.process_index == 0:
            logger.info("super().__init__() completed")

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # For deepspeed, fsdp or non-distributed models, create a reference model from scratch
            config = AutoConfig.from_pretrained(model_id)
            architecture = getattr(transformers, config.architectures[0])
            self.ref_model = architecture.from_pretrained(model_id, **model_init_kwargs)

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Liger loss
        if self.use_liger_loss:
            # Liger can trigger torch.compile/inductor issues on some shapes; disable its internal compile path.
            os.environ.setdefault("LIGER_DISABLE_TORCH_COMPILE", "1")
            if not is_liger_kernel_available():
                raise ImportError(
                    "Liger is required to use `liger_loss` as the GRPO loss. Run `pip install liger-kernel`."
                )
            # redirect the model.module forward to the model forward to ensure pre-forward hooks are called
            self._forward_redirection = _ForwardRedirection()

            self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                beta=self.beta,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                temperature=self.temperature,
                use_ref_model=self.beta != 0.0,
                loss_type=self.loss_type,
                max_completion_length=self.liger_max_completion_length,
            )

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # Keep logs sized to the generation batch to record only outputs from the latest model update.
        # For multi-turn generation: num_samples * num_generations * num_turns
        # num_samples = per_device_train_batch_size // num_generations
        # Total = per_device_train_batch_size * num_turns
        log_buffer_size = args.per_device_train_batch_size * self.num_turns
        self._logs = {
            "images": deque(maxlen=log_buffer_size),
            "prompt": deque(maxlen=log_buffer_size),
            "completion": deque(maxlen=log_buffer_size),
            "rewards": defaultdict(lambda: deque(maxlen=log_buffer_size)),
            "advantages": deque(maxlen=log_buffer_size),
            "turn_info": deque(maxlen=log_buffer_size),  # Store turn_info for table logging
        }
        self._trajectory_buffer = []
        
        # Completion saving configuration
        self.save_completions = getattr(args, 'save_completions', True)
        self.save_completions_path = getattr(args, 'save_completions_path', None)
        if self.save_completions_path is None:
            self.save_completions_path = os.path.join(args.output_dir, "completions")
        if self.accelerator.is_main_process and self.save_completions:
            os.makedirs(self.save_completions_path, exist_ok=True)
        self._completions_to_save = []

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install trl[vllm]` to use it."
                )

            if self.vllm_mode == "server":
                if self.accelerator.is_main_process:
                    if args.vllm_server_base_url is not None:
                        base_url = args.vllm_server_base_url
                    else:
                        base_url = f"http://{args.vllm_server_host}:{args.vllm_server_port}"
                    self.vllm_client = VLLMClient(base_url=base_url, connection_timeout=args.vllm_server_timeout)
                    self.vllm_client.init_communicator(device=torch.cuda.current_device())

            elif self.vllm_mode == "colocate":
                # Make sure vllm_tensor_parallel_size group size evenly divides the world size - each group should have
                # the same number of ranks
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                    # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                    self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                # vLLM requires the environment variables to be set for distributed training.
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                # Ensure distributed rendezvous variables are set without colliding across concurrent runs
                ensure_master_addr_port()

                if self.max_prompt_length is not None and self.max_completion_length is not None:
                    max_model_len = self.max_prompt_length + self.max_completion_length
                else:
                    max_model_len = None

                # vLLM + PEFT: to safely load merged LoRA weights, avoid bitsandbytes load/quantization
                llm_load_format = self.args.vllm_model_impl if hasattr(self.args, "vllm_model_impl") else None
                load_format = "bitsandbytes"
                quantization = "bitsandbytes"
                if is_peft_model(self.model):
                    load_format = "auto"  # vLLM expects a string; "auto" keeps dense load
                    quantization = None

                self.llm = LLM(
                    model=model.name_or_path,
                    tensor_parallel_size=args.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size
                    * self.vllm_tensor_parallel_size
                    * self.args.steps_per_generation,
                    max_model_len=max_model_len,
                    distributed_executor_backend="external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    # Latest vLLM v1 memory profiler is misled by the high default value (i.e., 32768) - thinking there's not enough memory
                    max_num_batched_tokens=4096,
                    model_impl=self.args.vllm_model_impl,
                    enable_sleep_mode=self.args.vllm_enable_sleep_mode,
                    load_format=load_format,
                    quantization=quantization,
                    # Important so temperature scaling/logit tweaking affects the TIS log probs
                    # logprobs_mode="processed_logprobs",
                )
                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=1)
            else:
                raise ValueError(f"vllm_mode must be either 'server' or 'colocate', got '{self.vllm_mode}'.")

            # vLLM specific sampling arguments
            self.guided_decoding_regex = args.vllm_guided_decoding_regex

            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            generation_kwargs = {
                "max_new_tokens": self.max_completion_length,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,
                "cache_implementation": args.cache_implementation,
            }
            if args.generation_kwargs is not None:
                generation_kwargs.update(args.generation_kwargs)
            self.generation_config = GenerationConfig(**generation_kwargs)

        # Track last synced adapter for vLLM so we can resync when switching (public/private)
        self._last_loaded_adapter = None

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                else:
                    # set device placement to True to make `prepare_model` move `reward_func` to device when using fsdp
                    self.reward_funcs[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True, device_placement=True
                    )
        
        # Log initialization completion
        if hasattr(self, 'accelerator') and self.accelerator.is_main_process:
            logger.info("PUBPRIGRPOTrainer.__init__() completed successfully")

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "image", "images"]

    def _get_train_sampler(self, dataset: Optional[Dataset] = None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...
        if dataset is None:
            dataset = self.train_dataset
        
        # Each prompt should appear once; num_generations duplication is handled inside _generate_multi_turn.
        # We set batch_size to the number of unique prompts per local step (before steps_per_generation splitting).
        unique_prompts_per_step = self.args.per_device_train_batch_size // self.num_generations
        if self.args.per_device_train_batch_size % self.num_generations != 0:
            raise ValueError(
                f"per_device_train_batch_size ({self.args.per_device_train_batch_size}) must be divisible by "
                f"num_generations ({self.num_generations}) so that each step has an integer number of unique prompts."
            )

        repeat_count = self.num_iterations * self.args.steps_per_generation

        if self.accelerator.num_processes > 1:
            sampler = DistributedRepeatSampler(
                dataset=dataset,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                mini_repeat_count=1,
                batch_size=unique_prompts_per_step,
                repeat_count=repeat_count,
                shuffle=self.shuffle_dataset,
                seed=self.args.seed,
                drop_last=False,
            )
        else:
            if self.accelerator.is_main_process:
                logger.info(
                    "Creating RepeatSampler: dataset_size=%s, mini_repeat_count=%s, batch_size=%s, repeat_count=%s",
                    len(dataset) if hasattr(dataset, "__len__") else "unknown",
                    1,
                    unique_prompts_per_step,
                    repeat_count,
                )
            sampler = RepeatSampler(
                data_source=dataset,
                mini_repeat_count=1,
                batch_size=unique_prompts_per_step,
                repeat_count=repeat_count,
                shuffle=self.shuffle_dataset,
                seed=self.args.seed,
            )
            if self.accelerator.is_main_process:
                logger.info("RepeatSampler created successfully")
        
        return sampler

    def get_train_dataloader(self) -> DataLoader:
        """
        Custom dataloader so that each local step receives
        (per_device_train_batch_size // num_generations) unique prompts,
        then `_prepare_inputs` duplicates them num_generations times during generation.
        The loader batch size is multiplied by steps_per_generation so we can
        split into that many accumulation slices without mixing prompts.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # Validate divisibility once more for clarity.
        if self.args.per_device_train_batch_size % self.num_generations != 0:
            raise ValueError(
                f"per_device_train_batch_size ({self.args.per_device_train_batch_size}) must be divisible by "
                f"num_generations ({self.num_generations})."
            )

        unique_prompts_per_step = self.args.per_device_train_batch_size // self.num_generations
        batch_size = unique_prompts_per_step * self.args.steps_per_generation

        # Build dataloader params (mirrors transformers.Trainer with our custom batch_size/sampler).
        data_collator = self.data_collator
        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # When using DistributedRepeatSampler, the sampler already handles the distributed nature
        # of the dataset (splitting indices per rank). If we pass this DataLoader to accelerator.prepare(),
        # Accelerate might attempt to shard it again (depending on version/detection), leading to
        # double sharding (e.g., using only 1/4 of data with 2 GPUs).
        # Since the inputs are text/metadata (not yet tensors on device), we can skip prepare().
        return DataLoader(self.train_dataset, **dataloader_params)

    def _get_eval_sampler(self, dataset: Optional[Dataset] = None) -> Sampler:
                                
        if dataset is None:
            dataset = self.eval_dataset
        
        # Each prompt should appear once; num_generations duplication is handled inside _generate_multi_turn.
        # We set batch_size to the number of unique prompts per local step (before steps_per_generation splitting).
        unique_prompts_per_step = self.args.per_device_eval_batch_size // self.num_generations
        
        if self.args.per_device_train_batch_size % self.num_generations != 0:
            raise ValueError(
                f"per_device_train_batch_size ({self.args.per_device_train_batch_size}) must be divisible by "
                f"num_generations ({self.num_generations}) so that each step has an integer number of unique prompts."
            )

        repeat_count = self.num_iterations * self.args.steps_per_generation

        if self.accelerator.num_processes > 1:
            sampler = DistributedRepeatSampler(
                dataset=dataset,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                mini_repeat_count=1,
                batch_size=unique_prompts_per_step,
                repeat_count=repeat_count,
                shuffle=self.shuffle_dataset,
                seed=self.args.seed,
                drop_last=False,
            )
        else:
            if self.accelerator.is_main_process:
                logger.info(
                    "Creating RepeatSampler: dataset_size=%s, mini_repeat_count=%s, batch_size=%s, repeat_count=%s",
                    len(dataset) if hasattr(dataset, "__len__") else "unknown",
                    1,
                    unique_prompts_per_step,
                    repeat_count,
                )
            sampler = RepeatSampler(
                data_source=dataset,
                mini_repeat_count=1,
                batch_size=unique_prompts_per_step,
                repeat_count=repeat_count,
                shuffle=self.shuffle_dataset,
                seed=self.args.seed,
            )
            if self.accelerator.is_main_process:
                logger.info("RepeatSampler created successfully")
        
        return sampler
    
    def get_eval_dataloader(self) -> DataLoader:
        """
        Custom dataloader so that each local step receives
        (per_device_train_batch_size // num_generations) unique prompts,
        then `_prepare_inputs` duplicates them num_generations times during generation.
        The loader batch size is multiplied by steps_per_generation so we can
        split into that many accumulation slices without mixing prompts.
        """
        if self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires a eval_dataset.")

        # Validate divisibility once more for clarity.
        if self.args.per_device_eval_batch_size % self.num_generations != 0:
            raise ValueError(
                f"per_device_eval_batch_size ({self.args.per_device_eval_batch_size}) must be divisible by "
                f"num_generations ({self.num_generations})."
            )

        unique_prompts_per_step = self.args.per_device_eval_batch_size // self.num_generations
        batch_size = unique_prompts_per_step * self.args.steps_per_generation

        # Build dataloader params (mirrors transformers.Trainer with our custom batch_size/sampler).
        data_collator = self.data_collator
        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(self.eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # When using DistributedRepeatSampler, the sampler already handles the distributed nature
        # of the dataset (splitting indices per rank). If we pass this DataLoader to accelerator.prepare(),
        # Accelerate might attempt to shard it again (depending on version/detection), leading to
        # double sharding (e.g., using only 1/4 of data with 2 GPUs).
        # Since the inputs are text/metadata (not yet tensors on device), we can skip prepare().
        return DataLoader(self.eval_dataset, **dataloader_params)
        
    @profiling_decorator
    def _get_last_hidden_state(
        self,
        unwrapped_model,
        input_ids,
        attention_mask,
        logits_to_keep,
        pixel_values=None,
        image_grid_thw=None,
        pixel_attention_mask=None,
        image_sizes=None,
    ):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model

        # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # For Qwen models:
        if image_grid_thw is not None and pixel_values is not None:
            model_inputs["image_grid_thw"] = image_grid_thw
        # For Gemma, SmolVLM2, LLaVa-Next etc.:
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        # For SmolVLM2
        if pixel_attention_mask is not None:
            model_inputs["pixel_attention_mask"] = pixel_attention_mask
        # For LLaVa-Next
        if image_sizes is not None:
            model_inputs["image_sizes"] = image_sizes

        # Only add logits_to_keep if the model supports it
        if "logits_to_keep" in self.model_kwarg_keys:
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

        last_hidden_state = unwrapped_model.model(**model_inputs).last_hidden_state
        # Exclude the last value: it corresponds to the next token pred
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
        last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    def get_high_entropy_mask(self, entropies: torch.Tensor, mask: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Returns a binary mask identifying tokens whose entropy exceeds a given quantile threshold.

        Args:
            entropies (`torch.Tensor`):
                Tensor of shape (batch_size, seq_len) with per-token entropy values.
            mask (`torch.Tensor`):
                Binary mask of the same shape as `entropies`, where `1` indicates valid tokens and `0` padding.
            threshold (`float`):
                Quantile threshold between `0.0` and `1.0` to select high-entropy tokens.

        Returns:
            `torch.Tensor`:
                Boolean mask of shape (batch_size, seq_len), where `True` indicates tokens with entropy >= threshold
                and `False` otherwise.
        """
        local = entropies[mask.bool()].float()

        # Use a negative pad_value as a sentinel because entropy values are always >= 0.
        # This guarantees that the sentinel cannot collide with any real entropy value.
        pad_value = -1e9

        # Pad across processes so that every rank has the same tensor length
        padded = self.accelerator.pad_across_processes(local, dim=0, pad_index=pad_value)
        gathered = self.accelerator.gather(padded)

        # Drop sentinel values (safe because no entropy can be negative)
        gathered = gathered[gathered != pad_value]

        if gathered.numel() == 0:
            return torch.zeros_like(entropies, dtype=torch.bool)

        entropy_threshold = torch.quantile(gathered, threshold)
        masked_entropies = entropies * mask.float()
        entropy_mask = masked_entropies >= entropy_threshold
        return entropy_mask & mask.bool()  # ensure padding tokens are always masked out

    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
    ) -> dict[str, Optional[torch.Tensor]]:
        """Compute log-probs and (optionally) entropies for each token."""
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
            if image_grid_thw is not None and pixel_values is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
                row_start, row_end = cum_rows[start].item(), cum_rows[start + batch_size].item()
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start : start + batch_size]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start : start + batch_size]
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[start : start + batch_size]

            # Only add logits_to_keep if the model supports it
            if "logits_to_keep" in self.model_kwarg_keys:
                # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

            logits = model(**model_inputs).logits
            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, H)
            # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
            logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)  # compute logprobs
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

    def _fix_param_name_to_vllm(self, name, extra_prefixes: Optional[list[str]] = None):
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for prefix in prefixes:
            name = name.replace(prefix, "")
        return name

    def _sync_fsdp1_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM."""
        # For FSDP1, we need to recurse into children and also use summon_full_params
        if visited is None:
            visited = set()
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp1_params_to_vllm(
                child_module, prefix=child_prefix, visited=visited
            )  # recurse into the child

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    full_name = self._fix_param_name_to_vllm(full_name, extra_prefixes=["_fsdp_wrapped_module."])

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self._should_skip_vllm_param(full_name):
                        continue

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param.data)])

    def _sync_fsdp2_params_to_vllm(self, module: nn.Module):
        # For FSDP2, module.state_dict() already covers all parameters, so no need for recursion
        for name, param in module.state_dict().items():
            if self._should_skip_vllm_param(name):
                continue
            if param.is_cpu:
                param = param.to(torch.device("cuda"))
            param = param.full_tensor()

            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                self.vllm_client.update_named_param(name, param)
            elif self.vllm_mode == "colocate":
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights([(name, param)])

    @profiling_decorator
    def _move_model_to_vllm(self, force: bool = False):
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if is_peft_model(self.model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            # TODO: does this work with FSDP?
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                    fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                    if fsdp_version == 1:
                        self._sync_fsdp1_params_to_vllm(
                            self.model
                        )  # use memory-efficient post-order traversal for FSDP
                    elif fsdp_version == 2:
                        self._sync_fsdp2_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        if self._should_skip_vllm_param(name):
                            continue
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            try:
                                llm_model.load_weights([(name, param.data)])
                            except AssertionError as e:
                                raise ValueError(
                                    f"vLLM load_weights shape mismatch for param '{name}' "
                                    f"(param={tuple(param.data.shape)}). This usually means the "
                                    f"weight name/shape after LoRA merge does not match vLLM's "
                                    f"backbone. Try removing adapters from the sync or verify the "
                                    f"adapter merge output."
                                ) from e

                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                if fsdp_version == 1:
                    self._sync_fsdp1_params_to_vllm(self.model)  # use memory-efficient post-order traversal for FSDP
                elif fsdp_version == 2:
                    self._sync_fsdp2_params_to_vllm(self.model)
            else:
                for name, param in self.model.named_parameters():
                    if self._should_skip_vllm_param(name):
                        continue
                    name = self._fix_param_name_to_vllm(name)
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            try:
                                llm_model.load_weights([(name, param.data)])
                            except AssertionError as e:
                                raise ValueError(
                                    f"vLLM load_weights shape mismatch for param '{name}' "
                                    f"(param={tuple(param.data.shape)}). This usually means the "
                                    f"weight name/shape after LoRA merge does not match vLLM's "
                                    f"backbone. Try removing adapters from the sync or verify the "
                                    f"adapter merge output."
                                ) from e

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()

        # Remember which adapter was used for this sync (active_adapter exists on PeftModel)
        if is_peft_model(self.model):
            self._last_loaded_adapter = self.model.active_adapter

    def _should_skip_vllm_param(self, name: str) -> bool:
        # Skip PEFT/adapter auxiliary params that vLLM backbone does not expect
        lowered = name.lower()
        return (
            "lora" in lowered
            or "adapter" in lowered
            or "modules_to_save" in lowered
            or "bias_finetune" in lowered
        )

    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        
        if self.accelerator.is_main_process and self._step == 0:
            logger.info(f"_prepare_inputs called for the first time (mode={mode})")
            if isinstance(generation_batch, dict):
                logger.info(f"Generation batch keys: {list(generation_batch.keys())}")
                if "prompt" in generation_batch or "problem" in generation_batch:
                    batch_size = len(generation_batch.get("prompt", generation_batch.get("problem", [])))
                    logger.info(f"Batch size: {batch_size}")
            else:
                # Fallback for list/other iterables
                logger.info(f"Generation batch type: {type(generation_batch).__name__}, len={len(generation_batch) if hasattr(generation_batch, '__len__') else 'unknown'}")
        
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                if self.accelerator.is_main_process:
                    logger.info(f"Generating completions (step={self._step}, generate_every={generate_every})")
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = split_pixel_values_by_grid(generation_batch)
                generation_batch = shuffle_sequence_dict(generation_batch)
                generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                self._buffered_inputs = [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]
                if self.accelerator.is_main_process:
                    logger.info(f"Completions generated and buffered ({len(self._buffered_inputs)} batches)")
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    @profiling_decorator
    def _calculate_trajectory_rewards(self, inputs, completions_per_trajectory, *args, **kwargs):
        """
        Calculate per-trajectory rewards.

        - Accuracy: 1.0 if the *last* turn matches the gold answer, else 0.0 (None -> 0.0)
        - Format: 0.5 if *all* turns contain <think>...</think> and <answer>...</answer>, else 0.0
        """
        device = self.accelerator.device
        num_trajectories = len(completions_per_trajectory)

        if num_trajectories == 0:
            empty = torch.zeros(0, device=device)
            return empty, empty, empty

        answers = [example.get("answer", "") for example in inputs for _ in range(self.num_generations)]

        # Accuracy on the last turn of each trajectory
        acc_inputs = [traj[-1]for traj in completions_per_trajectory]
        
        acc_scores_list = answer_tag_reward_fn(completions=acc_inputs, solution=answers)
        acc_scores = torch.tensor(
            [score for score in acc_scores_list],
            dtype=torch.float32,
            device=device,
        )

        # Format check on every turn of every trajectory
        flat_messages = [turn for traj in completions_per_trajectory for turn in traj]
        flat_fmt_scores = think_answer_format_reward(completions=flat_messages)
        flat_fmt_tensor = torch.tensor(flat_fmt_scores, dtype=torch.float32, device=device)

        num_turns = len(completions_per_trajectory[0])
        fmt_scores_matrix = flat_fmt_tensor.view(num_trajectories, num_turns)
        
        all_formatted = (fmt_scores_matrix >= 0.5).all(dim=1)
        format_rewards = torch.where(
            all_formatted,
            torch.full_like(acc_scores, 0.5),
            torch.zeros_like(acc_scores),
        )

        # format_compliance_rate = fmt_scores_matrix.float().mean(dim=1)
        # format_rewards = format_compliance_rate * 0.5

        rewards = acc_scores + format_rewards
        return rewards, acc_scores, format_rewards

    @profiling_decorator
    def _calculate_rewards(self, inputs, completions_per_trajectory):
        """
        Compute trajectory-level rewards (accuracy + format) and log per-component stats.

        Returns:
            rewards_per_trajectory: tensor of shape (num_trajectories,)
            acc_scores: tensor of shape (num_trajectories,)
            format_rewards: tensor of shape (num_trajectories,)
        """
        rewards_per_trajectory, acc_scores, format_rewards = self._calculate_trajectory_rewards(
            inputs=inputs, completions_per_trajectory=completions_per_trajectory
        )

        gathered_rewards = gather(rewards_per_trajectory)
        gathered_acc = gather(acc_scores)
        gathered_format = gather(format_rewards)

        mode = "train" if self.model.training else "eval"
        if gathered_acc.numel() > 0:
            self._metrics[mode]["rewards/accuracy"].append(gathered_acc.nanmean().item())
        if gathered_format.numel() > 0:
            self._metrics[mode]["rewards/format"].append(gathered_format.nanmean().item())

        # Log each component separately for easier inspection
        self._logs["rewards"]["accuracy"].extend(gathered_acc.tolist())
        self._logs["rewards"]["format"].extend(gathered_format.tolist())
        # Total reward (acc + format) for compatibility with downstream code
        self._logs["rewards"]["total"].extend(gathered_rewards.tolist())

        return rewards_per_trajectory, acc_scores, format_rewards

    def _generate_single_turn(self, prompts: list[str], images: Optional[list], max_completion_length: Optional[int] = None):
        device = self.accelerator.device
        
        # Use provided max_completion_length or fall back to default
        if max_completion_length is None:
            max_completion_length = self.max_completion_length

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is the sky?"}]}]
        kwargs = {}
        if images is not None:
            kwargs = {"images": images}
            for prompt, image_list in zip(prompts, images):
                if isinstance(prompt, list):  # i.e., when using conversational data
                    prepare_multimodal_messages(prompt, num_images=len(image_list))

        # Handle both message format and prompt format
        prompts_text = []
        for prompt in prompts:
            if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict) and "role" in prompt[0]:
                # Already in messages format, apply chat template directly
                formatted = maybe_apply_chat_template({"messages": prompt}, self.processing_class)
                prompts_text.append(formatted.get("prompt", formatted.get("messages", prompt)))
            else:
                # Regular prompt format
                formatted = maybe_apply_chat_template({"prompt": prompt}, self.processing_class)
                prompts_text.append(formatted.get("prompt", prompt))

        if self.use_vllm:
            # vLLM requires raw string prompts; coerce any non-string (e.g., chat messages) to string with chat template
            coerced_prompts_text = []
            for pt in prompts_text:
                if isinstance(pt, str):
                    coerced_prompts_text.append(pt)
                else:
                    if hasattr(self.processing_class, "apply_chat_template"):
                        try:
                            coerced = self.processing_class.apply_chat_template(pt, tokenize=False, add_generation_prompt=True)
                            coerced_prompts_text.append(coerced)
                            continue
                        except Exception:
                            pass
                    coerced_prompts_text.append(str(pt))
            prompts_text = coerced_prompts_text

        if images is not None:
            prompt_inputs = self.processing_class(text=prompts_text, padding=True, return_tensors="pt", **kwargs)
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            if self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode:
                # wake up colocated vLLM instances if needed
                torch.cuda.empty_cache()  # required to avoid OOM in some cases
                self.llm.wake_up()

            # First, update the vLLM weights if needed (also when adapter changes)
            active_adapter = self.model.active_adapter if is_peft_model(self.model) else None
            need_resync = self.state.global_step != self._last_loaded_step or active_adapter != self._last_loaded_adapter
            if need_resync:
                self._move_model_to_vllm(force=True)
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if images is not None:
                    all_images = gather_object(images)

                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]

                    if images is not None:
                        ordered_set_of_images = all_images[:: self.num_generations]
                    else:
                        ordered_set_of_images = None

                    with profiling_context(self, "vLLM.generate"):
                        output = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            images=ordered_set_of_images,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=max_completion_length,
                            truncate_prompt_tokens=self.max_prompt_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                            generation_kwargs=self.args.generation_kwargs,
                        )
                        payload = (output["prompt_ids"], output["completion_ids"], output["logprobs"])
                else:
                    payload = None

                # Broadcast the completions from the main process to all processes, ensuring each process receives its corresponding slice.
                obj_list = [payload]
                broadcast_object_list(obj_list, from_process=0)
                all_prompt_ids, all_completion_ids, all_logprobs = obj_list[0]

                # At this point, we only get 1 copy of each prompt, so we need to repeat them num_generations times
                all_prompt_ids = [ids for ids in all_prompt_ids for _ in range(self.num_generations)]

                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                prompt_ids = all_prompt_ids[process_slice]
                completion_ids = all_completion_ids[process_slice]
                logprobs = all_logprobs[process_slice]

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None

                generation_kwargs = {
                    "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": max_completion_length,
                    "truncate_prompt_tokens": self.max_prompt_length,
                    "guided_decoding": guided_decoding,
                    "logprobs": 0,  # only return the logprob of the generated token
                }
                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]

                    if images is not None:
                        gathered_images = [None for _ in range(self.vllm_tensor_parallel_size)]
                        torch.distributed.all_gather_object(gathered_images, images, group=self.tp_group)
                        all_images = [img for sublist in gathered_images for img in sublist]
                    else:
                        all_images = None
                else:
                    all_prompts_text = prompts_text
                    all_images = images

                if images is not None and all_images:
                    vllm_inputs = []
                    for prompt, image_list in zip(all_prompts_text, all_images):
                        vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image_list}})

                else:
                    vllm_inputs = all_prompts_text

                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=False)

                all_prompt_ids = [output.prompt_token_ids for output in all_outputs]
                all_completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
                all_logprobs = [
                    [next(iter(lp.values())).logprob for lp in output.logprobs]
                    for outputs in all_outputs
                    for output in outputs.outputs
                ]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    prompt_ids = all_prompt_ids[tp_slice]
                    completion_ids = all_completion_ids[tp_slice]
                    logprobs = all_logprobs[tp_slice]
                else:
                    prompt_ids = all_prompt_ids
                    completion_ids = all_completion_ids
                    logprobs = all_logprobs

                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=1)

        elif self.use_transformers_paged:
            # Re-process inputs for paged generation if needed
            # Note: images are already validated and preprocessed above
            paged_prompt_inputs = self.processing_class(text=prompts_text, **kwargs)
            previous_attn = self.model_wrapped.config._attn_implementation

            if is_flash_attn_2_available():
                self.model_wrapped.config._attn_implementation = "paged_attention"
            else:
                self.model_wrapped.config._attn_implementation = "sdpa_paged"
            
            # Create a modified generation config with the specified max_completion_length
            gen_config_dict = {k: v for k, v in self.generation_config.to_dict().items()}
            gen_config_dict.pop('max_new_tokens', None)  # Remove existing max_new_tokens to avoid duplicate argument
            generation_config = GenerationConfig(
                **gen_config_dict,
                max_new_tokens=max_completion_length
            )
            
            with (
                profiling_context(self, "transformers.generate_batch"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                # Cast to the appropriate dtype based on training configuration
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                with torch.inference_mode():
                    all_outputs = unwrapped_model.generate_batch(
                        paged_prompt_inputs.input_ids, generation_config=generation_config, progress_bar=False
                    )
                    unwrapped_model.train()  # restore training mode, as generate_batch forces eval mode
            completion_ids = [output.generated_tokens for output in all_outputs.values()]
            prompt_ids = paged_prompt_inputs.input_ids
            # Restore the original attention implementation, training mode
            self.model_wrapped.config._attn_implementation = previous_attn
            logprobs = None  # not used in this case

        else:
            # Regular generation path
            generate_inputs = self.processing_class(
                text=prompts_text,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                max_length=self.max_prompt_length,
                truncation=True,
                add_special_tokens=False,
                **kwargs,
            )
            generate_inputs = super()._prepare_inputs(generate_inputs)

            # Create a modified generation config with the specified max_completion_length
            gen_config_dict = {k: v for k, v in self.generation_config.to_dict().items()}
            gen_config_dict.pop('max_new_tokens', None)  # Remove existing max_new_tokens to avoid duplicate argument
            generation_config = GenerationConfig(
                **gen_config_dict,
                max_new_tokens=max_completion_length
            )

            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                prompt_completion_ids = unwrapped_model.generate(
                    **generate_inputs, generation_config=generation_config, disable_compile=True
                )
            # Compute prompt length and extract completion ids
            prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # Mask everything after the first EOS token
            is_eos = completion_ids == self.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool())]
            completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool())]
            logprobs = None  # not used in this case

        return prompt_ids, completion_ids, logprobs, forward_kwargs
    
    def _generate_multi_turn(self, prompts: list, images: Optional[list], problems: list[str], answers: list[str]):
        device = self.accelerator.device
        num_samples = len(prompts)
        
        all_turn_prompt_ids = []
        all_turn_completion_ids = []
        all_turn_logprobs = []
        turn_info = []
        all_forward_kwargs = []

        current_histories = [[[] for _ in range(self.num_generations)] for _ in range(num_samples)]
        remaining_agents = self.num_agents

        if self.accelerator.is_main_process:
            logger.info(f"Starting multi-turn generation: {num_samples} samples, {self.num_generations} generations, {self.num_turns} turns")

        with torch.no_grad(): # Ensure Inference Mode
            for turn_idx in range(self.num_turns):
                is_public_turn = (turn_idx % 2 == 0)
                agent_name = "public" if is_public_turn else "private"
                
                # Synchronize all processes before switching adapter
                self.accelerator.wait_for_everyone()
                self._switch_adapter(agent_name, self.model)
                self.accelerator.wait_for_everyone()
                
                turn_prompts = []
                for sample_idx in range(num_samples):
                    orig_prob = problems[sample_idx]
                    for gen_idx in range(self.num_generations):
                        hist = current_histories[sample_idx][gen_idx]
                        
                        last_public_output = next((out for agent, out in reversed(hist) if agent == "public"), None)
                        last_private_output = next((out for agent, out in reversed(hist) if agent == "private"), None)

                        if is_public_turn:
                            prev_outputs_str = "No previous outputs yet."
                            formatted_outputs = []
                            if last_public_output: formatted_outputs.append(f"Previous Orchestrator Output:\n{last_public_output}")
                            if last_private_output: formatted_outputs.append(f"Previous Worker Agent Output:\n{last_private_output}")
                            if formatted_outputs: prev_outputs_str = "\n\n".join(formatted_outputs)

                            content = PUBLIC_PROMPT.format(original_problem=orig_prob, previous_outputs=prev_outputs_str, num_agents=remaining_agents)
                            system_prompt = PUBLIC_SYSTEM_PROMPT
                        else:
                            content = PRIVATE_PROMPT.format(original_problem=orig_prob, orchestrator_instruction=last_public_output if last_public_output else "")
                            system_prompt = PRIVATE_SYSTEM_PROMPT

                        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
                        # Store messages directly - _generate_single_turn will handle template application
                        turn_prompts.append(messages)

                max_len = self.public_agent_max_completion_length if is_public_turn else self.private_agent_max_completion_length
                
                # Synchronize before generation
                self.accelerator.wait_for_everyone()
                ids_prompts, ids_completions, logprobs, fwd_kwargs = self._generate_single_turn(
                    turn_prompts, images=None, max_completion_length=max_len
                )
                # Synchronize after generation
                self.accelerator.wait_for_everyone()
                
                decoded = self.processing_class.batch_decode(ids_completions, skip_special_tokens=False)
                
                for i, content in enumerate(decoded):
                    sample_idx = i // self.num_generations
                    gen_idx = i % self.num_generations
                    answer = self._extract_answer_content(content)
                    current_histories[sample_idx][gen_idx].append((agent_name, answer))
                    turn_info.append((agent_name, turn_idx, sample_idx, gen_idx))

                all_turn_prompt_ids.extend(ids_prompts)
                all_turn_completion_ids.extend(ids_completions)
                if logprobs: all_turn_logprobs.extend(logprobs)
                else: all_turn_logprobs.extend([None] * len(ids_prompts))
                all_forward_kwargs.append(fwd_kwargs)
                
                if self.accelerator.is_main_process:
                    logger.info(f"Completed turn {turn_idx + 1}/{self.num_turns} ({agent_name}): {len(ids_completions)} completions")
                
                if is_public_turn: remaining_agents -= 1

        # Merge forward_kwargs from all turns (use the last one if they're all empty)
        merged_fwd_kwargs = {}
        for fwd_kw in all_forward_kwargs:
            if fwd_kw:
                merged_fwd_kwargs.update(fwd_kw)
        
        if self.accelerator.is_main_process:
            logger.info(f"Multi-turn generation completed: {len(all_turn_completion_ids)} total completions")

        return all_turn_prompt_ids, all_turn_completion_ids, all_turn_logprobs, merged_fwd_kwargs, turn_info

    def _generate(self, prompts: list[str], images: Optional[list]):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Use multi-turn generation for public-private agent setup
        # Extract problems and answers from inputs if available
        problems = getattr(self, '_current_problems', prompts)
        answers = getattr(self, '_current_answers', [""] * len(prompts))
        prompt_ids, completion_ids, logprobs, forward_kwargs, turn_info = self._generate_multi_turn(prompts, images, problems, answers)

        # Get completion length per sequence, used for logging
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()
        total_completion_tokens = agg_completion_lengths.sum()  # = num_items_in_batch, required for the DAPO loss
        
        # Store turn_info for later use
        self._current_turn_info = turn_info

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        return prompt_ids, completion_ids, total_completion_tokens, logprobs, forward_kwargs, turn_info

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Extract problem and answer from inputs
        problems = [x.get("problem", x.get("prompt", "")) for x in inputs]
        answers = [x.get("answer", "") for x in inputs]
        
        # Store for use in _generate
        self._current_problems = problems
        self._current_answers = answers
        
        # Create initial prompts using chat template and PUBLIC_PROMPT
        prompts = []
        for problem in problems:
            # Format public prompt for first turn
            public_prompt_text = PUBLIC_PROMPT.format(
                original_problem=problem,
                previous_outputs="No previous outputs yet.",
                num_agents=self.num_agents
            )
            # Apply chat template
            messages = [
                {"role": "system", "content": PUBLIC_SYSTEM_PROMPT},
                {"role": "user", "content": public_prompt_text}
            ]
            formatted = maybe_apply_chat_template({"messages": messages}, self.processing_class)
            prompts.append(formatted.get("messages", messages))

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        (
            prompt_ids_list,
            completion_ids_list,
            num_items_in_batch,
            sampling_per_token_logps_list,
            forward_kwargs,
            turn_info,
        ) = self._generate(prompts, images)

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            # sampling_per_token_logps is only used when vLLM is enabled with importance sampling correction
            # If vLLM is used, all logprobs should be available (not None)
            # If regular transformers is used, logprobs will be None and we set sampling_per_token_logps to None
            if any(logps is None for logps in sampling_per_token_logps_list):
                # If any logprobs are None, we're not using vLLM, so set to None
                sampling_per_token_logps = None
            else:
                logger.info("Sampling per token logps are available")
                # All logprobs are available (vLLM case), convert to tensors
                sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
                sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        num_images = [len(img_list) for img_list in images] if images is not None else None

        with torch.no_grad():
          
             generate_every = self.args.steps_per_generation * self.num_iterations
             if self.args.gradient_accumulation_steps % generate_every != 0 or (
                 self.use_vllm and self.vllm_importance_sampling_correction
             ):
                 old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                     self.model,
                     prompt_completion_ids,
                     attention_mask,
                     logits_to_keep,
                     batch_size,
                     num_images=num_images,
                     **forward_kwargs,
                 )
             else:
                 old_per_token_logps = None

             if self.use_vllm and self.vllm_importance_sampling_correction:
                 importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
                 importance_sampling_ratio = torch.clamp(
                     importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                 )

             if self.beta != 0.0:
                 if self.ref_model is not None:
                     ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                         self.ref_model,
                         prompt_completion_ids,
                         attention_mask,
                         logits_to_keep,
                         batch_size=batch_size,
                         num_images=num_images,
                         **forward_kwargs,
                     )
                 else:
                     with self.accelerator.unwrap_model(self.model).disable_adapter():
                         ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                             self.model,
                             prompt_completion_ids,
                             attention_mask,
                             logits_to_keep,
                             batch_size=batch_size,
                             num_images=num_images,
                             **forward_kwargs,
                         )
             else:
                 ref_per_token_logps = None

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        
        # Group completions by trajectory using turn_info to keep samples/generations isolated.
        num_samples = len(prompts)  # Original number of samples
        total_completions = num_samples * self.num_generations * self.num_turns

        completions_per_trajectory = [[None] * self.num_turns for _ in range(num_samples * self.num_generations)]
        for completion_text, info in zip(completions_text[:total_completions], turn_info[:total_completions]):
            _, turn_idx, sample_idx, gen_idx = info
            traj_idx = sample_idx * self.num_generations + gen_idx
            completions_per_trajectory[traj_idx][turn_idx] = completion_text

        # Fill any missing slots defensively
        for traj in completions_per_trajectory:
            for i in range(self.num_turns):
                if traj[i] is None:
                    raise ValueError(f"Completion text is missing for turn {i} of trajectory {traj}")

        # Convert to the expected format
        formatted_trajectories = []
        for trajectory_completions in completions_per_trajectory:
            formatted_trajectory = []
            for completion_text in trajectory_completions:
                if is_conversational(inputs[0]):
                    formatted_trajectory.append([{"role": "assistant", "content": completion_text}])
                else:
                    formatted_trajectory.append(completion_text)
            formatted_trajectories.append(formatted_trajectory)

        completions_per_trajectory = formatted_trajectories
        
        # Calculate rewards for each trajectory (total, accuracy, format)
        rewards_per_trajectory, acc_scores, format_rewards = self._calculate_rewards(inputs, completions_per_trajectory)
        
        # -------------------------------------------------------------------------
        # [LOGGING] Trajectory-level logging for WandB Table
        # -------------------------------------------------------------------------
        if hasattr(self, "_trajectory_buffer"):
            local_traj_logs = []
            
            for traj_idx, traj_content in enumerate(completions_per_trajectory):
                input_idx = traj_idx // self.num_generations
                gen_idx = traj_idx % self.num_generations
                problem_text = problems[input_idx]
                answer_text = answers[input_idx]
                
                # Retrieve rewards (scalars)
                reward_val = rewards_per_trajectory[traj_idx].item() if traj_idx < len(rewards_per_trajectory) else 0.0
                acc_val = acc_scores[traj_idx].item() if traj_idx < len(acc_scores) else 0.0
                fmt_val = format_rewards[traj_idx].item() if traj_idx < len(format_rewards) else 0.0
                
                log_entry = {
                    "step": self.state.global_step,
                    "sample_idx": input_idx,
                    "generation_idx": gen_idx,
                    "problem": problem_text,
                    "answer": answer_text,
                    "total_reward": reward_val,
                    "accuracy_reward": acc_val,
                    "format_reward": fmt_val,
                }
                
                # Extract turns
                for turn_i, turn_data in enumerate(traj_content):
                    agent = "public" if turn_i % 2 == 0 else "private"
                    
                    if isinstance(turn_data, list) and len(turn_data) > 0 and isinstance(turn_data[0], dict):
                        text = turn_data[0].get("content", "")
                    else:
                        text = str(turn_data)
                        
                    log_entry[f"turn_{turn_i}_{agent}"] = text
                
                local_traj_logs.append(log_entry)
            
            # Store locally first, gather only at log time to save bandwidth and memory
            self._trajectory_buffer.extend(local_traj_logs)
        
        # -------------------------------------------------------------------------
        # [JSON SAVE] Save completions to JSON file for later inspection
        # -------------------------------------------------------------------------
        if self.save_completions and self.accelerator.is_main_process:
            self._collect_completions_for_json(
                problems=problems,
                answers=answers,
                completions_per_trajectory=completions_per_trajectory,
                rewards_per_trajectory=rewards_per_trajectory,
                acc_scores=acc_scores,
                format_rewards=format_rewards,
            )

        # -------------------------------------------------------------------------
        # [FIX] Advantage Calculation Logic
        # GRPO: Group Relative Policy Optimization
        # We must group by Input (Sample) and normalize across Generations.
        # -------------------------------------------------------------------------

        # rewards_per_trajectory shape: (NumSamples * NumGenerations,)
        # Reshape to (NumSamples, NumGenerations) to compute stats per input problem
        rewards_by_sample = rewards_per_trajectory.view(-1, self.num_generations)
        
        # Compute Mean/Std across generations (dim=1) for each sample
        # This ensures we compare generations from the SAME input.
        mean_rewards = rewards_by_sample.mean(dim=1, keepdim=True)
        std_rewards = rewards_by_sample.std(dim=1, keepdim=True)
        
        # Calculate Advantages per trajectory
        if self.scale_rewards == "group":
            # Normalize within the group (Sample)
            advantages_by_sample = (rewards_by_sample - mean_rewards) / (std_rewards + 1e-4)
        elif self.scale_rewards == "batch":
            # Normalize across the entire batch
            advantages_by_sample = (rewards_by_sample - rewards_by_sample.mean()) / (rewards_by_sample.std() + 1e-4)
        else: # "none"
            advantages_by_sample = rewards_by_sample - mean_rewards

        # Flatten back to Trajectory level: (NumSamples * NumGenerations,)
        advantages_traj = advantages_by_sample.view(-1)

        # -------------------------------------------------------------------------
        # [FIX] Expand Advantages & Rewards to Turns
        # Data Layout in input_ids/completion_ids is: [Turn0_AllTrajs, Turn1_AllTrajs, ...]
        # So we need to match this layout (NumTurns, NumTraj) -> Flatten
        # -------------------------------------------------------------------------
        
        # Expand Advantages: (NumTraj,) -> (1, NumTraj) -> (NumTurns, NumTraj) -> Flatten
        advantages = advantages_traj.unsqueeze(0).repeat(self.num_turns, 1).reshape(-1)
        
        # Expand Rewards (for logging compatibility): Same logic
        # rewards_per_trajectory: (NumTraj,)
        rewards = rewards_per_trajectory.unsqueeze(0).repeat(self.num_turns, 1).reshape(-1)
        
        # For compatibility with existing code structure (rewards_per_func logic)
        rewards_per_func = rewards.unsqueeze(1) 

        # Metrics logging helper
        if self.scale_rewards == "batch":
            std_val = rewards_by_sample.std()
            is_std_zero = torch.isclose(std_val, torch.zeros_like(std_val))
            mean_std_log = std_val.item()
        else:
            is_std_zero = (std_rewards < 1e-6)
            mean_std_log = std_rewards.mean().item()
            
        mean_grouped_rewards_log = mean_rewards.mean().item()

        # Advantages and turn_info are already local (derived from local prompts/generations).
        # We need global advantages only for logging to match gathered prompts.
        # Use gather to collect advantages from all ranks for logging consistency.
        all_process_advantages = self.accelerator.gather(advantages)

        # No slicing needed for local training data as advantages is already local
        self._current_turn_info = turn_info

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        # Note: rewards_per_func is now (num_turns, 1) shape, so we need to handle it differently
        # if rewards_per_func.dim() == 2 and rewards_per_func.size(1) == 1:
        #     # Single reward function case
        #     mean_rewards = torch.nanmean(rewards_per_func.squeeze(1)).item()
        #     for reward_func_name in self.reward_func_names:
        #         self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
        #         self._metrics[mode][f"rewards/{reward_func_name}/std"].append(mean_std_log)
        # else:
        #     for i, reward_func_name in enumerate(self.reward_func_names):
        #         if rewards_per_func.size(1) > i:
        #             mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
        #             self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
        #             std_func_rewards = nanstd(rewards_per_func[:, i]).item()
        #             self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        # self._metrics[mode]["reward"].append(mean_grouped_rewards_log)
        # self._metrics[mode]["reward_std"].append(mean_std_log)
        # self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        gathered_prompts = gather_object(prompts_text)
        gathered_completions = gather_object(completions_text)
        gathered_turn_info = gather_object(turn_info)
        
        self._logs["prompt"].extend(gathered_prompts)
        self._logs["completion"].extend(gathered_completions)
        self._logs["turn_info"].extend(gathered_turn_info)
        # Log rewards
        if rewards_per_func.dim() == 2 and rewards_per_func.size(1) == 1:
            # Single reward function case
            for name in self.reward_func_names:
                self._logs["rewards"][name].extend(rewards_per_func.squeeze(1).tolist())
        else:
            for i, name in enumerate(self.reward_func_names):
                if rewards_per_func.size(1) > i:
                    self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        if images is not None:
            self._logs["images"].extend(gather_object(images))

        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            delta = delta[completion_mask.bool()]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            flat_is_ratio = importance_sampling_ratio[completion_mask.bool()]
            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )

        # [중요] output 딕셔너리 구성 시 'num_items_in_batch' 제거
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages, # advantages는 Flatten된 상태여야 함
            # "num_items_in_batch": num_items_in_batch,  <-- 제거됨 (Shuffle 에러 원인)
            "turn_info": turn_info,
        }
        
        # 필요한 부가 정보 추가
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        
        # kwargs 데이터 전달
        if "pixel_values" in forward_kwargs: output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs: output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs: output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs: output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs: output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if images is not None: output["num_images"] = num_images

        return output

    def compute_liger_loss(self, unwrapped_model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Get the last hidden state of the model
        last_hidden_state = self._get_last_hidden_state(
            unwrapped_model,
            input_ids,
            attention_mask,
            logits_to_keep,
            inputs.get("pixel_values"),
            inputs.get("image_grid_thw"),
            inputs.get("pixel_attention_mask"),
            inputs.get("image_sizes"),
        )

        # compute loss and metrics using liger grpo loss
        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=completion_ids,
            attention_mask=completion_mask,
            advantages=inputs["advantages"],
            bias=unwrapped_model.lm_head.bias,
            old_per_token_logps=inputs.get("old_per_token_logps"),
            ref_per_token_logps=inputs.get("ref_per_token_logps"),
        )
        # Extract metrics from the liger_grpo_loss output
        # KL divergence is the first metric when beta is non-zero
        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]

        mode = "train" if self.model.training else "eval"
        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).mean().item())
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather(clip_ratio).mean().item())
        return loss / self.current_gradient_accumulation_steps
    
    
    # [Helper Function] 특정 어댑터만 grad를 켜고, 나머지는 끄는 함수
    def _set_adapter_grads(self, model, active_adapter_name):
        # model.named_parameters()를 돌면서 public/private 이름에 따라 grad 제어
        # 주의: LoRA가 아닌 Base Model 전체를 학습 중이라면 로직 확인 필요 (현재는 Adapter 학습 가정)
        for name, param in model.named_parameters():
            # 'public' 혹은 'private'이 이름에 포함된 파라미터만 제어
            if "public" in name or "private" in name:
                if active_adapter_name in name:
                    param.requires_grad = True
                else:
                    # 현재 Step의 주인공이 아니면 grad를 꺼버림 -> DDP가 무시함
                    param.requires_grad = False

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        logger.info("compute_loss called")
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        turn_info = inputs.get("turn_info", [])
        
        # 일반 GRPO 처리
        if not turn_info:
            return self._compute_loss(model, inputs)

        public_indices = [i for i, info in enumerate(turn_info) if info[0] == "public"]
        private_indices = [i for i, info in enumerate(turn_info) if info[0] == "private"]
        
        mode = "train" if self.model.training else "eval"
        total_loss_detached = torch.tensor(0.0, device=self.accelerator.device)
        is_ddp_model = hasattr(model, "no_sync")
        
        # ==================================================================
        # DDP에서 두 번의 backward를 분리하여 수행할 때, gradient checkpointing과
        # 함께 사용하면 "undefined gradient" 에러가 발생합니다.
        # 해결: 모든 backward를 no_sync() 안에서 수행하고, 마지막에 수동 동기화
        # ==================================================================
        sync_context = model.no_sync() if is_ddp_model else nullcontext()
        
        with sync_context:
            # ==================================================================
            # STEP 1: Public Agent Update
            # ==================================================================
            self._switch_adapter("public", model)

            if public_indices:
                public_inputs = self._extract_agent_inputs(inputs, public_indices)
                
                if self.use_liger_loss:
                    unwrapped_model = self.accelerator.unwrap_model(model)
                    public_loss = self._forward_redirection(model, unwrapped_model, self.compute_liger_loss, unwrapped_model, public_inputs)
                else:
                    public_loss = self._compute_loss(model, public_inputs)
                
                public_loss_val = public_loss.detach()
                self._metrics[mode]["loss/public"].append(public_loss_val.item())
                total_loss_detached += public_loss_val

                self.accelerator.backward(public_loss)
                
                del public_loss
                if 'public_inputs' in locals(): del public_inputs

            # ==================================================================
            # STEP 2: Private Agent Update
            # ==================================================================
            self._switch_adapter("private", model)

            if private_indices:
                private_inputs = self._extract_agent_inputs(inputs, private_indices)
                
                if self.use_liger_loss:
                    unwrapped_model = self.accelerator.unwrap_model(model)
                    private_loss = self._forward_redirection(model, unwrapped_model, self.compute_liger_loss, unwrapped_model, private_inputs)
                else:
                    private_loss = self._compute_loss(model, private_inputs)

                private_loss_val = private_loss.detach()
                self._metrics[mode]["loss/private"].append(private_loss_val.item())
                total_loss_detached += private_loss_val
                
                self.accelerator.backward(private_loss)
                
                del private_loss
                if 'private_inputs' in locals(): del private_inputs

        # no_sync() 컨텍스트 밖에서 gradient 수동 동기화 수행
        # DDP에서 두 adapter에 대한 backward를 모두 완료한 후 동기화
        if is_ddp_model and model.training and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()

            for name, param in model.named_parameters():
                if param.grad is None:
                    continue

                if not param.requires_grad:
                    continue

                # 3. 수동 All-Reduce (SUM)
                # Mixed Precision 사용 시 param.grad는 Scaled 상태일 수 있으나,
                # 단순히 합치고 나누는 선형 연산이므로 All-Reduce -> Div는 안전합니다.
                torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
                param.grad.div_(world_size)

            # for param in model.parameters():
            #     if param.grad is not None:
            #         torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
            #         param.grad.div_(world_size)

        return total_loss_detached
        
    def _extract_agent_inputs(self, inputs, indices):
        extracted = {}
        for k, v in inputs.items():
            # Tensor인 경우 처리
            if isinstance(v, torch.Tensor):
                # [중요] 스칼라 텐서(dim=0)는 인덱싱 불가능하므로 그대로 복사
                if v.dim() == 0:
                    extracted[k] = v
                # 배치 크기(shape[0])가 prompt_ids와 같은 경우에만 슬라이싱
                elif v.shape[0] == len(inputs["prompt_ids"]):
                    extracted[k] = v[indices]
                else:
                    # shape이 안 맞으면(메타데이터 등) 그냥 복사
                    extracted[k] = v
            
            # 리스트인 경우 처리
            elif isinstance(v, list) and len(v) == len(inputs["prompt_ids"]):
                extracted[k] = [v[i] for i in indices]
            
            # 그 외(문자열, int 등)는 그대로 복사
            else:
                extracted[k] = v
        return extracted

    def _compute_loss(self, model, inputs):
        # ... [모델 Forward 및 Logprob 계산 기존 코드 동일] ...
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown importance sampling level: {self.importance_sampling_level}")

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # Loss Calculation 수정됨
        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dapo":
            # [수정] inputs["num_items_in_batch"] 대신 직접 계산
            # 로컬 배치의 총 토큰 수 계산
            local_total_tokens = completion_mask.sum()
            # DDP 환경인 경우 모든 프로세스의 토큰 수를 합침 (Global Norm)
            global_total_tokens = self.accelerator.gather(local_total_tokens).sum()
            
            # DAPO Normalizer: Total Tokens / Num Processes
            normalizer = global_total_tokens / self.accelerator.num_processes
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        
        # 1. 메트릭 계산 및 logs 딕셔너리 업데이트
        if mode in self._metrics:
            metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items() if len(val) > 0}
            if mode == "eval":
                metrics = {f"eval_{key}": val for key, val in metrics.items()}
            logs = {**logs, **metrics}
            self._metrics[mode].clear()

        # 2. Completions 및 Trajectory 로깅 준비
        # gather_object는 데드락 방지를 위해 모든 프로세스에서 실행
        all_trajectories = gather_object(self._trajectory_buffer) if hasattr(self, "_trajectory_buffer") else []
        
        # 로깅 후 버퍼 비우기
        if hasattr(self, "_trajectory_buffer"):
            self._trajectory_buffer.clear()

        # 메인 프로세스에서만 WandB 로깅 수행
        if self.accelerator.is_main_process:
            if self.log_completions:
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    # (A) Per-Turn Completions Table
                    if len(self._logs["prompt"]) > 0:
                        prompt_list = list(self._logs["prompt"])
                        completion_list = list(self._logs["completion"])
                        advantage_list = list(self._logs["advantages"])
                        turn_info_list = list(self._logs["turn_info"])
                        rewards_dict = {k: list(v) for k, v in self._logs["rewards"].items()}

                        column_lengths = [len(prompt_list), len(completion_list), len(advantage_list), len(turn_info_list)]
                        for v in rewards_dict.values():
                            if len(v) > 0: column_lengths.append(len(v))
                        
                        min_len = min(column_lengths) if column_lengths else 0

                        if min_len > 0:
                            # ... (데이터 가공 로직 동일) ...
                            sample_indices, generation_indices, turn_indices, agent_names = [], [], [], []
                            for info in turn_info_list[:min_len]:
                                if isinstance(info, (list, tuple)) and len(info) >= 4:
                                    agent_names.append(str(info[0]))
                                    turn_indices.append(int(info[1]))
                                    sample_indices.append(int(info[2]))
                                    generation_indices.append(int(info[3]))
                                else:
                                    agent_names.append("unknown"); turn_indices.append(-1); sample_indices.append(-1); generation_indices.append(-1)

                            table_data = {
                                "step": [self.state.global_step] * min_len,
                                "sample_idx": sample_indices,
                                "generation_idx": generation_indices,
                                "turn_idx": turn_indices,
                                "agent": agent_names,
                                "prompt": prompt_list[:min_len],
                                "completion": completion_list[:min_len],
                                "advantage": advantage_list[:min_len],
                            }
                            for r_k, r_v in rewards_dict.items():
                                if len(r_v) >= min_len: table_data[f"reward_{r_k}"] = r_v[:min_len]

                            if self._logs["images"]:
                                images_list = list(self._logs["images"])
                                if len(images_list) >= min_len:
                                    table_data["images"] = [[wandb.Image(img) for img in imgs] if imgs else [] for imgs in images_list[:min_len]]

                            try:
                                import pandas as pd
                                df = pd.DataFrame(table_data)
                                if self.wandb_log_unique_prompts:
                                    df = df.drop_duplicates(subset=["prompt", "turn_idx", "sample_idx", "generation_idx"])

                                table = wandb.Table(dataframe=df)
                                
                                # [FIX] Key 이름을 고정하고, commit=False를 사용하여 HF 로그와 병합
                                wandb.log({"generation_log/completions": table}, step=self.state.global_step, commit=False)
                                
                            except Exception as e:
                                logger.error(f"Failed to log completions table: {e}", exc_info=True)

                    # (B) Trajectories Table
                    if all_trajectories:
                        try:
                            import pandas as pd
                            flat_trajectories = []
                            for item in all_trajectories:
                                if isinstance(item, list): flat_trajectories.extend(item)
                                elif isinstance(item, dict): flat_trajectories.append(item)
                            
                            if flat_trajectories:
                                df_traj = pd.DataFrame(flat_trajectories)
                                traj_table = wandb.Table(dataframe=df_traj)
                                
                                # [FIX] Key 이름을 고정하고, commit=False 사용
                                wandb.log({"generation_log/trajectories": traj_table}, step=self.state.global_step, commit=False)
                        except Exception as e:
                            logger.error(f"Failed to log trajectories table: {e}", exc_info=True)

        # 3. JSON 저장 (기존 유지)
        if self.accelerator.is_main_process:
            self._save_completions_to_json(force=False)
        
        # 4. _logs 버퍼 클리어 (기존 유지)
        for key in self._logs:
            if isinstance(self._logs[key], deque): self._logs[key].clear()
            elif isinstance(self._logs[key], dict):
                for sub_key in self._logs[key]:
                    if isinstance(self._logs[key][sub_key], deque): self._logs[key][sub_key].clear()

        # 5. 부모 클래스 log 호출 (여기서 최종 commit=True가 발생하여 위 테이블과 메트릭이 함께 업로드됨)
        super().log(logs, start_time)

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        # Save any remaining completions before checkpoint
        if self.accelerator.is_main_process:
            self._save_completions_to_json(force=True)
        
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
        
    def _switch_adapter(self, adapter_name: str, model=None):
        """Safely switch the active LoRA adapter."""
        if model is None:
            model = self.model
            
        unwrapped_model = self.accelerator.unwrap_model(model)
        
        # Handle PeftModel wrapping
        if is_peft_model(unwrapped_model):
            unwrapped_model.set_adapter(adapter_name)
            return

        # Handle distributed wrapping (DDP/FSDP)
        if hasattr(model, "module"):
            if is_peft_model(model.module):
                model.module.set_adapter(adapter_name)
                return
        
        if is_peft_model(model):
            model.set_adapter(adapter_name)
            return
            
        logger.warning(f"Could not switch adapter to {adapter_name}. Model might not be a PeftModel.")
        
    def _extract_answer_content(self, content: str) -> str:
        """Extract content between <answer> and </answer> tags."""
        
        if "<answer>" in content:
            if "</answer>" in content:
                model_answer = content.split("<answer>")[-1].replace("</answer>", "")
            else:
                model_answer = content.split("<answer>")[-1]
        else:
            input_size = self.max_prompt_length // 2
            model_answer = content[-input_size:]    
        
        return model_answer
        
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        # compute_loss 내부에서 수동으로 backward를 수행하므로,
        # 여기서는 반환된 loss(logging용)를 가지고 후처리만 하면 됩니다.
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        # DDP 동기화를 위해 loss를 평균내어 로깅 준비 (backward는 이미 끝남) 
        if self.args.n_gpu > 1:
            loss = loss.mean()
        # loss는 이미 detach된 텐서여야 합니다 (compute_loss 수정 참조).
        return loss

    def _compute_pass_at_k(self, n: int, c: int, k: int) -> float:
        """
        Compute pass@k metric.
        
        Args:
            n: Total number of samples per problem
            c: Number of correct samples among n
            k: Number of samples to consider for pass@k
            
        Returns:
            pass@k probability
        """
        if n - c < k:
            return 1.0
        # Use the formula: pass@k = 1 - C(n-c, k) / C(n, k)
        # Computed as: 1 - prod((n-c-i)/(n-i) for i in range(k))
        result = 1.0
        for i in range(k):
            result *= (n - c - i) / (n - i)
        return 1.0 - result

    @torch.no_grad()
    def multi_generate_eval(
        self,
        eval_data: Union[Dataset, List[dict]],
        num_samples_per_problem: int = 10,
        k_values: List[int] = [1, 5, 10],
        num_agents: Optional[int] = None,
        max_completion_length_public: Optional[int] = None,
        max_completion_length_private: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Evaluate multi-agent generation performance using pass@k metric.
        
        This function runs multi-turn generation with public/private agents alternating,
        then compares the final private agent's answer with the ground truth answer.
        
        Args:
            eval_data: Evaluation dataset or list of dicts with 'problem' and 'answer' keys
            num_samples_per_problem: Number of generation samples per problem (n in pass@k)
            k_values: List of k values for pass@k computation (e.g., [1, 5, 10])
            num_turns: Number of turns (public-private pairs). If None, uses self.num_turns
            max_completion_length_public: Max completion length for public agent. 
                                          If None, uses self.public_agent_max_completion_length
            max_completion_length_private: Max completion length for private agent.
                                           If None, uses self.private_agent_max_completion_length
            batch_size: Batch size for processing. If None, uses all data at once.
            verbose: Whether to print progress information
            
        Returns:
            Dictionary containing:
                - pass@k scores for each k value
                - per_problem_results: detailed results per problem
                - overall_accuracy: simple accuracy (pass@1 equivalent)
                - total_problems: number of problems evaluated
                - num_samples_per_problem: n value used
        """
        device = self.accelerator.device
        self.model.eval()
        
        # Set default values
        if num_agents is None:
            num_agents = self.num_agents
        num_turns = num_agents * 2
        
        if max_completion_length_public is None:
            max_completion_length_public = self.public_agent_max_completion_length
        if max_completion_length_private is None:
            max_completion_length_private = self.private_agent_max_completion_length
        if batch_size is None:
            batch_size = len(eval_data)

        # Convert dataset to list if needed
        if isinstance(eval_data, Dataset):
            eval_data = [eval_data[i] for i in range(len(eval_data))]
        
        # Extract problems and answers
        problems = [item.get("problem", item.get("prompt", "")) for item in eval_data]
        answers = [item.get("answer", "") for item in eval_data]
        
        num_problems = len(problems)
        
        # Store results for each problem
        all_results = []
        
        # Process in batches
        for batch_start in range(0, num_problems, batch_size):
            batch_end = min(batch_start + batch_size, num_problems)
            batch_problems = problems[batch_start:batch_end]
            batch_answers = answers[batch_start:batch_end]
            
            # Generate multiple samples per problem
            batch_results = self._generate_multi_turn_eval_batch(
                problems=batch_problems,
                answers=batch_answers,
                num_samples=num_samples_per_problem,
                num_turns=num_turns,
                max_completion_length_public=max_completion_length_public,
                max_completion_length_private=max_completion_length_private,
                verbose=verbose,
            )
            
            all_results.extend(batch_results)
        
        # Compute pass@k metrics
        pass_at_k_results = {}
        for k in k_values:
            if k > num_samples_per_problem:
                if verbose and self.accelerator.is_main_process:
                    logger.warning(f"k={k} > num_samples_per_problem={num_samples_per_problem}, skipping")
                continue
            
            pass_at_k_sum = 0.0
            for result in all_results:
                n = result["num_samples"]
                c = result["num_correct"]
                pass_at_k_sum += self._compute_pass_at_k(n, c, k)
            
            pass_at_k_results[f"pass@{k}"] = pass_at_k_sum / num_problems
        
        # Compute overall accuracy (proportion of problems with at least one correct answer)
        problems_with_correct = sum(1 for r in all_results if r["num_correct"] > 0)
        overall_accuracy = problems_with_correct / num_problems if num_problems > 0 else 0.0
        
        # Compute average correct rate per problem
        avg_correct_rate = sum(r["num_correct"] / r["num_samples"] for r in all_results) / num_problems if num_problems > 0 else 0.0
        
        final_results = {
            **pass_at_k_results,
            "overall_accuracy": overall_accuracy,
            "avg_correct_rate": avg_correct_rate,
            "total_problems": num_problems,
            "num_samples_per_problem": num_samples_per_problem,
            "num_agents": num_agents,
            "per_problem_results": all_results,
        }
        
        if verbose and self.accelerator.is_main_process:
            logger.info(f"Evaluation complete:")
            for k, v in final_results.items():
                if k != "per_problem_results":
                    logger.info(f"  {k}: {v}")
        
        return final_results

    def _convert_messages_to_string(self, messages: list) -> str:
        """Convert a list of message dicts to a string using chat template."""
        if hasattr(self.processing_class, "apply_chat_template"):
            try:
                return self.processing_class.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        # Fallback: simple string conversion
        return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)

    def _generate_eval_single_turn(
        self, 
        prompts_as_messages: List[list], 
        max_completion_length: int
    ) -> List[str]:
        """
        Generate completions for evaluation, handling message-to-string conversion.
        
        Args:
            prompts_as_messages: List of message lists
            max_completion_length: Maximum tokens to generate
            
        Returns:
            List of decoded completion strings
        """
        device = self.accelerator.device
        
        # Convert all messages to strings
        prompts_text = [self._convert_messages_to_string(msgs) for msgs in prompts_as_messages]
        
        # Use vLLM if enabled
        if self.use_vllm:
            if self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode:
                torch.cuda.empty_cache()
                self.llm.wake_up()
            
            # Sync weights if adapter changed
            active_adapter = self.model.active_adapter if is_peft_model(self.model) else None
            need_resync = self.state.global_step != self._last_loaded_step or active_adapter != self._last_loaded_adapter
            if need_resync:
                self._move_model_to_vllm(force=True)
                self._last_loaded_step = self.state.global_step
            
            if self.vllm_mode == "colocate":
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None
                
                sampling_params = SamplingParams(
                    n=1,
                    repetition_penalty=self.repetition_penalty,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=-1 if self.top_k is None else self.top_k,
                    min_p=0.0 if self.min_p is None else self.min_p,
                    max_tokens=max_completion_length,
                    truncate_prompt_tokens=self.max_prompt_length,
                    guided_decoding=guided_decoding,
                )
                
                all_outputs = self.llm.generate(prompts_text, sampling_params=sampling_params, use_tqdm=False)
                completion_ids = [output.outputs[0].token_ids for output in all_outputs]
                
                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=1)
                    
                decoded = self.processing_class.batch_decode(completion_ids, skip_special_tokens=False)
            else:
                # Server mode
                all_prompts_text = gather_object(prompts_text)
                
                if self.accelerator.is_main_process:
                    output = self.vllm_client.generate(
                        prompts=all_prompts_text,
                        n=1,
                        repetition_penalty=self.repetition_penalty,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=-1 if self.top_k is None else self.top_k,
                        min_p=0.0 if self.min_p is None else self.min_p,
                        max_tokens=max_completion_length,
                        truncate_prompt_tokens=self.max_prompt_length,
                        guided_decoding_regex=self.guided_decoding_regex,
                    )
                    all_completion_ids = output["completion_ids"]
                else:
                    all_completion_ids = None
                
                obj_list = [all_completion_ids]
                broadcast_object_list(obj_list, from_process=0)
                all_completion_ids = obj_list[0]
                
                process_slice = slice(
                    self.accelerator.process_index * len(prompts_text),
                    (self.accelerator.process_index + 1) * len(prompts_text),
                )
                completion_ids = all_completion_ids[process_slice]
                decoded = self.processing_class.batch_decode(completion_ids, skip_special_tokens=False)
        else:
            # Use transformers generation
            generate_inputs = self.processing_class(
                text=prompts_text,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                max_length=self.max_prompt_length,
                truncation=True,
                add_special_tokens=False,
            )
            generate_inputs = {k: v.to(device) for k, v in generate_inputs.items()}
            
            gen_config_dict = {k: v for k, v in self.generation_config.to_dict().items()}
            gen_config_dict.pop('max_new_tokens', None)
            generation_config = GenerationConfig(
                **gen_config_dict,
                max_new_tokens=max_completion_length
            )
            
            with torch.no_grad():
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                prompt_completion_ids = unwrapped_model.generate(
                    **generate_inputs, 
                    generation_config=generation_config
                )
            
            prompt_length = generate_inputs["input_ids"].size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]
            decoded = self.processing_class.batch_decode(completion_ids, skip_special_tokens=False)
        
        return decoded

    def _generate_multi_turn_eval_batch(
        self,
        problems: List[str],
        answers: List[str],
        num_samples: int,
        num_turns: int,
        max_completion_length_public: int,
        max_completion_length_private: int,
        verbose: bool = True,
    ) -> List[dict]:
        """
        Generate multi-turn completions for a batch of problems and evaluate correctness.

        Args:
            problems: List of problem strings
            answers: List of answer strings
            num_samples: Number of samples to generate per problem
            num_turns: Total number of turns (public + private)
            max_completion_length_public: Max tokens for public agent
            max_completion_length_private: Max tokens for private agent
            verbose: Whether to print progress

        Returns:
            List of result dicts, one per problem, containing:
                - problem: the problem text
                - answer: the ground truth answer
                - num_samples: number of samples generated
                - num_correct: number of correct samples
                - sample_results: list of (final_answer, is_correct) tuples
                - trajectories: list of full trajectories for debugging
        """
        num_problems = len(problems)
        num_agents = num_turns // 2

        # Safety check
        if len(problems) != len(answers):
            raise ValueError(f"Problems and answers must have same length.")

        # Initialize histories: [num_problems][num_samples] -> list of (agent_name, answer)
        all_histories = [[[] for _ in range(num_samples)] for _ in range(num_problems)]
        
        if verbose and self.accelerator.is_main_process:
            logger.info(f"Starting evaluation generation: {num_problems} problems, "
                       f"{num_samples} samples each, {num_turns} turns")

        with torch.no_grad():
            for turn_idx in range(num_turns):
                is_public_turn = (turn_idx % 2 == 0)
                agent_name = "public" if is_public_turn else "private"
                remaining_agents = num_agents - (turn_idx // 2)
                
                # Switch adapter
                self.accelerator.wait_for_everyone()
                self._switch_adapter(agent_name, self.model)
                self.accelerator.wait_for_everyone()
                
                # Build prompts for all problem-sample combinations
                turn_prompts = []
                for prob_idx, problem in enumerate(problems):
                    for sample_idx in range(num_samples):
                        hist = all_histories[prob_idx][sample_idx]
                        
                        last_public_output = next(
                            (out for agent, out in reversed(hist) if agent == "public"), 
                            None
                        )
                        last_private_output = next(
                            (out for agent, out in reversed(hist) if agent == "private"), 
                            None
                        )
                        
                        if is_public_turn:
                            prev_outputs_str = "No previous outputs yet."
                            formatted_outputs = []
                            if last_public_output:
                                formatted_outputs.append(f"Previous Orchestrator Output:\n{last_public_output}")
                            if last_private_output:
                                formatted_outputs.append(f"Previous Worker Agent Output:\n{last_private_output}")
                            if formatted_outputs:
                                prev_outputs_str = "\n\n".join(formatted_outputs)
                            
                            content = PUBLIC_PROMPT.format(
                                original_problem=problem,
                                previous_outputs=prev_outputs_str,
                                num_agents=remaining_agents
                            )
                            system_prompt = PUBLIC_SYSTEM_PROMPT
                        else:
                            content = PRIVATE_PROMPT.format(
                                original_problem=problem,
                                orchestrator_instruction=last_public_output if last_public_output else ""
                            )
                            system_prompt = PRIVATE_SYSTEM_PROMPT
                        
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": content}
                        ]
                        turn_prompts.append(messages)
                
                # Generate completions
                max_len = max_completion_length_public if is_public_turn else max_completion_length_private
                
                self.accelerator.wait_for_everyone()
                decoded = self._generate_eval_single_turn(turn_prompts, max_len)
                self.accelerator.wait_for_everyone()
                
                # Update histories
                for i, content in enumerate(decoded):
                    prob_idx = i // num_samples
                    sample_idx = i % num_samples
                    answer = self._extract_answer_content(content)
                    all_histories[prob_idx][sample_idx].append((agent_name, answer))
                
                if verbose and self.accelerator.is_main_process:
                    logger.info(f"  Turn {turn_idx + 1}/{num_turns} ({agent_name}) completed")

        # Evaluate final answers
        results = []
        for prob_idx in range(num_problems):
            problem = problems[prob_idx]
            answer = answers[prob_idx]

            sample_results = []
            trajectories = []
            num_correct = 0

            for sample_idx in range(num_samples):
                hist = all_histories[prob_idx][sample_idx]
                trajectories.append(hist)

                # Get the last private agent's answer
                final_answer = None
                for agent, answer in reversed(hist):
                    if agent == "private":
                        final_answer = answer
                        break

                if final_answer is None:
                    final_answer = hist[-1][1] if hist else ""

                # Check correctness
                completion_for_reward = [[{"role": "assistant", "content": final_answer}]]
                try:
                    reward = answer_tag_reward_fn(completions=completion_for_reward, solution=[answer])
                    is_correct = reward[0] == 1.0 if reward[0] is not None else False
                except Exception as e:
                    if verbose and self.accelerator.is_main_process:
                        logger.warning(f"Error evaluating: {e}")
                    is_correct = False

                if is_correct:
                    num_correct += 1

                sample_results.append({
                    "final_answer": final_answer,
                    "is_correct": is_correct,
                })
            
            results.append({
                "problem": problem,
                "answer": answer,
                "num_samples": num_samples,
                "num_correct": num_correct,
                "sample_results": sample_results,
                "trajectories": trajectories,
            })
        
        return results

    def run_multi_agent_evaluation(
        self,
        eval_dataset: Optional[Union[Dataset, List[dict]]] = None,
        num_samples_per_problem: int = 10,
        k_values: List[int] = [1, 5, 10],
        output_path: Optional[str] = None,
        **kwargs
    ) -> dict[str, Any]:
        """
        Convenience method to run multi-agent evaluation and optionally save results.
        
        Args:
            eval_dataset: Dataset to evaluate. If None, uses self.eval_dataset
            num_samples_per_problem: Number of samples per problem for pass@k
            k_values: List of k values for pass@k metrics
            output_path: Path to save results as JSON. If None, results are only returned.
            **kwargs: Additional arguments passed to multi_generate_eval
            
        Returns:
            Evaluation results dictionary
        """
        if eval_dataset is None:
            if self.eval_dataset is None:
                raise ValueError("No evaluation dataset provided and self.eval_dataset is None")
            eval_dataset = self.eval_dataset
        
        results = self.multi_generate_eval(
            eval_data=eval_dataset,
            num_samples_per_problem=num_samples_per_problem,
            k_values=k_values,
            **kwargs
        )
        
        if output_path is not None and self.accelerator.is_main_process:
            import json
            # Remove non-serializable items for JSON export
            export_results = {k: v for k, v in results.items() if k != "per_problem_results"}
            
            # Add simplified per-problem results
            export_results["per_problem_summary"] = [
                {
                    "problem": r["problem"][:200] + "..." if len(r["problem"]) > 200 else r["problem"],
                    "num_correct": r["num_correct"],
                    "num_samples": r["num_samples"],
                    "accuracy": r["num_correct"] / r["num_samples"]
                }
                for r in results["per_problem_results"]
            ]
            
            with open(output_path, "w") as f:
                json.dump(export_results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        
        return results

    def _collect_completions_for_json(
        self,
        problems: List[str],
        answers: List[str],
        completions_per_trajectory: List[List],
        rewards_per_trajectory: torch.Tensor,
        acc_scores: Optional[torch.Tensor] = None,
        format_rewards: Optional[torch.Tensor] = None,
    ):
        """
        Collect completion data for JSON saving.
        This method is called from _generate_and_score_completions.
        
        Args:   
            problems: List of problem texts
            answers: List of answer texts
            completions_per_trajectory: List of trajectories, each containing turns
            rewards_per_trajectory: Tensor of rewards for each trajectory
            acc_scores: Tensor of accuracy scores (optional)
            format_rewards: Tensor of format rewards (optional)
        """
        num_samples = len(problems)
        
        for sample_idx in range(num_samples):
            sample_data = {
                "step": self.state.global_step,
                "problem": problems[sample_idx],
                "answer": answers[sample_idx],
                "generations": []
            }
            
            for gen_idx in range(self.num_generations):
                traj_idx = sample_idx * self.num_generations + gen_idx
                
                # Safety check for trajectory index
                if traj_idx >= len(completions_per_trajectory):
                    logger.warning(f"Trajectory index {traj_idx} out of bounds, skipping")
                    continue
                    
                traj_content = completions_per_trajectory[traj_idx]
                
                generation_data = {
                    "generation_idx": gen_idx,
                    "reward": rewards_per_trajectory[traj_idx].item() if traj_idx < len(rewards_per_trajectory) else 0.0,
                    "turns": []
                }
                
                # Add accuracy and format rewards if available (use the tensors directly)
                if acc_scores is not None and traj_idx < len(acc_scores):
                    generation_data["accuracy_reward"] = acc_scores[traj_idx].item()
                if format_rewards is not None and traj_idx < len(format_rewards):
                    generation_data["format_reward"] = format_rewards[traj_idx].item()
                
                # Extract turn information
                for turn_idx, turn_data in enumerate(traj_content):
                    agent = "public" if turn_idx % 2 == 0 else "private"
                    
                    if isinstance(turn_data, list) and len(turn_data) > 0 and isinstance(turn_data[0], dict):
                        text = turn_data[0].get("content", "")
                    else:
                        text = str(turn_data)
                    
                    turn_info = {
                        "turn_idx": turn_idx,
                        "agent": agent,
                        "completion": text
                    }
                    generation_data["turns"].append(turn_info)
                
                sample_data["generations"].append(generation_data)
            
            self._completions_to_save.append(sample_data)
    
    def _save_completions_to_json(self, force: bool = False):
        """
        Save collected completions to a single JSON file.
        Appends new completions to existing file if it exists.
        Called periodically during training.
        
        Args:
            force: If True, save even if buffer is small
        """
        if not self.save_completions or not self.accelerator.is_main_process:
            return
        
        if not self._completions_to_save:
            return
        
        # Save every logging step or when forced (at least 1 sample)
        if not force and len(self._completions_to_save) < 1:
            return
        
        import json
        from datetime import datetime
        
        # Ensure directory exists
        os.makedirs(self.save_completions_path, exist_ok=True)
        
        # Use a single file for all completions
        filename = "all_completions.json"
        filepath = os.path.join(self.save_completions_path, filename)
        
        # Load existing data if file exists
        existing_data = {"metadata": {}, "completions": []}
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load existing completions file, starting fresh: {e}")
                existing_data = {"metadata": {}, "completions": []}
        
        # Append new completions
        existing_data["completions"].extend(self._completions_to_save)
        
        # Update metadata
        step = self.state.global_step
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        existing_data["metadata"] = {
            "last_updated_step": step,
            "last_updated_timestamp": timestamp,
            "total_samples": len(existing_data["completions"]),
            "num_generations": self.num_generations,
            "num_turns": self.num_turns,
            "num_agents": self.num_agents,
            "model_name": getattr(self.model.config, "_name_or_path", "unknown"),
        }
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self._completions_to_save)} completions to {filepath} (total: {len(existing_data['completions'])})")
        except Exception as e:
            logger.error(f"Failed to save completions to JSON: {e}")
        
        # Clear the buffer after saving
        self._completions_to_save.clear()
        