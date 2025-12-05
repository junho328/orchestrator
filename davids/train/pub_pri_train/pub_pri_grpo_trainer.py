import inspect
import os
import re
import textwrap
from collections import defaultdict, deque
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Union, List

import datasets
import torch
import torch.utils.data
import transformers
from accelerate import logging
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
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
from davids.reward_utils.math_reward import accuracy_reward
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
            architecture = getattr(transformers, config.architectures[0])
            model = architecture.from_pretrained(model_id, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
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
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
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
                max_completion_length=self.max_completion_length,
            )

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # Keep logs sized to the generation batch to record only outputs from the latest model update.
        self._logs = {
            "images": deque(maxlen=args.generation_batch_size),
            "prompt": deque(maxlen=args.generation_batch_size),
            "completion": deque(maxlen=args.generation_batch_size),
            "rewards": defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
            "advantages": deque(maxlen=args.generation_batch_size),
        }

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
                    load_format="bitsandbytes",
                    quantization="bitsandbytes",
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
        
        if self.accelerator.is_main_process:
            logger.info(f"Creating RepeatSampler: dataset_size={len(dataset) if hasattr(dataset, '__len__') else 'unknown'}, "
                       f"mini_repeat_count={self.num_generations}, batch_size={self.args.generation_batch_size // self.num_generations}, "
                       f"repeat_count={self.num_iterations * self.args.steps_per_generation}")
        
        sampler = RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )
        
        if self.accelerator.is_main_process:
            logger.info("RepeatSampler created successfully")
        
        return sampler

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

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

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param.data)])

    def _sync_fsdp2_params_to_vllm(self, module: nn.Module):
        # For FSDP2, module.state_dict() already covers all parameters, so no need for recursion
        for name, param in module.state_dict().items():
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
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
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
                            llm_model.load_weights([(name, param.data)])
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
                    name = self._fix_param_name_to_vllm(name)
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()

        # Remember which adapter was used for this sync (active_adapter exists on PeftModel)
        if is_peft_model(self.model):
            self._last_loaded_adapter = self.model.active_adapter

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
    def _calculate_trajectory_rewards(self, inputs, prompts, completions_per_trajectory, completion_ids_list, turn_info):
        """
        Calculate rewards per trajectory.
        Accuracy (1.0): Check if LAST Private Answer == Solution.
        Format (0.5): Check if ALL turns have <think> and <answer>.
        """
        device = self.accelerator.device
        num_trajectories = len(completions_per_trajectory)
        rewards = torch.zeros(num_trajectories, device=device)
        
        # Map flat input structure back to problems/solutions
        # inputs contains the original raw data items (len = num_samples)
        solutions = [x.get("solution", "") for x in inputs]
        
        for traj_idx in range(num_trajectories):
            sample_idx = traj_idx // self.num_generations
            solution = solutions[sample_idx]
            trajectory_completions = completions_per_trajectory[traj_idx] # list of strings (contents)

            # 1. Format Reward (Check all turns)
            all_formatted = True
            for content in trajectory_completions:
                # Handle dictionary format if conversational
                if isinstance(content, list): content = content[0]["content"]
                
                # Use the imported reward function logic
                # Assuming think_answer_format_reward returns list of floats
                fmt_score = think_answer_format_reward([[{"role": "assistant", "content": content}]])[0]
                if fmt_score < 0.5:
                    all_formatted = False
                    break
            format_reward = 0.5 if all_formatted else 0.0

            # 2. Accuracy Reward (Check last private turn)
            # Find last private output in this trajectory
            last_private_content = None
            # Reconstruct turn info for this trajectory
            # Trajectory completions are ordered by time (Turn0, Turn1...)
            # We iterate backwards
            for i in range(len(trajectory_completions) - 1, -1, -1):
                # Turn info order in flattened list: [Turn0_AllSamples..., Turn1_AllSamples...]
                # This is tricky to map directly from completions_per_trajectory list.
                # Easier: just check the turn index logic (Even=Public, Odd=Private)
                # Last Private turn is always index 2*num_agents - 1 or generally odd
                if i % 2 != 0: # Odd index = Private
                    last_private_content = trajectory_completions[i]
                    if isinstance(last_private_content, list): last_private_content = last_private_content[0]["content"]
                    break
            
            accuracy = 0.0
            if last_private_content:
                acc_scores = accuracy_reward(
                    completions=[[{"role": "assistant", "content": last_private_content}]],
                    solution=[solution]
                )
                accuracy = acc_scores[0] if acc_scores[0] is not None else 0.0

            rewards[traj_idx] = accuracy + format_reward
            
        return rewards

    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        
        # 1. Trajectory별 Reward 계산
        rewards_per_trajectory = self._calculate_trajectory_rewards(
            inputs, prompts, completions, completion_ids_list, self._current_turn_info
        )
        
        # 2. GRPO 정규화
        num_samples = len(prompts)
        rewards_grouped = rewards_per_trajectory.view(num_samples, self.num_generations)
        
        mean_rewards = rewards_grouped.mean(dim=1, keepdim=True)
        std_rewards = rewards_grouped.std(dim=1, keepdim=True)
        
        if self.scale_rewards == "group":
            advantages_grouped = (rewards_grouped - mean_rewards) / (std_rewards + 1e-4)
        elif self.scale_rewards == "batch":
            batch_mean = rewards_per_trajectory.mean()
            batch_std = rewards_per_trajectory.std()
            advantages_grouped = (rewards_grouped - batch_mean) / (batch_std + 1e-4)
        else:
            advantages_grouped = rewards_grouped - mean_rewards

        # (Num_Trajectories,) 형태로 펼침
        advantages_per_trajectory = advantages_grouped.view(-1)

        # 3. 데이터 정렬 (Vectorized Operation으로 최적화)
        # Trajectory별 점수를 턴 횟수만큼 반복
        advantages = advantages_per_trajectory.repeat(self.num_turns)

        # 4. Distributed Training 처리 (Process Slice)
        num_local_turns = len(prompts) * self.num_generations * self.num_turns
        process_slice = slice(
            self.accelerator.process_index * num_local_turns,
            (self.accelerator.process_index + 1) * num_local_turns,
        )
        
        all_process_advantages = advantages.clone()
        
        if len(advantages) > process_slice.stop:
            advantages = advantages[process_slice]
        else:
            start_idx = min(process_slice.start, len(advantages))
            advantages = advantages[start_idx:]

        # 5. Logging
        mode = "train" if self.model.training else "eval"
        rewards_per_func = advantages.unsqueeze(1) 

        self._metrics[mode]["reward"].append(rewards_grouped.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._logs["advantages"].extend(all_process_advantages.tolist())
        
        # Reward 로깅 (Trajectory-Major -> Turn-Major 변환하여 기록)
        expanded_rewards = rewards_per_trajectory.repeat(self.num_turns)
        
        if self.reward_func_names:
            name = self.reward_func_names[0]
            self._logs["rewards"][name].extend(expanded_rewards.tolist())

        return rewards_per_func

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

    def _generate(self, prompts: list[str], images: Optional[list]):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Use multi-turn generation for public-private agent setup
        # Extract problems and solutions from inputs if available
        problems = getattr(self, '_current_problems', prompts)
        solutions = getattr(self, '_current_solutions', [""] * len(prompts))
        prompt_ids, completion_ids, logprobs, forward_kwargs, turn_info = self._generate_multi_turn(prompts, images, problems, solutions)

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

        # Extract problem and solution from inputs
        problems = [x.get("problem", x.get("prompt", "")) for x in inputs]
        solutions = [x.get("solution", "") for x in inputs]
        
        # Store for use in _generate
        self._current_problems = problems
        self._current_solutions = solutions
        
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
             # ... (old_per_token_logps 계산 등 기존 코드 유지) ...
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
        
        # Group completions by trajectory using batch operations
        # Structure: num_samples prompts -> each generates num_generations trajectories -> each trajectory has num_turns
        # So total completions = num_samples * num_generations * num_turns
        num_samples = len(prompts)  # Original number of samples
        total_completions = num_samples * self.num_generations * self.num_turns
        
        # Reshape completions_text into (num_samples, num_generations, num_turns) structure
        # First, ensure we have the right number of completions
        if len(completions_text) < total_completions:
            # Pad if necessary
            completions_text.extend([""] * (total_completions - len(completions_text)))
        
        # Reshape using tensor operations for efficiency
        completions_array = completions_text[:total_completions]
        completions_reshaped = [
            completions_array[sample_idx * self.num_generations * self.num_turns + 
                             gen_idx * self.num_turns:
                             sample_idx * self.num_generations * self.num_turns + 
                             gen_idx * self.num_turns + self.num_turns]
            for sample_idx in range(num_samples)
            for gen_idx in range(self.num_generations)
        ]
        
        # Convert to the expected format
        completions_per_trajectory = []
        for trajectory_completions in completions_reshaped:
            formatted_trajectory = []
            for completion_text in trajectory_completions:
                if is_conversational(inputs[0]):
                    formatted_trajectory.append([{"role": "assistant", "content": completion_text}])
                else:
                    formatted_trajectory.append(completion_text)
            completions_per_trajectory.append(formatted_trajectory)
        
        # Calculate rewards for each trajectory
        # Accuracy reward: check only the last private agent output
        # Format reward: check all turns
        rewards_per_trajectory = self._calculate_trajectory_rewards(
            inputs, prompts, completions_per_trajectory, completion_ids_list, turn_info
        )
        
        rewards_per_trajectory = rewards_per_trajectory.unsqueeze(1) 
        
        # 2. (Num_Trajectories, Num_Turns) 형태로 확장 (모든 턴에 같은 Trajectory Reward 복사)
        rewards_matrix = rewards_per_trajectory.repeat(1, self.num_turns)
        
        # 3. Transpose하여 (Num_Turns, Num_Trajectories) 형태로 변경
        #    이렇게 해야 Flatten 했을 때 [Turn0_AllTrajs, Turn1_AllTrajs...] 순서가 됨
        rewards_matrix_t = rewards_matrix.transpose(0, 1)
        
        # 4. Flatten하여 1차원 텐서로 변환
        rewards = rewards_matrix_t.reshape(-1)
        
        # For compatibility with existing code, create rewards_per_func structure
        # Shape: (num_samples * num_generations * num_turns, 1)
        rewards_per_func = rewards.unsqueeze(1) if rewards.dim() == 1 else rewards

        # Apply weights to each reward function's output and sum
        # Note: rewards_per_func is already a single column, so we just use it directly
        rewards = rewards_per_func.squeeze(1) if rewards_per_func.dim() > 1 else rewards_per_func

        # Compute grouped-wise rewards
        # Group by trajectory: each trajectory has num_turns turns
        # Structure: num_samples * num_generations trajectories, each with num_turns
        num_trajectories = len(prompts) * self.num_generations
        rewards_reshaped = rewards.view(num_trajectories, self.num_turns)
        mean_grouped_rewards = rewards_reshaped.mean(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_turns, dim=0)
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards in ["group", "none"]:
            # If self.scale_rewards = "none", we'll still log group level std
            std_rewards = rewards_reshaped.std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_turns, dim=0)
        elif self.scale_rewards == "batch":
            # Compute global std
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(
                f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
            )

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        # Slice to keep only the local part of the data
        # Note: advantages now has shape (num_trajectories * num_turns,)
        # But we need to account for distributed training
        num_local_samples = len(prompts)
        num_local_turns = num_local_samples * self.num_generations * self.num_turns
        process_slice = slice(
            self.accelerator.process_index * num_local_turns,
            (self.accelerator.process_index + 1) * num_local_turns,
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        if len(advantages) > process_slice.stop:
            advantages = advantages[process_slice]
        else:
            # If slice is out of bounds, just use what we have
            start_idx = min(process_slice.start, len(advantages))
            advantages = advantages[start_idx:]
        
        # Store turn_info for loss computation (slice to match advantages)
        if len(turn_info) > process_slice.stop:
            self._current_turn_info = turn_info[process_slice.start:process_slice.stop]
        else:
            self._current_turn_info = turn_info[start_idx:] if 'start_idx' in locals() else turn_info

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        # Note: rewards_per_func is now (num_turns, 1) shape, so we need to handle it differently
        if rewards_per_func.dim() == 2 and rewards_per_func.size(1) == 1:
            # Single reward function case
            mean_rewards = torch.nanmean(rewards_per_func.squeeze(1)).item()
            for reward_func_name in self.reward_func_names:
                self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
                self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards.mean().item())
        else:
            for i, reward_func_name in enumerate(self.reward_func_names):
                if rewards_per_func.size(1) > i:
                    mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
                    self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
                    std_func_rewards = nanstd(rewards_per_func[:, i]).item()
                    self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
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
            "advantages": rewards_per_func.squeeze(1) if rewards_per_func.dim() > 1 else rewards_per_func, # advantages는 Flatten된 상태여야 함
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
    
    
    # [수정된 compute_loss] OOM 방지를 위한 순차적 Backward 및 메모리 정리 적용
    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        logger.info("compute_loss called")
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        turn_info = inputs.get("turn_info", [])
        
        # 일반 GRPO (turn_info 없음) 처리
        if not turn_info:
            return self._compute_loss(model, inputs)

        public_indices = [i for i, info in enumerate(turn_info) if info[0] == "public"]
        private_indices = [i for i, info in enumerate(turn_info) if info[0] == "private"]
        
        mode = "train" if self.model.training else "eval"
        total_loss_detached = torch.tensor(0.0, device=self.accelerator.device)

        # ==================================================================
        # STEP 1: Public Agent Update & Memory Release
        # ==================================================================
        
        if public_indices:
            public_inputs = self._extract_agent_inputs(inputs, public_indices)
            if self.use_liger_loss:
                unwrapped_model = self.accelerator.unwrap_model(model)
                public_loss = self._forward_redirection(model, unwrapped_model, self.compute_liger_loss, unwrapped_model, public_inputs)
            else:
                public_loss = self._compute_loss(model, public_inputs)
            
            # 로깅용 값만 따로 저장 (Graph와 끊김)
            public_loss_val = public_loss.detach()
            self._metrics[mode]["loss/public"].append(public_loss_val.item())
            total_loss_detached += public_loss_val
        else:
            # DDP 동기화를 위한 Dummy Loss
            public_loss = sum(p.sum() for p in model.parameters() if p.requires_grad) * 0.0

        # [핵심] Backward 수행 -> 계산 그래프 메모리 해제됨 (retain_graph=False 기본값)
        self.accelerator.backward(public_loss)
        
        # [핵심] 텐서 변수 삭제 (Reference Count 감소 -> 메모리 반환 유도)
        del public_loss
        if 'public_inputs' in locals(): del public_inputs
        
        # 선택사항: VRAM 파편화가 심하면 강제 정리 (속도는 약간 느려질 수 있음)
        # torch.cuda.empty_cache() 

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
        else:
            private_loss = sum(p.sum() for p in model.parameters() if p.requires_grad) * 0.0

        # Backward 수행 -> 계산 그래프 메모리 해제
        self.accelerator.backward(private_loss)
        
        # 메모리 정리
        del private_loss
        if 'private_inputs' in locals(): del private_inputs
        
        # ==================================================================
        # 결과 반환
        # ==================================================================
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
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if is_rich_available():
                print_prompt_completions_sample(
                    self._logs["prompt"],
                    self._logs["completion"],
                    self._logs["rewards"],
                    self._logs["advantages"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._logs["prompt"]),
                    "prompt": self._logs["prompt"],
                    "completion": self._logs["completion"],
                    **self._logs["rewards"],
                    "advantage": self._logs["advantages"],
                }

                if self._logs["images"]:
                    table["images"] = []
                    for image_list in self._logs["images"]:
                        # Convert images to wandb Image objects for proper visualization
                        table["images"].append([wandb.Image(image) for image in image_list])

                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                wandb.log({"completions": wandb.Table(dataframe=df)})

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
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
        match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return content
    
    def _generate_multi_turn(self, prompts: list, images: Optional[list], problems: list[str], solutions: list[str]):
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