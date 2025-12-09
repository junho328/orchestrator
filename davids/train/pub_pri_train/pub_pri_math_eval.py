import os
import sys
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from datasets import load_dataset
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import (
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_quantization_config,
)

from davids.train.pub_pri_train.pub_pri_grpo_trainer import PUBPRIGRPOTrainer
from davids.train.pub_pri_train.grpo_config import GRPOConfig
from davids.reward_utils.think_answer_format_reward import think_answer_format_reward
from davids.reward_utils.math_reward import accuracy_reward


@dataclass
class EvalScriptArguments(ScriptArguments):
    """Script arguments for evaluation."""
    
    # Adapter paths for loading pre-trained adapters
    public_adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the pre-trained public adapter. If None, will use untrained adapter."},
    )
    private_adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the pre-trained private adapter. If None, will use untrained adapter."},
    )
    checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a checkpoint directory containing both adapters. "
                  "If provided, will look for 'public' and 'private' subdirectories."},
    )
    
    # Evaluation parameters
    num_samples_per_problem: int = field(
        default=8,
        metadata={"help": "Number of samples to generate per problem for pass@k calculation"},
    )
    k_values: str = field(
        default="1,4,8",
        metadata={"help": "Comma-separated list of k values for pass@k metrics (e.g., '1,4,8')"},
    )
    eval_batch_size: int = field(
        default=4,
        metadata={"help": "Number of problems to process in each batch"},
    )
    output_path: Optional[str] = field(
        default="eval_results.json",
        metadata={"help": "Path to save evaluation results"},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of evaluation samples to use. If None, use all."},
    )


@dataclass
class PUBPRIGRPOConfig(GRPOConfig):
    """Custom GRPO config for public-private agents."""

    num_agents: int = field(
        default=2,
        metadata={"help": "Number of agents (public-private pairs)"},
    )
    private_agent_max_completion_length: int = field(
        default=1024,
        metadata={"help": "Maximum completion length for the private agent"},
    )
    public_agent_max_completion_length: int = field(
        default=512,
        metadata={"help": "Maximum completion length for the public agent"},
    )


def load_adapters_from_checkpoint(
    base_model: AutoModelForCausalLM,
    checkpoint_path: str,
    public_adapter_path: Optional[str] = None,
    private_adapter_path: Optional[str] = None,
    logger: logging.Logger = None,
) -> PeftModel:
    """
    Load public and private adapters from checkpoint or individual paths.
    
    Args:
        base_model: The base model to add adapters to
        checkpoint_path: Path to checkpoint directory (contains adapter_model.safetensors or subdirs)
        public_adapter_path: Optional explicit path to public adapter
        private_adapter_path: Optional explicit path to private adapter
        logger: Logger instance
        
    Returns:
        PeftModel with both adapters loaded
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Determine adapter paths
    if checkpoint_path is not None:
        # Check if checkpoint_path directly contains adapter files
        if os.path.exists(os.path.join(checkpoint_path, "adapter_model.safetensors")) or \
           os.path.exists(os.path.join(checkpoint_path, "adapter_model.bin")):
            # Single adapter checkpoint - this might be a merged checkpoint
            # Try to load it and see if it has multiple adapters
            logger.info(f"Loading adapters from single checkpoint: {checkpoint_path}")
            model = PeftModel.from_pretrained(base_model, checkpoint_path, adapter_name="public")
            
            # Check if there are other adapter directories
            private_path = os.path.join(checkpoint_path, "private")
            if os.path.exists(private_path):
                logger.info(f"Loading private adapter from: {private_path}")
                model.load_adapter(private_path, adapter_name="private")
            else:
                # Create a copy of public adapter as private (for testing)
                logger.warning("No separate private adapter found. Using same weights for both adapters.")
                model.load_adapter(checkpoint_path, adapter_name="private")
        else:
            # Check for subdirectories
            public_path = public_adapter_path or os.path.join(checkpoint_path, "public")
            private_path = private_adapter_path or os.path.join(checkpoint_path, "private")
            
            if os.path.exists(public_path):
                logger.info(f"Loading public adapter from: {public_path}")
                model = PeftModel.from_pretrained(base_model, public_path, adapter_name="public")
            else:
                raise ValueError(f"Public adapter not found at: {public_path}")
            
            if os.path.exists(private_path):
                logger.info(f"Loading private adapter from: {private_path}")
                model.load_adapter(private_path, adapter_name="private")
            else:
                raise ValueError(f"Private adapter not found at: {private_path}")
    else:
        # Use explicit paths
        if public_adapter_path is None or private_adapter_path is None:
            raise ValueError("Either checkpoint_path or both public_adapter_path and private_adapter_path must be provided")
        
        logger.info(f"Loading public adapter from: {public_adapter_path}")
        model = PeftModel.from_pretrained(base_model, public_adapter_path, adapter_name="public")
        
        logger.info(f"Loading private adapter from: {private_adapter_path}")
        model.load_adapter(private_adapter_path, adapter_name="private")
    
    return model


def create_untrained_adapters(
    base_model: AutoModelForCausalLM,
    logger: logging.Logger = None,
) -> PeftModel:
    """
    Create new untrained adapters for baseline evaluation.
    
    Args:
        base_model: The base model to add adapters to
        logger: Logger instance
        
    Returns:
        PeftModel with both adapters initialized
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    base_peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
    )
    
    logger.info("Creating untrained public adapter...")
    model = get_peft_model(base_model, base_peft_config, adapter_name="public")
    
    logger.info("Creating untrained private adapter...")
    model.add_adapter("private", base_peft_config)
    
    return model


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    logger = logging.getLogger(__name__)
    
    # Parse arguments
    parser = TrlParser((EvalScriptArguments, PUBPRIGRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    # Parse k_values from comma-separated string
    k_values = [int(k.strip()) for k in script_args.k_values.split(",")]
    
    print("=" * 80, file=sys.stderr, flush=True)
    print("Multi-Agent Evaluation Script", file=sys.stderr, flush=True)
    print("=" * 80, file=sys.stderr, flush=True)
    
    ################
    # Model Configuration
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        training_args.model_init_kwargs["quantization_config"] = quantization_config
    
    ################
    # Dataset
    ################
    print("STEP 1: Loading evaluation dataset...", file=sys.stderr, flush=True)
    logger.info(f"Loading dataset: {script_args.dataset_name}")
    
    eval_dataset = load_dataset(script_args.dataset_name, split="test")
    print(f"Eval dataset loaded: {len(eval_dataset)} samples", file=sys.stderr, flush=True)
    
    # Shuffle with seed for reproducibility
    eval_dataset = eval_dataset.shuffle(seed=training_args.seed)
    
    # Limit samples if specified
    if script_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(script_args.max_eval_samples, len(eval_dataset))))
        print(f"Limited to {len(eval_dataset)} samples", file=sys.stderr, flush=True)
    
    # Map columns to expected format
    # The evaluation function expects 'problem' and 'solution' keys
    def prepare_eval_example(example):
        return {
            "problem": example.get("problem", example.get("question", "")),
            "solution": example.get("answer", example.get("solution", "")),
        }
    
    eval_dataset = eval_dataset.map(prepare_eval_example)
    print(f"Dataset preparation completed", file=sys.stderr, flush=True)
    
    ################
    # Model Loading
    ################
    print("=" * 80, file=sys.stderr, flush=True)
    print("STEP 2: Loading base model...", file=sys.stderr, flush=True)
    logger.info(f"Loading model from {model_args.model_name_or_path}...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    print("Base model loaded successfully", file=sys.stderr, flush=True)
    
    ################
    # Adapter Loading
    ################
    print("=" * 80, file=sys.stderr, flush=True)
    print("STEP 3: Loading adapters...", file=sys.stderr, flush=True)
    
    # Determine if we should load pre-trained adapters or create new ones
    has_checkpoint = script_args.checkpoint_path is not None
    has_adapter_paths = script_args.public_adapter_path is not None or script_args.private_adapter_path is not None
    
    if has_checkpoint or has_adapter_paths:
        # Load pre-trained adapters
        model = load_adapters_from_checkpoint(
            base_model=base_model,
            checkpoint_path=script_args.checkpoint_path,
            public_adapter_path=script_args.public_adapter_path,
            private_adapter_path=script_args.private_adapter_path,
            logger=logger,
        )
        print("Pre-trained adapters loaded successfully", file=sys.stderr, flush=True)
    else:
        # Create untrained adapters (baseline evaluation)
        logger.warning("No adapter paths provided. Creating untrained adapters for baseline evaluation.")
        model = create_untrained_adapters(base_model, logger)
        print("Untrained adapters created (baseline mode)", file=sys.stderr, flush=True)
    
    # List available adapters
    if hasattr(model, 'peft_config'):
        print(f"Available adapters: {list(model.peft_config.keys())}", file=sys.stderr, flush=True)
    
    ################
    # Create Trainer for Evaluation
    ################
    print("=" * 80, file=sys.stderr, flush=True)
    print("STEP 4: Creating PUBPRIGRPOTrainer...", file=sys.stderr, flush=True)
    
    # For evaluation only, we don't need a train dataset
    # Create a minimal trainer configuration
    training_args.do_train = False
    training_args.do_eval = True
    
    trainer = PUBPRIGRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=[think_answer_format_reward, accuracy_reward],
        train_dataset=eval_dataset,  # Required but not used for eval
        eval_dataset=eval_dataset,
        peft_config=None,  # Already a PeftModel
    )
    print("PUBPRIGRPOTrainer created successfully", file=sys.stderr, flush=True)
    
    ################
    # Run Evaluation
    ################
    print("=" * 80, file=sys.stderr, flush=True)
    print("STEP 5: Running multi-agent evaluation...", file=sys.stderr, flush=True)
    print(f"  - Samples per problem: {script_args.num_samples_per_problem}", file=sys.stderr, flush=True)
    print(f"  - k values: {k_values}", file=sys.stderr, flush=True)
    print(f"  - Batch size: {script_args.eval_batch_size}", file=sys.stderr, flush=True)
    print(f"  - Num agents: {training_args.num_agents}", file=sys.stderr, flush=True)
    
    results = trainer.run_multi_agent_evaluation(
        eval_dataset=eval_dataset,
        num_samples_per_problem=script_args.num_samples_per_problem,
        k_values=k_values,
        batch_size=script_args.eval_batch_size,
        output_path=script_args.output_path,
        verbose=True,
    )
    
    ################
    # Print Results
    ################
    print("=" * 80, file=sys.stderr, flush=True)
    print("EVALUATION RESULTS", file=sys.stderr, flush=True)
    print("=" * 80, file=sys.stderr, flush=True)
    
    for key, value in results.items():
        if key != "per_problem_results":
            print(f"  {key}: {value}", file=sys.stderr, flush=True)
    
    print("=" * 80, file=sys.stderr, flush=True)
    print(f"Results saved to: {script_args.output_path}", file=sys.stderr, flush=True)
    print("Evaluation completed!", file=sys.stderr, flush=True)
