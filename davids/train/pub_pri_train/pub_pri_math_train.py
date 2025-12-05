import os
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import torch
from datasets import load_dataset

from trl import (
    GRPOConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from davids.train.pub_pri_train.pub_pri_grpo_trainer import PUBPRIGRPOTrainer
from davids.reward_utils.think_answer_format_reward import think_answer_format_reward
from davids.reward_utils.math_reward import accuracy_reward

@dataclass
class PUBPRIGRPOConfig(GRPOConfig):
    """Custom script arguments extending TRL's ScriptArguments."""

    num_agents: int = field(
        default=2,
        metadata={"help": "Number of agents"},
    )
    
    private_agent_max_completion_length: int=field(
        default=1024,
        metadata={"help": "Maximum completion length for the private agent"},
    )
    public_agent_max_completion_length: int=field(
        default=512,
        metadata={"help": "Maximum completion length for the public agent"},
    )

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, PUBPRIGRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    # Get environment variables (set in run_single_math_train.sh)
    wandb_entity_env = os.environ.get('WANDB_ENTITY')
    wandb_run_name_env = os.environ.get('WANDB_RUN_NAME')
    wandb_project_env = os.environ.get('WANDB_PROJECT')
    
    # Always set run_name and project from environment variables if available
    # This ensures they are set even if --run_name argument parsing fails
    if wandb_entity_env and hasattr(training_args, 'entity'):
        training_args.entity = wandb_entity_env
    
    if wandb_run_name_env and hasattr(training_args, 'run_name'):
        training_args.run_name = wandb_run_name_env
    
    if wandb_project_env and hasattr(training_args, 'project'):
        training_args.project = wandb_project_env
    
    ################
    # Model & Processor
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        # training_args.model_init_kwargs["device_map"] = get_kbit_device_map()
        training_args.model_init_kwargs["quantization_config"] = quantization_config
    
    # Create two PEFT configs: public and private adapters
    base_peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
    )
    
    ################
    # Dataset
    ################
    import logging
    import sys
    
    # Setup logging to ensure it works in distributed environment
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    logger = logging.getLogger(__name__)
    
    # Use print as well for immediate output
    print("=" * 80, file=sys.stderr, flush=True)
    print("STEP 1: Loading datasets...", file=sys.stderr, flush=True)
    logger.info("Loading train dataset...")
    
    train_dataset = load_dataset(script_args.dataset_name, split="train")
    print(f"Train dataset loaded: {len(train_dataset)} samples", file=sys.stderr, flush=True)
    logger.info(f"Train dataset loaded: {len(train_dataset)} samples")
    
    print("Filtering train dataset...", file=sys.stderr, flush=True)
    train_dataset = train_dataset.filter(lambda x: x["level"] in ("Level 3", "Level 4", "Level 5"))
    print(f"Train dataset after filtering: {len(train_dataset)} samples", file=sys.stderr, flush=True)
    
    train_dataset = train_dataset.shuffle(seed=42)
    print("Loading eval dataset...", file=sys.stderr, flush=True)
    
    eval_dataset = load_dataset(script_args.dataset_name, split="test")
    print(f"Eval dataset loaded: {len(eval_dataset)} samples", file=sys.stderr, flush=True)
    eval_dataset = eval_dataset.filter(lambda x: x["level"] in ("Level 3", "Level 4", "Level 5"))
    print(f"Eval dataset after filtering: {len(eval_dataset)} samples", file=sys.stderr, flush=True)
    eval_dataset = eval_dataset.shuffle(seed=42)
    
    def filter_columns(example):
        return {"problem": example["problem"], "answer": example["answer"]}

    print("Mapping datasets to filter columns...", file=sys.stderr, flush=True)
    train_dataset = train_dataset.map(filter_columns)
    eval_dataset = eval_dataset.map(filter_columns)
    print("Dataset preparation completed", file=sys.stderr, flush=True)

    ################
    # Training
    ################
    
    print("=" * 80, file=sys.stderr, flush=True)
    print("STEP 2: Loading base model...", file=sys.stderr, flush=True)
    logger.info(f"Loading model from {model_args.model_name_or_path}...")
    
    base_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, 
                                                dtype=dtype)
    print("Base model loaded successfully", file=sys.stderr, flush=True)
    logger.info("Base model loaded successfully")
    
    print("STEP 3: Setting up PEFT adapters...", file=sys.stderr, flush=True)
    model = get_peft_model(base_model, base_peft_config, adapter_name="public")
    print("Public adapter added", file=sys.stderr, flush=True)
    model.add_adapter("private", base_peft_config)
    print("Private adapter added", file=sys.stderr, flush=True)
    logger.info("PEFT adapters configured")
    
    print("=" * 80, file=sys.stderr, flush=True)
    print("STEP 4: Creating PUBPRIGRPOTrainer...", file=sys.stderr, flush=True)
    logger.info("Creating PUBPRIGRPOTrainer...")
    trainer = PUBPRIGRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=[think_answer_format_reward, accuracy_reward],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=None,  # Already a PeftModel, so pass None to avoid merge_and_unload
    )
    print("PUBPRIGRPOTrainer created successfully", file=sys.stderr, flush=True)
    logger.info("PUBPRIGRPOTrainer created successfully")

    print("=" * 80, file=sys.stderr, flush=True)
    print("STEP 5: Starting training...", file=sys.stderr, flush=True)
    logger.info("Starting training...")
    trainer.train()
    print("Training completed", file=sys.stderr, flush=True)
    logger.info("Training completed")

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name.split("/")[-1], model_name=training_args.run_name)