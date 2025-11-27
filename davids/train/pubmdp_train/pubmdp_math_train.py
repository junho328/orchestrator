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

from davids.train.pubmdp_train.pubmdp_grpo_trainer import PUBMDPGRPOTrainer
from davids.train.utils.pubmdp_prompt import get_orchestrator_prompt
from davids.reward_utils.think_answer_format_reward import think_answer_format_reward
from davids.reward_utils.math_reward import accuracy_reward

@dataclass
class MultiGRPOConfig(GRPOConfig):
    """Custom script arguments extending TRL's ScriptArguments."""

    num_agents: int = field(
        default=3,
        metadata={"help": "Number of agents"},
    )

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, MultiGRPOConfig, ModelConfig))
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
    
    peft_config = get_peft_config(model_args)
    
    ################
    # Dataset
    ################
    
    train_dataset = load_dataset(script_args.dataset_name, split="train")
    train_dataset = train_dataset.filter(lambda x: x["level"] in ("Level 3", "Level 4", "Level 5"))
    train_dataset = train_dataset.shuffle(seed=42)
    eval_dataset = load_dataset(script_args.dataset_name, split="test")
    eval_dataset = eval_dataset.filter(lambda x: x["level"] in ("Level 3", "Level 4", "Level 5"))
    eval_dataset = eval_dataset.shuffle(seed=42)
    
    ORCHESTRATOR_SYSTEM_PROMPT = "You are a helpful assistant that decompose the problem into multiple sub-problems and solve them one by one."
    
    def make_conversation(example):
        """Create conversation format for orchestrator."""
        user_prompt = get_orchestrator_prompt(
            original_problem=example["problem"], 
            previous_outputs=[],
            num_agents=training_args.num_agents, 
        )
        return {
            "prompt": [
                {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "solution": example["answer"],
            "original_question": example["problem"],  # Store original question for agent prompts
        }

    train_dataset = train_dataset.map(make_conversation)
    eval_dataset = eval_dataset.map(make_conversation)

    ################
    # Training
    ################
    trainer = PUBMDPGRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[think_answer_format_reward, accuracy_reward],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name.split("/")[-1], model_name=training_args.run_name)