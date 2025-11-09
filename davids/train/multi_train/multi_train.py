#!/usr/bin/env python3
"""
Training script for orchestrator-agent model using GRPO, QLoRA, DeepSpeed ZeRO Stage 3, and Accelerator.

The model acts as both orchestrator (splitting tasks into subtasks with role instructions)
and as each agent (generating reasoning and answers).
"""

import os
import sys
import logging
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer
from accelerate import Accelerator
from accelerate.utils import set_seed
import deepspeed
from datasets import Dataset, train_test_split

# Import custom modules
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from davids.davids_bcb_reward import OrchestratorAgentReward
from davids.davids_grpotrainer import OrchestratorAgentGRPOTrainer
from davids.data_utils.orchestrator_process import OrchestratorDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_model_and_tokenizer(
    model_name_or_path: str,
    use_peft: bool = True,
    load_in_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    lora_target_modules: str = "all-linear",
    trust_remote_code: bool = True,
):
    """Setup model and tokenizer with QLoRA configuration."""
    logger.info(f"Loading model and tokenizer from {model_name_or_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Configure quantization if using 4-bit
    quantization_config = None
    if use_peft and load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16,
    )
    
    # Prepare model for k-bit training if using quantization
    if use_peft and load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA if using PEFT
    if use_peft:
        # Determine target modules
        # if lora_target_modules == "all-linear":
        #     # Common target modules for different model architectures
        #     if "qwen" in model_name_or_path.lower():
        #         target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        #     elif "llama" in model_name_or_path.lower():
        #         target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        #     elif "phi" in model_name_or_path.lower():
        #         target_modules = ["q_proj", "k_proj", "v_proj", "dense"]
        #     else:
        #         # Fallback: try to find linear layers
        #         target_modules = []
        #         for name, module in model.named_modules():
        #             if isinstance(module, torch.nn.Linear):
        #                 target_modules.append(name.split('.')[-1])
        #         target_modules = list(set(target_modules))
        # else:
        #     target_modules = lora_target_modules.split(',')
        
        logger.info(f"Using LoRA target modules: {lora_target_modules}")
        
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def create_dataset(
    tokenizer,
    dataset_name: str,
    split: str,
    max_routing_steps: int = 4,
    seed: int = 42,
    max_samples: Optional[int] = None,
    num_few_shot_examples: int = 2,
):
    """Create dataset for training."""
    logger.info(f"Creating dataset: {dataset_name}, split: {split}")
    
    dataset = OrchestratorDataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        split=split,
        max_routing_steps=max_routing_steps,
        seed=seed,
        apply_chat_template=True,
        max_prompt_length=1024,
        num_few_shot_examples=num_few_shot_examples,
    )
    
    # # Convert to HuggingFace Dataset format
    # data_dict = {
    #     "input_ids": [],
    #     "attention_mask": [],
    #     "prompt": [],
    #     "task_data": [],
    # }
    
    # dataset_size = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    # for i in range(dataset_size):
    #     item = dataset[i]
    #     data_dict["input_ids"].append(item["input_ids"])
    #     data_dict["attention_mask"].append(item["attention_mask"])
    #     data_dict["prompt"].append(item["prompt"])
    #     data_dict["task_data"].append(item["task_data"])
    
    # hf_dataset = Dataset.from_dict(data_dict)
    
    logger.info(f"Created dataset with {len(dataset)} examples")
    
    return dataset


def main(
    # Model configuration
    model_name_or_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    use_peft: bool = True,
    load_in_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    lora_target_modules: str = "all-linear",
    
    # Dataset configuration
    dataset_name: str = "bigcodebench",
    dataset_split: str = "v0.1.4",
    max_samples: Optional[int] = None,
    eval_ratio: Optional[str] = 0.1,
    max_routing_steps: int = 4,
    num_few_shot_examples: int = 2,
    
    # Training configuration
    output_dir: str = "/ext_hdd/jhna/davids",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-6,
    lr_scheduler_type: str = "cosine",
    warmup_steps: int = 100,
    max_grad_norm: float = 1.0,
    logging_steps: int = 1,
    save_strategy: str = "epoch",
    save_steps: int = 1,
    eval_steps: int = 1,
    save_total_limit: int = 3,
    
    # GRPO configuration
    num_generations: int = 16,
    max_completion_length: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.95,
    
    # Reward configuration
    format_reward_orchestrator: float = 0.5,
    format_reward_agent: float = 0.5,
    max_tokens: int = 1024,
    
    # DeepSpeed configuration
    deepspeed_config_path: Optional[str] = None,
    
    # Other
    seed: int = 42,
    trust_remote_code: bool = True,
    
    use_vllm: bool = True,
    gpu_memory_utilization: float = 0.8,
    tensor_parallel_size: int = 2,
):
    """Main training function."""
    
    # Set seed
    set_seed(seed)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        use_peft=use_peft,
        load_in_4bit=load_in_4bit,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        trust_remote_code=trust_remote_code,
    )
    
    # Create datasets
    train_dataset = create_dataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        split=dataset_split,
        max_routing_steps=max_routing_steps,
        seed=seed,
        max_samples=max_samples,
        num_few_shot_examples=num_few_shot_examples,
    )
    
    if eval_ratio != 0:
        
        train_dataset, eval_dataset = train_test_split(train_dataset, test_size=eval_ratio, random_state=seed)
    
    # Setup reward function
    reward_func = OrchestratorAgentReward(
        model=model,
        tokenizer=tokenizer,
        model_name_or_path=model_name_or_path,  # Pass model path for vLLM initialization
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        max_routing_steps=max_routing_steps,
        format_reward_orchestrator=format_reward_orchestrator,
        format_reward_agent=format_reward_agent,   
    )
    
    # GRPO configuration
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        max_grad_norm=max_grad_norm,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        temperature=temperature,
        top_p=top_p,
        bf16=True,
        report_to="wandb",
        project_name="davids",
        run_name="Qwen-7B-GRPO-G16",
    )
    
    # DeepSpeed configuration
    if deepspeed_config_path:
        logger.info(f"Using DeepSpeed config from {deepspeed_config_path}")
        grpo_config.deepspeed = deepspeed_config_path
    
    # Create trainer
    trainer = OrchestratorAgentGRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_ratio != 0 else None,
        reward_funcs=[reward_func],
        tokenizer=tokenizer,
        use_vllm=use_vllm,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving final model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train orchestrator-agent model with GRPO")
    
    # Model args
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--use_peft", action="store_true", default=True)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", type=str, default="all-linear")
    
    # Dataset args
    parser.add_argument("--dataset_name", type=str, default="bigcodebench")
    parser.add_argument("--dataset_split", type=str, default="v0.1.4")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--max_routing_steps", type=int, default=4)
    parser.add_argument("--num_few_shot_examples", type=int, default=2)
    
    # Training args
    parser.add_argument("--output_dir", type=str, default="/ext_hdd/jhna/davids")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--save_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=1)
    parser.add_argument("--save_total_limit", type=int, default=3)
    
    # GRPO args    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    
    # Reward args
    parser.add_argument("--format_reward_orchestrator", type=float, default=0.5)
    parser.add_argument("--format_reward_agent", type=float, default=0.5)
    parser.add_argument("--max_tokens", type=int, default=1024)
    
    # DeepSpeed args
    parser.add_argument("--deepspeed_config_path", type=str, default="./configs/deepspeed_zero3.yaml")
    
    # Other args
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    
    # vLLM args
    parser.add_argument("--use_vllm", action="store_true", default=True)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    
    args = parser.parse_args()
    
    main(**vars(args))

