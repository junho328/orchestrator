# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "math-verify",
#     "latex2sympy2_extended",
#     "trackio",
#     "kernels",
# ]
# ///

"""
pip install math_verify

# For Qwen/Qwen3-0.6B
pip install num2words==0.5.14

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/gspo.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --output_dir gspo-Qwen3-0.6B \
    --learning_rate 1e-5 \
    --dtype bfloat16 \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --use_peft \
    --lora_target_modules "q_proj", "v_proj" \
    --log_completions \
    --per_device_train_batch_size 8 \
    --num_generations 8 \
    --importance_sampling_level sequence \
    --epsilon 3e-4 \
    --epsilon_high 4e-4 \
    --beta 0.0 \
    --loss_type grpo \
    --gradient_accumulation_steps 2 \
    --steps_per_generation 8

"""

import os
from dataclasses import dataclass, field
from typing import Optional

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
    GRPOTrainer,
)
from davids.reward_utils.think_answer_format_reward import think_answer_format_reward
from davids.reward_utils.math_reward import accuracy_reward

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    # Get environment variables (set in run_single_math_train.sh)
    wandb_run_name_env = os.environ.get('WANDB_RUN_NAME')
    wandb_project_env = os.environ.get('WANDB_PROJECT')
    
    # Always set run_name and project from environment variables if available
    # This ensures they are set even if --run_name argument parsing fails
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
    eval_dataset = load_dataset(script_args.dataset_name, split="test")
    
    # dataset_dict = train_dataset.train_test_split(test_size=script_args.eval_ratio, seed=training_args.seed)
    # train_dataset = dataset_dict["train"]
    # eval_dataset = dataset_dict["test"]
    
    SYSTEM_PROMPT = (
        "You are a helpful assistant that solve complex math problems."
    )

    def make_conversation(example):
        
        user_prompt = f"""You first think about the reasoning process in the mind and then provide the answer.
        
        The reasoning process is enclosed within <think> and </think> tags and the final answer is enclosed within <answer> and </answer> tags.
        
        Example format:
        <think>
        [Your reasoning and thought process here]
        </think>
        <answer>
        [Your final answer here]
        </answer>
        
        Now, solve the following problem:

        Problem: {example["problem"]}
"""
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "solution": example["answer"],  # Pass testcase as solution for bcb_accuracy_reward
        }

    train_dataset = train_dataset.map(make_conversation)
    eval_dataset = eval_dataset.map(make_conversation)

    ################
    # Training
    ################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[accuracy_reward, think_answer_format_reward],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)