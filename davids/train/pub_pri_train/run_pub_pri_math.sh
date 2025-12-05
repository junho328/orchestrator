#!/bin/bash
# Single BigCodeBench training script using GRPO

# Set tokenizers parallelism to avoid warnings
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="/ext_hdd/jhna/.triton"
# export CUDA_VISIBLE_DEVICES=1,2

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCHESTRATOR_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Default values
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-Coder-1.5B-Instruct"}
DATASET_NAME=${DATASET_NAME:-"jhn9803/hendrycks-math-with-answers"}
OUTPUT_DIR=${OUTPUT_DIR:-"/ext_hdd/jhna/marllm"}

#TRAINING HYPERPARAMETERS
SEED=${SEED:-42}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
TOP_P=${TOP_P:-0.95}
TEMPERATURE=${TEMPERATURE:-0.6}
DTYPE=${DTYPE:-"bfloat16"}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
EVAL_STRATEGY=${EVAL_STRATEGY:-"no"}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-128}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-128}
BATCH_SIZE=${BATCH_SIZE:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-False}
STEPS_PER_GENERATION=${STEPS_PER_GENERATION:-1}
NUM_GENERATIONS=${NUM_GENERATIONS:-2}
BETA=${BETA:-0.0}

#VLLM GPU UTILIZATION
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.2}

#USE LIGER LOSS
USE_LIGER_LOSS=${USE_LIGER_LOSS:-True}

# Multi-Agent Config
NUM_AGENTS=${NUM_AGENTS:-1}
PUBLIC_AGENT_MAX_COMPLETION_LENGTH=${PUBLIC_AGENT_MAX_COMPLETION_LENGTH:-128}
PRIVATE_AGENT_MAX_COMPLETION_LENGTH=${PRIVATE_AGENT_MAX_COMPLETION_LENGTH:-128}

# Quantization: optional, set --load_in_4bit or --load_in_8bit if needed
# LOAD_IN_4BIT=${LOAD_IN_4BIT:-""}
# LOAD_IN_8BIT=${LOAD_IN_8BIT:-""}

WANDB_ENTITY=${WANDB_ENTITY:-"lamas-aipr"}
WANDB_PROJECT=${WANDB_PROJECT:-"pub-pri-math-train"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"qwen-1.5b-math-grpo-g4"}

# Export environment variables so they're available to the Python script
export WANDB_ENTITY
export WANDB_PROJECT
export WANDB_RUN_NAME

ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-"/home/jhna/orchestrator/davids/configs/ddp_config.yaml"}

# Change to orchestrator directory for proper module imports
cd "$ORCHESTRATOR_DIR"

# Run training with accelerate
accelerate launch \
    --config_file "$ACCELERATE_CONFIG"\
    -m davids.train.pub_pri_train.pub_pri_math_train \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate "$LEARNING_RATE" \
    --top_p "$TOP_P" \
    --temperature "$TEMPERATURE" \
    --dtype "$DTYPE" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --eval_strategy "$EVAL_STRATEGY" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --num_generations "$NUM_GENERATIONS" \
    --beta "$BETA" \
    --loss_type dr_grpo \
    --use_peft \
    --project "$WANDB_PROJECT" \
    --run_name "$WANDB_RUN_NAME" \
    --report_to wandb \
    --log_completions \
    --seed "$SEED" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --num_agents "$NUM_AGENTS" \
    --public_agent_max_completion_length "$PUBLIC_AGENT_MAX_COMPLETION_LENGTH" \
    --private_agent_max_completion_length "$PRIVATE_AGENT_MAX_COMPLETION_LENGTH" \
    --use_liger_loss "$USE_LIGER_LOSS" \
    # --use_vllm \
    # --vllm_mode colocate \
    # --vllm_gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \

