#!/bin/bash
# Single BigCodeBench training script using GRPO

# Set tokenizers parallelism to avoid warnings
export TOKENIZERS_PARALLELISM=false
# export CUDA_VISIBLE_DEVICES=0,2

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCHESTRATOR_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Default values
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-Coder-3B-Instruct"}
DATASET_NAME=${DATASET_NAME:-"jhn9803/hendrycks-math-with-answers"}
OUTPUT_DIR=${OUTPUT_DIR:-"/ext_hdd/jhna/marllm"}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
DTYPE=${DTYPE:-"bfloat16"}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-3}
EVAL_STRATEGY=${EVAL_STRATEGY:-"no"}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-1024}
BATCH_SIZE=${BATCH_SIZE:-16}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
NUM_GENERATIONS=${NUM_GENERATIONS:-16}
BETA=${BETA:-0.0}
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"all-linear"}
SEED=${SEED:-42}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}

# Quantization: optional, set --load_in_4bit or --load_in_8bit if needed
# LOAD_IN_4BIT=${LOAD_IN_4BIT:-""}
# LOAD_IN_8BIT=${LOAD_IN_8BIT:-""}

WANDB_PROJECT=${WANDB_PROJECT:-"single-math-train"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"qwen-1.5b-math-grpo-g16"}

# Export environment variables so they're available to the Python script
export WANDB_PROJECT
export WANDB_RUN_NAME

ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-"/home/jhna/orchestrator/davids/configs/deepspeed_zero.yaml"}

# Change to orchestrator directory for proper module imports
cd "$ORCHESTRATOR_DIR"

# Run training with accelerate
accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    -m davids.train.single_train.single_math_train \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate "$LEARNING_RATE" \
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
    # --load_in_4bit \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --lora_target_modules "$LORA_TARGET_MODULES" \
    --project "$WANDB_PROJECT" \
    --run_name "$WANDB_RUN_NAME" \
    --report_to wandb \
    --log_completions \
    --seed "$SEED" \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    "$@"

