#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="/home/work/aipr-jhna/huggingface_hub/Qwen2.5-Coder-3B-Instruct"

OUTPUT_DIR="/home/work/aipr-jhna/output/eval"

DTYPE="bfloat16"  
MAX_LEN=32768
MAX_NEW_TOKENS=3072
TEMP=0.6
TOP_P=0.95
GPU_UTIL=0.8

MODEL_ARGS="model_name=$MODEL_PATH,dtype=$DTYPE,max_model_length=$MAX_LEN,gpu_memory_utilization=$GPU_UTIL,generation_parameters={max_new_tokens:$MAX_NEW_TOKENS,temperature:$TEMP,top_p:$TOP_P}"

TASKS=(
  "gsm8k"
  "hendrycks_math"
  "math_500"
  "minervamath"
  "aime24"
  "aime25"
  ""

)


for TASK in "${TASKS[@]}"; do
  echo "Running task: $TASK"

  if [ "$TASK" = "aime24" ]; then
    PREFIX="community"
    CUSTOM_TASKS="--custom-tasks /home/work/aipr-jhna/orchestrator/aime_evals.py"
  elif [ "$TASK" = "aime25" ]; then
    PREFIX="community"
    CUSTOM_TASKS="--custom-tasks /home/work/aipr-jhna/orchestrator/aime_evals.py"
  elif [ "$TASK" = "minervamath" ]; then
    PREFIX="community"
    CUSTOM_TASKS="--custom-tasks /home/work/aipr-jhna/orchestrator/minervamath_evals.py"
  elif [ "$TASK" = "hendrycks_math" ]; then
    PREFIX="lighteval"
    CUSTOM_TASKS="--custom-tasks /home/work/aipr-jhna/orchestrator/hendrycks_math.py"
  else
    PREFIX="lighteval"
    CUSTOM_TASKS=""
  fi

  lighteval vllm $MODEL_ARGS "${PREFIX}|${TASK}|0|0" \
    --use-chat-template \
    --output-dir "$OUTPUT_DIR" \
    $CUSTOM_TASKS
done
