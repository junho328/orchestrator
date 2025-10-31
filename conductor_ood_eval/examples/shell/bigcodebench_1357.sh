#!/bin/bash
SERVER="0.0.0.0"
PORT="1357"
BASE_MODEL="inception-v15-2rounds-nofak"  # base name without number
export HOSTED_VLLM_API_BASE="http://$SERVER:$PORT/v1"
export FASTCHAT_WORKER_API_TIMEOUT=10800

DATASET=bigcodebench
BACKEND=openai
SPLIT=complete
SUBSET=hard # [hard, full] TBD, which subset to evaluate

for i in {1..5}; do
  MODEL="${BASE_MODEL}${i}"
  # MODEL="maxconductor-gpt4kmed2"
  echo "Running evaluation with MODEL=$MODEL"
  
  time bigcodebench.evaluate \
    --model $MODEL \
    --split $SPLIT \
    --subset $SUBSET \
    --execution "gradio" \
    --base_url $HOSTED_VLLM_API_BASE \
    --backend $BACKEND \
    --root results/bigcodebench \
    --max_new_tokens 1024 \
    --parallel 10 \
    --bs 50
done



# SERVER="0.0.0.0"
# PORT="4321"
# # MODEL="inception-v1-7c22-finetune-v1noempty2-2rounds" # Used for result file name identifier.
# # MODEL="conductormax-lowverbosity3" 
# # MODEL="inception-v11-1round-lowverbosity"
# # MODEL="inception-v1-1round-lowverbosity3"
# MODEL="conductormax-medreason-lowverbosity"
# export HOSTED_VLLM_API_BASE="http://$SERVER:$PORT/v1"
# export FASTCHAT_WORKER_API_TIMEOUT=10800

# DATASET=bigcodebench
# BACKEND=openai
# SPLIT=complete
# SUBSET=hard # [hard, full] TBD, which subset to evaluate
# # MODEL='gpt-5'
# time bigcodebench.evaluate \
#   --model $MODEL \
#   --split $SPLIT \
#   --subset $SUBSET \
#   --execution "gradio" \
#   --base_url $HOSTED_VLLM_API_BASE \
#   --backend $BACKEND \
#   --root results/bigcodebench \
#   --max_new_tokens 1024 \
#   --parallel 10 \
#   --bs 50 \


# # --base_url $HOSTED_VLLM_API_BASE \
#   # --max_new_tokens 32000 \
#   # --reasoning_effort "high" \
#   # --verbosity "low"