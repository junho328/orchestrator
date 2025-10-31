SERVER="0.0.0.0"
PORT="4321"
export HOSTED_VLLM_API_BASE="http://$SERVER:$PORT/v1"

MODEL="gemma-3-27b-it"

MAX_TOKENS=1024 # TBD

# Launch the MT-bench generation on model.
python third_party/FastChat/fastchat/llm_judge/gen_api_answer.py \
  --model "$MODEL" \
  --openai-api-base "$HOSTED_VLLM_API_BASE" \
  --answer-file "results/mt_bench/${MODEL}.jsonl" \
  --max-tokens "$MAX_TOKENS" \
  --parallel 20

# Ensure the answers exist.
test -s "results/mt_bench/${MODEL}.jsonl"
    --model $MODEL \
    --openai-api-base $HOSTED_VLLM_API_BASE \
    --answer-file results/mt_bench/${MODEL}.jsonl \
    --max-tokens $MAX_TOKENS \
    --parallel 10

# Launch the MT-bench judgment on the result (use OpenAI, not local vLLM).
unset HOSTED_VLLM_API_BASE
python third_party/FastChat/fastchat/llm_judge/gen_judgment.py \
  --mode single \
  --model-list "$MODEL" \
  --parallel 5 \
  --answer-file "results/mt_bench/${MODEL}.jsonl" \
  --judge-model gpt-4.1 \
  --output-file "results/mt_bench/model_judgment/${MODEL}_gpt-4.1_single.jsonl"
    --mode single \
    --model-list $MODEL \
    --parallel 10 \
    --answer-file results/mt_bench/${MODEL}.jsonl \
    --judge-model gpt-4.1 \
    --output-file results/mt_bench/model_judgment/${MODEL}_gpt-4.1_single.jsonl

# Print the result.
python third_party/FastChat/fastchat/llm_judge/show_result.py \
  --mode single \
  --input-file "results/mt_bench/model_judgment/${MODEL}_gpt-4.1_single.jsonl"
