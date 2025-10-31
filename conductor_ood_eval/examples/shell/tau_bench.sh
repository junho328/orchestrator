SERVER="0.0.0.0"
PORT="8081"

export HOSTED_VLLM_API_BASE="http://$SERVER:$PORT/v1/"

time python third_party/tau-bench/run.py \
    --env retail \
    --agent-strategy "act" \
    --model "$MODEL" \
    --model-provider hosted_vllm \
    --user-model gpt-4o \
    --user-model-provider openai \
    --user-strategy llm \
    --num-trials 1 \
    --temperature 0.6 \
    --start-index 1 \
    --end-index 10 \
    --max-concurrency 10