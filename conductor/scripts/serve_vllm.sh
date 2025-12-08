
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen2.5-3B-Instruct" --max-model-len=8192 \
    --tensor-parallel-size=1 \
    --gpu-memory-utilization=0.2 \
    --port 8324 &

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" --max-model-len=8192 \
    --tensor-parallel-size=1 \
    --gpu-memory-utilization=0.3 \
    --max-num-seqs=8 \
    --port 8325 &

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "google/gemma-2-2b-it" --max-model-len=8192 \
    --tensor-parallel-size=1 \
    --gpu-memory-utilization=0.3 \
    --max-num-seqs=16 \
    --port 8326 &

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "meta-llama/Llama-3.2-3B-Instruct" \
    --max-model-len=8192 \
    --tensor-parallel-size=1 \
    --gpu-memory-utilization=0.3 \
    --max-num-seqs=16 \
    --port 8327 &

# CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
#     --model "Qwen/Qwen2.5-32B-Instruct" --max-model-len=8192 \
#     --port 8324 &

# CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
#     --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" --max-model-len=8192 \
#     --port 8325 &

# CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
#     --model "google/gemma-3-27b-it" --max-model-len=8192 \
#     --port 8326 &

# CUDA_VISIBLE_DEVICES=5 python -m vllm.entrypoints.openai.api_server \
#     --model "meta-llama/Llama-3.1-8B-Instruct" --max-model-len=8192 \
#     --port 8327 &


# test
# curl -X POST http://localhost:8324/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "Qwen/Qwen2.5-3B-Instruct",
#     "messages": [
#       {"role": "user", "content": "What is the capital of France?"}
#     ],
#     "temperature": 0.7,
#     "max_tokens": 100
#   }'