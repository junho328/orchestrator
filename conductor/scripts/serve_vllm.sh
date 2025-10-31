CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen2.5-32B-Instruct" --max-model-len=8192 \
    --port 8324 &

CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" --max-model-len=8192 \
    --port 8325 &

CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
    --model "google/gemma-3-27b-it" --max-model-len=8192 \
    --port 8326 &

CUDA_VISIBLE_DEVICES=5 python -m vllm.entrypoints.openai.api_server \
    --model "meta-llama/Llama-3.1-8B-Instruct" --max-model-len=8192 \
    --port 8327 &



