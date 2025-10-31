CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --port 6543 \
    --max-model-len 8192