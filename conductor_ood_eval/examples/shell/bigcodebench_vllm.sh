MODEL="Qwen/Qwen3-32B"
export HOSTED_VLLM_API_BASE="http://slurm0us-a3nodeset-3:8324/v1" 
export FASTCHAT_WORKER_API_TIMEOUT=10800
export OPENAI_API_KEY=EMPTY

DATASET=bigcodebench
BACKEND=vllm-openai
SPLIT=complete
SUBSET=hard # [hard, full] TBD, which subset to evaluate

time bigcodebench.evaluate \
  --model $MODEL \
  --split $SPLIT \
  --subset $SUBSET \
  --execution "gradio" \
  --backend $BACKEND \
  --root results/bigcodebench \
  --base_url $HOSTED_VLLM_API_BASE \
  --parallel 10 \
  --bs 50 \
  --max_new_tokens 16000 \
  --top_p 0.8 \
  --top_k 20 \
  --presence_penalty 1.0 \
  --chat_template_kwargs '{"enable_thinking": true}'

#--gradio_endpoint https://lfsm-bigcodebench-evaluator-v2.hf.space/ \