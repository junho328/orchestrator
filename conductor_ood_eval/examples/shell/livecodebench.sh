export LITELLM_CONCURRENT_CALLS=8
export HOSTED_VLLM_API_BASE="http://slurm0us-gufnodeset-0:6543/v1"
export LITELLM_REQUEST_TIMEOUT=800

# Create the config file needed by lighteval
UNIQUE_ID=$(date +%s%N | cut -b1-13)
CONFIG_FILE="tmp_litellm_config_${UNIQUE_ID}.yaml"
cat > ${CONFIG_FILE} << EOF
model_parameters:
  model_name: "openai/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
  provider: "openai"
  api_key: "EMPTY"
  base_url: "$HOSTED_VLLM_API_BASE" 
  generation_parameters:
    temperature: 0.6
    max_new_tokens: 6500
EOF

# Launch the evalution of 10 samples in math_500.
lighteval endpoint litellm \
  "${CONFIG_FILE}" \
  "extended|lcb:codegeneration_v6|0|0" \
  --save-details


rm -f ${CONFIG_FILE}
