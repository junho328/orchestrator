SERVER="0.0.0.0"
PORT="2222"
export HOSTED_VLLM_API_BASE="http://$SERVER:$PORT/v1"

export LITELLM_CONCURRENT_CALLS=20
# Create the config file needed by lighteval.
UNIQUE_ID=$(date +%s%N | cut -b1-13)
CONFIG_FILE="tmp_litellm_config_${UNIQUE_ID}.yaml"
cat > ${CONFIG_FILE} << EOF
model_parameters:
  model_name: "gpt-4o"
  provider: "openai"
  api_key: "EMPTY"
  base_url: "$HOSTED_VLLM_API_BASE" 
  generation_parameters:
    temperature: 0.2
    max_new_tokens: 2048
EOF

lighteval endpoint litellm \
  "${CONFIG_FILE}" \
  "lighteval|arc_agi_2|0|0" \
  --save-details \