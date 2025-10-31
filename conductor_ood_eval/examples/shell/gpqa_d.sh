SERVER="0.0.0.0"
PORT="3333"
export HOSTED_VLLM_API_BASE="http://$SERVER:$PORT/v1"
# export HOSTED_VLLM_API_BASE="http://slurm0us-a3nodeset-2:8322/v1"
export LITELLM_REQUEST_TIMEOUT=10800

export LITELLM_CONCURRENT_CALLS=10

for run in {1..3}; do
echo "Running evaluation number $run"
# Create the config file needed by lighteval.
UNIQUE_ID=$(date +%s%N | cut -b1-13)
CONFIG_FILE="tmp_litellm_config_${UNIQUE_ID}.yaml"
cat > ${CONFIG_FILE} << EOF
model_parameters:
  model_name: "gpt-4o"
  provider: "openai"
  api_key: "none"
  base_url: "$HOSTED_VLLM_API_BASE" 
  generation_parameters:
    temperature: 0.2
    max_new_tokens: 1024
EOF

lighteval endpoint litellm \
  "${CONFIG_FILE}" \
  "lighteval|gpqa:diamond|0|0" \
  --save-details


rm -f ${CONFIG_FILE}
done


# # Create the config file needed by lighteval
# UNIQUE_ID=$(date +%s%N | cut -b1-13)
# CONFIG_FILE="tmp_litellm_config_${UNIQUE_ID}.yaml"
# cat > ${CONFIG_FILE} << EOF
# model_parameters:
#   model_name: "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"
#   generation_parameters:
#     temperature: 0.1
#     max_new_tokens: 4096
# EOF

# lighteval endpoint litellm \
#   "${CONFIG_FILE}" \
#   "lighteval|gpqa:diamond|0|0"

# rm -f ${CONFIG_FILE}