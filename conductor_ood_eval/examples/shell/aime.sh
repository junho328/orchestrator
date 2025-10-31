# SERVER="0.0.0.0"
# PORT="4321"
# export HOSTED_VLLM_API_BASE="http://$SERVER:$PORT/v1"
# # export HOSTED_VLLM_API_BASE="http://slurm0us-a3nodeset-3:8321/v1"

# export LITELLM_CONCURRENT_CALLS=10
# export LITELLM_REQUEST_TIMEOUT=10800 # 3 hours

# for run in {1..3}; do
# echo "Running evaluation number $run"
# # Create the config file needed by lighteval.
# UNIQUE_ID=$(date +%s%N | cut -b1-13)
# CONFIG_FILE="tmp_litellm_config_${UNIQUE_ID}.yaml"
# cat > ${CONFIG_FILE} << EOF
# model_parameters:
#   model_name: "gpt-4o"
#   provider: "openai"
#   api_key: "EMPTY"
#   base_url: "$HOSTED_VLLM_API_BASE" 
#   generation_parameters:
#     temperature: 0.2
#     max_new_tokens: 1024
# EOF

# # TODO: pass@16?
# lighteval endpoint litellm \
#   "${CONFIG_FILE}" \
#   "lighteval|aime25|0|0" \
#   --save-details \

# rm -f ${CONFIG_FILE}
# done


# # Create the config file needed by lighteval
# UNIQUE_ID=$(date +%s%N | cut -b1-13)
# CONFIG_FILE="tmp_litellm_config_${UNIQUE_ID}.yaml"
# cat > ${CONFIG_FILE} << EOF
# model_parameters:
#   model_name: "anthropic/claude-sonnet-4-20250514"
#   provider: "anthropic"
#   generation_parameters:
#     temperature: 0.1
#     max_new_tokens: 64000
#     extra_body:
#       thinking:
#         type: "enabled"
#         budget_tokens: 32768
# EOF

# lighteval endpoint litellm \
#   "${CONFIG_FILE}" \
#   "lighteval|aime25|0|0" \
#   --save-details \
#   --max-samples 3 \

# rm -f ${CONFIG_FILE}

# model_name: "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"

# export HOSTED_VLLM_API_BASE="http://slurm0us-a3nodeset-3:8323/v1"
# export LITELLM_CONCURRENT_CALLS=10
# export LITELLM_REQUEST_TIMEOUT=5800 # 1.5 hours
# # Create the config file needed by lighteval.
# UNIQUE_ID=$(date +%s%N | cut -b1-13)
# CONFIG_FILE="tmp_litellm_config_${UNIQUE_ID}.yaml"
# cat > ${CONFIG_FILE} << EOF
# model_parameters:
#   model_name: "openai/google/gemma-3-27b-it"
#   provider: "openai"
#   api_key: "EMPTY"
#   base_url: "$HOSTED_VLLM_API_BASE" 
#   generation_parameters:
#     temperature: 0.2
#     max_new_tokens: 16000
# EOF

# # TODO: pass@16?
# lighteval endpoint litellm \
#   "${CONFIG_FILE}" \
#   "lighteval|aime25|0|0" \
#   --save-details \

# rm -f ${CONFIG_FILE}


export HOSTED_VLLM_API_BASE="http://slurm0us-a3nodeset-2:8324/v1"
export LITELLM_CONCURRENT_CALLS=10
export LITELLM_REQUEST_TIMEOUT=1000
# Create the config file needed by lighteval.
UNIQUE_ID=$(date +%s%N | cut -b1-13)
CONFIG_FILE="tmp_litellm_config_${UNIQUE_ID}.yaml"
cat > ${CONFIG_FILE} << EOF
model_parameters:
  model_name: "openai/Qwen/Qwen3-32B"
  provider: "openai"
  api_key: "EMPTY"
  base_url: "$HOSTED_VLLM_API_BASE" 
  generation_parameters:
    temperature: 0.2
    max_new_tokens: 10000
    extra_body:
      top_p: 0.8
      top_k: 20
      presence_penalty: 1.0
      chat_template_kwargs: {"enable_thinking": true}
EOF

# TODO: pass@16?
lighteval endpoint litellm \
  "${CONFIG_FILE}" \
  "lighteval|aime25|0|0" \
  --save-details \

rm -f ${CONFIG_FILE}

# extra_body:
#       top_p: 0.8
#       top_k: 20
#       presence_penalty: 1.0
#       chat_template_kwargs: {"enable_thinking": true}