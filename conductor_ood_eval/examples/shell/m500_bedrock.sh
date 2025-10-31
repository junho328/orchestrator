# Create the config file needed by lighteval
UNIQUE_ID=$(date +%s%N | cut -b1-13)
CONFIG_FILE="tmp_litellm_config_${UNIQUE_ID}.yaml"
cat > ${CONFIG_FILE} << EOF
model_parameters:
  model_name: "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"
  generation_parameters:
    temperature: 0.1
    max_new_tokens: 4096
EOF

lighteval endpoint litellm \
  "${CONFIG_FILE}" \
  "lighteval|math_500|0|0"

rm -f ${CONFIG_FILE}