export LITELLM_CONCURRENT_CALLS=20
export USE_guf_TEMPLATE=true

# Create the config file needed by lighteval
UNIQUE_ID=$(date +%s%N | cut -b1-13)
CONFIG_FILE="tmp_litellm_config_${UNIQUE_ID}.yaml"
cat > ${CONFIG_FILE} << EOF
model_parameters:
  model_name: "gpt-4o"
  provider: "openai"
  api_key: "EMPTY"
  base_url: "http://localhost:8088/v1/" 
  generation_parameters:
    temperature: 0.1
    max_new_tokens: 2048
EOF

# ------ WARNING ------- #
# When running custom math500_pass_at1_metric.py, the conductor_engine must be configured with clean_final_response: False
# Ensure the conductor server is running with this attribute set correctly in the config file, otherwise the final response will be cleaned
# and the answer extraction will fail. 
# ------ WARNING ------- #

# Launch the evaluation of 10 samples in math_500 with custom metric
lighteval endpoint litellm \
  "${CONFIG_FILE}" \
  "lighteval|math_500|0|0" \
  --save-details \
  --custom-tasks examples/py/math500_pass_at1_metric.py

rm -f ${CONFIG_FILE}