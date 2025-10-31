

import boto3
from botocore.exceptions import ClientError


client = boto3.client("bedrock-runtime", region_name="us-east-1")


model_id = "anthropic.claude-3-haiku-20240307-v1:0"

resource_arn = f"arn:aws:bedrock:us-east-1::foundation-model/{model_id}"


user_message = "Describe the purpose of a 'hello world' program in one line."
conversation = [
    {
        "role": "user",
        "content": [{"text": user_message}],
    }
]

try:
    
    response = client.converse(
        modelId=model_id,
        messages=conversation,
        inferenceConfig={"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
    )

    
    response_text = response["output"]["message"]["content"][0]["text"]
    print(response_text)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)