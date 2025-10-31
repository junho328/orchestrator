# curl -X POST http://slurm0us-a3nodeset-3:8323/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Empty" \
#   -d '{
#     "model": "google/gemma-3-27b-it",
#     "messages": [{"role": "user", "content": "Write me python code for a binary search?"}],
#     "temperature": 0.1,
#     "max_tokens": 10240
#   }'

curl -X POST http://localhost:2468/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Empty" \
  -d '{
    "model": "xxx",
    "messages": [{"role": "user", "content": "Write a python program to calculate the nth number in the Fibonacci sequence. This is a hard question for which you should try to continually refine your answer."}],
    "temperature": 0.1,
    "max_tokens": 1024
  }'