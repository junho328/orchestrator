# OOD_eval: OAI-Compatible Server-Based Evaluation.
This repo presents an OAI server-based evaluation pipeline. It wraps up whatever models we have into an OAI server and then plugs into existing evaluation frameworks, including lighteval, tau-bench, etc.


## Weight download from anymouse huggingface repo
[Conductor Weights](https://huggingface.co/AnonSubmission38/Conductor)

## Quick Start

1. **Install dependencies**:
```bash
# install the uv first (optional, you can use another package manager as well).
# python 3.12 is preferred.
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python pin 3.12
uv venv
source .venv/bin/activate

# install the package.
uv pip install -r requirements.txt
```

3. **Install third-party libraries**:
```bash
# Install lighteval
uv pip install -e third_party/lighteval/
uv pip install -e third_party/lighteval[extended_tasks]

# Install mt-bench
uv pip install -e third_party/FastChat

# Install big-code-bench
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_BIGCODEBENCH="1.0.0"
uv pip install -e third_party/bigcodebench
```

4. **Launch the server & Test the API**:
```bash
# NOTE: Skip this part if you use your own hosted server
# Please directly use the example scripts below to evaluate your server!
python main.py --config configs/example.json
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Empty" \
  -d '{
    "model": "xxx",
    "messages": [{"role": "user", "content": "How many r in word strawberry?"}],
    "temperature": 0.1,
    "max_tokens": 1024
  }'
```

5. **Real evaluation**: 
- see examples/* for eval scripts.

## Architecture

```
├── models/
│   ├── __init__.py
│   ├── base.py              # Base model abstractions
│   ├── router_engine.py      # Router model implementation
│   └── single_engine.py      # Agent model implementations
├── server/
│   ├── __init__.py
│   ├── api.py              # OpenAI-compatible API
│   ├── config.py           # Configuration management
│   ├── error_handling.py   # Error handling and reliability
│   └── logging_config.py   # Logging configuration
├── third_party/             # Git submodules for evaluation frameworks
│   ├── lighteval/          # Hugging Face's LLM evaluation toolkit
│   └── tau-bench/          # Tool-Agent-User interaction benchmark
├── llm_clients.py          # Standalone LLM clients
├── model_utils.py          # Standalone model utilities
├── main.py                 # Server orchestration
└── requirements.txt
```
