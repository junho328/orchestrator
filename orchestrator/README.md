# Orchestrator Training

This project implements training code for an orchestrator model that coordinates multiple worker agents to solve complex tasks. The orchestrator learns to:
1. Decompose problems into subtasks
2. Select appropriate worker agents for each subtask
3. Generate role instructions for each agent

## Features

- **Training with PPO/GRPO**: Uses reinforcement learning to train the orchestrator based on final output rewards
- **BigCodeBench Dataset**: Trains on code generation tasks from BigCodeBench
- **Local HuggingFace Models**: Uses publicly available HuggingFace models instead of API calls
- **Multi-Agent Coordination**: Orchestrates multiple worker agents to solve complex tasks
- **Unittest-based Evaluation**: Uses BigCodeBench's unittest evaluation to compute rewards (1.0 if all tests pass, 0.0 otherwise)
- **Multi-GPU Support**: Supports distributed training with Accelerate (FSDP, DeepSpeed)

## Structure

```
orchestrator/
├── train.py                    # Main training script
├── trainers/
│   └── orchestrator_engine.py # Reward function and trainer classes
├── datasets/
│   ├── bigcodebench_dataset.py # BigCodeBench dataset loader
│   ├── orchestrator_dataset.py # Dataset class for training
│   └── few_shot_examples.py    # Few-shot examples configuration
├── utils/
│   ├── model_utils.py          # Utilities for loading local models
│   └── bigcodebench_eval.py    # BigCodeBench evaluation utilities
├── accelerate_configs/         # Accelerate configuration files
├── scripts/
│   └── launch.sh               # Launch script for multi-GPU training
├── cfgs/                       # Configuration files
│   ├── train.yaml             # Main training config
│   ├── model_cfg/             # Model configurations
│   ├── data_cfg/              # Dataset configurations
│   └── trainer_cfg/           # Trainer configurations
└── requirements.txt            # Dependencies

```

## Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install the orchestrator package in editable mode


```bash
cd /home/USER/orchestrator  # 프로젝트 루트로 이동
pip install -e .
```

또는 uv를 사용하는 경우:

```bash
cd /home/USER/orchestrator
uv pip install -e .
```

## Usage

### Single GPU Training

```bash
cd orchestrator
python train.py
```

### Multi-GPU Training with Accelerate

#### Option 1: Using accelerate launch directly

```bash
accelerate launch \
  --num_processes 4 \
  --main_process_port 29500 \
  --config_file accelerate_configs/deepspeed_zero3.yaml \
  train.py \
  model_cfg@_global_=base \
  data_cfg@_global_=bigcodebench \
  trainer_cfg@_global_=grpo \
  run_cfg@_global_=default
```

#### Option 2: Using launch script

```bash
# 8 GPUs with DeepSpeed Zero 3
./scripts/launch.sh 8 default

# With CPU offloading (for large models with limited GPU memory)
./scripts/launch.sh 8 default offload

# With FSDP instead of DeepSpeed
./scripts/launch.sh 8 default fsdp

# With custom Hydra args
./scripts/launch.sh 8 default trainer_cfg=ppo max_steps=1000
```

### Configuration Options

You can override any configuration using Hydra:

```bash
# Change model
python train.py model_name_or_path=Qwen/Qwen2.5-14B-Instruct

# Change trainer
python train.py trainer_cfg=ppo

# Change training parameters
python train.py train_batch_size=512 max_steps=1000

# Multi-GPU with custom config
accelerate launch --config_file accelerate_configs/fsdp.yaml train.py \
  model_name_or_path=Qwen/Qwen2.5-14B-Instruct \
  train_batch_size=256
```

### Evaluation

```bash
# Single GPU evaluation
python train.py evaluate_only=/path/to/checkpoint

# Multi-GPU evaluation
accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml \
  train.py evaluate_only=/path/to/checkpoint
```

## Accelerate Configuration Files

- **deepspeed_zero3.yaml**: DeepSpeed ZeRO Stage 3 (recommended for large models)
- **deepspeed_zero3_cpu_offloading.yaml**: DeepSpeed ZeRO Stage 3 with CPU offloading (for limited GPU memory)
- **deepspeed_zero1.yaml**: DeepSpeed ZeRO Stage 1 (less memory efficient but faster)
- **fsdp.yaml**: FSDP (Fully Sharded Data Parallel) - alternative to DeepSpeed
- **fsdp_qlora.yaml**: FSDP with QLoRA support

## Important Notes

1. **Multi-GPU Training**: 
   - Use `accelerate launch` or `./scripts/launch.sh` for distributed training
   - Adjust `num_processes` in the accelerate config to match your GPU count
   - Make sure to set `per_device_train_batch_size` appropriately so total batch size = `per_device_batch_size * num_gpus`

2. **Memory Management**:
   - For large models, use CPU offloading configs
   - Adjust `vllm_gpu_memory_utilization` if using vLLM
   - Worker models are loaded on each GPU independently

3. **Configuration**:
   - Worker models are specified in `cfgs/data_cfg/bigcodebench.yaml`
   - Each worker model needs to fit in GPU memory
   - Consider using smaller worker models if GPU memory is limited

## License

[Add your license here]
