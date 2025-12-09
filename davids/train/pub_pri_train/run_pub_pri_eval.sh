#!/bin/bash
# =============================================================================
# Multi-Agent Evaluation Script for Public-Private GRPO
# =============================================================================
#
# Usage:
#   bash run_pub_pri_eval.sh [OPTIONS]
#
# Options:
#   --checkpoint_path PATH   Path to trained checkpoint directory
#   --model_name MODEL       Base model name/path (default: Qwen/Qwen2.5-1.5B-Instruct)
#   --dataset DATASET        Dataset name (default: HuggingFaceH4/MATH-500)
#   --num_samples N          Samples per problem for pass@k (default: 8)
#   --k_values K             Comma-separated k values (default: 1,4,8)
#   --batch_size B           Eval batch size (default: 4)
#   --output_path PATH       Output JSON path (default: eval_results.json)
#   --max_samples N          Max eval samples (default: all)
#
# =============================================================================

set -e

# Multi-GPU settings
NUM_GPUS=1  # Number of GPUs to use (set to 1 for single GPU)
export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=0,1,2,3  # Uncomment to specify which GPUs

# Default values
MODEL_NAME="/home/work/aipr-jhna/huggingface_hub/Qwen2.5-Coder-3B-Instruct"
DATASET_NAME="/home/work/aipr-jhna/huggingface_hub/hendrycks-math-with-answers"
CHECKPOINT_PATH="/home/work/aipr-jhna/output/checkpoint-118"
PUBLIC_ADAPTER_PATH="/home/work/aipr-jhna/output/checkpoint-118/public"
PRIVATE_ADAPTER_PATH="/home/work/aipr-jhna/output/checkpoint-118/private"
NUM_SAMPLES_PER_PROBLEM=8
K_VALUES="1,4,8"
EVAL_BATCH_SIZE=4
OUTPUT_PATH="/home/work/aipr-jhna/output/eval/eval_results.json"
NUM_AGENTS=2
MAX_PROMPT_LENGTH=1024
MAX_COMPLETION_LENGTH=1024
PUBLIC_MAX_COMPLETION=512
PRIVATE_MAX_COMPLETION=1024
MAX_EVAL_SAMPLES=5000

# vLLM settings
USE_VLLM=True
VLLM_MODE="colocate"
VLLM_GPU_MEMORY_UTILIZATION=0.8
VLLM_TENSOR_PARALLEL_SIZE=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint_path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --public_adapter_path)
            PUBLIC_ADAPTER_PATH="$2"
            shift 2
            ;;
        --private_adapter_path)
            PRIVATE_ADAPTER_PATH="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --dataset)
            DATASET_NAME="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES_PER_PROBLEM="$2"
            shift 2
            ;;
        --k_values)
            K_VALUES="$2"
            shift 2
            ;;
        --batch_size)
            EVAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --max_samples)
            MAX_EVAL_SAMPLES="$2"
            shift 2
            ;;
        --num_agents)
            NUM_AGENTS="$2"
            shift 2
            ;;
        --public_max_completion)
            PUBLIC_MAX_COMPLETION="$2"
            shift 2
            ;;
        --private_max_completion)
            PRIVATE_MAX_COMPLETION="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --use_vllm)
            USE_VLLM=true
            shift 1
            ;;
        --vllm_mode)
            VLLM_MODE="$2"
            shift 2
            ;;
        --vllm_gpu_memory_utilization)
            VLLM_GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --vllm_tensor_parallel_size)
            VLLM_TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash run_pub_pri_eval.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --checkpoint_path PATH     Path to trained checkpoint directory"
            echo "  --public_adapter_path PATH Path to public adapter (alternative to checkpoint)"
            echo "  --private_adapter_path PATH Path to private adapter (alternative to checkpoint)"
            echo "  --model_name MODEL         Base model name/path"
            echo "  --dataset DATASET          Dataset name"
            echo "  --num_samples N            Samples per problem for pass@k"
            echo "  --k_values K               Comma-separated k values"
            echo "  --batch_size B             Eval batch size"
            echo "  --output_path PATH         Output JSON path"
            echo "  --max_samples N            Max eval samples"
            echo "  --num_agents N             Number of agents"
            echo "  --public_max_completion N  Max completion length for public agent"
            echo "  --private_max_completion N Max completion length for private agent"
            echo "  --num_gpus N               Number of GPUs to use (default: 1)"
            echo "  --use_vllm                 Enable vLLM for fast generation"
            echo "  --vllm_mode MODE           vLLM mode: 'colocate' or 'server' (default: colocate)"
            echo "  --vllm_gpu_memory_utilization F  GPU memory utilization (default: 0.8)"
            echo "  --vllm_tensor_parallel_size N    Tensor parallel size (default: 1)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print configuration
echo "============================================================================="
echo "Multi-Agent Evaluation Configuration"
echo "============================================================================="
echo "Base Model:              ${MODEL_NAME}"
echo "Dataset:                 ${DATASET_NAME}"
echo "Checkpoint Path:         ${CHECKPOINT_PATH:-'(none - using untrained adapters)'}"
echo "Public Adapter Path:     ${PUBLIC_ADAPTER_PATH:-'(from checkpoint)'}"
echo "Private Adapter Path:    ${PRIVATE_ADAPTER_PATH:-'(from checkpoint)'}"
echo "Samples per Problem:     ${NUM_SAMPLES_PER_PROBLEM}"
echo "K Values:                ${K_VALUES}"
echo "Batch Size:              ${EVAL_BATCH_SIZE}"
echo "Output Path:             ${OUTPUT_PATH}"
echo "Max Eval Samples:        ${MAX_EVAL_SAMPLES:-'(all)'}"
echo "Num Agents:              ${NUM_AGENTS}"
echo "Public Max Completion:   ${PUBLIC_MAX_COMPLETION}"
echo "Private Max Completion:  ${PRIVATE_MAX_COMPLETION}"
echo "Num GPUs:                ${NUM_GPUS}"
echo "Use vLLM:                ${USE_VLLM}"
if [ "${USE_VLLM}" = true ]; then
    echo "vLLM Mode:               ${VLLM_MODE}"
    echo "vLLM GPU Mem Util:       ${VLLM_GPU_MEMORY_UTILIZATION}"
    echo "vLLM TP Size:            ${VLLM_TENSOR_PARALLEL_SIZE}"
fi
echo "============================================================================="

# Build base arguments
ARGS="--model_name_or_path ${MODEL_NAME} \
    --dataset_name ${DATASET_NAME} \
    --num_samples_per_problem ${NUM_SAMPLES_PER_PROBLEM} \
    --k_values ${K_VALUES} \
    --eval_batch_size ${EVAL_BATCH_SIZE} \
    --output_path ${OUTPUT_PATH} \
    --num_agents ${NUM_AGENTS} \
    --public_agent_max_completion_length ${PUBLIC_MAX_COMPLETION} \
    --private_agent_max_completion_length ${PRIVATE_MAX_COMPLETION} \
    --max_prompt_length ${MAX_PROMPT_LENGTH} \
    --max_completion_length ${PRIVATE_MAX_COMPLETION} \
    --output_dir /home/work/aipr-jhna/output/eval \
    --use_vllm \
    --vllm_mode ${VLLM_MODE} \
    --vllm_gpu_memory_utilization ${VLLM_GPU_MEMORY_UTILIZATION} \
    --vllm_tensor_parallel_size ${VLLM_TENSOR_PARALLEL_SIZE}"

# Add checkpoint path if provided
if [ -n "${CHECKPOINT_PATH}" ]; then
    ARGS="${ARGS} --checkpoint_path ${CHECKPOINT_PATH}"
fi

# Add explicit adapter paths if provided
if [ -n "${PUBLIC_ADAPTER_PATH}" ]; then
    ARGS="${ARGS} --public_adapter_path ${PUBLIC_ADAPTER_PATH}"
fi
if [ -n "${PRIVATE_ADAPTER_PATH}" ]; then
    ARGS="${ARGS} --private_adapter_path ${PRIVATE_ADAPTER_PATH}"
fi

# Add max samples if provided
if [ -n "${MAX_EVAL_SAMPLES}" ]; then
    ARGS="${ARGS} --max_eval_samples ${MAX_EVAL_SAMPLES}"
fi

# Add vLLM settings if enabled
if [ "${USE_VLLM}" = true ]; then
    ARGS="${ARGS} --use_vllm"
    ARGS="${ARGS} --vllm_mode ${VLLM_MODE}"
    ARGS="${ARGS} --vllm_gpu_memory_utilization ${VLLM_GPU_MEMORY_UTILIZATION}"
    ARGS="${ARGS} --vllm_tensor_parallel_size ${VLLM_TENSOR_PARALLEL_SIZE}"
fi

# Build final command with accelerate for multi-GPU
if [ "${NUM_GPUS}" -gt 1 ]; then
    CMD="accelerate launch --num_processes ${NUM_GPUS} -m davids.train.pub_pri_train.pub_pri_math_eval ${ARGS}"
else
    CMD="python -m davids.train.pub_pri_train.pub_pri_math_eval ${ARGS}"
fi

echo ""
echo "Running command:"
echo "${CMD}"
echo ""

# Run evaluation
eval ${CMD}

echo ""
echo "============================================================================="
echo "Evaluation completed! Results saved to: ${OUTPUT_PATH}"
echo "============================================================================="

