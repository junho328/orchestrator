#!/bin/bash

# Multi-agent BigCodeBench Instruct Test Script
# This script runs multi-agent inference on BigCodeBench instruct dataset

export PATH=./bin:$PATH

# Configuration
export MODEL_DIR=${1:-"Qwen/Qwen2.5-Coder-14B-Instruct"}  # Default model if not provided
export TP=${2:-2}  # Tensor parallelism
export OUTPUT_DIR=${3:-"/ext_hdd/jhna/lamas/bcb_results/qwen2.5_coder_14b_instruct_multi"}  # Output directory
export CONFIG_PATH=${4:-"/home/jhna/MARTI/marti/cli/configs/ma_code_chain.yaml"}  # Multi-agent config

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "Multi-Agent BigCodeBench Instruct Test"
echo "=========================================="
echo "Model: ${MODEL_DIR}"
echo "Tensor Parallelism: ${TP}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Config Path: ${CONFIG_PATH}"
echo "=========================================="

run_multi_agent_benchmark() {
    # SPLIT=$1
    # SUBSET=$2
    SPLIT="instruct"
    SUBSET="full"

    echo "Running multi-agent inference for ${SPLIT} split, ${SUBSET} subset..."

    # Generate code completions using multi-agent workflow
    python generate_multi_agent.py \
        --model ${MODEL_DIR} \
        --config ${CONFIG_PATH} \
        --split ${SPLIT} \
        --subset ${SUBSET} \
        --greedy \
        --bs 1 \
        --temperature 0.8 \
        --n_samples 1 \
        --resume \
        --backend vllm \
        --tp ${TP} \
        --save_path ${OUTPUT_DIR}/completion.jsonl

    echo "Multi-agent code generation completed!"

    # Sanitize and calibrate the generated samples
    echo "Sanitizing generated samples..."
    python sanitize.py \
        --samples ${OUTPUT_DIR}/completion.jsonl \
        --calibrate

    # Evaluate the sanitized and calibrated completions
    echo "Evaluating generated samples..."
    python evaluate.py \
        --split ${SPLIT} \
        --subset ${SUBSET} \
        --no-gt \
        --samples ${OUTPUT_DIR}/completion-sanitized-calibrated.jsonl
    
    echo "Evaluation completed!"
    
    # Clean up environment after evaluation
    echo "Cleaning up environment..."
    pids=$(ps -u $(id -u) -o pid,comm | grep 'bigcodebench' | awk '{print $1}'); 
    if [ -n "$pids" ]; then 
        echo $pids | xargs -r kill; 
    fi
    rm -rf /tmp/*
}

# Run multi-agent benchmark
echo "Starting multi-agent benchmark..."
run_multi_agent_benchmark

echo "=========================================="
echo "Multi-Agent BigCodeBench Test Completed!"
echo "Results saved in: ${OUTPUT_DIR}"
echo "=========================================="