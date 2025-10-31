#!/bin/bash

offload_found=false
zero1_found=false
offload_optim_found=false
args=()
for arg in "$@"; do
  if [ "$arg" = "offload" ]; then
    offload_found=true
  elif [ "$arg" = "offload_optim" ]; then
    offload_optim_found=true
  elif [ "$arg" = "zero1" ]; then
    zero1_found=true
  else
    args+=("$arg")
  fi
done

if [ "$offload_optim_found" = true ]; then
  echo "Offload optim found in arguments. Using deepspeed_zero3_cpu_offloading_optim.yaml."
  config="accelerate_configs/deepspeed_zero3_cpu_offloading_optim.yaml"
elif [ "$offload_found" = true ]; then
  echo "Offload found in arguments. Using deepspeed_zero3_cpu_offloading.yaml."
  config="accelerate_configs/deepspeed_zero3_cpu_offloading.yaml"
elif [ "$zero1_found" = true ]; then
  echo "Zero1 found in arguments. Using deepspeed_zero1.yaml."
  config="accelerate_configs/deepspeed_zero1.yaml"
else
  echo "No offload, zero1, or offload_optim found in arguments. Defaulting to deepspeed_zero3.yaml."
  config="accelerate_configs/deepspeed_zero3.yaml"
fi

nproc=${args[0]}
arg2=${args[1]:-"default"}

prefix="cfgs/run_cfg/"
if [[ "$arg2" == $prefix* ]]; then
    arg2="${arg2#$prefix}"
fi

extra_args=("${args[@]:2}")

# generate random port
RND_PORT=$(($RANDOM % 1000 + 12000))
echo $RND_PORT


echo "Running the following command:"
echo "accelerate launch --num_processes $nproc --main_process_port $RND_PORT --config_file $config train.py run_cfg@_global_=$arg2 ${extra_args[@]}"

accelerate launch --num_processes "$nproc" \
  --main_process_port "$RND_PORT" \
  --config_file "$config" \
  train.py run_cfg@_global_="$arg2" "${extra_args[@]}"

