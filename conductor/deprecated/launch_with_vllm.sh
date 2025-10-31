#!/bin/bash
# usage: bash run_vllm.sh NUM_VLLM_GPUS NUM_TRAIN_GPUS YAML \
#        [model_name_or_path=…] [...]

# gpus for vllm
num_gpus=$1
# gpus for training
num_gpus_2=$2

# locate yaml
if [[ "$3" == cfgs/run_cfg/* ]]; then
  yaml_file="$3"
else
  yaml_file="cfgs/run_cfg/$3"
fi

# parse extra args, capture model override
custom_model=""
extra_args=()
for arg in "${@:4}"; do
  if [[ "$arg" == model_name_or_path=* ]]; then
    custom_model="${arg#model_name_or_path=}"
  else
    extra_args+=("$arg")
  fi
done

echo "starting vllm wrapper…"

cleanup() {
  echo "cleaning up background processes"
  kill $(jobs -p) 2>/dev/null || true
  unset CUDA_VISIBLE_DEVICES
  pkill -9 -f 'trl vllm-serve' || true
}
trap 'cleanup; exit 1' SIGINT SIGTERM
trap 'cleanup' EXIT

# read values from yaml (may be empty)
model_name=$(grep -m1 '^model_name_or_path:' "$yaml_file" | awk '{print $2}')
host=$(grep -m1 '^vllm_server_host:' "$yaml_file" | awk '{print $2}')
port=$(grep -m1 '^vllm_server_port:' "$yaml_file" | awk '{print $2}')
gpu_mem=$(grep -m1 '^vllm_gpu_memory_utilization:' "$yaml_file" \
         | awk '{print $2}')

: "${gpu_mem:=0.9}"

# optional model override
if [[ -n "$custom_model" ]]; then
  model_name="$custom_model"
  if grep -q '^model_name_or_path:' "$yaml_file"; then
    sed -i "s|^model_name_or_path:.*|model_name_or_path: $model_name|" \
      "$yaml_file"
  else
    echo "model_name_or_path: $model_name" >> "$yaml_file"
  fi
fi

[[ -z "$model_name" ]] && {
  echo "error: model_name_or_path missing"; exit 1; }

# build optional host/port flags
host_flag=""
port_flag=""
if [[ -n "$host" ]]; then host_flag="--host $host"; fi
if [[ -n "$port" ]]; then port_flag="--port $port"; fi

# devices for vllm
server_devs=$(seq -s, 0 $((num_gpus - 1)))

# prefix caching flag if env var set
if [[ -n "$enable_prefix_caching" ]]; then
  prefix_flag="--enable_prefix_caching $enable_prefix_caching"
else
  prefix_flag=""
fi

echo "Running the following vLLM server command:"
echo "CUDA_VISIBLE_DEVICES=$server_devs trl vllm-serve --model $model_name --tensor-parallel-size 1 --data-parallel-size $num_gpus --gpu-memory-utilization $gpu_mem $host_flag $port_flag $prefix_flag"
# launch one vllm-serve (pipeline/data parallel >1 gpu)
CUDA_VISIBLE_DEVICES=$server_devs trl vllm-serve \
  --model "$model_name" \
  --tensor-parallel-size 1 \
  --data-parallel-size "$num_gpus" \
  --gpu-memory-utilization "$gpu_mem" \
  --enforce-eager true \
  $host_flag $port_flag $prefix_flag \
  > vllm_server.log 2>&1 &

# pick values for readiness probe
probe_host=${host:-localhost}
probe_port=${port:-8000}

echo -n "waiting for vllm server"
elapsed=0
until curl -s "http://${probe_host}:${probe_port}/health" >/dev/null 2>&1; do
  sleep 5
  elapsed=$((elapsed + 5))
  echo -n "."
done
echo " up!"

# choose gpus for training
start_dev=$num_gpus
end_dev=$((num_gpus + num_gpus_2 - 1))
train_devs=$(seq -s, $start_dev $end_dev)
echo "training gpus: $train_devs"

# hand off to launch.sh
CUDA_VISIBLE_DEVICES=$train_devs bash scripts/launch.sh "$num_gpus_2" \
  "$yaml_file" "${extra_args[@]}"

unset CUDA_VISIBLE_DEVICES
