#!/bin/bash

# gpus for vllm
num_gpus=$1
# run yaml path
orig_yaml=$2

# resolve path if shorthand
if [[ "$orig_yaml" != */* ]]; then
  orig_yaml="cfgs/run_cfg/$orig_yaml"
fi

shift 2  # leave only extra args
custom_model=""
extra_args=()
for arg in "$@"; do
  [[ "$arg" == model_name_or_path=* ]] && custom_model="${arg#*=}" \
    || extra_args+=("$arg")
done

echo "starting vllm server only…"

cleanup() {
  echo "cleaning up"
  kill $(jobs -p) 2>/dev/null || true
  pkill -9 -f 'trl vllm-serve' || true
  [[ -n "$tmp_yaml" && -f "$tmp_yaml" ]] && rm -f "$tmp_yaml"
}
trap 'cleanup; exit 1' SIGINT SIGTERM
trap 'cleanup' EXIT

# read values from original yaml (may be empty/missing)
model=$(grep -m1 '^model_name_or_path:' "$orig_yaml" | awk '{print $2}')
host=$(grep -m1 '^vllm_server_host:' "$orig_yaml" | awk '{print $2}')
port=$(grep -m1 '^vllm_server_port:' "$orig_yaml" | awk '{print $2}')
gpu_mem=$(grep -m1 '^vllm_gpu_memory_utilization:' "$orig_yaml" \
          | awk '{print $2}')
: "${gpu_mem:=0.9}"

[[ -z "$model" && -z "$custom_model" ]] && {
  echo "error: model_name_or_path missing"; exit 1; }

[[ -n "$custom_model" ]] && model="$custom_model"

# choose defaults only for the COPY / probe
: "${host:=0.0.0.0}"
: "${port:=8000}"
base_url="http://${host}:${port}"

# build host/port flags only if present in orig yaml
host_flag=""; port_flag=""
grep -q '^vllm_server_host:' "$orig_yaml" && host_flag="--host $host"
grep -q '^vllm_server_port:' "$orig_yaml" && port_flag="--port $port"

# create live_vllm_sessions copy
run_dir=$(dirname "$orig_yaml")
session_dir="$run_dir/live_vllm_sessions"
mkdir -p "$session_dir"
tmp_yaml="$session_dir/$(basename "${orig_yaml%.*}")_live.yaml"
cp "$orig_yaml" "$tmp_yaml"

# inject connection info into copy
sed -i "/^use_vllm:/d" "$tmp_yaml"
sed -i "/^vllm_mode:/d" "$tmp_yaml"
sed -i "/^vllm_server_host:/d" "$tmp_yaml"
sed -i "/^vllm_server_port:/d" "$tmp_yaml"
sed -i "/^vllm_server_base_url:/d" "$tmp_yaml"
{
  echo "use_vllm: true"
  echo "vllm_mode: server"
  echo "vllm_server_host: $host"
  echo "vllm_server_port: $port"
  echo "vllm_server_base_url: $base_url"
} >> "$tmp_yaml"

# override model in copy if needed
if grep -q '^model_name_or_path:' "$tmp_yaml"; then
  sed -i "s|^model_name_or_path:.*|model_name_or_path: $model|" "$tmp_yaml"
else
  echo "model_name_or_path: $model" >> "$tmp_yaml"
fi

echo "live session yaml: $tmp_yaml"

# launch vllm (pipeline/data parallel when >1 gpu)
devs=$(seq -s, 0 $((num_gpus - 1)))
CUDA_VISIBLE_DEVICES=$devs trl vllm-serve \
  --model "$model" \
  --tensor-parallel-size 1 \
  --data-parallel-size "$num_gpus" \
  --gpu-memory-utilization "$gpu_mem" \
  $host_flag $port_flag "${extra_args[@]}" \
  > vllm_server.log 2>&1 &

# wait until healthy
echo -n "waiting for vllm server"
until curl -s "$base_url/health" >/dev/null 2>&1; do
  sleep 5; echo -n "."
done
echo " up!"

echo "server running; connect with: launch.sh N_GPUS \"$tmp_yaml\" …"
wait
