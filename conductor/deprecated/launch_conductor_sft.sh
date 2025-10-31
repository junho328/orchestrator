export WANDB_DISABLED=true

CUDA_VISIBLE_DEVICES=4,5 accelerate launch \
  --config_file accelerate_configs/deepspeed_zero3_cpu_offloading.yaml \
  --num_processes 2 --main_process_port 19031 \
  train.py run_cfg@_global_=sft_conductor \
  dataset_local_directory='${hydra:runtime.cwd}/data/router_demos/debug' \
  output_dir=debug/qwen7bi_sft_coldstart