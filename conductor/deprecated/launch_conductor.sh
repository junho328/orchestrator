export MASTER_ADDR=localhost
#mix_dmrp with offloading
CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
  --config_file accelerate_configs/deepspeed_zero3_cpu_offloading.yaml \
  --num_processes 2 --main_process_port 21298 \
  train.py --config-name run_conductor_mix_mmmrlc max_prompt_length=8192 beta=0.0 chunk_size=10 \
  cost_bonus_weight=0.00 per_device_train_batch_size=1
