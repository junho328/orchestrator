# mix_dmrp with offloading - eval only
export WANDB_DISABLED=true
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file accelerate_configs/deepspeed_zero3_cpu_offloading.yaml \
  --num_processes 2 --main_process_port 11418 \
  train.py --config-name run_conductor_mix_jrlm max_prompt_length=4096 beta=0.0 \
  trainer_args.temperature=0.0 score_repeats=1 report_to=null evaluate_only="results/conductor_mix_jrlm/qwen7bi/conductor_grpo_qwen7bi_temp0.2_v0c_beta0_max2048_old4/2025.07.18035301/checkpoint-160"