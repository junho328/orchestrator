# to deal with built in huggingface wandb
export WANDB_DISABLED=true

python -m custom_data.countdown_coldstart --config-name generate_coldstart_data output_file=debug_gensft.jsonl data_limit=100