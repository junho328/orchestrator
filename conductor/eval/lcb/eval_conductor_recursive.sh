export WANDB_DISABLED=true
for i in {1..2}; do
    echo "Starting eval $i..."
    CUDA_VISIBLE_DEVICES=1 scripts/launch.sh 1 cfgs/run_cfg/run_recursion_mix_mmrl.yaml offload report_to=null output_dir=debug \
    evaluate_only=path/to/checkpoint \
    task_module=guf.tasks.livecodebench task_class=LiveCodeBenchTask eval_max_samples=2000 chunk_size=20 temperature=0.2 recursion_question_format=v1 \
    training_tasks='["livecodebench"]' router_question_format_style=v1_7c2_2 closed_models='["gemini-2.5-pro", "claude-sonnet-4-20250514", "gpt-5"]' || true
    echo "Eval $i done"
done

