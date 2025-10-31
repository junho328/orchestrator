export WANDB_DISABLED=true
for i in {1..3}; do
    echo "Starting eval $i..."
    CUDA_VISIBLE_DEVICES=0 scripts/launch.sh 1 cfgs/run_cfg/run_recursion_mix_mmrl.yaml offload report_to=null output_dir=debug \
    evaluate_only=path/to/checkpoint \
    task_module=guf.tasks.rlpr task_class=RLPRTask eval_max_samples=2000 chunk_size=20 temperature=0.2 recursion_question_format=v1 gemini_thinking_budget=128 claude_thinking_budget=0 gpt_reasoning_effort=minimal \
    training_tasks='["rlpr"]' router_question_format_style=v1_7c2_2_ood closed_models='["gemini-2.5-pro", "claude-sonnet-4-20250514", "gpt-5"]' max_model_length=32000 || true
    echo "Eval $i done"
done



