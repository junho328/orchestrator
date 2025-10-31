export WANDB_DISABLED=true
for i in {1..3}; do
    echo "Starting eval $i..."
    CUDA_VISIBLE_DEVICES=4 scripts/launch.sh 1 cfgs/run_cfg/run_conductor_mix_mmrl.yaml offload report_to=null output_dir=debug \
    evaluate_only=path/to/checkpoint \
    task_module=guf.tasks.mmlu task_class=MMLUTask eval_max_samples=2000 chunk_size=20 temperature=0.2 gemini_thinking_budget=32768 claude_thinking_budget=32768 \
    max_agent_tokens=64000 gpt_reasoning_effort=high final_agent_knowledge=false \
    training_tasks='["mmlu"]' router_question_format_style=v1_7c2_2_ood closed_models='["gemini-2.5-pro", "claude-sonnet-4-20250514", "gpt-5"]' || true
    echo "Eval $i done"
done