from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def hendrycks_math_prompt(line, task_name: str=None):
    # Prompt template adapted from
    # - simple-evals: https://github.com/openai/simple-evals/blob/6e84f4e2aed6b60f6a0c7b8f06bbbf4bfde72e58/math_eval.py#L17
    # - Llama 3: https://huggingface.co/datasets/meta-llama/Llama-3.2-1B-Instruct-evals/viewer/Llama-3.2-1B-Instruct-evals__math__details?views%5B%5D=llama_32_1b_instruct_evals__math__details
    # Note that it is important to have the final answer in a box for math-verify to work correctly
    MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["question"]),
        choices=[line["answer"]],
        gold_index=0,
    )
    
task = LightevalTaskConfig(
    name="hendrycks_math",
    prompt_function=minervamath_prompt,
    suite=["community"],
    hf_subset="",
    hf_repo="jhn9803/hendrycks-math-with-answers",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[
        Metrics.math_pass_at_1_1n,
        Metrics.math_pass_at_1_4n,
        Metrics.math_pass_at_1_8n,
        # Metrics.math_pass_at_1_16n,
        # Metrics.math_pass_at_1_32n,
        # Metrics.math_pass_at_1_64n,
    ],
)

# STORE YOUR EVALS
TASKS_TABLE = [task]
