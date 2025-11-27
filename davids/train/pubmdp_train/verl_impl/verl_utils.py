import re
from davids.train.utils.pubmdp_prompt import WORKER_PROMPT, ORCHESTRATOR_PROMPT, get_orchestrator_prompt

def make_worker_prompt(original_problem: str, orchestrator_instruction: str) -> str:
    """Create a prompt for a worker agent using WORKER_PROMPT template."""
    return WORKER_PROMPT.format(
        original_problem=original_problem,
        orchestrator_instruction=orchestrator_instruction
    )

def extract_answer(text: str) -> str:
    """Extract answer from <answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

