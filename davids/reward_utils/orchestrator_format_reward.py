"""Reward function for checking orchestrator output format."""
from davids.train.utils.orchestrator_prompt import parse_orchestrator_output


def orchestrator_format_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
    """
    Reward function that checks if the orchestrator output is in the correct format.
    The orchestrator should output two Python lists: subtasks and role_instructions.
    
    Args:
        completions (`list[list[dict[str, str]]]`):
            List of completions to be evaluated. Each completion must be a list of one message, i.e. a dictionary
            containing the key `"content"` with the value being the text of the completion.
        **kwargs:
            Additional keyword arguments. This function does not use them, but they are required in the function
            signature to ensure compatibility with trainers like [`GRPOTrainer`].
    
    Returns:
        `list[float]`:
            A list of rewards, where each reward is 1.0 if the completion matches the expected format, otherwise 0.0.
    """
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        parsed = parse_orchestrator_output(content)
        rewards.append(1.0 if parsed is not None else 0.0)
    return rewards

