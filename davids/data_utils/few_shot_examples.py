"""Few-shot examples for orchestrator prompts."""
from typing import List, Dict

# Few-shot examples for orchestrator training
# Each example should follow the format used in the prompt template

FEW_SHOT_EXAMPLES = [
    {
        "user_question": "Write a Python function to calculate the factorial of a number.",
        "model_id": [0, 3],
        "subtasks": [
            "Write a recursive function to calculate factorial.",
            "Test the function with various inputs and verify correctness."
        ],
        "access_list": [[], [0]]
    },
    {
        "user_question": "Implement a binary search algorithm in Python.",
        "model_id": [0, 1],
        "subtasks": [
            "Implement the core binary search algorithm with proper boundary handling.",
            "Add edge case testing and validation for the binary search implementation."
        ],
        "access_list": [[], [0]]
    },
    {
        "user_question": "Create a function that sorts a list using quicksort algorithm.",
        "model_id": [0],
        "subtasks": [
            "Implement quicksort algorithm with proper pivot selection and partitioning."
        ],
        "access_list": [[]]
    }
]


def format_few_shot_example(example: Dict) -> str:
    """
    Format a few-shot example into the prompt format.
    
    Args:
        example: Dictionary with 'user_question', 'model_id', 'subtasks', 'access_list'
        
    Returns:
        Formatted string representation of the example
    """
    formatted = f"USER QUESTION:\n{example['user_question']}\n\n"
    formatted += f"AVAILABLE LANGUAGE MODELS:\n"
    # Models will be formatted in the main prompt
    
    formatted += "\nmodel_id = ["
    formatted += ", ".join(str(mid) for mid in example['model_id'])
    formatted += "]\n\n"
    
    formatted += "subtasks = [\n"
    for subtask in example['subtasks']:
        formatted += f'    "{subtask}",\n'
    formatted += "]\n\n"
    
    formatted += "access_list = [\n"
    for access in example['access_list']:
        if access == "all":
            formatted += '    "all",\n'
        elif len(access) == 0:
            formatted += "    [],\n"
        else:
            formatted += f"    {access},\n"
    formatted += "]\n"
    
    return formatted


def get_few_shot_examples(num_examples: int = 3) -> str:
    """
    Get few-shot examples formatted for the prompt.
    
    Args:
        num_examples: Number of examples to include
        
    Returns:
        Formatted string with few-shot examples
    """
    examples = FEW_SHOT_EXAMPLES[:num_examples]
    formatted_examples = []
    
    for example in examples:
        formatted_examples.append(format_few_shot_example(example))
    
    return "\n\n---\n\n".join(formatted_examples)


