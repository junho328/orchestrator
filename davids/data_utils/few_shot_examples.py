"""Few-shot examples for orchestrator prompts."""
from typing import List, Dict


# Few-shot examples for orchestrator training
# Each example should show subtasks and role_instructions as Python lists
FEW_SHOT_EXAMPLES = [
    {
        "user_question": "Write a Python function to calculate the factorial of a number.",
        "subtasks": [
            "Write a recursive function to calculate factorial.",
            "Test the function with various inputs and verify correctness."
        ],
        "role_instructions": [
            "You are a code generator. Write clean, efficient Python code.",
            "You are a code tester. Verify the correctness of code with edge cases."
        ]
    },
    {
        "user_question": "Implement a binary search algorithm in Python.",
        "subtasks": [
            "Implement the core binary search algorithm with proper boundary handling.",
            "Add edge case testing and validation for the binary search implementation."
        ],
        "role_instructions": [
            "You are an algorithm expert. Implement efficient algorithms with proper error handling.",
            "You are a quality assurance engineer. Test code thoroughly and identify edge cases."
        ]
    },
    {
        "user_question": "Create a function that sorts a list using quicksort algorithm.",
        "subtasks": [
            "Implement quicksort algorithm with proper pivot selection and partitioning."
        ],
        "role_instructions": [
            "You are a sorting algorithm specialist. Implement efficient sorting algorithms with optimal time complexity."
        ]
    }
]


def format_few_shot_example(example: Dict) -> str:
    """
    Format a few-shot example into the prompt format.
    
    Args:
        example: Dictionary with 'user_question', 'subtasks', 'role_instructions'
        
    Returns:
        Formatted string representation of the example
    """
    formatted = f"USER QUESTION:\n{example['user_question']}\n\n"
    
    formatted += "subtasks = [\n"
    for subtask in example['subtasks']:
        # Escape quotes and newlines
        escaped_subtask = subtask.replace('"', '\\"').replace('\n', '\\n')
        formatted += f'    "{escaped_subtask}",\n'
    formatted += "]\n\n"
    
    formatted += "role_instructions = [\n"
    for role_instruction in example['role_instructions']:
        # Escape quotes and newlines
        escaped_role = role_instruction.replace('"', '\\"').replace('\n', '\\n')
        formatted += f'    "{escaped_role}",\n'
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
