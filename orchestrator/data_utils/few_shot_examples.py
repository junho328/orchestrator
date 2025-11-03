"""Few-shot examples for orchestrator prompts."""
from typing import List, Dict

# Few-shot examples for orchestrator training
# Each example should follow the format used in the prompt template

FEW_SHOT_EXAMPLES = [
    {
        "user_question": "Subject: Physics. A converging lens is placed 30.0 cm to the right of a diverging lens of focal length 10.0 cm. A beam of parallel light enters the diverging lens from the left, and the beam is again parallel when it emerges from the converging lens. Calculate the focal length of the converging lens.",
        "model_id": [3, 5],
        "subtasks": [
            "Calculate the focal length of the converging lens and provide the final result in the required format. Show your working in <idea> </idea> tags.",
            "Verify the solution proposed by the previous model and refine if necessary. Format the final answer in the required format, using standard mathematical notation and units."
        ],
        "access_list": [[], ["all"]],
        "assistant_response": "Given the simplicity of the problem and the required formatting, we'll use two models that can handle physics calculations. Model 3 can perform the required physics calculation and determine the focal length of the converging lens based on the given conditions. Subsequently, Model 5 can verify the solution, format the result according to the required guidelines, and return the final response."
    },
    {
        "user_question": "Solve the following math problem step by step: A palindrome is a number that reads the same forwards and backwards. The sum of a particular set of three consecutive positive integers is a three-digit palindrome. If the sum is less than 220, what is the greatest possible value for the largest of the three integers in the set?",
        "model_id": [0, 2, 0],
        "subtasks": [
            "Find a set of three consecutive positive integers whose sum is a three-digit palindrome smaller than 220 and calculate the largest of those numbers.",
            "Verify if the number found by the previous agent is the largest set of three consecutive positive integers whose sum is a three-digit palindrome smaller than 220.",
            "Ensure the final answer is formatted correctly according to the rules and provide the necessary LaTeX representation."
        ],
        "access_list": [[], ["all"], ["all"]],
        "assistant_response": "We'll need at least 3 models to solve this. We can use model 0 to first try and solve the question, model 2 can check if the answer is correct given the constraints, and then with the additional context, model 0 can finalize and re-attempt the problem if necessary."
    }
]


def format_few_shot_example(example: Dict) -> str:
    """
    Format a few-shot example into the prompt format.
    
    Args:
        example: Dictionary with 'user_question', 'model_id', 'subtasks', 'access_list', 'assistant_response' (optional)
        
    Returns:
        Formatted string representation of the example
    """
    formatted = f"EXAMPLE:\n"
    formatted += f"Question: {example['user_question']}\n\n"
    
    # Add assistant response if available
    if 'assistant_response' in example and example['assistant_response']:
        formatted += f"Assistant Response: {example['assistant_response']}\n\n"
    
    formatted += "model_id = ["
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
        num_examples: Number of examples to include (will use available examples, max 2)
        
    Returns:
        Formatted string with few-shot examples
    """
    examples = FEW_SHOT_EXAMPLES[:min(num_examples, len(FEW_SHOT_EXAMPLES))]
    formatted_examples = []
    
    for i, example in enumerate(examples, 1):
        formatted_example = format_few_shot_example(example)
        # Replace "EXAMPLE:" with numbered example
        formatted_example = formatted_example.replace("EXAMPLE:\n", f"EXAMPLE {i}:\n")
        formatted_examples.append(formatted_example)
    
    return "\n\n---\n\n".join(formatted_examples)
