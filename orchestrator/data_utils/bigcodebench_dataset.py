"""BigCodeBench dataset loader for orchestrator training."""
from datasets import load_dataset
from typing import List, Dict, Optional
import json
import logging

logger = logging.getLogger(__name__)


def load_bigcodebench_dataset(
    split: str = "v0.1.4",
    max_samples: Optional[int] = None,
    seed: int = 42
) -> List[Dict]:
    """
    Load BigCodeBench dataset.
    
    Args:
        split: Dataset split ('train', 'test', or version like 'v0.1.4')
        max_samples: Maximum number of samples to load (None for all)
        seed: Random seed for sampling
        
    Returns:
        List of dataset examples
    """
    try:
        # Map common split names to actual BigCodeBench splits
        split_mapping = {
            "train": "v0.1.4",  # Use latest version for training
        }
        actual_split = split_mapping.get(split, split)
        
        # Load BigCodeBench dataset
        dataset = load_dataset("bigcode/bigcodebench", split=actual_split)
        if split != actual_split:
            logger.info(f"Mapped split '{split}' to '{actual_split}'")
        logger.info(f"Loaded BigCodeBench {actual_split} dataset with {len(dataset)} examples")
        
        if max_samples and max_samples > 0 and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
            logger.info(f"Limited to {max_samples} samples")
        
        # Convert to list of dicts
        examples = []
        for item in dataset:
            example = {
                "task_id": item.get("task_id", ""),
                "instruction": item.get("instruction", ""),
                "test": item.get("test", ""),
                "entry_point": item.get("entry_point", ""),
                "canonical_solution": item.get("canonical_solution", ""),
                "base_question": item.get("instruction", ""),  # For compatibility
            }
            examples.append(example)
        
        return examples
        
    except Exception as e:
        logger.error(f"Error loading BigCodeBench dataset: {e}")
        # Return empty list or raise
        raise


def format_code_generation_prompt(instruction: str, test: str = "") -> str:
    """
    Format a code generation prompt from BigCodeBench instruction.
    
    Args:
        instruction: The task instruction
        test: Optional test code
        
    Returns:
        Formatted prompt string
    """
    prompt = f"Task: {instruction}\n\n"
    
    prompt += "Please provide a self-contained Python script that solves the given task."
    
    return prompt

