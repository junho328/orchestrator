from datasets import load_dataset
from typing import List, Dict, Optional
from .few_shot_examples import get_few_shot_examples
from transformers import PreTrainedTokenizerBase
import logging

logger = logging.getLogger(__name__)


def load_bcb_dataset(
    split: str = "v0.1.4",
    seed: int = 42
):
    """Load BigCodeBench dataset."""
    bcb_dataset = load_dataset("bigcode/bigcodebench", split=split)
    logger.info(f"Loaded BigCodeBench {split} dataset with {len(bcb_dataset)} examples")
    
    return bcb_dataset


class OrchestratorDataset:
    """
    Dataset for orchestrator-agent training.
    
    The orchestrator generates subtasks and role_instructions as Python lists.
    Each agent will then use the same LLM to generate responses.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_name: str,
        split: str,
        max_routing_steps: int = 4,
        seed: int = 42,
        apply_chat_template: bool = True,
        max_prompt_length: int = 1024,
        num_few_shot_examples: int = 2
    ):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.split = split
        self.max_routing_steps = max_routing_steps
        self.seed = seed
        self.apply_chat_template = apply_chat_template
        self.max_prompt_length = max_prompt_length
        self.num_few_shot_examples = num_few_shot_examples
        
        if num_few_shot_examples > 0:
            self.few_shot_examples = get_few_shot_examples(num_few_shot_examples)
        else:
            self.few_shot_examples = None
            
        logger.info(f"Loading {self.dataset_name} {self.split} dataset...")
        
        if self.dataset_name == "bigcodebench":
            self.dataset = load_bcb_dataset(split=self.split, seed=self.seed)
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
        logger.info(f"Loaded {len(self.dataset)} examples")
    
    def __len__(self):
        return len(self.dataset)
    
    def _format_orchestrator_prompt(self, instruction: str):
        """
        Format the orchestrator prompt.
        
        The orchestrator should output:
        - subtasks: List of subtasks as Python list
        - role_instructions: List of role instructions for each agent as Python list
        """
        
        prompt = f"""Your role as an assistant involves obtaining answers to questions by an iterative process of querying language models, each with a different role.

You are given a user-provided question. Your objective is to output a sequence of up to {self.max_routing_steps} workflow steps.

Each step is made of two elements: A subtask and a role instruction for the agent that will handle that subtask.

A subtask could directly ask the language model to solve the given question from scratch, refine the solution of the previous subtask in the sequence, or perform any other completely different task that would facilitate later language models in the sequence to answer the original question with their expertise.

The role instruction should specify what role or expertise the agent should take when solving this subtask.

Based on your answer, the first agent will be prompted with the user question, the first subtask, and the first role instruction. Each following agent in the sequence will be prompted with the full history of the previous subtasks and responses, and will be asked to accomplish its relative subtask with its assigned role.

The answer of the final agent will be provided back as the final solution to the user.

Your response should be provided as two Python lists.

The first list should be called subtasks, and contain the strings that will be used to prompt each agent.

The second list should be called role_instructions, and contain the role instruction strings for each corresponding agent.

For instance:

{self.few_shot_examples}

USER QUESTION:

{instruction}

MAXIMUM NUMBER OF ROUTING STEPS: {self.max_routing_steps}

Please provide your response as two Python lists: subtasks and role_instructions.
""" 
        
        return prompt
    
    def __getitem__(self, idx: int):
        """Get a single dataset item."""
        
        data = self.dataset[idx]
        
        instruction = data.get("instruction", "")
        
        orchestrator_prompt = self._format_orchestrator_prompt(
            instruction=instruction,
        )
        
        system_msg = "You are a helpful orchestrator that coordinates agents to solve complex tasks. You break down problems into subtasks and assign role instructions to each agent."
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": orchestrator_prompt},
            {"role": "assistant", "content": "Here is your solution."}
        ]
        
        if self.apply_chat_template:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = orchestrator_prompt
        
        tokenized_prompt = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_prompt_length,
            padding=False,
            return_tensors=None,
        )
        
        output = {
            "input_ids": tokenized_prompt["input_ids"],
            "attention_mask": tokenized_prompt["attention_mask"],
            "prompt": prompt_text,
            "task_data": {
                "task_id": data.get("task_id", ""),
                "instruction": instruction,
                "test": data.get("test", ""),
                "entry_point": data.get("entry_point", ""),
                "canonical_solution": data.get("canonical_solution", ""),
            }
        }
        
        return output
