"""Dataset class for orchestrator training."""
from typing import List, Dict, Optional
from transformers import PreTrainedTokenizerBase
from datasets import Dataset as HFDataset
from .bigcodebench_dataset import load_bigcodebench_dataset, format_code_generation_prompt
from .few_shot_examples import get_few_shot_examples
import logging

logger = logging.getLogger(__name__)


class OrchestratorDataset:
    """Dataset class for orchestrator training with BigCodeBench."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        worker_models: List[str],
        max_routing_steps: int = 5,
        split: str = "v0.1.4",
        max_samples: Optional[int] = None,
        seed: int = 42,
        apply_chat_template: bool = True,
        max_prompt_length: int = 2048,
        tokenizer_kwargs: Optional[Dict] = None,
        num_few_shot_examples: int = 3,
        few_shot_examples: Optional[str] = None,
    ):
        """
        Initialize orchestrator dataset.
        
        Args:
            tokenizer: Tokenizer for processing
            worker_models: List of worker model names
            max_routing_steps: Maximum routing steps
            split: Dataset split
            max_samples: Max samples to load
            seed: Random seed
            apply_chat_template: Whether to apply chat template
            max_prompt_length: Maximum prompt length
            tokenizer_kwargs: Additional tokenizer kwargs
            num_few_shot_examples: Number of few-shot examples to include
            few_shot_examples: Custom few-shot examples string (if None, uses default)
        """
        self.tokenizer = tokenizer
        self.worker_models = worker_models
        self.max_routing_steps = max_routing_steps
        self.apply_chat_template = apply_chat_template
        self.max_prompt_length = max_prompt_length
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.num_few_shot_examples = num_few_shot_examples
        
        # Set few-shot examples
        if few_shot_examples is not None:
            self.few_shot_examples = few_shot_examples
        else:
            self.few_shot_examples = get_few_shot_examples(num_few_shot_examples)
        
        # Load BigCodeBench dataset
        logger.info(f"Loading BigCodeBench {split} dataset...")
        raw_data = load_bigcodebench_dataset(
            split=split,
            max_samples=max_samples,
            seed=seed
        )
        
        # Convert to HuggingFace Dataset
        self.dataset = HFDataset.from_list(raw_data)
        logger.info(f"Loaded {len(self.dataset)} examples")
    
    def __len__(self):
        return len(self.dataset)
    
    def _format_available_models(self) -> str:
        """Format available models list."""
        formatted = ""
        for i, model_name in enumerate(self.worker_models):
            formatted += f"  {i}: {model_name}\n"
        return formatted
    
    def _format_orchestrator_prompt(
        self,
        instruction: str,
        test: str = "",
    ) -> str:
        """Format prompt for orchestrator using the new format."""
        prompt = """Your role as an assistant involves obtaining answers to questions by an iterative process of querying powerful language models, each with a different skillset.

You are given a user-provided question and a list of available numbered language models with their metadata. Your objective is to output a sequence of up to 5 workflow steps.

Each routing is made of three elements: A language model, its assigned subtask to accomplish, and an "access list" of past workflow steps it will see in its context when trying to accomplish the subtask.

A subtask could directly ask the language model to solve the given question from scratch, refine the solution of the previous subtask in the sequence, or perform any other completely different task that would facilitate later language models in the sequence to answer the original question with their expertise.

Based on your answer, the first model selected will be prompted with the user question and the first subtask you define. Each following model in the sequence will be prompted with the history of the previous subtask and response messages specified in its access list, and will be asked to accomplish its relative subtask. The answer of the final model and subtask will be provided back as the final solution to the user.

Your response should be provided as three Python lists.
The first list should be called model_id, and contain the integers corresponding to the numbered language models in the sequence you want to prompt.

The second list should be called subtasks, and contain the strings that will be used to prompt the corresponding language model specified in model_id.

The third list should be called access_list, and contain the lists of past routing messages (subtasks and assistant responses) from the previous routing steps to include in the context in the current routing step.
You can pass the string "all" for any of the routing steps in access_list to provide all the previous routing messages in the language model's context. Alternatively, if you want an agent to attempt its subtask without any access to previous routing steps, you can pass an empty list.

For instance:

{few_shot_examples}

USER QUESTION:

{user_question}

AVAILABLE LANGUAGE MODELS:

{available_models}

MAXIMUM NUMBER OF ROUTING STEPS: {max_routing_steps}

Please provide your response as three Python lists: model_id, subtasks, and access_list.
"""
        
        # Format available models
        available_models_str = self._format_available_models()
        
        # Format user question (combine instruction and test if available)
        user_question = f"Task: {instruction}\n\n"

        user_question += "Please provide a self-contained Python script that solves the given task."
        
        # Replace placeholders
        prompt = prompt.format(
            few_shot_examples=self.few_shot_examples,
            user_question=user_question,
            available_models=available_models_str,
            max_routing_steps=self.max_routing_steps,
        )
        
        return prompt
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single example."""
        data = self.dataset[idx]
        
        base_question = data.get("instruction", "")
        test_code = data.get("test", "")
        
        # Format orchestrator prompt
        orchestrator_prompt = self._format_orchestrator_prompt(
            instruction=base_question,
            test=test_code,
        )
        
        # Create messages
        system_msg = "You are a helpful orchestrator that coordinates worker agents to solve complex tasks. You break down problems into subtasks, select appropriate agents, and specify how agents can access previous work."
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": orchestrator_prompt},
            {"role": "assistant", "content": "Here is your solution."}
        ]
        
        # Apply chat template if requested
        if self.apply_chat_template:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **self.tokenizer_kwargs
            )
        else:
            prompt_text = orchestrator_prompt
        
        # Tokenize
        encoded = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_prompt_length,
            padding=False,
            return_tensors=None,
        )
        
        # Prepare return dict
        example = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "prompt": prompt_text,
            "base_question": base_question,
            "task_data": {
                "instruction": data.get("instruction", ""),
                "test": data.get("test", ""),
                "entry_point": data.get("entry_point", ""),
                "canonical_solution": data.get("canonical_solution", ""),
                "task_id": data.get("task_id", ""),
            }
        }
        
        return example

