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
    bcb_dataset = load_dataset("bigcode/bigcodebench", split=split)
    logger.info(f"Loaded BigCodeBench {split} dataset with {len(bcb_dataset)} examples")
    
    return bcb_dataset

class OrchestratorDataset:
    
    def __init__(
        self,
        tokenizer : PreTrainedTokenizerBase,
        dataset_name : str,
        split : str,
        worker_models : List[str],
        max_routing_steps : int = 4,
        seed : int = 42,
        apply_chat_template : bool = True,
        max_prompt_length : int = 1024,
        num_few_shot_examples : int = 2
    ):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.split = split
        self.worker_models = worker_models
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
    
    def _format_available_models(self):
        
        formatted_available_models = ""
        
        for i, model_name in enumerate(self.worker_models):
            formatted_available_models += f"{i}: {model_name}\n"
        
        return formatted_available_models
    
    def _format_orchestrator_prompt(self, 
                                    instruction: str,
                                    ):
        
        prompt = f"""Your role as an assistant involves obtaining answers to questions by an iterative process of querying powerful language models, each with a different skillset.

You are given a user-provided question and a list of available numbered language models and their metadata. Your objective is to output a sequence of up to {self.max_routing_steps} workflow steps.

Each routing is made of two elements: A language model and its assigned subtask to accomplish.

A subtask could directly ask the language model to solve the given question from scratch, refine the solution of the previous subtask in the sequence, or perform any other completely different task that would facilitate later language models in the sequence to answer the original question with their expertise.

Based on your answer, the first model selected will be prompted with the user question and the first subtask you define. Each following model in the sequence will be prompted with the full history of the previous subtasks and responses, and will be asked to accomplish its relative subtask. The answer of the final model and subtask will be provided back as the final solution to the user.

Your response should be provided as two Python lists.

The first list should be called model_id, and contain the integers corresponding to the numbered language models in the sequence you want to prompt.

The second list should be called subtasks, and contain the strings that will be used to prompt the corresponding language model specified in model_id.

For instance:

{self.few_shot_examples}

USER QUESTION:

{instruction}

AVAILABLE LANGUAGE MODELS:

{formatted_available_models}

MAXIMUM NUMBER OF ROUTING STEPS: {self.max_routing_steps}

Please provide your response as three Python lists: model_id, subtasks, and access_list.
""" 

        formatted_available_models = self._format_available_models()
        
        user_question = f"Task: {instruction}\n\n"
        
        prompt = prompt.format(
            instruction=instruction,
            formatted_available_models=formatted_available_models,
        )
        
        return prompt
    
    def __getitem__(self, idx: int):
        
        data = self.dataset[idx]
        
        instruction = data.get("instruction", "")
        
        orchestrator_prompt = self._format_orchestrator_prompt(
            instruction=instruction,
        )
        
        system_msg = "You are a helpful orchestrator that coordinates worker agents to solve complex tasks. You break down problems into subtasks, select appropriate agents, and specify how agents can access previous work."
        
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
        
        tokenized_prompt = self.tokenizer(
            orchestrator_prompt,
            truncation=True,
            max_length=self.max_prompt_length,
            padding=False,
            return_tensors=None,
        )
        
        output = {
            "input_ids": tokenized_prompt["input_ids"],
            "attention_mask": tokenized_prompt["attention_mask"],
            "prompt": orchestrator_prompt,
            "task_data": {
                "task_id": data.get("task_id", ""),
                "instruction": instruction,
                "test": data.get("test", ""),
                "entry_point": data.get("entry_point", ""),
                "canonical_solution": data.get("canonical_solution", ""),
            }
        }
        
        return output

        

