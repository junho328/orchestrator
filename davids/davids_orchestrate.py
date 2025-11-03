import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
import abc
import accelerate
import time
from torch import nn
import torch.nn.functional as F
import json
import numpy as np
from datetime import datetime
import random
import re
from typing import Any, Callable, Optional, Union, Sequence, List, Dict, Tuple
from itertools import islice
from transformers import PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, PPOTrainer, GRPOConfig, PPOConfig
from accelerate.utils import gather
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import logging
import subprocess
import sys

from ..bcb_eval.sanitize_unittest import sanitize_code, untrusted_check

logging.getLogger("transformers").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available, falling back to transformers")

def load_model_and_tokenizer(
    model_name_or_path: str, 
    device_map: str = "auto", 
    use_vllm: bool = True,
    vllm_kwargs: Optional[Dict] = None,
    **kwargs
):
    """
    Load a model and tokenizer. Can use vLLM for faster inference.
    
    Args:
        model_name_or_path: Path to model
        device_map: Device mapping for transformers
        use_vllm: Whether to use vLLM (faster)
        vllm_kwargs: Additional kwargs for vLLM LLM initialization
        **kwargs: Additional kwargs for transformers
        
    Returns:
        model (vLLM LLM or transformers model), tokenizer
    """
    logger.info(f"Loading model: {model_name_or_path}")
    
    if use_vllm and VLLM_AVAILABLE:
        # Use vLLM for faster inference
        vllm_kwargs = vllm_kwargs or {}
        model = LLM(
            model=model_name_or_path,
            trust_remote_code=True,
            dtype="float16" if torch.cuda.is_available() else "float32",
            **vllm_kwargs
        )
        # vLLM has built-in tokenizer
        tokenizer = None
        logger.info("Loaded model with vLLM")
        return model, tokenizer
    else:
        # Fall back to transformers
        if not use_vllm:
            logger.info("Using transformers (vLLM disabled)")
        else:
            logger.warning("vLLM not available, using transformers")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            **kwargs
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            **kwargs
        )
        
        return model, tokenizer

def generate_with_model(
    model,
    tokenizer: Optional[AutoTokenizer],
    messages: List[Dict[str, str]],
    max_tokens: int = 1024,
    temperature: float = 0.8,
    **kwargs
) -> str:
    """
    Generate text using either vLLM or transformers model.
    
    Args:
        model: vLLM LLM instance or transformers model
        tokenizer: Tokenizer (None if using vLLM, as it has built-in tokenizer)
        messages: Chat messages in format [{"role": "...", "content": "..."}]
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        **kwargs: Additional generation parameters
        
    Returns:
        Generated text
    """
    # Check if using vLLM (vLLM LLM instances have 'llm_engine' attribute)
    if VLLM_AVAILABLE and hasattr(model, 'llm_engine'):
        # vLLM LLM instance
        from vllm import SamplingParams
        
        # Convert messages to prompt string
        if isinstance(messages, list) and len(messages) > 0:
            if "content" in messages[0] and "role" in messages[0]:
                # Use vLLM's tokenizer to apply chat template
                prompt = model.get_tokenizer().apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                prompt = str(messages)
        else:
            prompt = str(messages)
        
        # Create sampling params
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Generate with vLLM
        outputs = model.generate([prompt], sampling_params)
        
        # Extract generated text
        generated_text = outputs[0].outputs[0].text
        
        return generated_text.strip()
    
    else:
        # Fall back to transformers
        # Apply chat template if messages format
        if isinstance(messages, list) and len(messages) > 0:
            if "content" in messages[0] and "role" in messages[0]:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                text = messages
        else:
            text = str(messages)
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()

def is_tensor(t):
    return isinstance(t, torch.Tensor)

class CustomReward(abc.ABC):
    """Abstract base class for reward functions."""
    
    def link_with_trainer(self, trainer, tokenizer, numeric_answer=False):
        """Link reward function with trainer."""
        self.__name__ = self.__class__.__name__
        self._numeric_answer = numeric_answer
        self.trainer = trainer

    @abc.abstractmethod
    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs,
    ):
        raise NotImplementedError
    
class OrchestratorReward(CustomReward):
    
    def __init__(
        self,
        worker_models: List[str],
        max_tokens: int = 1024,
        temperature: float = 0.8,
        max_routing_steps: int = 4,
        output_dir: Optional[str] = None,
        coordination_log_dir: Optional[str] = None,
        chunk_size: int = 16,
        score_repeats: int = 1,
        use_local_models: bool = True,
        device_map: str = "auto",
        format_bonus: float = 0.5,
        vllm_kwargs: Optional[Dict] = None,
        *args,
        **kwargs,
    ):
        self.worker_models = worker_models
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_routing_steps = max_routing_steps
        self.output_dir = output_dir
        self.coordination_log_dir = coordination_log_dir
        self.chunk_size = chunk_size
        self.score_repeats = score_repeats
        self.use_local_models = use_local_models
        self.device_map = device_map
        self.format_bonus = format_bonus

        # Load worker models
        self.worker_models_dict = {}
        self.worker_tokenizers = {}

        if use_local_models:
            logger.info(f"Loading {len(worker_models)} worker models...")
            for i, model_name in enumerate(worker_models):
                try:
                    model, tokenizer = load_model_and_tokenizer(
                        model_name,
                        device_map=device_map,
                        use_vllm=True,
                        vllm_kwargs=vllm_kwargs
                    )
                    self.worker_models_dict[model_name] = model
                    self.worker_tokenizers[model_name] = tokenizer
                    logger.info(f"Loaded worker model {i}: {model_name}")
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {e}")
                    raise
                
        # Parsing regex patterns
        self.SUBTASK_RE = re.compile(
            r"<subtask[^>]*>(.*?)</subtask>",
            re.DOTALL | re.IGNORECASE
        )
        self.AGENT_ID_RE = re.compile(
            r"<agent[^>]*>(\d+)</agent>",
            re.IGNORECASE
        )
        self.ROLE_INSTRUCTION_RE = re.compile(
            r"<role_instruction[^>]*>(.*?)</role_instruction>",
            re.DOTALL | re.IGNORECASE
        )
        
        # Also try to parse Python list format
        self.MODEL_ID_LIST_RE = re.compile(
            r"model_id\s*=\s*\[(.*?)\]",
            re.DOTALL | re.IGNORECASE
        )
        self.SUBTASKS_LIST_RE = re.compile(
            r"subtasks\s*=\s*\[(.*?)\]",
            re.DOTALL | re.IGNORECASE
        )
        
        # Regex for extracting answer from worker responses
        self.ANSWER_RE = re.compile(
            r"<answer>(.*?)</answer>",
            re.DOTALL | re.IGNORECASE
        )
        
        if coordination_log_dir:
            os.makedirs(coordination_log_dir, exist_ok=True)
            self.debug_log_dir = os.path.join(coordination_log_dir, "transcript_debug")
            os.makedirs(self.debug_log_dir, exist_ok=True)
        
        self.__name__ = "OrchestratorReward"
        self.full_log = self._create_full_log()
        
    def _create_full_log(self):
        """Create empty log structure."""
        return {
            "total_question_cnt": 0,
            "total_correct_cnt": 0,
            "format_error_cnt": 0,
            "total_access_cnt": 0,
            "avg_subtask_length": 0,
            "total_format_correct_cnt": 0,
        }
        
    def _parse_orchestrator_output(self, completion: str) -> Tuple[List[str], List[int], List[str], bool]:
        """
        Parse orchestrator output to extract:
        - subtasks: List of decomposed subtasks
        - agent_ids: List of agent IDs (indices into worker_models)
        - role_instructions: List of role instructions for each agent
        - is_python_list_format: Whether parsing succeeded with Python list format
        
        Returns:
            (subtasks, agent_ids, role_instructions, is_python_list_format)
        """
        subtasks = []
        agent_ids = []
        role_instructions = []
        is_python_list_format = False
        
        # Try Python list format first
        model_id_match = self.MODEL_ID_LIST_RE.search(completion)
        subtasks_match = self.SUBTASKS_LIST_RE.search(completion)
        
        if model_id_match and subtasks_match:
            # Parse model_id list
            model_id_str = model_id_match.group(1)
            try:
                agent_ids = [int(x.strip()) for x in model_id_str.split(',')]
            except:
                pass
            
            # Parse subtasks list
            subtasks_str = subtasks_match.group(1)
            # Extract strings from list
            subtask_matches = re.findall(r'"(.*?)"', subtasks_str, re.DOTALL)
            subtasks = [m.replace('\\n', '\n').strip() for m in subtask_matches]
            
            # Check if Python list format parsing succeeded
            if subtasks and agent_ids and len(subtasks) == len(agent_ids):
                # Validate agent IDs are valid
                if all(0 <= aid < len(self.worker_models) for aid in agent_ids):
                    is_python_list_format = True
        
        # Fallback to XML format
        if not subtasks or not agent_ids:
            subtask_matches = self.SUBTASK_RE.findall(completion)
            subtasks = [match.strip() for match in subtask_matches]
            
            agent_matches = self.AGENT_ID_RE.findall(completion)
            agent_ids = []
            for match in agent_matches:
                try:
                    agent_id = int(match)
                    if 0 <= agent_id < len(self.worker_models):
                        agent_ids.append(agent_id)
                except ValueError:
                    pass
            
            role_matches = self.ROLE_INSTRUCTION_RE.findall(completion)
            role_instructions = [match.strip() for match in role_matches]
        
        # Ensure all lists have same length (pad if necessary)
        max_len = max(len(subtasks), len(agent_ids))
        subtasks.extend([""] * (max_len - len(subtasks)))
        agent_ids.extend([-1] * (max_len - len(agent_ids)))
        role_instructions.extend([""] * (max_len - len(role_instructions)))
        
        return subtasks, agent_ids, role_instructions, is_python_list_format
    
    def _extract_answer(self, text: str) -> str:
        """
        Extract content between <answer> and </answer> tags.
        Returns the extracted content, or the original text if no answer tags found.
        """
        match = self.ANSWER_RE.search(text)
        if match:
            return match.group(1).strip()
        return text.strip()
    
    def _prepare_worker_messages(
        self,
        subtask: str,
        role_instruction: str,
        base_question: str,
        history: List[Dict] = None,
    ) -> List[Dict[str, str]]:
        """Prepare messages for worker agent."""
        if history is None:
            history = []
        
        user_content = f"Task: {base_question}\n\n"
        if role_instruction:
            user_content += f"Your Role: {role_instruction}\n\n"
        user_content += f"Your Subtask: {subtask}\n\n"
        
        if history:
            user_content += "\nPrevious agent responses:\n"
            for i, h in enumerate(history):
                # Extract answer from previous responses if available
                prev_answer = self._extract_answer(h['response'])
                user_content += f"Agent {h['agent_id']} answer: {prev_answer}\n\n"
        
        user_content += """Please solve your subtask following this format:
1. Put your reasoning process between <think> and </think> tags.
2. Put your final answer/solution between <answer> and </answer> tags.

Example format:
<think>
[Your reasoning and thought process here]
</think>
<answer>
[Your final solution here]
</answer>"""
        
        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "I'll solve this step by step."}
        ]
        
    def _query_worker_model(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> str:
        """Query a worker model with messages."""
        if not self.use_local_models:
            raise ValueError("Only local models supported")
        
        if model_name not in self.worker_models_dict:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.worker_models_dict[model_name]
        tokenizer = self.worker_tokenizers[model_name]
        
        return generate_with_model(
            model,
            tokenizer,
            messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            **kwargs
        )
    
    def _evaluate_code_solution(
        self,
        solution: str,
        test_code: str,
        entry_point: str,
    ) -> float:
        """
        Evaluate a code solution using BigCodeBench's unittest-based evaluation.
        Returns reward: 1.0 if all tests pass, 0.0 otherwise.
        
        This uses the same evaluation method as BigCodeBench:
        1. Sanitize the code to extract only necessary functions/classes
        2. Run unittest tests
        3. Return 1.0 if all tests pass, 0.0 otherwise
        """
        try:
            if not test_code:
                return 0.0
            
            sanitized_code = sanitize_code(solution, entrypoint=entry_point)
            
            if not sanitized_code:
                logger.debug("Sanitized code is empty")
                return 0.0
            
            try:
                status, details = untrusted_check(
                    code=sanitized_code,
                    test_code=test_code,
                    entry_point=entry_point,
                    max_as_limit=128 * 1024,  # 128 MB
                    max_data_limit=4 * 1024,   # 4 MB
                    max_stack_limit=5,         # 5 MB
                    min_time_limit=1.0,
                    gt_time_limit=20.0,
                )
                
                if status == "pass":
                    return 1.0
                else:
                    logger.debug(f"Test failed with status: {status}, details: {details}")
                    return 0.0
                
            except Exception as e:
                logger.debug(f"Error during unittest evaluation: {e}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error evaluating solution: {e}")
            return 0.0
        
    def _multi_turn_coordination(
        self,
        prompt: str,
        completion: str,
        idx: int,
        **kwargs,
    ) -> Tuple[float, Dict, Dict]:
        """
        Execute multi-turn coordination and compute reward.
        
        Returns:
            (reward, counters_log, per_step_log)
        """
        start_time = time.time()
        
        # Extract task data from kwargs (passed from dataset)
        if isinstance(kwargs.get("task_data"), list):
            task_data = kwargs["task_data"][idx] if idx < len(kwargs["task_data"]) else {}
        else:
            task_data = kwargs.get("task_data", {})
        
        if isinstance(kwargs.get("base_question"), list):
            base_question = kwargs["base_question"][idx] if idx < len(kwargs["base_question"]) else ""
        else:
            base_question = kwargs.get("base_question", "")
        
        test_code = task_data.get("test", "") if isinstance(task_data, dict) else ""
        entry_point = task_data.get("entry_point", "") if isinstance(task_data, dict) else ""
        
        counters_log = {
            "total_question_cnt": 1,
            "total_correct_cnt": 0,
            "format_error_cnt": 0,
            "total_access_cnt": 0,
            "avg_subtask_length": 0,
            "total_format_correct_cnt": 0,
        }
        
        per_step_log = {
            "subtasks": [],
            "agent_ids": [],
            "role_instructions": [],
            "worker_responses": [],
            "parsing_error": None,
        }
        
        reward = 0.0
        parsing_error = None
        accuracy_reward = 0.0
        format_reward = 0.0
        
        try:
            subtasks, agent_ids, role_instructions, is_python_list_format = self._parse_orchestrator_output(completion)
            
            if not subtasks or not agent_ids:
                raise ValueError("Failed to parse orchestrator output")
            
            if len(subtasks) != len(agent_ids):
                raise ValueError("Mismatched subtasks and agent_ids")
            
            # Check for valid agent IDs
            if any(aid < 0 or aid >= len(self.worker_models) for aid in agent_ids):
                raise ValueError("Invalid agent IDs")
            
            counters_log["total_access_cnt"] = len(agent_ids)
            counters_log["avg_subtask_length"] = sum(len(s) for s in subtasks) / len(subtasks) if subtasks else 0
            
            # Check format reward: Python list format으로 제대로 파싱되었는지 확인
            if is_python_list_format:
                format_reward = self.format_bonus if self.format_bonus > 0 else 0.25  # Default 0.25 if format_bonus not set
                counters_log["total_format_correct_cnt"] = 1
            
            # Execute workflow
            history = []
            final_response = ""
            
            for i, (subtask, agent_id) in enumerate(zip(subtasks, agent_ids)):
                role_instruction = role_instructions[i] if i < len(role_instructions) else ""
                
                # Prepare messages for worker
                messages = self._prepare_worker_messages(
                    subtask=subtask,
                    role_instruction=role_instruction,
                    base_question=base_question,
                    history=history,
                )
                
                # Query worker model
                model_name = self.worker_models[agent_id]
                worker_response = self._query_worker_model(model_name, messages)
                
                history.append({
                    "agent_id": agent_id,
                    "subtask": subtask,
                    "response": worker_response,
                })
                
                per_step_log["subtasks"].append(subtask)
                per_step_log["agent_ids"].append(agent_id)
                per_step_log["role_instructions"].append(role_instruction)
                per_step_log["worker_responses"].append(worker_response)
            
            # Final response: extract answer from the last worker response
            # Only use content between <answer> tags for final evaluation
            if history:
                raw_response = history[-1]["response"]
                final_response = self._extract_answer(raw_response)
            else:
                final_response = ""
            
            # Evaluate final solution - accuracy reward
            if final_response and test_code:
                accuracy_reward = self._evaluate_code_solution(
                    final_response,
                    test_code,
                    entry_point,
                )
            else:
                accuracy_reward = 0.0
            
            if accuracy_reward > 0:
                counters_log["total_correct_cnt"] = 1
            
            # Total reward = accuracy reward + format reward
            reward = accuracy_reward + format_reward
                
        except Exception as e:
            logger.error(f"Error in coordination: {e}")
            parsing_error = str(e)
            per_step_log["parsing_error"] = parsing_error
            counters_log["format_error_cnt"] = 1
            reward = 0.0
            accuracy_reward = 0.0
            format_reward = 0.0
        
        per_step_log["reward"] = reward
        per_step_log["accuracy_reward"] = accuracy_reward
        per_step_log["format_reward"] = format_reward
        per_step_log["execution_time"] = time.time() - start_time
        
        return reward, counters_log, per_step_log
    
    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs,
    ) -> List[float]:
        
        batch_rewards = []
        num_items = len(completions)
        
        for chunk_start in range(0, num_items, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, num_items)
            prompts_chunk = prompts[chunk_start:chunk_end]
            completions_chunk = completions[chunk_start:chunk_end]
            indices_chunk = range(chunk_start, chunk_end)
            
            worker_fn = partial(self._multi_turn_coordination, **kwargs)
            
            with ThreadPoolExecutor(max_workers=min(8, len(indices_chunk))) as pool:
                results = list(pool.map(
                    lambda args: worker_fn(*args),
                    prompts_chunk,
                    completions_chunk,
                    indices_chunk,
                ))
                
                for reward, counters_log, per_step_log in results:
                    batch_rewards.append(reward)
                    
                    # Update full log
                    for key in counters_log:
                        if key in self.full_log:
                            self.full_log[key] += counters_log[key]
                            
        return batch_rewards
    
class OrchestratorGRPOTrainer(GRPOTrainer):
    """Custom GRPO trainer for orchestrator models with accuracy and format reward statistics."""
    
    def __init__(self, *args, logging_prob=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.logging_prob = logging_prob
    
    def evaluate(self, *args, **kwargs):
        """Override evaluate to add orchestrator-specific metrics."""
        orig_ngen = self.args.num_generations
        try:
            self.args.num_generations = 1
            self.num_generations = 1
            self.per_device_eval_batch_size = 20
            self.args.per_device_eval_batch_size = 20
            results = super().evaluate(*args, **kwargs)
        finally:
            self.args.num_generations = orig_ngen
        
        # Log orchestrator statistics
        if self.reward_funcs:
            rw = self.reward_funcs[0]
            print('\n\n--- Orchestrator Statistics: ---\n\n')
            assert rw.full_log, 'No full log found. Full log required for statistics'
            
            log = rw.full_log
            if log['total_question_cnt'] > 0:
                # Accuracy statistics
                accuracy = log['total_correct_cnt'] / log['total_question_cnt'] * 100.0
                print(f"Accuracy: {log['total_correct_cnt']} / {log['total_question_cnt']} -> {accuracy:.3f}%")
                
                # Format reward statistics
                format_acc = log['total_format_correct_cnt'] / log['total_question_cnt'] * 100.0
                print(f"Format Reward: {log['total_format_correct_cnt']} / {log['total_question_cnt']} -> {format_acc:.3f}%")
                
                # Other statistics
                print(f"Format Errors: {log['format_error_cnt']}")
                print(f"Total Access Count: {log['total_access_cnt']}")
                print(f"Avg Subtask Length: {log['avg_subtask_length']:.2f}")
        
        return results