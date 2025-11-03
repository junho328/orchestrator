"""Orchestrator engine for training orchestrator models with PPO/GRPO."""
import os
import abc
import torch
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

from ..utils.model_utils import load_model_and_tokenizer, generate_with_model
from ..utils.bigcodebench_eval import sanitize_code, untrusted_check

logging.getLogger("transformers").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

import threading
_log_write_lock = threading.Lock()


def is_tensor(t):
    """Check if object is a torch tensor."""
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
    """Reward function for orchestrator training with BigCodeBench."""
    
    def __init__(
        self,
        worker_models: List[str],
        max_tokens: int = 2048,
        temperature: float = 0.8,
        max_routing_steps: int = 5,
        output_dir: Optional[str] = None,
        coordination_log_dir: Optional[str] = None,
        chunk_size: int = 8,
        score_repeats: int = 1,
        use_local_models: bool = True,
        device_map: str = "auto",
        format_bonus: float = 0.0,
        *args,
        **kwargs,
    ):
        """
        Initialize OrchestratorReward.
        
        Args:
            worker_models: List of HuggingFace model paths for worker agents
            max_tokens: Max tokens for worker generation
            temperature: Temperature for generation
            max_routing_steps: Maximum number of routing steps
            output_dir: Output directory for logs
            coordination_log_dir: Directory for coordination logs
            chunk_size: Batch size for processing
            score_repeats: Number of times to repeat scoring
            use_local_models: Whether to use local models (vs API)
            device_map: Device mapping for models
            format_bonus: Bonus for correct format
        """
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
                        device_map=device_map
                    )
                    self.worker_models_dict[model_name] = model
                    self.worker_tokenizers[model_name] = tokenizer
                    logger.info(f"Loaded worker model {i+1}/{len(worker_models)}: {model_name}")
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
        self.ACCESS_LIST_RE = re.compile(
            r"access_list\s*=\s*\[(.*?)\]",
            re.DOTALL | re.IGNORECASE
        )
        
        # Logging setup
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
        }
    
    def _parse_orchestrator_output(self, completion: str) -> Tuple[List[str], List[int], List[str]]:
        """
        Parse orchestrator output to extract:
        - subtasks: List of decomposed subtasks
        - agent_ids: List of agent IDs (indices into worker_models)
        - role_instructions: List of role instructions for each agent
        
        Returns:
            (subtasks, agent_ids, role_instructions)
        """
        subtasks = []
        agent_ids = []
        role_instructions = []
        
        # Try Python list format first
        model_id_match = self.MODEL_ID_LIST_RE.search(completion)
        subtasks_match = self.SUBTASKS_LIST_RE.search(completion)
        access_list_match = self.ACCESS_LIST_RE.search(completion)
        
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
        
        return subtasks, agent_ids, role_instructions
    
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
                user_content += f"Agent {h['agent_id']} response: {h['response']}\n\n"
        
        user_content += "Please solve your subtask and provide your solution."
        
        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "Here is your solution."}
        ]
    
    def _query_worker_model(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
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
            if not solution or not test_code:
                return 0.0
            
            # Sanitize the code (extract only necessary parts based on entry_point)
            sanitized_code = sanitize_code(solution, entrypoint=entry_point)
            
            if not sanitized_code:
                logger.debug("Sanitized code is empty")
                return 0.0
            
            # Run unittest-based evaluation
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
                
                # Return 1.0 if all tests pass, 0.0 otherwise
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
        
        try:
            # Parse orchestrator output
            subtasks, agent_ids, role_instructions = self._parse_orchestrator_output(completion)
            
            if not subtasks or not agent_ids:
                raise ValueError("Failed to parse orchestrator output")
            
            if len(subtasks) != len(agent_ids):
                raise ValueError("Mismatched subtasks and agent_ids")
            
            # Check for valid agent IDs
            if any(aid < 0 or aid >= len(self.worker_models) for aid in agent_ids):
                raise ValueError("Invalid agent IDs")
            
            counters_log["total_access_cnt"] = len(agent_ids)
            counters_log["avg_subtask_length"] = sum(len(s) for s in subtasks) / len(subtasks) if subtasks else 0
            
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
            
            # Final response is the last worker response
            if history:
                final_response = history[-1]["response"]
            
            # Evaluate final solution
            if final_response and test_code:
                correctness_reward = self._evaluate_code_solution(
                    final_response,
                    test_code,
                    entry_point,
                )
            else:
                correctness_reward = 0.0
            
            reward = correctness_reward
            
            if correctness_reward > 0:
                counters_log["total_correct_cnt"] = 1
            
            if self.format_bonus > 0:
                reward += self.format_bonus
            
        except Exception as e:
            logger.error(f"Error in coordination: {e}")
            parsing_error = str(e)
            per_step_log["parsing_error"] = parsing_error
            counters_log["format_error_cnt"] = 1
            reward = 0.0
        
        per_step_log["reward"] = reward
        per_step_log["execution_time"] = time.time() - start_time
        
        return reward, counters_log, per_step_log
    
    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs,
    ) -> List[float]:
        """
        Compute rewards for a batch of prompts and completions.
        
        Args:
            prompts: List of input prompts
            completions: List of orchestrator completions
            **kwargs: Additional arguments (base_question, task_data, etc.)
            
        Returns:
            List of reward values
        """
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


class CustomGRPOTrainer(GRPOTrainer):
    """Custom GRPO trainer with orchestrator-specific logging."""
    
    def __init__(self, *args, logging_prob=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.logging_prob = logging_prob
    
    def evaluate(self, *args, **kwargs):
        """Override evaluate to add orchestrator-specific metrics."""
        orig_ngen = self.args.num_generations
        try:
            self.args.num_generations = 1
            self.num_generations = 1
            results = super().evaluate(*args, **kwargs)
        finally:
            self.args.num_generations = orig_ngen
        
        # Log orchestrator statistics
        if self.reward_funcs:
            rw = self.reward_funcs[0]
            if hasattr(rw, "full_log"):
                print("\n\n--- Orchestrator Statistics: ---\n\n")
                log = rw.full_log
                if log["total_question_cnt"] > 0:
                    acc = log["total_correct_cnt"] / log["total_question_cnt"]
                    print(f"Accuracy: {log['total_correct_cnt']} / {log['total_question_cnt']} -> {acc * 100:.2f}%")
                    print(f"Format errors: {log['format_error_cnt']}")
                    print(f"Avg subtask length: {log['avg_subtask_length']:.2f}")
        
        return results


class CustomPPOTrainer(PPOTrainer):
    """Custom PPO trainer with orchestrator-specific logging."""
    
    def __init__(self, *args, logging_prob=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.logging_prob = logging_prob
    
    def evaluate(self, *args, **kwargs):
        """Override evaluate to add orchestrator-specific metrics."""
        results = super().evaluate(*args, **kwargs)
        
        # Log orchestrator statistics
        if self.reward_funcs:
            rw = self.reward_funcs[0]
            if hasattr(rw, "full_log"):
                print("\n\n--- Orchestrator Statistics: ---\n\n")
                log = rw.full_log
                if log["total_question_cnt"] > 0:
                    acc = log["total_correct_cnt"] / log["total_question_cnt"]
                    print(f"Accuracy: {log['total_correct_cnt']} / {log['total_question_cnt']} -> {acc * 100:.2f}%")
        
        return results

