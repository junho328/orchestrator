import re
import logging
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

from .bcb_eval.sanitize_unittest import sanitize_code, untrusted_check

logging.getLogger("transformers").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class CustomReward(ABC):
    """Abstract base class for reward functions."""

    @abstractmethod
    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs,
    ) -> List[float]:
        raise NotImplementedError


class OrchestratorAgentReward(CustomReward):
    """
    Simple reward function that evaluates:
    1. Format reward: Whether output can be properly parsed (orchestrator format + agent format)
    2. Accuracy reward: Whether final code passes all unit tests
    
    Reward structure:
    - Format reward (orchestrator): 0.25 if Python list format is correct
    - Format reward (agent): 0.25 if tags are properly formatted
    - Accuracy reward: 1.0 if all unit tests pass
    """
    
    def __init__(
        self,
        max_routing_steps: int = 4,
        format_reward_orchestrator: float = 0.5,
        format_reward_agent: float = 0.5,
        *args,
        **kwargs,
    ):
        self.max_routing_steps = max_routing_steps
        self.format_reward_orchestrator = format_reward_orchestrator
        self.format_reward_agent = format_reward_agent
        
        # Regex patterns for parsing orchestrator output
        self.SUBTASKS_LIST_RE = re.compile(
            r"subtasks\s*=\s*\[(.*?)\]",
            re.DOTALL | re.IGNORECASE
        )
        self.ROLE_INSTRUCTIONS_LIST_RE = re.compile(
            r"role_instructions\s*=\s*\[(.*?)\]",
            re.DOTALL | re.IGNORECASE
        )
        
        # Regex for extracting agent responses
        self.REASONING_RE = re.compile(
            r"<start_think>(.*?)<end_think>",
            re.DOTALL | re.IGNORECASE
        )
        self.ANSWER_RE = re.compile(
            r"<start_answer>(.*?)<end_answer>",
            re.DOTALL | re.IGNORECASE
        )
        
        self.__name__ = "OrchestratorAgentReward"
    
    def _parse_orchestrator_output(self, completion: str) -> tuple[List[str], List[str], bool]:
        """
        Parse orchestrator output to extract subtasks and role_instructions.
        
        Returns:
            (subtasks, role_instructions, is_valid_format)
        """
        subtasks = []
        role_instructions = []
        is_valid_format = False
        
        # Try to parse Python list format
        subtasks_match = self.SUBTASKS_LIST_RE.search(completion)
        role_instructions_match = self.ROLE_INSTRUCTIONS_LIST_RE.search(completion)
        
        if subtasks_match and role_instructions_match:
            # Parse subtasks list
            subtasks_str = subtasks_match.group(1)
            subtask_matches = re.findall(r'"(.*?)"', subtasks_str, re.DOTALL)
            subtasks = [m.replace('\\n', '\n').replace('\\"', '"').strip() for m in subtask_matches]
            
            # Parse role_instructions list
            role_instructions_str = role_instructions_match.group(1)
            role_matches = re.findall(r'"(.*?)"', role_instructions_str, re.DOTALL)
            role_instructions = [m.replace('\\n', '\n').replace('\\"', '"').strip() for m in role_matches]
            
            # Check if format is valid
            if subtasks and role_instructions and len(subtasks) == len(role_instructions):
                if len(subtasks) <= self.max_routing_steps:
                    is_valid_format = True
        
        return subtasks, role_instructions, is_valid_format
    
    def _extract_answer(self, text: str) -> str:
        """Extract content between <start_answer> and <end_answer> tags."""
        match = self.ANSWER_RE.search(text)
        if match:
            return match.group(1).strip()
        return ""
    
    def _check_agent_format(self, text: str) -> bool:
        """Check if agent response has proper format with both tags."""
        has_reasoning = bool(self.REASONING_RE.search(text))
        has_answer = bool(self.ANSWER_RE.search(text))
        return has_reasoning and has_answer
    
    def _evaluate_code_solution(
        self,
        solution: str,
        test_code: str,
        entry_point: str,
    ) -> float:
        """
        Evaluate a code solution using BigCodeBench's unittest-based evaluation.
        Returns reward: 1.0 if all tests pass, 0.0 otherwise.
        """
        try:
            if not test_code or not solution:
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
                    logger.debug(f"Test failed with status: {status}")
                    return 0.0
                
            except Exception as e:
                logger.debug(f"Error during unittest evaluation: {e}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error evaluating solution: {e}")
            return 0.0
    
    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs,
    ) -> List[float]:
        """
        Compute rewards for a batch of prompts and completions.
        
        Args:
            prompts: List of prompt strings
            completions: List of completion strings (orchestrator outputs)
            **kwargs: Additional arguments, including:
                - task_data: List of dicts with 'test', 'entry_point' keys
                - agent_responses: List of agent response strings (if available)
        
        Returns:
            List of reward values
        """
        batch_rewards = []
        num_items = len(completions)
        
        # Extract task data
        task_data_list = kwargs.get("task_data", [])
        agent_responses_list = kwargs.get("agent_responses", [])
        
        for idx in range(num_items):
            completion = completions[idx]
            reward = 0.0
            
            try:
                # 1. Parse orchestrator output and check format
                subtasks, role_instructions, is_valid_format = self._parse_orchestrator_output(completion)
                
                format_reward_orchestrator = 0.0
                if is_valid_format:
                    format_reward_orchestrator = self.format_reward_orchestrator
                else:
                    # Format is invalid, give 0 reward
                    batch_rewards.append(0.0)
                    continue
                
                # 2. Check agent format (if agent responses are provided)
                format_reward_agent = 0.0
                if agent_responses_list and idx < len(agent_responses_list):
                    agent_response = agent_responses_list[idx]
                    if self._check_agent_format(agent_response):
                        format_reward_agent = self.format_reward_agent
                
                # 3. Extract final code and evaluate with unittest
                accuracy_reward = 0.0
                if task_data_list and idx < len(task_data_list):
                    task_data = task_data_list[idx]
                    test_code = task_data.get("test", "")
                    entry_point = task_data.get("entry_point", "")
                    
                    # Extract final code from agent response or completion
                    final_code = ""
                    if agent_responses_list and idx < len(agent_responses_list):
                        final_code = self._extract_answer(agent_responses_list[idx])
                    else:
                        # Fallback: try to extract from completion
                        final_code = self._extract_answer(completion)
                    
                    if final_code and test_code:
                        accuracy_reward = self._evaluate_code_solution(
                            final_code,
                            test_code,
                            entry_point,
                        )
                
                # Total reward = format rewards + accuracy reward
                reward = format_reward_orchestrator + format_reward_agent + accuracy_reward
                
            except Exception as e:
                logger.error(f"Error computing reward for item {idx}: {e}")
                reward = 0.0
            
            batch_rewards.append(reward)
        
        return batch_rewards
