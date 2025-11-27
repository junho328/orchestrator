import torch
import re
from typing import List, Dict, Any

class PubMDPRewardManager:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def compute_rewards(self, trajectories: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Compute rewards for a batch of trajectories.
        
        Args:
            trajectories: List of dictionaries, each containing:
                - 'solution': Ground truth answer
                - 'steps': List of steps, where each step has 'role' ('orchestrator' or 'agent') and 'completion_text'
        
        Returns:
            Tensor of rewards matching the structure required by the trainer.
            Usually (batch_size, ) or (batch_size, num_steps) depending on algorithm.
            For GRPO, typically we return a reward for the whole trajectory or per step.
        """
        rewards = []
        
        for traj in trajectories:
            solution = traj['solution']
            steps = traj['steps']
            
            # 1. Accuracy Reward (Check final agent answer)
            # Find the last agent answer
            last_agent_answer = ""
            for step in reversed(steps):
                if step['role'] == 'agent':
                    # Extract <answer> content
                    content = step['completion_text']
                    match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL | re.IGNORECASE)
                    if match:
                        last_agent_answer = match.group(1).strip()
                    else:
                        last_agent_answer = content.strip()
                    break
            
            acc_reward = self.check_accuracy(last_agent_answer, solution)
            
            # 2. Format Reward (Check all agent steps)
            fmt_reward = 1.0 # Default to 1.0, penalize if any agent fails
            for step in steps:
                if step['role'] == 'agent':
                    content = step['completion_text']
                    if not (("<think>" in content) and ("</think>" in content) and 
                            ("<answer>" in content) and ("</answer>" in content)):
                        fmt_reward = 0.0
                        break
            
            # Total reward for this trajectory
            total_reward = acc_reward + fmt_reward
            rewards.append(total_reward)
            
        return torch.tensor(rewards)

    def check_accuracy(self, answer: str, solution: str) -> float:
        # Simple exact match or substring match for math problems
        # Using a simplified version of the logic found in davids.reward_utils.math_reward
        # Ideally import the exact function
        if solution.strip() in answer.strip():
            return 1.0
        return 0.0

# Helper to match standard VeRL interface if needed
def compute_score(data_batch, tokenizer):
    # data_batch usually contains 'prompts', 'completions', 'ground_truth'
    # But for multi-turn, the data structure is more complex.
    # We assume the rollout worker has pre-computed/structured the data.
    pass



