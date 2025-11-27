import torch
from typing import List, Dict, Any
from davids.train.utils.pubmdp_prompt import get_orchestrator_prompt
from .verl_utils import make_worker_prompt, extract_answer

class PubMDPRollout:
    def __init__(self, actor, tokenizer, num_agents=3, num_generations=4):
        """
        Args:
            actor: The VeRL actor module (or wrapped model) capable of generation.
            tokenizer: Tokenizer.
            num_agents: Number of turns (Orch -> Agent) per problem.
            num_generations: Number of parallel generations per problem (G in GRPO).
        """
        self.actor = actor
        self.tokenizer = tokenizer
        self.num_agents = num_agents
        self.num_generations = num_generations

    def generate_rollout(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute the multi-agent loop for a batch of problems.
        
        Args:
            batch: Dictionary containing 'problem', 'solution', etc.
                   Assume batch size B.
        
        Returns:
            List of trajectories (length B * num_generations).
        """
        original_problems = batch['problem'] # List of strings
        solutions = batch['solution']       # List of strings
        
        # Expand batch for GRPO generations
        # B -> B * G
        expanded_problems = []
        expanded_solutions = []
        for p, s in zip(original_problems, solutions):
            expanded_problems.extend([p] * self.num_generations)
            expanded_solutions.extend([s] * self.num_generations)
            
        current_batch_size = len(expanded_problems)
        
        # Initialize state for each trajectory
        # state = {'previous_outputs': [], 'remaining_agents': N}
        states = [{'previous_outputs': [], 'remaining_agents': self.num_agents} 
                  for _ in range(current_batch_size)]
        
        # Store full trajectory for training
        # Each trajectory will contain a list of (prompt, completion) pairs
        trajectories = [{'steps': [], 'problem': p, 'solution': s} 
                        for p, s in zip(expanded_problems, expanded_solutions)]

        # Main Loop
        for turn in range(self.num_agents):
            # 1. Orchestrator Step
            orch_prompts = []
            for i, state in enumerate(states):
                prompt_text = get_orchestrator_prompt(
                    original_problem=expanded_problems[i],
                    previous_outputs=state['previous_outputs'],
                    num_agents=state['remaining_agents']
                )
                # Format as chat if model expects it, or raw text
                # Assuming simple raw text or applying template here
                orch_prompts.append(prompt_text)
                
            # Generate Orchestrator outputs
            # Note: self.actor.generate should handle tokenization and generation
            orch_outputs = self.actor.generate(orch_prompts) 
            
            # Process Orchestrator outputs
            for i, output in enumerate(orch_outputs):
                # Store this step
                trajectories[i]['steps'].append({
                    'role': 'orchestrator',
                    'prompt': orch_prompts[i],
                    'completion_text': output
                })
            
            # 2. Worker Agent Step
            worker_prompts = []
            for i, state in enumerate(states):
                orch_instr = orch_outputs[i]
                prompt_text = make_worker_prompt(
                    original_problem=expanded_problems[i],
                    orchestrator_instruction=orch_instr
                )
                worker_prompts.append(prompt_text)
                
            # Generate Worker outputs
            worker_outputs = self.actor.generate(worker_prompts)
            
            # Process Worker outputs
            for i, output in enumerate(worker_outputs):
                # Update state
                agent_answer = extract_answer(output)
                states[i]['previous_outputs'].append(agent_answer)
                states[i]['remaining_agents'] -= 1
                
                # Store step
                trajectories[i]['steps'].append({
                    'role': 'agent',
                    'prompt': worker_prompts[i],
                    'completion_text': output
                })
                
        return trajectories



