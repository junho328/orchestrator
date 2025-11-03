import json
import os
import yaml
from typing import List, Dict, Any
from abc import ABC, abstractmethod
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from model import DecoderBase, make_model


class MultiAgentDecoder(DecoderBase):
    """
    Multi-agent decoder that uses chain-of-agents workflow for code generation.
    """
    
    def __init__(
        self,
        name: str,
        config_path: str,
        backend: str = "hf",
        batch_size: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 1280,
        tp: int = 1,
        **kwargs
    ):
        super().__init__(name, batch_size, temperature, max_new_tokens, **kwargs)
        
        self.config_path = config_path
        self.backend = backend
        self.tp = tp
        
        # Load multi-agent configuration
        self.load_config()
        
        # Initialize shared model decoder (more efficient than separate models)
        self.shared_decoder = None
        self.agent_roles = []
        self.agent_templates = []
        
        # Store kwargs for model initialization
        self.kwargs = kwargs
        
        self._init_shared_model()
        self._load_agent_configs()
    
    def load_config(self):
        """Load multi-agent configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.workflow_args = self.config.get('workflow_args', {})
        self.agents_config = self.config.get('agents', [])
        
        print(f"Loaded multi-agent config: {self.config_path}")
        print(f"Workflow: {self.config.get('agent_workflow', 'chain-of-agents')}")
        print(f"Number of rounds: {self.workflow_args.get('num_rounds', 2)}")
    
    def _init_shared_model(self):
        """Initialize shared model decoder for all agents."""
        print(f"Initializing shared model: {self.name}")
        self.shared_decoder = make_model(
            model=self.name,
            backend=self.backend,
            batch_size=self.batch_size,
            temperature=self.temperature,
            tp=self.tp,
            trust_remote_code=self.kwargs.get('trust_remote_code', False),
            tokenizer_name=self.kwargs.get('tokenizer_name', None),
            tokenizer_legacy=self.kwargs.get('tokenizer_legacy', False),
        )
        print("Shared model initialized successfully")
    
    def _load_agent_configs(self):
        """Load agent configurations from YAML."""
        for agent_config in self.agents_config:
            agent_id = list(agent_config.keys())[0]
            agent_info = agent_config[agent_id]
            
            role = agent_info.get('role', agent_id)
            chat_template = agent_info.get('chat_template', [])
            
            self.agent_roles.append(role)
            self.agent_templates.append(chat_template)
            
            print(f"Loaded agent config: {agent_id} ({role})")
            if chat_template:
                print(f"  Template preview: {chat_template[0][:100]}...")
    
    def is_direct_completion(self) -> bool:
        return False
    
    def codegen(self, prompts: List[str], do_sample: bool = True, num_samples: int = 200) -> List[str]:
        """
        Generate code using multi-agent chain workflow.
        """
        results = []
        
        for prompt in prompts:
            # Run multi-agent chain
            final_output = self._run_agent_chain(prompt, do_sample, num_samples)
            results.append(final_output)
        
        return results
    
    def _run_agent_chain(self, query: str, do_sample: bool, num_samples: int) -> str:
        """
        Run the agent chain: planner -> implementer
        """
        num_rounds = self.workflow_args.get('num_rounds', 2)
        
        # First agent: Code Planner
        if len(self.agent_templates) >= 1 and self.agent_templates[0]:
            planner_template = self.agent_templates[0][0]
            planner_prompt = planner_template.format(query=query)
            
            print(f"Running {self.agent_roles[0]}...")
            # print(f"Planner prompt: {planner_prompt[:200]}...")
            
            planner_outputs = self.shared_decoder.codegen(
                [planner_prompt], do_sample=do_sample, num_samples=1
            )
            planner_result = planner_outputs[0]
            
            # print(f"Planner output: {planner_result[:200]}...")
        else:
            planner_result = ""
        
        # Second agent: Code Implementer
        if len(self.agent_templates) >= 2 and self.agent_templates[1]:
            implementer_template = self.agent_templates[1][0]
            implementer_prompt = implementer_template.format(
                query=query, 
                solution=planner_result
            )
            
            print(f"Running {self.agent_roles[1]}...")
            # print(f"Implementer prompt: {implementer_prompt[:200]}...")
            
            implementer_outputs = self.shared_decoder.codegen(
                [implementer_prompt], do_sample=do_sample, num_samples=1
            )
            final_result = implementer_outputs[0]
            
            # print(f"Implementer output: {final_result[:200]}...")
        else:
            final_result = planner_result
        
        return final_result
    
    def extract_code_from_output(self, output: str) -> str:
        """
        Extract Python code from the agent output.
        """
        # Look for code blocks
        if "```python" in output:
            start_idx = output.find("```python") + 9
            end_idx = output.find("```", start_idx)
            if end_idx != -1:
                return output[start_idx:end_idx].strip()
        
        # If no code block found, return the whole output
        return output.strip()


def make_multi_agent_model(
    model: str,
    config_path: str,
    backend: str = "hf",
    batch_size: int = 1,
    temperature: float = 0.8,
    max_new_tokens: int = 1280,
    tp: int = 1,
    **kwargs
):
    """
    Create a multi-agent model for code generation.
    """
    return MultiAgentDecoder(
        name=model,
        config_path=config_path,
        backend=backend,
        batch_size=batch_size,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        tp=tp,
        **kwargs
    )
