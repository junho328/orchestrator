import os
import json
import argparse
import yaml
from typing import List, Dict, Any

from multi_agent_decoder import make_multi_agent_model
from data import get_bigcodebench, write_jsonl
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)


def multi_agent_codegen(
    model,
    save_path: str,
    split: str,
    subset: str = "full",
    greedy: bool = False,
    strip_newlines: bool = False,
    n_samples: int = 1,
    id_range=None,
    resume: bool = True,
):
    """
    Generate code using multi-agent chain workflow.
    """
    dataset = get_bigcodebench(subset=subset)
    
    # Create save_path if it doesn't exist
    dirname = os.path.dirname(save_path)
    if not os.path.exists(dirname) and dirname != "":
        os.makedirs(dirname)
    
    task_ids, prompts, complete_prompts = [], [], []
    for task_id, task in dataset.items():
        task_ids.append(task_id)
        complete_prompts.append(task["complete_prompt"])
        prompt = task[f"{split}_prompt"]
        prompt = prompt.strip("\n") if strip_newlines else prompt
        prompts.append(prompt)
    
    print(f"Processing {len(prompts)} prompts with multi-agent workflow...")
    
    # Generate outputs using multi-agent chain
    outputs = model.codegen(prompts, do_sample=not greedy, num_samples=n_samples)
    assert outputs, "No outputs from model!"
    
    # Extract code from outputs
    samples = []
    for task_id, complete_prompt, completion in zip(task_ids, complete_prompts, outputs):
        # Extract Python code from the multi-agent output
        extracted_code = model.extract_code_from_output(completion)
        
        # For instruct tasks, we use the extracted code directly
        samples.append(dict(task_id=task_id, solution=extracted_code))
    
    print(f"Generated {len(samples)} samples")
    write_jsonl(save_path, samples)


def main():
    parser = argparse.ArgumentParser(description="Multi-agent code generation for BigCodeBench")
    parser.add_argument("--model", required=True, type=str, help="Model name or path")
    parser.add_argument("--config", required=True, type=str, help="Path to multi-agent config YAML file")
    parser.add_argument("--split", required=True, type=str, choices=["complete", "instruct"])
    parser.add_argument("--subset", default="full", type=str, choices=["full", "hard"])
    parser.add_argument("--save_path", default=None, type=str, help="Path to save generated samples")
    parser.add_argument("--bs", default=1, type=int, help="Batch size")
    parser.add_argument("--n_samples", default=1, type=int, help="Number of samples per prompt")
    parser.add_argument("--temperature", default=0.8, type=float, help="Sampling temperature")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding")
    parser.add_argument("--strip_newlines", action="store_true", help="Strip newlines from prompts")
    parser.add_argument("--resume", action="store_true", help="Resume from existing file")
    parser.add_argument("--id_range", nargs=2, type=int, help="Range of task IDs to process")
    parser.add_argument("--backend", default="hf", type=str, 
                       choices=["vllm", "hf", "openai", "mistral", "anthropic", "google"],
                       help="Backend for model inference")
    parser.add_argument("--base_url", default=None, type=str, help="Base URL for API models")
    parser.add_argument("--tp", default=1, type=int, help="Tensor parallelism")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code")
    parser.add_argument("--tokenizer_legacy", action="store_true", help="Use legacy tokenizer")
    parser.add_argument("--tokenizer_name", default=None, type=str, help="Tokenizer name")
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    # Set greedy decoding parameters
    if args.greedy or (args.temperature == 0 and args.n_samples == 1):
        args.temperature = 0
        args.bs = 1
        args.n_samples = 1
        args.greedy = True
        print("Greedy decoding ON (--greedy): setting bs=1, n_samples=1, temperature=0")
    
    if args.id_range is not None:
        assert len(args.id_range) == 2, "id_range must be a list of length 2"
        assert args.id_range[0] < args.id_range[1], "id_range must be increasing"
        args.id_range = tuple(args.id_range)
    
    # Create multi-agent model
    print(f"Initializing multi-agent model with config: {args.config}")
    model_runner = make_multi_agent_model(
        model=args.model,
        config_path=args.config,
        backend=args.backend,
        batch_size=args.bs,
        temperature=args.temperature,
        max_new_tokens=1024,
        tp=args.tp,
        trust_remote_code=args.trust_remote_code,
        tokenizer_name=args.tokenizer_name,
        tokenizer_legacy=args.tokenizer_legacy,
    )
    
    # Set save path
    extra = "-" + args.subset if args.subset != "full" else ""
    if not args.save_path:
        model_name = args.model.replace("/", "--")
        args.save_path = f"{model_name}--bigcodebench{extra}-{args.split}--multi-agent-{args.backend}-{args.temperature}-{args.n_samples}.jsonl"
    
    print(f"Save path: {args.save_path}")
    
    # Generate code
    multi_agent_codegen(
        model=model_runner,
        save_path=args.save_path,
        split=args.split,
        subset=args.subset,
        greedy=args.greedy,
        strip_newlines=args.strip_newlines,
        n_samples=args.n_samples,
        resume=args.resume,
        id_range=args.id_range,
    )


if __name__ == "__main__":
    main()
