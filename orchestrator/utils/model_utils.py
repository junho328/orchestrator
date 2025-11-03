"""Utilities for loading and using local HuggingFace models."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_name_or_path: str, device_map: str = "auto", **kwargs):
    """Load a HuggingFace model and tokenizer locally."""
    logger.info(f"Loading model: {model_name_or_path}")
    
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
    tokenizer,
    messages: List[Dict[str, str]],
    max_tokens: int = 1024,
    temperature: float = 0.7,
    **kwargs
) -> str:
    """Generate text using a local model with chat template."""
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

