"""Hydra utilities for orchestrator training."""
import datasets
import trl
import torch
import numpy
import transformers
from datasets import load_dataset


def fix_pad_token(tokenizer, model_name):
    """Fix pad token for various model types."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        assert tokenizer.pad_token_id != tokenizer.eos_token_id
    return tokenizer

def wrap_as_list(*args, **kwargs):
    """Wrap arguments as a list."""
    to_return = []
    for element in args:
        to_return.append(element)
    for element in kwargs.values():
        to_return.append(element)
    return to_return


def wrap_as_dict(*args, dict_keys, **kwargs):
    """Wrap arguments as a dict."""
    all_values = list(args) + list(kwargs.values())
    assert len(all_values) == len(dict_keys)
    return {k: v for k, v in zip(dict_keys, all_values)}


# Import trl components for Hydra instantiation
from trl import GRPOConfig, PPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

