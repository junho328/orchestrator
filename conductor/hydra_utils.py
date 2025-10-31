import datasets 
import trl 
import torch 
import numpy 
import transformers 
from datasets import load_dataset 


def fix_pad_token (tokenizer ,model_name ):
    if tokenizer .pad_token is None :
        if "Llama"in model_name :
            tokenizer .pad_token ="<|reserved_special_token_5|>"
        elif "Qwen"in model_name :
            tokenizer .pad_token ="<|fim_pad|>"
        else :
            raise NotImplementedError 
    else :
        assert tokenizer .pad_token_id !=tokenizer .eos_token_id 
    return tokenizer 


def wrap_as_list (*args ,**kwargs ):
    to_return =[]
    for element in args :
        to_return .append (element )
    for element in kwargs .values ():
        to_return .append (element )
    return to_return 


def wrap_as_dict (*args ,dict_keys ,**kwargs ):
    all_values =list (args )+list (kwargs .values ())
    assert len (all_values )==len (dict_keys )
    return {k :v for k ,v in zip (dict_keys ,all_values )}
