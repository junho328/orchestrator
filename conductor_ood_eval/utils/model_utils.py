


import os 
import torch 
import torch .nn as nn 
import numpy as np 
from typing import List ,Dict ,Any ,Optional ,Tuple 
from transformers import AutoTokenizer 

import torch 
from typing import Optional ,Union ,List ,Tuple ,Unpack 

from transformers .models .qwen2 .modeling_qwen2 import (
CausalLMOutputWithPast ,
KwargsForCausalLM ,
)
from transformers .cache_utils import Cache 


DEBUG =True 


def qwen2_forward (
self ,
input_ids :torch .LongTensor =None ,
attention_mask :Optional [torch .Tensor ]=None ,
position_ids :Optional [torch .LongTensor ]=None ,
past_key_values :Optional [Union [Cache ,List [torch .FloatTensor ]]]=None ,
inputs_embeds :Optional [torch .FloatTensor ]=None ,
labels :Optional [torch .LongTensor ]=None ,
use_cache :Optional [bool ]=None ,
output_attentions :Optional [bool ]=None ,
output_hidden_states :Optional [bool ]=None ,
return_dict :Optional [bool ]=None ,
cache_position :Optional [torch .LongTensor ]=None ,
logits_to_keep :Union [int ,torch .Tensor ]=0 ,
action_layer :torch .nn .Module =None ,
**kwargs :Unpack [KwargsForCausalLM ],
)->Union [Tuple ,CausalLMOutputWithPast ]:
    output_attentions =output_attentions if output_attentions is not None else self .config .output_attentions 
    output_hidden_states =(
    output_hidden_states if output_hidden_states is not None else self .config .output_hidden_states 
    )
    return_dict =return_dict if return_dict is not None else self .config .use_return_dict 


    outputs =self .model (
    input_ids =input_ids ,
    attention_mask =attention_mask ,
    position_ids =position_ids ,
    past_key_values =past_key_values ,
    inputs_embeds =inputs_embeds ,
    use_cache =use_cache ,
    output_attentions =output_attentions ,
    output_hidden_states =output_hidden_states ,
    return_dict =return_dict ,
    cache_position =cache_position ,
    **kwargs ,
    )

    hidden_states =outputs [0 ]

    slice_indices =slice (-logits_to_keep ,None )if isinstance (logits_to_keep ,int )else logits_to_keep 
    logits =self .lm_head (hidden_states [:,slice_indices ,:])

    assert action_layer is not None ,"Action layer is not None"
    return action_layer (hidden_states [:,-2 ,:])

class SVDParameterManager :


    @staticmethod 
    def load_svd_weights (model_name :str ,device :str ="cpu")->Dict [str ,torch .Tensor ]:


        svd_file =os .path .join (
        "path/to/svd/weights",
        model_name .replace ("/","_"),
        "svd_weights.pt"
        )
        if not os .path .exists (svd_file ):
            raise FileNotFoundError (f"SVD file not found: {svd_file}")
        return torch .load (svd_file ,map_location =device )


    @staticmethod 
    def filter_svd_weights_by_layers (
    svd_weights :Dict [str ,torch .Tensor ],
    layer_indices :Optional [List [int ]]=None 
    )->Dict [str ,torch .Tensor ]:

        if layer_indices is None :
            return svd_weights 

        filtered_weights ={}
        for key ,tensor in svd_weights .items ():

            base_key =key .rsplit (".",1 )[0 ]if "."in key else key 


            if "model.layers."not in base_key :
                filtered_weights [key ]=tensor 
                continue 


            for idx in layer_indices :
                if f"model.layers.{idx}."in base_key :
                    filtered_weights [key ]=tensor 
                    break 

        return filtered_weights 


class EvaluationManager :


    @staticmethod 
    def get_action (model :nn .Module ,linear_layer :nn .Module ,tokenizer :AutoTokenizer ,
    messages :List [Dict [str ,str ]],inference :bool =True )->np .ndarray :

        formatted_text =EvaluationManager ._format_messages (messages ,tokenizer )

        input_ids =tokenizer (formatted_text ,return_tensors ="pt").input_ids .to (model .device )
        if inference :
            with torch .no_grad ():
                action =model (input_ids ,action_layer =linear_layer )
            return action .float ().cpu ().numpy ().squeeze ()
        else :
            return model (input_ids ,action_layer =linear_layer )

    @staticmethod 
    def _format_messages (messages :List [Dict [str ,str ]],tokenizer :AutoTokenizer )->str :


        if hasattr (tokenizer ,'apply_chat_template')and tokenizer .chat_template :
            try :
                return tokenizer .apply_chat_template (messages ,tokenize =False ,add_generation_prompt =True )
            except Exception :

                pass 


        formatted_parts =[]
        for msg in messages :
            role =msg ["role"]
            content =msg ["content"]

            if role =="system":
                formatted_parts .append (f"System: {content}")
            elif role =="user":
                formatted_parts .append (f"User: {content}")
            elif role =="assistant":
                formatted_parts .append (f"Assistant: {content}")

        return "\n".join (formatted_parts )+"\nAssistant:"


class ModelParameterApplier :


    @staticmethod 
    def apply_cma_parameters (model :nn .Module ,linear_layer :nn .Module ,
    flat_params :np .ndarray ,svd_weights :Dict [str ,torch .Tensor ])->None :

        model_dict =model .state_dict ()
        offset =0 


        for full_key in model_dict .keys ():
            sv_key =f"{full_key}.S"
            if sv_key in svd_weights :
                device =model .get_parameter (full_key ).device 
                dtype =model .get_parameter (full_key ).dtype 

                S =svd_weights [sv_key ].to (device ,dtype )
                s_size =S .numel ()

                scale_chunk =flat_params [offset :offset +s_size ]
                offset +=s_size 
                scale_factors =torch .from_numpy (scale_chunk ).to (device ,dtype )+1.0 

                U =svd_weights [f"{full_key}.U"].to (device ,dtype )
                V =svd_weights [f"{full_key}.V"].to (device ,dtype )

                scaled_S =S *scale_factors 
                new_param =(U @torch .diag_embed (scaled_S )@V .transpose (-1 ,-2 ))*(
                S .sum ()/scaled_S .sum ()
                )

                model .get_parameter (full_key ).data .copy_ (new_param )


        device =linear_layer .weight .device 
        dtype =linear_layer .weight .dtype 
        w_size =linear_layer .weight .numel ()

        if flat_params .shape [0 ]-offset >=w_size :
            w_chunk =flat_params [offset :offset +w_size ]
            w_tensor =torch .from_numpy (w_chunk ).to (device ,dtype )
            linear_layer .weight .data .copy_ (w_tensor .view_as (linear_layer .weight ))

        return model ,linear_layer 


def load_model_config (log_file_path :str )->Dict [str ,Any ]:

    import json 

    with open (log_file_path ,'r')as f :
        log_data =json .load (f )


    return log_data [0 ]['configs']


def find_model_files (model_path :str )->Tuple [Optional [str ],Optional [str ]]:

    log_file =None 
    model_file =None 


    for file in os .listdir (model_path ):
        if file .endswith ('_log.json'):
            log_file =os .path .join (model_path ,file )
            break 


    models_dir =os .path .join (model_path ,"models")
    if os .path .exists (models_dir ):
        for ext in ['.npy','.pt']:
            candidate =os .path .join (models_dir ,f"best_model{ext}")
            if os .path .exists (candidate ):
                model_file =candidate 
                break 

    return model_file ,log_file 