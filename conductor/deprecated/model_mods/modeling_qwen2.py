import torch 
from typing import Optional ,Union ,List ,Tuple ,Unpack 

from transformers .models .qwen2 .modeling_qwen2 import (
CausalLMOutputWithPast ,
)
from transformers .cache_utils import Cache 




def forward (
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
    )

    hidden_states =outputs [0 ]

    slice_indices =slice (-logits_to_keep ,None )if isinstance (logits_to_keep ,int )else logits_to_keep 
    logits =self .lm_head (hidden_states [:,slice_indices ,:])

    if action_layer is not None :
        return action_layer (hidden_states [:,-1 ,:])

    loss =None 
    if labels is not None :
        loss =self .loss_function (logits =logits ,labels =labels ,vocab_size =self .config .vocab_size ,**kwargs )

    if not return_dict :
        output =(logits ,)+outputs [1 :]
        return (loss ,)+output if loss is not None else output 

    return CausalLMOutputWithPast (
    loss =loss ,
    logits =logits ,
    past_key_values =outputs .past_key_values ,
    hidden_states =outputs .hidden_states ,
    attentions =outputs .attentions ,
    )