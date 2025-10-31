import os 
import abc 
import torch 
import accelerate 
import time 
from torch import nn 
import torch .nn .functional as F 
from contextlib import nullcontext 
import re 

from typing import Any ,Callable ,Optional ,Union ,Dict ,List 
from transformers import Trainer 
from accelerate .utils import broadcast_object_list ,gather_object 

from trl .data_utils import is_conversational 
from trl .import_utils import is_vllm_available 
from trl .extras .profiling import profiling_context 
from trl .models import unwrap_model_for_generation 
from trl .trainer .grpo_trainer import nanstd ,gather_object ,unwrap_model_for_generation ,GRPOTrainer 
from trl .trainer .utils import pad 
from transformers import Trainer 

import logging 
from .conductor_engine import ConductorReward ,gufGRPOTrainer 
logging .getLogger ("google_genai").setLevel (logging .WARNING )


def is_tensor (t ):
    if isinstance (t ,torch .Tensor ):
        return True 
    return False 


if is_vllm_available ():
    from vllm import LLM ,SamplingParams 
    from vllm .sampling_params import GuidedDecodingParams 


def log_tensor_info (tensor ):
    print (f"Shape: {tensor.shape}")
    print (f"Max: {tensor.max().item()}")
    print (f"Min: {tensor.min().item()}")
    print (f"Mean: {tensor.float().mean().item()}")
    print (f"Nan: {torch.isnan(tensor).any().item()}")


def _setup_server_mapping (servers :Union [str ,Dict [str ,str ]],llm_names )->Dict [str ,str ]:

    server_map ={}
    if isinstance (servers ,str ):
        server_list =servers .split (",")
        if len (server_list )==1 :
            for m in llm_names :
                server_map [m ]=server_list [0 ].strip ()
        elif len (server_list )==len (llm_names ):
            for i ,m in enumerate (llm_names ):
                server_map [m ]=server_list [i ].strip ()
        else :
            raise ValueError ("Server count mismatch.")
    elif isinstance (servers ,dict ):
        server_map =servers .copy ()
    return server_map 


dummy_debugging_completion ="""
model_id = [0]
subtasks = ["You are an expert problem solver. Solve this problem."]
access_list = [[]]
"""


class gufrecursionGRPOTrainer (gufGRPOTrainer ):
    def __init__ (
    self ,
    *args ,




    train_recursion_rounds =2 ,



    normalize_rewards_per_recursion_round =True ,

    eval_recursion_rounds =2 ,


    recursion_discount_factor =0.2 ,


    use_recursion_transition_payoff :bool =False ,
    recursion_transition_payoff :dict =None ,
    recursion_penalty_for_unnecessary_rounds =0.0 ,



    recursion_round_processor :Callable [
    [Dict [str ,Any ],str ,str ],List [str ]
    ]=None ,


    use_dummy_debugging_completion :bool =False ,
    **kwargs ,
    ):

        gufGRPOTrainer .__init__ (self ,*args ,**kwargs )
        self .recursion_rounds =train_recursion_rounds 





        self .normalize_rewards_per_recursion_round =normalize_rewards_per_recursion_round 
        self .use_recursion_transition_payoff =use_recursion_transition_payoff 
        if self .use_recursion_transition_payoff :
            print (f"Using recursion transition payoff")
        self .recursion_transition_payoff =recursion_transition_payoff or {"RR":1 ,"WR":1.5 ,"RW":0 ,"WW":0.5 }


        if not (train_recursion_rounds ==1 ):

            assert self .num_generations %train_recursion_rounds ==0 ,"Number of generations must be divisible by train_recursion_rounds"

        self .generations_per_round =self .num_generations //train_recursion_rounds 

        self .recursion_discount_factor =recursion_discount_factor 
        self .recursion_penalty_for_unnecessary_rounds =(
        recursion_penalty_for_unnecessary_rounds 
        )
        if recursion_penalty_for_unnecessary_rounds >0.0 :
            raise NotImplementedError (
            "recursion penalty for unnecessary rounds is not implemented yet."
            )

        assert len (
        self .reward_funcs )==1 ,"gufrecursionGRPOTrainer only supports a single reward function"
        self .conductor_reward =self .reward_funcs [0 ]
        assert isinstance (
        self .conductor_reward ,ConductorReward ),"gufrecursionGRPOTrainer only supports ConductorReward as the reward function"
        if not getattr (self .conductor_reward ,"cache_final_response",False ):
            raise ValueError ("gufrecursionGRPOTrainer only supports ConductorReward with cache_final_response=True")
        if recursion_round_processor is None :
            print ('WARNING: Using default recursion round processor, only meant'
            ' for debugging purposes.')

            def default_recursion_round_processor (
            input_sample :Dict [str ,Any ],
            prompt_sample :str ,
            completion_sample :str ,
            **meta :Any )->List [str ]:
                print ('WARNING: Using default recursion round processor, only '
                'meant for debugging purposes.')
                return prompt_sample +completion_sample 
            recursion_round_processor =default_recursion_round_processor 

        self .recursion_round_processor =recursion_round_processor 
        self .use_dummy_completion =use_dummy_debugging_completion 

    def _generate_and_score_completions (
    self ,inputs :list [dict [str ,Union [torch .Tensor ,Any ]]]
    )->dict [str ,Union [torch .Tensor ,Any ]]:


        all_keys =set ().union (*[example .keys ()for example in inputs ])

        inputs =[{key :example .get (key ,None )
        for key in all_keys }for example in inputs ]

        device =self .accelerator .device 
        mode ="train"if self .model .training else "eval"

        prompts =[x ["prompt"]for x in inputs ]
        num_prompts =len (prompts )
        num_prompts_per_round =num_prompts //self .recursion_rounds 
        prompts_text =prompts .copy ()
        if is_conversational (inputs [0 ]):
            raise NotImplementedError 
        unique_prompts =prompts [::self .num_generations ]
        unique_inputs =inputs [::self .num_generations ]
        unique_inputs =[
        {k :v for k ,v in input_example .items ()if not (k =='prompt')}
        for input_example in unique_inputs ]

        assert set (prompts_text )==set (unique_prompts )

        all_prompts_raw =[None for _ in range (num_prompts )]
        all_prompts_text =[None for _ in range (num_prompts )]
        all_prompt_ids =[None for _ in range (num_prompts )]
        all_completions_text =[None for _ in range (num_prompts )]
        all_completion_ids_list =[None for _ in range (num_prompts )]
        all_rewards =[None for _ in range (num_prompts )]
        all_rewards_per_func =[None for _ in range (num_prompts )]

        latest_recursion_text =[""for _ in range (num_prompts_per_round )]
        recursion_ids_previous_rounds =[]
        prev_round_metadata =None 

        for recursion_round_id in range (self .recursion_rounds ):
            recursion_round_offset_start =(
            self .generations_per_round *recursion_round_id )
            recursion_round_offset_end =(
            recursion_round_offset_start +self .generations_per_round )

            prompts_this_round_raw =[
            item for i ,item in enumerate (
            unique_prompts )for _ in range (self .generations_per_round )]

            prompts_this_round_text =[
            item +latest_recursion_text [i ]for i ,item in enumerate (
            prompts_this_round_raw )]

            recursion_ids_this_round =[
            pid 
            for start in range (0 ,num_prompts ,self .num_generations )
            for pid in range (start +recursion_round_offset_start ,
            start +recursion_round_offset_end )
            ]

            assert all ([p1 ==prompts [pid ]for p1 ,pid in zip (
            prompts_this_round_raw ,recursion_ids_this_round )]),"Failed prompt matching check before generation."

            input_this_round =[inputs [pid ]
            for pid in recursion_ids_this_round ]
            assert all ([p1 ==p2 for p1 ,p2 in zip (
            prompts_this_round_raw ,[input_s ["prompt"]
            for input_s in input_this_round ])]),"Failed prompt matching check before generation."

            inputs_this_round =[
            {**input_example ,"prompt":prompt }for input_example ,prompt 
            in zip (input_this_round ,prompts_this_round_text )
            ]


            for d in inputs_this_round :
                d ["recursion_round_id"]=recursion_round_id 


            if prev_round_metadata is not None :
                for inc_id in range (len (inputs_this_round )):
                    meta =prev_round_metadata [inc_id ]if inc_id <len (prev_round_metadata )else {}
                    if isinstance (meta ,dict ):
                        inputs_this_round [inc_id ]["prev_history"]=meta .get ("history",{})
                        inputs_this_round [inc_id ]["prev_final_response"]=meta .get ("final_worker_response","")


            prompt_inputs_this_round =self .processing_class (
            text =prompts_this_round_text ,return_tensors ="pt",padding =True ,
            padding_side ="left",add_special_tokens =False 
            )
            prompt_inputs_this_round =Trainer ._prepare_inputs (
            self ,prompt_inputs_this_round )
            prompt_ids_tr =prompt_inputs_this_round ["input_ids"]
            prompt_mask_this_round =prompt_inputs_this_round ["attention_mask"]

            if self .max_prompt_length is not None :




                if prompt_ids_tr .shape [1 ]>self .max_prompt_length :
                    print (
                    f'WARNING: Trimming prompt to max_prompt_length, prompts of size {prompt_ids_tr.shape[1]} too long...')
                    raise NotImplementedError (
                    "recursion prompt trimming not implemented")
                    prompt_ids_tr =prompt_ids_tr [:,-self .max_prompt_length :]
                    prompt_mask_this_round =prompt_mask_this_round [:,-
                    self .max_prompt_length :]
                    prompts_this_round_text =self .processing_class .batch_decode (
                    prompt_ids_tr ,skip_special_tokens =False ,
                    clean_up_tokenization_spaces =False ,
                    )
                prompts_this_round_text =[
                re .sub (rf"^({re.escape(self.processing_class.pad_token)})+","",text )for text in prompts_this_round_text 
                ]


            if recursion_round_id >0 :
                prompts_this_round_text =[
                text +"<|im_end|>\n<|im_start|>assistant\n"+"Let's think step by step whether or not I need to make any changes to the previous routing strategy."for text in prompts_this_round_text 
                ]

            (
            prompt_ids_tr ,prompt_completion_ids_tr ,completion_ids_tr ,
            completions_tr ,_ ,
            )=self ._generate_completions (
            inputs =inputs_this_round ,
            prompts =prompts_this_round_text ,
            prompts_text =prompts_this_round_text ,
            prompt_ids =prompt_ids_tr ,
            prompt_mask =prompt_mask_this_round ,
            recursion_round_id =recursion_round_id ,
            )

            _ ,_ ,_ ,_ ,completion_ids_l_tr =self ._build_completion_metadata (
            completion_ids =completion_ids_tr ,
            prompt_mask =prompt_mask_this_round ,
            )
            if self .use_dummy_completion :
                dummy_completions =[dummy_debugging_completion for _ in range (
                len (completions_tr ))]

                rewards_tr ,rewards_per_func_tr =self ._compute_rewards (
                inputs =inputs_this_round ,
                prompts =prompts_this_round_text ,
                completions =dummy_completions ,
                completion_ids_list =completion_ids_l_tr ,
                recursion_round_id =recursion_round_id ,
                )
            else :
                rewards_tr ,rewards_per_func_tr =self ._compute_rewards (
                inputs =inputs_this_round ,
                prompts =prompts_this_round_text ,
                completions =completions_tr ,
                completion_ids_list =completion_ids_l_tr ,
                recursion_round_id =recursion_round_id ,
                )


            cached_conductor_metadata =(
            self .conductor_reward .get_cached_metadata (reset =True )
            )

            for inc_id ,pid in enumerate (recursion_ids_this_round ):
                all_prompts_raw [pid ]=prompts_this_round_raw [inc_id ]
                all_prompts_text [pid ]=prompts_this_round_text [inc_id ]
                all_prompt_ids [pid ]=prompt_ids_tr [inc_id ]
                all_completions_text [pid ]=completions_tr [inc_id ]
                all_rewards [pid ]=rewards_tr [inc_id ]
                all_completion_ids_list [pid ]=completion_ids_l_tr [inc_id ]


                all_rewards_per_func [pid ]=rewards_per_func_tr [inc_id ]



                if recursion_round_id >0 :
                    if self .use_recursion_transition_payoff :
                        if not self .use_recursion_transition_payoff :
                            print ("[trans] gated: flag False")
                        elif len (recursion_ids_previous_rounds )==0 :
                            print ("[trans] gated: no prev ids")
                        else :
                            print (f"[trans] applying payoff: prev_pid={recursion_ids_previous_rounds[-1][inc_id]} inc_id={inc_id}")
                        prev_pid =recursion_ids_previous_rounds [-1 ][inc_id ]
                        prev_meta =prev_round_metadata [inc_id ]if prev_round_metadata and inc_id <len (prev_round_metadata )else {}
                        curr_meta =cached_conductor_metadata [inc_id ]
                        prev_correct =bool (prev_meta .get ("history",{}).get ("correctness_reward",False ))
                        curr_correct =bool (curr_meta .get ("history",{}).get ("correctness_reward",False ))
                        key =("R"if prev_correct else "W")+("R"if curr_correct else "W")
                        payoff =self .recursion_transition_payoff .get (key ,all_rewards [pid ])
                        payoff_tensor =rewards_tr [inc_id ].new_tensor (payoff )




                        all_rewards [pid ]=payoff_tensor 
                        all_rewards [prev_pid ]=payoff_tensor 

            for current_round_id ,pids in enumerate (
            recursion_ids_previous_rounds 
            ):
                if self .use_recursion_transition_payoff :
                    continue 
                discount_this_round =(
                self .recursion_discount_factor **(current_round_id +1 ))

                for inc_id ,(pid ,current_rw )in enumerate (
                zip (pids ,rewards_tr )
                ):

                    all_rewards [pid ]=(
                    1 -discount_this_round )*all_rewards [pid ]+(
                    discount_this_round *current_rw )





            assert len (cached_conductor_metadata )==len (completions_tr ),(f"Cached metadata length:  {len(cached_conductor_metadata)} "
            f"does not match number of completions: {len(completions_tr)}.")
            for inc_id ,(input_s ,raw_prompt_s ,completion_s ,meta_s 
            )in enumerate (zip (
            inputs_this_round ,prompts_this_round_raw ,
            completions_tr ,cached_conductor_metadata 
            )):
                latest_recursion_text [inc_id ]=self .recursion_round_processor (
                input_sample =input_s ,
                prompt_sample =raw_prompt_s ,
                completion_sample =completion_s ,
                **meta_s ,
                )



            prev_round_metadata =cached_conductor_metadata 

            recursion_ids_previous_rounds .append (recursion_ids_this_round )



        assert all ([p ==p2 for p ,p2 in zip (prompts ,all_prompts_raw )]),"Failed prompt matching check after generation."

        prompt_inputs =self .processing_class (
        text =all_prompts_text ,return_tensors ="pt",padding =True ,
        padding_side ="left",add_special_tokens =False 
        )

        prompt_inputs =Trainer ._prepare_inputs (self ,prompt_inputs )
        prompt_ids ,prompt_mask =prompt_inputs ["input_ids"],prompt_inputs ["attention_mask"]

        completion_ids =[torch .tensor (ids ,device =device )
        for ids in all_completion_ids_list ]
        completion_ids =pad (
        completion_ids ,padding_value =self .processing_class .pad_token_id 
        )
        prompt_completion_ids =torch .cat (
        [prompt_ids ,completion_ids ],dim =1 
        )

        (
        completion_mask ,
        is_eos ,
        completion_lengths ,
        attention_mask ,
        completion_ids_list ,
        )=self ._build_completion_metadata (
        completion_ids =completion_ids ,
        prompt_mask =prompt_mask ,
        )


        old_per_token_logps ,ref_per_token_logps =(
        self ._compute_reference_logprobs (
        prompt_completion_ids =prompt_completion_ids ,
        attention_mask =attention_mask ,
        completion_ids =completion_ids ,
        )
        )

        completions_text =self .processing_class .batch_decode (
        completion_ids ,skip_special_tokens =True 
        )

        for c1 ,c2 in zip (all_completions_text ,completions_text ):
            assert c1 ==c2 ,"Completion texts do not match after decoding."

        rewards =torch .stack (all_rewards ,dim =0 )
        rewards_per_func =rewards .unsqueeze (-1 )

        if self .normalize_rewards_per_recursion_round :

            mean_grouped_rewards =rewards .view (
            -1 ,self .recursion_rounds ,self .generations_per_round ).mean (
            dim =-1 ,keepdim =True )
            std_grouped_rewards =rewards .view (
            -1 ,self .recursion_rounds ,self .generations_per_round ).std (
            dim =-1 ,keepdim =True )
            is_std_zero =torch .isclose (
            std_grouped_rewards ,torch .zeros_like (std_grouped_rewards ))


            mean_grouped_rewards =mean_grouped_rewards .repeat_interleave (
            self .generations_per_round ,dim =-1 ).view (-1 )
            std_grouped_rewards =std_grouped_rewards .repeat_interleave (
            self .generations_per_round ,dim =-1 ).view (-1 )
            advantages =rewards -mean_grouped_rewards 
        else :

            mean_grouped_rewards =rewards .view (
            -1 ,self .num_generations ).mean (dim =1 )
            std_grouped_rewards =rewards .view (
            -1 ,self .num_generations ).std (dim =1 )

            is_std_zero =torch .isclose (
            std_grouped_rewards ,torch .zeros_like (std_grouped_rewards ))


            mean_grouped_rewards =mean_grouped_rewards .repeat_interleave (
            self .num_generations ,dim =0 )
            std_grouped_rewards =std_grouped_rewards .repeat_interleave (
            self .num_generations ,dim =0 )
            advantages =rewards -mean_grouped_rewards 

        if self .scale_rewards :
            advantages =advantages /(std_grouped_rewards +1e-4 )







        all_process_advantages =advantages .clone ()
        assert advantages .size (0 )==prompt_ids .size (0 )==completion_ids .size (0 ),"Batch size mismatch among returned tensors."



        if mode =="train":
            self .state .num_input_tokens_seen +=self .accelerator .gather (
            attention_mask .sum ()).sum ().item ()
        self ._metrics [mode ]["num_tokens"]=[self .state .num_input_tokens_seen ]


        agg_completion_lengths =self .accelerator .gather (completion_lengths )
        self ._metrics [mode ]["completions/mean_length"].append (
        agg_completion_lengths .float ().mean ().item ())
        self ._metrics [mode ]["completions/min_length"].append (
        agg_completion_lengths .float ().min ().item ())
        self ._metrics [mode ]["completions/max_length"].append (
        agg_completion_lengths .float ().max ().item ())


        agg_terminated_with_eos =self .accelerator .gather (is_eos .any (dim =1 ))
        term_completion_lengths =agg_completion_lengths [agg_terminated_with_eos ]
        clipped_completions_ratio =1 -len (term_completion_lengths )/len (agg_completion_lengths )
        self ._metrics [mode ]["completions/clipped_ratio"].append (
        clipped_completions_ratio )
        if len (term_completion_lengths )==0 :
            term_completion_lengths =torch .zeros (1 ,device =device )
        self ._metrics [mode ]["completions/mean_terminated_length"].append (
        term_completion_lengths .float ().mean ().item ())
        self ._metrics [mode ]["completions/min_terminated_length"].append (
        term_completion_lengths .float ().min ().item ())
        self ._metrics [mode ]["completions/max_terminated_length"].append (
        term_completion_lengths .float ().max ().item ())


        for i ,reward_func_name in enumerate (self .reward_func_names ):
            mean_rewards =torch .nanmean (rewards_per_func [:,i ]).item ()
            self ._metrics [mode ][f"rewards/{reward_func_name}/mean"].append (
            mean_rewards )
            std_rewards =nanstd (rewards_per_func [:,i ]).item ()
            self ._metrics [mode ][f"rewards/{reward_func_name}/std"].append (
            std_rewards )
        self ._metrics [mode ]["reward"].append (
        mean_grouped_rewards .mean ().item ())
        self ._metrics [mode ]["reward_std"].append (
        std_grouped_rewards .mean ().item ())
        self ._metrics [mode ]["frac_reward_zero_std"].append (
        is_std_zero .float ().mean ().item ())


        self ._textual_logs ["prompt"].extend (gather_object (prompts_text ))
        self ._textual_logs ["completion"].extend (
        gather_object (completions_text ))
        for i ,name in enumerate (self .reward_func_names ):
            self ._textual_logs ["rewards"][name ].extend (
            rewards_per_func [:,i ].tolist ())
        self ._textual_logs ["advantages"].extend (
        all_process_advantages .tolist ())

        return {
        "prompt_ids":prompt_ids ,
        "prompt_mask":prompt_mask ,
        "completion_ids":completion_ids ,
        "completion_mask":completion_mask ,
        "advantages":advantages ,
        "old_per_token_logps":old_per_token_logps ,
        "ref_per_token_logps":ref_per_token_logps ,
        }

    def _generate_completions (
    self ,
    inputs :list [dict [str ,Union [torch .Tensor ,Any ]]],
    prompts :list [Any ],
    prompts_text :list [str ],
    prompt_ids :torch .Tensor ,
    prompt_mask :torch .Tensor ,
    recursion_round_id :Optional [int ]=None ,
    )->dict [str ,Any ]:
        device =self .accelerator .device 
        if self .use_dummy_completion :
            print ("WARNING: Using test recursion logic, "
            "not using vLLM server for generation.")
            all_prompts_text =gather_object (prompts_text )
            if self .accelerator .is_main_process :
                ordered_set_of_prompts =all_prompts_text [::self .num_generations ]
                completions_text =[
                "sample completion"for _ in all_prompts_text ]
                if recursion_round_id is not None :
                    completions_text =["completion {i} for round {round_id}".format (
                    i =i ,round_id =recursion_round_id )
                    for i in range (len (all_prompts_text ))]
                completion_ids =[self .processing_class (
                text =sc ,add_special_tokens =False 
                )['input_ids']for sc in completions_text ]
            else :
                completion_ids =[None ]*len (all_prompts_text )

            completion_ids =broadcast_object_list (
            completion_ids ,from_process =0 )
            slc =slice (
            self .accelerator .process_index *len (prompts ),
            (self .accelerator .process_index +1 )*len (prompts ),
            )
            completion_ids =completion_ids [slc ]
            completion_ids =[torch .tensor (ids ,device =device )
            for ids in completion_ids ]
            completion_ids =pad (
            completion_ids ,padding_value =self .processing_class .pad_token_id 
            )
            prompt_completion_ids =torch .cat (
            [prompt_ids ,completion_ids ],dim =1 )
        elif self .use_vllm :
            if self .state .global_step !=self ._last_loaded_step :
                self ._move_model_to_vllm ()
                self ._last_loaded_step =self .state .global_step 

            if self .vllm_mode =="server":
                all_prompts_text =gather_object (prompts_text )
                if self .accelerator .is_main_process :
                    ordered_set_of_prompts =all_prompts_text [::self .num_generations ]
                    with profiling_context (self ,"vLLM.generate"):
                        completion_ids =self .vllm_client .generate (
                        prompts =ordered_set_of_prompts ,
                        n =self .generations_per_round ,
                        repetition_penalty =self .repetition_penalty ,
                        temperature =self .temperature ,
                        top_p =self .top_p ,
                        top_k =-1 if self .top_k is None else self .top_k ,
                        min_p =0.0 if self .min_p is None else self .min_p ,
                        max_tokens =self .max_completion_length ,
                        guided_decoding_regex =self .guided_decoding_regex ,
                        generation_kwargs =self .args .generation_kwargs ,
                        )
                else :
                    completion_ids =[None ]*len (all_prompts_text )

                completion_ids =broadcast_object_list (
                completion_ids ,from_process =0 )
                slc =slice (
                self .accelerator .process_index *len (prompts ),
                (self .accelerator .process_index +1 )*len (prompts ),
                )
                completion_ids =completion_ids [slc ]

            elif self .vllm_mode =="colocate":
                if self .guided_decoding_regex :
                    guided_decoding =GuidedDecodingParams (
                    backend ="outlines",regex =self .guided_decoding_regex 
                    )
                else :
                    guided_decoding =None 

                gen_kwargs ={
                "n":1 ,
                "repetition_penalty":self .repetition_penalty ,
                "temperature":self .temperature ,
                "top_p":self .top_p ,
                "top_k":-1 if self .top_k is None else self .top_k ,
                "min_p":0.0 if self .min_p is None else self .min_p ,
                "max_tokens":self .max_completion_length ,
                "guided_decoding":guided_decoding ,
                }

                extra_kwargs =getattr (self .args ,"generation_kwargs",None )
                if extra_kwargs :
                    gen_kwargs .update (extra_kwargs )
                sampling_params =SamplingParams (**gen_kwargs )

                if self .vllm_tensor_parallel_size >1 :
                    orig_size =len (prompts_text )
                    gathered =[None ]*self .vllm_tensor_parallel_size 
                    torch .distributed .all_gather_object (
                    gathered ,prompts_text ,group =self .tp_group 
                    )
                    all_prompts_text =[p for sub in gathered for p in sub ]
                else :
                    all_prompts_text =prompts_text 

                with profiling_context (self ,"vLLM.generate"):
                    all_outputs =self .llm .generate (
                    all_prompts_text ,sampling_params =sampling_params ,use_tqdm =False 
                    )

                completion_ids =[
                out .token_ids for outs in all_outputs for out in outs .outputs 
                ]

                if self .vllm_tensor_parallel_size >1 :
                    local_rank =torch .distributed .get_rank (
                    group =self .tp_group )
                    tp_slc =slice (local_rank *orig_size ,
                    (local_rank +1 )*orig_size )
                    completion_ids =completion_ids [tp_slc ]

            completion_ids =[torch .tensor (ids ,device =device )
            for ids in completion_ids ]
            completion_ids =pad (
            completion_ids ,padding_value =self .processing_class .pad_token_id 
            )
            prompt_completion_ids =torch .cat (
            [prompt_ids ,completion_ids ],dim =1 )
        else :
            with unwrap_model_for_generation (
            self .model_wrapped ,
            self .accelerator ,
            gather_deepspeed3_params =self .args .ds3_gather_for_generation ,
            )as unwrapped_model :
                with (
                FSDP .summon_full_params (self .model_wrapped ,recurse =False )
                if self .is_fsdp_enabled 
                else nullcontext ()
                ):
                    prompt_completion_ids =unwrapped_model .generate (
                    prompt_ids ,
                    attention_mask =prompt_mask ,
                    generation_config =self .generation_config ,
                    )

            prompt_len =prompt_ids .size (1 )
            prompt_ids =prompt_completion_ids [:,:prompt_len ]
            completion_ids =prompt_completion_ids [:,prompt_len :]

        completions_text =self .processing_class .batch_decode (
        completion_ids ,skip_special_tokens =True 
        )
        if is_conversational (inputs [0 ]):
            raise NotImplementedError (
            "Conversational inputs are not supported in "
            "gufrecursionGRPOTrainer."
            )
        else :
            completions =completions_text 

        return (
        prompt_ids ,prompt_completion_ids ,completion_ids ,completions_text ,
        completions )

    def _build_completion_metadata (
    self ,
    completion_ids :torch .Tensor ,
    prompt_mask :torch .Tensor ,
    )->tuple [
    torch .Tensor ,
    torch .Tensor ,
    torch .Tensor ,
    torch .Tensor ,
    list [list [int ]],
    ]:
        device =completion_ids .device 


        is_eos =completion_ids ==self .processing_class .eos_token_id 
        eos_idx =torch .full (
        (is_eos .size (0 ),),is_eos .size (1 ),dtype =torch .long ,device =device 
        )
        eos_idx [is_eos .any (dim =1 )]=is_eos .int ().argmax (dim =1 )[
        is_eos .any (dim =1 )]


        seq_idx =torch .arange (is_eos .size (1 ),device =device ).expand (
        is_eos .size (0 ),-1 
        )
        completion_mask =(seq_idx <=eos_idx .unsqueeze (1 )).int ()

        if self .mask_truncated_completions :
            trunc =~is_eos .any (dim =1 )
            completion_mask =completion_mask *(~trunc ).unsqueeze (1 ).int ()

        completion_lengths =completion_mask .sum (1 )
        attention_mask =torch .cat ([prompt_mask ,completion_mask ],dim =1 )


        completion_ids_list =[
        [tok .item ()for tok ,m in zip (row ,mrow )if m ]
        for row ,mrow in zip (completion_ids ,completion_mask )
        ]

        return (
        completion_mask ,
        is_eos ,
        completion_lengths ,
        attention_mask ,
        completion_ids_list ,
        )

    def _compute_rewards (
    self ,
    inputs :list [dict [str ,Union [torch .Tensor ,Any ]]],
    prompts :list [Any ],
    completions :list [str ]|list [Any ],
    completion_ids_list :list [list [int ]],
    recursion_round_id :Optional [int ]=None ,
    )->tuple [torch .Tensor ,torch .Tensor ]:

        if self .use_dummy_completion :
            rewards_per_func =torch .zeros (
            (len (inputs ),len (self .reward_funcs )),device =self .accelerator .device 
            )
            if recursion_round_id is not None :
                rewards_per_func =rewards_per_func +recursion_round_id 
            print ('WARNING: Using dummy zeros rewards for debugging recursion training')
        else :
            rewards_per_func =self ._calculate_rewards (
            inputs ,prompts ,completions ,completion_ids_list 
            )
        rewards =(
        rewards_per_func *self .reward_weights .to (self .accelerator .device )
        .unsqueeze (0 )
        ).nansum (dim =1 )
        return rewards ,rewards_per_func 

    def _compute_reference_logprobs (
    self ,
    prompt_completion_ids :torch .Tensor ,
    attention_mask :torch .Tensor ,
    completion_ids :torch .Tensor ,
    )->tuple [Optional [torch .Tensor ],Optional [torch .Tensor ]]:
        mode ="train"if self .model .training else "eval"
        logits_to_keep =completion_ids .size (1 )
        batch_size =(
        self .args .per_device_train_batch_size 
        if mode =="train"
        else self .args .per_device_eval_batch_size 
        )

        with torch .no_grad ():
            need_old =(
            self .num_iterations >1 
            or self .args .steps_per_generation 
            >self .args .gradient_accumulation_steps 
            )
            old_lp =(
            self ._get_per_token_logps (
            self .model ,
            prompt_completion_ids ,
            attention_mask ,
            logits_to_keep ,
            batch_size ,
            )
            if need_old 
            else None 
            )

            if self .beta !=0.0 :
                ref_model =self .ref_model or self .model 
                ctx =(
                self .accelerator .unwrap_model (self .model ).disable_adapter ()
                if self .ref_model is None 
                else nullcontext ()
                )
                with ctx :
                    ref_lp =self ._get_per_token_logps (
                    ref_model ,
                    prompt_completion_ids ,
                    attention_mask ,
                    logits_to_keep ,
                    )
            else :
                ref_lp =None 

        return old_lp ,ref_lp 

    def _unpad_completions (
    self ,
    completion_ids :torch .Tensor ,
    completion_mask :Optional [torch .Tensor ]=None ,
    padding_value :int |None =None ,
    padding_side :str ="right",
    )->list [torch .Tensor ]:

        if padding_value is None :
            padding_value =self .processing_class .pad_token_id 


        if completion_mask is None :
            if padding_side =="right":
                keep_mask =completion_ids .ne (padding_value )
            elif padding_side =="left":

                keep_mask =completion_ids .ne (padding_value ).cumsum (1 ).bool ()
            else :
                raise ValueError ("padding_side must be 'left' or 'right'")
        else :
            keep_mask =completion_mask .bool ()


        unpadded =[]
        for row ,mask in zip (completion_ids ,keep_mask ):
            unpadded .append (row [mask ].clone ())

        return unpadded 


    def evaluate (self ,*args ,**kwargs ):
        import json 
        from datetime import datetime 

        orig_ngen =self .args .num_generations 
        orig_recursion_rounds =self .recursion_rounds 
        orig_num_generations =getattr (self ,"num_generations",orig_ngen )
        orig_gpr =getattr (self ,"generations_per_round",None )
        orig_eval_bsz =getattr (self ,"per_device_eval_batch_size",None )
        orig_args_eval_bsz =getattr (self .args ,"per_device_eval_batch_size",None )

        try :
            eval_rounds =getattr (self ,"eval_recursion_rounds",2 )or 2 
            self .recursion_rounds =eval_rounds 
            self .args .num_generations =eval_rounds 
            self .num_generations =eval_rounds 
            self .generations_per_round =self .num_generations //self .recursion_rounds 

            self .per_device_eval_batch_size =20 
            self .args .per_device_eval_batch_size =20 



            assert self .reward_funcs [0 ].cache_final_response ,"gufrecursionGRPOTrainer only supports ConductorReward with cache_final_response=True"


            results =super (gufGRPOTrainer ,self ).evaluate (*args ,**kwargs )

        finally :
            self .args .num_generations =orig_ngen 
            self .num_generations =orig_num_generations 
            self .recursion_rounds =orig_recursion_rounds 
            if orig_gpr is not None :
                self .generations_per_round =orig_gpr 
            if orig_eval_bsz is not None :
                self .per_device_eval_batch_size =orig_eval_bsz 
            if orig_args_eval_bsz is not None :
                self .args .per_device_eval_batch_size =orig_args_eval_bsz 


        if self .reward_funcs :
            rw =self .reward_funcs [0 ]
            print ('\n\n --- Accuracy Statistics (Normal aggregate) --- \n')
            for name ,task_dict in rw .full_log .items ():
                if name .startswith ("recursion/")or name =="global_recursion":
                    continue 
                total =task_dict ["total_question_cnt"]
                correct =task_dict ["total_correct_cnt"]
                acc =(correct /total *100.0 )if total else 0.0 
                print (f"{name} | Accuracy: {correct} / {total} -> {acc:.3f}%")

            print ('\n --- Accuracy Statistics (recursion-only, final round) --- \n')
            for name ,task_dict in rw .full_log .items ():




                correct =task_dict .get ("recursion_correct_cnt",0 )
                total =task_dict .get ("recursion_total_cnt",0 )
                acc =(correct /total *100.0 )if total else 0.0 
                print (f"{name} | Accuracy: {correct} / {total} -> {acc:.3f}%")

                print ('\n-------recursion analytics -------')
                return_cnt =task_dict .get ("recursion_return_cnt",0 )
                return_correct_cnt =task_dict .get ("recursion_return_correct_cnt",0 )
                return_acc =(return_correct_cnt /return_cnt *100.0 )if return_cnt else 0.0 
                print (f"recursion direct return accuracy: {return_correct_cnt} / {return_cnt} -> {return_acc:.3f}%")
                round_cnt =task_dict .get ("recursion_round_cnt",0 )
                return_ratio =return_cnt /round_cnt if round_cnt else 0.0 
                print (f"recursion return ratio: {return_cnt} / {round_cnt} -> {return_ratio:.3f}%")
                revise_correct_cnt =task_dict .get ("recursion_revise_strategy_correct_cnt",0 )
                revise_cnt =task_dict .get ("recursion_revise_strategy_cnt",0 )
                revise_acc =(revise_correct_cnt /revise_cnt *100.0 )if revise_cnt else 0.0 
                print (f"recursion revise strategy accuracy: {revise_correct_cnt} / {revise_cnt} -> {revise_acc:.3f}%")
                revise_ratio =revise_cnt /round_cnt if round_cnt else 0.0 
                print (f"recursion revise strategy ratio: {revise_cnt} / {round_cnt} -> {revise_ratio:.3f}%")


            try :
                from pathlib import Path 
                out_dir =Path (rw .evaluate_only )
                out_dir .mkdir (parents =True ,exist_ok =True )
                out_path =out_dir /"eval_results.txt"

                try :
                    from filelock import FileLock 
                    lock =FileLock (str (out_path )+".lock")
                    lock_ctx =lock 
                except Exception :
                    from contextlib import nullcontext 
                    lock_ctx =nullcontext ()

                task_label =None 
                if self .reward_funcs and hasattr (self .reward_funcs [0 ],"task"):
                    task_label =self .reward_funcs [0 ].task .__class__ .__name__ 

                with lock_ctx :
                    with open (out_path ,"a",encoding ="utf-8")as f :
                        f .write ("\n=== Evaluation Results (recursion) ===\n")
                        f .write (f"timestamp: {datetime.now().isoformat()}\n")
                        if task_label :
                            f .write (f"task: {task_label}\n")
                        try :
                            f .write (json .dumps ({"results":results },indent =2 )+"\n")
                        except Exception :
                            f .write (str (results )+"\n")

                        try :
                            cfg =rw .config_snapshot ()if hasattr (rw ,"config_snapshot")else {
                            k :v for k ,v in rw .__dict__ .items ()if not k .startswith ("_")
                            }
                            f .write ("-- Eval Config --\n")
                            f .write (json .dumps ({"Conductor Config":cfg },indent =2 ,default =str ))
                        except Exception as e :
                            f .write (f"Conductor config snapshot failed: {e}\n")

                        f .write ("--- Normal aggregate ---\n")
                        for name ,task_dict in rw .full_log .items ():
                            if name .startswith ("recursion/")or name =="global_recursion":
                                continue 
                            total =task_dict ["total_question_cnt"]
                            correct =task_dict ["total_correct_cnt"]
                            acc =(correct /total *100.0 )if total else 0.0 
                            f .write (f"{name} | Accuracy: {correct} / {total} -> {acc:.3f}%\n")

                        f .write ("--- recursion-only (final round) ---\n")
                        for name ,task_dict in rw .full_log .items ():




                            correct =task_dict .get ("recursion_correct_cnt",0 )
                            total =task_dict .get ("recursion_total_cnt",0 )
                            acc =(correct /total *100.0 )if total else 0.0 
                            f .write (f"{name} | Accuracy: {correct} / {total} -> {acc:.3f}%\n")

                            f .write ('\n-------recursion analytics -------')
                            return_cnt =task_dict .get ("recursion_return_cnt",0 )
                            return_correct_cnt =task_dict .get ("recursion_return_correct_cnt",0 )
                            return_acc =(return_correct_cnt /return_cnt *100.0 )if return_cnt else 0.0 
                            f .write (f"recursion direct return accuracy: {return_correct_cnt} / {return_cnt} -> {return_acc:.3f}%")
                            round_cnt =task_dict .get ("recursion_round_cnt",0 )
                            return_ratio =return_cnt /round_cnt if round_cnt else 0.0 
                            f .write (f"recursion return ratio: {return_cnt} / {round_cnt} -> {return_ratio:.3f}%")
                            revise_correct_cnt =task_dict .get ("recursion_revise_strategy_correct_cnt",0 )
                            revise_cnt =task_dict .get ("recursion_revise_strategy_cnt",0 )
                            revise_acc =(revise_correct_cnt /revise_cnt *100.0 )if revise_cnt else 0.0 
                            f .write (f"recursion revise strategy accuracy: {revise_correct_cnt} / {revise_cnt} -> {revise_acc:.3f}%")
                            revise_ratio =revise_cnt /round_cnt if round_cnt else 0.0 
                            f .write (f"recursion revise strategy ratio: {revise_cnt} / {round_cnt} -> {revise_ratio:.3f}%")

                print (f"Wrote evaluation summary to {out_path}")
            except Exception as e :
                print (f"Failed to write eval summary: {e}")

        return results 