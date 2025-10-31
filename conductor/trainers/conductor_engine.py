import os 
import abc 
import torch 
import accelerate 
import time 
from torch import nn 
import torch .nn .functional as F 
import json 
import numpy as np 
from datetime import datetime 
import random 
import re 
import ast 
import sys 
import warnings 
from typing import Any ,Callable ,Optional ,Union ,Sequence 
from itertools import islice 
from transformers import PreTrainedTokenizer ,AutoModelForCausalLM 
from trl import GRPOTrainer 
from trl import GRPOConfig 
from trl .extras .profiling import profiling_context ,profiling_decorator 
from trl .data_utils import apply_chat_template ,is_conversational ,maybe_apply_chat_template 
from accelerate .utils import gather 
from .math_parsing_util import (
extract_answer ,
strip_answer_string ,
)
from functools import partial 
from .conductor_utils import (query_llm ,_is_all_access ,_match_name ,_extract_any ,_check_all_access ,_check_ctop_access_list ,
setup_task_data ,_write_log ,_calculate_cost_bonus ,_ascribe_history_binary ,_ascribe_history_complex ,_ascribe_history_positional_complex ,contains_subtask_misspecification )
from typing import List ,Dict ,Tuple 
from concurrent .futures import ThreadPoolExecutor ,as_completed 
import copy 
import unicodedata 
import importlib 
import threading 
import logging 
logging .getLogger ("google_genai").setLevel (logging .WARNING )

_log_write_lock =threading .Lock ()

logging .getLogger ("google_genai").setLevel (logging .WARNING )



def is_tensor (t ):
    if isinstance (t ,torch .Tensor ):
        return True 
    return False 


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


class CustomTrainer (abc .ABC ):
    def __init__ (
    self ,
    tokenizer ,
    reward_functions =None ,
    output_dir =None ,
    logging_prob =0.0 ,
    ):

        if reward_functions is None :


            reward_functions =self .reward_funcs 
        for rw in reward_functions :
            if isinstance (rw ,CustomReward ):
                rw .link_with_trainer (

                trainer =self ,
                tokenizer =tokenizer ,
                )
        self .logging_dir =output_dir 
        self .logging_prob =logging_prob 
        self .last_logged_iter =-1 
        self .do_log_to_file =False 
        if logging_prob >0.0 and output_dir is not None :
            self .do_log_to_file =True 
            self .logging_dir =output_dir +'/completions_logs'
            os .makedirs (self .logging_dir ,exist_ok =True )
            self .logging_file =self .logging_dir +'/log.txt'

    def log_to_file (self ,*args ,**kwargs ):
        if self .do_log_to_file :
            if not (self .state .global_step ==self .last_logged_iter ):
                with open (f"{self.logging_file}","a")as f :
                    f .write ("\n\n============================\n"+
                    f"Global step: {self.state.global_step}")
                self .last_logged_iter =self .state .global_step 
            if random .random ()<self .logging_prob :
                for log_value in args :
                    assert isinstance (log_value ,str )
                    with open (f"{self.logging_file}","a")as f :
                        f .write ("\n\n==============\n"+log_value )
                for log_name ,log_value in kwargs .items ():
                    assert isinstance (log_value ,str )
                    with open (f"{self.logging_dir}/{log_name}.txt","a")as f :
                        f .write ("\n\n==============\n"+log_value )


    def log_metric (self ,**kwargs ):
        logged_dict ={}
        for log_name ,log_value in kwargs .items ():
            if is_tensor (log_value ):
                log_value =log_value .mean ().item ()



            elif isinstance (log_value ,(list ,tuple )):
                log_value =np .mean (log_value )
            else :
                log_value =float (log_value )
            logged_dict [log_name ]=log_value 
            if self .accelerator .is_main_process :
                self ._metrics [log_name ].append (log_value )
        return logged_dict 


class CustomReward (abc .ABC ):
    def link_with_trainer (
    self ,trainer ,tokenizer ,numeric_answer =False ):
        self .__name__ =self .__class__ .__name__ 
        self ._numeric_answer =numeric_answer 
        self ._format_rw ,self ._extract_solution =(
        self .make_answer_check_and_functions (numeric_answer =numeric_answer ))
        self .trainer =trainer 

    def make_answer_check_and_functions (self ,numeric_answer ):
        def format_reward_fn (completions ,start_think_tag ,end_think_tag ,start_solution_tag ,end_solution_tag ,**kwargs ):
            rewards =[]
            for completion in completions :
                try :
                    completion =f"{start_think_tag}"+completion 
                    regex =f"^{re.escape(start_think_tag)}([\s\S]*?){re.escape(end_think_tag)}\n\n"
                    regex +=f"{re.escape(start_solution_tag)}([\s\S]*?){re.escape(end_solution_tag)}$"
                    match =re .search (regex ,completion ,re .DOTALL )
                    rewards .append (1.0 if match and len (
                    match .groups ())==2 else 0.0 )
                except Exception :
                    rewards .append (0.0 )
            return rewards 

        def solution_extraction_fn (completion ,start_think_tag ,end_think_tag ,start_solution_tag ,end_solution_tag ,**kwargs ):
            match =re .search (
            f"{re.escape(start_solution_tag)}(.*?){re.escape(end_solution_tag)}",completion ,re .DOTALL )
            hyp =match .group (1 ).strip ()

            if numeric_answer :
                hyp =extract_answer (hyp )
                hyp =strip_answer_string (hyp )
            return hyp 

        return [format_reward_fn ,solution_extraction_fn ]

    @abc .abstractmethod 
    def __call__ (
    self ,
    prompts ,
    completions ,
    start_think_tag ,
    end_think_tag ,
    start_solution_tag ,
    end_solution_tag ,
    **kwargs ,
    ):
        raise NotImplementedError 


class ConductorReward (CustomReward ):
    def __init__ (
    self ,
    open_models =[],
    closed_models =[],
    servers =[],
    ports =[],
    max_tokens =1024 ,
    temperature =0.7 ,
    task_module ="",
    task_class_name ="",
    use_guf =False ,
    user_content_format ="include_overall_task",
    random_names =['Fish','Cat','Bird','Dog'],
    chunk_size =8 ,
    model_id_format ="numeric",
    overall_task ="countdown",
    score_repeats =2 ,
    training_tasks =['deepmath','mmlu','livecodebenchv6','rlpr'],
    include_overall_task =True ,
    dump_log_data =False ,
    output_dir =None ,
    allow_interrupt =False ,
    store_full_log =False ,
    coordination_log_dir =None ,
    cost_bonus_weight =0.0 ,
    access_history_type ="binary",
    format_bonus =0.0 ,
    final_agent_knowledge =False ,
    gemini_thinking_budget =1024 ,
    claude_thinking_budget =1024 ,
    gpt_reasoning_effort ='minimal',
    anthropic_platform ='bedrock',
    evaluate_only =False ,
    cache_final_response =False ,
    share_agent_subtasks =False ,
    non_chain_bonus =0.0 ,
    *args ,
    **kwargs ):



        self .available_models =closed_models +open_models 
        print (f"Available models: {self.available_models}")
        self .servers =servers 
        self .ports =ports 
        self .max_tokens =max_tokens 
        self .temperature =temperature 
        self .user_content_format =user_content_format 
        self .access_history_type =access_history_type 
        self .format_bonus =format_bonus 
        self .evaluate_only =evaluate_only 
        self .share_agent_subtasks =share_agent_subtasks 

        self .USER_QUESTION_RE =re .compile (
        r"""
            USER\ *QUESTION:          # anchor label
            \s*                       # maybe spaces / newlines
            (.*?)                     # capture the actual question lazily
            (?=                       # stop at…
                \n\s*\n               #   a blank line        OR
            | AVAILABLE\ *LANGUAGE  #   “AVAILABLE LANGUAGE MODELS:”
            | <\|im_start\|>        #   next chat segment
            | $                     #   end of string
            )
            """,
        re .I |re .S |re .X ,
        )

        self .ANSWER_RE =re .compile (
        r"<answer>\s*(.*?)\s*</answer>",re .I |re .S )

        print (f'Using B200 server at {self.servers} \n')
        self .servers =_setup_server_mapping (self .servers ,open_models )


        self .max_routing_steps =5 

        self .include_overall_task =include_overall_task 


        self .random_names =random_names 
        self .model_id_format =model_id_format 
        self .chunk_size =chunk_size 
        self .score_repeats =score_repeats 

        self .allow_interrupt =allow_interrupt 
        self .cache_final_response =cache_final_response 

        self .training_tasks =training_tasks or [
        'deepmath','mmlu','polaris','rlpr']
        self .full_log =self ._create_full_log ()
        self .output_dir =output_dir 
        self .dump_log_data =dump_log_data 
        self .store_full_log =store_full_log 
        self .coordination_log_dir =coordination_log_dir 
        self .cost_bonus_weight =cost_bonus_weight 
        self .non_chain_bonus =non_chain_bonus 
        self .final_agent_knowledge =final_agent_knowledge 

        self .anthropic_platform =anthropic_platform 
        self .gemini_thinking_budget =gemini_thinking_budget 
        self .claude_thinking_budget =claude_thinking_budget 
        self .gpt_reasoning_effort =gpt_reasoning_effort 
        self .use_guf =use_guf 
        if self .use_guf :
            target_mod =importlib .import_module (task_module )
            self .task =getattr (target_mod ,task_class_name )(
            llm_names =self .available_models )
            print (
            f"Task {self.task.__class__.__name__} initialized with models: {self.available_models}")

            print (f'TASK: {self.task}')
            self .reward_func =self .task ._calculate_reward 

        if self .cache_final_response :
            self .final_response_cache =[]
            self .history_cache =[]
        self .debug_log_dir =f'{self.coordination_log_dir}/transcript_debug'
        if self .debug_log_dir :
            os .makedirs (self .debug_log_dir ,exist_ok =True )
            self .correct_log_file_path =os .path .join (
            self .debug_log_dir ,"correct_completions.txt")
            self .incorrect_log_file_path =os .path .join (
            self .debug_log_dir ,"incorrect_completions.txt")
        else :
            self .correct_log_file_path ="correct_completions.txt"
            self .incorrect_log_file_path ="incorrect_completions.txt"

        self .recursion_round_id =0 

        self .__name__ ="ConductorReward"

    def config_snapshot (self ):
        keys =[
        "available_models","servers","ports","max_tokens","temperature",
        "user_content_format","access_history_type","format_bonus",
        "evaluate_only","share_agent_subtasks","max_routing_steps",
        "include_overall_task","random_names","model_id_format",
        "chunk_size","score_repeats","allow_interrupt","training_tasks",
        "output_dir","dump_log_data","store_full_log","coordination_log_dir",
        "cost_bonus_weight","anthropic_platform","gemini_thinking_budget","final_agent_knowledge",
        "claude_thinking_budget","gpt_reasoning_effort","use_guf","cache_final_response",
        ]
        cfg ={k :getattr (self ,k ,None )for k in keys if hasattr (self ,k )}
        if hasattr (self ,"task"):
            try :
                cfg ["task_name"]=self .task .__class__ .__name__ 
            except Exception :
                cfg ["task_name"]=str (type (getattr (self ,"task",None )))
        return cfg 

    def get_cached_metadata (self ,reset =False ):
        if self .cache_final_response :
            metadata_to_return =[{'final_worker_response':resp ,
            'history':history }
            for resp ,history in zip (self .final_response_cache ,self .history_cache )]
            if reset :
                self .final_response_cache =[]
                self .history_cache =[]
        else :
            raise ValueError 
        return metadata_to_return 

    def _write_conversation_log (
    self ,
    router_completion :str ,
    subtasks ,
    model_ids ,
    access_list ,
    messages ,
    agent_responses ,
    correct :bool ,
    base_question :str ,
    task_type :str ,
    available_models :List [str ],
    subtask_critique_flag :bool ,
    non_chain_topology_flag :bool ,
    recursion_round_id :int ,
    recursion_return_flag :bool ,
    recursion_prev_history :dict ,
    task_data :dict ,
    ):

        file_path =self .correct_log_file_path if correct else self .incorrect_log_file_path 
        if self .evaluate_only :

            with _log_write_lock :
                if not hasattr (self ,"eval_log_date")or not self .eval_log_date :
                    self .eval_log_date =datetime .now ().strftime ("%m%d%H%M")

            subdir =os .path .join (self .evaluate_only ,f"eval_{task_type}")
            os .makedirs (subdir ,exist_ok =True )
            key =(task_type ,bool (correct ))
            fname =f"eval_transcript_{'correct' if correct else 'incorrect'}_{self.eval_log_date}.txt"
            with _log_write_lock :
                if not hasattr (self ,"_eval_transcript_files"):
                    self ._eval_transcript_files ={}
                if key not in self ._eval_transcript_files :
                    self ._eval_transcript_files [key ]=os .path .join (subdir ,fname )

            file_path =self ._eval_transcript_files [key ]


        if task_type =="math500"or task_type =="mmlu":
            correct_answer =task_data ["answer"]
        elif task_type =="rlpr":
            correct_answer =task_data ["ground_truth"]
        elif task_type =="livecodebench":
            correct_answer ='N/A'
        else :
            correct_answer ='N/A'

        try :
            with _log_write_lock :

                _write_log (file_path ,task_type ,base_question ,router_completion ,subtasks ,model_ids ,access_list ,messages ,agent_responses ,available_models ,correct_answer =correct_answer )


                if subtask_critique_flag :
                    flagged_subtasks_file_name =f'flagged_subtasks_correct.txt'if correct else f'flagged_subtasks_incorrect.txt'
                    flagged_subtasks_file_path =os .path .join (self .debug_log_dir ,flagged_subtasks_file_name )
                    _write_log (flagged_subtasks_file_path ,task_type ,base_question ,router_completion ,subtasks ,model_ids ,access_list ,messages ,agent_responses ,available_models )


                if non_chain_topology_flag :
                    non_chain_topology_file_name =f'non_chain_topology_correct.txt'if correct else f'non_chain_topology_incorrect.txt'
                    non_chain_topology_file_path =os .path .join (self .debug_log_dir ,non_chain_topology_file_name )
                    _write_log (non_chain_topology_file_path ,task_type ,base_question ,router_completion ,subtasks ,model_ids ,access_list ,messages ,agent_responses ,available_models )

                if recursion_round_id >0 :
                    recursion_round_file_name =f'recursion_round_{recursion_round_id}_correct.txt'if correct else f'recursion_round_{recursion_round_id}_incorrect.txt'
                    recursion_round_file_path =os .path .join (self .debug_log_dir ,recursion_round_file_name )
                    _write_log (recursion_round_file_path ,task_type ,base_question ,router_completion ,subtasks ,model_ids ,access_list ,messages ,agent_responses ,
                    available_models =available_models ,recursion_prev_history =recursion_prev_history ,correct_answer =correct_answer )
        except Exception as e :

            print (f"[WARN] Could not write conversation log: {e}")

    def _create_full_log (self ):

        def _blank ():
            return {
            "all_access_cnt":0 ,
            "invalid_access_cnt":0 ,
            "model_id_error_cnt":0 ,
            "format_error_cnt":0 ,
            "total_question_cnt":0 ,
            "total_correct_cnt":0 ,
            "total_access_cnt":0 ,
            "interrupt_cnt":0 ,
            "avg_subtask_length":0 ,
            "total_cost":0 ,
            "request_timeout_cnt":0 ,
            "format_bonus_cnt":0 ,
            "invalid_ctop_access_cnt":0 ,
            "subtask_misspecification_cnt":0 ,
            "non_chain_topology_cnt":0 ,
            "non_chain_topology_correct_cnt":0 ,
            "complex_topology_future_error_cnt":0 ,
            "recursion_return_cnt":0 ,
            "recursion_round_cnt":0 ,
            "recursion_return_correct_cnt":0 ,
            "recursion_revise_strategy_cnt":0 ,
            "recursion_revise_strategy_correct_cnt":0 ,
            "model_none_returns_cnt":{i :0 for i in range (len (self .available_models ))},
            "model_selected_counts":
            {i :0 for i in range (len (self .available_models ))},
            "model_visible_counts":
            {i :0 for i in range (len (self .available_models ))},
            "position_visible_counts":
            {i :0 for i in range (self .max_routing_steps )},
            "position_model_counts":
            {p :{i :0 for i in range (len (self .available_models ))}
            for p in range (self .max_routing_steps )},
            "total_position_counts":
            {p :0 for p in range (self .max_routing_steps )},
            "chosen_num_routing_steps":
            {i +1 :0 for i in range (self .max_routing_steps )},
            }


        full_log ={"global":_blank ()}
        for task in self .training_tasks :
            full_log [task ]=_blank ()

        if self .cache_final_response :

            full_log ["global_recursion"]=_blank ()
            for task in self .training_tasks :
                full_log [f"recursion/{task}"]=_blank ()

        return full_log 

    def _parse_completions (self ,completion :str ,model_id_format :str ='numeric')->Tuple [List ,List ,List ,Dict ]:



        text =completion .replace ("`","")
        text =re .sub (r"^[ \t]*[-*•]\s*","",text ,
        flags =re .MULTILINE )


        try :
            text =re .sub (r"<think>.*?</think>","",text ,flags =re .DOTALL )
        except Exception :
            pass 


        raw_model_ids =_extract_any (
        text ,["model_id","model id","model ids"])
        subtasks =_extract_any (text ,["subtasks","subtask"])
        access_list =_extract_any (
        text ,["access_list","access list","access"])


        if self .recursion_round_id >0 :
            if subtasks ==[]and raw_model_ids ==[]and access_list ==[]:
                return subtasks ,raw_model_ids ,access_list ,{}


        model_ids =[]
        if model_id_format =='numeric':
            for mid in raw_model_ids :
                if isinstance (mid ,int ):
                    model_ids .append (mid )
                elif isinstance (mid ,str )and mid .strip ().isdigit ():
                    model_ids .append (int (mid .strip ()))
                else :
                    print (f"Ignoring non-numeric model_id {mid}")
        elif model_id_format =='random_names':
            for mid in raw_model_ids :
                if _match_name (mid ,self .random_names )is not None :
                    model_ids .append (_match_name (mid ,self .random_names ))
                else :

                    raise ValueError (f"Invalid model_id: {mid}")
        else :
            raise ValueError (f"Invalid model_id_format: {model_id_format}")

        parsing_error_dict =self ._check_parsing (
        subtasks ,model_ids ,access_list ,completion )


        return subtasks ,model_ids ,access_list ,parsing_error_dict 

    def _prepare_agent_messages (
    self ,
    subtask :str ,
    access_list ,
    history ,
    idx ,
    think_tags :bool =False ,
    all_subtasks :List [str ]=[],
    mid :int =None ,
    **kwargs ,
    )->List [dict ]:

        if self .use_guf :
            user_content =kwargs ['base_question'][idx ]if isinstance (
            kwargs ['base_question'],list )else kwargs ['base_question']
            user_content =str (
            user_content )if user_content is not None else ""
            if user_content =="":
                raise ValueError (f"Base question is empty for index {idx}")
        else :
            raise ValueError (
            f"WARNING: base question retrieval not active. Check self.use_guf = True")

        if self .include_overall_task :

            user_content =(f"You are tasked with solving a subtask within an overall task. For example you may be asked to solve a math problem using a specific approach, verify the correctness of a given answer, check that a solution meets some criteria, or any other task. "
            f"Focus on solving your subtask. If you need additional context in order to solve your subtask, you may check the overall task for any necessary additional information. "
            f"You may also possibly be shown other related subtasks and other agents' responses to those subtasks, which will be demarcated by <Subtask assigned to Agent ...> and <Agent ... response> tags respectively. "
            f"You may use this information to help you solve your subtask, for instance by reflecting on possible mistakes made by the other agents' attempts or leveraging good ideas in their responses. "

            f"Here is the overall task: <overall_task> {user_content} </overall_task>\n")


        if self .recursion_round_id >0 :

            include_past_round =_check_all_access (access_list )
            if include_past_round :
                try :

                    prev_history =kwargs ["prev_history"][idx ]if isinstance (kwargs ["prev_history"],list )else kwargs ["prev_history"]

                    prev_subtasks =prev_history ["subtasks"]
                    prev_model_ids =prev_history ["model_ids"]
                    prev_responses =prev_history ["agent_responses"]

                    user_content +=("This overall task was already attempted by a collection of agents, each with assigned subtasks, in a previous round. Below, you will find the subtasks assigned to those agents "
                    "along with their responses. You may also consult this information in order to solve your subtask and the overall task, for instance by reflecting on possible mistakes "
                    "made by the earlier agent attempts or leveraging good ideas in their responses. "
                    )
                    user_content +=("\n\nHere is the previous round's information:\n\n"
                    "PREVIOUS ROUND SUBTASK ASSIGNMENT AND AGENT RESPONSES\n"

                    )
                    for subtask ,model_id ,response in zip (prev_subtasks ,prev_model_ids ,prev_responses ):
                        user_content +=(
                        f"\n<Previous round subtask assigned to Agent {model_id}>{subtask}\n"
                        f"</Previous round subtask assigned to Agent {model_id}>\n"
                        f"\n<Previous round Agent {model_id} response>{response}</Previous round Agent {model_id} response>\n"
                        )
                except Exception as e :
                    print (f"[WARN] Could not add previous round's information: {e}")

        if self .access_history_type =="binary":
            indices =_ascribe_history_binary (history ,access_list )
        elif self .access_history_type =="choose_id":
            indices =_ascribe_history_complex (history ,access_list )
        elif self .access_history_type =="choose_position":
            indices =_ascribe_history_positional_complex (history ,access_list )
        else :
            raise ValueError (f"Invalid access history type: {self.access_history_type}")


        for i in indices :
            if i >=len (history ["subtasks"])or i >=len (history ["model_ids"])or i >=len (history ["agent_responses"]):
                raise ValueError (
                f"Index out of range: {i} while constructing message history")
            prev_subtask =history ["subtasks"][i ]
            model_id =history ["model_ids"][i ]
            response =history ["agent_responses"][i ]

            if response is not None :
                agent_output =response .strip ()

            else :
                agent_output =""
            if agent_output :
                user_content +=(
                f"\n<Subtask assigned to Agent {model_id}>{prev_subtask}"
                f"</Subtask assigned to Agent {model_id}>"
                f"\n<Agent {model_id} response>{agent_output}</Agent {model_id} response>"
                )

        if self .share_agent_subtasks :
            additional_content =(f"\n Here are the subtasks assigned to the other agents in order to solve the overall user question. You may use this information to help you understand how your assigned "
            f"subtask fits into the overall task, and how other agents may use your response in order to solve their subtasks, towards solving the overall user question.  "
            f"The subtasks are as follows: {all_subtasks}")
            user_content +=additional_content 



        user_content +=f"\n\nYour assigned subtask: {subtask}"

        if self .final_agent_knowledge :
            workflow_length =len (history ['model_ids'])

            if len (history ['agent_responses'])==workflow_length -1 :
                final_agent_knowledge =f"As the final agent in the workflow, your response will be used as the final answer to the overall user question. Hence, after working through your subtask, ensure you return the solved user question according to the formatting instructions provided. "
                user_content +=final_agent_knowledge 

        assistant_prompt ="Let me solve this step by step."
        if think_tags :
            assistant_prompt +="<think>"

        if 'claude'in self .available_models [mid ]and self .claude_thinking_budget :

            return [
            {"role":"user","content":user_content }]

        return [
        {"role":"user","content":user_content },
        {"role":"assistant","content":assistant_prompt },
        ]

    def _check_parsing (self ,subtasks ,model_ids ,access_list ,completion ):

        parsing_error_dict ={
        "model_id_error":[],
        "access_list_empty_error":[],
        "completion_unparseable_error":[],
        "router_output_length_mismatch_error":[],
        "invalid_complex_topology_error":[],
        }


        if subtasks ==[]and model_ids ==[]and access_list ==[]:
            parsing_error_dict ["completion_unparseable_error"].append (
            f"Completion unparseable | Completion: {completion}")

            return parsing_error_dict 


        for id in model_ids :
            if not isinstance (id ,int ):
                parsing_error_dict ["model_id_error"].append (
                f"Model ID {id} is non-integer (type={type(id)})")
            elif id <0 or id >=len (self .available_models ):
                parsing_error_dict ["model_id_error"].append (
                f"Model ID {id} is out of range ")


        if not access_list :
            parsing_error_dict ["access_list_empty_error"].append (
            f"Access list is empty | subtasks: {subtasks}, models: {model_ids}, access_list: {access_list}")
        else :

            for agent_access in access_list :
                if self .access_history_type =="binary":
                    if not _is_all_access (agent_access ):
                        if agent_access !=[]:
                            parsing_error_dict ["access_list_empty_error"].append (
                            f"Access list is empty | Invalid agent access: {agent_access}")
                else :
                    valid_access =_check_ctop_access_list (agent_access ,access_list ,self .available_models ,ctop_type =self .access_history_type )
                    if not valid_access :
                        parsing_error_dict ["invalid_complex_topology_error"].append (
                        f"Invalid complex topology | agent access: {access_list}")


        if len (subtasks )!=len (model_ids )or len (subtasks )!=len (access_list ):
            parsing_error_dict ["router_output_length_mismatch_error"].append (f"Router output length mismatch | Completion: {completion}, \n"
            f"subtasks: {len(subtasks)}, models: {len(model_ids)}, access_list: {len(access_list)}")


        if any (parsing_error_dict .values ()):
            return parsing_error_dict 


        return {}

    def _update_counters_log (self ,counters_log ,subtasks ,model_ids ,access_list ):


        counters_log ["total_access_cnt"]=len (model_ids )


        counters_log ["all_access_cnt"]=_check_all_access (access_list )


        for mid in model_ids :
            counters_log ["model_selected_counts"][mid ]+=1 


        if self .access_history_type =="choose_id":
            flat_access_list =[item for elem in access_list for item in (elem if isinstance (elem ,list )else [elem ])]
            for mid in flat_access_list :
                counters_log ["model_visible_counts"][mid ]+=1 

        elif self .access_history_type =="choose_position":
            flat_access_list =[item for elem in access_list for item in (elem if isinstance (elem ,list )else [elem ])]
            for mid in flat_access_list :
                counters_log ["position_visible_counts"][mid ]+=1 


        for pos ,mid in enumerate (model_ids ):
            counters_log ["position_model_counts"][pos ][mid ]+=1 
            counters_log ["total_position_counts"][pos ]+=1 


        counters_log ["chosen_num_routing_steps"][len (model_ids )]+=1 


        counters_log ["avg_subtask_length"]+=sum (len (subtask )
        for subtask in subtasks )/len (subtasks )

        return counters_log 

    def _create_coordination_log_file (self ,function_name :str )->str :

        log_filename =f"{function_name}.json"
        if self .output_dir is not None :
            log_file_path =os .path .join (self .output_dir ,log_filename )
            return log_file_path 
        else :
            return ""

    def _merge_counts (self ,dst :dict ,src :dict ):

        for k ,v in src .items ():
            if isinstance (v ,dict ):
                dst .setdefault (k ,{})
                self ._merge_counts (dst [k ],v )
            else :
                dst [k ]=dst .get (k ,0 )+v 

    def _update_full_log (self ,counters_log ,task_type ):


        self ._merge_counts (self .full_log ["global"],counters_log )


        self ._merge_counts (self .full_log [task_type ],counters_log )

        if self .cache_final_response :

            if counters_log .get ("recursion_round_cnt",0 )>0 :
                self ._merge_counts (self .full_log ["global_recursion"],counters_log )
                self ._merge_counts (self .full_log [f"recursion/{task_type}"],counters_log )

        return self .full_log 

    def _prepare_full_log (self ,full_log ,store_converted_log =False ):

        full_log_converted ={}

        for name ,log_dict in full_log .items ():
            if log_dict ["total_access_cnt"]>0 :
                full_log_converted [f"{name}/all_access_ratio"]=log_dict ["all_access_cnt"]/log_dict ["total_access_cnt"]
                full_log_converted [f"{name}/invalid_access_ratio"]=log_dict ["invalid_access_cnt"]/log_dict ["total_access_cnt"]
                full_log_converted [f"{name}/invalid_ctop_access_cnt"]=log_dict ["invalid_ctop_access_cnt"]/log_dict ["total_access_cnt"]
                full_log_converted [f"{name}/model_id_error_ratio"]=log_dict ["model_id_error_cnt"]/log_dict ["total_access_cnt"]
            if log_dict ["total_question_cnt"]>0 :
                full_log_converted [f"{name}/format_error_ratio"]=log_dict ["format_error_cnt"]/log_dict ["total_question_cnt"]
                full_log_converted [f"{name}/total_correct_ratio"]=log_dict ["total_correct_cnt"]/log_dict ["total_question_cnt"]
                full_log_converted [f"{name}/interrupt_ratio"]=log_dict ["interrupt_cnt"]/log_dict ["total_question_cnt"]
                full_log_converted [f"{name}/avg_subtask_length"]=log_dict ["avg_subtask_length"]/log_dict ["total_question_cnt"]
                full_log_converted [f"{name}/total_cost"]=log_dict ["total_cost"]
                full_log_converted [f"{name}/request_timeout_cnt"]=log_dict ["request_timeout_cnt"]
                full_log_converted [f"{name}/complex_topology_future_error_cnt"]=log_dict ["complex_topology_future_error_cnt"]/log_dict ["total_question_cnt"]
                full_log_converted [f"{name}/avg_format_bonus"]=log_dict ["format_bonus_cnt"]/log_dict ["total_question_cnt"]
                full_log_converted [f"{name}/subtask_misspecification_ratio"]=log_dict ["subtask_misspecification_cnt"]/log_dict ["total_question_cnt"]
                full_log_converted [f"{name}/non_chain_topology_ratio"]=log_dict ["non_chain_topology_cnt"]/log_dict ["total_question_cnt"]

            if log_dict ["non_chain_topology_cnt"]>0 :
                full_log_converted [f"{name}/non_chain_topology_correct_ratio"]=log_dict ["non_chain_topology_correct_cnt"]/log_dict ["non_chain_topology_cnt"]

            if log_dict ["recursion_return_cnt"]>0 :
                full_log_converted [f"{name}/recursion_return_correct_ratio"]=log_dict ["recursion_return_correct_cnt"]/log_dict ["recursion_return_cnt"]

            if log_dict ["recursion_round_cnt"]>0 :
                full_log_converted [f"{name}/recursion_return_ratio"]=log_dict ["recursion_return_cnt"]/log_dict ["recursion_round_cnt"]
                full_log_converted [f"{name}/recursion_correct_ratio"]=log_dict ["recursion_correct_cnt"]/log_dict ["recursion_round_cnt"]


            if log_dict ["recursion_revise_strategy_cnt"]>0 :
                full_log_converted [f"{name}/recursion_revise_strategy_correct_ratio"]=log_dict ["recursion_revise_strategy_correct_cnt"]/log_dict ["recursion_revise_strategy_cnt"]


            for mid ,cnt in log_dict ["model_none_returns_cnt"].items ():
                if log_dict ["total_access_cnt"]>0 :
                    full_log_converted [f"{name}/none_returns/{self.available_models[mid]}"]=cnt 


            for mid ,cnt in log_dict ["model_selected_counts"].items ():
                if log_dict ["total_access_cnt"]>0 :
                    full_log_converted [f"{name}/router_selection_ratio/{self.available_models[mid]}"]=cnt /log_dict ["total_access_cnt"]


            for mid ,cnt in log_dict ["model_visible_counts"].items ():
                if log_dict ["total_access_cnt"]>0 :
                    full_log_converted [f"{name}/router_selection_visibility_ratio/{self.available_models[mid]}"]=cnt /log_dict ["total_access_cnt"]


            for pos ,cnt in log_dict ["position_visible_counts"].items ():
                if log_dict ["total_access_cnt"]>0 :
                    full_log_converted [f"{name}/router_selection_visibility_ratio_pos{pos+1}"]=cnt /log_dict ["total_access_cnt"]


            for pos ,mdict in log_dict ["position_model_counts"].items ():
                tot =log_dict ["total_position_counts"][pos ]
                if tot >0 :
                    for mid ,cnt in mdict .items ():
                        full_log_converted [f"{name}/router_selection_ratio_pos{pos+1}/{self.available_models[mid]}"]=cnt /tot 


            total_chosen_num_routing_steps =sum (log_dict ["chosen_num_routing_steps"].values ())
            if total_chosen_num_routing_steps >0 :
                full_log_converted [f"{name}/chosen_num_routing_steps_ratio"]={
                k :v /total_chosen_num_routing_steps for k ,v in log_dict ["chosen_num_routing_steps"].items ()
                }

        if self .store_full_log :
            self .full_log_converted =full_log_converted 

        return full_log_converted 

    def _print_response (self ,step_log ,completions ,idx ,correct ,include_responses =False ,**kwargs ):
        if correct :
            prefix_str =f"------ Correct completion ------ \n"
        else :
            prefix_str =f"------ Incorrect completion ------ \n"

        if include_responses :
            prefix_str +=f" ------ Full multi-step coordination ------ \n"

        print (prefix_str +
        f"Sequence of chosen models: {[self.available_models[id] for id in step_log['model_ids']]} \n"
        f"Access list: {step_log['access_list']} \n"
        f"Subtasks: {step_log['subtasks']} \n")

    def _critique_subtask (self ,reply :str )->bool :

        return contains_subtask_misspecification (reply )


    def _count_non_chain_topology (self ,access_list :List [List [int ]])->bool :

        if self .access_history_type =="choose_position":
            if len (access_list )==1 or len (access_list )==2 :
                return False 
            if len (access_list )==3 :
                chain1 =access_list ==[[],[0 ],[0 ,1 ]]
                chain2 =access_list ==[[],[0 ],[1 ]]
                if chain1 or chain2 :
                    return False 
                return True 

            if len (access_list )==4 :
                chain1 =access_list ==[[],[0 ],[0 ,1 ],[0 ,1 ,2 ]]
                chain2 =access_list ==[[],[0 ],[1 ],[2 ]]
                if chain1 or chain2 :
                    return False 
                return True 

            if len (access_list )==5 :
                chain1 =access_list ==[[],[0 ],[0 ,1 ],[0 ,1 ,2 ],[0 ,1 ,2 ,3 ]]
                chain2 =access_list ==[[],[0 ],[1 ],[2 ],[3 ]]
                if chain1 or chain2 :
                    return False 
                return True 


        return False 



    def _multi_turn_coordination (self ,prompt ,completion ,idx ,recursion_kwargs =None ,**kwargs ):

        log_file_path =self ._create_coordination_log_file (
        "multi_turn_coordination")



        if recursion_kwargs :
            kwargs ={**kwargs ,**recursion_kwargs }

        task_type =kwargs ["dataset_source"][idx ]if isinstance (kwargs .get (
        "dataset_source",None ),list )else kwargs .get ("dataset_source",None )
        if task_type is None :
            task_type =kwargs ["task_type"][idx ]if isinstance (
            kwargs ["task_type"],list )else kwargs ["task_type"]


        log_data ={
        "timestamp":time .time (),
        "input_parameters":{
        "temperature":self .temperature ,
        "max_tokens":self .max_tokens ,
        },
        "prompt":prompt ,
        "conductor_completion":completion ,
        "task_type":task_type ,
        "coordination_steps":[],
        "final_result":None ,
        "execution_time":None ,
        "reward":None ,
        "log_file_path":log_file_path ,
        }

        start_time =time .time ()


        recursion_round_id =kwargs .get ("recursion_round_id",0 )
        if isinstance (recursion_round_id ,list ):
            recursion_round_id =recursion_round_id [idx ]
        try :
            self .recursion_round_id =int (recursion_round_id )
        except Exception as e :
            print (f"WARNING: Could not parse recursion_round_id: {e}")
            self .recursion_round_id =0 



        history ={
        "subtasks":[],
        "model_ids":[],
        "agent_responses":[],
        "messages":[],
        }


        scorer =copy .copy (self )


        counters_log ={
        "all_access_cnt":0 ,
        "invalid_access_cnt":0 ,
        "model_id_error_cnt":0 ,
        "format_error_cnt":0 ,
        "total_question_cnt":0 ,
        "total_correct_cnt":0 ,
        "total_access_cnt":0 ,
        "interrupt_cnt":0 ,
        "avg_subtask_length":0 ,
        "model_id_error_cnt":0 ,
        "total_cost":0 ,
        "request_timeout_cnt":0 ,
        "format_bonus_cnt":0 ,
        "invalid_ctop_access_cnt":0 ,
        "subtask_misspecification_cnt":0 ,
        "non_chain_topology_cnt":0 ,
        "non_chain_topology_correct_cnt":0 ,
        "complex_topology_future_error_cnt":0 ,
        "recursion_correct_cnt":0 ,
        "recursion_round_cnt":0 ,
        "recursion_return_cnt":0 ,
        "recursion_return_correct_cnt":0 ,
        "recursion_revise_strategy_cnt":0 ,
        "recursion_revise_strategy_correct_cnt":0 ,
        "model_selected_counts":{i :0 for i in range (len (self .available_models ))},
        "model_none_returns_cnt":{i :0 for i in range (len (self .available_models ))},
        "model_visible_counts":{i :0 for i in range (len (self .available_models ))},
        "position_visible_counts":{i :0 for i in range (self .max_routing_steps )},
        "position_model_counts":{pos :{i :0 for i in range (len (self .available_models ))}for pos in range (self .max_routing_steps )},
        "total_position_counts":{pos :0 for pos in range (self .max_routing_steps )},
        "chosen_num_routing_steps":{i +1 :0 for i in range (self .max_routing_steps )},
        }

        local_rewards =[]
        access_list =[]
        final_resp =""
        parsing_error_dict ={}
        counters_log ["total_question_cnt"]+=1 
        if self .recursion_round_id >0 :
            counters_log ["recursion_round_cnt"]+=1 
        non_chain_topology_flag =False 
        recursion_return_flag =False 
        subtask_critique_flag =False 
        for repeat_idx in range (self .score_repeats ):
            try :
                per_step_log ={}

                subtasks ,model_ids ,access_list ,parsing_error_dict =scorer ._parse_completions (
                completion ,model_id_format =scorer .model_id_format )


                if self .recursion_round_id >0 :
                    if subtasks ==[]and model_ids ==[]and access_list ==[]:
                        recursion_return_flag =True 

                if parsing_error_dict :
                    error_msg =next (
                    (msgs [0 ]for k ,msgs in parsing_error_dict .items ()if msgs ))
                    if len (error_msg )>100 :
                        error_msg ="..."+error_msg [-250 :]
                    raise ValueError (f'Format error: {error_msg}')


                history ["subtasks"].extend (subtasks )
                history ["model_ids"].extend (model_ids )

                task_data =setup_task_data (kwargs ,idx )


                for subtask ,mid in zip (subtasks ,model_ids ):
                    msg =scorer ._prepare_agent_messages (subtask =subtask ,access_list =access_list ,history =history ,idx =idx ,
                    repeat_idx =repeat_idx ,user_content_format =scorer .user_content_format ,all_subtasks =subtasks ,mid =mid ,**kwargs )
                    model_name =scorer .available_models [mid ]

                    history ["messages"].append (msg )
                    agent_reply =query_llm (model_name ,messages =msg ,max_tokens =scorer .max_tokens ,
                    temperature =scorer .temperature ,server =scorer .servers .get (model_name ,None ),port =scorer .ports .get (model_name ,None ),
                    gemini_thinking_budget =scorer .gemini_thinking_budget ,claude_thinking_budget =scorer .claude_thinking_budget ,
                    anthropic_platform =scorer .anthropic_platform ,gpt_reasoning_effort =scorer .gpt_reasoning_effort )


                    subtask_critique_flag =scorer ._critique_subtask (agent_reply )

                    if agent_reply is None or agent_reply =="":
                        counters_log ["model_none_returns_cnt"][mid ]+=1 

                    history ["agent_responses"].append (agent_reply )


                if recursion_return_flag :
                        final_resp =kwargs ["prev_final_response"][idx ]if isinstance (kwargs ["prev_final_response"],list )else kwargs ["prev_final_response"]


                else :
                    if self .evaluate_only :
                        final_resp =""
                        for response in reversed (history ["agent_responses"]):
                            if response and response .strip ():
                                final_resp =response 
                                break 

                    else :
                        final_resp =history ["agent_responses"][-1 ]if history ["agent_responses"]else ""


                if self .use_guf :

                    submodule_package_root =os .path .abspath (
                    os .path .join (os .path .dirname (__file__ ),'..','proj_guf'))
                    if submodule_package_root not in sys .path :
                        sys .path .insert (0 ,submodule_package_root )
                    try :
                        correctness_reward =scorer .reward_func (
                        [final_resp ],[task_data ])[0 ]
                    finally :

                        if sys .path [0 ]==submodule_package_root :
                            sys .path .pop (0 )
                else :
                    raise ValueError (
                    "Check self.use_guf. All reward scoring now managed through guf task reward functions")


                cost_bonus ,total_cost =_calculate_cost_bonus (model_ids ,history ["messages"],history ["agent_responses"],
                cost_bonus_weight =self .cost_bonus_weight ,
                llm_names =self .available_models )


                reward =correctness_reward 

                if self .cost_bonus_weight :
                    reward +=cost_bonus 

                if self .format_bonus :
                    reward +=self .format_bonus 



                if repeat_idx ==0 :
                    if not recursion_return_flag :

                        counters_log =scorer ._update_counters_log (
                        counters_log ,subtasks ,model_ids ,access_list )
                        counters_log ["total_cost"]+=total_cost 

                        if self .access_history_type =="choose_position":
                            non_chain_topology_flag =self ._count_non_chain_topology (access_list )
                            if non_chain_topology_flag :
                                counters_log ["non_chain_topology_cnt"]+=1 
                                if correctness_reward :
                                    counters_log ["non_chain_topology_correct_cnt"]+=1 
                                    if self .non_chain_bonus :
                                        reward +=self .non_chain_bonus 

                    if self .format_bonus :
                        counters_log ["format_bonus_cnt"]+=1 
                    if subtask_critique_flag :
                        counters_log ["subtask_misspecification_cnt"]+=1 

                    if self .recursion_round_id >0 :
                        if recursion_return_flag :
                            counters_log ["recursion_return_cnt"]+=1 
                            if correctness_reward :
                                counters_log ["recursion_return_correct_cnt"]+=1 
                        else :
                            counters_log ["recursion_revise_strategy_cnt"]+=1 
                            if correctness_reward :
                                counters_log ["recursion_revise_strategy_correct_cnt"]+=1 

                    if self .debug_log_dir :
                        if self .recursion_round_id >0 :
                            recursion_prev_history =kwargs ["prev_history"][idx ]if isinstance (kwargs ["prev_history"],list )else kwargs ["prev_history"]
                        else :
                            recursion_prev_history ={}

                        self ._write_conversation_log (
                        router_completion =completion ,
                        subtasks =subtasks ,
                        model_ids =model_ids ,
                        access_list =access_list ,
                        messages =history ["messages"],
                        agent_responses =history ["agent_responses"],
                        correct =correctness_reward ,
                        base_question =kwargs ["base_question"][idx ]if isinstance (kwargs .get ("base_question",None ),list )else kwargs .get ("base_question",None ),
                        task_type =task_type ,
                        available_models =self .available_models ,
                        subtask_critique_flag =subtask_critique_flag ,
                        non_chain_topology_flag =non_chain_topology_flag ,
                        recursion_round_id =recursion_round_id ,
                        recursion_return_flag =recursion_return_flag ,
                        recursion_prev_history =recursion_prev_history ,
                        task_data =task_data ,
                        )

            except Exception as e :
                print (f"Error processing completion {idx}: {e}")


                counters_log ["format_error_cnt"]+=1 
                if parsing_error_dict :
                    if parsing_error_dict ["model_id_error"]:
                        counters_log ["model_id_error_cnt"]+=1 
                    if parsing_error_dict ["access_list_empty_error"]:
                        counters_log ["invalid_access_cnt"]+=1 
                    if parsing_error_dict ["invalid_complex_topology_error"]:
                        counters_log ["invalid_ctop_access_cnt"]+=1 
                elif 'timed out'in str (e ).lower ()or 'connection error'in str (e ).lower ()or 'Key limit exceeded'in str (e ).lower ():
                    counters_log ['request_timeout_cnt']+=1 
                elif 'tries to see future'in str (e ).lower ():
                    counters_log ['complex_topology_future_error_cnt']+=1 
                else :
                    print (
                    f"Format failure but empty parsing dict -> Uncaught error: {e}")

                correctness_reward =0.0 
                reward =0.0 

            local_rewards .append (reward )


            per_step_log ={
            'subtasks':subtasks if not parsing_error_dict else None ,
            'model_ids':model_ids if not parsing_error_dict else None ,
            'selected_agents':[self .available_models [mid ]for mid in model_ids ]if not parsing_error_dict else None ,
            'access_list':access_list if not parsing_error_dict else None ,
            'agent_responses':history ["agent_responses"]if not parsing_error_dict else None ,
            'total_turns':len (model_ids )if not parsing_error_dict else None ,
            'parsing_error_dict':parsing_error_dict 
            }

            if correctness_reward :
                counters_log ["total_correct_cnt"]+=1 
                if self .recursion_round_id >0 :
                    counters_log ["recursion_correct_cnt"]+=1 


        log_data ["execution_time"]=time .time ()-start_time 
        log_data ["coordination_steps"].append (per_step_log .copy ())
        log_data ["final_result"]=final_resp 
        log_data ["execution_time"]=time .time ()-start_time 
        log_data ["reward"]=sum (local_rewards )/len (local_rewards )

        return sum (local_rewards )/len (local_rewards ),counters_log ,per_step_log ,idx ,task_type ,log_data ,correctness_reward 

    def __call__ (
    self ,
    prompts ,
    completions ,
    start_think_tag :str ="<think>",
    end_think_tag :str ="</think>",
    start_solution_tag :str ="<answer>",
    end_solution_tag :str ="</answer>",
    max_workers :int =8 ,
    **kwargs ,
    ):

        batch_rewards =[]
        num_items =len (completions )

        for chunk_start in range (0 ,num_items ,self .chunk_size ):
            chunk_end =min (chunk_start +self .chunk_size ,num_items )
            prompts_chunk =prompts [chunk_start :chunk_end ]
            completions_chunk =completions [chunk_start :chunk_end ]
            indices_chunk =range (chunk_start ,chunk_end )

            worker_fn =partial (self ._multi_turn_coordination ,**kwargs )


            recursion_kwargs_list =[]
            for i ,idx in enumerate (indices_chunk ):
                recursion_kwargs ={}
                for k in ("recursion_round_id","prev_history","prev_final_response"):
                    if k in kwargs :
                        v =kwargs [k ]
                        recursion_kwargs [k ]=v [idx ]if isinstance (v ,list )else v 
                recursion_kwargs_list .append (recursion_kwargs )

            with ThreadPoolExecutor (max_workers =min (max_workers ,len (indices_chunk )))as pool :
                for reward ,counters_log ,per_step_log ,idx ,task_type ,log_data ,correctness_reward in pool .map (worker_fn ,prompts_chunk ,completions_chunk ,indices_chunk ,recursion_kwargs_list ):

                    batch_rewards .append (reward )
                    if log_data ["final_result"]is not None :
                        response =log_data ["final_result"]
                    else :
                        response =(
                        "ERROR: Failed to obatin a final response.")
                    if self .cache_final_response :
                        self .final_response_cache .append (response )
                        hist_cache ={
                        "subtasks":per_step_log ["subtasks"],
                        "model_ids":per_step_log ["model_ids"],
                        "access_list":per_step_log ["access_list"],
                        "agent_responses":per_step_log ["agent_responses"],
                        "reward":reward ,
                        "correctness_reward":correctness_reward ,
                        }
                        self .history_cache .append (hist_cache )

                    if reward ==0 and per_step_log ["agent_responses"]is not None :


                        if not per_step_log ["parsing_error_dict"]:
                            if random .random ()<0.2 :
                                self ._print_response (
                                per_step_log ,completions ,idx ,correct =False ,**kwargs )

                    if reward >self .format_bonus +self .non_chain_bonus :
                        if random .random ()<0.1 :

                            self ._print_response (
                            per_step_log ,completions ,idx ,correct =True ,**kwargs )


                    self .full_log =self ._update_full_log (
                    counters_log ,task_type )

                    if self .dump_log_data and log_data ["log_file_path"]:
                        with open (log_data ["log_file_path"],"w")as f :
                            json .dump (log_data ,f ,indent =2 )


            full_log_converted =self ._prepare_full_log (self .full_log )


            if hasattr (self ,"trainer"):
                try :
                    self .trainer .log (full_log_converted )
                except Exception :
                    self .trainer .log_metric (**full_log_converted )

        return batch_rewards 


class CustomGRPOTrainer (GRPOTrainer ,CustomTrainer ):
    def __init__ (
    self ,
    *args ,
    logging_prob =0.0 ,
    **kwargs ):

        GRPOTrainer .__init__ (self ,*args ,**kwargs )
        CustomTrainer .__init__ (
        self ,
        tokenizer =self .processing_class ,
        reward_functions =self .reward_funcs ,
        output_dir =self .args .output_dir ,
        logging_prob =logging_prob ,
        )


    def evaluate (self ,*args ,**kwargs ):
        orig_ngen =self .args .num_generations 
        try :

            self .args .num_generations =1 
            self .num_generations =1 
            self .per_device_eval_batch_size =64 
            results =super ().evaluate (*args ,**kwargs )
        finally :
            self .args .num_generations =orig_ngen 


        if self .reward_funcs :
            rw =self .reward_funcs [0 ]
            print ('\n\n --- Accuracy Statistics: --- \n\n')
            assert rw .full_log ,'No full log found. Full log required for statistics'
            for name ,task_dict in rw .full_log .items ():
                print (
                f"{name} | Accuracy: {task_dict['total_correct_cnt']} / {task_dict['total_question_cnt']} -> {task_dict['total_correct_cnt'] / task_dict['total_question_cnt'] * 100:.3f}% \n")




        return results 


class gufGRPOTrainer (GRPOTrainer ,CustomTrainer ):
    def __init__ (
    self ,
    *args ,
    logging_prob =0.0 ,
    **kwargs ):

        GRPOTrainer .__init__ (self ,*args ,**kwargs )
        CustomTrainer .__init__ (
        self ,
        tokenizer =self .processing_class ,
        reward_functions =self .reward_funcs ,
        output_dir =self .args .output_dir ,
        logging_prob =logging_prob ,
        )

    @profiling_decorator 
    def _calculate_rewards (self ,inputs ,prompts ,completions ,completion_ids_list ):
        device =self .accelerator .device 
        rewards_per_func =torch .zeros (
        len (prompts ),len (self .reward_funcs ),device =device )


        excluded_keys ={"prompt","completion","completion_ids"}


        all_keys =set ().union (*[example .keys ()for example in inputs ])
        keys =[key for key in all_keys if key not in excluded_keys ]


        reward_kwargs ={
        key :[example .get (key ,None )for example in inputs ]
        for key in keys 
        }


        reward_kwargs ["trainer_state"]=self .state 

        for i ,(reward_func ,reward_processing_class ,reward_func_name )in enumerate (
        zip (self .reward_funcs ,self .reward_processing_classes ,
        self .reward_func_names )
        ):
            with profiling_context (self ,reward_func_name ):

                if isinstance (reward_func ,nn .Module ):
                    if is_conversational (inputs [0 ]):
                        messages =[{"messages":p +c }
                        for p ,c in zip (prompts ,completions )]
                        texts =[apply_chat_template (x ,reward_processing_class )[
                        "text"]for x in messages ]
                    else :
                        texts =[p +c for p ,c in zip (prompts ,completions )]
                    reward_inputs =reward_processing_class (
                    text =texts ,return_tensors ="pt",padding =True ,padding_side ="right",add_special_tokens =False 
                    )
                    reward_inputs =super ()._prepare_inputs (reward_inputs )
                    with torch .inference_mode ():
                        rewards_per_func [:,i ]=reward_func (
                        **reward_inputs ).logits [:,0 ]
                else :
                    output_reward_func =reward_func (
                    prompts =prompts ,completions =completions ,completion_ids =completion_ids_list ,**reward_kwargs 
                    )

                    output_reward_func =[
                    reward if reward is not None else torch .nan for reward in output_reward_func ]

                    rewards_per_func [:,i ]=torch .tensor (
                    output_reward_func ,dtype =torch .float32 ,device =device )


        if torch .isnan (rewards_per_func ).all (dim =1 ).any ():
            nan_row_idx =torch .isnan (rewards_per_func ).all (
            dim =1 ).nonzero (as_tuple =True )[0 ][0 ]
            row_reward_kwargs ={key :value [nan_row_idx ]
            for key ,value in reward_kwargs .items ()}
            row_reward_kwargs ["prompt"]=prompts [nan_row_idx ]
            row_reward_kwargs ["completion"]=completions [nan_row_idx ]
            warnings .warn (
            f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
            "Please ensure that at least one reward function returns a valid reward."
            )



        rewards_per_func =gather (rewards_per_func )
        return rewards_per_func 



    def _generate_and_score_completions (
    self ,inputs :list [dict [str ,Union [torch .Tensor ,Any ]]]
    )->dict [str ,Union [torch .Tensor ,Any ]]:


        all_keys =set ().union (*[example .keys ()for example in inputs ])

        inputs =[{key :example .get (key ,None )
        for key in all_keys }for example in inputs ]

        return super ()._generate_and_score_completions (inputs )


    def evaluate (self ,*args ,**kwargs ):
        orig_ngen =self .args .num_generations 
        try :

            self .args .num_generations =1 
            self .num_generations =1 
            self .per_device_eval_batch_size =20 
            self .args .per_device_eval_batch_size =20 
            results =super ().evaluate (*args ,**kwargs )
        finally :
            self .args .num_generations =orig_ngen 


        if self .reward_funcs :
            rw =self .reward_funcs [0 ]
            print ('\n\n --- Accuracy Statistics: --- \n\n')
            assert rw .full_log ,'No full log found. Full log required for statistics'
            for name ,task_dict in rw .full_log .items ():
                if task_dict ['total_question_cnt']>0 :
                    print (
                    f"{name} | Accuracy: {task_dict['total_correct_cnt']} / {task_dict['total_question_cnt']} -> {task_dict['total_correct_cnt'] / task_dict['total_question_cnt'] * 100:.3f}% \n")


                none_counts =task_dict .get ("model_none_returns_cnt",{})
                total_nones =sum (none_counts .values ())if none_counts else 0 
                per_model_str =", ".join (
                f"{rw.available_models[mid]}: {cnt}"for mid ,cnt in none_counts .items ()
                )
                print (f"{name} | None returns per model: {per_model_str}")

                none_percentage =total_nones /task_dict ['total_question_cnt']if task_dict ['total_question_cnt']>0 else 0.0 

                print (f"{name} | Total None returns: {total_nones} / {task_dict['total_question_cnt']} questions -> {none_percentage * 100:.3f}% \n")


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
                        f .write ("\n=== Evaluation Results ===\n")
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

                        f .write ("--- Accuracy Statistics ---\n")
                        for name ,task_dict in rw .full_log .items ():
                            total =task_dict ["total_question_cnt"]
                            correct =task_dict ["total_correct_cnt"]
                            acc =(correct /total *100.0 )if total else 0.0 
                            f .write (f"{name} | Accuracy: {correct} / {total} -> {acc:.3f}%\n")



                            none_counts =task_dict .get ("model_none_returns_cnt",{})
                            total_nones =sum (none_counts .values ())if none_counts else 0 
                            per_model_str =", ".join (
                            f"{rw.available_models[mid]}: {cnt}"for mid ,cnt in none_counts .items ()
                            )
                            f .write (f"{name} | None returns per model: {per_model_str}\n")
                            none_rate =(total_nones /total *100.0 )if total else 0.0 
                            f .write (f"{name} | Total None returns: {total_nones} / {total} questions -> {none_rate:.3f}%\n")

                print (f"Wrote evaluation summary to {out_path}")
            except Exception as e :
                print (f"Failed to write eval summary: {e}")



        return results 
