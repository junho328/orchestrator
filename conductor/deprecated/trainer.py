

import os 
import torch 
import torch .nn as nn 
import numpy as np 
from typing import Dict ,Tuple ,List ,Optional ,Union ,Any 
from dataclasses import dataclass 
from transformers import AutoTokenizer ,AutoModelForCausalLM ,AutoConfig 
from guf .model_mods .modeling_qwen2 import forward as qwen2_forward 
from guf .run_tasks import create_task 
from guf .utils import track_episode_tokens ,configure_split_dir 


@dataclass 
class WorkerContext :



    router_model :Optional [nn .Module ]=None 
    tokenizer :Optional [AutoTokenizer ]=None 
    linear_layer :Optional [nn .Linear ]=None 
    task_instance :Optional [Any ]=None 


    debug :bool =False 
    debug_log_dir :Optional [str ]=None 
    log_dir :Optional [str ]=None 
    task_name :Optional [str ]=None 
    max_tokens :Optional [int ]=None 
    temperature :Optional [float ]=None 
    max_turns :Optional [int ]=None 
    router_model_name :Optional [str ]=None 
    llm_names :Optional [List [str ]]=None 
    servers :Dict [str ,str ]=None 
    ports :Dict [str ,int ]=None 
    test_split_enabled :bool =False 
    valid_ratio :float =0.5 
    test_ratio :float =0.2 
    seed :int =42 
    use_consultant :bool =True 
    last_token_predict :bool =False 
    assigned_gpu :int =1 

    def __post_init__ (self ):

            if self .servers is None :
                self .servers ={}
            if self .ports is None :
                self .ports ={}

    def initialize_from_config (self ,config :Dict [str ,Any ])->None :

        for key ,value in config .items ():
            if hasattr (self ,key ):
                setattr (self ,key ,value )

            elif key =="model_name":
                self .router_model_name =value 

            elif key in ["valid_ratio","test_ratio","use_consultant","log_dir"]:
                setattr (self ,key ,value )


        if not hasattr (self ,'log_dir')or self .log_dir is None :
            import tempfile 
            import os 

            self .log_dir =tempfile .mkdtemp (prefix ="worker_log_")
            print (f"Warning: log_dir was None, using temporary directory: {self.log_dir}")


class SVDParameterManager :


    @staticmethod 
    def compose_model_weights (learnable_params :Dict ,svd_weights :Dict )->Dict :

        composed ={}
        for k ,v in learnable_params .items ():
            if k =="action_layer.weight":
                continue 
            U =svd_weights [k +".U"]
            V =svd_weights [k +".V"]
            S =svd_weights [k +".S"]
            scale =v +1 
            composed [k ]=(U @torch .diag_embed (S *scale )@V .T )*(
            S .sum ()/(S *scale ).sum ()
            )
        return composed 

    @staticmethod 
    @torch .no_grad ()
    def load_model_weights (model :nn .Module ,learnable_params :Dict ,svd_weights :Dict ):

        new_w =SVDParameterManager .compose_model_weights (learnable_params ,svd_weights )
        for k ,v in new_w .items ():
            model .get_parameter (k ).copy_ (v )
        return new_w 

    @staticmethod 
    def backpropagate_gradients (model :nn .Module ,learnable_params :Dict ,svd_weights :Dict ):

        new_w =SVDParameterManager .compose_model_weights (learnable_params ,svd_weights )
        for k in learnable_params :
            if k !="action_layer.weight":
                new_w [k ].backward (model .get_parameter (k ).grad )

    @staticmethod 
    def load_svd_weights (model_name :str ,device :str ="cpu")->Dict [str ,torch .Tensor ]:

        svd_file =os .path .join (
        "decomposed_models",
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
    def get_action (
    model :nn .Module ,
    linear_layer :nn .Module ,
    tokenizer :AutoTokenizer ,
    messages :List [Dict ],
    inference :bool =True ,
    last_token_predict :bool =True ,
    )->torch .Tensor :


        if "qwen3"in model .name_or_path .lower ():
            text =tokenizer .apply_chat_template (
            messages ,tokenize =False ,add_generation_prompt =True ,
            enable_thinking =False 
            )
        else :
            text =tokenizer .apply_chat_template (
            messages ,tokenize =False ,add_generation_prompt =True ,
            )


        tokenized =tokenizer (text ,return_tensors ="pt")
        input_ids =tokenized .input_ids .to (model .device )
        attention_mask =tokenized .attention_mask .to (model .device )

        def _execute_without_generation (input_ids :torch .Tensor ,attention_mask :torch .Tensor ,
        action_layer :nn .Module )->torch .Tensor :

            action =model (input_ids ,attention_mask =attention_mask ,action_layer =action_layer )
            return action .float ().cpu ().numpy ().squeeze ()

        def _execute_with_generation (input_ids :torch .Tensor ,attention_mask :torch .Tensor ,
        action_layer :nn .Module )->torch .Tensor :


            outputs =model .generate (
            input_ids ,
            attention_mask =attention_mask ,
            return_dict_in_generate =True ,
            output_hidden_states =True ,
            max_new_tokens =2048 ,
            )

            last_hidden_state =outputs .hidden_states [-1 ][-1 ][:,-1 ,:]
            return action_layer (last_hidden_state ).float ().cpu ().numpy ().squeeze ()

        if not last_token_predict :
            if inference :
                with torch .no_grad ():
                    return _execute_without_generation (input_ids ,attention_mask ,linear_layer )
            else :
                return _execute_without_generation (input_ids ,attention_mask ,linear_layer )
        else :
            if inference :
                with torch .no_grad ():
                    return _execute_with_generation (input_ids ,attention_mask ,linear_layer )
            else :
                return _execute_with_generation (input_ids ,attention_mask ,linear_layer )

    @staticmethod 
    def _mk_iter_dir (root :str ,it :int )->str :

        path =os .path .join (root ,f"iter_{it}")
        os .makedirs (path ,exist_ok =True )
        return path 

    @staticmethod 
    def evaluate_episode (
    context :WorkerContext ,
    task_id :int ,
    split :str ,
    iter_idx :int ,
    eps_explore :float =0.0 ,
    debug_log_file :Optional [str ]=None 
    )->Tuple :

        try :

            obs =context .task_instance .reset (task_id =task_id ,split =split )
            done =False 
            turn_num =1 
            sampled_ids =[]
            previous_agent_id =-1 


            episode_router_tokens =0 
            episode_agent_input_tokens =0 
            episode_agent_output_tokens =0 

            while not done :

                if context .debug and debug_log_file is not None :
                    with open (debug_log_file ,"a")as log_f :
                        log_f .write (f"Turn {turn_num}/{context.task_instance.max_turns}\n")

                        if turn_num ==context .task_instance .max_turns :
                            log_f .write ("This is the final allowed turn\n")


                router_messages =(
                context .task_instance ._format_router_messages ()
                if hasattr (context .task_instance ,"_format_router_messages")
                else obs 
                )


                if context .debug and debug_log_file is not None :
                    with open (debug_log_file ,"a")as log_f :
                        log_f .write (f"Router prompt: ")
                        for msg in router_messages :
                            log_f .write (f"{msg['role']}: {msg['content']}\n")


                logits =EvaluationManager .get_action (
                context .router_model ,
                context .linear_layer ,
                context .tokenizer ,
                router_messages ,
                True ,
                context .last_token_predict ,
                )


                probs =np .exp (logits -logits .max ())
                probs /=probs .sum ()


                if eps_explore >0 :
                    probs =(1.0 -eps_explore )*probs +eps_explore /len (probs )
                    probs /=probs .sum ()


                predicted_agent_id =np .random .choice (range (len (probs )),p =probs )


                consecutive_selection =(predicted_agent_id ==previous_agent_id and previous_agent_id !=-1 )


                if context .debug and debug_log_file is not None :
                    with open (debug_log_file ,"a")as log_f :
                        top_idx =np .argmax (logits )
                        log_f .write (
                        f"Router response: {logits} (Top index: {top_idx}, "
                        f"Model: {context.llm_names[top_idx]})\n")
                        if consecutive_selection :
                            log_f .write (f"Consecutive selection detected - will reuse previous response\n")
                        if turn_num ==context .task_instance .max_turns :
                            log_f .write (f"This is the final turn - episode will terminate after this\n")


                if hasattr (context .task_instance ,"_format_agent_messages"):
                    agent_messages =context .task_instance ._format_agent_messages (predicted_agent_id )
                else :
                    agent_messages =context .task_instance .messages 


                if context .debug and debug_log_file is not None :
                    with open (debug_log_file ,"a")as log_f :
                        log_f .write (f"Agent prompt: ")
                        for msg in agent_messages :
                            log_f .write (f"{msg['role']}: {msg['content']}\n")


                agent_model_name =context .llm_names [predicted_agent_id ]

                agent_response =context .task_instance .response if consecutive_selection else None 


                obs ,reward ,done ,obs_action =context .task_instance .step (probs ,sampling =False ,
                preselected_agent_id =predicted_agent_id )


                actual_sampled_id =predicted_agent_id 
                sampled_ids .append (actual_sampled_id )


                if not consecutive_selection :
                    agent_response =context .task_instance .response 


                turn_tokens =track_episode_tokens (
                context ,
                router_messages ,
                agent_messages ,
                agent_response ,
                agent_model_name ,
                debug_log_file 
                )


                episode_router_tokens +=turn_tokens ["router_tokens"]
                episode_agent_input_tokens +=turn_tokens ["agent_input_tokens"]
                episode_agent_output_tokens +=turn_tokens ["agent_output_tokens"]

                previous_agent_id =actual_sampled_id 


                if context .debug and debug_log_file is not None :
                    with open (debug_log_file ,"a")as log_f :
                        log_f .write (
                        f"Router logits : {logits}\n"
                        f"Router probs  : {probs}\n"
                        f"Sampled agent : {context.llm_names[actual_sampled_id]}\n"
                        )


                        if consecutive_selection :
                            log_f .write (f"[OPTIMIZATION] Reused previous response (consecutive selection)\n")
                        else :
                            log_f .write (f"Agent response:\n{context.task_instance.response}\n")

                        if done :
                            log_f .write ("\n=== EPISODE COMPLETED ===\n")
                            if consecutive_selection :
                                log_f .write ("Reason: Same agent selected consecutively\n")
                            elif turn_num >=context .task_instance .max_turns :
                                log_f .write ("Reason: Maximum turns reached\n")
                            log_f .write (f"Final reward: {reward}\n")

                        log_f .write ("\n")


                turn_num +=1 


            if debug_log_file is not None :
                with open (debug_log_file ,"a")as log_f :
                    log_f .write ("\n===== TOKEN STATISTICS SUMMARY =====\n")
                    log_f .write (f"Episode Router Tokens: {episode_router_tokens}\n")
                    log_f .write (f"Episode Agent Input Tokens: {episode_agent_input_tokens}\n")
                    log_f .write (f"Episode Agent Output Tokens: {episode_agent_output_tokens}\n")
                    log_f .write (
                    f"Total Episode Tokens: {episode_router_tokens + episode_agent_input_tokens + episode_agent_output_tokens}\n")
                    if context .task_instance .num_turns >0 :
                        avg_tokens_per_turn =(
                        episode_router_tokens +episode_agent_input_tokens +episode_agent_output_tokens )/context .task_instance .num_turns 
                        log_f .write (f"Average Tokens Per Turn: {avg_tokens_per_turn:.2f}\n")
                    log_f .write ("=======================================\n")


            token_stats ={
            "router_tokens":episode_router_tokens ,
            "agent_input_tokens":episode_agent_input_tokens ,
            "agent_output_tokens":episode_agent_output_tokens ,
            "total_tokens":episode_router_tokens +episode_agent_input_tokens +episode_agent_output_tokens ,
            "num_turns":context .task_instance .num_turns 
            }


            return (
            reward ,
            context .task_instance .num_turns ,
            obs_action ,
            sampled_ids ,
            context .task_instance .response ,
            token_stats 
            )

        except Exception as e :
            print (f"ERROR in episode evaluation: {e}")
            import traceback 
            traceback .print_exc ()

            return (-1.0 ,0 ,[],[],"Error occurred",{
            "router_tokens":0 ,
            "agent_input_tokens":0 ,
            "agent_output_tokens":0 ,
            "total_tokens":0 ,
            "num_turns":0 
            })



_worker_process_context =WorkerContext ()

def _init_worker (config :Dict )->None :

    global _worker_process_context 
    _worker_process_context .initialize_from_config (config )


def _do_eval (args :Tuple )->Tuple :

    global _worker_process_context 


    if len (args )==6 :
        task_id ,split_arg ,model_state_dict ,linear_layer_state_dict ,servers_dict ,iter_idx =args 
        eps_explore =0.0 
    elif len (args )==7 :
        (task_id ,split_arg ,model_state_dict ,linear_layer_state_dict ,
        servers_dict ,iter_idx ,eps_explore )=args 
    else :
        raise ValueError ("do_eval expects either 6 or 7 arguments.")


    debug_log_file =None 
    if _worker_process_context .debug and _worker_process_context .debug_log_dir is not None :
        iter_dir =EvaluationManager ._mk_iter_dir (_worker_process_context .debug_log_dir ,iter_idx )
        log_filename =f"debug_{task_id}_{split_arg}.txt"
        debug_log_file =os .path .join (iter_dir ,log_filename )
        with open (debug_log_file ,"w")as f :
            f .write (f"Debug log for task_id: {task_id}, split: {split_arg}\n")


    if _worker_process_context .router_model is None :
        worker_pid =os .getpid ()
        assigned_gpu =getattr (_worker_process_context ,'assigned_gpu',1 )
        device_str =f"cuda:{assigned_gpu}"

        print (f"[Worker {worker_pid}] Initializing evaluation components on {device_str}...")


        _worker_process_context .router_model =AutoModelForCausalLM .from_pretrained (
        _worker_process_context .router_model_name ,
        torch_dtype =torch .bfloat16 ,
        attn_implementation ="flash_attention_2",
        device_map =device_str ,
        )



        if "qwen"in _worker_process_context .router_model_name .lower ():
            _worker_process_context .router_model .forward =qwen2_forward .__get__ (
            _worker_process_context .router_model ,
            type (_worker_process_context .router_model )
            )


        _worker_process_context .tokenizer =AutoTokenizer .from_pretrained (
        _worker_process_context .router_model_name 
        )


        _worker_process_context .linear_layer =nn .Linear (
        in_features =_worker_process_context .router_model .config .hidden_size ,
        out_features =len (_worker_process_context .llm_names ),
        bias =False ,
        ).to (device_str ).to (torch .bfloat16 )


        _worker_process_context .task_instance =create_task (
        _worker_process_context .task_name ,
        llm_names =_worker_process_context .llm_names ,
        seed =_worker_process_context .seed ,
        max_tokens =_worker_process_context .max_tokens ,
        temperature =_worker_process_context .temperature ,
        max_turns =_worker_process_context .max_turns ,
        servers =servers_dict ,
        ports =_worker_process_context .ports ,
        valid_ratio =_worker_process_context .valid_ratio ,
        test_ratio =_worker_process_context .test_ratio ,
        log_dir =getattr (_worker_process_context ,'log_dir',None ),
        )

        print (f"[Worker {worker_pid}] Evaluation components initialized on {device_str}.")


    _worker_process_context .router_model .load_state_dict (model_state_dict )
    _worker_process_context .linear_layer .load_state_dict (linear_layer_state_dict )


    if split_arg =="test"and (not hasattr (_worker_process_context .task_instance ,'data_splits')or 
    _worker_process_context .task_instance .data_splits is None or 
    "test"not in _worker_process_context .task_instance .data_splits ):
        worker_pid =os .getpid ()
        print (f"[Worker {worker_pid}] Test split requested but not loaded - loading now")
        _worker_process_context .task_instance .data_splits =_worker_process_context .task_instance ._load_data (
        seed =np .random .randint (0 ,10000 ),
        split ="train",
        validation =True ,
        valid_ratio =getattr (_worker_process_context ,'valid_ratio',0.5 ),
        test_split =True ,
        test_ratio =getattr (_worker_process_context ,'test_ratio',0.2 )
        )
        if debug_log_file is not None :
            with open (debug_log_file ,"a")as f :
                f .write (f"Loaded test split with {len(_worker_process_context.task_instance.data_splits.get('test', []))} samples\n")


    return EvaluationManager .evaluate_episode (
    context =_worker_process_context ,
    task_id =task_id ,
    split =split_arg ,
    iter_idx =iter_idx ,
    eps_explore =eps_explore ,
    debug_log_file =debug_log_file 
    )


class RouterInfrastructure :


    def __init__ (
    self ,
    task :str ,
    model_name :str ,
    llm_names :List [str ],
    log_dir :str ,
    temperature :float =0.8 ,
    max_tokens :int =512 ,
    max_turns :int =5 ,
    servers :Union [str ,Dict [str ,str ]]=None ,
    ports :Dict [str ,int ]=None ,
    num_workers :int =8 ,
    debug :bool =False ,
    debug_log_dir :Optional [str ]=None ,
    eval_workers :int =2 ,
    test_ratio :float =0.2 ,
    valid_ratio :float =0.5 ,
    seed :int =42 ,
    configure_splits :bool =True ,
    max_samples :int =-1 ,
    use_consultant :bool =True ,
    worker_gpu_assignments :Optional [List [int ]]=None ,
    ):

        self .task =task 
        self .model_name =model_name 
        self .llm_names =llm_names 
        self .log_dir =log_dir 
        self .temperature =temperature 
        self .max_tokens =max_tokens 
        self .max_turns =max_turns 
        self .num_workers =num_workers 
        self .debug =debug 
        self .eval_workers =eval_workers 
        self .test_ratio =test_ratio 
        self .valid_ratio =valid_ratio 
        self .seed =seed 
        self .max_samples =max_samples 
        self .use_consultant =use_consultant 


        self .worker_gpu_assignments =worker_gpu_assignments or [1 ]*num_workers 

        if debug and worker_gpu_assignments :
            print (f"Worker GPU assignments: {worker_gpu_assignments}")
            gpu_counts ={}
            for gpu_id in worker_gpu_assignments :
                gpu_counts [gpu_id ]=gpu_counts .get (gpu_id ,0 )+1 
            for gpu_id ,count in gpu_counts .items ():
                print (f"  GPU {gpu_id}: {count} workers")


        self .debug_log_dir =debug_log_dir 
        os .makedirs (log_dir ,exist_ok =True )
        configure_split_dir (self .log_dir )

        if debug and debug_log_dir is None :
            self .debug_log_dir =os .path .join (log_dir ,"debug_logs")
            os .makedirs (self .debug_log_dir ,exist_ok =True )
            if configure_splits :
                configure_split_dir (self .log_dir )


        self .servers =self ._setup_server_mapping (servers )
        self .ports =ports if ports is not None else {}


        self ._initialize_dataset_sizes (
        test_ratio =self .test_ratio ,
        valid_ratio =self .valid_ratio 
        )


        self .param_manager =SVDParameterManager ()

    def _setup_server_mapping (self ,servers :Union [str ,Dict [str ,str ]])->Dict [str ,str ]:

        server_map ={}
        if isinstance (servers ,str ):
            server_list =servers .split (",")
            if len (server_list )==1 :
                for m in self .llm_names :
                    server_map [m ]=server_list [0 ].strip ()
            elif len (server_list )==len (self .llm_names ):
                for i ,m in enumerate (self .llm_names ):
                    server_map [m ]=server_list [i ].strip ()
            else :
                raise ValueError ("Server count mismatch.")
        elif isinstance (servers ,dict ):
            server_map =servers .copy ()
        return server_map 

    def _initialize_dataset_sizes (
    self ,
    test_ratio :float =None ,
    valid_ratio :float =None 
    ):


        test_ratio =test_ratio if test_ratio is not None else self .test_ratio 
        valid_ratio =valid_ratio if valid_ratio is not None else self .valid_ratio 


        temp_task =create_task (
        self .task ,
        seed =self .seed ,
        llm_names =self .llm_names ,
        max_tokens =self .max_tokens ,
        temperature =self .temperature ,
        max_turns =self .max_turns ,
        servers =self .servers ,
        ports =self .ports ,
        valid_ratio =valid_ratio ,
        test_ratio =test_ratio ,
        max_samples =self .max_samples ,
        )


        temp_task .data_splits =temp_task ._load_data (
        seed =self .seed ,
        split ="train",
        validation =True ,
        valid_ratio =valid_ratio ,
        test_split =True ,
        test_ratio =test_ratio 
        )

        self .train_dataset_size =len (temp_task .data_splits ["train"])
        self .valid_dataset_size =len (temp_task .data_splits ["valid"])
        self .test_dataset_size =len (temp_task .data_splits ["test"])

    def dprint (self ,*args ,**kwargs ):

        if self .debug :
            print (*args ,**kwargs )

    def initialize_models (self ,layer_indices :Optional [List [int ]]=None ):


        svd_weights =SVDParameterManager .load_svd_weights (
        self .model_name ,device ="cuda:0"
        )


        if layer_indices is not None :
            svd_weights =SVDParameterManager .filter_svd_weights_by_layers (
            svd_weights ,layer_indices 
            )
            if self .debug :
                print (f"Filtered SVD weights to include only layers: {layer_indices}")
                print (f"Remaining SVD weight components: {len(svd_weights)}")


        model_config =AutoConfig .from_pretrained (self .model_name )
        model =AutoModelForCausalLM .from_pretrained (
        self .model_name ,
        torch_dtype =torch .bfloat16 ,
        attn_implementation ="flash_attention_2",
        device_map ="cuda:0",
        )


        if "qwen"in self .model_name .lower ():
            model .forward =qwen2_forward .__get__ (model ,type (model ))


        tokenizer =AutoTokenizer .from_pretrained (self .model_name )


        linear_layer =nn .Linear (
        in_features =model_config .hidden_size ,
        out_features =len (self .llm_names ),
        bias =False ,
        ).to ("cuda:0").to (torch .bfloat16 )

        return model ,tokenizer ,linear_layer ,svd_weights 


    def compose_model_weights (self ,learnable_params ,svd_weights ):

        return SVDParameterManager .compose_model_weights (learnable_params ,svd_weights )

    def load_model_weights (self ,model ,learnable_params ,svd_weights ):

        return SVDParameterManager .load_model_weights (model ,learnable_params ,svd_weights )

    def backpropagate_gradients (self ,model ,learnable_params ,svd_weights ):

        return SVDParameterManager .backpropagate_gradients (model ,learnable_params ,svd_weights )


    def get_action (self ,model ,linear_layer ,tokenizer ,messages ,inference =True ,last_token_predict =False ):

        return EvaluationManager .get_action (model ,linear_layer ,tokenizer ,messages ,inference ,last_token_predict )


    do_eval =staticmethod (_do_eval )