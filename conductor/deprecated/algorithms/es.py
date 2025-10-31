

import os 
import json 
from typing import Dict ,Tuple ,List ,Optional ,Any 
import multiprocessing as mp 
import time 
import numpy as np 
import cma 
from tqdm import tqdm 
import torch 
import re 
import pickle 
import signal 

try :
    import wandb 
    _WANDB_AVAILABLE =True 
except ImportError :
    _WANDB_AVAILABLE =False 

from guf .trainer import (
RouterInfrastructure ,
SVDParameterManager ,
EvaluationManager ,
WorkerContext ,
_worker_process_context 
)
from guf .utils import calculate_agent_stats ,aggregate_token_statistics 


try :
    from guf .debug_worker_lifecycle import get_worker_logger 
    _DEBUG_LOGGING_AVAILABLE =True 
except ImportError :
    _DEBUG_LOGGING_AVAILABLE =False 
    print ("Warning: Debug logging not available. Create debug_worker_lifecycle.py for detailed logs.")


class ParameterApplier :


    @staticmethod 
    def apply_params_to_model (
    model :torch .nn .Module ,
    linear_layer :torch .nn .Linear ,
    svd_weights :Dict [str ,torch .Tensor ],
    flat_params :np .ndarray ,
    use_structured_router :bool =False 
    )->None :

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

                del U ,V ,S ,scaled_S ,new_param 
                torch .cuda .empty_cache ()


        device =linear_layer .weight .device 
        dtype =linear_layer .weight .dtype 
        w_size =linear_layer .weight .numel ()


        if flat_params .shape [0 ]-offset >=w_size :
            w_chunk =flat_params [offset :offset +w_size ]
            w_tensor =torch .from_numpy (w_chunk ).to (device ,dtype )
            linear_layer .weight .data .copy_ (w_tensor .view_as (linear_layer .weight ))
            del w_tensor 
            torch .cuda .empty_cache ()



def _do_eval_cma (args :Tuple )->Tuple :

    global _worker_process_context 


    logger =None 
    if _DEBUG_LOGGING_AVAILABLE :
        try :
            logger =get_worker_logger ()
            logger .log_event ("eval_start",{"args_length":len (args ),"worker_pid":os .getpid ()})
        except Exception as e :
            print (f"Debug logging initialization failed: {e}")


    signal .signal (signal .SIGALRM ,signal .SIG_DFL )
    signal .alarm (0 )

    try :

        if len (args )>=10 :
            (task_id ,split_arg ,flat_params ,svd_weights_cpu ,iteration_idx ,
            eps_explore ,servers_dict ,use_structured_router ,closed_model_config ,
            agent_configs )=args [:10 ]
        elif len (args )>=9 :
            (task_id ,split_arg ,flat_params ,svd_weights_cpu ,iteration_idx ,
            eps_explore ,servers_dict ,use_structured_router ,closed_model_config )=args [:9 ]
            agent_configs =None 
        elif len (args )>=8 :
            (task_id ,split_arg ,flat_params ,svd_weights_cpu ,iteration_idx ,
            eps_explore ,servers_dict ,use_structured_router )=args [:8 ]
            closed_model_config =None 
            agent_configs =None 
        elif len (args )==7 :
            (task_id ,split_arg ,flat_params ,svd_weights_cpu ,iteration_idx ,
            eps_explore ,servers_dict )=args 
            use_structured_router =False 
            closed_model_config =None 
            agent_configs =None 
        else :
            (task_id ,split_arg ,flat_params ,svd_weights_cpu ,iteration_idx ,eps_explore )=args 
            servers_dict =_worker_process_context .servers if _worker_process_context .servers else {}
            use_structured_router =False 
            closed_model_config =None 
            agent_configs =None 

        if logger :
            logger .log_event ("eval_args_parsed",{
            "task_id":task_id ,
            "split":split_arg ,
            "iteration":iteration_idx ,
            "has_closed_config":closed_model_config is not None ,
            "has_agent_configs":agent_configs is not None ,
            "use_structured_router":use_structured_router ,
            "servers_count":len (servers_dict )if servers_dict else 0 
            })


        _worker_process_context .using_closed_models =(closed_model_config is not None )


        use_consultant =getattr (_worker_process_context ,'use_consultant',True )


        if agent_configs :
            from guf .utils import set_worker_agent_configs 
            set_worker_agent_configs (agent_configs )
            if logger :
                logger .log_event ("agent_configs_set",{"agent_count":len (agent_configs )})


        if _worker_process_context .using_closed_models :
            signal .alarm (0 )

            os .environ ["USING_CLOSED_MODELS"]="1"
            if logger :
                logger .log_event ("using_closed_models",{"disabled_timeouts":True })
        else :
            os .environ .pop ("USING_CLOSED_MODELS",None )
            if logger :
                logger .log_event ("using_local_models",{"timeouts_enabled":True })


        if _worker_process_context .router_model is None :
            worker_pid =os .getpid ()


            assigned_gpu =getattr (_worker_process_context ,'assigned_gpu',1 )
            device_str =f"cuda:{assigned_gpu}"

            if logger :
                logger .log_event ("model_init_start",{
                "worker_pid":worker_pid ,
                "assigned_gpu":assigned_gpu 
                })

            print (f"[Worker {worker_pid}] Initializing CMA-ES evaluation components on {device_str}...")


            from transformers import AutoModelForCausalLM ,AutoTokenizer 
            from guf .model_mods .modeling_qwen2 import forward as qwen2_forward 

            try :

                _worker_process_context .router_model =AutoModelForCausalLM .from_pretrained (
                _worker_process_context .router_model_name ,
                torch_dtype =torch .bfloat16 ,
                attn_implementation ="flash_attention_2",
                device_map =device_str ,
                )
                if logger :
                    logger .log_event ("router_model_loaded",{
                    "model_name":_worker_process_context .router_model_name ,
                    "device":device_str 
                    })


                if "qwen"in _worker_process_context .router_model_name .lower ():
                    _worker_process_context .router_model .forward =qwen2_forward .__get__ (
                    _worker_process_context .router_model ,
                    type (_worker_process_context .router_model )
                    )
                    if logger :
                        logger .log_event ("qwen2_patch_applied",{})

                _worker_process_context .tokenizer =AutoTokenizer .from_pretrained (
                _worker_process_context .router_model_name 
                )
                if logger :
                    logger .log_event ("tokenizer_loaded",{})


                consultant_outputs =1 if use_consultant else 0 


                _worker_process_context .linear_layer =torch .nn .Linear (
                in_features =_worker_process_context .router_model .config .hidden_size ,
                out_features =len (_worker_process_context .llm_names )+consultant_outputs ,
                bias =False ,
                ).to (device_str ).to (torch .bfloat16 )

                if logger :
                    logger .log_event ("linear_layer_created",{
                    "output_features":len (_worker_process_context .llm_names )+consultant_outputs ,
                    "use_consultant":use_consultant ,
                    "device":device_str 
                    })


                from guf .run_tasks import create_task 


                valid_ratio =getattr (_worker_process_context ,'valid_ratio',0.5 )
                test_ratio =getattr (_worker_process_context ,'test_ratio',0.2 )


                create_task_args ={
                "task_name":_worker_process_context .task_name ,
                "llm_names":_worker_process_context .llm_names ,
                "max_tokens":_worker_process_context .max_tokens ,
                "temperature":_worker_process_context .temperature ,
                "max_turns":_worker_process_context .max_turns ,
                "servers":servers_dict ,
                "ports":_worker_process_context .ports ,
                "valid_ratio":valid_ratio ,
                "test_ratio":test_ratio ,
                "use_structured_router":use_structured_router ,
                "seed":_worker_process_context .seed ,
                "use_consultant":use_consultant ,
                }

                worker_log_dir =getattr (_worker_process_context ,'log_dir',None )
                if worker_log_dir is None :
                    import tempfile 
                    worker_log_dir =tempfile .mkdtemp (prefix ="task_log_")
                    print (f"Warning: Worker log_dir was None, using temporary directory: {worker_log_dir}")

                create_task_args ["log_dir"]=worker_log_dir 


                if closed_model_config and closed_model_config .get ("model_types")=="closed":
                    together_flags =closed_model_config .get ("together_flags",{})

                    for i ,agent_name in enumerate (_worker_process_context .llm_names ):
                        if agent_configs and agent_name in agent_configs :
                            actual_model =agent_configs [agent_name ]["model_name"]
                        else :
                            actual_model =agent_name 

                        if "deepseek"in actual_model .lower ():
                            create_task_args ["together"]=together_flags .get (actual_model ,True )
                            print (f"[Worker {worker_pid}] Using together={create_task_args['together']} for DeepSeek API model")
                            if logger :
                                logger .log_event ("deepseek_together_config",{"together":create_task_args ['together']})
                            break 


                _worker_process_context .task_instance =create_task (**create_task_args )
                if logger :
                    logger .log_event ("task_created",{"task_name":_worker_process_context .task_name })

                print (f"[Worker {worker_pid}] Task created. Initial data_splits: "
                f"{list(_worker_process_context.task_instance.data_splits.keys()) if hasattr(_worker_process_context.task_instance, 'data_splits') and _worker_process_context.task_instance.data_splits else 'None'}")


                test_split_enabled =hasattr (_worker_process_context ,
                'test_split_enabled')and _worker_process_context .test_split_enabled 
                if test_split_enabled or split_arg =="test":
                    print (f"[Worker {worker_pid}] Loading test split for evaluation")
                    if logger :
                        logger .log_event ("loading_test_split",{})
                    _worker_process_context .task_instance .data_splits =_worker_process_context .task_instance ._load_data (
                    seed =np .random .randint (0 ,100000 ),
                    split ="train",
                    validation =True ,
                    valid_ratio =valid_ratio ,
                    test_split =True ,
                    test_ratio =test_ratio ,
                    )

                print (f"[Worker {worker_pid}] CMA-ES evaluation components initialized on {device_str}.")
                if logger :
                    logger .log_event ("model_init_complete",{
                    "worker_pid":worker_pid ,
                    "device":device_str 
                    })

            except Exception as e :
                if logger :
                    logger .log_event ("model_init_error",{
                    "error":str (e ),
                    "error_type":type (e ).__name__ ,
                    "worker_pid":worker_pid 
                    })
                raise 


        if logger :
            logger .log_event ("applying_parameters",{"flat_params_size":len (flat_params )})

        ParameterApplier .apply_params_to_model (
        _worker_process_context .router_model ,
        _worker_process_context .linear_layer ,
        svd_weights_cpu ,
        flat_params ,
        use_structured_router 
        )


        if split_arg =="test"and (not hasattr (_worker_process_context .task_instance ,'data_splits')or 
        "test"not in _worker_process_context .task_instance .data_splits ):
            worker_pid =os .getpid ()
            print (f"[Worker {worker_pid}] Test split requested but not loaded - loading now")
            if logger :
                logger .log_event ("loading_missing_test_split",{})
            _worker_process_context .task_instance .data_splits =_worker_process_context .task_instance ._load_data (
            seed =np .random .randint (0 ,100000 ),
            split ="train",
            validation =True ,
            valid_ratio =getattr (_worker_process_context ,'valid_ratio',0.5 ),
            test_split =True ,
            test_ratio =getattr (_worker_process_context ,'test_ratio',0.2 )
            )


        debug_log_file =None 
        if _worker_process_context .debug and _worker_process_context .debug_log_dir is not None :
            iter_dir =EvaluationManager ._mk_iter_dir (_worker_process_context .debug_log_dir ,iteration_idx )
            log_filename =f"debug_cma_{task_id}_{split_arg}.txt"
            debug_log_file =os .path .join (iter_dir ,log_filename )
            with open (debug_log_file ,"w")as f :
                f .write (f"CMA-ES debug log for task_id: {task_id}, split: {split_arg}\n")


                has_data_splits =(hasattr (_worker_process_context .task_instance ,'data_splits')and 
                _worker_process_context .task_instance .data_splits is not None )

                if has_data_splits :
                    f .write (f"Task has data_splits: {list(_worker_process_context.task_instance.data_splits.keys())}\n")
                    for k ,v in _worker_process_context .task_instance .data_splits .items ():
                        f .write (f"Split {k} size: {len(v)}\n")
                else :
                    f .write ("Task instance has no data_splits attribute or it's None\n")

                if use_structured_router :
                    f .write ("Using hybrid structured router approach with action layer\n")
                if closed_model_config :
                    f .write ("Using closed model configuration with API models\n")
                if agent_configs :
                    f .write (f"Using custom agent configurations: {list(agent_configs.keys())}\n")

                f .write (f"Consultant feature enabled: {use_consultant}\n")

                if _worker_process_context .using_closed_models :
                    f .write ("Timeout handling: DISABLED for closed models (API calls)\n")
                else :
                    f .write ("Timeout handling: ENABLED for local models\n")


                assigned_gpu =getattr (_worker_process_context ,'assigned_gpu',1 )
                f .write (f"Worker GPU assignment: cuda:{assigned_gpu}\n")


        try :
            if logger :
                logger .log_event ("evaluation_start",{"task_id":task_id ,"split":split_arg })


            if _worker_process_context .using_closed_models :
                signal .alarm (0 )


            obs =_worker_process_context .task_instance .reset (task_id =task_id ,split =split_arg )
            done =False 
            turn_num =1 
            sampled_ids =[]
            previous_agent_id =-1 


            episode_router_tokens =0 
            episode_agent_input_tokens =0 
            episode_agent_output_tokens =0 


            agent_messages_list =[]
            agent_responses_list =[]

            if logger :
                logger .log_event ("episode_loop_start",{"max_turns":_worker_process_context .task_instance .max_turns })

            while not done :
                if logger :
                    logger .log_event ("turn_start",{"turn":turn_num ,"previous_agent":previous_agent_id })


                if _worker_process_context .using_closed_models :
                    signal .alarm (0 )


                if _worker_process_context .debug and debug_log_file is not None :
                    with open (debug_log_file ,"a")as log_f :
                        log_f .write (f"Turn {turn_num}/{_worker_process_context.task_instance.max_turns}\n")

                        if turn_num ==_worker_process_context .task_instance .max_turns :
                            log_f .write ("This is the final allowed turn\n")


                if use_structured_router :
                    router_messages =_worker_process_context .task_instance ._obs 
                else :
                    router_messages =(
                    _worker_process_context .task_instance ._format_router_messages ()
                    if hasattr (_worker_process_context .task_instance ,"_format_router_messages")
                    else obs 
                    )


                if _worker_process_context .debug and debug_log_file is not None :
                    with open (debug_log_file ,"a")as log_f :
                        log_f .write (f"Router prompt: ")
                        for msg in router_messages :

                            log_f .write (f"{msg['role']}: {msg['content']}\n")


                if logger :
                    logger .log_event ("router_action_start",{"turn":turn_num })

                logits =EvaluationManager .get_action (
                _worker_process_context .router_model ,
                _worker_process_context .linear_layer ,
                _worker_process_context .tokenizer ,
                router_messages ,
                True ,
                _worker_process_context .last_token_predict 
                )

                if logger :
                    logger .log_event ("router_action_complete",{"turn":turn_num ,"logits_shape":logits .shape if hasattr (logits ,'shape')else len (logits )})


                if use_consultant :
                    consult_flag =logits [-1 ]
                    logits =logits [:-1 ]
                else :
                    consult_flag =None 


                probs =np .exp (logits -logits .max ())
                probs /=probs .sum ()


                if eps_explore >0 :
                    probs =(1.0 -eps_explore )*probs +eps_explore /len (probs )
                    probs /=probs .sum ()


                predicted_agent_id =np .random .choice (range (len (probs )),p =probs )


                consecutive_selection =(predicted_agent_id ==previous_agent_id and previous_agent_id !=-1 )


                will_use_consultant =False 
                if use_consultant and consult_flag is not None and (not hasattr (_worker_process_context .task_instance ,
                'pending_suggestion')or _worker_process_context .task_instance .pending_suggestion is None ):
                    will_use_consultant =consult_flag >0 

                if logger :
                    logger .log_event ("agent_selection",{
                    "turn":turn_num ,
                    "predicted_agent_id":int (predicted_agent_id ),
                    "consecutive_selection":consecutive_selection ,
                    "will_use_consultant":will_use_consultant if use_consultant else None ,
                    "probs":probs .tolist ()
                    })


                if _worker_process_context .debug and debug_log_file is not None :
                    with open (debug_log_file ,"a")as log_f :
                        top_idx =np .argmax (logits )
                        log_f .write (
                        f"Router response: {logits} (Top index: {top_idx}, "
                        f"Model: {_worker_process_context.llm_names[top_idx]})\n")

                        if use_consultant :
                            log_f .write (f"Consult flag: {consult_flag:.4f}, Will use consultant: {will_use_consultant}\n")
                        if consecutive_selection :
                            log_f .write (f"Consecutive selection detected - will reuse previous response\n")
                        if turn_num ==_worker_process_context .task_instance .max_turns :
                            log_f .write (f"This is the final turn - episode will terminate after this\n")


                if use_structured_router :
                    agent_messages =_worker_process_context .task_instance ._format_agent_messages (predicted_agent_id +1 )
                else :
                    if hasattr (_worker_process_context .task_instance ,"_format_agent_messages"):
                        if use_consultant and will_use_consultant :
                            agent_messages =_worker_process_context .task_instance ._format_consult_messages (
                            predicted_agent_id )
                        else :
                            agent_messages =_worker_process_context .task_instance ._format_agent_messages (predicted_agent_id )
                    else :
                        agent_messages =_worker_process_context .task_instance .messages 


                if not consecutive_selection :
                    agent_messages_list .append (agent_messages )


                if _worker_process_context .debug and debug_log_file is not None :
                    with open (debug_log_file ,"a")as log_f :

                        if use_consultant and will_use_consultant :
                            log_f .write (f"[CONSULTANT PREDICTED] Agent will be used as consultant\n")
                        log_f .write (f"Agent prompt: ")
                        for msg in agent_messages :
                            log_f .write (f"{msg['role']}: {msg['content']}\n")


                agent_model_name =_worker_process_context .llm_names [predicted_agent_id ]

                agent_response =_worker_process_context .task_instance .response if consecutive_selection else None 


                if _worker_process_context .using_closed_models :
                    signal .alarm (0 )


                if logger :
                    logger .log_event ("environment_step_start",{
                    "turn":turn_num ,
                    "agent_model":agent_model_name ,
                    "consecutive":consecutive_selection 
                    })

                if use_structured_router :

                    if consecutive_selection :

                        action ="RETURN"
                    else :

                        action =f"<step {turn_num}>\n<description>\nSolve this problem step by step.\n</description>\n<agent>\n{predicted_agent_id + 1}\n</agent>\n</step {turn_num}>"
                    obs ,reward ,done ,obs_action =_worker_process_context .task_instance .step (action )
                else :

                    if use_consultant :
                        obs ,reward ,done ,obs_action =_worker_process_context .task_instance .step (
                        probs ,sampling =False ,preselected_agent_id =predicted_agent_id ,
                        consult_flag =consult_flag ,
                        )
                    else :
                        obs ,reward ,done ,obs_action =_worker_process_context .task_instance .step (
                        probs ,sampling =False ,preselected_agent_id =predicted_agent_id 
                        )

                if logger :
                    logger .log_event ("environment_step_complete",{
                    "turn":turn_num ,
                    "reward":float (reward ),
                    "done":done 
                    })


                actual_sampled_id =predicted_agent_id 
                sampled_ids .append (actual_sampled_id )


                if not consecutive_selection :
                    agent_response =_worker_process_context .task_instance .response 

                    agent_responses_list .append (agent_response )


                consultant_actually_used =False 
                if use_consultant and hasattr (_worker_process_context .task_instance ,
                'pending_suggestion')and _worker_process_context .task_instance .pending_suggestion is not None :
                    consultant_actually_used =True 


                from guf .utils import track_episode_tokens 
                turn_tokens =track_episode_tokens (
                _worker_process_context ,
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

                if logger :
                    logger .log_event ("turn_complete",{
                    "turn":turn_num ,
                    "agent_id":int (actual_sampled_id ),
                    "consultant_used":consultant_actually_used ,
                    "tokens":turn_tokens ,
                    "done":done 
                    })


                if _worker_process_context .debug and debug_log_file is not None :
                    with open (debug_log_file ,"a")as log_f :
                        log_f .write (
                        f"Router logits : {logits}\n"
                        f"Router probs  : {probs}\n"
                        f"Sampled agent : {_worker_process_context.llm_names[actual_sampled_id]}\n"
                        )


                        if use_consultant :
                            if consultant_actually_used :
                                log_f .write (
                                f"[CONSULTANT USED] Generated suggestion: {_worker_process_context.task_instance.pending_suggestion}\n")
                            elif will_use_consultant :
                                log_f .write (
                                f"[CONSULTANT SKIPPED] Prediction was consultant, but not used (pending_suggestion exists or other reason)\n")


                        if consecutive_selection :
                            log_f .write (f"[OPTIMIZATION] Reused previous response (consecutive selection)\n")
                        else :
                            if use_consultant and consultant_actually_used :
                                log_f .write (
                                f"[CONSULTANT] Response used for suggestion generation:\n{_worker_process_context.task_instance.response}\n")
                            else :
                                log_f .write (f"Agent response:\n{_worker_process_context.task_instance.response}\n")

                        if done :
                            log_f .write ("\n=== EPISODE COMPLETED ===\n")
                            if consecutive_selection :
                                log_f .write ("Reason: Same agent selected consecutively\n")
                            elif turn_num >=_worker_process_context .task_instance .max_turns :
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
                    if _worker_process_context .task_instance .num_turns >0 :
                        avg_tokens_per_turn =(
                        episode_router_tokens +episode_agent_input_tokens +episode_agent_output_tokens )/_worker_process_context .task_instance .num_turns 
                        log_f .write (f"Average Tokens Per Turn: {avg_tokens_per_turn:.2f}\n")
                    log_f .write ("=======================================\n")


            token_stats ={
            "router_tokens":episode_router_tokens ,
            "agent_input_tokens":episode_agent_input_tokens ,
            "agent_output_tokens":episode_agent_output_tokens ,
            "total_tokens":episode_router_tokens +episode_agent_input_tokens +episode_agent_output_tokens ,
            "num_turns":_worker_process_context .task_instance .num_turns 
            }

            if logger :
                logger .log_event ("evaluation_complete",{
                "task_id":task_id ,
                "reward":float (reward ),
                "num_turns":_worker_process_context .task_instance .num_turns ,
                "token_stats":token_stats ,
                "sampled_agents":sampled_ids 
                })

            return (
            reward ,
            _worker_process_context .task_instance .num_turns ,
            obs_action ,
            sampled_ids ,
            _worker_process_context .task_instance .response ,
            token_stats ,
            agent_messages_list ,
            agent_responses_list 
            )

        except Exception as e :
            if logger :
                logger .log_event ("evaluation_error",{
                "error":str (e ),
                "error_type":type (e ).__name__ ,
                "task_id":task_id 
                })
            print (f"ERROR in CMA-ES episode evaluation: {e}")
            import traceback 
            traceback .print_exc ()
            return (-1.0 ,0 ,[],[],"Error occurred",{
            "router_tokens":0 ,
            "agent_input_tokens":0 ,
            "agent_output_tokens":0 ,
            "total_tokens":0 ,
            "num_turns":0 
            },[],[])

    except Exception as e :
        if logger :
            logger .log_event ("eval_critical_error",{
            "error":str (e ),
            "error_type":type (e ).__name__ ,
            "task_id":task_id if 'task_id'in locals ()else "unknown"
            })
        print (f"CRITICAL ERROR in CMA-ES episode evaluation: {e}")
        import traceback 
        traceback .print_exc ()
        return (-1.0 ,0 ,[],[],"Critical error occurred",{
        "router_tokens":0 ,
        "agent_input_tokens":0 ,
        "agent_output_tokens":0 ,
        "total_tokens":0 ,
        "num_turns":0 
        },[],[])


def _evaluate_with_structured_router (
context :WorkerContext ,
task_id :int ,
split :str ,
iter_idx :int ,
eps_explore :float =0.0 ,
debug_log_file :Optional [str ]=None 
)->Tuple :

    import torch 
    import numpy as np 
    from collections import Counter 


    task =context .task_instance 
    task .reset (task_id =task_id ,split =split )


    if not hasattr (task ,'use_structured_router')or not task .use_structured_router :
        if debug_log_file :
            with open (debug_log_file ,"a")as f :
                f .write (f"Warning: Task does not support structured router, falling back to standard evaluation\n")
        return EvaluationManager .evaluate_episode (
        context =context ,
        task_id =task_id ,
        split =split ,
        iter_idx =iter_idx ,
        eps_explore =eps_explore ,
        debug_log_file =debug_log_file 
        )


    agent_selections =[]
    max_turns =task .max_turns 


    if debug_log_file :
        with open (debug_log_file ,"a")as f :
            f .write (f"CMA-ES debug log for task_id: {task_id}, split: {split}\n")
            f .write (f"Using hybrid structured router approach with action layer\n")
            f .write (f"Starting structured evaluation for task {task_id}, split {split}\n")
            f .write (f"Max turns: {max_turns}\n\n")
            f .write (f"Problem description: {task.problem_description}\n\n")


    done =False 
    num_turns =0 
    reward =0.0 

    try :
        while not done and num_turns <max_turns :

            obs =task ._obs 

            if debug_log_file :
                with open (debug_log_file ,"a")as f :
                    f .write (f"\n===== Turn {num_turns + 1} =====\n")
                    f .write (f"--- Router Input ---\n")
                    for message in obs :
                        f .write (f"Role: {message['role']}\n")
                        f .write (f"Content:\n{message['content']}\n\n")



            with torch .no_grad ():
                action =EvaluationManager .get_action (
                context .router_model ,
                context .linear_layer ,
                context .tokenizer ,
                obs ,
                True ,
                context .last_token_predict ,
                )


            logits =action 
            probs =np .exp (logits -logits .max ())
            probs /=probs .sum ()


            if eps_explore >0 :
                probs =(1.0 -eps_explore )*probs +eps_explore /len (probs )
                probs /=probs .sum ()


            agent_id =np .argmax (probs )

            if debug_log_file :
                with open (debug_log_file ,"a")as f :
                    f .write (f"Router logits : {logits}\n")
                    f .write (f"Router probs  : {probs}\n")
                    f .write (f"Sampled agent : {context.llm_names[agent_id]}\n\n")



            text =context .tokenizer .apply_chat_template (
            obs ,
            tokenize =False ,
            add_generation_prompt =True 
            )
            inputs =context .tokenizer (text ,return_tensors ="pt").to (context .router_model .device )


            with torch .no_grad ():
                output_ids =context .router_model .generate (
                **inputs ,
                max_new_tokens =512 ,
                do_sample =True ,
                temperature =0.1 ,
                num_return_sequences =1 ,
                )


            input_length =inputs .input_ids .shape [1 ]
            router_response =context .tokenizer .decode (
            output_ids [0 ][input_length :],
            skip_special_tokens =True ,
            )

            if debug_log_file :
                with open (debug_log_file ,"a")as f :
                    f .write (f"--- Router Generated Description ---\n")
                    f .write (f"{router_response}\n\n")


            description =task ._get_job_description (router_response )
            parsed_agent_id =task ._get_next_agent (router_response )-1 



            modified_response =router_response 


            modified_response =re .sub (
            r'<agent>\s*\d+\s*</agent>',
            f'<agent>\n{agent_id + 1}\n</agent>',
            modified_response 
            )

            if debug_log_file :
                with open (debug_log_file ,"a")as f :
                    f .write (f"--- Modified Router Response ---\n")
                    f .write (f"Using action layer selected agent: {agent_id + 1}\n")
                    f .write (f"{modified_response}\n\n")


            agent_messages =task ._format_agent_messages (agent_id +1 )

            if debug_log_file :
                with open (debug_log_file ,"a")as f :
                    f .write (f"--- Agent Input ---\n")
                    for message in agent_messages :
                        f .write (f"Role: {message['role']}\n")
                        f .write (f"Content:\n{message['content']}\n\n")


            obs ,step_reward ,done ,_ =task .step (modified_response )


            if debug_log_file :
                with open (debug_log_file ,"a")as f :
                    f .write (f"--- Agent Response ---\n")
                    f .write (f"{task.latest_response}\n\n")


            agent_selections .append (agent_id )


            reward =step_reward 

            if debug_log_file :
                with open (debug_log_file ,"a")as f :
                    f .write (f"--- Step Results ---\n")
                    f .write (f"Step done: {done}\n")
                    f .write (f"Current reward: {reward}\n")
                    f .write (f"\n")


            num_turns +=1 


            if num_turns >=max_turns :
                done =True 
                if debug_log_file :
                    with open (debug_log_file ,"a")as f :
                        f .write (f"Reached max_turns ({max_turns}). Ending episode.\n")


        if debug_log_file :
            with open (debug_log_file ,"a")as f :
                f .write (f"\n===== Episode Summary =====\n")
                f .write (f"Episode complete after {num_turns} turns\n")
                f .write (f"Final reward: {reward}\n")
                f .write (f"Agent selections (indices): {agent_selections}\n")


                agent_models =[context .llm_names [idx ]for idx in agent_selections ]
                f .write (f"Agent models: {agent_models}\n")


                if agent_selections :
                    unique_agents =len (set (agent_selections ))
                    agent_counts =Counter (agent_selections )
                    f .write (f"Unique agents used: {unique_agents} out of {len(context.llm_names)}\n")
                    f .write (f"Agent usage counts: {dict(agent_counts)}\n")


                f .write (f"\n===== Final Solution =====\n")
                f .write (f"{task.latest_response}\n")


        if not isinstance (reward ,(int ,float ))or np .isnan (reward ):
            if debug_log_file :
                with open (debug_log_file ,"a")as f :
                    f .write (f"Invalid reward: {reward}. Setting to -1.0\n")
            reward =-1.0 


        gamma =0.9 if split =="train"else 1.0 
        discounted_reward =float (reward )*(gamma **num_turns )
        return (discounted_reward ,task_id ,split ,agent_selections )

    except Exception as e :

        import traceback 
        if debug_log_file :
            with open (debug_log_file ,"a")as f :
                f .write (f"===== ERROR =====\n")
                f .write (f"Error during evaluation: {str(e)}\n")
                f .write (traceback .format_exc ())
                f .write ("\n")


        print (f"Structured evaluation error: {str(e)}")
        return (-1.0 ,-1 ,-1 ,[])

def _calculate_diversity_metrics (agent_ids :list ,num_agents :int )->dict :


    valid_ids =[aid for aid in agent_ids if 0 <=aid <num_agents ]
    if not valid_ids :
        return {
        "entropy":0.0 ,
        "gini_diversity":0.0 ,
        "unique_ratio":0.0 
        }


    counts =np .zeros (num_agents ,dtype =float )
    for aid in valid_ids :
        counts [aid ]+=1.0 


    total =counts .sum ()
    probs =counts /total if total >0 else counts 


    entropy =0.0 
    for p in probs :
        if p >0 :
            entropy -=p *np .log (p )


    gini =1.0 -np .sum (probs **2 )


    unique_ratio =np .count_nonzero (counts )/num_agents 

    return {
    "entropy":float (entropy ),
    "gini_diversity":float (gini ),
    "unique_ratio":float (unique_ratio )
    }

def _init_cma_worker (config :Dict )->None :

    global _worker_process_context 


    logger =None 
    if _DEBUG_LOGGING_AVAILABLE :
        try :
            logger =get_worker_logger ()
            logger .log_event ("worker_init_start",{
            "config_keys":list (config .keys ()),
            "worker_pid":os .getpid ()
            })
        except Exception as e :
            print (f"Debug logging initialization failed in worker init: {e}")

    try :

        _worker_process_context .initialize_from_config (config )


        worker_gpu_assignments =config .get ('worker_gpu_assignments',[1 ])



        import multiprocessing as mp 
        current_process =mp .current_process ()


        worker_index =0 
        if hasattr (current_process ,'_identity')and current_process ._identity :

            worker_index =current_process ._identity [0 ]-1 
        elif hasattr (current_process ,'name')and 'ForkPoolWorker'in current_process .name :

            try :
                worker_index =int (current_process .name .split ('-')[-1 ])-1 
            except (ValueError ,IndexError ):
                worker_index =0 


        if worker_index <len (worker_gpu_assignments ):
            assigned_gpu =worker_gpu_assignments [worker_index ]
        else :

            assigned_gpu =worker_gpu_assignments [worker_index %len (worker_gpu_assignments )]


        _worker_process_context .assigned_gpu =assigned_gpu 

        if logger :
            logger .log_event ("worker_gpu_assigned",{
            "worker_index":worker_index ,
            "assigned_gpu":assigned_gpu ,
            "worker_pid":os .getpid (),
            "process_name":current_process .name if hasattr (current_process ,'name')else "unknown"
            })


        if "agent_configs"in config and config ["agent_configs"]:
            from guf .utils import set_worker_agent_configs 
            set_worker_agent_configs (config ["agent_configs"])

            if logger :
                logger .log_event ("agent_configs_loaded",{
                "agent_count":len (config ["agent_configs"]),
                "agent_names":list (config ["agent_configs"].keys ())
                })


            if config .get ("debug",False ):
                worker_pid =os .getpid ()
                print (f"[Worker {worker_pid}] Agent configurations loaded:")
                for agent_name ,agent_config in config ["agent_configs"].items ():
                    print (f"  {agent_name} -> {agent_config['model_name']}")
                    if agent_config .get ("payload"):
                        print (f"    Payload: {agent_config['payload']}")


        worker_pid =os .getpid ()
        print (f"[Worker {worker_pid}] CMA-ES worker initialized on GPU cuda:{assigned_gpu}")

        if logger :
            logger .log_event ("worker_init_success",{
            "worker_pid":worker_pid ,
            "assigned_gpu":assigned_gpu ,
            "servers":_worker_process_context .servers ,
            "llm_names":_worker_process_context .llm_names 
            })


        if config .get ("debug",False ):
            print (f"[Worker {worker_pid}] GPU assignment: cuda:{assigned_gpu}")
            print (f"[Worker {worker_pid}] Server mappings: {_worker_process_context.servers}")
            print (f"[Worker {worker_pid}] Valid ratio: {config.get('valid_ratio', 0.5)}")
            print (f"[Worker {worker_pid}] Test ratio: {config.get('test_ratio', 0.2)}")
            if 'test_split_enabled'in config :
                print (f"[Worker {worker_pid}] Test split enabled: {config.get('test_split_enabled', False)}")

    except Exception as e :
        if logger :
            logger .log_event ("worker_init_error",{
            "error":str (e ),
            "error_type":type (e ).__name__ ,
            "worker_pid":os .getpid ()
            })
        print (f"Error in worker initialization: {e}")
        raise 


class CMAEvolutionTrainer :


    def __init__ (
    self ,
    infrastructure :RouterInfrastructure ,
    num_iters :int =1000 ,
    test_interval :int =10 ,
    save_interval :int =1 ,
    num_repeats :int =16 ,
    popsize_override :int =0 ,
    sigma0 :float =0.03 ,
    seed :int =42 ,
    num_tests :int =100 ,
    test_size :int =100 ,
    servers :Dict [str ,str ]=None ,
    opt_layer_indices :Optional [List [int ]]=None ,
    diversity_bonus_weight :float =0.0 ,
    cost_bonus_weight :float =0.0 ,
    turn_bonus_weight :float =0.0 ,
    use_structured_router :bool =False ,
    closed_model_config :Optional [Dict ]=None ,
    agent_configs :Optional [Dict ]=None ,
    use_consultant :bool =True ,
    resume_from_training :bool =True ,
    last_token_predict :bool =False ,
    wandb_run =None ,
    ):

        self .infra =infrastructure 
        self .num_iters =num_iters 
        self .test_interval =test_interval 
        self .save_interval =save_interval 
        self .num_repeats =num_repeats 
        self .sigma0 =sigma0 
        self .seed =seed 
        self .num_tests =num_tests 
        self .test_size =test_size 
        self .popsize_override =popsize_override 
        self .servers =servers or {}
        self .opt_layer_indices =opt_layer_indices 
        self .resume_from_training =resume_from_training 
        self .last_token_predict =last_token_predict 


        self .diversity_bonus_weight =diversity_bonus_weight 
        self .cost_bonus_weight =cost_bonus_weight 
        self .turn_bonus_weight =turn_bonus_weight 

        self .use_structured_router =use_structured_router 
        self .closed_model_config =closed_model_config 
        self .agent_configs =agent_configs or {}
        self .use_consultant =use_consultant 
        self .wandb_run =wandb_run 


        self .cumulative_closed_source_cost =0.0 


        self .valid_ratio =getattr (self .infra ,'valid_ratio',0.5 )
        self .test_ratio =getattr (self .infra ,'test_ratio',0.2 )


        if self .diversity_bonus_weight >0 :
            print (f"[CMA-ES] Using diversity bonus with weight: {self.diversity_bonus_weight}")
        if self .cost_bonus_weight >0 :
            print (f"[CMA-ES] Using cost penalty with weight: {self.cost_bonus_weight}")
        if self .turn_bonus_weight >0 :
            print (f"[CMA-ES] Using turn bonus with weight: {self.turn_bonus_weight}")
        if self .use_structured_router :
            print (f"[CMA-ES] Using hybrid structured router approach with action layer and task descriptions")
        if self .closed_model_config :
            print (f"[CMA-ES] Using closed model configuration with API calls")
        if self .agent_configs :
            print (f"[CMA-ES] Using custom agent configurations for {len(self.agent_configs)} agents")


        if self .opt_layer_indices :
            print (f"[CMA-ES] Selectively training layers: {self.opt_layer_indices}")


        if hasattr (self .infra ,'worker_gpu_assignments')and self .infra .worker_gpu_assignments :
            gpu_counts ={}
            for gpu_id in self .infra .worker_gpu_assignments :
                gpu_counts [gpu_id ]=gpu_counts .get (gpu_id ,0 )+1 
            print (f"[CMA-ES] Multi-GPU worker distribution:")
            print (f"  Router: cuda:0")
            for gpu_id ,count in sorted (gpu_counts .items ()):
                print (f"  Workers on cuda:{gpu_id}: {count}")


        self .model_config ,self .num_learnable_params ,self .svd_weights_cpu =self ._setup_svd_info ()

        self .diag_dir =os .path .join (self .infra .log_dir ,"es_diagnostics")
        os .makedirs (self .diag_dir ,exist_ok =True )


        self .ckpt_dir =os .path .join (self .infra .log_dir ,"es_ckpts")
        os .makedirs (self .ckpt_dir ,exist_ok =True )

        self .log_file =os .path .join (self .infra .log_dir ,"es_log.json")
        self .action_weights_file =os .path .join (self .infra .log_dir ,"action_weights_evolution.json")


        self .action_weights_data ={}


        model_save_dir =os .path .join (self .infra .log_dir ,"models")
        os .makedirs (model_save_dir ,exist_ok =True )
        self .best_model_path =os .path .join (model_save_dir ,"best_model.npy")
        self .best_score =-np .inf 
        self .best_solution =None 
        self .best_iter =-1 


        self .log_data =[
        {
        "configs":{
        "task":self .infra .task ,
        "model_name":self .infra .model_name ,
        "llm_names":self .infra .llm_names ,
        "log_dir":self .infra .log_dir ,
        "num_iters":self .num_iters ,
        "test_interval":self .test_interval ,
        "num_repeats":self .num_repeats ,
        "sigma0":self .sigma0 ,
        "seed":self .seed ,
        "num_tests":self .num_tests ,
        "test_size":self .test_size ,
        "opt_layer_indices":self .opt_layer_indices ,
        "diversity_bonus_weight":self .diversity_bonus_weight ,
        "cost_bonus_weight":self .cost_bonus_weight ,
        "turn_bonus_weight":self .turn_bonus_weight ,
        "use_structured_router":self .use_structured_router ,
        "hybrid_approach":self .use_structured_router ,
        "closed_model_config":self .closed_model_config is not None ,
        "valid_ratio":self .valid_ratio ,
        "test_ratio":self .test_ratio ,
        "temperature":self .infra .temperature ,
        "max_tokens":self .infra .max_tokens ,
        "max_turns":self .infra .max_turns ,
        "use_consultant":self .use_consultant ,
        "agent_configs":self .agent_configs ,
        "num_agents":len (self .infra .llm_names ),
        "last_token_predict":self .last_token_predict ,

        "gpu_config":{
        "router_gpu":"cuda:0",
        "worker_gpu_assignments":getattr (self .infra ,'worker_gpu_assignments',[1 ]),
        "total_workers":self .infra .num_workers ,
        }
        }
        }
        ]


        if resume_from_training and os .path .exists (self .log_file ):
            with open (self .log_file ,"r")as f :
                self .log_data =json .load (f )
            print ("[CMA-ES] load the existing log file")


            try :
                best_validation_entry =None 
                for entry in self .log_data :
                    if (entry .get ("type")=="valid"and 
                    entry .get ("is_new_best",False )and 
                    entry .get ("best_score")is not None ):

                        if (best_validation_entry is None or 
                        entry .get ("iter",-1 )>best_validation_entry .get ("iter",-1 )):
                            best_validation_entry =entry 

                if best_validation_entry :
                    self .best_score =best_validation_entry ["best_score"]
                    self .best_iter =best_validation_entry ["best_iter"]


                    if os .path .exists (self .best_model_path ):
                        try :
                            self .best_solution =np .load (self .best_model_path )
                            print (f"[CMA-ES] Restored best model: score={self.best_score:.4f}, iter={self.best_iter}")
                        except Exception as e :
                            print (f"[CMA-ES] Warning: Could not load best model file: {e}")

                            self .best_score =-np .inf 
                            self .best_solution =None 
                            self .best_iter =-1 
                    else :
                        print (
                        f"[CMA-ES] Best model state restored from logs: score={self.best_score:.4f}, iter={self.best_iter}")
                        print (f"[CMA-ES] Model file not found at: {self.best_model_path}")
                else :
                    print ("[CMA-ES] No previous best validation results found in log")

            except Exception as e :
                print (f"[CMA-ES] Warning: Error during resume state restoration: {e}")
                print ("[CMA-ES] Continuing with fresh state")

                pass 


        if not resume_from_training :
            with open (self .log_file ,"w")as f :
                json .dump (self .log_data ,f ,indent =2 )

            with open (self .action_weights_file ,"w")as f :
                json .dump ({},f )
            print ("[CMA-ES] build a fresh log file")

    def _setup_svd_info (self )->Tuple [Dict ,int ,Dict [str ,torch .Tensor ]]:


        svd_weights_disk =SVDParameterManager .load_svd_weights (
        self .infra .model_name ,device ="cpu"
        )

        if self .opt_layer_indices is not None :
            svd_weights_cpu =SVDParameterManager .filter_svd_weights_by_layers (
            svd_weights_disk ,self .opt_layer_indices 
            )
            filtered_count =len (svd_weights_disk )-len (svd_weights_cpu )
            print (f"[CMA-ES] Filtered out {filtered_count} SVD weight components based on layer selection")
            print (f"[CMA-ES] Keeping {len(svd_weights_cpu)} SVD weight components for training")
        else :
            svd_weights_cpu ={k :v .cpu ()for k ,v in svd_weights_disk .items ()}


        num_singular =0 
        for name ,tensor in svd_weights_cpu .items ():
            if name .endswith (".S"):
                num_singular +=tensor .numel ()


        from transformers import AutoConfig 
        model_config =AutoConfig .from_pretrained (self .infra .model_name )
        hidden_size =model_config .hidden_size 
        num_llms =len (self .infra .llm_names )+(1 if self .use_consultant else 0 )
        action_param_count =hidden_size *num_llms 
        num_learnable_params =num_singular +action_param_count 

        if self .use_structured_router :
            print (f"[CMA-ES] Using hybrid structured router approach with action layer")
            print (f"[CMA-ES] Total learnable parameters: {num_learnable_params} (SVD: {num_singular}, Action: {action_param_count})")

        return model_config ,num_learnable_params ,svd_weights_cpu 

    def _extract_action_layer_weights (self ,solution :np .ndarray )->np .ndarray :


        singular_value_count =0 
        for k ,v in self .svd_weights_cpu .items ():
            if k .endswith (".S"):
                singular_value_count +=v .numel ()


        action_weights =solution [singular_value_count :]
        return action_weights 

    def _analyze_action_layer_weights (self ,action_weights :np .ndarray )->Dict [str ,Any ]:


        hidden_dim =action_weights .shape [0 ]//len (self .infra .llm_names )
        weights =action_weights .reshape (len (self .infra .llm_names ),hidden_dim )


        analysis ={
        "mean_per_agent":[float (weights [i ].mean ())for i in range (len (self .infra .llm_names ))],
        "std_per_agent":[float (weights [i ].std ())for i in range (len (self .infra .llm_names ))],
        "l2_norm_per_agent":[float (np .linalg .norm (weights [i ]))for i in range (len (self .infra .llm_names ))],
        "agent_names":[name .split ('/')[-1 ]for name in self .infra .llm_names ],
        }

        return analysis 

    def _get_episode_agent_selections (self ,results :List )->List [List [int ]]:

        episode_selections =[]

        for result in results :
            if len (result )>=4 and result [0 ]!=-1.0 :
                agent_ids =result [3 ]
                episode_selections .append (agent_ids )

        return episode_selections 

    def _calculate_episode_diversity_metrics (self ,episode_selections :List [List [int ]])->Dict [str ,Any ]:

        num_agents =len (self .infra .llm_names )
        num_episodes =len (episode_selections )


        result ={
        "avg_unique_agents_per_episode":0.0 ,
        "single_agent_episode_count":0 ,
        "single_agent_episode_pct":0.0 
        }

        if num_episodes ==0 :
            return result 


        unique_per_episode =[len (set (episode ))for episode in episode_selections ]
        avg_unique =sum (unique_per_episode )/num_episodes 


        single_agent_episodes =sum (1 for unique in unique_per_episode if unique ==1 )

        result ["avg_unique_agents_per_episode"]=float (avg_unique )
        result ["single_agent_episode_count"]=single_agent_episodes 
        result ["single_agent_episode_pct"]=single_agent_episodes /num_episodes *100 if num_episodes >0 else 0 

        return result 

    def _calculate_diversity_bonus (self ,agent_selections :List [int ])->float :

        if not agent_selections or self .diversity_bonus_weight ==0 :
            return 0.0 


        metrics =_calculate_diversity_metrics (agent_selections ,len (self .infra .llm_names ))


        return metrics ["entropy"]*self .diversity_bonus_weight 

    def _calculate_cost_bonus (self ,agent_selections :List [int ],agent_messages_list :List ,agent_responses_list :List )->float :

        if not agent_selections or self .cost_bonus_weight ==0 :
            return 0.0 

        from guf .cost import Calculator 

        total_cost =0.0 
        for i ,agent_id in enumerate (agent_selections ):
            agent_model =self .infra .llm_names [agent_id ]


            if i <len (agent_messages_list )and i <len (agent_responses_list ):
                input_messages =agent_messages_list [i ]
                output_response =agent_responses_list [i ]


                calc =Calculator (agent_model )
                agent_cost =calc .calculate_total_cost (input_messages ,output_response ,form ="formatted")


                agent_cost *=1000 
                total_cost +=agent_cost 

        return -total_cost *self .cost_bonus_weight 

    def _calculate_turn_bonus (self ,agent_selections :List [int ])->float :

        if not agent_selections or self .turn_bonus_weight ==0 :
            return 0.0 

        num_turns =len (agent_selections )
        max_turns =self .infra .max_turns 



        turn_bonus =-((num_turns -1 )/max_turns )/4 *self .turn_bonus_weight 
        return turn_bonus 


    def _calculate_closed_source_costs (self ,results :List =None )->Dict [str ,float ]:

        if not results :
            return {
            "closed_source_iteration_cost_usd":0.0 ,
            "closed_source_cumulative_cost_usd":self .cumulative_closed_source_cost 
            }

        from guf .cost import Calculator 



        CLOSED_LLM_NAMES =[
        "gpt-4o-mini",
        "claude-3-7-sonnet-20250219",
        "gemini-1.5-pro",
        "deepseek-ai/DeepSeek-V3",
        "gpt-4.1",
        "claude-sonnet-4-20250514",
        "gemini-2.5-pro",
        ]

        closed_source_models =set ()
        if self .closed_model_config :

            closed_source_models .update (CLOSED_LLM_NAMES )
        else :

            for model in self .infra .llm_names :
                if model in CLOSED_LLM_NAMES :
                    closed_source_models .add (model )


        iteration_cost =0.0 
        agent_cost_breakdown ={}

        for result in results :

            if len (result )<8 or result [0 ]==-1.0 :
                continue 


            agent_selections =result [3 ]if len (result )>=4 else []
            agent_messages_list =result [6 ]if len (result )>=7 else []
            agent_responses_list =result [7 ]if len (result )>=8 else []


            for i ,agent_id in enumerate (agent_selections ):
                agent_model =self .infra .llm_names [agent_id ]


                if agent_model not in closed_source_models :
                    continue 


                if i <len (agent_messages_list )and i <len (agent_responses_list ):
                    input_messages =agent_messages_list [i ]
                    output_response =agent_responses_list [i ]


                    try :
                        calc =Calculator (agent_model )
                        agent_cost =calc .calculate_total_cost (input_messages ,output_response ,form ="formatted")
                        iteration_cost +=agent_cost 


                        if agent_model not in agent_cost_breakdown :
                            agent_cost_breakdown [agent_model ]={"cost":0.0 ,"count":0 }
                        agent_cost_breakdown [agent_model ]["cost"]+=agent_cost 
                        agent_cost_breakdown [agent_model ]["count"]+=1 

                    except Exception as e :
                        if self .infra .debug :
                            print (f"Warning: Error calculating cost for {agent_model}: {e}")
                        continue 


        if self .infra .debug and iteration_cost >0 :
            print (f"[Cost Tracking] Iteration cost: ${iteration_cost:.6f}")
            for model ,info in agent_cost_breakdown .items ():
                print (f"  {model}: ${info['cost']:.6f} ({info['count']} interactions)")


        self .cumulative_closed_source_cost +=iteration_cost 

        return {
        "closed_source_iteration_cost_usd":float (iteration_cost ),
        "closed_source_cumulative_cost_usd":float (self .cumulative_closed_source_cost )
        }

    def train (self ):


        logger =None 
        if _DEBUG_LOGGING_AVAILABLE :
            try :
                logger =get_worker_logger ()
                logger .log_event ("trainer_init",{
                "num_iters":self .num_iters ,
                "num_agents":len (self .infra .llm_names ),
                "use_closed_models":self .closed_model_config is not None ,
                "gpu_config":getattr (self .infra ,'worker_gpu_assignments',[1 ])
                })
            except Exception as e :
                print (f"Trainer debug logging initialization failed: {e}")


        mp .set_start_method ("spawn",force =True )
        worker_config ={
        "router_model_name":self .infra .model_name ,
        "llm_names":self .infra .llm_names ,
        "debug":self .infra .debug ,
        "debug_log_dir":self .infra .debug_log_dir ,
        "log_dir":self .infra .log_dir ,
        "task_name":self .infra .task ,
        "max_tokens":self .infra .max_tokens ,
        "temperature":self .infra .temperature ,
        "max_turns":self .infra .max_turns ,
        "ports":self .infra .ports ,
        "servers":self .servers ,
        "valid_ratio":self .valid_ratio ,
        "test_ratio":self .test_ratio ,
        "test_split_enabled":True ,
        "seed":self .infra .seed ,
        "agent_configs":self .agent_configs ,
        "use_consultant":self .use_consultant ,
        "cost_bonus_weight":self .cost_bonus_weight ,
        "turn_bonus_weight":self .turn_bonus_weight ,
        "last_token_predict":self .last_token_predict ,

        "worker_gpu_assignments":getattr (self .infra ,'worker_gpu_assignments',[1 ]),
        }

        if logger :
            logger .log_event ("creating_worker_pool",{
            "num_workers":self .infra .num_workers ,
            "gpu_assignments":worker_config ["worker_gpu_assignments"]
            })


        pool =mp .Pool (
        self .infra .num_workers ,
        initializer =_init_cma_worker ,
        initargs =(worker_config ,),
        )

        if logger :
            logger .log_event ("worker_pool_created",{
            "pool_size":self .infra .num_workers ,
            "gpu_assignments":worker_config ["worker_gpu_assignments"]
            })


        x0 =np .zeros (self .num_learnable_params ,dtype =np .float32 )
        pop_size =self .popsize_override if self .popsize_override >0 else int (
        np .ceil (4 +3 *np .log (self .num_learnable_params )))
        print (f"[CMA-ES] #params={self.num_learnable_params}, popsize={pop_size}, sigma0={self.sigma0}")


        start_iter =0 
        latest_ckpt_path =os .path .join (self .ckpt_dir ,"ckpt_latest.pkl")
        if self .resume_from_training and os .path .exists (latest_ckpt_path ):
            self .solver =pickle .load (open (latest_ckpt_path ,'rb'))
            start_iter =self .solver .countiter 
            print (f"[CMA-ES] successfully load the ckpt from {latest_ckpt_path}.")
            print (f"[CMA-ES] start from iter {start_iter}.")
            if logger :
                logger .log_event ("solver_loaded",{"start_iter":start_iter })
        else :
            self .solver =cma .CMAEvolutionStrategy (
            x0 =x0 ,
            sigma0 =self .sigma0 ,
            inopts ={
            "popsize":pop_size ,
            "seed":self .seed if self .seed >0 else 42 ,
            "CMA_diagonal":True ,
            },
            )
            print (f"[CMA-ES] successfully build a fresh cma solver.")
            if logger :
                logger .log_event ("solver_created",{"popsize":pop_size ,"sigma0":self .sigma0 })


        eval_func =_do_eval_cma 


        model_save_dir =os .path .join (self .infra .log_dir ,"models")
        os .makedirs (model_save_dir ,exist_ok =True )
        self .best_model_path =os .path .join (model_save_dir ,"best_model.npy")
        self .best_score =-np .inf 
        self .best_solution =None 
        self .best_iter =-1 


        np_random =np .random .RandomState (seed =self .seed )
        total_agent_usage ={name :0 for name in self .infra .llm_names }

        for i in range (start_iter ,self .num_iters +1 ):
            if logger :
                logger .log_event ("iteration_start",{"iteration":i ,"total_iters":self .num_iters })

            start_time =time .time ()


            solutions =self .solver .ask ()
            train_args =[]
            for sol in solutions :
                task_ids =np_random .randint (0 ,self .infra .train_dataset_size ,size =self .num_repeats )
                for tid in task_ids :

                    train_arg =[
                    int (tid ),
                    "train",
                    sol .astype (np .float32 ),
                    self .svd_weights_cpu ,
                    i ,
                    0.0 ,
                    self .servers ,
                    self .use_structured_router ,
                    ]


                    if self .closed_model_config :
                        train_arg .append (self .closed_model_config )
                    else :
                        train_arg .append (None )


                    train_arg .append (self .agent_configs )

                    train_args .append (tuple (train_arg ))

            if logger :
                logger .log_event ("starting_train_rollouts",{
                "iteration":i ,
                "num_solutions":len (solutions ),
                "total_rollouts":len (train_args )
                })


            results =list (
            tqdm (
            pool .imap (eval_func ,train_args ),
            total =len (train_args ),
            desc =f"[CMA-ES] Iter {i} - Train Rollouts",
            leave =False ,
            )
            )

            if logger :
                successful_results =[r for r in results if r [0 ]!=-1.0 ]
                logger .log_event ("train_rollouts_complete",{
                "iteration":i ,
                "total_results":len (results ),
                "successful_results":len (successful_results ),
                "failed_results":len (results )-len (successful_results )
                })


            train_token_stats =aggregate_token_statistics (results )


            closed_source_costs =self ._calculate_closed_source_costs (results )
            train_token_stats .update (closed_source_costs )


            task_results_by_solution =[[]for _ in solutions ]
            solution_agent_selections =[[]for _ in solutions ]
            solution_episode_selections =[[]for _ in solutions ]
            solution_agent_messages =[[]for _ in solutions ]
            solution_agent_responses =[[]for _ in solutions ]
            for idx ,res in enumerate (results ):
                sol_idx =idx //self .num_repeats 
                task_results_by_solution [sol_idx ].append (res )
                if len (res )>=4 and res [0 ]!=-1.0 :
                    solution_agent_selections [sol_idx ].extend (res [3 ])
                    solution_episode_selections [sol_idx ].append (res [3 ])

                    if len (res )>=8 :
                        solution_agent_messages [sol_idx ].extend (res [6 ])
                        solution_agent_responses [sol_idx ].extend (res [7 ])

            rewards =np .zeros (len (solutions ))
            base_rewards =[]
            diversity_bonuses =[]
            cost_bonuses =[]
            turn_bonuses =[]

            for sol_idx ,sol_res in enumerate (task_results_by_solution ):
                ep_rewards =[r [0 ]for r in sol_res if r [0 ]!=-1.0 ]

                if not ep_rewards :
                    rewards [sol_idx ]=-1.0 
                    base_rewards .append (0.0 )
                    diversity_bonuses .append (0.0 )
                    cost_bonuses .append (0.0 )
                    turn_bonuses .append (0.0 )
                else :
                    base_reward =np .mean (ep_rewards )
                    diversity_bonus =self ._calculate_diversity_bonus (solution_agent_selections [sol_idx ])
                    cost_bonus =self ._calculate_cost_bonus (
                    solution_agent_selections [sol_idx ],
                    solution_agent_messages [sol_idx ],
                    solution_agent_responses [sol_idx ]
                    )
                    turn_bonus =self ._calculate_turn_bonus (
                    solution_agent_selections [sol_idx ],
                    )

                    final_reward =base_reward +diversity_bonus +cost_bonus +turn_bonus 

                    rewards [sol_idx ]=final_reward 
                    base_rewards .append (base_reward )
                    diversity_bonuses .append (diversity_bonus )
                    cost_bonuses .append (cost_bonus )
                    turn_bonuses .append (turn_bonus )

            self .solver .tell (solutions ,-rewards )


            all_agent_ids =[aid for seq in solution_agent_selections for aid in seq ]
            agent_stats ,total_agent_usage =calculate_agent_stats (all_agent_ids ,self .infra .llm_names ,
            total_agent_usage )
            diversity_metrics =_calculate_diversity_metrics (all_agent_ids ,len (self .infra .llm_names ))
            all_episodes =[ep for sub in solution_episode_selections for ep in sub ]
            episode_diversity =self ._calculate_episode_diversity_metrics (all_episodes )


            pop_best =float (np .max (rewards ))
            pop_mean =float (np .mean (rewards ))
            pop_std =float (np .std (rewards ))

            base_mean =float (np .mean (base_rewards ))if base_rewards else 0.0 
            bonus_mean =float (np .mean (diversity_bonuses ))if diversity_bonuses else 0.0 
            cost_bonus_mean =float (np .mean (cost_bonuses ))if cost_bonuses else 0.0 
            turn_bonus_mean =float (np .mean (turn_bonuses ))if turn_bonuses else 0.0 
            iter_time =time .time ()-start_time 
            train_log_entry ={
            "iter":i ,
            "type":"train",
            "pop_best":pop_best ,
            "pop_mean":pop_mean ,
            "pop_std":pop_std ,
            "base_reward_mean":base_mean ,
            "diversity_bonus_mean":bonus_mean ,
            "diversity_bonus_weight":self .diversity_bonus_weight ,
            "cost_bonus_mean":cost_bonus_mean ,
            "cost_bonus_weight":self .cost_bonus_weight ,
            "turn_bonus_mean":turn_bonus_mean ,
            "iter_time_seconds":iter_time ,
            **agent_stats ,
            **diversity_metrics ,
            **episode_diversity ,
            "cma_sigma":float (self .solver .sigma ),
            "token_stats":train_token_stats ,
            }
            self .log_data .append (train_log_entry )

            if logger :
                logger .log_event ("iteration_complete",{
                "iteration":i ,
                "pop_best":pop_best ,
                "pop_mean":pop_mean ,
                "iter_time":iter_time ,
                "successful_episodes":len (all_agent_ids )
                })


            if self .wandb_run is not None and _WANDB_AVAILABLE :

                wandb_log =dict ()
                for key ,value in train_log_entry .items ():
                    if type (value )is not dict :
                        wandb_log [f"train/{key}"]=value 
                    else :
                        for sub_key ,sub_value in value .items ():
                            wandb_log [f"train/{key}/{sub_key}"]=sub_value 

                wandb_log ["train/base_reward"]=base_mean 
                wandb_log ["train/diversity_bonus"]=bonus_mean 
                wandb_log ["train/cost_bonus"]=cost_bonus_mean 
                wandb_log ["train/turn_bonus"]=turn_bonus_mean 

                self .wandb_run .log (wandb_log )



            if i %self .save_interval ==0 :
                iter_ckpt_path =os .path .join (self .ckpt_dir ,f'ckpt_iter_{i}.pkl')
                latest_ckpt_path =os .path .join (self .ckpt_dir ,"ckpt_latest.pkl")
                open (iter_ckpt_path ,'wb').write (self .solver .pickle_dumps ())
                open (latest_ckpt_path ,'wb').write (self .solver .pickle_dumps ())
                print (f"[CMA-ES] Save ckpt at iter {i} into {iter_ckpt_path} and overwrite the ckpt_latest.pkl")


            if i %self .test_interval ==0 :
                if logger :
                    logger .log_event ("validation_start",{"iteration":i })

                current_solution =self .solver .result .xfavorite 
                test_args =[]
                for _ in range (self .num_tests ):
                    vid =np_random .randint (0 ,self .infra .valid_dataset_size )


                    test_arg =[
                    int (vid ),
                    "valid",
                    current_solution .astype (np .float32 ),
                    self .svd_weights_cpu ,
                    i ,
                    0.0 ,
                    self .servers ,
                    self .use_structured_router ,
                    ]


                    if self .closed_model_config :
                        test_arg .append (self .closed_model_config )
                    else :
                        test_arg .append (None )


                    test_arg .append (self .agent_configs )

                    test_args .append (tuple (test_arg ))

                test_results =list (
                tqdm (
                pool .imap (eval_func ,test_args ),
                total =len (test_args ),
                desc =f"[CMA-ES] Iter {i} - Valid Rollouts",
                leave =False ,
                )
                )

                valid_token_stats =aggregate_token_statistics (test_results )


                valid_closed_source_costs =self ._calculate_closed_source_costs (test_results )
                valid_token_stats .update (valid_closed_source_costs )

                valid_agent_ids =[aid for r in test_results if len (r )>=4 and r [0 ]!=-1.0 for aid in r [3 ]]
                valid_agent_stats ,_ =calculate_agent_stats (valid_agent_ids ,self .infra .llm_names )
                valid_diversity =_calculate_diversity_metrics (valid_agent_ids ,len (self .infra .llm_names ))
                valid_episodes =[r [3 ]for r in test_results if len (r )>=4 and r [0 ]!=-1.0 ]
                valid_episode_diversity =self ._calculate_episode_diversity_metrics (valid_episodes )

                test_scores =[r [0 ]for r in test_results if r [0 ]!=-1.0 ]
                if test_scores :
                    test_score =float (np .mean (test_scores ))


                    is_new_best =test_score >self .best_score 


                    iter_model_path =os .path .join (model_save_dir ,f"model_iter_{i}.npy")
                    np .save (iter_model_path ,current_solution )
                    print (f"[CMA-ES] Model saved at iter {i} with validation score: {test_score:.4f}")


                    if is_new_best :
                        self .best_score =test_score 
                        self .best_solution =current_solution .copy ()
                        self .best_iter =i 


                        np .save (self .best_model_path ,self .best_solution )

                        print (f"[CMA-ES] New best model saved at iter {i} with validation score: {self.best_score:.4f}")
                        if logger :
                            logger .log_event ("new_best_model",{
                            "iteration":i ,
                            "score":self .best_score 
                            })


                    valid_log_entry ={
                    "iter":i ,
                    "type":"valid",
                    "test_score":test_score ,
                    "best_score":self .best_score ,
                    "is_new_best":is_new_best ,
                    "best_iter":self .best_iter ,
                    **valid_agent_stats ,
                    **valid_diversity ,
                    **valid_episode_diversity ,
                    "token_stats":valid_token_stats ,
                    }
                    self .log_data .append (valid_log_entry )

                    if logger :
                        logger .log_event ("validation_complete",{
                        "iteration":i ,
                        "score":test_score ,
                        "is_new_best":is_new_best 
                        })


                    if self .wandb_run is not None and _WANDB_AVAILABLE :
                        wandb_valid_log ={
                        "iteration":i ,
                        "valid/score":test_score ,
                        "valid/best_score":self .best_score ,
                        "valid/is_new_best":is_new_best ,
                        "valid/agent_diversity_entropy":valid_diversity .get ("entropy",0.0 ),
                        "valid/total_tokens":valid_token_stats .get ("total_tokens",0 ),
                        "valid/avg_tokens_per_episode":valid_token_stats .get ("avg_tokens_per_episode",0.0 ),
                        }

                        self .wandb_run .log (wandb_valid_log )


            with open (self .log_file ,"w")as f :
                import json 
                json .dump (self .log_data ,f ,indent =2 )


            torch .cuda .empty_cache ()


        if logger :
            logger .log_event ("training_complete",{
            "best_score":self .best_score ,
            "best_iter":self .best_iter ,
            "total_iterations":self .num_iters 
            })

        print (f"[CMA-ES] Training complete. Best validation score: {self.best_score:.4f} at iteration {self.best_iter}")
        if self .best_solution is not None :
            print (f"[CMA-ES] Best model saved to: {self.best_model_path}")

        pool .close ()
        pool .join ()
        return self .log_data 

    def run_test (self ,solution =None ):

        print ("[CMA-ES] Running test evaluation...")


        if solution is None :
            if hasattr (self ,'best_solution')and self .best_solution is not None :
                solution =self .best_solution 
            elif os .path .exists (self .best_model_path ):
                try :
                    solution =np .load (self .best_model_path )
                    print (f"[CMA-ES] Loaded best model from {self.best_model_path}")
                except Exception as e :
                    print (f"[CMA-ES] Error loading best model: {e}")
                    if hasattr (self ,'solver')and hasattr (self .solver ,'result'):
                        solution =self .solver .result .xfavorite 
                        print ("[CMA-ES] Using final model instead")
                    else :
                        print ("[CMA-ES] No model available for testing")
                        return None 
            elif hasattr (self ,'solver')and hasattr (self .solver ,'result'):
                solution =self .solver .result .xfavorite 
                print ("[CMA-ES] No saved best model found, using final model")
            else :
                print ("[CMA-ES] No model available for testing")
                return None 


        worker_config ={
        "router_model_name":self .infra .model_name ,
        "llm_names":self .infra .llm_names ,
        "debug":self .infra .debug ,
        "debug_log_dir":self .infra .debug_log_dir ,
        "task_name":self .infra .task ,
        "max_tokens":self .infra .max_tokens ,
        "temperature":self .infra .temperature ,
        "max_turns":self .infra .max_turns ,
        "ports":self .infra .ports ,
        "servers":self .servers ,
        "valid_ratio":getattr (self .infra ,'valid_ratio',0.5 ),
        "test_ratio":getattr (self .infra ,'test_ratio',0.2 ),
        "test_split_enabled":True ,
        "seed":self .infra .seed ,
        "agent_configs":self .agent_configs ,
        "cost_bonus_weight":self .cost_bonus_weight ,
        "turn_bonus_weight":self .turn_bonus_weight ,
        "use_consultant":self .use_consultant ,
        "last_token_predict":self .last_token_predict ,

        "worker_gpu_assignments":getattr (self .infra ,'worker_gpu_assignments',[1 ]),
        }

        import multiprocessing as mp 
        mp .set_start_method ("spawn",force =True )

        pool =mp .Pool (
        self .infra .num_workers ,
        initializer =_init_cma_worker ,
        initargs =(worker_config ,),
        )


        np_random =np .random .RandomState (seed =self .seed )
        test_args =[]


        num_test_samples =self .test_size 

        for _ in range (num_test_samples ):
            tid =np_random .randint (0 ,self .infra .test_dataset_size )


            test_arg =[
            int (tid ),
            "test",
            solution .astype (np .float32 ),
            self .svd_weights_cpu ,
            -1 ,
            0.0 ,
            self .servers ,
            self .use_structured_router ,
            ]


            if hasattr (self ,'closed_model_config')and self .closed_model_config :
                test_arg .append (self .closed_model_config )
            else :
                test_arg .append (None )


            test_arg .append (self .agent_configs )

            test_args .append (tuple (test_arg ))


        from tqdm import tqdm 
        test_results =list (
        tqdm (
        pool .imap (_do_eval_cma ,test_args ),
        total =len (test_args ),
        desc =f"[CMA-ES] Test Evaluation",
        )
        )
        pool .close ()
        pool .join ()


        test_token_stats =aggregate_token_statistics (test_results )


        test_closed_source_costs =self ._calculate_closed_source_costs (test_results )
        test_token_stats .update (test_closed_source_costs )

        test_agent_ids =[aid for r in test_results if len (r )>=4 and r [0 ]!=-1.0 for aid in r [3 ]]
        test_agent_stats ,_ =calculate_agent_stats (test_agent_ids ,self .infra .llm_names )
        test_diversity =_calculate_diversity_metrics (test_agent_ids ,len (self .infra .llm_names ))
        test_episodes =[r [3 ]for r in test_results if len (r )>=4 and r [0 ]!=-1.0 ]
        test_episode_diversity =self ._calculate_episode_diversity_metrics (test_episodes )

        test_scores =[r [0 ]for r in test_results if r [0 ]!=-1.0 ]
        test_score =float (np .mean (test_scores ))if test_scores else 0.0 


        test_entry ={
        "type":"test",
        "test_score":test_score ,
        "num_samples":len (test_scores ),
        "validation_best_score":self .best_score if hasattr (self ,'best_score')else None ,
        **test_agent_stats ,
        **test_diversity ,
        **test_episode_diversity ,
        "token_stats":test_token_stats ,
        }

        self .log_data .append (test_entry )


        if self .wandb_run is not None and _WANDB_AVAILABLE :
            wandb_test_log ={
            "test/score":test_score ,
            "test/num_samples":len (test_scores ),
            "test/agent_diversity_entropy":test_diversity .get ("entropy",0.0 ),
            "test/total_tokens":test_token_stats .get ("total_tokens",0 ),
            "test/avg_tokens_per_episode":test_token_stats .get ("avg_tokens_per_episode",0.0 ),
            }

            self .wandb_run .log (wandb_test_log )


        print (f"[CMA-ES] Test evaluation complete. Score: {test_score:.4f}")
        with open (self .log_file ,"w")as f :
            import json 
            json .dump (self .log_data ,f ,indent =2 )

        return test_entry 