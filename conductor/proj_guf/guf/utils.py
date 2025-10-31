import asyncio 
import re 
import os ,json ,random 
import tempfile 
import time 
from typing import Optional ,Any 
from typing import Dict ,List ,Tuple 
from tenacity import retry ,stop_after_attempt ,wait_exponential 







from guf .cost import track_cost ,get_cost_summary ,Calculator 

from collections import Counter 
import numpy as np 
from scipy .stats import entropy 
import threading 


from guf .debug_worker_lifecycle import get_worker_logger 


try :
    import fcntl 

    HAS_FCNTL =True 
except ImportError :

    HAS_FCNTL =False 
    import portalocker 

_SPLIT_DIR :str =None 


_WORKER_AGENT_CONFIGS ={}


def configure_split_dir (base_dir :str ):

    global _SPLIT_DIR 
    _SPLIT_DIR =os .path .join (base_dir ,"data_splits")


def set_worker_agent_configs (configs ):

    global _WORKER_AGENT_CONFIGS 
    _WORKER_AGENT_CONFIGS =configs .copy ()if configs else {}


def get_worker_agent_configs ():

    global _WORKER_AGENT_CONFIGS 
    return _WORKER_AGENT_CONFIGS .copy ()


def _resolve_agent_model_and_payload (agent_name :str )->Tuple [str ,dict ]:

    global _WORKER_AGENT_CONFIGS 

    if agent_name in _WORKER_AGENT_CONFIGS :
        config =_WORKER_AGENT_CONFIGS [agent_name ]
        actual_model =config .get ("model_name",agent_name )
        payload =config .get ("payload",{})
        return actual_model ,payload 
    else :

        return agent_name ,{}


def debug_agent_resolution (agent_name :str )->dict :

    global _WORKER_AGENT_CONFIGS 

    actual_model ,payload =_resolve_agent_model_and_payload (agent_name )

    return {
    "input_agent_name":agent_name ,
    "resolved_model_name":actual_model ,
    "payload":payload ,
    "was_resolved":agent_name in _WORKER_AGENT_CONFIGS ,
    "available_configs":list (_WORKER_AGENT_CONFIGS .keys ())if _WORKER_AGENT_CONFIGS else []
    }


def validate_agent_configs (agent_configs :dict )->List [str ]:

    errors =[]

    if not isinstance (agent_configs ,dict ):
        errors .append ("agent_configs must be a dictionary")
        return errors 

    for agent_name ,config in agent_configs .items ():
        if not isinstance (config ,dict ):
            errors .append (f"Config for agent '{agent_name}' must be a dictionary")
            continue 

        if "model_name"not in config :
            errors .append (f"Config for agent '{agent_name}' missing 'model_name' key")

        if "payload"not in config :
            errors .append (f"Config for agent '{agent_name}' missing 'payload' key")
        elif not isinstance (config ["payload"],dict ):
            errors .append (f"Payload for agent '{agent_name}' must be a dictionary")

    return errors 


def log_agent_resolution_summary (agent_names :List [str ],debug :bool =True )->dict :

    global _WORKER_AGENT_CONFIGS 

    summary ={
    "total_agents":len (agent_names ),
    "configured_agents":len (_WORKER_AGENT_CONFIGS ),
    "resolutions":{},
    "unique_models":set (),
    }

    for agent_name in agent_names :
        actual_model ,payload =_resolve_agent_model_and_payload (agent_name )
        summary ["resolutions"][agent_name ]={
        "actual_model":actual_model ,
        "has_payload":bool (payload ),
        "payload_keys":list (payload .keys ())if payload else []
        }
        summary ["unique_models"].add (actual_model )

    summary ["unique_models"]=list (summary ["unique_models"])
    summary ["num_unique_models"]=len (summary ["unique_models"])

    if debug :
        print ("=== AGENT RESOLUTION SUMMARY ===")
        print (f"Total agents: {summary['total_agents']}")
        print (f"Configured agents: {summary['configured_agents']}")
        print (f"Unique models: {summary['num_unique_models']}")
        print ("\nResolution details:")
        for agent ,details in summary ["resolutions"].items ():
            print (f"  {agent} -> {details['actual_model']}")
            if details ["has_payload"]:
                print (f"    Payload: {details['payload_keys']}")
        print ("================================")

    return summary 


def _is_oai_model (model :str )->bool :
    return "gpt"in model .lower ()


def _is_anthropic_model (model :str )->bool :
    return "claude"in model .lower ()


def _is_gemini_model (model :str )->bool :
    return "gemini"in model .lower ()


def _is_deepseek_model (model :str )->bool :
    return "deepseek"in model .lower ()


def _is_llama_model (model :str )->bool :
    return "llama"in model .lower ()


def _is_gemma3_model (model :str )->bool :
    return "gemma3"in model .lower ()


def _is_qwen_model (model :str )->bool :
    return "qwen"in model .lower ()


def _print_retry_message (retry_state ):
    print (
    f"Retrying attempt {retry_state.attempt_number}/15 "
    +f"after error: {retry_state.outcome.exception()}"
    )


def log_api_call_wrapper (original_func ):


    def wrapper (*args ,**kwargs ):
        logger =get_worker_logger ()
        start_time =time .time ()


        model_name =args [0 ]if args else "unknown"

        logger .log_event ("api_call_start",{
        "model":model_name ,
        "function":original_func .__name__ ,
        "worker_pid":os .getpid (),
        "thread_id":threading .get_ident ()if 'threading'in globals ()else "unknown"
        })

        try :
            result =original_func (*args ,**kwargs )
            duration =time .time ()-start_time 
            logger .log_event ("api_call_success",{
            "model":model_name ,
            "function":original_func .__name__ ,
            "duration":duration ,
            "response_length":len (result )if isinstance (result ,str )else 0 ,
            "worker_pid":os .getpid ()
            })
            return result 
        except Exception as e :
            duration =time .time ()-start_time 
            logger .log_event ("api_call_error",{
            "model":model_name ,
            "function":original_func .__name__ ,
            "duration":duration ,
            "error":str (e ),
            "error_type":type (e ).__name__ ,
            "worker_pid":os .getpid ()
            })
            raise 

    return wrapper 


@retry (
stop =stop_after_attempt (5 ),
wait =wait_exponential (multiplier =2 ,min =2 ,max =30 ),
reraise =True ,
before_sleep =_print_retry_message ,
)







































def query_llm (
model :str ,
messages :List [Dict ],
max_tokens :int ,
temperature :float ,
server :str =None ,
port :int =None ,
debug :bool =False ,
together :bool =True ,
**kwargs 
)->str :


    import signal 
    old_alarm =signal .alarm (0 )

    try :

        actual_model ,agent_payload =_resolve_agent_model_and_payload (model )


        merged_kwargs ={**agent_payload ,**kwargs }


        if debug and actual_model !=model :
            print (f"=== DEBUG: Agent Resolution ===")
            print (f"Agent name: {model}")
            print (f"Actual model: {actual_model}")
            if agent_payload :
                print (f"Agent payload: {agent_payload}")
            if kwargs :
                print (f"Additional kwargs: {kwargs}")
            if merged_kwargs !=kwargs :
                print (f"Merged parameters: {merged_kwargs}")
            print ("==============================")


        if debug :
            print ("\n=== DEBUG: MESSAGES SENT TO LLM API ===")
            print (f"Original model parameter: {model}")
            print (f"Resolved model: {actual_model}")
            print (f"Server: {server}, Port: {port}")
            for i ,msg in enumerate (messages ):
                print (f"Message {i + 1} - Role: {msg['role']}")

                content =msg ['content']
                if len (content )>500 :
                    print (f"Content: {content}")
                else :
                    print (f"Content: {content}")
            if merged_kwargs :
                print (f"Model parameters: {merged_kwargs}")
            print ("=======================================\n")

        try :

            result =_query_llm (
            actual_model ,messages ,max_tokens ,temperature ,server ,port ,debug ,together ,**merged_kwargs 
            )
            return result 
        except Exception as e :
            error_msg =f"Failed to query LLM (model: {actual_model}"
            if actual_model !=model :
                error_msg +=f", original: {model}"
            error_msg +=f"): {e}"
            print (error_msg )
            return ""
    finally :

        if old_alarm >0 :
            signal .alarm (old_alarm )


def extract_answer (text :str )->Optional [str ]:

    if not text :
        return None 

    match =re .search (r"<answer>(.*?)</answer>",text ,flags =re .IGNORECASE |re .DOTALL )
    if match :
        return match .group (1 ).strip ()
    return None 


def batch_completion (
model_name :str ,
batch_messages :List [List [Dict ]],
max_tokens :int ,
temperature :float ,
max_concurrency :int ,
server :str =None ,
port :int =None ,
debug :bool =False ,
together :bool =True ,
**kwargs 
)->List [str ]:


    actual_model ,agent_payload =_resolve_agent_model_and_payload (model_name )
    merged_kwargs ={**agent_payload ,**kwargs }

    if debug :
        print (f"=== BATCH COMPLETION DEBUG ===")
        print (f"Input model: {model_name}")
        print (f"Resolved model: {actual_model}")
        print (f"Batch size: {len(batch_messages)}")
        print (f"Max concurrency: {max_concurrency}")
        if agent_payload :
            print (f"Agent payload: {agent_payload}")
        if merged_kwargs !=kwargs :
            print (f"Merged parameters: {merged_kwargs}")
        print ("==============================")

    async def _async_batch_completion ():
        semaphore =asyncio .Semaphore (max_concurrency )

        async def process_with_semaphore (prompt ):
            async with semaphore :
                return await asyncio .to_thread (
                query_llm ,
                actual_model ,
                prompt ,
                max_tokens ,
                temperature ,
                server ,
                port ,
                debug ,
                together ,
                **merged_kwargs 
                )

        tasks =[process_with_semaphore (prompt )for prompt in batch_messages ]
        return await asyncio .gather (*tasks )

    responses =asyncio .run (_async_batch_completion ())


    cost_summary =get_cost_summary ()
    if actual_model in [model_summary ["model"]for model_summary in cost_summary ["models"]]:
        for model_summary in cost_summary ["models"]:
            if model_summary ["model"]==actual_model :
                print (f"\nCumulative usage for {actual_model}"+
                (f" (via {model_name})"if actual_model !=model_name else "")+":")
                print (
                f"Total tokens: {model_summary['total_tokens']} "
                f"(Input: {model_summary['total_input_tokens']}, "
                f"Output: {model_summary['total_output_tokens']})"
                )
                print (f"Total cost: ${model_summary['total_cost']:.6f}")
                break 

    return responses 


def calculate_agent_stats (
episode_agents :list ,
llm_names :list ,
total_agent_usage :dict =None ,
entropy_history :list =None 
)->dict :



    dist =Counter (episode_agents )
    agent_dist ={name :int (dist .get (j ,0 ))for j ,name in enumerate (llm_names )}


    if total_agent_usage is None :
        total_agent_usage ={name :0 for name in llm_names }

    for j ,name in enumerate (llm_names ):
        total_agent_usage [name ]+=int (dist .get (j ,0 ))


    counts =np .array ([dist .get (j ,0 )for j in range (len (llm_names ))])
    if counts .sum ()>0 :
        probs =counts /counts .sum ()
        agent_selection_entropy =float (entropy (probs ,base =2 ))
    else :
        agent_selection_entropy =0.0 


    mean_entropy =float (np .mean (entropy_history ))if entropy_history else None 

    stats ={
    "agent_distribution":agent_dist ,
    "total_agent_usage":dict (total_agent_usage ),
    "agent_selection_entropy":agent_selection_entropy ,
    }

    if mean_entropy is not None :
        stats ["mean_entropy"]=mean_entropy 

    return stats ,total_agent_usage 




class TokenStatisticsTracker :


    _instance =None 

    @classmethod 
    def get_instance (cls ):

        if cls ._instance is None :
            cls ._instance =TokenStatisticsTracker ()
        return cls ._instance 

    def __init__ (self ):

        self .reset ()

    def reset (self ):

        self .router_tokens =[]
        self .agent_input_tokens =[]
        self .agent_output_tokens =[]
        self .total_tokens =[]
        self .num_turns =[]
        self .num_episodes =0 

    def add_episode (self ,token_stats :Dict [str ,int ]):

        if not token_stats :
            return 

        self .router_tokens .append (token_stats .get ("router_tokens",0 ))
        self .agent_input_tokens .append (token_stats .get ("agent_input_tokens",0 ))
        self .agent_output_tokens .append (token_stats .get ("agent_output_tokens",0 ))
        self .total_tokens .append (token_stats .get ("total_tokens",0 ))
        self .num_turns .append (token_stats .get ("num_turns",0 ))
        self .num_episodes +=1 

    def get_stats (self )->Dict [str ,Any ]:

        if self .num_episodes ==0 :
            return {
            "avg_router_tokens":0 ,
            "avg_agent_input_tokens":0 ,
            "avg_agent_output_tokens":0 ,
            "avg_total_tokens":0 ,
            "avg_turns":0 ,
            "total_router_tokens":0 ,
            "total_agent_input_tokens":0 ,
            "total_agent_output_tokens":0 ,
            "total_tokens":0 ,
            "episodes_tracked":0 
            }

        return {
        "avg_router_tokens":float (np .mean (self .router_tokens )),
        "avg_agent_input_tokens":float (np .mean (self .agent_input_tokens )),
        "avg_agent_output_tokens":float (np .mean (self .agent_output_tokens )),
        "avg_total_tokens":float (np .mean (self .total_tokens )),
        "avg_turns":float (np .mean (self .num_turns )),
        "total_router_tokens":int (np .sum (self .router_tokens )),
        "total_agent_input_tokens":int (np .sum (self .agent_input_tokens )),
        "total_agent_output_tokens":int (np .sum (self .agent_output_tokens )),
        "total_tokens":int (np .sum (self .total_tokens )),
        "episodes_tracked":self .num_episodes 
        }


def count_router_tokens (model_name :str ,messages :List [Dict ])->int :

    calculator =Calculator (model_name ,messages )
    return calculator .calculate_input_token_length ()


def count_agent_tokens (model_name :str ,messages :List [Dict ],response :str =None )->Tuple [int ,int ]:


    input_calculator =Calculator (model_name ,messages )
    input_tokens =input_calculator .calculate_input_token_length ()


    output_tokens =0 
    if response :
        output_calculator =Calculator (model_name ,output_sequence_string =response )
        output_tokens =output_calculator .calculate_output_token_length_GPT ()

    return input_tokens ,output_tokens 


def track_episode_tokens (
context :Any ,
router_messages :List [Dict ],
agent_messages :List [Dict ],
agent_response :str ,
agent_model_name :str ,
debug_log_file :Optional [str ]=None 
)->Dict [str ,int ]:


    router_tokens =count_router_tokens (context .router_model_name ,router_messages )


    agent_input_tokens ,agent_output_tokens =count_agent_tokens (
    agent_model_name ,agent_messages ,agent_response 
    )


    if debug_log_file is not None :
        with open (debug_log_file ,"a")as log_f :
            log_f .write (f"Router input tokens: {router_tokens}\n")
            log_f .write (f"Agent input tokens: {agent_input_tokens}\n")
            log_f .write (f"Agent output tokens: {agent_output_tokens}\n")
            log_f .write (f"Total tokens: {router_tokens + agent_input_tokens + agent_output_tokens}\n")

    return {
    "router_tokens":router_tokens ,
    "agent_input_tokens":agent_input_tokens ,
    "agent_output_tokens":agent_output_tokens ,
    "total_tokens":router_tokens +agent_input_tokens +agent_output_tokens 
    }


def extract_token_stats_from_results (results :List [Tuple ])->List [Dict ]:

    token_stats_list =[]

    for result in results :

        if len (result )>=6 and isinstance (result [5 ],dict ):

            if result [0 ]==-1.0 :
                continue 

            token_stats_list .append (result [5 ])

    return token_stats_list 


def aggregate_token_statistics (results :List [Tuple ])->Dict [str ,Any ]:


    token_stats_list =extract_token_stats_from_results (results )


    tracker =TokenStatisticsTracker ()


    tracker .reset ()


    for stats in token_stats_list :
        tracker .add_episode (stats )


    return tracker .get_stats ()


def _split_path (task_name :str ,seed :int ,v_ratio :float ,t_ratio :float )->str :

    split_root =_SPLIT_DIR or os .path .join (os .getcwd (),".data_splits")
    os .makedirs (split_root ,exist_ok =True )
    filename =f"{task_name}_{seed}_v{v_ratio}_t{t_ratio}.json"
    return os .path .join (split_root ,filename )


def get_or_create_indices (
task_name :str ,
dataset_len :int ,
seed :int ,
valid_ratio :float ,
test_ratio :float 
)->Dict [str ,List [int ]]:

    path =_split_path (task_name ,seed ,valid_ratio ,test_ratio )
    lock_path =path +".lock"


    if os .path .exists (path ):
        try :
            with open (path ,"r")as f :
                return json .load (f )
        except (json .JSONDecodeError ,IOError )as e :

            print (f"Warning: Corrupted split file {path}, regenerating... Error: {e}")
            try :
                os .remove (path )
            except :
                pass 


    os .makedirs (os .path .dirname (path ),exist_ok =True )


    if not HAS_FCNTL :

        max_retries =10 
        for retry in range (max_retries ):
            if os .path .exists (path ):
                try :
                    with open (path ,"r")as f :
                        return json .load (f )
                except :
                    time .sleep (0.1 *(retry +1 ))
                    continue 


            try :

                rnd =random .Random (seed )
                ids =list (range (dataset_len ))
                rnd .shuffle (ids )

                n_test =int (dataset_len *test_ratio )
                n_valid =int (dataset_len *valid_ratio )

                split ={
                "test":ids [:n_test ],
                "valid":ids [n_test :n_test +n_valid ],
                "train":ids [n_test +n_valid :]
                }


                temp_fd ,temp_path =tempfile .mkstemp (dir =os .path .dirname (path ),text =True )
                try :
                    with os .fdopen (temp_fd ,'w')as temp_file :
                        json .dump (split ,temp_file )

                    os .replace (temp_path ,path )
                    return split 
                except :
                    try :
                        os .remove (temp_path )
                    except :
                        pass 
                    raise 
            except Exception as e :
                if retry ==max_retries -1 :
                    raise RuntimeError (f"Failed to create split file after {max_retries} retries: {e}")
                time .sleep (0.1 *(retry +1 ))
                continue 


    os .makedirs (os .path .dirname (lock_path ),exist_ok =True )


    max_retries =10 
    retry_delay =0.1 

    for retry in range (max_retries ):
        try :

            with open (lock_path ,'w')as lock_file :
                try :
                    fcntl .flock (lock_file .fileno (),fcntl .LOCK_EX |fcntl .LOCK_NB )


                    if os .path .exists (path ):
                        try :
                            with open (path ,"r")as f :
                                result =json .load (f )
                                fcntl .flock (lock_file .fileno (),fcntl .LOCK_UN )
                                return result 
                        except :

                            pass 


                    rnd =random .Random (seed )
                    ids =list (range (dataset_len ))
                    rnd .shuffle (ids )

                    n_test =int (dataset_len *test_ratio )
                    n_valid =int (dataset_len *valid_ratio )

                    split ={
                    "test":ids [:n_test ],
                    "valid":ids [n_test :n_test +n_valid ],
                    "train":ids [n_test +n_valid :]
                    }


                    temp_fd ,temp_path =tempfile .mkstemp (dir =os .path .dirname (path ),text =True )
                    try :
                        with os .fdopen (temp_fd ,'w')as temp_file :
                            json .dump (split ,temp_file )

                        os .replace (temp_path ,path )
                    except :

                        try :
                            os .remove (temp_path )
                        except :
                            pass 
                        raise 


                    fcntl .flock (lock_file .fileno (),fcntl .LOCK_UN )


                    try :
                        os .remove (lock_path )
                    except :
                        pass 

                    return split 

                except IOError :

                    time .sleep (retry_delay *(retry +1 ))


                    if os .path .exists (path ):
                        try :
                            with open (path ,"r")as f :
                                return json .load (f )
                        except :

                            pass 

        except Exception as e :
            if retry ==max_retries -1 :

                raise RuntimeError (f"Failed to create/read split file after {max_retries} retries: {e}")


    print (f"Warning: Failed to acquire lock, creating splits without synchronization")
    rnd =random .Random (seed )
    ids =list (range (dataset_len ))
    rnd .shuffle (ids )

    n_test =int (dataset_len *test_ratio )
    n_valid =int (dataset_len *valid_ratio )

    return {
    "test":ids [:n_test ],
    "valid":ids [n_test :n_test +n_valid ],
    "train":ids [n_test +n_valid :]
    }