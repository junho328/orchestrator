

import os 
import time 
import signal 
import threading 
import multiprocessing as mp 
from enum import Enum 
from typing import Dict ,List ,Any ,Tuple ,Optional ,Union 
from dataclasses import dataclass 
import traceback 
import numpy as np 
import torch 
import atexit 

class JobType (Enum ):
    TRAINING_EPISODE ="training_episode"
    CODE_EXECUTION ="code_execution"
    ROUTER_INFERENCE ="router_inference"

@dataclass 
class JobResult :

    success :bool 
    result :Any 
    error :Optional [str ]=None 
    execution_time :float =0.0 
    worker_pid :Optional [int ]=None 

class UnifiedJobManager :


    _instance =None 
    _lock =threading .Lock ()

    def __new__ (cls ):
        if cls ._instance is None :
            with cls ._lock :
                if cls ._instance is None :
                    cls ._instance =super ().__new__ (cls )
        return cls ._instance 

    def __init__ (self ):
        if hasattr (self ,'_initialized'):
            return 
        self ._initialized =True 
        self .pool =None 
        self .num_workers =None 
        self .worker_config =None 
        self ._cleanup_registered =False 

    def initialize (self ,num_workers :int ,worker_config :Dict ):

        if self .pool is not None :
            print ("[JobManager] Pool already initialized, skipping")
            return 

        self .num_workers =num_workers 
        self .worker_config =worker_config .copy ()

        try :

            try :
                mp .set_start_method ("spawn",force =True )
            except RuntimeError :

                pass 


            self .pool =mp .Pool (
            num_workers ,
            initializer =_init_unified_worker ,
            initargs =(worker_config ,),
            )


            if not self ._cleanup_registered :
                atexit .register (self .cleanup )
                self ._cleanup_registered =True 

            print (f"[JobManager] Initialized unified pool with {num_workers} workers")

        except Exception as e :
            print (f"[JobManager] Failed to initialize pool: {e}")
            raise 

    def submit_training_job (self ,
    task_id :int ,
    split :str ,
    flat_params :np .ndarray ,
    svd_weights_cpu :Dict ,
    iteration_idx :int ,
    eps_explore :float ,
    servers_dict :Dict ,
    use_structured_router :bool ,
    closed_model_config :Optional [Dict ]=None ,
    agent_configs :Optional [Dict ]=None )->mp .pool .AsyncResult :

        if self .pool is None :
            raise RuntimeError ("JobManager not initialized. Call initialize() first.")

        job_data ={
        'task_id':task_id ,
        'split':split ,
        'flat_params':flat_params ,
        'svd_weights_cpu':svd_weights_cpu ,
        'iteration_idx':iteration_idx ,
        'eps_explore':eps_explore ,
        'servers_dict':servers_dict ,
        'use_structured_router':use_structured_router ,
        'closed_model_config':closed_model_config ,
        'agent_configs':agent_configs ,
        }

        return self .pool .apply_async (
        _execute_unified_job ,
        (JobType .TRAINING_EPISODE ,job_data )
        )

    def _is_running_in_worker (self )->bool :

        try :
            import multiprocessing as mp 
            current_process =mp .current_process ()


            if hasattr (current_process ,'name'):
                process_name =current_process .name 
                if ('SpawnPoolWorker'in process_name or 
                'ForkPoolWorker'in process_name or 
                'PoolWorker'in process_name ):
                    return True 


            global _worker_initialized 
            if _worker_initialized :
                return True 

            return False 

        except Exception :

            return False 

    def submit_code_execution_job (self ,
    sample :Dict ,
    generation :str ,
    timeout :int ,
    debug :bool =False ,
    using_closed_models :bool =False )->mp .pool .AsyncResult :



        class MockAsyncResult :
            def __init__ (self ,result ):
                self ._result =result 

            def get (self ,timeout =None ):
                return self ._result 

            def ready (self ):
                return True 

            def successful (self ):
                return True 



        if self ._is_running_in_worker ():
            if debug :
                print ("[JobManager] Detected nested job submission, using direct execution")


            try :

                import sys 
                current_module =sys .modules [__name__ ]

                if hasattr (current_module ,'_code_executor')and current_module ._code_executor is not None :
                    code_executor =current_module ._code_executor 
                    result ,metadata =code_executor .check_correctness (
                    sample ,generation ,timeout ,debug ,using_closed_models 
                    )
                    return MockAsyncResult ((result ,metadata ))
                else :

                    from guf .tasks .livecodebench_direct_executor import DirectCodeExecutor 
                    direct_executor =DirectCodeExecutor ()
                    result ,metadata =direct_executor .check_correctness (
                    sample ,generation ,timeout ,debug ,using_closed_models 
                    )
                    return MockAsyncResult ((result ,metadata ))

            except Exception as e :
                if debug :
                    print (f"[JobManager] Direct execution failed: {e}")


        if self .pool is None :
            raise RuntimeError ("JobManager not initialized. Call initialize() first.")

        job_data ={
        'sample':sample ,
        'generation':generation ,
        'timeout':timeout ,
        'debug':debug ,
        'using_closed_models':using_closed_models ,
        }

        return self .pool .apply_async (
        _execute_unified_job ,
        (JobType .CODE_EXECUTION ,job_data )
        )

    def submit_batch_jobs (self ,job_type :JobType ,job_data_list :List [Dict ])->List [mp .pool .AsyncResult ]:

        if self .pool is None :
            raise RuntimeError ("JobManager not initialized. Call initialize() first.")

        futures =[]
        for job_data in job_data_list :
            future =self .pool .apply_async (
            _execute_unified_job ,
            (job_type ,job_data )
            )
            futures .append (future )

        return futures 

    def get_pool_status (self )->Dict [str ,Any ]:

        if self .pool is None :
            return {"status":"not_initialized"}

        try :

            return {
            "status":"active",
            "num_workers":self .num_workers ,
            "pool_object":str (self .pool ),
            }
        except Exception as e :
            return {
            "status":"error",
            "error":str (e )
            }

    def cleanup (self ):

        if self .pool is not None :
            try :
                print ("[JobManager] Terminating unified pool...")
                self .pool .terminate ()
                self .pool .join ()
                self .pool =None 
                print ("[JobManager] Unified pool cleaned up successfully")
            except Exception as e :
                print (f"[JobManager] Error during cleanup: {e}")
                self .pool =None 

    def force_cleanup (self ):

        if self .pool is not None :
            try :
                print ("[JobManager] Force terminating unified pool...")
                self .pool .terminate ()

                self .pool =None 
                print ("[JobManager] Unified pool force cleaned up")
            except Exception as e :
                print (f"[JobManager] Error during force cleanup: {e}")
                self .pool =None 


_job_manager =UnifiedJobManager ()

def get_job_manager ()->UnifiedJobManager :

    return _job_manager 


_worker_context =None 
_code_executor =None 
_worker_initialized =False 


def _init_unified_worker (config :Dict ):

    global _worker_context ,_code_executor ,_worker_initialized 

    if _worker_initialized :
        return 

    worker_pid =os .getpid ()
    logger =None 

    try :

        try :
            from guf .debug_worker_lifecycle import get_worker_logger 
            logger =get_worker_logger ()
            logger .log_event ("worker_init_start",{
            "config_keys":list (config .keys ()),
            "worker_pid":worker_pid 
            })
        except Exception :

            pass 


        from guf .trainer import WorkerContext 


        _worker_context =WorkerContext ()
        _worker_context .initialize_from_config (config )


        try :
            from guf .tasks .livecodebench_direct_executor import DirectCodeExecutor 
            _code_executor =DirectCodeExecutor ()
        except ImportError :
            print (f"[Worker {worker_pid}] Warning: DirectCodeExecutor not available, code execution will be limited")
            _code_executor =None 


        if config .get ("agent_configs"):
            try :
                from guf .utils import set_worker_agent_configs 
                set_worker_agent_configs (config ["agent_configs"])
            except ImportError :
                print (f"[Worker {worker_pid}] Warning: Could not set agent configs")


        worker_gpu_assignments =config .get ('worker_gpu_assignments',[1 ])


        import multiprocessing as mp 
        current_process =mp .current_process ()

        worker_index =0 
        if hasattr (current_process ,'_identity')and current_process ._identity :

            worker_index =current_process ._identity [0 ]-1 
        elif hasattr (current_process ,'name')and 'SpawnPoolWorker'in current_process .name :

            try :
                worker_index =int (current_process .name .split ('-')[-1 ])-1 
            except (ValueError ,IndexError ):

                worker_index =worker_pid %len (worker_gpu_assignments )
        elif hasattr (current_process ,'name')and 'ForkPoolWorker'in current_process .name :

            try :
                worker_index =int (current_process .name .split ('-')[-1 ])-1 
            except (ValueError ,IndexError ):
                worker_index =worker_pid %len (worker_gpu_assignments )
        else :

            worker_index =worker_pid %len (worker_gpu_assignments )


        if worker_index <len (worker_gpu_assignments ):
            assigned_gpu =worker_gpu_assignments [worker_index ]
        else :

            assigned_gpu =worker_gpu_assignments [worker_index %len (worker_gpu_assignments )]


        _worker_context .assigned_gpu =assigned_gpu 


        if "agent_configs"in config and config ["agent_configs"]:
            from guf .utils import set_worker_agent_configs 
            set_worker_agent_configs (config ["agent_configs"])

            if logger :
                logger .log_event ("agent_configs_loaded",{
                "agent_count":len (config ["agent_configs"]),
                "agent_names":list (config ["agent_configs"].keys ())
                })


            if config .get ("debug",False ):
                print (f"[Worker {worker_pid}] Agent configurations loaded:")
                for agent_name ,agent_config in config ["agent_configs"].items ():
                    print (f"  {agent_name} -> {agent_config['model_name']}")
                    if agent_config .get ("payload"):
                        print (f"    Payload: {agent_config['payload']}")


        print (
        f"[Worker {worker_pid}] CMA-ES worker initialized on GPU cuda:{assigned_gpu} "
        f"(index {worker_index}/{len(worker_gpu_assignments)}, process: {current_process.name})"
        )


        if config .get ("debug",False ):
            print (f"[Worker {worker_pid}] Process identity: {getattr(current_process, '_identity', 'None')}")
            print (f"[Worker {worker_pid}] Process name: {getattr(current_process, 'name', 'None')}")
            print (f"[Worker {worker_pid}] Worker index: {worker_index}/{len(worker_gpu_assignments)}")
            print (f"[Worker {worker_pid}] GPU assignment: cuda:{assigned_gpu}")
            print (f"[Worker {worker_pid}] Available GPUs: {worker_gpu_assignments}")
            print (f"[Worker {worker_pid}] Server mappings: {_worker_context.servers}")
            print (f"[Worker {worker_pid}] Valid ratio: {config.get('valid_ratio', 0.5)}")
            print (f"[Worker {worker_pid}] Test ratio: {config.get('test_ratio', 0.2)}")
            if 'test_split_enabled'in config :
                print (f"[Worker {worker_pid}] Test split enabled: {config.get('test_split_enabled', False)}")

        _worker_initialized =True 

    except Exception as e :
        if logger :
            logger .log_event ("worker_init_error",{
            "error":str (e ),
            "error_type":type (e ).__name__ ,
            "worker_pid":worker_pid 
            })
        print (f"Error in worker initialization: {e}")
        raise 

def _execute_unified_job (job_type :JobType ,job_data :Dict )->Any :

    global _worker_context ,_code_executor 

    worker_pid =os .getpid ()
    start_time =time .time ()

    try :
        if job_type ==JobType .TRAINING_EPISODE :
            result =_execute_training_episode (job_data )
        elif job_type ==JobType .CODE_EXECUTION :
            result =_execute_code_execution (job_data )
        else :
            raise ValueError (f"Unknown job type: {job_type}")

        execution_time =time .time ()-start_time 
        return result 

    except Exception as e :
        execution_time =time .time ()-start_time 
        error_msg =f"Error executing {job_type.value} job in worker {worker_pid}: {e}"
        print (error_msg )
        traceback .print_exc ()


        if job_type ==JobType .TRAINING_EPISODE :
            return (-1.0 ,0 ,[],[],"Error occurred",{
            "router_tokens":0 ,"agent_input_tokens":0 ,
            "agent_output_tokens":0 ,"total_tokens":0 ,"num_turns":0 
            },[],[])
        elif job_type ==JobType .CODE_EXECUTION :
            return [[-1 ]],{"error":error_msg ,"error_code":-5 ,"execution_time":execution_time }
        else :
            raise 

def _execute_training_episode (job_data :Dict )->Tuple :

    global _worker_context 


    task_id =job_data ['task_id']
    split =job_data ['split']
    flat_params =job_data ['flat_params']
    svd_weights_cpu =job_data ['svd_weights_cpu']
    iteration_idx =job_data ['iteration_idx']
    eps_explore =job_data ['eps_explore']
    servers_dict =job_data ['servers_dict']
    use_structured_router =job_data ['use_structured_router']
    closed_model_config =job_data .get ('closed_model_config')
    agent_configs =job_data .get ('agent_configs')


    if agent_configs :
        try :
            from guf .utils import set_worker_agent_configs 
            set_worker_agent_configs (agent_configs )
        except ImportError :
            pass 


    if closed_model_config :
        _worker_context .using_closed_models =True 
        os .environ ["USING_CLOSED_MODELS"]="1"
    else :
        _worker_context .using_closed_models =False 
        os .environ .pop ("USING_CLOSED_MODELS",None )


    if _worker_context .router_model is None :
        _initialize_worker_models ()


    try :
        from guf .algorithms .es import ParameterApplier 
        ParameterApplier .apply_params_to_model (
        _worker_context .router_model ,
        _worker_context .linear_layer ,
        svd_weights_cpu ,
        flat_params ,
        use_structured_router 
        )
    except Exception as e :
        print (f"Error applying parameters: {e}")
        raise 


    try :
        from guf .trainer import EvaluationManager 


        debug_log_file =None 
        if _worker_context .debug and _worker_context .debug_log_dir is not None :
            iter_dir =EvaluationManager ._mk_iter_dir (_worker_context .debug_log_dir ,iteration_idx )
            log_filename =f"debug_unified_{task_id}_{split}.txt"
            debug_log_file =os .path .join (iter_dir ,log_filename )
            with open (debug_log_file ,"w")as f :
                f .write (f"Unified job manager debug log for task_id: {task_id}, split: {split}\n")


        return EvaluationManager .evaluate_episode (
        context =_worker_context ,
        task_id =task_id ,
        split =split ,
        iter_idx =iteration_idx ,
        eps_explore =eps_explore ,
        debug_log_file =debug_log_file 
        )

    except Exception as e :
        print (f"Error in training episode evaluation: {e}")
        traceback .print_exc ()
        return (-1.0 ,0 ,[],[],"Error occurred",{
        "router_tokens":0 ,"agent_input_tokens":0 ,
        "agent_output_tokens":0 ,"total_tokens":0 ,"num_turns":0 
        },[],[])

def _execute_code_execution (job_data :Dict )->Tuple :

    global _code_executor 

    if _code_executor is None :
        return [[-1 ]],{
        "error":"Code executor not available",
        "error_code":-5 ,
        "error_message":"DirectCodeExecutor not initialized"
        }

    sample =job_data ['sample']
    generation =job_data ['generation']
    timeout =job_data ['timeout']
    debug =job_data ['debug']
    using_closed_models =job_data ['using_closed_models']

    try :

        return _code_executor .check_correctness (
        sample ,generation ,timeout ,debug ,using_closed_models 
        )
    except Exception as e :
        print (f"Error in code execution: {e}")
        traceback .print_exc ()
        return [[-1 ]],{
        "error":str (e ),
        "error_code":-5 ,
        "error_message":f"Code execution failed: {e}"
        }

def _initialize_worker_models ():

    global _worker_context 

    worker_pid =os .getpid ()
    assigned_gpu =getattr (_worker_context ,'assigned_gpu',1 )
    device_str =f"cuda:{assigned_gpu}"

    print (f"[Worker {worker_pid}] Initializing models on {device_str}...")

    try :

        from transformers import AutoModelForCausalLM ,AutoTokenizer 

        _worker_context .router_model =AutoModelForCausalLM .from_pretrained (
        _worker_context .router_model_name ,
        torch_dtype =torch .bfloat16 ,
        attn_implementation ="flash_attention_2",
        device_map =device_str ,
        )


        if "qwen"in _worker_context .router_model_name .lower ():
            from guf .model_mods .modeling_qwen2 import forward as qwen2_forward 
            _worker_context .router_model .forward =qwen2_forward .__get__ (
            _worker_context .router_model ,
            type (_worker_context .router_model )
            )


        _worker_context .tokenizer =AutoTokenizer .from_pretrained (
        _worker_context .router_model_name 
        )


        use_consultant =getattr (_worker_context ,'use_consultant',True )
        consultant_outputs =1 if use_consultant else 0 

        _worker_context .linear_layer =torch .nn .Linear (
        in_features =_worker_context .router_model .config .hidden_size ,
        out_features =len (_worker_context .llm_names )+consultant_outputs ,
        bias =False ,
        ).to (device_str ).to (torch .bfloat16 )


        from guf .run_tasks import create_task 

        valid_ratio =getattr (_worker_context ,'valid_ratio',0.5 )
        test_ratio =getattr (_worker_context ,'test_ratio',0.2 )

        create_task_args ={
        "task_name":_worker_context .task_name ,
        "llm_names":_worker_context .llm_names ,
        "max_tokens":_worker_context .max_tokens ,
        "temperature":_worker_context .temperature ,
        "max_turns":_worker_context .max_turns ,
        "servers":_worker_context .servers ,
        "ports":_worker_context .ports ,
        "valid_ratio":valid_ratio ,
        "test_ratio":test_ratio ,
        "use_structured_router":getattr (_worker_context ,'use_structured_router',False ),
        "seed":_worker_context .seed ,
        "use_consultant":use_consultant ,
        "use_verifier":_worker_context .use_verifier ,
        }

        worker_log_dir =getattr (_worker_context ,'log_dir',None )
        if worker_log_dir :
            create_task_args ["log_dir"]=worker_log_dir 

        _worker_context .task_instance =create_task (**create_task_args )

        print (f"[Worker {worker_pid}] Models initialized successfully on {device_str}")

    except Exception as e :
        print (f"[Worker {worker_pid}] Error initializing models: {e}")
        traceback .print_exc ()
        raise 

def cleanup_job_manager ():

    global _job_manager 
    if _job_manager is not None :
        _job_manager .cleanup ()

def force_cleanup_job_manager ():

    global _job_manager 
    if _job_manager is not None :
        _job_manager .force_cleanup ()


atexit .register (cleanup_job_manager )