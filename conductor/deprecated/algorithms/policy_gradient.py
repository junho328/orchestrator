

import os 
import json 
import torch 
import numpy as np 
from typing import Dict ,List ,Optional ,Tuple ,Any 
from tqdm import tqdm 
import matplotlib .pyplot as plt 
import multiprocessing as mp 

from guf .trainer import (
RouterInfrastructure ,
SVDParameterManager ,
EvaluationManager ,
_worker_process_context ,
_init_worker 
)
from guf .utils import calculate_agent_stats ,aggregate_token_statistics 
from guf .run_tasks import create_task 



def _init_pg_worker (config :Dict )->None :

    global _worker_process_context 


    _init_worker (config )


    worker_pid =os .getpid ()
    print (f"[Worker {worker_pid}] PG worker initialized")
    print (f"[Worker {worker_pid}] Valid ratio: {config.get('valid_ratio', 0.5)}")
    print (f"[Worker {worker_pid}] Test ratio: {config.get('test_ratio', 0.2)}")
    print (f"[Worker {worker_pid}] Test split enabled: {config.get('test_split_enabled', False)}")
    print (f"[Worker {worker_pid}] Seed: {config.get('seed', 42)}")


    seed =config .get ('seed',42 )


    if config .get ("test_split_enabled",False ):
        print (f"[Worker {worker_pid}] Test split enabled, ensuring it's loaded")


        task_name =_worker_process_context .task_name 
        llm_names =_worker_process_context .llm_names 
        max_tokens =_worker_process_context .max_tokens 
        temperature =_worker_process_context .temperature 
        max_turns =_worker_process_context .max_turns 
        ports =_worker_process_context .ports 
        servers =_worker_process_context .servers 
        valid_ratio =config .get ("valid_ratio",0.5 )
        test_ratio =config .get ("test_ratio",0.2 )


        if _worker_process_context .task_instance is None :

            print (f"[Worker {worker_pid}] Creating new task instance with seed={seed}, valid_ratio={valid_ratio}, test_ratio={test_ratio}")
            _worker_process_context .task_instance =create_task (
            task_name ,
            llm_names =llm_names ,
            seed =seed ,
            max_tokens =max_tokens ,
            temperature =temperature ,
            max_turns =max_turns ,
            servers =servers ,
            ports =ports ,
            valid_ratio =valid_ratio ,
            test_ratio =test_ratio ,
            )


        if not hasattr (_worker_process_context .task_instance ,
        'data_splits')or _worker_process_context .task_instance .data_splits is None :
            print (f"[Worker {worker_pid}] data_splits not initialized, loading now with seed={seed}")

            _worker_process_context .task_instance .data_splits =_worker_process_context .task_instance ._load_data (
            seed =seed ,
            split ="train",
            validation =True ,
            valid_ratio =valid_ratio ,
            test_split =True ,
            test_ratio =test_ratio 
            )

        elif "test"not in _worker_process_context .task_instance .data_splits :
            print (f"[Worker {worker_pid}] Test split missing, reloading data with seed={seed}")

            _worker_process_context .task_instance .data_splits =_worker_process_context .task_instance ._load_data (
            seed =seed ,
            split ="train",
            validation =True ,
            valid_ratio =valid_ratio ,
            test_split =True ,
            test_ratio =test_ratio 
            )


        if _worker_process_context .task_instance .data_splits is not None and "test"in _worker_process_context .task_instance .data_splits :
            print (
            f"[Worker {worker_pid}] Test split loaded with {len(_worker_process_context.task_instance.data_splits['test'])} samples")
        else :
            print (f"[Worker {worker_pid}] WARNING: Failed to load test split")


class PolicyGradientTrainer :


    def __init__ (
    self ,
    infrastructure :RouterInfrastructure ,
    num_iters :int =100 ,
    test_interval :int =10 ,
    validation_samples :int =-1 ,
    batch_size :int =512 ,
    init_std :float =1e-3 ,
    lr :float =2e-3 ,
    max_grad_norm :float =1e-3 ,
    entropy_coef :float =0.35 ,
    l2_coef :float =1e-4 ,
    neg_reward :float =-1e-3 ,
    eps_explore :float =0.05 ,
    stable :bool =False ,
    opt_layer_indices :Optional [List [int ]]=None ,
    ):

        self .infra =infrastructure 
        self .num_iters =num_iters 
        self .test_interval =test_interval 
        self .validation_samples =validation_samples 
        self .batch_size =batch_size 
        self .init_std =init_std 
        self .lr =lr 
        self .max_grad_norm =max_grad_norm 
        self .entropy_coef =entropy_coef 
        self .l2_coef =l2_coef if not stable else 0.0 
        self .neg_reward =neg_reward 
        self .eps_explore =eps_explore if not stable else 0.0 
        self .stable =stable 
        self .opt_layer_indices =opt_layer_indices 


        self .valid_ratio =getattr (self .infra ,'valid_ratio',0.5 )
        self .test_ratio =getattr (self .infra ,'test_ratio',0.2 )


        self .seed =getattr (self .infra ,'seed',42 )


        print (f"[PG] Using seed={self.seed}, valid_ratio={self.valid_ratio}, test_ratio={self.test_ratio}")


        self .total_agent_usage ={name :0 for name in infrastructure .llm_names }


        self .best_score =-float ('inf')
        self .best_iter =-1 


        self .model_save_dir =os .path .join (self .infra .log_dir ,"models")
        os .makedirs (self .model_save_dir ,exist_ok =True )
        self .best_model_path =os .path .join (self .model_save_dir ,"best_model.pt")


        if self .opt_layer_indices :
            print (f"[PG] Selectively training layers: {self.opt_layer_indices}")


        self .model ,self .tokenizer ,self .linear_layer ,self .svd_weights =(
        self .infra .initialize_models (layer_indices =opt_layer_indices )
        )


        self .learnable_params =self .initialize_learnable_params ()


        self .optimizer =torch .optim .Adam (
        list (self .learnable_params .values ()),lr =self .lr 
        )


        self .model .eval ()
        for k in self .learnable_params :
            if k !="action_layer.weight":
                self .model .get_parameter (k ).requires_grad_ (True )


        self .log_file =os .path .join (self .infra .log_dir ,"pg_log.json")
        self .log_data =[{"configs":self .get_config_dict ()}]
        with open (self .log_file ,"w")as f :
            json .dump (self .log_data ,f ,indent =2 )

    def get_config_dict (self )->Dict :

        return {
        "task":self .infra .task ,
        "model_name":self .infra .model_name ,
        "llm_names":self .infra .llm_names ,
        "log_dir":self .infra .log_dir ,
        "num_iters":self .num_iters ,
        "test_interval":self .test_interval ,
        "validation_samples":self .validation_samples ,
        "batch_size":self .batch_size ,
        "init_std":self .init_std ,
        "lr":self .lr ,
        "max_grad_norm":self .max_grad_norm ,
        "entropy_coef":self .entropy_coef ,
        "l2_coef":self .l2_coef ,
        "neg_reward":self .neg_reward ,
        "eps_explore":self .eps_explore ,
        "temperature":self .infra .temperature ,
        "max_tokens":self .infra .max_tokens ,
        "max_turns":self .infra .max_turns ,
        "debug":self .infra .debug ,
        "stable":self .stable ,
        "opt_layer_indices":self .opt_layer_indices ,
        "valid_ratio":self .valid_ratio ,
        "test_ratio":self .test_ratio ,
        "seed":self .seed ,
        "diversity_bonus_weight":0.0 ,
        }

    def _setup_worker_pool (self ,for_eval :bool =False ):


        worker_config ={
        "debug":self .infra .debug ,
        "debug_log_dir":self .infra .debug_log_dir ,
        "task_name":self .infra .task ,
        "max_tokens":self .infra .max_tokens ,
        "temperature":self .infra .temperature ,
        "max_turns":self .infra .max_turns ,
        "router_model_name":self .infra .model_name ,
        "llm_names":self .infra .llm_names ,
        "ports":self .infra .ports ,
        "servers":self .infra .servers ,
        "valid_ratio":self .valid_ratio ,
        "test_ratio":self .test_ratio ,
        "seed":self .seed ,
        "test_split_enabled":True 
        }


        if self .infra .debug :
            print (f"[PG] Creating worker pool with seed={self.seed}, valid_ratio={self.valid_ratio}, test_ratio={self.test_ratio}")


        pool_size =self .infra .eval_workers if for_eval else self .infra .num_workers 
        mp .set_start_method ("spawn",force =True )
        pool =mp .Pool (
        pool_size ,
        initializer =_init_pg_worker ,
        initargs =(worker_config ,)
        )

        return pool 

    def initialize_learnable_params (self )->Dict [str ,torch .nn .Parameter ]:

        learnable_params ={}
        for k ,v in self .svd_weights .items ():
            if k .endswith (".S"):
                learnable_params [k [:-2 ]]=torch .nn .Parameter (
                torch .randn_like (v )*self .init_std ,requires_grad =True 
                )


        self .linear_layer .weight .data =(
        torch .randn_like (self .linear_layer .weight .data )*self .init_std 
        )
        learnable_params ["action_layer.weight"]=self .linear_layer .weight 

        return learnable_params 

    def train (self ):


        diag_dir =os .path .join (self .infra .log_dir ,"pg_diagnostics")
        os .makedirs (diag_dir ,exist_ok =True )

        best_score =-1.0 
        try :
            for i in range (self .num_iters +1 ):
                print (f"Iteration {i}/{self.num_iters}")
                self .optimizer .zero_grad ()
                self .model .zero_grad ()
                self .linear_layer .zero_grad ()


                SVDParameterManager .load_model_weights (
                self .model ,self .learnable_params ,self .svd_weights 
                )
                model_state_dict =self .model .state_dict ()
                linear_layer_state_dict =self .linear_layer .state_dict ()


                np .random .seed (self .seed +i )
                seeds =np .random .randint (0 ,self .infra .train_dataset_size ,size =self .batch_size )
                train_args =[
                (
                int (seed ),
                "train",
                model_state_dict ,
                linear_layer_state_dict ,
                self .infra .servers ,
                i ,
                self .eps_explore 
                )
                for seed in seeds 
                ]


                eval_pool =self ._setup_worker_pool (for_eval =True )
                results =list (
                tqdm (
                eval_pool .imap (self .infra .do_eval ,train_args ),
                total =len (train_args ),
                desc =f"Iter {i} train",
                leave =False ,
                )
                )
                eval_pool .close ()
                eval_pool .join ()


                train_token_stats =aggregate_token_statistics (results )


                self .update_model (results ,i ,diag_dir )


                self .log_data [-1 ]["token_stats"]=train_token_stats 
                with open (self .log_file ,"w")as f :
                    json .dump (self .log_data ,f ,indent =2 )


                if i >0 and i %self .test_interval ==0 :
                    best_score =self .run_validation (
                    model_state_dict ,linear_layer_state_dict ,i ,best_score 
                    )

                    weight_file =os .path .join (self .infra .log_dir ,f"pg_model_{i}.pt")
                    torch .save (self .learnable_params ,weight_file )

            print ("Training finished.")
            return self .log_data 

        finally :

            torch .cuda .empty_cache ()

    def update_model (self ,results ,iter_idx ,diag_dir ):


        diag_file =os .path .join (diag_dir ,f"iter_{iter_idx}_diagnostics.txt")
        diagnostics =[]


        episode_rewards =[]
        batch_obs ,batch_acts ,batch_rewards =[],[],[]
        episode_agents =[]
        entropy_history =[]

        total_loss =0.0 
        grad_norms ={}


        for reward ,num_turns ,obs_act ,sampled_ids ,response ,token_stats in results :
            if reward ==-1.0 :
                continue 
            episode_rewards .append (reward )


            r =(1.0 if reward >0 else self .neg_reward )/num_turns 
            for obs ,act in obs_act :
                batch_obs .append (obs )
                batch_acts .append (act )
                batch_rewards .append (r )
            episode_agents .extend (sampled_ids )


        if not episode_rewards :
            diagnostics .append (f"Iteration {iter_idx}: No valid rewards; skipping update.\n")
            with open (diag_file ,"w")as f :
                f .writelines (diagnostics )
            return 

        avg_reward =float (np .mean (episode_rewards ))
        diagnostics .append (f"Iteration {iter_idx}: avg train reward = {avg_reward:.4f}\n")


        baseline =float (np .mean (batch_rewards ))
        diagnostics .append (f"Baseline (average per-step reward): {baseline:.4f}\n\n")


        diagnostics .append ("[Diagnostic] BEFORE update:\n")
        for name ,param in self .learnable_params .items ():
            diagnostics .append (
            f"  Param {name}: "
            f"min={param.data.min().item():.4f}, "
            f"max={param.data.max().item():.4f}, "
            f"mean={param.data.mean().item():.4f}, "
            f"std={param.data.std().item():.4f}\n"
            )


        for obs ,act ,r in zip (batch_obs ,batch_acts ,batch_rewards ):
            logits =EvaluationManager .get_action (
            self .model ,self .linear_layer ,self .tokenizer ,obs ,inference =False 
            )
            log_probs =torch .nn .functional .log_softmax (logits ,dim =-1 )
            probs =torch .exp (log_probs )
            entropy =-(probs *log_probs ).sum ()
            entropy_history .append (entropy .item ())

            advantage =r if self .stable else r -baseline 
            loglikelihood =log_probs .squeeze ()[act ]
            pg_loss =-(loglikelihood *advantage +self .entropy_coef *entropy )/len (batch_obs )

            if self .l2_coef >0.0 :
                l2_loss =self .l2_coef *(self .linear_layer .weight **2 ).mean ()
                loss =pg_loss +l2_loss 
            else :
                loss =pg_loss 

            loss .backward ()
            total_loss +=loss .item ()


        torch .nn .utils .clip_grad_norm_ (
        list (self .learnable_params .values ()),self .max_grad_norm 
        )
        self .optimizer .step ()

        diagnostics .append (f"Total loss for update: {total_loss:.4f}\n")
        diagnostics .append ("[Diagnostic] Gradient norms:\n")
        for name ,param in self .learnable_params .items ():
            if param .grad is not None :
                grad_norm =param .grad .data .norm ().item ()
                grad_norms [name ]=grad_norm 
                diagnostics .append (f"  {name}: grad_norm={grad_norm:.4f}\n")


        diagnostics .append ("[Diagnostic] AFTER update:\n")
        for name ,param in self .learnable_params .items ():
            diagnostics .append (
            f"  Param {name}: min={param.data.min().item():.4f}, "
            f"max={param.data.max().item():.4f}, "
            f"mean={param.data.mean().item():.4f}, "
            f"std={param.data.std().item():.4f}\n"
            )


        with open (diag_file ,"w")as f :
            f .writelines (diagnostics )


        agent_stats ,self .total_agent_usage =calculate_agent_stats (
        episode_agents ,
        self .infra .llm_names ,
        self .total_agent_usage ,
        entropy_history 
        )


        entry ={
        "iter":iter_idx ,
        "type":"train",
        "avg_reward":avg_reward ,
        "total_loss":total_loss ,
        "num_samples":len (episode_rewards ),
        **agent_stats 
        }
        self .log_data .append (entry )


        with open (self .log_file ,"w")as f :
            json .dump (self .log_data ,f ,indent =2 )


        self ._create_gradient_plot (grad_norms ,avg_reward ,iter_idx ,diag_dir )

    def _create_gradient_plot (self ,grad_norms ,avg_reward ,iter_idx ,diag_dir ):

        plt .figure (figsize =(10 ,5 ))
        names =list (grad_norms .keys ())
        norms =[grad_norms [name ]for name in names ]

        sorted_indices =np .argsort (norms )[::-1 ]
        sorted_names =[names [i ]for i in sorted_indices [:20 ]]
        sorted_norms =[norms [i ]for i in sorted_indices [:20 ]]

        plt .bar (range (len (sorted_names )),sorted_norms )
        plt .xticks (range (len (sorted_names )),sorted_names ,rotation =45 ,ha ="right")
        plt .ylabel ("Gradient Norm")
        plt .title (f"Iteration {iter_idx} Gradient Norms (Top 20)\nAvg Reward: {avg_reward:.4f}")
        plt .tight_layout ()

        plot_file =os .path .join (diag_dir ,f"iter_{iter_idx}_grad_norms.png")
        plt .savefig (plot_file )
        plt .close ()

    def run_validation (self ,model_state_dict ,linear_layer_state_dict ,iter_idx ,best_score ):

        if self .validation_samples ==-1 :
            indices =list (range (self .infra .valid_dataset_size ))
        else :

            np .random .seed (self .seed +10000 +iter_idx )
            indices =np .random .choice (
            self .infra .valid_dataset_size ,
            size =min (self .validation_samples ,self .infra .valid_dataset_size ),
            replace =False ,
            )

        eval_args =[
        (
        int (idx ),
        "valid",
        model_state_dict ,
        linear_layer_state_dict ,
        self .infra .servers ,
        iter_idx ,
        0.0 
        )
        for idx in indices 
        ]


        eval_pool =self ._setup_worker_pool (for_eval =True )
        test_results =list (
        tqdm (
        eval_pool .imap (self .infra .do_eval ,eval_args ),
        total =len (indices ),
        desc =f"Iter {iter_idx} valid",
        leave =False ,
        )
        )
        eval_pool .close ()
        eval_pool .join ()


        valid_token_stats =aggregate_token_statistics (test_results )

        test_scores =[r [0 ]for r in test_results if r [0 ]!=-1.0 ]
        if not test_scores :
            return best_score 

        valid_agent_ids =[]
        for result in test_results :
            if len (result )>=4 and result [0 ]!=-1.0 :
                valid_agent_ids .extend (result [3 ])

        valid_agent_stats ,_ =calculate_agent_stats (
        valid_agent_ids ,
        self .infra .llm_names 
        )

        test_score =float (np .mean (test_scores ))
        new_best =max (best_score ,test_score )


        is_new_best =test_score >best_score 
        if is_new_best :
            self .best_score =test_score 
            self .best_iter =iter_idx 


            torch .save (self .learnable_params ,self .best_model_path )


            iter_best_path =os .path .join (self .model_save_dir ,f"best_model_iter_{iter_idx}.pt")
            torch .save (self .learnable_params ,iter_best_path )

            print (f"New best model saved at iter {iter_idx} with validation score: {test_score:.4f}")

        entry ={
        "iter":iter_idx ,
        "type":"valid",
        "avg_reward":test_score ,
        "best_score":new_best ,
        "is_new_best":is_new_best ,
        "best_iter":self .best_iter ,
        "num_samples":len (test_scores ),
        **valid_agent_stats ,
        "token_stats":valid_token_stats 
        }
        self .log_data .append (entry )

        with open (self .log_file ,"w")as f :
            json .dump (self .log_data ,f ,indent =2 )

        print (f"Validation avg reward: {test_score:.4f} (best {new_best:.4f})")

        return new_best 

    def run_test (self ,model_path :Optional [str ]=None ,test_size :int =1000 ):

        print ("[PG] Running test evaluation with best model...")





        model_loaded =False 

        if model_path and os .path .exists (model_path ):
            try :
                self .learnable_params =torch .load (model_path )
                print (f"[PG] Loaded model from {model_path}")
                model_loaded =True 
            except Exception as e :
                print (f"[PG] Error loading specified model: {e}")

        if not model_loaded and os .path .exists (self .best_model_path ):
            try :
                self .learnable_params =torch .load (self .best_model_path )
                print (f"[PG] Loaded best model from {self.best_model_path}")
                model_loaded =True 
            except Exception as e :
                print (f"[PG] Error loading best model: {e}")

        if not model_loaded :
            print ("[PG] Using current model parameters")


        SVDParameterManager .load_model_weights (
        self .model ,self .learnable_params ,self .svd_weights 
        )
        model_state_dict =self .model .state_dict ()
        linear_layer_state_dict =self .linear_layer .state_dict ()


        if not hasattr (self .infra ,'test_dataset_size')or self .infra .test_dataset_size ==0 :
            print ("[PG] Test dataset not initialized, initializing now...")

            self .infra ._initialize_dataset_sizes (
            test_ratio =self .test_ratio ,
            valid_ratio =self .valid_ratio 
            )

        test_dataset_size =self .infra .test_dataset_size 
        if test_dataset_size ==0 :
            print ("[PG] Error: Test dataset size is 0 or not available")
            return None 

        print (f"[PG] Test dataset has {test_dataset_size} samples")


        num_samples =min (test_size ,test_dataset_size )
        if num_samples <test_dataset_size :

            np .random .seed (self .seed +20000 )
            test_indices =np .random .choice (test_dataset_size ,size =num_samples ,replace =False )
        else :
            test_indices =range (test_dataset_size )

        print (f"[PG] Evaluating on {len(test_indices)} out of {test_dataset_size} test samples")


        eval_args =[
        (
        int (idx ),
        "test",
        model_state_dict ,
        linear_layer_state_dict ,
        self .infra .servers ,
        -1 ,
        0.0 
        )
        for idx in test_indices 
        ]


        eval_pool =self ._setup_worker_pool (for_eval =True )
        test_results =list (
        tqdm (
        eval_pool .imap (self .infra .do_eval ,eval_args ),
        total =len (eval_args ),
        desc ="[PG] Test Evaluation"
        )
        )
        eval_pool .close ()
        eval_pool .join ()


        test_token_stats =aggregate_token_statistics (test_results )


        test_agent_ids =[]
        for result in test_results :
            if len (result )>=4 and result [0 ]!=-1.0 :
                test_agent_ids .extend (result [3 ])

        test_agent_stats ,_ =calculate_agent_stats (
        test_agent_ids ,self .infra .llm_names 
        )

        test_scores =[r [0 ]for r in test_results if r [0 ]!=-1.0 ]
        test_score =float (np .mean (test_scores ))if test_scores else 0.0 


        test_entry ={
        "type":"test",
        "test_score":test_score ,
        "num_samples":len (test_scores ),
        "test_size":test_size ,
        "best_score":self .best_score if hasattr (self ,'best_score')else None ,
        "best_iter":self .best_iter if hasattr (self ,'best_iter')else -1 ,
        "validation_best_score":self .best_score if hasattr (self ,'best_score')else None ,
        **test_agent_stats ,
        "token_stats":test_token_stats ,
        }


        self .log_data .append (test_entry )
        with open (self .log_file ,"w")as f :
            json .dump (self .log_data ,f ,indent =2 )

        print (f"[PG] Test evaluation complete. Score: {test_score:.4f}")
        if hasattr (self ,'best_score')and self .best_score is not None :
            print (f"[PG] Best validation score: {self.best_score:.4f} at iteration {self.best_iter}")

        return test_entry 