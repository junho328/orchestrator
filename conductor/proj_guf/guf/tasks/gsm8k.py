

import re 
from typing import Dict ,List ,Any ,Optional 
from datasets import load_dataset 
import numpy as np 
from guf .core import Task 
from guf .utils import extract_answer 


class GSM8KTask (Task ):



    USER_PROMPT_TEMPLATE =(
    "Solve the following math problem step by step: {question} "
    "Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example "
    "<answer> 42 </answer>. Think step by step inside <think> tags, but keep it short."
    )

    def __init__ (
    self ,
    llm_names :List [str ],
    seed :int =42 ,
    max_tokens :int =1024 ,
    temperature :float =0.8 ,
    max_turns :int =3 ,
    servers :Dict [str ,str ]=None ,
    ports :Dict [str ,int ]=None ,
    track_costs :bool =True ,
    debug :bool =False ,
    together :bool =True ,
    valid_ratio :float =0.2 ,
    max_samples :int =-1 ,
    test_ratio :float =0.2 ,
    use_consultant :bool =True ,
    log_dir :Optional [str ]=None ,
    ):

        super ().__init__ (
        llm_names ,
        seed =seed ,
        max_tokens =max_tokens ,
        temperature =temperature ,
        max_turns =max_turns ,
        servers =servers ,
        ports =ports ,
        track_costs =track_costs ,
        debug =debug ,
        together =together ,
        valid_ratio =valid_ratio ,
        max_samples =max_samples ,
        test_ratio =test_ratio ,
        use_consultant =use_consultant ,
        log_dir =log_dir ,
        )

    def _get_user_prompt_template (self )->str :

        return self .USER_PROMPT_TEMPLATE 

    def _format_base_prompt (self ,task_data :Dict )->str :

        return self .user_prompt_template .format (question =task_data ["question"])

    def _load_data (
    self ,
    seed :int ,
    split :str ="train",
    validation :bool =False ,
    valid_ratio :float =None ,
    max_samples :int =None ,
    test_split :bool =False ,
    test_ratio :float =None 
    )->Dict [str ,Any ]:


        if valid_ratio is None :
            valid_ratio =self .valid_ratio 
        if max_samples is None :
            max_samples =self .max_samples 
        if test_ratio is None :
            test_ratio =self .test_ratio 


        ds =load_dataset ("gsm8k","main",split ="train")


        from guf .utils import get_or_create_indices 

        split_indices =get_or_create_indices (
        task_name ="gsm8k",
        dataset_len =len (ds ),
        seed =self .split_seed ,
        valid_ratio =valid_ratio ,
        test_ratio =test_ratio 
        )

        def _take (idxs :List [int ]):
            return ds .select (idxs )

        data_splits ={
        "train":_take (split_indices ["train"]),
        "valid":_take (split_indices ["valid"]),
        "test":_take (split_indices ["test"])
        }


        if max_samples !=-1 :
            data_splits ["train"]=data_splits ["train"].shuffle (seed =seed ).select (range (min (max_samples ,len (data_splits ["train"]))))
            valid_cap =int (max_samples *valid_ratio /(1.0 -valid_ratio ))if valid_ratio <1.0 else len (
            data_splits ["valid"])
            data_splits ["valid"]=data_splits ["valid"].shuffle (seed =seed ).select (range (min (valid_cap ,len (data_splits ["valid"]))))
            test_cap =int (max_samples *test_ratio /(1.0 -test_ratio ))if test_ratio <1.0 else len (
            data_splits ["test"])
            data_splits ["test"]=data_splits ["test"].shuffle (seed =seed ).select (range (min (test_cap ,len (data_splits ["test"]))))


        for k ,v in data_splits .items ():
            print (f"Split {k} contains {len(v)} examples")

        return data_splits 

    def _calculate_reward (self ,completions :List [str ],task_data :List [Dict ],debug :bool =False )->List [float ]:

        rewards =[]

        for completion ,data in zip (completions ,task_data ):

            answer =extract_answer (completion )
            if answer is None :
                if debug :
                    print ("=== DEBUG: No <answer> tag found in response ===")
                    print ("Question:",data ["question"][:100 ],"...")
                    print ("Reference answer:",data ["answer"])
                    print ("Response snippet:",completion [:100 ],"...")
                rewards .append (0.0 )
                continue 


            pred_numbers =re .findall (r"[-+]?\d*\.\d+|\d+",answer )
            ref_numbers =re .findall (r"[-+]?\d*\.\d+|\d+",data ["answer"])

            if not pred_numbers or not ref_numbers :
                if debug :
                    print ("=== DEBUG: No numbers found in answer ===")
                    print ("Question:",data ["question"][:100 ],"...")
                    print ("Reference answer:",data ["answer"])
                    print ("Extracted answer:",answer )
                rewards .append (0.0 )
                continue 


            try :
                pred_final =float (pred_numbers [-1 ])
                ref_final =float (ref_numbers [-1 ])
            except Exception as e :
                if debug :
                    print ("=== DEBUG: Error converting numbers to float ===")
                    print ("Error:",e )
                rewards .append (0.0 )
                continue 


            is_correct =abs (pred_final -ref_final )<1e-5 

            if debug :
                print ("=== DEBUG: GSM8K Answer Evaluation ===")
                print ("Question:",data ["question"][:100 ],"...")
                print ("Reference answer:",data ["answer"])
                print ("Extracted answer:",answer )
                print ("Reference final number:",ref_final )
                print ("Predicted final number:",pred_final )
                print ("Correct:",is_correct )
                print ("Reward:",1.0 if is_correct else 0.0 )
                print ("===========================")

            rewards .append (1.0 if is_correct else 0.0 )

        return rewards 


if __name__ =="__main__":

    llms =["gpt-4o-mini","claude-3-7-sonnet-20250219"]
    task =GSM8KTask (llm_names =llms ,max_turns =3 ,max_tokens =1024 ,debug =True )
    obs =task .reset (split ="test")
    done =False 

    while not done :
        action =np .random .rand (len (llms ))
        obs ,reward ,done ,_ =task .step (action )

    print ("Final response:",task .response )
    print ("Reward:",reward )