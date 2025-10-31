

import re 
from typing import Dict ,List ,Any ,Optional 
from datasets import load_dataset 
from guf .core import Task 
import numpy as np 


class CountdownTask (Task ):



    USER_PROMPT_TEMPLATE =(
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) one or multiple times but "
    "each number can only be used once. Show your work in <THINKING_TAG> </THINKING_TAG> tags. "
    "And return the final equation in <answer> </answer> tags, for example "
    "<answer> (1 + 2) / 3 </answer>; do not include = in the <answer> tags, only the equation. "
    "Think step by step inside <THINKING_TAG> tags, but keep it short."
    )

    def __init__ (
    self ,
    llm_names :List [str ],
    seed :int =42 ,
    max_tokens :int =512 ,
    temperature :float =0.8 ,
    max_turns :int =5 ,
    include_format_reward :bool =True ,
    servers :Dict [str ,str ]=None ,
    ports :Dict [str ,int ]=None ,
    track_costs :bool =True ,
    debug :bool =False ,
    together :bool =True ,
    valid_ratio :float =0.5 ,
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
        self .include_format_reward =include_format_reward 
        self .num_agents =len (llm_names )

    def _get_user_prompt_template (self )->str :

        return self .USER_PROMPT_TEMPLATE .replace ("THINKING_TAG",self .thinking_tag )

    def _format_base_prompt (self ,task_data :Dict )->str :

        return self .user_prompt_template .format (
        numbers =task_data ["nums"],
        target =task_data ["target_number"],
        )

    def _load_data (self ,seed :int ,split :str ="train",validation :bool =False ,
    valid_ratio :float =None ,max_samples :int =None ,
    test_split :bool =False ,test_ratio :float =None )->Dict :


        if valid_ratio is None :
            valid_ratio =self .valid_ratio 
        if max_samples is None :
            max_samples =self .max_samples 
        if test_ratio is None :
            test_ratio =self .test_ratio 


        ds =load_dataset ("Jiayi-Pan/Countdown-Tasks-3to4",split ="train")

        ds =ds .rename_column ("target","target_number")


        from guf .utils import get_or_create_indices 
        split_indices =get_or_create_indices (
        task_name ="countdown",
        dataset_len =len (ds ),
        seed =self .split_seed ,
        valid_ratio =valid_ratio ,
        test_ratio =test_ratio 
        )


        def _take (idxs ):
            return ds .select (idxs )

        data_splits ={
        "train":_take (split_indices ["train"]),
        "valid":_take (split_indices ["valid"]),
        "test":_take (split_indices ["test"])
        }


        if max_samples !=-1 :
            data_splits ["train"]=data_splits ["train"].shuffle (seed =seed ).select (range (min (max_samples ,len (data_splits ["train"]))))
            valid_cap =int (max_samples *valid_ratio /(1.0 -valid_ratio ))if valid_ratio <1.0 else len (data_splits ["valid"])
            data_splits ["valid"]=data_splits ["valid"].shuffle (seed =seed ).select (range (min (valid_cap ,len (data_splits ["valid"]))))
            test_cap =int (max_samples *test_ratio /(1.0 -test_ratio ))if test_ratio <1.0 else len (data_splits ["test"])
            data_splits ["test"]=data_splits ["test"].shuffle (seed =seed ).select (range (min (test_cap ,len (data_splits ["test"]))))


        for k ,v in data_splits .items ():
            print (f"Split {k} contains {len(v)} examples")


        for split_name in data_splits :

            data_splits [split_name ]=data_splits [split_name ].add_column (
            "task_type",
            ["countdown"]*len (data_splits [split_name ])
            )

        return data_splits 



    def _calculate_reward (self ,completions :List [str ],task_data :List [Dict [str ,Any ]],debug :bool =False )->List [float ]:

        rewards :List [float ]=[]

        targets =[data ['target_number']for data in task_data ]
        nums_list =[data ['nums']for data in task_data ]

        for completion ,gt ,numbers in zip (completions ,targets ,nums_list ):
            r =0.0 

            if '<answer>'in completion and '</answer>'in completion :

                expr =completion .split ('<answer>')[1 ].split ('</answer>')[0 ].strip ()

                if '='in expr :
                    expr =expr .split ('=')[0 ].strip ()
                try :

                    used_numbers =[int (n )for n in re .findall (r"\d+",expr )]
                except Exception as e :
                    if debug :
                        print ("=== DEBUG: Error parsing numbers ===")
                        print ("Error:",e )
                        print (f"Expression: {expr}")
                    r =0.0 
                    rewards .append (r )
                    continue 


                if sorted (used_numbers )!=sorted (numbers ):
                    if debug :
                        print ("=== DEBUG: Numbers mismatch ===")
                        print ("Target:",gt )
                        print ("Expected numbers:",sorted (numbers ))
                        print ("Used numbers:",sorted (used_numbers ))
                        print ("Expression:",expr )
                    r =0.0 
                else :

                    if not re .match (r"^[\d+\-*/().\s]+$",expr ):
                        if debug :
                            print ("=== DEBUG: Invalid characters in equation ===")
                            print ("Equation:",expr )
                        r =0.0 
                    else :
                        try :
                            result =eval (expr ,{"__builtins__":None },{})
                            if abs (float (result )-float (gt ))<1e-5 :
                                r =1.0 
                            else :
                                r =0.0 
                        except Exception as e :
                            if debug :
                                print ("=== DEBUG: Error evaluating equation ===")
                                print ("Equation:",expr )
                                print ("Error:",e )
                            r =0.0 
            else :
                if debug :
                    print ("=== DEBUG: No <answer> tag found ===")
                    print (f"Completion snippet: {completion[:100]}")
                r =0.0 

            if debug :
                print ("=== DEBUG: Countdown Answer Evaluation ===")
                print ("Target:",gt )
                print ("Numbers:",numbers )
                print ("Extracted expression:",expr if '<answer>'in completion else None )
                print ("Reward:",r )
                print ("===========================")

            rewards .append (r )

        assert len (rewards )==len (targets ),f"{len(rewards)} vs {len(targets)}"
        return rewards 


if __name__ =="__main__":

    llms =["gpt-4o-mini","claude-3-7-sonnet-20250219","gemini-1.5-pro","deepseek-ai/DeepSeek-V3"]
    task =CountdownTask (llm_names =llms ,debug =True )
    obs =task .reset (split ="test")
    done =False 

    while not done :
        action =np .random .rand (len (llms ))
        obs ,reward ,done ,obs_act =task .step (action )

    print ("Final response:",task .response )
    print ("Reward:",reward )