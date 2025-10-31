

import re 
from typing import Dict ,List ,Any ,Optional 
from datasets import load_dataset ,Dataset ,concatenate_datasets 
import numpy as np 
from guf .core import Task 
from guf .utils import extract_answer 
from guf .cost import reset_costs 


class MixCGTask (Task ):



    COUNTDOWN_PROMPT_TEMPLATE =(
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) one or multiple times but "
    "each number can only be used once. Show your work in <think> </think> tags. "
    "And return the final equation in <answer> </answer> tags, for example "
    "<answer> (1 + 2) / 3 </answer>; do not include = in the <answer> tags, only the equation. "
    "Think step by step inside <think> tags, but keep it short."
    )

    GSM8K_PROMPT_TEMPLATE =(
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
    max_turns :int =5 ,
    include_format_reward :bool =True ,
    task_ratios :Dict [str ,float ]=None ,
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


        if task_ratios is None :
            self .task_ratios ={"countdown":0.5 ,"gsm8k":0.5 }
        else :
            self .task_ratios =task_ratios 


        total =sum (self .task_ratios .values ())
        if total !=1.0 :
            for k in self .task_ratios :
                self .task_ratios [k ]/=total 


        self .current_task_type ="mixed"
        self .include_format_reward =include_format_reward 

    def _get_user_prompt_template (self )->str :



        if not hasattr (self ,'current_task_type')or self .current_task_type in ['mixed',None ]:

            return self .GSM8K_PROMPT_TEMPLATE 

        if self .current_task_type =="countdown":
            return self .COUNTDOWN_PROMPT_TEMPLATE 
        else :
            return self .GSM8K_PROMPT_TEMPLATE 

    def _format_base_prompt (self ,task_data :Dict )->str :


        self .current_task_type =task_data .get ("task_type","gsm8k")

        if self .current_task_type =="countdown":
            return self .COUNTDOWN_PROMPT_TEMPLATE .format (
            numbers =task_data ["nums"],
            target =task_data ["target"],
            )
        else :
            return self .GSM8K_PROMPT_TEMPLATE .format (
            question =task_data ["question"]
            )

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



        from datasets import load_dataset ,concatenate_datasets 

        countdown_raw =load_dataset ("Jiayi-Pan/Countdown-Tasks-3to4",split ="train").map (lambda x :{"task_type":"countdown"})
        gsm8k_raw =load_dataset ("gsm8k","main",split ="train").map (lambda x :{"task_type":"gsm8k"})

        combined_raw =concatenate_datasets ([countdown_raw ,gsm8k_raw ])
        total_len =len (combined_raw )



        split_indices =self ._get_or_make_splits (
        raw_dataset_len =total_len ,
        task_name ="mix_c_g"
        )



        def _take (idx_list ):
            return combined_raw .select (idx_list )

        data_splits ={
        "train":_take (split_indices ["train"]),
        "valid":_take (split_indices ["valid"]),
        "test":_take (split_indices ["test"])
        }

        print (
        f"[MixCGTask] deterministic splits – "
        f"train {len(data_splits['train'])}, "
        f"valid {len(data_splits['valid'])}, "
        f"test  {len(data_splits['test'])}"
        )
        return data_splits 

    def reset (self ,task_id :int =-1 ,split :str ="train"):


        if self .track_costs :
            reset_costs ()

        if self .use_consultant :
            self .pending_suggestion =None 


        if self .data_splits is None :
            self .data_splits =self ._load_data (
            seed =self .np_random .randint (0 ,10000 ),
            split =split ,
            validation =True ,
            valid_ratio =self .valid_ratio ,
            max_samples =self .max_samples 
            )

        self .data_split =self .data_splits [split ]
        if task_id <0 :
            self .task_id =self .np_random .randint (0 ,len (self .data_split ))
        else :
            self .task_id =task_id 


        task_data =self .data_split [self .task_id ]
        self .current_task_type =task_data .get ("task_type","gsm8k")


        user_prompt =self ._format_user_prompt (task_data )



        if self .max_turns >1 and len (self .llm_names )>1 :

            self .messages =[
            {"role":"system","content":self .router_system_prompt .format (num_agents =len (self .llm_names ))},
            {"role":"user","content":user_prompt },
            ]
        else :

            self .messages =[
            {"role":"system","content":self .system_prompt },
            {"role":"user","content":user_prompt },
            {"role":"assistant","content":self .assistant_prompt }
            ]


        self ._obs =self ._get_obs ()
        self .obs_act =[]
        self .num_turns =0 
        self .prev_agent_id =-1 
        self .response =None 

        return self ._obs 

    def _calculate_reward (self ,completions :List [str ],task_data :List [Dict [str ,Any ]],debug :bool =False )->List [float ]:

        rewards =[]

        for completion ,data in zip (completions ,task_data ):
            task_type =data .get ("task_type","gsm8k")

            if task_type =="countdown":

                r =self ._calculate_countdown_reward (completion ,data ,debug )
            else :

                r =self ._calculate_gsm8k_reward (completion ,data ,debug )

            rewards .append (r )

        return rewards 

    def _calculate_countdown_reward (self ,completion :str ,task_data :Dict [str ,Any ],debug :bool =False )->float :

        r =0.0 
        gt =task_data ['target']
        numbers =task_data ['nums']


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
                return 0.0 


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

        return r 

    def _calculate_gsm8k_reward (self ,completion :str ,task_data :Dict [str ,Any ],debug :bool =False )->float :


        answer =extract_answer (completion )
        if answer is None :
            if debug :
                print ("=== DEBUG: No <answer> tag found in response ===")
                print ("Question:",task_data ["question"][:100 ],"...")
                print ("Reference answer:",task_data ["answer"])
                print ("Response snippet:",completion [:100 ],"...")
            return 0.0 


        pred_numbers =re .findall (r"[-+]?\d*\.\d+|\d+",answer )
        ref_numbers =re .findall (r"[-+]?\d*\.\d+|\d+",task_data ["answer"])

        if not pred_numbers or not ref_numbers :
            if debug :
                print ("=== DEBUG: No numbers found in answer ===")
                print ("Question:",task_data ["question"][:100 ],"...")
                print ("Reference answer:",task_data ["answer"])
                print ("Extracted answer:",answer )
            return 0.0 


        try :
            pred_final =float (pred_numbers [-1 ])
            ref_final =float (ref_numbers [-1 ])
        except Exception as e :
            if debug :
                print ("=== DEBUG: Error converting numbers to float ===")
                print ("Error:",e )
            return 0.0 


        is_correct =abs (pred_final -ref_final )<1e-5 

        if debug :
            print ("=== DEBUG: GSM8K Answer Evaluation ===")
            print ("Question:",task_data ["question"][:100 ],"...")
            print ("Reference answer:",task_data ["answer"])
            print ("Extracted answer:",answer )
            print ("Reference final number:",ref_final )
            print ("Predicted final number:",pred_final )
            print ("Correct:",is_correct )
            print ("Reward:",1.0 if is_correct else 0.0 )
            print ("===========================")

        return 1.0 if is_correct else 0.0 


if __name__ =="__main__":

    llms =["gpt-4o-mini","claude-3-7-sonnet-20250219","gemini-1.5-pro","deepseek-ai/DeepSeek-V3"]
    task =MixCGTask (
    llm_names =llms ,
    max_turns =5 ,
    max_tokens =1024 ,
    debug =True ,
    task_ratios ={"countdown":0.5 ,"gsm8k":0.5 }
    )


    obs =task .reset (split ="train")
    print (f"Current task type: {task.current_task_type}")
    done =False 

    while not done :
        action =np .random .rand (len (llms ))
        obs ,reward ,done ,_ =task .step (action )

    print ("Final response:",task .response )
    print ("Reward:",reward )