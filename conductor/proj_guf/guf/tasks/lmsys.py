

import pandas as pd 
import re 
import time 
from typing import Dict ,List ,Any ,Optional 
from datasets import Dataset 
from guf .core import Task 
from guf .utils import query_llm 
import numpy as np 
import os 


class LMSYSJudgeTask (Task ):



    USER_PROMPT_TEMPLATE ="{prompt}"

    JUDGE_SYSTEM_PROMPT =(
    "You are an expert judge evaluating the quality of AI responses. "
    "You will be shown a user prompt and two responses: one from a router system "
    "that coordinates multiple AI agents, and one from a single AI model. "
    "Your task is to determine which response is better overall in terms of "
    "helpfulness, accuracy, clarity, and completeness."
    )

    JUDGE_USER_PROMPT ="""
Please evaluate these two responses to the following prompt:

**User Prompt:**
{prompt}

**Response A (Router System):**
{router_response}

**Response B (Single Model):**
{baseline_response}

Please compare these responses and determine which is better. Consider:
- Helpfulness and relevance to the prompt
- Accuracy of information
- Clarity and organization
- Completeness of the answer

Respond with exactly one of these three options:
- "router_win" if Response A (Router System) is better
- "router_lose" if Response B (Single Model) is better  
- "tie" if both responses are roughly equal in quality

Your response:"""

    def __init__ (
    self ,
    llm_names :List [str ],
    baseline_llm :str ="gpt-4o-mini",
    judge_llm :str ="gpt-4o-mini",
    data_path :str ="your/path/here/train.csv",
    seed :int =42 ,
    max_tokens :int =1024 ,
    temperature :float =0.7 ,
    max_turns :int =3 ,
    servers :Dict [str ,str ]=None ,
    ports :Dict [str ,int ]=None ,
    track_costs :bool =True ,
    debug :bool =False ,
    together :bool =True ,
    valid_ratio :float =0.2 ,
    max_samples :int =-1 ,
    test_ratio :float =0.2 ,
    use_structured_router :bool =False ,
    judge_max_retries :int =3 ,
    judge_retry_delay :float =1.0 ,
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
        use_structured_router =use_structured_router ,
        )

        self .baseline_llm =baseline_llm 
        self .judge_llm =judge_llm 
        self .data_path =data_path 
        self .judge_max_retries =judge_max_retries 
        self .judge_retry_delay =judge_retry_delay 


        self ._current_baseline_response =None 
        self ._current_prompt =None 

        if debug :
            print (f"[LMSYSJudgeTask] Initialized with:")
            print (f"  Router agents: {llm_names}")
            print (f"  Baseline LLM: {baseline_llm}")
            print (f"  Judge LLM: {judge_llm}")
            print (f"  Data path: {data_path}")
            print (f"  Judge retries: {judge_max_retries}")

    def _get_user_prompt_template (self )->str :

        return self .USER_PROMPT_TEMPLATE 

    def _format_base_prompt (self ,task_data :Dict )->str :

        return self .USER_PROMPT_TEMPLATE .format (prompt =task_data ["prompt"])

    def _load_data (self ,seed :int ,split :str ="train",validation :bool =False ,
    valid_ratio :float =None ,max_samples :int =None ,
    test_split :bool =False ,test_ratio :float =None )->Dict :


        if valid_ratio is None :
            valid_ratio =self .valid_ratio 
        if max_samples is None :
            max_samples =self .max_samples 
        if test_ratio is None :
            test_ratio =self .test_ratio 


        if not os .path .exists (self .data_path ):
            raise FileNotFoundError (f"LMSYS data file not found: {self.data_path}")

        df =pd .read_csv (self .data_path )

        if "prompt"not in df .columns :
            raise ValueError (f"Expected 'prompt' column in {self.data_path}")


        dataset_dict ={"prompt":df ["prompt"].tolist ()}


        dataset_dict ["index"]=list (range (len (df )))


        ds =Dataset .from_dict (dataset_dict )

        if self .debug :
            print (f"[LMSYSJudgeTask] Loaded {len(ds)} prompts from {self.data_path}")


        from guf .utils import get_or_create_indices 
        split_indices =get_or_create_indices (
        task_name ="lmsys_judge",
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
            np .random .seed (seed )
            for split_name in ["train","valid","test"]:
                split_data =data_splits [split_name ]
                if len (split_data )>max_samples :
                    indices =np .random .choice (len (split_data ),max_samples ,replace =False )
                    data_splits [split_name ]=split_data .select (indices .tolist ())


        if self .debug :
            for k ,v in data_splits .items ():
                print (f"[LMSYSJudgeTask] Split {k} contains {len(v)} examples")

        return data_splits 

    def reset (self ,task_id :int =-1 ,split :str ="train"):


        obs =super ().reset (task_id ,split )


        current_data =self .data_split [self .task_id ]
        self ._current_prompt =current_data ["prompt"]
        self ._current_baseline_response =self ._get_baseline_response (self ._current_prompt )

        if self .debug :
            print (f"[LMSYSJudgeTask] Generated baseline response for task {task_id}")
            print (f"  Prompt: {self._current_prompt[:100]}...")
            print (f"  Baseline: {self._current_baseline_response[:100]}...")

        return obs 

    def _get_baseline_response (self ,prompt :str )->str :

        messages =[
        {"role":"user","content":prompt }
        ]

        try :
            response =query_llm (
            model =self .baseline_llm ,
            messages =messages ,
            max_tokens =self .max_tokens ,
            temperature =self .temperature ,
            server =self .servers .get (self .baseline_llm )if self .servers else None ,
            port =self .ports .get (self .baseline_llm )if self .ports else None ,
            debug =self .debug ,
            together =self .together 
            )
            return response 
        except Exception as e :
            if self .debug :
                print (f"[LMSYSJudgeTask] Error getting baseline response: {e}")
            return "Error: Could not generate baseline response"

    def _get_judge_decision_with_retry (self ,prompt :str ,router_response :str ,baseline_response :str )->str :

        for attempt in range (self .judge_max_retries ):
            try :
                judge_messages =[
                {"role":"system","content":self .JUDGE_SYSTEM_PROMPT },
                {"role":"user","content":self .JUDGE_USER_PROMPT .format (
                prompt =prompt ,
                router_response =router_response ,
                baseline_response =baseline_response 
                )}
                ]

                judge_response =query_llm (
                model =self .judge_llm ,
                messages =judge_messages ,
                max_tokens =100 ,
                temperature =0.1 ,
                server =self .servers .get (self .judge_llm )if self .servers else None ,
                port =self .ports .get (self .judge_llm )if self .ports else None ,
                debug =self .debug ,
                together =self .together 
                )


                judge_response_clean =judge_response .lower ().strip ()


                judge_response_clean =re .sub (r'[^\w\s]',' ',judge_response_clean )
                judge_response_clean =' '.join (judge_response_clean .split ())

                if "router_win"in judge_response_clean or "router win"in judge_response_clean :
                    return "router_win"
                elif "router_lose"in judge_response_clean or "router lose"in judge_response_clean :
                    return "router_lose"
                elif "tie"in judge_response_clean :
                    return "tie"
                else :

                    if "response a"in judge_response_clean and ("better"in judge_response_clean or "superior"in judge_response_clean ):
                        return "router_win"
                    elif "response b"in judge_response_clean and ("better"in judge_response_clean or "superior"in judge_response_clean ):
                        return "router_lose"
                    elif "equal"in judge_response_clean or "same"in judge_response_clean :
                        return "tie"
                    else :

                        if attempt <self .judge_max_retries -1 :
                            if self .debug :
                                print (f"[LMSYSJudgeTask] Unclear judge response (attempt {attempt + 1}): {judge_response}")
                                print (f"[LMSYSJudgeTask] Retrying in {self.judge_retry_delay} seconds...")
                            time .sleep (self .judge_retry_delay )
                            continue 
                        else :

                            if self .debug :
                                print (f"[LMSYSJudgeTask] All judge attempts failed. Final response: {judge_response}")
                            return "tie"

            except Exception as e :
                if attempt <self .judge_max_retries -1 :
                    if self .debug :
                        print (f"[LMSYSJudgeTask] Judge error (attempt {attempt + 1}): {e}")
                        print (f"[LMSYSJudgeTask] Retrying in {self.judge_retry_delay} seconds...")
                    time .sleep (self .judge_retry_delay )
                    continue 
                else :
                    if self .debug :
                        print (f"[LMSYSJudgeTask] All judge attempts failed with error: {e}")
                    return "tie"


        return "tie"

    def step (self ,action ,sampling :bool =False ):


        obs ,_ ,done ,obs_act =super ().step (action ,sampling )



        if done :

            final_reward =self ._calculate_final_reward ()
            return obs ,final_reward ,done ,obs_act 
        else :

            return obs ,0.0 ,done ,obs_act 

    def _calculate_final_reward (self )->float :

        if not self .response or not self ._current_baseline_response :
            if self .debug :
                print ("[LMSYSJudgeTask] Missing response for final evaluation")
            return 0.0 


        judge_decision =self ._get_judge_decision_with_retry (
        self ._current_prompt ,
        self .response ,
        self ._current_baseline_response 
        )


        if judge_decision =="router_win":
            reward =1.0 
        elif judge_decision =="tie":
            reward =0.5 
        else :
            reward =0.0 

        if self .debug :
            print ("=== DEBUG: Final Judge-based Evaluation ===")
            print (f"Prompt: {self._current_prompt[:100]}..."if len (self ._current_prompt )>100 else f"Prompt: {self._current_prompt}")
            print (f"Router Response: {self.response[:200]}..."if len (self .response )>200 else f"Router Response: {self.response}")
            print (f"Baseline Response: {self._current_baseline_response[:200]}..."if len (self ._current_baseline_response )>200 else f"Baseline Response: {self._current_baseline_response}")
            print (f"Judge Decision: {judge_decision}")
            print (f"Final Reward: {reward}")
            print ("="*40 )

        return reward 

    def _calculate_reward (self ,completions :List [str ],task_data :List [Dict [str ,Any ]],debug :bool =False )->List [float ]:


        return [0.0 ]*len (completions )