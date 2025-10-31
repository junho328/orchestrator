

import re 
import random 
from typing import Dict ,List ,Optional ,Any 
from datasets import load_dataset ,Dataset 
from guf .core import Task 
from guf .utils import extract_answer 


class HHRLHFTask (Task ):



    USER_PROMPT_TEMPLATE =(
    "You will be shown two conversations between a human and an assistant. Your task is to determine which conversation is better in terms of helpfulness, harmlessness, and overall quality.\n\n"
    "Conversation 1:\n{conversation_1}\n\n"
    "Conversation 2:\n{conversation_2}\n\n"
    "Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example "
    "<answer>0</answer> if Conversation 1 is better, or <answer>1</answer> if Conversation 2 is better. "
    "Your answer should be either 0 or 1, where 0 means Conversation 1 is preferred and 1 means Conversation 2 is preferred. "
    "Think step by step inside <think> tags, but keep it short."
    )

    def __init__ (
    self ,
    llm_names :List [str ],
    seed :int =42 ,
    max_tokens :int =512 ,
    temperature :float =0.1 ,
    max_turns :int =1 ,
    servers :Optional [Dict [str ,str ]]=None ,
    ports :Optional [Dict [str ,int ]]=None ,
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
        return self .user_prompt_template .format (
        conversation_1 =task_data ["conversation_1"],
        conversation_2 =task_data ["conversation_2"]
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


        if valid_ratio is None :
            valid_ratio =self .valid_ratio 
        if max_samples is None :
            max_samples =self .max_samples 
        if test_ratio is None :
            test_ratio =self .test_ratio 


        train_ds =load_dataset ("Anthropic/hh-rlhf",split ="train")
        test_ds =load_dataset ("Anthropic/hh-rlhf",split ="test")

        print (f"Loaded train from HH-RLHF 'train': {len(train_ds)} samples")
        print (f"Loaded test from HH-RLHF 'test': {len(test_ds)} samples")


        def process_dataset (dataset ,dataset_seed ):
            processed_data =[]
            rng =random .Random (dataset_seed )

            for i ,example in enumerate (dataset ):
                chosen =example ["chosen"].strip ()
                rejected =example ["rejected"].strip ()



                item_seed =dataset_seed +i 
                item_rng =random .Random (item_seed )

                if item_rng .random ()<0.5 :
                    conversation_1 =chosen 
                    conversation_2 =rejected 
                    correct_answer =0 
                else :
                    conversation_1 =rejected 
                    conversation_2 =chosen 
                    correct_answer =1 

                processed_data .append ({
                "conversation_1":conversation_1 ,
                "conversation_2":conversation_2 ,
                "correct_answer":correct_answer ,
                "original_chosen":chosen ,
                "original_rejected":rejected 
                })

            return processed_data 


        processed_train =process_dataset (train_ds ,seed )
        processed_test =process_dataset (test_ds ,seed +10000 )


        train_ratio =1.0 -valid_ratio -test_ratio 


        available_train_total =len (processed_train )
        available_test =len (processed_test )

        print (f"Available samples: train_source={available_train_total}, test_source={available_test}")
        print (f"Target ratios: train={train_ratio:.1f}, valid={valid_ratio:.1f}, test={test_ratio:.1f}")


        max_total_from_test =available_test /test_ratio 
        max_total_from_train_source =available_train_total /(train_ratio +valid_ratio )

        max_total_samples =min (max_total_from_test ,max_total_from_train_source )


        final_train_size =int (max_total_samples *train_ratio )
        final_valid_size =int (max_total_samples *valid_ratio )
        final_test_size =int (max_total_samples *test_ratio )

        print (f"Max total samples achievable: {int(max_total_samples)}")
        print (f"Final samples with correct ratios: train={final_train_size}, valid={final_valid_size}, test={final_test_size}")


        from guf .utils import get_or_create_indices 


        needed_from_train_source =final_train_size +final_valid_size 


        train_source_indices =get_or_create_indices (
        task_name ="hh-rlhf-source",
        dataset_len =available_train_total ,
        seed =self .split_seed ,
        valid_ratio =final_valid_size /needed_from_train_source ,
        test_ratio =0.0 
        )


        def _take (data_list :List [Dict ],indices :List [int ],max_size :int =None ):
            selected =[data_list [i ]for i in indices ]
            if max_size is not None and len (selected )>max_size :

                rng =random .Random (seed )
                rng .shuffle (selected )
                selected =selected [:max_size ]
            return selected 


        data_splits ={
        "train":_take (processed_train ,train_source_indices ["train"],final_train_size ),
        "valid":_take (processed_train ,train_source_indices ["valid"],final_valid_size ),
        "test":_take (processed_test ,list (range (len (processed_test ))),final_test_size )
        }


        for split_name in data_splits :
            data_splits [split_name ]=Dataset .from_list (data_splits [split_name ])



        if max_samples !=-1 :


            pass 


        for k ,v in data_splits .items ():
            print (f"Final split {k} contains {len(v)} examples")

        return data_splits 

    def _calculate_reward (
    self ,
    completions :List [str ],
    task_data :List [Dict ],
    debug :bool =False 
    )->List [float ]:

        rewards =[]

        for completion ,data in zip (completions ,task_data ):

            answer =extract_answer (completion )
            if answer is None :
                if debug :
                    print ("=== DEBUG: No <answer> tag found in response ===")
                    print ("Conversation 1 snippet:",data ["conversation_1"][:100 ],"...")
                    print ("Conversation 2 snippet:",data ["conversation_2"][:100 ],"...")
                    print ("Correct answer:",data ["correct_answer"])
                    print ("Response snippet:",completion [:100 ],"...")
                rewards .append (0.0 )
                continue 


            pred_answer =answer .strip ()


            predicted =None 


            if pred_answer in ['0','1']:
                predicted =int (pred_answer )
            elif '0'in pred_answer and '1'not in pred_answer :
                predicted =0 
            elif '1'in pred_answer and '0'not in pred_answer :
                predicted =1 
            else :

                digits =re .findall (r'\d',pred_answer )
                for digit in digits :
                    if digit in ['0','1']:
                        predicted =int (digit )
                        break 


                if predicted is None :
                    try :
                        potential =int (pred_answer )
                        if potential in [0 ,1 ]:
                            predicted =potential 
                    except (ValueError ,TypeError ):
                        pass 

            if predicted is None :
                if debug :
                    print ("=== DEBUG: Could not parse answer as 0 or 1 ===")
                    print ("Extracted answer:",answer )
                    print ("Correct answer:",data ["correct_answer"])
                rewards .append (0.0 )
                continue 


            is_correct =(predicted ==data ["correct_answer"])

            if debug :
                print ("=== DEBUG: HH-RLHF Preference Evaluation ===")
                print ("Conversation 1 snippet:",data ["conversation_1"][:100 ],"...")
                print ("Conversation 2 snippet:",data ["conversation_2"][:100 ],"...")
                print ("Correct answer:",data ["correct_answer"])
                print ("Extracted answer:",answer )
                print ("Predicted choice:",predicted )
                print ("Correct:",is_correct )
                print ("Reward:",1.0 if is_correct else 0.0 )
                print ("===========================")

            rewards .append (1.0 if is_correct else 0.0 )

        return rewards 


if __name__ =="__main__":

    import numpy as np 

    llms =["gpt-4o-mini"]
    task =HHRLHFTask (llm_names =llms ,max_turns =1 ,max_tokens =512 ,debug =True ,max_samples =10 )
    obs =task .reset (split ="test")
    done =False 

    while not done :
        action =np .random .rand (len (llms ))
        obs ,reward ,done ,_ =task .step (action )

    print ("Final response:",task .response )
    print ("Reward:",reward )