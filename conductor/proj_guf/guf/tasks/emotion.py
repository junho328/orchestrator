

import re 
from typing import Dict ,List ,Optional ,Any 
from datasets import load_dataset 
from guf .core import Task 
from guf .utils import extract_answer 


class EmotionTask (Task ):



    EMOTION_LABELS ={
    0 :"sadness",
    1 :"joy",
    2 :"love",
    3 :"anger",
    4 :"fear",
    5 :"surprise"
    }


    EMOTION_TO_NUMBER ={v :k for k ,v in EMOTION_LABELS .items ()}


    USER_PROMPT_TEMPLATE =(
    "Classify the emotion expressed in the following text into one of these six categories: "
    "sadness, joy, love, anger, fear, surprise.\n\n"
    "Text: {text}\n\n"
    "Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example "
    "<answer>love</answer>. Provide only the emotion name (one word) within the <answer> tags. "
    "The emotion must be one of: sadness, joy, love, anger, fear, surprise. "
    "Think step by step inside <think> tags, but keep it short."
    )

    def __init__ (
    self ,
    llm_names :List [str ],
    seed :int =42 ,
    max_tokens :int =512 ,
    temperature :float =0.8 ,
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

        return self .user_prompt_template .format (text =task_data ["text"])

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


        try :

            ds =load_dataset ("dair-ai/emotion",split ="unsplit")
        except :

            ds =load_dataset ("dair-ai/emotion",split ="train")


        ds =ds .select (range (len (ds )))


        from guf .utils import get_or_create_indices 

        split_indices =get_or_create_indices (
        task_name ="emotion",
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
            valid_cap =int (max_samples *valid_ratio /(1.0 -valid_ratio ))if valid_ratio <1.0 else len (data_splits ["valid"])
            data_splits ["valid"]=data_splits ["valid"].shuffle (seed =seed ).select (range (min (valid_cap ,len (data_splits ["valid"]))))
            test_cap =int (max_samples *test_ratio /(1.0 -test_ratio ))if test_ratio <1.0 else len (data_splits ["test"])
            data_splits ["test"]=data_splits ["test"].shuffle (seed =seed ).select (range (min (test_cap ,len (data_splits ["test"]))))


        for k ,v in data_splits .items ():
            print (f"Split {k} contains {len(v)} examples")

        return data_splits 

    def _calculate_reward (
    self ,
    completions :List [str ],
    task_data :List [Dict ],
    debug :bool =False 
    )->List [float ]:

        rewards =[]

        for completion ,data in zip (completions ,task_data ):

            pred =extract_answer (completion )
            if pred is None :
                if debug :
                    print ("=== DEBUG: No <answer> tag found in response ===")
                    print ("Text:",data ["text"][:100 ],"...")
                    print ("Response snippet:",completion [:100 ],"...")
                rewards .append (0.0 )
                continue 


            pred =pred .strip ().lower ()


            ref_label =data ["label"]


            if isinstance (ref_label ,int ):
                ref_emotion =self .EMOTION_LABELS .get (ref_label ,"").lower ()
            else :

                ref_str =str (ref_label ).strip ().lower ()


                if " "in ref_str :

                    parts =ref_str .split ()
                    ref_emotion =parts [-1 ]
                else :

                    try :
                        ref_num =int (ref_str )
                        ref_emotion =self .EMOTION_LABELS .get (ref_num ,"").lower ()
                    except ValueError :

                        ref_emotion =ref_str 


            is_correct =pred ==ref_emotion 


            if not is_correct :

                pred_normalized =pred .replace ("_","").replace ("-","").replace (" ","")
                ref_normalized =ref_emotion .replace ("_","").replace ("-","").replace (" ","")
                is_correct =pred_normalized ==ref_normalized 


                if not is_correct and pred in self .EMOTION_TO_NUMBER :
                    is_correct =pred ==ref_emotion 

            if debug :
                print ("=== DEBUG: Emotion Classification Evaluation ===")
                print ("Text:",data ["text"][:100 ],"...")
                print ("Reference label:",data ["label"])
                print ("Reference emotion:",ref_emotion )
                print ("Extracted prediction:",pred )
                print ("Correct:",is_correct )
                print ("Reward:",1.0 if is_correct else 0.0 )
                print ("===========================")

            rewards .append (1.0 if is_correct else 0.0 )

        return rewards 


if __name__ =="__main__":

    llms =["gpt-4o-mini"]
    task =EmotionTask (llm_names =llms ,max_turns =1 ,max_tokens =512 ,debug =True )
    obs =task .reset (split ="test")
    done =False 

    while not done :
        import numpy as np 

        action =np .random .rand (len (llms ))
        obs ,reward ,done ,_ =task .step (action )

    print ("Final response:",task .response )
    print ("Reward:",reward )