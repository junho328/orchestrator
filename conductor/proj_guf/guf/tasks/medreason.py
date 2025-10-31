

import re 
from typing import Dict ,List ,Optional ,Any 
from datasets import load_dataset 
from guf .core import Task 
from guf .utils import extract_answer 


class MedReasonTask (Task ):



    USER_PROMPT_TEMPLATE =(
    "{question}\n"
    "{options}\n\n"
    "Show your work in <THINKING_TAG> </THINKING_TAG> tags. And return the final equation in <answer> </answer> tags, for example "
    "<answer> A </answer>.Provide only the letter corresponding to your choice (A, B, C, or D) within the <answer> tags. "
    "Think step by step inside <THINKING_TAG> tags, but keep it short."
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

        return self .USER_PROMPT_TEMPLATE .replace ("THINKING_TAG",self .thinking_tag )

    def _format_base_prompt (self ,task_data :Dict )->str :
        return self .user_prompt_template .format (
        question =task_data ["question"],
        options =task_data ["options"]
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


        ds =load_dataset ("UCSC-VLAA/MedReason",split ="train")

        ds =ds .select (range (len (ds )))


        from guf .utils import get_or_create_indices 

        split_indices =get_or_create_indices (
        task_name ="medreason",
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


        for split_name in data_splits :

            data_splits [split_name ]=data_splits [split_name ].add_column (
            "task_type",
            ["medreason"]*len (data_splits [split_name ])
            )
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
                    print ("=== DEBUG: No <answer> tag found ===",completion )
                rewards .append (0.0 )
                continue 

            pred =pred .strip ().upper ()
            if pred not in {"A","B","C","D"}and pred :
                pred =pred [0 ]


            opts =re .findall (r'([ABCD])\.\s*([^\n\r]+)',data ["options"])
            letter_map ={ltr :txt .strip ()for ltr ,txt in opts }


            ref =data ["answer"].split (".")[0 ].strip ()


            candidates =[ref ]
            if ":"in ref :
                candidates .append (ref .split (":")[-1 ].strip ())
            if " is "in ref .lower ():

                parts =re .split (r"\s+is\s+",ref ,flags =re .IGNORECASE )
                candidates .append (parts [-1 ].strip ())


            correct_letter =None 
            for cand in candidates :
                for ltr ,txt in letter_map .items ():
                    if txt .lower ()==cand .lower ():
                        correct_letter =ltr 
                        break 
                if correct_letter :
                    break 

            if correct_letter is None and debug :
                print ("=== DEBUG: Could not map ref answer to letter ===")
                print ("Reference piece:",ref )
                print ("Letter map:",letter_map )

            is_correct =(pred ==correct_letter )
            if debug :
                print (f"=== DEBUG: Pred {pred} vs Correct {correct_letter} → {is_correct}")

            rewards .append (1.0 if is_correct else 0.0 )

        return rewards 


if __name__ =="__main__":

    llms =["gpt-4o-mini"]
    task =MedReasonTask (llm_names =llms ,debug =True ,max_samples =20 )
    splits =task ._load_data (seed =0 ,split ="train",validation =True ,valid_ratio =0.1 ,max_samples =20 )
    for split in ("train","valid"):
        sample =splits [split ][0 ]
        print (task ._format_user_prompt (sample ))
        demo ="<think>...</think><answer> A </answer>"
        print ("Reward:",task ._calculate_reward ([demo ],[sample ],debug =True ))