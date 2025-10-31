

from typing import Dict ,List ,Optional ,Any 
from datasets import load_dataset 
from guf .core import Task 
from guf .utils import extract_answer 


class SearchArenaTask (Task ):


    USER_PROMPT_TEMPLATE =(
    "Compare the following two model responses to the same user query.\n"
    "Response A:\n{response_a}\n\n"
    "Response B:\n{response_b}\n\n"
    "Based on their quality, decide which model performed better, choosing from: model_a, model_b, tie, or tie (bothbad). "
    "Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, "
    "for example <answer> model_a </answer>. Think step by step inside <think> tags, but keep it short."
    )


    MAX_MESSAGE_WORDS =500 

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
    valid_ratio :float =0.5 ,
    max_samples :int =-1 ,
    test_ratio :float =0.2 ,
    use_consultant :bool =True ,
    log_dir :Optional [str ]=None ,
    ):

        super ().__init__ (
        llm_names =llm_names ,
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

        def render (msgs :List [Dict [str ,str ]])->str :
            return "\n".join (f"{m['role']}: {m['content']}"for m in msgs )

        response_a =render (task_data ["messages_a"])
        response_b =render (task_data ["messages_b"])
        return self .user_prompt_template .format (
        response_a =response_a ,
        response_b =response_b 
        )

    def _load_data (
    self ,
    seed :int ,
    split :str ="test",
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


        ds =load_dataset ("lmarena-ai/search-arena-v1-7k",split ="test")


        ds =ds .remove_columns ([
        "timestamp",
        "system_a_metadata",
        "system_b_metadata",
        "conv_metadata"
        ])


        def short_enough (ex :Dict [str ,Any ])->bool :
            word_count =0 
            for side in ("messages_a","messages_b"):
                for m in ex .get (side ,[]):
                    word_count +=len (m .get ("content","").split ())
                    if word_count >self .MAX_MESSAGE_WORDS :
                        return False 
            return True 

        ds =ds .filter (short_enough ,num_proc =1 )
        print (f"Filtered dataset contains {len(ds)} examples after length filter.")


        from guf .utils import get_or_create_indices 

        split_indices =get_or_create_indices (
        task_name ="search_arena",
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
        "test":_take (split_indices ["test"]),
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

    def _calculate_reward (
    self ,
    completions :List [str ],
    task_data :List [Dict [str ,Any ]],
    debug :bool =False 
    )->List [float ]:

        rewards =[]
        for completion ,data in zip (completions ,task_data ):
            answer =extract_answer (completion )
            if answer is None :
                if debug :
                    print ("=== DEBUG: No <answer> tag found ===")
                    print ("True winner:",data ["winner"])
                rewards .append (0.0 )
                continue 

            pred =answer .strip ().lower ()
            true =data ["winner"].strip ().lower ()
            is_correct =(pred ==true )
            if debug :
                print (f"=== DEBUG: SearchArena Eval ===")
                print (f" Predicted: {pred!r}, True: {true!r}, Correct: {is_correct}")
                print ("==============================")
            rewards .append (1.0 if is_correct else 0.0 )

        return rewards 
