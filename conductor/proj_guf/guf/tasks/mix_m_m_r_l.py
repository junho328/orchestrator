

from typing import Dict ,List ,Any ,Optional 
import numpy as np 
from datasets import concatenate_datasets 
from guf .core import Task 
from guf .tasks .medreason import MedReasonTask 
from guf .tasks .countdown import CountdownTask 
from guf .tasks .math500 import MathTask 
from guf .tasks .rlpr import RLPRTask 
from guf .tasks .livecodebench import LiveCodeBenchTask 
from guf .tasks .mmlu import MMLUTask 


class MixMMRLTask (Task ):



    TASK_FIELD_WHITELIST ={
    "mmlu":["question","choices","answer","subject"],
    "math500":["problem","answer"],
    "rlpr":["rlpr_conversation","ability","ground_truth","style","reward_model"],
    "livecodebench":[
    "question_title","question_content","platform","question_id",
    "contest_id","contest_date","starter_code","difficulty",
    "public_test_cases","private_test_cases","metadata"
    ]
    }

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
    balanced_sampling :bool =True ,
    use_consultant :bool =True ,
    log_dir :Optional [str ]=None ,
    finetune :bool =False ,

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
        self .balanced_sampling =balanced_sampling 
        self .finetune =finetune 


        self .mmlu_task =MMLUTask (
        llm_names =llm_names ,seed =seed ,max_tokens =max_tokens ,
        temperature =temperature ,max_turns =max_turns ,servers =servers ,
        ports =ports ,track_costs =False ,debug =debug ,together =together ,
        valid_ratio =valid_ratio ,test_ratio =test_ratio ,use_consultant =use_consultant ,
        )

        self .math_task =MathTask (
        llm_names =llm_names ,seed =seed ,max_tokens =max_tokens ,
        temperature =temperature ,max_turns =max_turns ,servers =servers ,
        ports =ports ,track_costs =False ,debug =debug ,together =together ,
        valid_ratio =valid_ratio ,test_ratio =test_ratio ,use_consultant =use_consultant ,
        )

        self .rlpr_task =RLPRTask (
        llm_names =llm_names ,seed =seed ,max_tokens =max_tokens ,
        temperature =temperature ,max_turns =max_turns ,servers =servers ,
        ports =ports ,track_costs =False ,debug =debug ,together =together ,
        valid_ratio =valid_ratio ,test_ratio =test_ratio ,use_consultant =use_consultant ,
        )

        self .livecodebench_task =LiveCodeBenchTask (
        llm_names =llm_names ,seed =seed ,max_tokens =max_tokens ,
        temperature =temperature ,max_turns =max_turns ,servers =servers ,
        ports =ports ,track_costs =False ,debug =debug ,together =together ,
        valid_ratio =valid_ratio ,test_ratio =test_ratio ,use_consultant =use_consultant ,
        )

    def _get_user_prompt_template (self )->str :

        return "{problem_content}"

    def _standardize_dataset_schema (self ,dataset ,task_type :str ):

        from datasets import Features ,Value 

        def standardize_sample (sample ):

            result =dict (sample )

            result ["task_type"]=task_type 


            if task_type =="mmlu"and "answer"in sample :

                answer_idx =int (sample ["answer"])
                result ["answer"]=chr (ord ('A')+answer_idx )
            elif task_type in ["rlpr","livecodebench"]and "answer"not in sample :

                result ["answer"]=""
            elif "answer"in sample :

                result ["answer"]=str (sample ["answer"])if sample ["answer"]is not None else ""
            else :

                result ["answer"]=""

            return result 


        result_dataset =dataset .map (standardize_sample )


        target_features =result_dataset .features .copy ()


        target_features ["answer"]=Value ("string")
        target_features ["task_type"]=Value ("string")


        result_dataset =result_dataset .cast (Features (target_features ))

        return result_dataset 

    def _format_base_prompt (self ,task_data :Dict )->str :

        task_type =task_data .get ("task_type","unknown")


        expected_fields =self .TASK_FIELD_WHITELIST .get (task_type ,[])
        clean_task_data ={k :v for k ,v in task_data .items ()if k in expected_fields }

        if task_type =="mmlu":
            return self .mmlu_task ._format_base_prompt (clean_task_data )
        elif task_type =="math500":
            return self .math_task ._format_base_prompt (clean_task_data )
        elif task_type =="rlpr":
            return self .rlpr_task ._format_base_prompt (clean_task_data )
        elif task_type =="livecodebench":
            return self .livecodebench_task ._format_base_prompt (clean_task_data )
        else :
            raise ValueError (f"Unknown task type: {task_type}")

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


        mmlu_splits =self .mmlu_task ._load_data (
        seed =seed ,split =split ,validation =validation ,
        valid_ratio =valid_ratio ,max_samples =-1 ,
        test_split =test_split ,test_ratio =test_ratio ,respect_ratios =True 
        )

        math_splits =self .math_task ._load_data (
        seed =seed ,split =split ,validation =validation ,
        valid_ratio =valid_ratio ,max_samples =-1 ,
        test_split =test_split ,test_ratio =test_ratio 
        )

        rlpr_splits =self .rlpr_task ._load_data (
        seed =seed ,split =split ,validation =validation ,
        valid_ratio =valid_ratio ,max_samples =-1 ,
        test_split =test_split ,test_ratio =test_ratio 
        )

        livecodebench_splits =self .livecodebench_task ._load_data (
        seed =seed ,split =split ,validation =validation ,
        valid_ratio =valid_ratio ,max_samples =-1 ,
        test_split =test_split ,test_ratio =test_ratio 
        )

        combined_splits ={}

        for split_name in ["train","valid","test"]:
            mmlu_splits [split_name ]=self ._standardize_dataset_schema (mmlu_splits [split_name ],"mmlu")
            math_splits [split_name ]=self ._standardize_dataset_schema (math_splits [split_name ],"math500")
            rlpr_splits [split_name ]=self ._standardize_dataset_schema (rlpr_splits [split_name ],"rlpr")
            livecodebench_splits [split_name ]=self ._standardize_dataset_schema (livecodebench_splits [split_name ],"livecodebench")

            mmlu_data =mmlu_splits [split_name ]
            math_data =math_splits [split_name ]
            rlpr_data =rlpr_splits [split_name ]
            livecodebench_data =livecodebench_splits [split_name ]

            use_balanced_sampling =self .balanced_sampling and (split_name =="train")

            if use_balanced_sampling :

                min_samples =min (
                len (mmlu_data ),
                len (math_data ),
                len (rlpr_data ),
                len (livecodebench_data )
                )

                if self .debug :
                    print (f"Split {split_name} - Original sizes: "
                    f"MMLU={len(mmlu_data)}, "
                    f"Math500={len(math_data)}, "
                    f"RLPR={len(rlpr_data)}, "
                    f"LiveCodeBench={len(livecodebench_data)}")
                    print (f"Using {min_samples} samples from each task for balanced sampling")


                np .random .seed (seed )
                mmlu_sampled =mmlu_data .shuffle (seed =seed ).select (range (min_samples ))
                math_sampled =math_data .shuffle (seed =seed +2 ).select (range (min_samples ))
                rlpr_sampled =rlpr_data .shuffle (seed =seed +3 ).select (range (min_samples ))
                livecodebench_sampled =livecodebench_data .shuffle (seed =seed +4 ).select (range (min_samples ))


                combined_data =concatenate_datasets ([
                mmlu_sampled ,
                math_sampled ,
                rlpr_sampled ,
                livecodebench_sampled 
                ])
            else :

                combined_data =concatenate_datasets ([
                mmlu_data ,
                math_data ,
                rlpr_data ,
                livecodebench_data 
                ])


            combined_data =combined_data .shuffle (seed =seed +len (split_name ))


            if split_name =="train"and max_samples >0 and len (combined_data )>max_samples :
                combined_data =combined_data .select (range (max_samples ))

            combined_splits [split_name ]=combined_data 


        for split_name ,data in combined_splits .items ():
            if self .debug :
                task_counts ={}
                for item in data :
                    task_type =item .get ("task_type","unknown")
                    task_counts [task_type ]=task_counts .get (task_type ,0 )+1 

                print (f"Mixed split {split_name} contains {len(data)} examples:")
                for task_type ,count in task_counts .items ():
                    percentage =(count /len (data ))*100 
                    print (f"  {task_type}: {count} ({percentage:.1f}%)")

        if self .finetune :

            if "train"in combined_splits :
                combined_splits ["train"]=combined_splits ["train"].shuffle (seed =seed +123 )


        return combined_splits 

    def _calculate_reward (self ,completions :List [str ],task_data :List [Dict ],debug :bool =False )->List [float ]:

        rewards =[]


        task_groups ={}
        for i ,data in enumerate (task_data ):
            task_type =data .get ("task_type","unknown")
            if task_type not in task_groups :
                task_groups [task_type ]=[]
            task_groups [task_type ].append ((i ,data ,completions [i ]))


        for task_type ,items in task_groups .items ():
            indices ,batch_data ,batch_completions =zip (*items )


            expected_fields =self .TASK_FIELD_WHITELIST .get (task_type ,[])
            clean_batch_data =[]
            for data in batch_data :
                clean_data ={k :v for k ,v in data .items ()if k in expected_fields }
                clean_batch_data .append (clean_data )


            if False :
                print (f"\n=== DEBUG: Processing {task_type} task ===")
                print (f"Number of items: {len(clean_batch_data)}")
                print (f"Expected fields for {task_type}: {expected_fields}")
                if clean_batch_data :
                    sample_data =clean_batch_data [0 ]
                    print (f"Actual fields being passed: {list(sample_data.keys())}")
                    print (f"Sample data preview: {dict(list(sample_data.items())[:3])}")
            elif task_type =="mmlu":
                batch_rewards =self .mmlu_task ._calculate_reward (
                batch_completions ,clean_batch_data ,debug =debug 
                )

            elif task_type =="math500":
                batch_rewards =self .math_task ._calculate_reward (
                batch_completions ,clean_batch_data ,debug =debug 
                )

            elif task_type =="rlpr":
                batch_rewards =self .rlpr_task ._calculate_reward (
                batch_completions ,clean_batch_data ,debug =debug 
                )

            elif task_type =="livecodebench":
                batch_rewards =self .livecodebench_task ._calculate_reward (batch_completions ,clean_batch_data ,debug =debug )

            else :
                batch_rewards =[0.0 ]*len (batch_completions )
                if debug :
                    print (f"Warning: Unknown task type {task_type}, assigning 0 reward")



            for idx ,reward in zip (indices ,batch_rewards ):
                while len (rewards )<=idx :
                    rewards .append (0.0 )
                rewards [idx ]=reward 

        return rewards 


if __name__ =="__main__":

    import numpy as np 

    llms =["gpt-4o-mini"]


    print ("=== Testing Balanced Sampling ===")
    task_balanced =MixMMRLTask (
    llm_names =llms ,
    debug =True ,
    max_samples =50 ,
    balanced_sampling =True 
    )
    obs =task_balanced .reset (split ="test")
    done =False 

    while not done :
        action =np .random .rand (len (llms ))
        obs ,reward ,done ,_ =task_balanced .step (action )

    print ("Final response:",task_balanced .response )
    print ("Reward:",reward )

    print ("\n=== Testing Unbalanced Sampling ===")
    task_unbalanced =MixMMRLTask (
    llm_names =llms ,
    debug =True ,
    max_samples =50 ,
    balanced_sampling =False 
    )
    obs =task_unbalanced .reset (split ="test")
    done =False 

    while not done :
        action =np .random .rand (len (llms ))
        obs ,reward ,done ,_ =task_unbalanced .step (action )

    print ("Final response:",task_unbalanced .response )
    print ("Reward:",reward )