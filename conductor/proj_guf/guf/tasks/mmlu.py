

import re 
from typing import Dict ,List ,Optional ,Any 
from datasets import load_dataset 
from guf .core import Task 
from guf .utils import extract_answer 


class MMLUTask (Task ):



    USER_PROMPT_TEMPLATE =(
    "Subject: {subject}\n\n"
    "Question: {question}\n\n"
    "Options:\n{choices}\n\n"
    "Show your work in <THINKING_TAG> </THINKING_TAG> tags. And return the final equation in <answer> </answer> tags, for example "
    "<answer>B</answer>. Provide only the letter corresponding to your choice (A, B, C, or D) within the <answer> tags. "
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
    use_structured_router :bool =False ,
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
        use_structured_router =use_structured_router ,
        use_consultant =use_consultant ,
        log_dir =log_dir ,
        )

    def _get_user_prompt_template (self )->str :

        return self .USER_PROMPT_TEMPLATE .replace ("THINKING_TAG",self .thinking_tag )

    def _format_base_prompt (self ,task_data :Dict )->str :


        choices =task_data ["choices"]
        formatted_choices =""
        for i ,choice in enumerate (choices ):
            letter =chr (ord ('A')+i )
            formatted_choices +=f"{letter}. {choice}\n"


        formatted_choices =formatted_choices .strip ()

        return self .user_prompt_template .format (
        subject =task_data ["subject"],
        question =task_data ["question"],
        choices =formatted_choices 
        )

    def _load_data (
    self ,
    seed :int ,
    split :str ="train",
    validation :bool =False ,
    valid_ratio :float =None ,
    max_samples :int =None ,
    test_split :bool =False ,
    test_ratio :float =None ,
    respect_ratios :bool =True 
    )->Dict [str ,Any ]:


        if valid_ratio is None :
            valid_ratio =self .valid_ratio 
        if max_samples is None :
            max_samples =self .max_samples 
        if test_ratio is None :
            test_ratio =self .test_ratio 


        train_ratio =1.0 -valid_ratio -test_ratio 


        dataset_full =load_dataset ("cais/mmlu","all")


        print (f"MMLU dataset structure: {type(dataset_full)}")
        if hasattr (dataset_full ,'keys'):
            print (f"Available splits: {list(dataset_full.keys())}")


            split_mapping ={
            'train':'auxiliary_train',
            'valid':'validation',
            'test':'test'
            }

            original_splits ={}
            for our_split ,mmlu_split in split_mapping .items ():
                if mmlu_split in dataset_full .keys ():
                    original_splits [our_split ]=dataset_full [mmlu_split ]
                    print (f"Loaded {our_split} from MMLU '{mmlu_split}': {len(original_splits[our_split])} samples")
                else :
                    raise ValueError (f"Required MMLU split '{mmlu_split}' not found in dataset")

            if not respect_ratios :
                data_splits =original_splits 
            else :


                available_train =len (original_splits ['train'])
                available_valid =len (original_splits ['valid'])
                available_test =len (original_splits ['test'])

                print (f"Available samples: train={available_train}, valid={available_valid}, test={available_test}")
                print (f"Target ratios: train={train_ratio:.1f}, valid={valid_ratio:.1f}, test={test_ratio:.1f}")



                max_total_from_train =available_train /train_ratio if train_ratio >0 else float ('inf')
                max_total_from_valid =available_valid /valid_ratio if valid_ratio >0 else float ('inf')
                max_total_from_test =available_test /test_ratio if test_ratio >0 else float ('inf')


                max_total =min (max_total_from_train ,max_total_from_valid ,max_total_from_test )


                final_train =int (max_total *train_ratio )
                final_valid =int (max_total *valid_ratio )
                final_test =int (max_total *test_ratio )

                print (f"Max total samples achievable: {max_total:.0f}")
                print (f"Final samples with correct ratios: train={final_train}, valid={final_valid}, test={final_test}")


                data_splits ={}
                for split_name ,count in [('train',final_train ),('valid',final_valid ),('test',final_test )]:
                    if count >0 :
                        data_splits [split_name ]=original_splits [split_name ].shuffle (seed =seed ).select (range (count ))
                    else :

                        data_splits [split_name ]=original_splits [split_name ].select ([])
        else :
            raise ValueError ("MMLU dataset does not have the expected split structure")


        for k ,v in data_splits .items ():
            if "task_type"not in v .column_names :
                data_splits [k ]=v .add_column (
                "task_type",
                ["mmlu"]*len (v )
                )


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

            pred =extract_answer (completion )
            if pred is None :
                if debug :
                    print ("=== DEBUG: No <answer> tag found ===")
                    print ("Question:",data ["question"][:100 ],"...")
                    print ("Subject:",data ["subject"])
                    print ("Response snippet:",completion [:100 ],"...")
                rewards .append (0.0 )
                continue 


            pred =pred .strip ().upper ()



            letter_match =re .search (r'[ABCD]',pred )
            if letter_match :
                pred_letter =letter_match .group ()
            else :

                number_match =re .search (r'[0123]',pred )
                if number_match :
                    pred_number =int (number_match .group ())
                    pred_letter =chr (ord ('A')+pred_number )
                else :
                    if debug :
                        print (f"=== DEBUG: Could not extract valid answer from: '{pred}' ===")
                    rewards .append (0.0 )
                    continue 



            correct_index =data ["answer"]
            correct_letter =chr (ord ('A')+int (correct_index ))


            is_correct =(pred_letter ==correct_letter )

            if debug :
                print ("=== DEBUG: MMLU Answer Evaluation ===")
                print ("Question:",data ["question"][:100 ],"...")
                print ("Subject:",data ["subject"])
                print ("Choices:",data ["choices"])
                print ("Correct index:",correct_index )
                print ("Correct letter:",correct_letter )
                print ("Extracted answer:",pred )
                print ("Predicted letter:",pred_letter )
                print ("Correct:",is_correct )
                print ("Reward:",1.0 if is_correct else 0.0 )
                print ("===========================")

            rewards .append (1.0 if is_correct else 0.0 )

        return rewards 


if __name__ =="__main__":

    import numpy as np 

    llms =["gpt-4o-mini"]
    task =MMLUTask (llm_names =llms ,debug =True ,max_samples =20 )


    splits =task ._load_data (seed =42 ,split ="all",validation =True ,valid_ratio =0.1 ,max_samples =20 )


    sample =splits ["train"][0 ]
    formatted_prompt =task ._format_base_prompt (sample )
    print ("=== Sample Formatted Prompt ===")
    print (formatted_prompt )
    print ("\n=== Sample Data ===")
    print (f"Subject: {sample['subject']}")
    print (f"Question: {sample['question']}")
    print (f"Choices: {sample['choices']}")
    print (f"Correct answer index: {sample['answer']}")


    demo_responses =[
    "<think>This is about polynomials.</think><answer>B</answer>",
    "<think>Let me think.</think><answer>1</answer>",
    "<think>Reasoning...</think><answer>The answer is A</answer>",
    "<think>Analysis...</think><answer>C.</answer>"
    ]

    test_data =[sample ]*len (demo_responses )
    rewards =task ._calculate_reward (demo_responses ,test_data ,debug =True )
    print (f"\nDemo rewards: {rewards}")