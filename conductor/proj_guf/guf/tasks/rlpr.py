

import re 
import json 
from typing import Dict ,List ,Optional ,Any 
from datasets import load_dataset 
import numpy as np 
from guf .core import Task 
from guf .utils import extract_answer 


class RLPRTask (Task ):



    USER_PROMPT_TEMPLATE =(
    "Subject: {ability}\n\n"
    "{question}\n\n"
    "Show your work in <THINKING_TAG> </THINKING_TAG> tags. And return the final equation in <answer> </answer> tags, for example "
    "<answer>1.99</answer>. "
    "For your final answer, follow these formatting guidelines:\n"
    "1. For numerical values with units, include the appropriate units (e.g., '50400 N', '1.13e4 N.m^2/C')\n"
    "2. For percentages, use decimal format (e.g., '0.1033' for 10.33%)\n"
    "3. For monetary values, use standard decimal format (e.g., '5200' for $5,200)\n"
    "4. For scientific notation, prefer decimal format when possible (e.g., '11300' rather than '1.13e4')\n"
    "5. For formulas and equations, use standard mathematical notation (e.g., 'F=qvB')\n"
    "6. Provide only the final numerical result or formula without units unless specifically required\n"
    "7. Use reasonable precision (2-4 significant figures) to avoid overprecision\n"
    "Think step by step inside <THINKING_TAG> tags, but keep it short."
    )

    def __init__ (
    self ,
    llm_names :List [str ],
    seed :int =42 ,
    max_tokens :int =1536 ,
    temperature :float =0.7 ,
    max_turns :int =3 ,
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


        question =self ._extract_user_question (task_data ["rlpr_conversation"])


        try :
            formatted_prompt =self .user_prompt_template .format (
            ability =task_data ["ability"],
            question =question 
            )
            return formatted_prompt 
        except Exception as e :
            if self .debug :
                print (f"DEBUG: ERROR in formatting: {e}")
                print ("DEBUG: USER_PROMPT_TEMPLATE:",self .user_prompt_template )
            return f"Error formatting prompt: {e}. Please check the problem text and format string."

    def _extract_user_question (self ,prompt :str )->str :

        try :

            if isinstance (prompt ,str ):
                prompt_data =json .loads (prompt )
            else :
                prompt_data =prompt 


            for message in prompt_data :
                if message .get ("role")=="user":
                    return message .get ("content","")

            return "No user question found"
        except (json .JSONDecodeError ,TypeError )as e :
            if self .debug :
                print (f"DEBUG: Error parsing prompt: {e}")
            return str (prompt )

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


        ds =load_dataset ("openbmb/RLPR-Train-Dataset",split ="train")

        ds =ds .select (range (len (ds )))

        if 'prompt'in ds .column_names :
            ds =ds .rename_column ('prompt','rlpr_conversation')


        def process_example (example ):
            try :

                if isinstance (example ["reward_model"],str ):
                    reward_data =json .loads (example ["reward_model"])
                else :
                    reward_data =example ["reward_model"]

                example ["ground_truth"]=reward_data .get ("ground_truth","")
                example ["style"]=reward_data .get ("style","rule")
            except (json .JSONDecodeError ,TypeError ):
                example ["ground_truth"]=""
                example ["style"]="rule"

            return example 


        ds =ds .map (process_example )


        def filter_long_answers (example ):
            ground_truth =example .get ("ground_truth","")
            return len (ground_truth )<=30 

        ds =ds .filter (filter_long_answers )

        if self .debug :
            print (f"After filtering long answers (>30 chars): {len(ds)} examples remaining")


        from guf .utils import get_or_create_indices 

        split_indices =get_or_create_indices (
        task_name ="rlpr",
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


        for split_name in data_splits :

            data_splits [split_name ]=data_splits [split_name ].add_column (
            "task_type",
            ["rlpr"]*len (data_splits [split_name ])
            )


        if max_samples !=-1 :
            data_splits ["train"]=data_splits ["train"].shuffle (seed =seed ).select (range (min (max_samples ,len (data_splits ["train"]))))
            valid_cap =int (max_samples *valid_ratio /(1.0 -valid_ratio ))if valid_ratio <1.0 else len (data_splits ["valid"])
            data_splits ["valid"]=data_splits ["valid"].shuffle (seed =seed ).select (range (min (valid_cap ,len (data_splits ["valid"]))))
            test_cap =int (max_samples *test_ratio /(1.0 -test_ratio ))if test_ratio <1.0 else len (data_splits ["test"])
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
                    print ("Ground truth:",data .get ("ground_truth",""))
                    print ("Response snippet:",completion [:100 ],"...")
                rewards .append (0.0 )
                continue 


            pred_answer =self ._normalize_answer (answer )
            ref_answer =self ._normalize_answer (data .get ("ground_truth",""))


            is_correct =pred_answer ==ref_answer 


            if not is_correct :

                is_correct =self ._flexible_match (pred_answer ,ref_answer )

            if debug :
                print ("=== DEBUG: RLPR Answer Evaluation ===")
                print ("Subject:",data .get ("ability","Unknown"))
                print ("Ground truth:",data .get ("ground_truth",""))
                print ("Extracted answer:",answer )
                print ("Normalized reference:",ref_answer )
                print ("Normalized prediction:",pred_answer )


                if not is_correct :
                    pred_num =self ._extract_number_from_string (pred_answer )
                    ref_num =self ._extract_number_from_string (ref_answer )
                    pred_currency =self ._convert_currency_to_standard (pred_answer )
                    ref_currency =self ._convert_currency_to_standard (ref_answer )

                    print ("DEBUG: Extracted numbers:",pred_num ,"vs",ref_num )
                    print ("DEBUG: Currency conversion:",pred_currency ,"vs",ref_currency )


                    if '%'in pred_answer or '%'in ref_answer :
                        print ("DEBUG: Percentage detection in pred/ref:",'%'in pred_answer ,'/','%'in ref_answer )

                print ("Correct:",is_correct )
                print ("Reward:",1.0 if is_correct else 0.0 )
                print ("===========================")

            rewards .append (1.0 if is_correct else 0.0 )

        return rewards 

    def _normalize_answer (self ,answer :str )->str :

        if not answer :
            return ""


        normalized =answer .strip ()
        normalized =re .sub (r'^\n+|\n+$','',normalized )


        normalized =re .sub (r'\s+',' ',normalized ).strip ()

        return normalized 

    def _flexible_match (self ,pred :str ,ref :str )->bool :


        if pred ==ref :
            return True 


        if pred .lower ()==ref .lower ():
            return True 


        try :

            if '%'in pred and '%'not in ref :
                pred_num =float (pred .replace ('%',''))/100 
                ref_num =float (ref )
                return abs (pred_num -ref_num )<1e-6 
            elif '%'in ref and '%'not in pred :
                ref_num =float (ref .replace ('%',''))/100 
                pred_num =float (pred )
                return abs (pred_num -ref_num )<1e-6 
        except (ValueError ,TypeError ):
            pass 


        try :
            pred_num =float (pred .replace (',','').replace ('$','').replace ('%',''))
            ref_num =float (ref .replace (',','').replace ('$','').replace ('%',''))

            tolerance =max (1e-6 ,abs (ref_num )*1e-3 )if abs (ref_num )>1 else 1e-6 
            return abs (pred_num -ref_num )<tolerance 
        except (ValueError ,TypeError ):
            pass 


        pred_clean =self ._extract_number_from_string (pred )
        ref_clean =self ._extract_number_from_string (ref )
        if pred_clean is not None and ref_clean is not None :

            tolerance =max (1e-6 ,abs (ref_clean )*1e-3 )if abs (ref_clean )>1 else 1e-6 
            return abs (pred_clean -ref_clean )<tolerance 


        try :
            pred_millions =self ._convert_currency_to_standard (pred )
            ref_millions =self ._convert_currency_to_standard (ref )
            if pred_millions is not None and ref_millions is not None :
                tolerance =max (1e-6 ,abs (ref_millions )*1e-3 )if abs (ref_millions )>1 else 1e-6 
                return abs (pred_millions -ref_millions )<tolerance 
        except (ValueError ,TypeError ):
            pass 

        return False 

    def _extract_number_from_string (self ,text :str )->Optional [float ]:

        import re 
        try :

            clean_text =re .sub (r'[A-Za-z/\^\.\s]+$','',text .strip ())
            clean_text =clean_text .replace ('$','').replace (',','').replace ('%','')


            if 'e'in clean_text .lower ():
                return float (clean_text )


            match =re .search (r'-?\d+\.?\d*',clean_text )
            if match :
                return float (match .group ())
        except (ValueError ,TypeError ):
            pass 
        return None 

    def _convert_currency_to_standard (self ,text :str )->Optional [float ]:

        text =text .lower ().strip ()
        try :

            if 'million'in text :
                number_part =text .replace ('$','').replace ('million','').replace (',','').strip ()
                return float (number_part )*1_000_000 


            elif 'thousand'in text or 'k'in text :
                number_part =text .replace ('$','').replace ('thousand','').replace ('k','').replace (',','').strip ()
                return float (number_part )*1_000 


            elif '$'in text or text .replace ('.','').replace (',','').isdigit ():
                number_part =text .replace ('$','').replace (',','').strip ()
                return float (number_part )

        except (ValueError ,TypeError ):
            pass 
        return None 


if __name__ =="__main__":

    print ("=== RLPR Task Testing ===")


    llms =["gpt-4o-mini","claude-3-7-sonnet-20250219"]


    task =RLPRTask (
    llm_names =llms ,
    max_turns =3 ,
    max_tokens =1536 ,
    debug =True ,
    max_samples =20 
    )

    print ("Task initialized successfully!")
    print (f"LLM names: {task.llm_names}")
    print (f"Max turns: {task.max_turns}")
    print (f"Max tokens: {task.max_tokens}")


    print ("\n=== Testing Data Loading ===")
    try :
        data_splits =task ._load_data (
        seed =42 ,
        split ="train",
        validation =True ,
        valid_ratio =0.2 ,
        max_samples =50 
        )

        print ("Data loading successful!")
        for split_name ,split_data in data_splits .items ():
            print (f"{split_name}: {len(split_data)} examples")


            if len (split_data )>0 :
                example =split_data [0 ]
                print (f"  Example - Subject: {example.get('ability', 'Unknown')}")
                print (f"  Ground truth: {example.get('ground_truth', 'None')}")

    except Exception as e :
        print (f"Error loading data: {e}")
        import traceback 
        traceback .print_exc ()


    print ("\n=== Testing Prompt Formatting ===")
    try :
        if 'data_splits'in locals ()and len (data_splits ['test'])>0 :
            test_example =data_splits ['test'][0 ]
            formatted_prompt =task ._format_user_prompt (test_example )
            print ("Formatted prompt preview:")
            print (formatted_prompt [:500 ]+"..."if len (formatted_prompt )>500 else formatted_prompt )
    except Exception as e :
        print (f"Error formatting prompt: {e}")


    print ("\n=== Testing Reward Calculation ===")
    test_cases =[

    ("0.084","8.4%","Percentage conversion"),
    ("8.4%","0.084","Reverse percentage conversion"),


    ("2435.86","2.44e3","Scientific notation"),
    ("1.13e4","11300","Scientific to decimal"),


    ("$0.7617 million","$761700","Currency millions"),
    ("$112,480.11","$112480.50","Currency precision"),


    ("782 A","782","Units vs no units"),
    ("70.9246 amu","71.00","Precision difference"),


    ("3.083","3.083","Exact match"),
    ("F=qvB","F=qvB","Formula match"),
    ]

    for pred ,ref ,description in test_cases :

        mock_data =[{"ground_truth":ref ,"ability":"Test"}]
        mock_completions =[f"<answer>{pred}</answer>"]

        rewards =task ._calculate_reward (mock_completions ,mock_data ,debug =True )
        print (f"{description}: {pred} vs {ref} -> Reward: {rewards[0]}")
        print ("-"*50 )


    print ("\n=== Testing Environment ===")
    try :
        obs =task .reset (split ="test")
        print ("Environment reset successful!")
        print (f"Initial observation length: {len(obs)}")


        action =np .random .rand (len (llms ))
        obs ,reward ,done ,info =task .step (action )

        print (f"Step completed - Reward: {reward}, Done: {done}")
        print (f"Final response preview: {task.response[:100] if task.response else 'None'}...")

    except Exception as e :
        print (f"Error in environment testing: {e}")
        import traceback 
        traceback .print_exc ()

    print ("\n=== Testing Complete ===")