

import re 
import json 
from typing import Dict ,List ,Optional ,Any 
from datasets import load_dataset 
import numpy as np 
from guf .core import Task 
from guf .utils import extract_answer 


class DeepMathTask (Task ):



    USER_PROMPT_TEMPLATE =(
    "Solve the following mathematical problem step by step: {question} "
    "Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags. "
    "For your final answer, follow these guidelines:\n"
    "1. Provide ONLY the final result without explanations, equations, or steps\n"
    "2. Use standard LaTeX notation for mathematical expressions with single backslashes (not double)\n"
    "3. Keep the answer as concise as possible\n"
    "4. For limits, derivatives, and integrals, express the final value directly\n"
    "5. For yes/no questions, simply answer 'Yes' or 'No'\n"
    "6. IMPORTANT: Your final answer MUST be enclosed in <answer> </answer> tags\n"
    "7. Format examples: \\delta (not \\\\delta), \\frac{a}{b} (not \\\\frac{a}{b})\n"
    "Example: <answer>\\frac{1}{2}</answer>\n"
    "Think step by step inside <think> tags, but keep it short."
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

        return self .USER_PROMPT_TEMPLATE 

    def _format_base_prompt (self ,task_data :Dict )->str :


        if "question"not in task_data :
            if self .debug :
                print (f"WARNING: Field 'question' not found. Available fields: {list(task_data.keys())}")
            return f"Error: Field 'question' not found in task data."


        question_text =task_data ["question"]


        if isinstance (question_text ,dict )or (question_text is None ):
            question_text =json .dumps (question_text )if isinstance (question_text ,dict )else "No question provided"


        question_text =str (question_text ).replace ("{","{{").replace ("}","}}")


        try :
            return self .user_prompt_template .format (question =question_text )
        except Exception as e :

            try :
                return self .USER_PROMPT_TEMPLATE .replace ("{question}",str (question_text ))
            except Exception :

                return f"Error formatting prompt: {e}. Please check the problem text and format string."

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


        ds =load_dataset ("zwhe99/DeepMath-103K",split ="train")


        from guf .utils import get_or_create_indices 

        split_indices =get_or_create_indices (
        task_name ="deepmath",
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

            valid_cap =int (max_samples *valid_ratio /(1.0 -valid_ratio ))if valid_ratio <1.0 else len (
            data_splits ["valid"])
            data_splits ["valid"]=data_splits ["valid"].shuffle (seed =seed ).select (range (min (valid_cap ,len (data_splits ["valid"]))))

            test_cap =int (max_samples *test_ratio /(1.0 -test_ratio ))if test_ratio <1.0 else len (
            data_splits ["test"])
            data_splits ["test"]=data_splits ["test"].shuffle (seed =seed ).select (range (min (test_cap ,len (data_splits ["test"]))))


        for k ,v in data_splits .items ():
            print (f"Split {k} contains {len(v)} examples")

        return data_splits 

    def _safe_regex_replace (self ,text ,pattern ,replacement ):

        try :
            return re .sub (pattern ,replacement ,text )
        except Exception as e :
            if self .debug :
                print (f"WARNING: Regex replacement failed - pattern: {pattern}, error: {e}")
            return text 

    def _normalize_backslashes (self ,text ):


        text =text .rstrip ('\\')


        try :

            for cmd in ['delta','frac','text','infty','infinity','sin','cos','tan','log','ln','exp']:
                text =text .replace (f'\\\\{cmd}',f'\\{cmd}')


            text =text .replace ('\\\\delta\'\\\\','\\delta\'')
            text =text .replace ('\\\\delta\'','\\delta\'')


            text =text .replace ('-2\\\\delta\'\\\\','-2\\delta\'')
            text =text .replace ('-2\\\\delta\'','-2\\delta\'')


            text =text .replace ('-2\\\\delta','-2\\delta')
        except Exception as e :
            if self .debug :
                print (f"WARNING: Error normalizing backslashes: {e}")

        return text 

    def _normalize_answer (self ,answer :str )->str :

        try :

            normalized =self ._normalize_backslashes (answer )


            normalized =normalized .strip ()
            normalized =re .sub (r'^\n+|\n+$','',normalized )



            normalized =normalized .replace ('\\[','OPEN_BRACKET')
            normalized =normalized .replace ('\\]','CLOSE_BRACKET')
            normalized =normalized .replace ('\\(','OPEN_PAREN')
            normalized =normalized .replace ('\\)','CLOSE_PAREN')


            normalized =self ._safe_regex_replace (normalized ,r'\\boxed\{([^}]*)\}',r'\1')
            normalized =self ._safe_regex_replace (normalized ,r'\\begin\{[^}]*\}(.*?)\\end\{[^}]*\}',r'\1')


            normalized =self ._safe_regex_replace (normalized ,r'\\left\(','(')
            normalized =self ._safe_regex_replace (normalized ,r'\\right\)',')')
            normalized =self ._safe_regex_replace (normalized ,r'\\left\[','[')
            normalized =self ._safe_regex_replace (normalized ,r'\\right\]',']')


            normalized =normalized .replace ('OPEN_BRACKET','\\[')
            normalized =normalized .replace ('CLOSE_BRACKET','\\]')
            normalized =normalized .replace ('OPEN_PAREN','\\(')
            normalized =normalized .replace ('CLOSE_PAREN','\\)')


            normalized =self ._safe_regex_replace (normalized ,r'\s+','')


            normalized =self ._safe_regex_replace (normalized ,r'\\dfrac',r'\\frac')
            normalized =self ._safe_regex_replace (normalized ,r'\\text\{([^}]*)\}',r'\1')


            for func in ['sin','cos','tan','log','ln','exp','infty']:
                normalized =self ._safe_regex_replace (normalized ,r'\\'+func ,func )


            if normalized .lower ()=='yes'or normalized .lower ()=='no':
                normalized =normalized .lower ().capitalize ()


            normalized =normalized .replace ('\\infinity','infinity')
            normalized =normalized .replace ('\\infty','infinity')


            normalized =self ._safe_regex_replace (normalized ,r'\\(left|right|big|Big|bigg|Bigg)',r'')
            normalized =self ._safe_regex_replace (normalized ,r'\{([^{}]+)\}',r'\1')


            normalized =normalized .replace ('\\delta\'','delta_prime')
            normalized =normalized .replace ('-2\\delta\'','-2delta_prime')

            return normalized 

        except Exception as e :

            if self .debug :
                print (f"WARNING: Error in _normalize_answer: {e}")
                print (f"Original answer: {answer}")


            return answer .strip ().replace (" ","")

    def _calculate_reward (self ,completions :List [str ],task_data :List [Dict ],debug :bool =False )->List [float ]:

        rewards =[]

        for completion ,data in zip (completions ,task_data ):
            try :

                answer =extract_answer (completion )
                if answer is None :
                    if debug :
                        print ("=== DEBUG: No <answer> tag found in response ===")
                        print ("Question:",str (data .get ("question",""))[:100 ],"...")
                        print ("Reference answer:",data .get ("final_answer",""))
                        print ("Response snippet:",completion [:100 ],"...")
                    rewards .append (0.0 )
                    continue 


                ref_original =data .get ("final_answer","")


                pred_answer =self ._normalize_answer (answer )
                ref_answer =self ._normalize_answer (ref_original )


                if "-2\\delta\'"in ref_original or "-2delta\'"in ref_original :

                    special_normalize =lambda x :x .replace ('\\\\delta\'','\\delta\'').replace ('-2\\\\delta\'','-2\\delta\'')
                    normalized_pred =special_normalize (answer )
                    normalized_ref =special_normalize (ref_original )


                    if "-2\\delta\'"in normalized_pred and "-2\\delta\'"in normalized_ref :
                        if debug :
                            print ("=== DEBUG: Special case match for -2\\delta' ===")
                        is_correct =True 
                        rewards .append (1.0 )
                        continue 


                is_correct =pred_answer ==ref_answer 


                if not is_correct :

                    if pred_answer .lower ()in ["yes","no"]and ref_answer .lower ()in ["yes","no"]:
                        is_correct =pred_answer .lower ()==ref_answer .lower ()


                    if ("infinity"in pred_answer .lower ()and "infinity"in ref_answer .lower ()):
                        is_correct =True 


                    try :

                        frac_pattern =r'\\frac\{([^{}]+)\}\{([^{}]+)\}'

                        pred_frac_match =re .search (frac_pattern ,answer )
                        ref_frac_match =re .search (frac_pattern ,ref_original )

                        if pred_frac_match and ref_frac_match :
                            pred_num =int (pred_frac_match .group (1 ))
                            pred_denom =int (pred_frac_match .group (2 ))
                            ref_num =int (ref_frac_match .group (1 ))
                            ref_denom =int (ref_frac_match .group (2 ))

                            pred_decimal =pred_num /pred_denom 
                            ref_decimal =ref_num /ref_denom 

                            is_correct =abs (pred_decimal -ref_decimal )<1e-9 
                    except Exception :

                        pass 

                if debug :
                    print ("=== DEBUG: DeepMath Answer Evaluation ===")
                    print ("Question:",str (data .get ("question",""))[:100 ],"...")
                    print ("Reference answer:",ref_original )
                    print ("Extracted answer:",answer )
                    print ("Normalized reference:",ref_answer )
                    print ("Normalized prediction:",pred_answer )
                    print ("Correct:",is_correct )
                    print ("Reward:",1.0 if is_correct else 0.0 )
                    print ("===========================")

                rewards .append (1.0 if is_correct else 0.0 )

            except Exception as e :
                if debug :
                    print (f"ERROR in _calculate_reward: {e}")
                    print (f"Completion: {completion[:100]}...")
                rewards .append (0.0 )

        return rewards 


if __name__ =="__main__":

    llms =["gpt-4o-mini","claude-3-7-sonnet-20250219"]
    task =DeepMathTask (llm_names =llms ,max_turns =3 ,max_tokens =1536 ,debug =True )
    obs =task .reset (split ="train")
    done =False 

    while not done :
        action =np .random .rand (len (llms ))
        obs ,reward ,done ,_ =task .step (action )

    print ("Final response:",task .response )
    print ("Reward:",reward )