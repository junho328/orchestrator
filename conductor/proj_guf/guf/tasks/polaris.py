

import re 
from typing import Dict ,List ,Optional ,Any 
from datasets import load_dataset 
import numpy as np 
from guf .core import Task 
from guf .utils import extract_answer 


class PolarisTask (Task ):



    USER_PROMPT_TEMPLATE =(
    "Solve the following mathematical problem step by step: {problem} "
    "Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example "
    "<answer>8\\sqrt{{10}}</answer>. "
    "For your final answer, follow these guidelines:\n"
    "1. Provide ONLY the final result without explanations or steps\n"
    "2. Use standard LaTeX notation: \\frac{{a}}{{b}} for fractions, \\sqrt{{n}} for square roots\n"
    "3. For coordinates or tuples, use parentheses: (a,b,c) or \\left(a,b,c\\right)\n"
    "4. For text answers, use \\text{{answer}} when appropriate\n"
    "5. For multiple choice, include the option letter: \\textbf{{(A)}}\n"
    "6. For simple numbers, provide just the value (e.g., <answer>42</answer>)\n"
    "7. For yes/no questions, answer with 'Yes' or 'No'\n"
    "8. Do not use \\boxed{{}} around your answer\n"
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


        problem_text =task_data ["problem"].replace ("{","{{").replace ("}","}}")


        try :
            formatted_prompt =self .user_prompt_template .format (problem =problem_text )
            return formatted_prompt 
        except Exception as e :

            if self .debug :
                print (f"DEBUG: ERROR in formatting: {e}")
                print ("DEBUG: USER_PROMPT_TEMPLATE:",self .user_prompt_template )

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


        ds =load_dataset ("POLARIS-Project/Polaris-Dataset-53K",split ="train")

        ds =ds .select (range (len (ds )))


        from guf .utils import get_or_create_indices 

        split_indices =get_or_create_indices (
        task_name ="polaris-53k",
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

    def _calculate_reward (self ,completions :List [str ],task_data :List [Dict ],debug :bool =False )->List [float ]:

        rewards =[]

        for completion ,data in zip (completions ,task_data ):

            answer =self ._extract_answer_flexible (completion )

            if answer is None :
                if debug :
                    print ("=== DEBUG: No answer found in response ===")
                    print ("Problem:",data ["problem"][:100 ],"...")
                    print ("Reference answer:",data ["answer"])
                    print ("Response snippet:",completion [:200 ],"...")
                rewards .append (0.0 )
                continue 


            pred_answer =self ._normalize_answer (answer )
            ref_answer =self ._normalize_answer (data ["answer"])


            is_correct =pred_answer ==ref_answer 


            if not is_correct :

                is_correct =self ._flexible_comparison (pred_answer ,ref_answer ,debug )

            if debug :
                print ("=== DEBUG: POLARIS Answer Evaluation ===")
                print ("Problem:",data ["problem"][:100 ],"...")
                print ("Reference answer:",data ["answer"])
                print ("Extracted answer:",answer )
                print ("Normalized reference:",ref_answer )
                print ("Normalized prediction:",pred_answer )
                print ("Correct:",is_correct )
                print ("Reward:",1.0 if is_correct else 0.0 )
                print ("===========================")

            rewards .append (1.0 if is_correct else 0.0 )

        return rewards 

    def _extract_answer_flexible (self ,completion :str )->Optional [str ]:

        if not completion :
            return None 


        answer =extract_answer (completion )
        if answer is not None :
            return answer .strip ()



        lines =completion .strip ().split ('\n')
        for line in reversed (lines ):
            line =line .strip ()
            if not line :
                continue 


            if line .lower ().startswith (('therefore','thus','so','hence','finally')):

                parts =line .split (' ',1 )
                if len (parts )>1 :
                    candidate =parts [1 ].strip ()
                    if self ._looks_like_answer (candidate ):
                        return candidate 
                continue 


            if self ._looks_like_answer (line ):
                return line 



        import re 


        equals_matches =re .findall (r'=\s*([^=\n]+?)(?:\s|$|\n)',completion )
        for match in reversed (equals_matches ):
            candidate =match .strip ()
            if self ._looks_like_answer (candidate ):
                return candidate 



        math_patterns =[
        r'\\frac\{[^}]*\}\{[^}]*\}',
        r'\\sqrt(?:\[[^\]]*\])?\{[^}]*\}',
        r'-?\d+(?:\.\d+)?',
        r'\\text\{[^}]*\}',
        r'\([^)]*\)',
        r'[A-Z]'
        ]

        for pattern in math_patterns :
            matches =re .findall (pattern ,completion )
            if matches :
                candidate =matches [-1 ]
                if self ._looks_like_answer (candidate ):
                    return candidate 

        return None 

    def _looks_like_answer (self ,text :str )->bool :

        if not text or len (text .strip ())==0 :
            return False 

        text =text .strip ()


        if len (text )>200 :
            return False 


        import re 
        answer_patterns =[
        r'^-?\d+(?:\.\d+)?$',
        r'\\frac\{',
        r'\\sqrt',
        r'\\text\{',
        r'^\([^)]+\)$',
        r'^[A-Z]$',
        r'^(Yes|No)$',
        r'\\pi|\\theta|\\alpha|\\beta',
        r'\\cdot|\\times|\\div',
        ]

        for pattern in answer_patterns :
            if re .search (pattern ,text ):
                return True 


        exclude_patterns =[
        r'problem|question|given|find|determine|calculate|solve',
        r'let|suppose|assume|consider|since|because',
        r'step|first|second|third|next|then|now',
        r'we have|we get|we need|we want|we can',
        ]

        text_lower =text .lower ()
        for pattern in exclude_patterns :
            if re .search (pattern ,text_lower ):
                return False 

        return True 

    def _normalize_answer (self ,answer :str )->str :

        if self .debug :
            print (f"DEBUG: Starting normalization of answer: '{answer}'")


        normalized =answer .strip ()
        normalized =re .sub (r'^\n+|\n+$','',normalized )


        patterns_to_remove =[
        (r'\\boxed\{([^}]*)\}',r'\1'),
        (r'\\begin\{[^}]*\}(.*?)\\end\{[^}]*\}',r'\1'),
        (r'\\\[(.*?)\\\]',r'\1'),
        (r'\\\((.*?)\\\)',r'\1'),
        (r'\\text\{([^}]*)\}',r'\1'),
        (r'\\textbf\{([^}]*)\}',r'\1'),
        ]

        for pattern ,replacement in patterns_to_remove :
            try :
                normalized =re .sub (pattern ,replacement ,normalized ,flags =re .DOTALL )
            except Exception as e :
                if self .debug :
                    print (f"DEBUG: Error applying pattern {pattern}: {e}")


        normalized =re .sub (r'\s+',' ',normalized ).strip ()


        latex_replacements =[
        (r'\\dfrac',r'\\frac'),
        (r'\\left\(','('),
        (r'\\right\)',')'),
        (r'\\left\[','['),
        (r'\\right\]',']'),
        (r'\\left\{','{'),
        (r'\\right\}','}'),
        ]

        for old ,new in latex_replacements :
            normalized =normalized .replace (old ,new )


        if '='in normalized and ','not in normalized and '('not in normalized :
            try :

                eq_match =re .match (r'^[a-zA-Z]+\s*=\s*(.+)$',normalized )
                if eq_match :
                    normalized =eq_match .group (1 ).strip ()
            except Exception :
                pass 


        if '\\frac'in normalized :
            try :

                def replace_fraction (match ):
                    numerator =match .group (1 )
                    denominator =match .group (2 )
                    return f"({numerator}/{denominator})"


                simple_frac_pattern =r'\\frac\{([^{}]+)\}\{([^{}]+)\}'
                if re .search (simple_frac_pattern ,normalized ):

                    normalized =re .sub (simple_frac_pattern ,replace_fraction ,normalized )
            except Exception as e :
                if self .debug :
                    print (f"DEBUG: Error processing fractions: {e}")


        try :

            if normalized .startswith ('{')and normalized .endswith ('}'):

                brace_count =0 
                can_remove =True 
                for i ,char in enumerate (normalized [1 :-1 ],1 ):
                    if char =='{':
                        brace_count +=1 
                    elif char =='}':
                        brace_count -=1 
                        if brace_count <0 :
                            can_remove =False 
                            break 
                if can_remove and brace_count ==0 :
                    normalized =normalized [1 :-1 ]
        except Exception :
            pass 

        if self .debug :
            print (f"DEBUG: Final normalized answer: '{normalized}'")

        return normalized 

    def _flexible_comparison (self ,pred :str ,ref :str ,debug :bool =False )->bool :


        if pred .lower ()in ['yes','no']or ref .lower ()in ['yes','no']:
            return pred .lower ()==ref .lower ()


        if '('in pred and '('in ref and ')'in pred and ')'in ref :
            try :

                pred_nums =re .findall (r'-?\d+(?:\.\d+)?(?:/\d+)?',pred )
                ref_nums =re .findall (r'-?\d+(?:\.\d+)?(?:/\d+)?',ref )
                if len (pred_nums )==len (ref_nums )and len (pred_nums )>0 :
                    return pred_nums ==ref_nums 
            except Exception :
                pass 


        if self ._check_mathematical_equivalence (pred ,ref ,debug ):
            return True 


        def strip_non_essential (s ):

            s =re .sub (r'[^\w\d\+\-\*/\(\)\{\}\\\.,=<>]','',s )
            return s .lower ()

        stripped_pred =strip_non_essential (pred )
        stripped_ref =strip_non_essential (ref )

        if stripped_pred ==stripped_ref :
            return True 


        def remove_latex (s ):

            s =re .sub (r'\\[a-zA-Z]+\{([^}]*)\}',r'\1',s )
            s =re .sub (r'\\[a-zA-Z]+','',s )
            return s .replace (' ','').lower ()

        clean_pred =remove_latex (pred )
        clean_ref =remove_latex (ref )

        if clean_pred ==clean_ref :
            return True 


        pred_choice =re .search (r'\b([A-Z])\b',pred )
        ref_choice =re .search (r'\b([A-Z])\b',ref )
        if pred_choice and ref_choice :
            return pred_choice .group (1 )==ref_choice .group (1 )


        try :
            pred_num =float (pred .replace (',',''))
            ref_num =float (ref .replace (',',''))
            return abs (pred_num -ref_num )<1e-10 
        except :
            pass 

        return False 

    def _check_mathematical_equivalence (self ,pred :str ,ref :str ,debug :bool =False )->bool :

        try :




            if '\\frac'in pred and '\\frac'in ref :

                pred_fractions =re .findall (r'\\frac\{([^}]*)\}\{([^}]*)\}',pred )
                ref_fractions =re .findall (r'\\frac\{([^}]*)\}\{([^}]*)\}',ref )

                if len (pred_fractions )==1 and len (ref_fractions )==1 :
                    pred_num ,pred_den =pred_fractions [0 ]
                    ref_num ,ref_den =ref_fractions [0 ]


                    if ('\\sqrt'in pred_num and '\\sqrt'in ref_num and 
                    '\\sqrt'in pred_den and '\\sqrt'in ref_den ):

                        pred_sqrt_num =re .search (r'\\sqrt(?:\[[^\]]*\])?\{([^}]*)\}',pred_num )
                        pred_sqrt_den =re .search (r'\\sqrt(?:\[[^\]]*\])?\{([^}]*)\}',pred_den )
                        ref_combined =re .search (r'\\sqrt(?:\[[^\]]*\])?\{([^}]*)\}',ref )

                        if pred_sqrt_num and pred_sqrt_den and ref_combined :

                            expected =f"{pred_sqrt_num.group(1)}/{pred_sqrt_den.group(1)}"
                            if expected ==ref_combined .group (1 ):
                                return True 


            if '^'in pred or '^'in ref :

                pred_norm =pred .replace ('\\cdot','*').replace ('\\times','*')
                ref_norm =ref .replace ('\\cdot','*').replace ('\\times','*')

                if pred_norm ==ref_norm :
                    return True 




            return False 

        except Exception as e :
            if debug :
                print (f"DEBUG: Error in mathematical equivalence check: {e}")
            return False 

if __name__ =="__main__":

    llms =["gpt-4o-mini","claude-3-7-sonnet-20250219"]
    task =PolarisTask (llm_names =llms ,max_turns =3 ,max_tokens =1536 ,debug =True )
    obs =task .reset (split ="test")
    done =False 

    while not done :
        action =np .random .rand (len (llms ))
        obs ,reward ,done ,_ =task .step (action )

    print ("Final response:",task .response )
    print ("Reward:",reward )