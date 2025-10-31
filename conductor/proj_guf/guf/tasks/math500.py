

import re 
from typing import Dict ,List ,Optional ,Any 
from datasets import load_dataset 
import numpy as np 
from guf .core import Task 
from guf .utils import extract_answer 


class MathTask (Task ):



    USER_PROMPT_TEMPLATE =(
    "Solve the following math problem step by step: {problem} "
    "Show your work in <THINKING_TAG> </THINKING_TAG> tags. And return the final equation in <answer> </answer> tags. "
    "For your final answer, follow these guidelines:\n"
    "1. Provide ONLY the final result without explanations, equations, or steps\n"
    "2. For equations with a single solution, provide just the value (e.g., '5' not 'x=5')\n"
    "3. For multiple solutions, list the values separated by commas (e.g., '3, 5, 7' not 'x = 3, 5, 7')\n"
    "4. Use standard LaTeX notation: \\frac{{a}}{{b}} for fractions, \\sqrt{{n}} for square roots\n"
    "5. For angle measurements, include the degree symbol (e.g., '90^\\circ')\n"
    "6. Do not use \\boxed{{}} around your answer\n"
    "Example: <answer>3\\sqrt{{13}}</answer>"
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
    additional_fallbacks :bool =True ,
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

        self .additional_fallbacks =additional_fallbacks 
        print (f"Additional fallbacks: {self.additional_fallbacks}")

    def _get_user_prompt_template (self )->str :

        return self .USER_PROMPT_TEMPLATE .replace ("THINKING_TAG",self .thinking_tag )

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


        ds =load_dataset ("HuggingFaceH4/MATH-500",split ="test")

        ds =ds .select (range (len (ds )))


        from guf .utils import get_or_create_indices 

        split_indices =get_or_create_indices (
        task_name ="math-500",
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
            ["math500"]*len (data_splits [split_name ])
            )



        return data_splits 

    def _calculate_reward (self ,completions :List [str ],task_data :List [Dict ],debug :bool =False )->List [float ]:

        rewards =[]

        for completion ,data in zip (completions ,task_data ):

            answer =extract_answer (completion )
            if answer is None :
                if debug :
                    print ("=== DEBUG: No <answer> tag found in response ===")
                    print ("Problem:",data ["problem"][:100 ],"...")
                    print ("Reference answer:",data ["answer"])
                    print ("Response snippet:",completion [:100 ],"...")
                rewards .append (0.0 )
                continue 


            pred_answer =self ._normalize_answer (answer )
            ref_answer =self ._normalize_answer (data ["answer"])


            is_correct =pred_answer ==ref_answer 


            if not is_correct :


                if ","in pred_answer and ","in ref_answer :
                    pred_values =[v .strip ()for v in pred_answer .split (",")]
                    ref_values =[v .strip ()for v in ref_answer .split (",")]
                    is_correct =sorted (pred_values )==sorted (ref_values )


                if not is_correct and "="in pred_answer :

                    value_after_equals =pred_answer .split ("=")[-1 ].strip ()
                    is_correct =value_after_equals ==ref_answer 


                if not is_correct and "="in pred_answer and ","in pred_answer :
                    values_after_equals =pred_answer .split ("=")[-1 ].strip ()
                    pred_values =[v .strip ()for v in values_after_equals .split (",")]
                    ref_values =[v .strip ()for v in ref_answer .split (",")]
                    is_correct =sorted (pred_values )==sorted (ref_values )

                if self .additional_fallbacks :

                    if not is_correct :
                        pred_no_degree =re .sub (r'\^\s*(\\?circ|circ)','',pred_answer )
                        pred_no_degree =pred_no_degree .replace ('°','')
                        if pred_no_degree ==ref_answer :
                            is_correct =True 


                    if not is_correct :
                        ref_untext =ref_answer 

                        m =re .fullmatch (r'\\text\{(.*)\}',ref_untext )
                        if m :
                            ref_untext =m .group (1 )
                        else :
                            m =re .fullmatch (r'\\text\((.*)\)',ref_untext )
                            if m :
                                ref_untext =m .group (1 )

                        changed =True 
                        while changed and len (ref_untext )>=2 :
                            changed =False 
                            for a ,b in [('(',')'),('[',']'),('{','}')]:
                                if ref_untext .startswith (a )and ref_untext .endswith (b ):
                                    ref_untext =ref_untext [1 :-1 ]
                                    changed =True 
                        if pred_answer ==ref_untext :
                            is_correct =True 

                        if not is_correct and pred_answer .lower ()==ref_untext .lower ():
                            is_correct =True 


                    if not is_correct and ('\\in'in ref_answer or '∈'in ref_answer ):
                        if '\\in'in ref_answer :
                            rhs =ref_answer .split ('\\in')[-1 ].strip ()
                        else :
                            rhs =ref_answer .split ('∈')[-1 ].strip ()
                        if pred_answer ==rhs :
                            is_correct =True 


                    if not is_correct :
                        def _drop_paren_fractions (s :str )->str :
                            return re .sub (r'\(([^()]+\/[^()]+)\)',r'\1',s )
                        p2 =_drop_paren_fractions (pred_answer )
                        r2 =_drop_paren_fractions (ref_answer )
                        if p2 ==r2 :
                            is_correct =True 


                    if not is_correct and '\\pm'in ref_answer :
                        ref_plus =ref_answer .replace ('\\pm','+')
                        ref_minus =ref_answer .replace ('\\pm','-')

                        if pred_answer ==ref_plus or pred_answer ==ref_minus :
                            is_correct =True 

                        elif ','in pred_answer :
                            pred_values =[v .strip ()for v in pred_answer .split (',')if v .strip ()]
                            if sorted (pred_values )==sorted ([ref_plus ,ref_minus ]):
                                is_correct =True 


                    if not is_correct :
                        strip_commas =lambda s :re .sub (r'(,|\\!)','',s )
                        if strip_commas (pred_answer )==strip_commas (ref_answer ):
                            is_correct =True 



            if debug :
                print ("=== DEBUG: MATH-500 Answer Evaluation ===")
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

    def _normalize_answer (self ,answer :str )->str :

        if self .debug :
            print (f"DEBUG: Starting normalization of answer: '{answer}'")


        normalized =answer .strip ()
        normalized =re .sub (r'^\n+|\n+$','',normalized )
        if self .debug :
            print (f"DEBUG: After stripping whitespace: '{normalized}'")



        if '\\boxed'in normalized :
            try :
                old_value =normalized 
                normalized =re .sub (r'\\boxed\{([^}]*)\}',r'\1',normalized )
                if self .debug :
                    print (f"DEBUG: After removing \\boxed: '{normalized}'")
            except Exception as e :
                if self .debug :
                    print (f"DEBUG: Error removing \\boxed: {e}")


        if '\\begin'in normalized :
            try :
                old_value =normalized 
                normalized =re .sub (r'\\begin\{[^}]*\}(.*?)\\end\{[^}]*\}',r'\1',normalized ,flags =re .DOTALL )
                if self .debug :
                    print (f"DEBUG: After removing \\begin/\\end: '{normalized}'")
            except Exception as e :
                if self .debug :
                    print (f"DEBUG: Error removing \\begin/\\end: {e}")


        if '\\['in normalized :
            try :
                old_value =normalized 
                normalized =re .sub (r'\\[([^]]*)\]',r'\1',normalized )
                if self .debug :
                    print (f"DEBUG: After removing \\[...\\]: '{normalized}'")
            except Exception as e :
                if self .debug :
                    print (f"DEBUG: Error removing \\[...\\]: {e}")


        if '\\('in normalized :
            try :
                old_value =normalized 
                normalized =re .sub (r'\\\((.*?)\\\)',r'\1',normalized )
                if self .debug :
                    print (f"DEBUG: After removing \\(...\\): '{normalized}'")
            except Exception as e :
                if self .debug :
                    print (f"DEBUG: Error removing \\(...\\): {e}")


        old_value =normalized 
        normalized =re .sub (r'\s+','',normalized )
        if self .debug :
            print (f"DEBUG: After removing all whitespace: '{normalized}'")


        if '\\dfrac'in normalized :
            try :
                old_value =normalized 
                normalized =re .sub (r'\\dfrac',r'\\frac',normalized )
                if self .debug :
                    print (f"DEBUG: After standardizing fractions: '{normalized}'")
            except Exception as e :
                if self .debug :
                    print (f"DEBUG: Error standardizing fractions: {e}")


        if '\\sqrt{'in normalized :
            try :
                old_value =normalized 


                parts =[]
                i =0 
                while i <len (normalized ):
                    if normalized [i :i +6 ]=='\\sqrt{':

                        i +=6 
                        brace_level =1 
                        content_start =i 
                        while i <len (normalized )and brace_level >0 :
                            if normalized [i ]=='{':
                                brace_level +=1 
                            elif normalized [i ]=='}':
                                brace_level -=1 
                            i +=1 
                        if brace_level ==0 :

                            content =normalized [content_start :i -1 ]
                            parts .append (f"\\sqrt{content}")
                        else :

                            parts .append (normalized [i -len (normalized [i :]):i ])
                    else :
                        parts .append (normalized [i ])
                        i +=1 
                normalized =''.join (parts )
                if self .debug :
                    print (f"DEBUG: After handling square roots: '{normalized}'")
            except Exception as e :
                if self .debug :
                    print (f"DEBUG: Error handling square roots: {e}")

                normalized =old_value 


        if '='in normalized and ','not in normalized :
            try :
                old_value =normalized 
                normalized =re .sub (r'^[a-zA-Z]+\s*=\s*','',normalized )
                if self .debug :
                    print (f"DEBUG: After removing variable assignments: '{normalized}'")
            except Exception as e :
                if self .debug :
                    print (f"DEBUG: Error removing variable assignments: {e}")


        if '\\text'in normalized :
            try :
                old_value =normalized 
                normalized =re .sub (r'\\text\{([^}]*)\}',r'\1',normalized )
                if self .debug :
                    print (f"DEBUG: After standardizing LaTeX text: '{normalized}'")
            except Exception as e :
                if self .debug :
                    print (f"DEBUG: Error standardizing LaTeX text: {e}")


        if '\\frac'in normalized :
            try :
                old_value =normalized 
                if self .debug :
                    print (f"DEBUG: Before handling fractions: '{normalized}'")
                    frac_matches =re .findall (r'\\frac\{([^}]*)\}\{([^}]*)\}',normalized )
                    print (f"DEBUG: Found {len(frac_matches)} fraction matches: {frac_matches}")



                result =[]
                i =0 
                while i <len (normalized ):
                    if normalized [i :i +6 ]=='\\frac{':

                        i +=6 

                        brace_level =1 
                        numerator_start =i 
                        while i <len (normalized )and brace_level >0 :
                            if normalized [i ]=='{':
                                brace_level +=1 
                            elif normalized [i ]=='}':
                                brace_level -=1 
                            i +=1 

                        if brace_level !=0 or i >=len (normalized )or normalized [i ]!='{':

                            if self .debug :
                                print (f"DEBUG: Syntax error in fraction at position {i}")
                            result .append (normalized [numerator_start -6 :i ])
                            continue 

                        numerator =normalized [numerator_start :i -1 ]


                        i +=1 
                        brace_level =1 
                        denominator_start =i 
                        while i <len (normalized )and brace_level >0 :
                            if normalized [i ]=='{':
                                brace_level +=1 
                            elif normalized [i ]=='}':
                                brace_level -=1 
                            i +=1 

                        if brace_level !=0 :

                            if self .debug :
                                print (f"DEBUG: Syntax error in fraction denominator at position {i}")
                            result .append (normalized [numerator_start -6 :i ])
                            continue 

                        denominator =normalized [denominator_start :i -1 ]
                        result .append (f"({numerator}/{denominator})")
                    else :
                        result .append (normalized [i ])
                        i +=1 

                normalized =''.join (result )
                if self .debug :
                    print (f"DEBUG: After converting fractions: '{normalized}'")
            except Exception as e :
                if self .debug :
                    print (f"DEBUG: Error converting fractions: {e}")
                    import traceback 
                    traceback .print_exc ()

                normalized =old_value 


        try :
            old_value =normalized 
            normalized =re .sub (r'\\(left|right|big|Big|bigg|Bigg)',r'',normalized )
            if self .debug :
                print (f"DEBUG: After removing LaTeX formatting: '{normalized}'")
        except Exception as e :
            if self .debug :
                print (f"DEBUG: Error removing LaTeX formatting: {e}")


        try :
            old_value =normalized 
            normalized =re .sub (r'\{([^{}]+)\}',r'\1',normalized )
            if self .debug :
                print (f"DEBUG: After removing braces: '{normalized}'")
        except Exception as e :
            if self .debug :
                print (f"DEBUG: Error removing braces: {e}")

        if self .debug :
            print (f"DEBUG: Final normalized answer: '{normalized}'")

        return normalized 


if __name__ =="__main__":

    llms =["gpt-4o-mini","claude-3-7-sonnet-20250219"]
    task =MathTask (llm_names =llms ,max_turns =3 ,max_tokens =1536 ,debug =True )
    obs =task .reset (split ="test")
    done =False 

    while not done :
        action =np .random .rand (len (llms ))
        obs ,reward ,done ,_ =task .step (action )

    print ("Final response:",task .response )
    print ("Reward:",reward )