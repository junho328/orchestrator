

import re 
import json 
from typing import Dict ,List ,Any ,Optional 
from datasets import load_dataset ,Dataset ,concatenate_datasets 
import numpy as np 
from guf .core import Task 
from guf .utils import extract_answer 
from guf .cost import reset_costs 


class MixCDTask (Task ):



    COUNTDOWN_PROMPT_TEMPLATE =(
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) one or multiple times but "
    "each number can only be used once. Show your work in <think> </think> tags. "
    "And return the final equation in <answer> </answer> tags, for example "
    "<answer> (1 + 2) / 3 </answer>; do not include = in the <answer> tags, only the equation. "
    "Think step by step inside <think> tags, but keep it short."
    )

    DEEPMATH_PROMPT_TEMPLATE =(
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
    temperature :float =0.8 ,
    max_turns :int =5 ,
    include_format_reward :bool =True ,
    task_ratios :Dict [str ,float ]=None ,
    balanced_sampling :bool =True ,
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


        if task_ratios is None :
            self .task_ratios ={"countdown":0.5 ,"deepmath":0.5 }
        else :
            self .task_ratios =task_ratios 


        if self .balanced_sampling :
            total =sum (self .task_ratios .values ())
            if total !=1.0 :
                for k in self .task_ratios :
                    self .task_ratios [k ]/=total 


        self .current_task_type ="mixed"
        self .include_format_reward =include_format_reward 

    def _get_user_prompt_template (self )->str :



        if not hasattr (self ,'current_task_type')or self .current_task_type in ['mixed',None ]:

            return self .DEEPMATH_PROMPT_TEMPLATE 

        if self .current_task_type =="countdown":
            return self .COUNTDOWN_PROMPT_TEMPLATE 
        else :
            return self .DEEPMATH_PROMPT_TEMPLATE 

    def _format_base_prompt (self ,task_data :Dict )->str :


        self .current_task_type =task_data .get ("task_type","deepmath")

        if self .current_task_type =="countdown":
            return self .COUNTDOWN_PROMPT_TEMPLATE .format (
            numbers =task_data ["nums"],
            target =task_data ["target"],
            )
        else :

            question_text =task_data .get ("question","")


            if isinstance (question_text ,dict )or (question_text is None ):
                question_text =json .dumps (question_text )if isinstance (question_text ,dict )else "No question provided"


            question_text =str (question_text ).replace ("{","{{").replace ("}","}}")


            try :
                return self .DEEPMATH_PROMPT_TEMPLATE .format (question =question_text )
            except Exception as e :

                try :
                    return self .DEEPMATH_PROMPT_TEMPLATE .replace ("{question}",str (question_text ))
                except Exception :

                    return f"Error formatting prompt: {e}. Please check the problem text and format string."

    def get_balanced_sample_sizes (self ,available_countdown :int ,available_deepmath :int ,
    target_total :int =None )->tuple :

        countdown_ratio =self .task_ratios .get ("countdown",0.5 )
        deepmath_ratio =self .task_ratios .get ("deepmath",0.5 )

        if target_total is None :

            max_countdown_from_ratio =int (available_countdown /countdown_ratio )
            max_deepmath_from_ratio =int (available_deepmath /deepmath_ratio )
            target_total =min (max_countdown_from_ratio ,max_deepmath_from_ratio )


        target_countdown =int (target_total *countdown_ratio )
        target_deepmath =int (target_total *deepmath_ratio )


        actual_countdown =min (target_countdown ,available_countdown )
        actual_deepmath =min (target_deepmath ,available_deepmath )


        if actual_countdown <target_countdown :

            scaling_factor =actual_countdown /target_countdown 
            actual_deepmath =int (target_deepmath *scaling_factor )
            limiting_factor ="countdown"
        elif actual_deepmath <target_deepmath :

            scaling_factor =actual_deepmath /target_deepmath 
            actual_countdown =int (target_countdown *scaling_factor )
            limiting_factor ="deepmath"
        else :
            limiting_factor =None 

        if self .debug and limiting_factor :
            print (f"[MixCDTask] {limiting_factor} dataset is the limiting factor for balanced sampling")

        return actual_countdown ,actual_deepmath 

    def _load_data (
    self ,
    seed :int ,
    split :str ="train",
    validation :bool =False ,
    valid_ratio :float =None ,
    max_samples :int =None ,
    test_split :bool =False ,
    test_ratio :float =None 
    )->Dict [str ,Dataset ]:

        from datasets import load_dataset ,concatenate_datasets 
        import numpy as np 



        deepmath_raw =load_dataset ("zwhe99/DeepMath-103K",split ="train")
        countdown_raw =load_dataset ("Jiayi-Pan/Countdown-Tasks-3to4",split ="train")

        deepmath_raw =deepmath_raw .map (lambda x :{"task_type":"deepmath"})
        countdown_raw =countdown_raw .map (lambda x :{"task_type":"countdown"})

        print (f"[MixCDTask] Raw dataset sizes - DeepMath: {len(deepmath_raw)}, Countdown: {len(countdown_raw)}")

        if self .balanced_sampling :
            print (f"[MixCDTask] Using BALANCED sampling with ratios - "
            f"Countdown: {self.task_ratios.get('countdown', 0.5):.1%}, "
            f"DeepMath: {self.task_ratios.get('deepmath', 0.5):.1%}")



            target_total =max_samples if max_samples is not None and max_samples >0 else None 
            actual_countdown_samples ,actual_deepmath_samples =self .get_balanced_sample_sizes (
            len (countdown_raw ),len (deepmath_raw ),target_total 
            )

            print (f"[MixCDTask] Balanced sampling - Countdown: {actual_countdown_samples}, DeepMath: {actual_deepmath_samples}")



            np .random .seed (seed )


            countdown_indices =np .random .choice (len (countdown_raw ),actual_countdown_samples ,replace =False )
            deepmath_indices =np .random .choice (len (deepmath_raw ),actual_deepmath_samples ,replace =False )


            countdown_subset =countdown_raw .select (countdown_indices .tolist ())
            deepmath_subset =deepmath_raw .select (deepmath_indices .tolist ())



            combined_raw =concatenate_datasets ([countdown_subset ,deepmath_subset ])
            total_len =len (combined_raw )

            print (f"[MixCDTask] Final balanced dataset size: {total_len} "
            f"(Countdown: {actual_countdown_samples}, DeepMath: {actual_deepmath_samples})")


            countdown_ratio =self .task_ratios .get ("countdown",0.5 )
            deepmath_ratio =self .task_ratios .get ("deepmath",0.5 )
            balanced_task_name =(f"mix_c_d_balanced_c{actual_countdown_samples}_d{actual_deepmath_samples}_"
            f"cr{countdown_ratio:.3f}_dr{deepmath_ratio:.3f}_total{total_len}")
            task_name =balanced_task_name 

        else :
            print (f"[MixCDTask] Using UNBALANCED sampling - concatenating all available data")



            combined_raw =concatenate_datasets ([deepmath_raw ,countdown_raw ])
            total_len =len (combined_raw )


            if max_samples is not None and max_samples >0 and max_samples <total_len :
                np .random .seed (seed )
                selected_indices =np .random .choice (total_len ,max_samples ,replace =False )
                combined_raw =combined_raw .select (selected_indices .tolist ())
                total_len =len (combined_raw )
                print (f"[MixCDTask] Applied max_samples limit: {total_len}")

            print (f"[MixCDTask] Final unbalanced dataset size: {total_len} "
            f"(DeepMath: {len(deepmath_raw)}, Countdown: {len(countdown_raw)})")


            task_name =f"mix_c_d_full_total{total_len}"



        split_indices =self ._get_or_make_splits (
        raw_dataset_len =total_len ,
        task_name =task_name 
        )



        def _take (idx_list ):
            return combined_raw .select (idx_list )

        data_splits ={
        "train":_take (split_indices ["train"]),
        "valid":_take (split_indices ["valid"]),
        "test":_take (split_indices ["test"])
        }

        sampling_type ="balanced"if self .balanced_sampling else "unbalanced"
        print (f"[MixCDTask] Deterministic splits ({sampling_type}) - "
        f"train: {len(data_splits['train'])}, "
        f"valid: {len(data_splits['valid'])}, "
        f"test: {len(data_splits['test'])}")

        return data_splits 

    def reset (self ,task_id :int =-1 ,split :str ="train"):


        if self .track_costs :
            reset_costs ()

        if self .use_consultant :
            self .pending_suggestion =None 


        if self .data_splits is None :
            self .data_splits =self ._load_data (
            seed =self .np_random .randint (0 ,10000 ),
            split =split ,
            validation =True ,
            valid_ratio =self .valid_ratio ,
            max_samples =self .max_samples 
            )

        self .data_split =self .data_splits [split ]
        if task_id <0 :
            self .task_id =self .np_random .randint (0 ,len (self .data_split ))
        else :
            self .task_id =task_id 


        task_data =self .data_split [self .task_id ]
        self .current_task_type =task_data .get ("task_type","deepmath")


        user_prompt =self ._format_user_prompt (task_data )



        if self .max_turns >1 and len (self .llm_names )>1 :

            self .messages =[
            {"role":"system","content":self .router_system_prompt .format (num_agents =len (self .llm_names ))},
            {"role":"user","content":user_prompt },
            ]
        else :

            self .messages =[
            {"role":"system","content":self .system_prompt },
            {"role":"user","content":user_prompt },
            {"role":"assistant","content":self .assistant_prompt }
            ]


        self ._obs =self ._get_obs ()
        self .obs_act =[]
        self .num_turns =0 
        self .prev_agent_id =-1 
        self .response =None 
        self .pending_suggestion =None 

        return self ._obs 

    def _calculate_reward (self ,completions :List [str ],task_data :List [Dict [str ,Any ]],debug :bool =False )->List [float ]:

        rewards =[]

        for completion ,data in zip (completions ,task_data ):
            task_type =data .get ("task_type","deepmath")

            if task_type =="countdown":

                r =self ._calculate_countdown_reward (completion ,data ,debug )
            else :

                r =self ._calculate_deepmath_reward (completion ,data ,debug )

            rewards .append (r )

        return rewards 

    def _calculate_countdown_reward (self ,completion :str ,task_data :Dict [str ,Any ],debug :bool =False )->float :

        r =0.0 
        gt =task_data ['target']
        numbers =task_data ['nums']


        if '<answer>'in completion and '</answer>'in completion :

            expr =completion .split ('<answer>')[1 ].split ('</answer>')[0 ].strip ()

            if '='in expr :
                expr =expr .split ('=')[0 ].strip ()
            try :

                used_numbers =[int (n )for n in re .findall (r"\d+",expr )]
            except Exception as e :
                if debug :
                    print ("=== DEBUG: Error parsing numbers ===")
                    print ("Error:",e )
                    print (f"Expression: {expr}")
                return 0.0 


            if sorted (used_numbers )!=sorted (numbers ):
                if debug :
                    print ("=== DEBUG: Numbers mismatch ===")
                    print ("Target:",gt )
                    print ("Expected numbers:",sorted (numbers ))
                    print ("Used numbers:",sorted (used_numbers ))
                    print ("Expression:",expr )
                r =0.0 
            else :

                if not re .match (r"^[\d+\-*/().\s]+$",expr ):
                    if debug :
                        print ("=== DEBUG: Invalid characters in equation ===")
                        print ("Equation:",expr )
                    r =0.0 
                else :
                    try :
                        result =eval (expr ,{"__builtins__":None },{})
                        if abs (float (result )-float (gt ))<1e-5 :
                            r =1.0 
                        else :
                            r =0.0 
                    except Exception as e :
                        if debug :
                            print ("=== DEBUG: Error evaluating equation ===")
                            print ("Equation:",expr )
                            print ("Error:",e )
                        r =0.0 
        else :
            if debug :
                print ("=== DEBUG: No <answer> tag found ===")
                print (f"Completion snippet: {completion[:100]}")
            r =0.0 

        if debug :
            print ("=== DEBUG: Countdown Answer Evaluation ===")
            print ("Target:",gt )
            print ("Numbers:",numbers )
            print ("Extracted expression:",expr if '<answer>'in completion else None )
            print ("Reward:",r )
            print ("===========================")

        return r 

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

    def _calculate_deepmath_reward (self ,completion :str ,task_data :Dict [str ,Any ],debug :bool =False )->float :

        try :

            answer =extract_answer (completion )
            if answer is None :
                if debug :
                    print ("=== DEBUG: No <answer> tag found in response ===")
                    print ("Question:",str (task_data .get ("question",""))[:100 ],"...")
                    print ("Reference answer:",task_data .get ("final_answer",""))
                    print ("Response snippet:",completion [:100 ],"...")
                return 0.0 


            ref_original =task_data .get ("final_answer","")


            pred_answer =self ._normalize_answer (answer )
            ref_answer =self ._normalize_answer (ref_original )


            if "-2\\delta\'"in ref_original or "-2delta\'"in ref_original :

                special_normalize =lambda x :x .replace ('\\\\delta\'','\\delta\'').replace ('-2\\\\delta\'','-2\\delta\'')
                normalized_pred =special_normalize (answer )
                normalized_ref =special_normalize (ref_original )


                if "-2\\delta\'"in normalized_pred and "-2\\delta\'"in normalized_ref :
                    if debug :
                        print ("=== DEBUG: Special case match for -2\\delta' ===")
                    return 1.0 


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
                print ("Question:",str (task_data .get ("question",""))[:100 ],"...")
                print ("Reference answer:",ref_original )
                print ("Extracted answer:",answer )
                print ("Normalized reference:",ref_answer )
                print ("Normalized prediction:",pred_answer )
                print ("Correct:",is_correct )
                print ("Reward:",1.0 if is_correct else 0.0 )
                print ("===========================")

            return 1.0 if is_correct else 0.0 

        except Exception as e :
            if debug :
                print (f"ERROR in _calculate_deepmath_reward: {e}")
                print (f"Completion: {completion[:100]}...")
            return 0.0 


if __name__ =="__main__":

    llms =["gpt-4o-mini","claude-3-7-sonnet-20250219","gemini-1.5-pro","deepseek-ai/DeepSeek-V3"]


    task_balanced =MixCDTask (
    llm_names =llms ,
    max_turns =5 ,
    max_tokens =1536 ,
    debug =True ,
    balanced_sampling =True ,
    task_ratios ={"countdown":0.5 ,"deepmath":0.5 }
    )


    task_unbalanced =MixCDTask (
    llm_names =llms ,
    max_turns =5 ,
    max_tokens =1536 ,
    debug =True ,
    balanced_sampling =False 
    )


    print ("=== Testing Balanced Version ===")
    obs =task_balanced .reset (split ="train")
    print (f"Current task type: {task_balanced.current_task_type}")

    print ("\n=== Testing Unbalanced Version ===")
    obs =task_unbalanced .reset (split ="train")
    print (f"Current task type: {task_unbalanced.current_task_type}")