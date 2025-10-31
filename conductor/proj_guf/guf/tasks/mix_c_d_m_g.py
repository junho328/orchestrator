

import re 
import json 
from typing import Dict ,List ,Any ,Optional 
from datasets import load_dataset ,Dataset ,concatenate_datasets 
import numpy as np 
from guf .core import Task 
from guf .utils import extract_answer 
from guf .cost import reset_costs 


class MixCDMGTask (Task ):



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

    MEDREASON_PROMPT_TEMPLATE =(
    "{question}\n"
    "{options}\n\n"
    "Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example "
    "<answer> A </answer>. Provide only the letter corresponding to your choice (A, B, C, or D) within the <answer> tags. "
    "Think step by step inside <think> tags, but keep it short."
    )

    GSM8K_PROMPT_TEMPLATE =(
    "Solve the following math problem step by step: {question} "
    "Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example "
    "<answer> 42 </answer>. Think step by step inside <think> tags, but keep it short."
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
            self .task_ratios ={
            "countdown":0.25 ,
            "deepmath":0.25 ,
            "medreason":0.25 ,
            "gsm8k":0.25 
            }
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

            return self .GSM8K_PROMPT_TEMPLATE 

        if self .current_task_type =="countdown":
            return self .COUNTDOWN_PROMPT_TEMPLATE 
        elif self .current_task_type =="deepmath":
            return self .DEEPMATH_PROMPT_TEMPLATE 
        elif self .current_task_type =="medreason":
            return self .MEDREASON_PROMPT_TEMPLATE 
        else :
            return self .GSM8K_PROMPT_TEMPLATE 

    def _format_base_prompt (self ,task_data :Dict )->str :


        self .current_task_type =task_data .get ("task_type","gsm8k")

        if self .current_task_type =="countdown":
            return self .COUNTDOWN_PROMPT_TEMPLATE .format (
            numbers =task_data ["nums"],
            target =task_data ["target"],
            )
        elif self .current_task_type =="deepmath":

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
        elif self .current_task_type =="medreason":
            return self .MEDREASON_PROMPT_TEMPLATE .format (
            question =task_data ["question"],
            options =task_data ["options"]
            )
        else :
            return self .GSM8K_PROMPT_TEMPLATE .format (question =task_data ["question"])

    def get_balanced_sample_sizes (self ,available_sizes :Dict [str ,int ],target_total :int =None )->Dict [str ,int ]:

        task_names =["countdown","deepmath","medreason","gsm8k"]

        if target_total is None :

            max_totals =[]
            for task in task_names :
                ratio =self .task_ratios .get (task ,0.25 )
                if ratio >0 :
                    max_from_task =int (available_sizes [task ]/ratio )
                    max_totals .append (max_from_task )
            target_total =min (max_totals )if max_totals else 0 


        target_samples ={}
        for task in task_names :
            ratio =self .task_ratios .get (task ,0.25 )
            target_samples [task ]=int (target_total *ratio )


        actual_samples ={}
        limiting_factors =[]

        for task in task_names :
            available =available_sizes [task ]
            target =target_samples [task ]
            actual =min (target ,available )
            actual_samples [task ]=actual 

            if actual <target :
                limiting_factors .append (task )


        if limiting_factors :

            scaling_factors =[]
            for task in limiting_factors :
                if target_samples [task ]>0 :
                    scaling_factor =actual_samples [task ]/target_samples [task ]
                    scaling_factors .append (scaling_factor )

            if scaling_factors :
                min_scaling =min (scaling_factors )


                for task in task_names :
                    actual_samples [task ]=int (target_samples [task ]*min_scaling )

                if self .debug :
                    print (f"[MixCDMGTask] Limiting factors: {limiting_factors}. Applied scaling factor: {min_scaling:.3f}")

        return actual_samples 

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
        medreason_raw =load_dataset ("UCSC-VLAA/MedReason",split ="train")
        gsm8k_raw =load_dataset ("gsm8k","main",split ="train")


        deepmath_raw =deepmath_raw .map (lambda x :{"task_type":"deepmath"})
        countdown_raw =countdown_raw .map (lambda x :{"task_type":"countdown"})
        medreason_raw =medreason_raw .map (lambda x :{"task_type":"medreason"})
        gsm8k_raw =gsm8k_raw .map (lambda x :{"task_type":"gsm8k"})

        available_sizes ={
        "deepmath":len (deepmath_raw ),
        "countdown":len (countdown_raw ),
        "medreason":len (medreason_raw ),
        "gsm8k":len (gsm8k_raw )
        }

        print (f"[MixCDMGTask] Raw dataset sizes - DeepMath: {available_sizes['deepmath']}, "
        f"Countdown: {available_sizes['countdown']}, MedReason: {available_sizes['medreason']}, "
        f"GSM8K: {available_sizes['gsm8k']}")

        if self .balanced_sampling :
            print (f"[MixCDMGTask] Using BALANCED sampling with ratios - "
            f"Countdown: {self.task_ratios.get('countdown', 0.25):.1%}, "
            f"DeepMath: {self.task_ratios.get('deepmath', 0.25):.1%}, "
            f"MedReason: {self.task_ratios.get('medreason', 0.25):.1%}, "
            f"GSM8K: {self.task_ratios.get('gsm8k', 0.25):.1%}")



            target_total =max_samples if max_samples is not None and max_samples >0 else None 
            actual_samples =self .get_balanced_sample_sizes (available_sizes ,target_total )

            print (f"[MixCDMGTask] Balanced sampling - Countdown: {actual_samples['countdown']}, "
            f"DeepMath: {actual_samples['deepmath']}, MedReason: {actual_samples['medreason']}, "
            f"GSM8K: {actual_samples['gsm8k']}")



            np .random .seed (seed )


            countdown_indices =np .random .choice (len (countdown_raw ),actual_samples ['countdown'],replace =False )
            deepmath_indices =np .random .choice (len (deepmath_raw ),actual_samples ['deepmath'],replace =False )
            medreason_indices =np .random .choice (len (medreason_raw ),actual_samples ['medreason'],replace =False )
            gsm8k_indices =np .random .choice (len (gsm8k_raw ),actual_samples ['gsm8k'],replace =False )


            countdown_subset =countdown_raw .select (countdown_indices .tolist ())
            deepmath_subset =deepmath_raw .select (deepmath_indices .tolist ())
            medreason_subset =medreason_raw .select (medreason_indices .tolist ())
            gsm8k_subset =gsm8k_raw .select (gsm8k_indices .tolist ())



            combined_raw =concatenate_datasets ([
            countdown_subset ,
            deepmath_subset ,
            medreason_subset ,
            gsm8k_subset 
            ])
            total_len =len (combined_raw )

            print (f"[MixCDMGTask] Final balanced dataset size: {total_len} "
            f"(Countdown: {actual_samples['countdown']}, DeepMath: {actual_samples['deepmath']}, "
            f"MedReason: {actual_samples['medreason']}, GSM8K: {actual_samples['gsm8k']})")


            ratios_str ="_".join ([f"{k}{v:.3f}"for k ,v in self .task_ratios .items ()])
            samples_str ="_".join ([f"{k}{v}"for k ,v in actual_samples .items ()])
            balanced_task_name =f"mix_c_d_m_g_balanced_{samples_str}_{ratios_str}_total{total_len}"
            task_name =balanced_task_name 

        else :
            print (f"[MixCDMGTask] Using UNBALANCED sampling - concatenating all available data")



            combined_raw =concatenate_datasets ([deepmath_raw ,countdown_raw ,medreason_raw ,gsm8k_raw ])
            total_len =len (combined_raw )


            if max_samples is not None and max_samples >0 and max_samples <total_len :
                np .random .seed (seed )
                selected_indices =np .random .choice (total_len ,max_samples ,replace =False )
                combined_raw =combined_raw .select (selected_indices .tolist ())
                total_len =len (combined_raw )
                print (f"[MixCDMGTask] Applied max_samples limit: {total_len}")

            print (f"[MixCDMGTask] Final unbalanced dataset size: {total_len} "
            f"(DeepMath: {len(deepmath_raw)}, Countdown: {len(countdown_raw)}, "
            f"MedReason: {len(medreason_raw)}, GSM8K: {len(gsm8k_raw)})")


            task_name =f"mix_c_d_m_g_full_total{total_len}"



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
        print (f"[MixCDMGTask] Deterministic splits ({sampling_type}) - "
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
        self .current_task_type =task_data .get ("task_type","gsm8k")


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

        return self ._obs 

    def _calculate_reward (self ,completions :List [str ],task_data :List [Dict [str ,Any ]],debug :bool =False )->List [float ]:

        rewards =[]

        for completion ,data in zip (completions ,task_data ):
            task_type =data .get ("task_type","gsm8k")

            if task_type =="countdown":
                r =self ._calculate_countdown_reward (completion ,data ,debug )
            elif task_type =="deepmath":
                r =self ._calculate_deepmath_reward (completion ,data ,debug )
            elif task_type =="medreason":
                r =self ._calculate_medreason_reward (completion ,data ,debug )
            else :
                r =self ._calculate_gsm8k_reward (completion ,data ,debug )

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

    def _calculate_medreason_reward (self ,completion :str ,task_data :Dict [str ,Any ],debug :bool =False )->float :

        pred =extract_answer (completion )
        if pred is None :
            if debug :
                print ("=== DEBUG: No <answer> tag found ===",completion )
            return 0.0 

        pred =pred .strip ().upper ()
        if pred not in {"A","B","C","D"}and pred :
            pred =pred [0 ]


        opts =re .findall (r'([ABCD])\.\s*([^\n\r]+)',task_data ["options"])
        letter_map ={ltr :txt .strip ()for ltr ,txt in opts }


        ref =task_data ["answer"].split (".")[0 ].strip ()


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
            print (f"=== DEBUG: MedReason Pred {pred} vs Correct {correct_letter} → {is_correct}")

        return 1.0 if is_correct else 0.0 

    def _calculate_gsm8k_reward (self ,completion :str ,task_data :Dict [str ,Any ],debug :bool =False )->float :


        answer =extract_answer (completion )
        if answer is None :
            if debug :
                print ("=== DEBUG: No <answer> tag found in response ===")
                print ("Question:",task_data ["question"][:100 ],"...")
                print ("Reference answer:",task_data ["answer"])
                print ("Response snippet:",completion [:100 ],"...")
            return 0.0 


        pred_numbers =re .findall (r"[-+]?\d*\.\d+|\d+",answer )
        ref_numbers =re .findall (r"[-+]?\d*\.\d+|\d+",task_data ["answer"])

        if not pred_numbers or not ref_numbers :
            if debug :
                print ("=== DEBUG: No numbers found in answer ===")
                print ("Question:",task_data ["question"][:100 ],"...")
                print ("Reference answer:",task_data ["answer"])
                print ("Extracted answer:",answer )
            return 0.0 


        try :
            pred_final =float (pred_numbers [-1 ])
            ref_final =float (ref_numbers [-1 ])
        except Exception as e :
            if debug :
                print ("=== DEBUG: Error converting numbers to float ===")
                print ("Error:",e )
            return 0.0 


        is_correct =abs (pred_final -ref_final )<1e-5 

        if debug :
            print ("=== DEBUG: GSM8K Answer Evaluation ===")
            print ("Question:",task_data ["question"][:100 ],"...")
            print ("Reference answer:",task_data ["answer"])
            print ("Extracted answer:",answer )
            print ("Reference final number:",ref_final )
            print ("Predicted final number:",pred_final )
            print ("Correct:",is_correct )
            print ("Reward:",1.0 if is_correct else 0.0 )
            print ("===========================")

        return 1.0 if is_correct else 0.0 


if __name__ =="__main__":

    llms =["gpt-4o-mini","claude-3-7-sonnet-20250219","gemini-1.5-pro","deepseek-ai/DeepSeek-V3"]


    task_balanced =MixCDMGTask (
    llm_names =llms ,
    max_turns =5 ,
    max_tokens =1536 ,
    debug =True ,
    balanced_sampling =True ,
    task_ratios ={"countdown":0.25 ,"deepmath":0.25 ,"medreason":0.25 ,"gsm8k":0.25 }
    )


    task_unbalanced =MixCDMGTask (
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