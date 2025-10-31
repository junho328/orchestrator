

import re 
from typing import Dict ,List ,Any ,Optional 
from datasets import load_dataset ,Dataset ,concatenate_datasets 
from guf .core import Task 
from guf .utils import extract_answer 
from guf .cost import reset_costs 


class MixMCMMLTask (Task ):



    TASK_FIELD_WHITELIST ={
    "medreason":["question","options","answer"],
    "countdown":["nums","target"],
    "math500":["problem","answer"],
    "mmlu":["question","choices","answer","subject"],
    "livecodebench":[
    "question_title","question_content","platform","question_id",
    "contest_id","contest_date","starter_code","difficulty",
    "public_test_cases","private_test_cases","metadata"
    ]
    }


    MEDREASON_PROMPT_TEMPLATE =(
    "{question}\n"
    "{options}\n\n"
    "Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example "
    "<answer>A</answer>. Provide only the letter corresponding to your choice (A, B, C, or D) within the <answer> tags. "
    "Think step by step inside <think> tags, but keep it short."
    )

    COUNTDOWN_PROMPT_TEMPLATE =(
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) one or multiple times but "
    "each number can only be used once. Show your work in <think> </think> tags. "
    "And return the final equation in <answer> </answer> tags, for example "
    "<answer>(1 + 2) / 3</answer>; do not include = in the <answer> tags, only the equation. "
    "Think step by step inside <think> tags, but keep it short."
    )

    MATH500_PROMPT_TEMPLATE =(
    "Solve the following math problem step by step: {problem} "
    "Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags. "
    "For your final answer, follow these guidelines:\n"
    "1. Provide ONLY the final result without explanations, equations, or steps\n"
    "2. For equations with a single solution, provide just the value (e.g., '5' not 'x=5')\n"
    "3. For multiple solutions, list the values separated by commas (e.g., '3, 5, 7' not 'x = 3, 5, 7')\n"
    "4. Use standard LaTeX notation: \\frac{{a}}{{b}} for fractions, \\sqrt{{n}} for square roots\n"
    "5. For angle measurements, include the degree symbol (e.g., '90^\\circ')\n"
    "6. Do not use \\boxed{{}} around your answer\n"
    "Example: <answer>3\\sqrt{{13}}</answer>"
    "Think step by step inside <think> tags, but keep it short."
    )

    MMLU_PROMPT_TEMPLATE =(
    "Subject: {subject}\n\n"
    "Question: {question}\n\n"
    "Options:\n{choices}\n\n"
    "Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example "
    "<answer>B</answer>. Provide only the letter corresponding to your choice (A, B, C, or D) within the <answer> tags. "
    "Think step by step inside <think> tags, but keep it short."
    )

    LIVECODEBENCH_PROMPT_TEMPLATE =(
    "### Question:\n{question_content}\n\n"
    "### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). "
    "Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
    "```python\n# YOUR CODE HERE\n```\n\n"
    "### Answer: (use the provided format with backticks)\n\n"
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
            "medreason":0.2 ,
            "countdown":0.2 ,
            "math500":0.2 ,
            "mmlu":0.2 ,
            "livecodebench":0.2 
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

            return self .MEDREASON_PROMPT_TEMPLATE 

        if self .current_task_type =="medreason":
            return self .MEDREASON_PROMPT_TEMPLATE 
        elif self .current_task_type =="countdown":
            return self .COUNTDOWN_PROMPT_TEMPLATE 
        elif self .current_task_type =="math500":
            return self .MATH500_PROMPT_TEMPLATE 
        elif self .current_task_type =="mmlu":
            return self .MMLU_PROMPT_TEMPLATE 
        elif self .current_task_type =="livecodebench":
            return self .LIVECODEBENCH_PROMPT_TEMPLATE 
        else :
            return self .MEDREASON_PROMPT_TEMPLATE 

    def _format_base_prompt (self ,task_data :Dict )->str :


        self .current_task_type =task_data .get ("task_type","medreason")


        expected_fields =self .TASK_FIELD_WHITELIST .get (self .current_task_type ,[])
        clean_task_data ={k :v for k ,v in task_data .items ()if k in expected_fields }

        if self .current_task_type =="medreason":
            return self .MEDREASON_PROMPT_TEMPLATE .format (
            question =clean_task_data ["question"],
            options =clean_task_data ["options"]
            )
        elif self .current_task_type =="countdown":
            return self .COUNTDOWN_PROMPT_TEMPLATE .format (
            numbers =clean_task_data ["nums"],
            target =clean_task_data ["target"],
            )
        elif self .current_task_type =="math500":

            problem_text =clean_task_data ["problem"].replace ("{","{{").replace ("}","}}")
            try :
                return self .MATH500_PROMPT_TEMPLATE .format (problem =problem_text )
            except Exception as e :

                try :
                    return self .MATH500_PROMPT_TEMPLATE .replace ("{problem}",str (clean_task_data ["problem"]))
                except Exception :
                    return f"Error formatting prompt: {e}. Please check the problem text and format string."
        elif self .current_task_type =="mmlu":

            choices =clean_task_data ["choices"]
            formatted_choices =""
            for i ,choice in enumerate (choices ):
                letter =chr (ord ('A')+i )
                formatted_choices +=f"{letter}. {choice}\n"
            formatted_choices =formatted_choices .strip ()

            return self .MMLU_PROMPT_TEMPLATE .format (
            subject =clean_task_data ["subject"],
            question =clean_task_data ["question"],
            choices =formatted_choices 
            )
        elif self .current_task_type =="livecodebench":
            return self .LIVECODEBENCH_PROMPT_TEMPLATE .format (
            question_content =clean_task_data ["question_content"]
            )
        else :
            return f"Error: Unknown task type {self.current_task_type}"

    def get_balanced_sample_sizes (self ,available_counts :Dict [str ,int ],target_total :int =None )->Dict [str ,int ]:

        if target_total is None :

            max_totals =[]
            for task_name ,ratio in self .task_ratios .items ():
                if ratio >0 and task_name in available_counts :
                    max_total_from_task =int (available_counts [task_name ]/ratio )
                    max_totals .append (max_total_from_task )
            target_total =min (max_totals )if max_totals else 0 


        target_counts ={}
        for task_name ,ratio in self .task_ratios .items ():
            target_counts [task_name ]=int (target_total *ratio )


        actual_counts ={}
        for task_name ,target_count in target_counts .items ():
            if task_name in available_counts :
                actual_counts [task_name ]=min (target_count ,available_counts [task_name ])
            else :
                actual_counts [task_name ]=0 


        scaling_factors =[]
        for task_name ,target_count in target_counts .items ():
            if target_count >0 and task_name in actual_counts :
                scaling_factor =actual_counts [task_name ]/target_count 
                scaling_factors .append ((scaling_factor ,task_name ))

        if scaling_factors :
            limiting_factor_ratio ,limiting_task =min (scaling_factors )
            if limiting_factor_ratio <1.0 :

                for task_name in actual_counts :
                    actual_counts [task_name ]=int (target_counts [task_name ]*limiting_factor_ratio )

                if self .debug :
                    print (f"[MixMCMMLTask] {limiting_task} dataset is the limiting factor for balanced sampling")

        return actual_counts 

    def _standardize_dataset_schema (self ,dataset ,task_type :str ):

        from datasets import Features ,Value 

        def standardize_sample (sample ):

            result =dict (sample )


            result ["task_type"]=task_type 


            if task_type =="mmlu"and "answer"in sample :

                answer_idx =int (sample ["answer"])
                result ["answer"]=chr (ord ('A')+answer_idx )
            elif task_type in ["countdown","livecodebench"]and "answer"not in sample :

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

        import numpy as np 


        print (f"[MixMCMMLTask] Loading datasets...")


        medreason_raw =load_dataset ("UCSC-VLAA/MedReason",split ="train")
        medreason_raw =self ._standardize_dataset_schema (medreason_raw ,"medreason")


        countdown_raw =load_dataset ("Jiayi-Pan/Countdown-Tasks-3to4",split ="train")
        countdown_raw =self ._standardize_dataset_schema (countdown_raw ,"countdown")


        math500_raw =load_dataset ("HuggingFaceH4/MATH-500",split ="test")
        math500_raw =self ._standardize_dataset_schema (math500_raw ,"math500")


        mmlu_raw =load_dataset ("cais/mmlu","all")["auxiliary_train"]
        mmlu_raw =self ._standardize_dataset_schema (mmlu_raw ,"mmlu")


        livecodebench_raw =load_dataset ("livecodebench/code_generation_lite",split ="test",version_tag ="release_v1",trust_remote_code =True )
        livecodebench_raw =self ._standardize_dataset_schema (livecodebench_raw ,"livecodebench")

        available_counts ={
        "medreason":len (medreason_raw ),
        "countdown":len (countdown_raw ),
        "math500":len (math500_raw ),
        "mmlu":len (mmlu_raw ),
        "livecodebench":len (livecodebench_raw )
        }

        print (f"[MixMCMMLTask] Raw dataset sizes - {available_counts}")

        if self .balanced_sampling :
            print (f"[MixMCMMLTask] Using BALANCED sampling with ratios - {self.task_ratios}")


            target_total =max_samples if max_samples is not None and max_samples >0 else None 
            actual_counts =self .get_balanced_sample_sizes (available_counts ,target_total )

            print (f"[MixMCMMLTask] Balanced sampling - {actual_counts}")


            np .random .seed (seed )


            subsets =[]
            for task_name ,count in actual_counts .items ():
                if count >0 :
                    if task_name =="medreason":
                        indices =np .random .choice (len (medreason_raw ),count ,replace =False )
                        subset =medreason_raw .select (indices .tolist ())
                    elif task_name =="countdown":
                        indices =np .random .choice (len (countdown_raw ),count ,replace =False )
                        subset =countdown_raw .select (indices .tolist ())
                    elif task_name =="math500":
                        indices =np .random .choice (len (math500_raw ),count ,replace =False )
                        subset =math500_raw .select (indices .tolist ())
                    elif task_name =="mmlu":
                        indices =np .random .choice (len (mmlu_raw ),count ,replace =False )
                        subset =mmlu_raw .select (indices .tolist ())
                    elif task_name =="livecodebench":
                        indices =np .random .choice (len (livecodebench_raw ),count ,replace =False )
                        subset =livecodebench_raw .select (indices .tolist ())

                    subsets .append (subset )


            combined_raw =concatenate_datasets (subsets )
            total_len =len (combined_raw )

            print (f"[MixMCMMLTask] Final balanced dataset size: {total_len}")


            task_name =f"mix_mcmml_balanced_total{total_len}"

        else :
            print (f"[MixMCMMLTask] Using UNBALANCED sampling - concatenating all available data")


            combined_raw =concatenate_datasets ([
            medreason_raw ,countdown_raw ,math500_raw ,mmlu_raw ,livecodebench_raw 
            ])
            total_len =len (combined_raw )


            if max_samples is not None and max_samples >0 and max_samples <total_len :
                np .random .seed (seed )
                selected_indices =np .random .choice (total_len ,max_samples ,replace =False )
                combined_raw =combined_raw .select (selected_indices .tolist ())
                total_len =len (combined_raw )
                print (f"[MixMCMMLTask] Applied max_samples limit: {total_len}")

            print (f"[MixMCMMLTask] Final unbalanced dataset size: {total_len}")


            task_name =f"mix_mcmml_full_total{total_len}"


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
        print (f"[MixMCMMLTask] Deterministic splits ({sampling_type}) - "
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
        self .current_task_type =task_data .get ("task_type","medreason")


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


        task_groups ={}
        for i ,data in enumerate (task_data ):
            task_type =data .get ("task_type","medreason")
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


            if task_type =="medreason":
                batch_rewards =[self ._calculate_medreason_reward (comp ,data ,debug )
                for comp ,data in zip (batch_completions ,clean_batch_data )]
            elif task_type =="countdown":
                batch_rewards =[self ._calculate_countdown_reward (comp ,data ,debug )
                for comp ,data in zip (batch_completions ,clean_batch_data )]
            elif task_type =="math500":
                batch_rewards =[self ._calculate_math500_reward (comp ,data ,debug )
                for comp ,data in zip (batch_completions ,clean_batch_data )]
            elif task_type =="mmlu":
                batch_rewards =[self ._calculate_mmlu_reward (comp ,data ,debug )
                for comp ,data in zip (batch_completions ,clean_batch_data )]
            elif task_type =="livecodebench":

                batch_rewards =[]
                for completion ,data in zip (batch_completions ,clean_batch_data ):
                    reward =self ._calculate_livecodebench_reward (completion ,data ,debug =debug )
                    batch_rewards .append (reward )
            else :
                batch_rewards =[0.0 ]*len (batch_completions )
                if debug :
                    print (f"Warning: Unknown task type {task_type}, assigning 0 reward")


            for idx ,reward in zip (indices ,batch_rewards ):
                while len (rewards )<=idx :
                    rewards .append (0.0 )
                rewards [idx ]=reward 

        return rewards 

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
            print (f"=== DEBUG: MedReason - Pred {pred} vs Correct {correct_letter} → {is_correct}")

        return 1.0 if is_correct else 0.0 

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

        if debug :
            print ("=== DEBUG: Countdown Answer Evaluation ===")
            print ("Target:",gt )
            print ("Numbers:",numbers )
            print ("Reward:",r )
            print ("===========================")

        return r 

    def _calculate_math500_reward (self ,completion :str ,task_data :Dict [str ,Any ],debug :bool =False )->float :


        answer =extract_answer (completion )
        if answer is None :
            if debug :
                print ("=== DEBUG: No <answer> tag found in response ===")
                print ("Problem:",task_data ["problem"][:100 ],"...")
                print ("Reference answer:",task_data ["answer"])
                print ("Response snippet:",completion [:100 ],"...")
            return 0.0 


        pred_answer =self ._normalize_math_answer (answer )
        ref_answer =self ._normalize_math_answer (task_data ["answer"])


        is_correct =pred_answer ==ref_answer 


        if not is_correct :

            if ","in pred_answer and ","in ref_answer :
                pred_values =[v .strip ()for v in pred_answer .split (",")]
                ref_values =[v .strip ()for v in ref_answer .split (",")]
                is_correct =sorted (pred_values )==sorted (ref_values )


            if not is_correct and "="in pred_answer :
                value_after_equals =pred_answer .split ("=")[-1 ].strip ()
                is_correct =value_after_equals ==ref_answer 

        if debug :
            print ("=== DEBUG: Math500 Answer Evaluation ===")
            print ("Problem:",task_data ["problem"][:100 ],"...")
            print ("Reference answer:",task_data ["answer"])
            print ("Extracted answer:",answer )
            print ("Normalized reference:",ref_answer )
            print ("Normalized prediction:",pred_answer )
            print ("Correct:",is_correct )
            print ("===========================")

        return 1.0 if is_correct else 0.0 

    def _calculate_mmlu_reward (self ,completion :str ,task_data :Dict [str ,Any ],debug :bool =False )->float :


        pred =extract_answer (completion )
        if pred is None :
            if debug :
                print ("=== DEBUG: No <answer> tag found ===")
                print ("Question:",task_data ["question"][:100 ],"...")
                print ("Subject:",task_data ["subject"])
                print ("Response snippet:",completion [:100 ],"...")
            return 0.0 


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
                return 0.0 


        correct_letter =task_data ["answer"].strip ().upper ()


        is_correct =(pred_letter ==correct_letter )

        if debug :
            print ("=== DEBUG: MMLU Answer Evaluation ===")
            print ("Question:",task_data ["question"][:100 ],"...")
            print ("Subject:",task_data ["subject"])
            print ("Choices:",task_data ["choices"])
            print ("Correct letter:",correct_letter )
            print ("Extracted answer:",pred )
            print ("Predicted letter:",pred_letter )
            print ("Correct:",is_correct )
            print ("===========================")

        return 1.0 if is_correct else 0.0 

    def _calculate_livecodebench_reward (self ,completion :str ,task_data :Dict [str ,Any ],debug :bool =False )->float :



        import inspect 


        in_batch_processing =False 
        for frame_info in inspect .stack ():
            if 'run_task_with_batches'in frame_info .function or 'batch_completion'in frame_info .function :
                in_batch_processing =True 
                break 

        if in_batch_processing :
            return self ._calculate_livecodebench_reward_for_batch_processing (completion ,task_data ,debug )
        else :
            return self ._calculate_livecodebench_reward_for_cma_training (completion ,task_data ,debug )

    def _calculate_livecodebench_reward_for_batch_processing (self ,completion :str ,task_data :Dict [str ,Any ],
    debug :bool =False )->float :

        try :
            from guf .tasks .livecodebench import CodeGenerationProblem ,extract_code 
            from guf .tasks .livecodebench_utils import codegen_metrics 
        except ImportError :
            if debug :
                print ("=== DEBUG: LiveCodeBench evaluation modules not available ===")
            return 0.0 

        try :
            task_data_sample =[CodeGenerationProblem (**task_data ).get_evaluation_sample ()]


            task_completion =[[extract_code (completion )]]

            codegen_return =codegen_metrics (task_data_sample ,task_completion ,debug =debug )
            metrics ,results ,final_metadata =codegen_return [0 ],codegen_return [1 ],codegen_return [2 ]
            reward =metrics ['pass@1']

            if debug :
                print ("=== DEBUG: LiveCodeBench Answer Evaluation (Batch Processing) ===")
                print (f"Completion: {completion[:100]}...")
                print (f"Extracted code: {task_completion[0][0][:100]}...")
                print (f"Pass@1: {reward}")
                print ("===========================")


            return 1.0 if reward ==1.0 else 0.0 
        except Exception as e :
            if debug :
                print (f"=== DEBUG: Error in LiveCodeBench evaluation (Batch Processing): {e} ===")
            return 0.0 

    def _calculate_livecodebench_reward_for_cma_training (self ,completion :str ,task_data :Dict [str ,Any ],
    debug :bool =False )->float :



        import os 
        import time 
        from datetime import datetime 


        base_log_dir =getattr (self ,'log_dir','logs')


        timestamp =datetime .now ().strftime ("%Y%m%d_%H%M%S")
        debug_dir =os .path .join (base_log_dir ,f"debug_livecodebench_{timestamp}")


        base_debug_pattern =os .path .join (base_log_dir ,"debug_livecodebench_")
        existing_dirs =[d for d in os .listdir (base_log_dir )if d .startswith ("debug_livecodebench_")]if os .path .exists (
        base_log_dir )else []

        if existing_dirs :

            existing_dirs .sort ()
            debug_dir =os .path .join (base_log_dir ,existing_dirs [-1 ])

        os .makedirs (debug_dir ,exist_ok =True )
        debug_file =os .path .join (debug_dir ,f"livecodebench_debug_{int(time.time())}_{os.getpid()}.txt")

        def log_debug (message ):
            with open (debug_file ,"a")as f :
                f .write (f"{time.strftime('%H:%M:%S')} - {message}\n")
                f .flush ()

        log_debug ("=== LiveCodeBench evaluation started (CMA-ES TRAINING VERSION) ===")
        log_debug (f"Debug file location: {debug_file}")
        log_debug (f"Completion length: {len(completion)}")
        log_debug (f"Task data keys: {list(task_data.keys())}")


        log_debug (f"Problem title: {task_data.get('question_title', 'N/A')}")
        log_debug (f"Platform: {task_data.get('platform', 'N/A')}")
        log_debug (f"Difficulty: {task_data.get('difficulty', 'N/A')}")

        try :
            from guf .tasks .livecodebench import CodeGenerationProblem ,extract_code 
            import json 
            import sys 
            import signal 
            import time 
            from io import StringIO 
            from unittest .mock import patch 
            from types import ModuleType 

            log_debug ("Successfully imported LiveCodeBench modules")


            log_debug ("Creating task data sample...")
            problem =CodeGenerationProblem (**task_data )
            sample =problem .get_evaluation_sample ()
            log_debug (f"Task data sample created successfully")


            if 'input_output'in sample :
                try :
                    io_data =json .loads (sample ['input_output'])
                    log_debug (f"Number of test cases: {len(io_data.get('inputs', []))}")
                    log_debug (f"Function name (if call-based): {io_data.get('fn_name', 'N/A')}")
                    log_debug (f"Test type: {'call-based' if io_data.get('fn_name') else 'stdin-based'}")


                    if io_data .get ('inputs')and len (io_data ['inputs'])>0 :
                        log_debug (f"First test input: {io_data['inputs'][0][:200]}...")
                    if io_data .get ('outputs')and len (io_data ['outputs'])>0 :
                        log_debug (f"First expected output: {io_data['outputs'][0][:200]}...")
                except Exception as e :
                    log_debug (f"Error parsing input_output: {e}")
                    return 0.0 


            log_debug ("=== FULL COMPLETION ===")
            log_debug (completion )
            log_debug ("=== END COMPLETION ===")


            log_debug ("Extracting code from completion...")
            extracted_code =extract_code (completion )
            log_debug (f"Extracted code length: {len(extracted_code)}")

            log_debug ("=== EXTRACTED CODE ===")
            log_debug (extracted_code )
            log_debug ("=== END EXTRACTED CODE ===")

            if len (extracted_code )==0 :
                log_debug ("WARNING: No code extracted from completion!")
                return 0.0 


            log_debug ("Using signal-based timeout evaluation for CMA-ES training")

            try :
                io_data =json .loads (sample ['input_output'])
                timeout_seconds =6 


                class TimeoutException (Exception ):
                    pass 

                def timeout_handler (signum ,frame ):
                    raise TimeoutException ("Code execution timeout")

                if io_data .get ('fn_name')is None :

                    log_debug ("Evaluating stdin-based problem")

                    results =[]
                    for i ,(test_input ,expected_output )in enumerate (zip (io_data ['inputs'],io_data ['outputs'])):
                        log_debug (f"Running test case {i + 1}/{len(io_data['inputs'])}")
                        log_debug (f"Test input: {test_input[:100]}...")
                        log_debug (f"Expected output: {expected_output[:100]}...")

                        try :

                            signal .signal (signal .SIGALRM ,timeout_handler )
                            signal .alarm (timeout_seconds )


                            captured_output =StringIO ()


                            exec_globals ={
                            '__builtins__':__builtins__ ,
                            'input':lambda :test_input ,
                            }


                            with patch ('sys.stdin',StringIO (test_input )):
                                with patch ('sys.stdout',captured_output ):

                                    exec (extracted_code ,exec_globals )


                            actual_output =captured_output .getvalue ().strip ()
                            expected_output_clean =expected_output .strip ()

                            signal .alarm (0 )

                            log_debug (f"Actual output: {actual_output[:100]}...")

                            if actual_output ==expected_output_clean :
                                results .append (True )
                                log_debug (f"Test {i + 1}: PASSED")
                            else :
                                results .append (False )
                                log_debug (f"Test {i + 1}: FAILED")
                                log_debug (f"  Expected: '{expected_output_clean[:200]}'")
                                log_debug (f"  Actual:   '{actual_output[:200]}'")

                        except TimeoutException :
                            signal .alarm (0 )
                            results .append (False )
                            log_debug (f"Test {i + 1}: TIMEOUT")
                        except Exception as e :
                            signal .alarm (0 )
                            results .append (False )
                            log_debug (f"Test {i + 1}: ERROR - {str(e)[:200]}")
                            import traceback 
                            log_debug (f"Error traceback: {traceback.format_exc()}")

                else :

                    log_debug ("Evaluating function-based problem")
                    fn_name =io_data ['fn_name']

                    results =[]
                    for i ,(test_input_json ,expected_output_json )in enumerate (
                    zip (io_data ['inputs'],io_data ['outputs'])):
                        log_debug (f"Running function test case {i + 1}/{len(io_data['inputs'])}")

                        try :

                            test_inputs =[json .loads (line )for line in test_input_json .split ('\n')]
                            expected_output =json .loads (expected_output_json )

                            log_debug (f"Function inputs: {test_inputs}")
                            log_debug (f"Expected output: {expected_output}")


                            signal .signal (signal .SIGALRM ,timeout_handler )
                            signal .alarm (timeout_seconds )


                            exec_globals ={'__builtins__':__builtins__ }


                            exec (extracted_code ,exec_globals )


                            if 'class Solution'in extracted_code and fn_name !='Solution':

                                solution_instance =exec_globals ['Solution']()
                                target_function =getattr (solution_instance ,fn_name )
                            else :

                                target_function =exec_globals [fn_name ]


                            actual_output =target_function (*test_inputs )

                            signal .alarm (0 )


                            if isinstance (actual_output ,tuple ):
                                actual_output =list (actual_output )

                            log_debug (f"Function output: {actual_output}")

                            if actual_output ==expected_output :
                                results .append (True )
                                log_debug (f"Function test {i + 1}: PASSED")
                            else :
                                results .append (False )
                                log_debug (f"Function test {i + 1}: FAILED")
                                log_debug (f"  Expected: {expected_output}")
                                log_debug (f"  Actual:   {actual_output}")

                        except TimeoutException :
                            signal .alarm (0 )
                            results .append (False )
                            log_debug (f"Function test {i + 1}: TIMEOUT")
                        except Exception as e :
                            signal .alarm (0 )
                            results .append (False )
                            log_debug (f"Function test {i + 1}: ERROR - {str(e)[:200]}")
                            import traceback 
                            log_debug (f"Function error traceback: {traceback.format_exc()}")


                passed =sum (1 for r in results if r )
                total =len (results )
                reward =1.0 if passed ==total and total >0 else 0.0 

                log_debug (f"=== FINAL RESULTS ===")
                log_debug (f"Individual test results: {results}")
                log_debug (f"Tests passed: {passed}/{total}")
                log_debug (f"Final reward: {reward}")
                log_debug (f"=== END RESULTS ===")

                return reward 

            except Exception as eval_e :
                log_debug (f"CMA-ES evaluation failed: {eval_e}")
                import traceback 
                log_debug (f"CMA-ES evaluation traceback: {traceback.format_exc()}")
                return 0.0 

        except ImportError as e :
            log_debug (f"IMPORT ERROR: LiveCodeBench evaluation modules not available: {e}")
            return 0.0 
        except Exception as e :
            log_debug (f"Setup failed: {e}")
            import traceback 
            log_debug (f"Setup traceback: {traceback.format_exc()}")
            return 0.0 

    def _normalize_math_answer (self ,answer :str )->str :

        if not answer :
            return ""


        normalized =answer .strip ()
        normalized =re .sub (r'^\n+|\n+$','',normalized )


        try :
            normalized =re .sub (r'\\boxed\{([^}]*)\}',r'\1',normalized )
            normalized =re .sub (r'\\begin\{[^}]*\}(.*?)\\end\{[^}]*\}',r'\1',normalized ,flags =re .DOTALL )
            normalized =re .sub (r'\\[([^]]*)\]',r'\1',normalized )
            normalized =re .sub (r'\\\((.*?)\\\)',r'\1',normalized )
        except Exception :
            pass 


        normalized =re .sub (r'\s+','',normalized )


        try :
            normalized =re .sub (r'\\dfrac',r'\\frac',normalized )
            normalized =re .sub (r'\\text\{([^}]*)\}',r'\1',normalized )
        except Exception :
            pass 


        if '='in normalized and ','not in normalized :
            try :
                normalized =re .sub (r'^[a-zA-Z]+\s*=\s*','',normalized )
            except Exception :
                pass 

        return normalized 


if __name__ =="__main__":

    import numpy as np 

    llms =["gpt-4o-mini"]


    print ("=== Testing Balanced Sampling ===")
    task_balanced =MixMCMMLTask (
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
    task_unbalanced =MixMCMMLTask (
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