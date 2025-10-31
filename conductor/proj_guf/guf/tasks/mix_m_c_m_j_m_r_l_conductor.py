


import random 
import multiprocessing 
from typing import Dict ,List ,Any ,Optional 
import numpy as np 
from datasets import Dataset ,concatenate_datasets 
from guf .core import Task 
from guf .tasks .medreason import MedReasonTask 
from guf .tasks .countdown import CountdownTask 
from guf .tasks .math500 import MathTask 
from guf .tasks .japanese_financial import JapaneseFinancialTask 
from guf .tasks .mmlu import MMLUTask 
from guf .tasks .rlpr import RLPRTask 
from guf .tasks .livecodebench import LiveCodeBenchTask 
import signal 
import json 
from guf .tasks .livecodebench_utils import check_correctness 
from guf .utils import extract_answer 

class MixMCMJMRLTask (Task ):




    TASK_FIELD_WHITELIST ={
    "medreason":["question","options","answer"],
    "countdown":["nums","target_number"],
    "math500":["problem","answer"],
    "japanese_financial":[

    "sentence","target","polarity",

    "question","choices","answer","context",

    "jp_task_type"
    ],
    "mmlu":["question","choices","answer","subject"],
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
    servers :Optional [Dict [str ,str ]]=None ,
    ports :Optional [Dict [str ,int ]]=None ,
    track_costs :bool =True ,
    debug :bool =False ,
    together :bool =True ,
    valid_ratio :float =0.2 ,
    max_samples :int =-1 ,
    test_ratio :float =0.2 ,
    balanced_sampling :bool =True ,
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


        task_common_args ={
        "llm_names":llm_names ,
        "seed":seed ,
        "max_tokens":max_tokens ,
        "temperature":temperature ,
        "max_turns":max_turns ,
        "servers":servers ,
        "ports":ports ,
        "track_costs":False ,
        "debug":debug ,
        "together":together ,
        "valid_ratio":valid_ratio ,
        "test_ratio":test_ratio ,
        "use_consultant":use_consultant ,
        "log_dir":log_dir ,
        }

        self .medreason_task =MedReasonTask (**task_common_args )
        self .countdown_task =CountdownTask (**task_common_args )
        self .math_task =MathTask (**task_common_args )


        self .japanese_task =JapaneseFinancialTask (
        prompt_language ="jp",
        **task_common_args 
        )

        self .mmlu_task =MMLUTask (**task_common_args )
        self .rlpr_task =RLPRTask (**task_common_args )
        self .livecodebench_task =LiveCodeBenchTask (**task_common_args )

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

            if task_type =="countdown"and "target"in result :
                result ["target_number"]=int (result .pop ("target"))


            if "target"not in result :
                result ["target"]=""

            return result 


        result_dataset =dataset .map (standardize_sample )


        target_features =result_dataset .features .copy ()


        target_features ["answer"]=Value ("string")


        if "target"not in target_features :
            target_features ["target"]=Value ("string")


        result_dataset =result_dataset .cast (Features (target_features ))

        return result_dataset 

    def _format_base_prompt (self ,task_data :Dict )->str :

        task_type =task_data .get ("task_type","unknown")


        expected_fields =self .TASK_FIELD_WHITELIST .get (task_type ,[])
        clean_task_data ={k :v for k ,v in task_data .items ()if k in expected_fields }


        if "task_type"in task_data :
            clean_task_data ["task_type"]=task_data ["task_type"]

        if task_type =="medreason":
            return self .medreason_task ._format_base_prompt (clean_task_data )
        elif task_type =="countdown":
            return self .countdown_task ._format_base_prompt (clean_task_data )
        elif task_type =="math500":
            return self .math_task ._format_base_prompt (clean_task_data )
        elif task_type =="japanese_financial":
            return self .japanese_task ._format_base_prompt (clean_task_data )
        elif task_type =="mmlu":
            return self .mmlu_task ._format_base_prompt (clean_task_data )
        elif task_type =="rlpr":
            return self .rlpr_task ._format_base_prompt (clean_task_data )
        elif task_type =="livecodebench":
            return self .livecodebench_task ._format_base_prompt (clean_task_data )
        else :
            raise ValueError (f"Unknown task type: {task_type}")

    def _create_unified_features (self ):

        from datasets import Features ,Value ,Sequence 

        return Features ({
        "jp_task_type":Value ("string"),
        "answer":Value ("string"),
        "question":Value ("string"),
        "choices":Sequence (Value ("string")),
        "subject":Value ("string"),
        "sentence":Value ("string"),
        "target":Value ("string"),
        "polarity":Value ("string"),
        "nums":Sequence (Value ("int64")),
        "target_number":Value ("int64"),
        "problem":Value ("string"),
        "context":Value ("string"),
        })

    def _standardize_jp_financial (self ,japanese_raw ):

        normalized_japanese_data =[]
        for item in japanese_raw :

            normalized_item ={
            "jp_task_type":item .get ("jp_task_type",""),
            "answer":"",
            "question":"",
            "choices":[],
            "subject":"",
            "sentence":"",
            "target":"",
            "polarity":"",
            "nums":[],
            "target_number":0 ,
            "problem":"",
            "context":"",
            }

            task_type =item .get ("jp_task_type","chabsa")

            if task_type =="chabsa":

                normalized_item ["sentence"]=item .get ("sentence","")
                normalized_item ["target"]=item .get ("target","")
                normalized_item ["polarity"]=item .get ("polarity","")
                normalized_item ["answer"]=str (item .get ("polarity",""))
            else :

                normalized_item ["question"]=item .get ("question","")
                normalized_item ["context"]=item .get ("context","")

                choices =item .get ("choices",[])
                choice_strings =[]
                for choice in choices :
                    if isinstance (choice ,dict ):
                        choice_strings .append (str (choice .get ("text","")))
                    else :
                        choice_strings .append (str (choice ))
                normalized_item ["choices"]=choice_strings 
                normalized_item ["answer"]=str (item .get ("answer",0 ))

            normalized_japanese_data .append (normalized_item )


        unified_features =self ._create_unified_features ()
        japanese_standardized =Dataset .from_list (normalized_japanese_data ,features =unified_features )

        return japanese_standardized 

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


        medreason_splits =self .medreason_task ._load_data (
        seed =seed ,split =split ,validation =validation ,
        valid_ratio =valid_ratio ,max_samples =-1 ,
        test_split =test_split ,test_ratio =test_ratio 
        )

        countdown_splits =self .countdown_task ._load_data (
        seed =seed ,split =split ,validation =validation ,
        valid_ratio =valid_ratio ,max_samples =-1 ,
        test_split =test_split ,test_ratio =test_ratio 
        )

        math_splits =self .math_task ._load_data (
        seed =seed ,split =split ,validation =validation ,
        valid_ratio =valid_ratio ,max_samples =-1 ,
        test_split =test_split ,test_ratio =test_ratio 
        )

        japanese_splits =self .japanese_task ._load_data (
        seed =seed ,split =split ,validation =validation ,
        valid_ratio =valid_ratio ,max_samples =-1 ,
        test_split =test_split ,test_ratio =test_ratio 
        )

        mmlu_splits =self .mmlu_task ._load_data (
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



            japanese_data =self ._standardize_jp_financial (japanese_splits [split_name ])


            medreason_data =self ._standardize_dataset_schema (medreason_splits [split_name ],"medreason")
            countdown_data =self ._standardize_dataset_schema (countdown_splits [split_name ],"countdown")
            math_data =self ._standardize_dataset_schema (math_splits [split_name ],"math500")
            japanese_data =self ._standardize_dataset_schema (japanese_data ,"japanese_financial")
            mmlu_data =self ._standardize_dataset_schema (mmlu_splits [split_name ],"mmlu")
            rlpr_data =self ._standardize_dataset_schema (rlpr_splits [split_name ],"rlpr")
            livecodebench_data =self ._standardize_dataset_schema (livecodebench_splits [split_name ],"livecodebench")

            if self .balanced_sampling :

                def get_length (data ):
                    return len (data )

                min_samples =min (
                get_length (medreason_data ),
                get_length (countdown_data ),
                get_length (math_data ),
                get_length (japanese_data ),
                get_length (mmlu_data ),
                get_length (rlpr_data ),
                get_length (livecodebench_data )
                )

                if self .debug :
                    print (f"Split {split_name} - Original sizes: "
                    f"MedReason={get_length(medreason_data)}, "
                    f"Countdown={get_length(countdown_data)}, "
                    f"Math500={get_length(math_data)}, "
                    f"Japanese={get_length(japanese_data)}, "
                    f"MMLU={get_length(mmlu_data)}, "
                    f"RLPR={get_length(rlpr_data)}, "
                    f"LiveCodeBench={get_length(livecodebench_data)}")
                    print (f"Using {min_samples} samples from each task for balanced sampling")


                def sample_data (data ,num_samples ,seed_offset ):
                    if hasattr (data ,'shuffle')and hasattr (data ,'select'):

                        return data .shuffle (seed =seed +seed_offset ).select (range (num_samples ))
                    else :

                        np .random .seed (seed +seed_offset )
                        indices =np .random .choice (len (data ),size =num_samples ,replace =False )
                        return [data [i ]for i in indices ]












                np .random .seed (seed )
                mmlu_sampled =mmlu_data .shuffle (seed =seed ).select (range (min_samples ))
                math_sampled =math_data .shuffle (seed =seed +2 ).select (range (min_samples ))
                rlpr_sampled =rlpr_data .shuffle (seed =seed +3 ).select (range (min_samples ))
                livecodebench_sampled =livecodebench_data .shuffle (seed =seed +4 ).select (range (min_samples ))
                medreason_sampled =medreason_data .shuffle (seed =seed +5 ).select (range (min_samples ))
                countdown_sampled =countdown_data .shuffle (seed =seed +6 ).select (range (min_samples ))
                japanese_sampled =japanese_data .shuffle (seed =seed +7 ).select (range (min_samples ))











                combined_data =concatenate_datasets ([
                mmlu_sampled ,
                math_sampled ,
                rlpr_sampled ,
                livecodebench_sampled ,
                medreason_sampled ,
                countdown_sampled ,
                japanese_sampled ,
                ])
            else :









                combined_data =concatenate_datasets ([
                mmlu_data ,
                math_data ,
                rlpr_data ,
                livecodebench_data ,
                medreason_data ,
                countdown_data ,
                japanese_data ,
                ])


            combined_data =combined_data .shuffle (seed =seed +len (split_name ))


            if max_samples >0 and len (combined_data )>max_samples :
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

                if "task_type"in data :
                    clean_data ["task_type"]=data ["task_type"]
                clean_batch_data .append (clean_data )

            if task_type =="medreason":
                batch_rewards =self .medreason_task ._calculate_reward (
                batch_completions ,clean_batch_data ,debug =debug 
                )
            elif task_type =="countdown":
                batch_rewards =self .countdown_task ._calculate_reward (
                batch_completions ,clean_batch_data ,debug =debug 
                )
            elif task_type =="math500":
                batch_rewards =self .math_task ._calculate_reward (
                batch_completions ,clean_batch_data ,debug =debug 
                )
            elif task_type =="japanese_financial":
                batch_rewards =self .japanese_task ._calculate_reward (
                batch_completions ,clean_batch_data ,debug =debug 
                )
            elif task_type =="mmlu":
                batch_rewards =self .mmlu_task ._calculate_reward (
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

    def _calculate_livecodebench_reward_for_cma_training (
    self ,completion :str ,task_data :Dict [str ,Any ],debug :bool =False 
    )->float :


        import os 
        import time 
        import tempfile 
        from guf .tasks .livecodebench import CodeGenerationProblem 


        base_log_dir =self .log_dir 


        if base_log_dir is None :
            base_log_dir =tempfile .mkdtemp (prefix ="livecodebench_debug_")



        debug_dir =os .path .join (base_log_dir ,"debug_livecodebench")
        os .makedirs (debug_dir ,exist_ok =True )


        pid =os .getpid ()
        debug_file =os .path .join (debug_dir ,f"livecodebench_debug_{int(time.time())}_{pid}.txt")

        def log_debug (message ):
            try :
                with open (debug_file ,"a")as f :
                    f .write (f"{time.strftime('%H:%M:%S')} - {message}\n")
                    f .flush ()
            except Exception :

                pass 

        log_debug ("=== LiveCodeBench evaluation started (CMA-ES TRAINING VERSION - NO MULTIPROCESSING) ===")
        log_debug (f"Debug file location: {debug_file}")
        log_debug (f"Base log directory: {base_log_dir}")
        log_debug (f"Completion length: {len(completion)}")
        log_debug (f"Task data keys: {list(task_data.keys())}")


        log_debug (f"Problem title: {task_data.get('question_title', 'N/A')}")
        log_debug (f"Platform: {task_data.get('platform', 'N/A')}")
        log_debug (f"Difficulty: {task_data.get('difficulty', 'N/A')}")


        saved_handlers ={}
        for sig in [signal .SIGALRM ,signal .SIGTERM ,signal .SIGINT ]:
            saved_handlers [sig ]=signal .signal (sig ,signal .SIG_DFL )

        try :

            generated_code =extract_answer (completion )

            log_debug ("=== FULL COMPLETION ===")
            log_debug (completion )
            log_debug ("=== END COMPLETION ===")

            if generated_code is None :
                log_debug ("WARNING: No code found in <answer> tags!")
                return 0.0 

            log_debug (f"Extracted code length: {len(generated_code)}")
            log_debug ("=== EXTRACTED CODE ===")
            log_debug (generated_code )
            log_debug ("=== END EXTRACTED CODE ===")


            try :
                problem =CodeGenerationProblem (**task_data )
                evaluation_sample =problem .get_evaluation_sample ()


                sample ={
                "input_output":evaluation_sample ["input_output"]
                }
                log_debug ("Successfully transformed task data to evaluation format")
            except Exception as e :
                log_debug (f"Error transforming task data: {e}")
                import traceback 
                log_debug (f"Transform traceback: {traceback.format_exc()}")
                return 0.0 


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


            timeout =6 
            log_debug ("Using new check_correctness without nested multiprocessing")

            try :

                result ,metadata =check_correctness (
                sample ,
                generation =generated_code ,
                timeout =timeout ,
                debug =False 
                )

                log_debug (f"check_correctness returned: result={result}, metadata={metadata}")


                if isinstance (result ,list ):
                    for i ,test_result in enumerate (result ):
                        if test_result ==True :
                            log_debug (f"Test {i + 1}: PASSED")
                        elif test_result ==-2 :
                            log_debug (f"Test {i + 1}: WRONG ANSWER")
                        elif test_result ==-3 :
                            log_debug (f"Test {i + 1}: TIME LIMIT EXCEEDED")
                        elif test_result ==-4 :
                            log_debug (f"Test {i + 1}: RUNTIME ERROR")
                        elif test_result ==-5 :
                            log_debug (f"Test {i + 1}: TEST RUNNER ERROR")
                        else :
                            log_debug (f"Test {i + 1}: FAILED (code: {test_result})")


                if metadata :
                    log_debug (f"Metadata: {metadata}")
                    if 'error_message'in metadata :
                        log_debug (f"Error message: {metadata['error_message']}")
                    if 'error'in metadata :
                        log_debug (f"Error details: {metadata['error']}")


                if isinstance (result ,list )and result :

                    passed_tests =sum (1 for r in result if r ==True )
                    total_tests =len (result )


                    reward =1.0 if passed_tests ==total_tests and total_tests >0 else 0.0 

                    log_debug (f"=== FINAL RESULTS ===")
                    log_debug (f"Individual test results: {result}")
                    log_debug (f"Tests passed: {passed_tests}/{total_tests}")
                    log_debug (f"Final reward: {reward}")
                    log_debug (f"=== END RESULTS ===")

                    return reward 
                else :
                    log_debug (f"Invalid result format: {result}")
                    return 0.0 

            except Exception as e :
                log_debug (f"Error during code evaluation: {e}")
                import traceback 
                log_debug (f"Evaluation traceback: {traceback.format_exc()}")
                return 0.0 

        except Exception as e :
            log_debug (f"Setup failed: {e}")
            import traceback 
            log_debug (f"Setup traceback: {traceback.format_exc()}")
            return 0.0 
        finally :

            for sig ,handler in saved_handlers .items ():
                signal .signal (sig ,handler )

            log_debug ("=== LiveCodeBench evaluation completed ===")

if __name__ =="__main__":

    import numpy as np 

    llms =["gpt-4o-mini"]


    print ("=== Testing Balanced Sampling ===")
    task_balanced =MixMCMJMRLTask (
    llm_names =llms ,
    debug =True ,
    max_samples =35 ,
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
    task_unbalanced =MixMCMJMRLTask (
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