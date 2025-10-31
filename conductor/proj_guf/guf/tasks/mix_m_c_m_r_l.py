

from typing import Dict ,List ,Any ,Optional 
import numpy as np 
from datasets import concatenate_datasets 
from guf .core import Task 
from guf .tasks .medreason import MedReasonTask 
from guf .tasks .countdown import CountdownTask 
from guf .tasks .math500 import MathTask 
from guf .tasks .rlpr import RLPRTask 
from guf .tasks .livecodebench import LiveCodeBenchTask 


class MixMCMRLTask (Task ):



    TASK_FIELD_WHITELIST ={
    "medreason":["question","options","answer"],
    "countdown":["nums","target"],
    "math500":["problem","answer"],
    "rlpr":["prompt","ability","ground_truth","style","reward_model"],
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


        self .medreason_task =MedReasonTask (
        llm_names =llm_names ,seed =seed ,max_tokens =max_tokens ,
        temperature =temperature ,max_turns =max_turns ,servers =servers ,
        ports =ports ,track_costs =False ,debug =debug ,together =together ,
        valid_ratio =valid_ratio ,test_ratio =test_ratio ,use_consultant =use_consultant ,
        )

        self .countdown_task =CountdownTask (
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

    def _format_base_prompt (self ,task_data :Dict )->str :

        task_type =task_data .get ("task_type","unknown")


        expected_fields =self .TASK_FIELD_WHITELIST .get (task_type ,[])
        clean_task_data ={k :v for k ,v in task_data .items ()if k in expected_fields }

        if task_type =="medreason":
            return self .medreason_task ._format_base_prompt (clean_task_data )
        elif task_type =="countdown":
            return self .countdown_task ._format_base_prompt (clean_task_data )
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

            medreason_data =medreason_splits [split_name ].add_column (
            "task_type",["medreason"]*len (medreason_splits [split_name ])
            )
            countdown_data =countdown_splits [split_name ].add_column (
            "task_type",["countdown"]*len (countdown_splits [split_name ])
            )
            math_data =math_splits [split_name ].add_column (
            "task_type",["math500"]*len (math_splits [split_name ])
            )
            rlpr_data =rlpr_splits [split_name ].add_column (
            "task_type",["rlpr"]*len (rlpr_splits [split_name ])
            )
            livecodebench_data =livecodebench_splits [split_name ].add_column (
            "task_type",["livecodebench"]*len (livecodebench_splits [split_name ])
            )

            if self .balanced_sampling :

                min_samples =min (
                len (medreason_data ),
                len (countdown_data ),
                len (math_data ),
                len (rlpr_data ),
                len (livecodebench_data )
                )

                if self .debug :
                    print (f"Split {split_name} - Original sizes: "
                    f"MedReason={len(medreason_data)}, "
                    f"Countdown={len(countdown_data)}, "
                    f"Math500={len(math_data)}, "
                    f"RLPR={len(rlpr_data)}, "
                    f"LiveCodeBench={len(livecodebench_data)}")
                    print (f"Using {min_samples} samples from each task for balanced sampling")


                np .random .seed (seed )
                medreason_sampled =medreason_data .shuffle (seed =seed ).select (range (min_samples ))
                countdown_sampled =countdown_data .shuffle (seed =seed +1 ).select (range (min_samples ))
                math_sampled =math_data .shuffle (seed =seed +2 ).select (range (min_samples ))
                rlpr_sampled =rlpr_data .shuffle (seed =seed +3 ).select (range (min_samples ))
                livecodebench_sampled =livecodebench_data .shuffle (seed =seed +4 ).select (range (min_samples ))


                combined_data =concatenate_datasets ([
                medreason_sampled ,
                countdown_sampled ,
                math_sampled ,
                rlpr_sampled ,
                livecodebench_sampled 
                ])
            else :

                combined_data =concatenate_datasets ([
                medreason_data ,
                countdown_data ,
                math_data ,
                rlpr_data ,
                livecodebench_data 
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
                clean_batch_data .append (clean_data )


            if False :
                print (f"\n=== DEBUG: Processing {task_type} task ===")
                print (f"Number of items: {len(clean_batch_data)}")
                print (f"Expected fields for {task_type}: {expected_fields}")
                if clean_batch_data :
                    sample_data =clean_batch_data [0 ]
                    print (f"Actual fields being passed: {list(sample_data.keys())}")
                    print (f"Sample data preview: {dict(list(sample_data.items())[:3])}")

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
            elif task_type =="rlpr":
                batch_rewards =self .rlpr_task ._calculate_reward (
                batch_completions ,clean_batch_data ,debug =debug 
                )
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


if __name__ =="__main__":

    import numpy as np 

    llms =["gpt-4o-mini"]


    print ("=== Testing Balanced Sampling ===")
    task_balanced =MixMCMRLTask (
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
    task_unbalanced =MixMCMRLTask (
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