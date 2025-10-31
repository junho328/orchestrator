


import json 
import requests 
import argparse 
import time 
import os 
import tempfile 
import multiprocessing 
import subprocess 
import sys 
import types 
import unittest 
import shutil 
from datetime import datetime 
from typing import List ,Dict ,Any ,Optional ,Tuple ,Union 
from collections import Counter ,defaultdict 
from concurrent .futures import ProcessPoolExecutor ,as_completed 
import numpy as np 
import re 
import signal 
import resource 
from contextlib import contextmanager 
import builtins 


PASS ="pass"
FAIL ="fail"
TIMEOUT ="timeout"

_SUCCESS =0 
_FAILED =1 
_TIMEOUT =2 
_UNKNOWN =3 

_mapping ={_SUCCESS :PASS ,_FAILED :FAIL ,_TIMEOUT :TIMEOUT ,_UNKNOWN :None }


TIMEOUT_LIMIT =float (os .getenv ("BIGCODEBENCH_TIMEOUT_PER_TASK",240 ))


def load_bigcodebench_dataset (subset :str ="full"):

    try :
        from datasets import load_dataset 


        extra ="-"+subset .capitalize ()if subset !="full"else ""
        dataset_name =f"bigcode/bigcodebench{extra}"

        print (f"Loading dataset: {dataset_name}")
        dataset =load_dataset (dataset_name ,split ="v0.1.4")


        problems ={}
        for task in dataset :
            problems [task ["task_id"]]=task 

        return problems 

    except ImportError :
        print ("datasets library not available. Install with: pip install datasets")
        return None 
    except Exception as e :
        print (f"Error loading dataset: {e}")
        return None 


def format_bigcodebench_prompt (task :Dict [str ,Any ],split :str ="complete")->str :

    try :
        prompt =task [f"{split}_prompt"]
        return prompt 
    except KeyError :
        raise Exception (f"Invalid split {split} for BigCodeBench task {task.get('task_id', 'unknown')}")


def sanitize_code (code :str ,entry_point :str )->str :



    lines =code .split ('\n')
    sanitized_lines =[]

    for line in lines :
        stripped =line .strip ()

        if any (dangerous in stripped .lower ()for dangerous in [
        'exec(','eval(','__import__(',
        'input(','raw_input(',
        'os.system(','os.popen(',
        'import sys','from sys import'
        ]):

            sanitized_lines .append (f"# SANITIZED: {line}")
            continue 
        sanitized_lines .append (line )

    return '\n'.join (sanitized_lines )


@contextmanager 
def time_limit (seconds :float ):


    def signal_handler (signum ,frame ):
        raise TimeoutError ("Operation timed out")

    old_handler =signal .signal (signal .SIGALRM ,signal_handler )
    signal .alarm (int (seconds ))
    try :
        yield 
    finally :
        signal .alarm (0 )
        signal .signal (signal .SIGALRM ,old_handler )


@contextmanager 
def safe_environment ():


    original_limits ={}

    try :

        try :
            current_as =resource .getrlimit (resource .RLIMIT_AS )
            original_limits ['as']=current_as 
            new_as_limit =min (500 *1024 *1024 ,current_as [1 ])
            if new_as_limit <current_as [0 ]:
                resource .setrlimit (resource .RLIMIT_AS ,(new_as_limit ,current_as [1 ]))
        except (ValueError ,OSError ):

            pass 


        try :
            current_cpu =resource .getrlimit (resource .RLIMIT_CPU )
            original_limits ['cpu']=current_cpu 
            new_cpu_limit =min (60 ,current_cpu [1 ])
            if new_cpu_limit <current_cpu [0 ]:
                resource .setrlimit (resource .RLIMIT_CPU ,(new_cpu_limit ,current_cpu [1 ]))
        except (ValueError ,OSError ):

            pass 

        yield 

    finally :

        for limit_type ,original_limit in original_limits .items ():
            try :
                if limit_type =='as':
                    current_limit =resource .getrlimit (resource .RLIMIT_AS )

                    new_limit =(min (original_limit [0 ],current_limit [1 ]),current_limit [1 ])
                    resource .setrlimit (resource .RLIMIT_AS ,new_limit )
                elif limit_type =='cpu':
                    current_limit =resource .getrlimit (resource .RLIMIT_CPU )

                    new_limit =(min (original_limit [0 ],current_limit [1 ]),current_limit [1 ])
                    resource .setrlimit (resource .RLIMIT_CPU ,new_limit )
            except (ValueError ,OSError ):

                pass 


@contextmanager 
def create_tempdir ():

    temp_dir =tempfile .mkdtemp ()
    old_cwd =os .getcwd ()
    try :
        os .chdir (temp_dir )
        yield temp_dir 
    finally :
        try :
            os .chdir (old_cwd )
        except :
            pass 
        try :
            shutil .rmtree (temp_dir ,ignore_errors =True )
        except :

            try :
                import subprocess 
                subprocess .run (['rm','-rf',temp_dir ],check =False ,capture_output =True )
            except :
                pass 


@contextmanager 
def swallow_io ():

    old_stdout =sys .stdout 
    old_stderr =sys .stderr 

    try :
        sys .stdout =open (os .devnull ,'w')
        sys .stderr =open (os .devnull ,'w')
        yield 
    finally :
        sys .stdout .close ()
        sys .stderr .close ()
        sys .stdout =old_stdout 
        sys .stderr =old_stderr 


def unsafe_execute (
entry_point :str ,
code :str ,
test_code :str ,
timeout :float ,
max_as_limit :float ,
max_data_limit :float ,
max_stack_limit :float ,
stat ,
details ,
):

    with safe_environment (),create_tempdir ():
        import os 
        import shutil 
        import builtins 

        rmtree =shutil .rmtree 
        rmdir =os .rmdir 
        chdir =os .chdir 

        module_name ="__test__"
        new_module =types .ModuleType (module_name )
        new_module .__dict__ .update ({
        '__builtins__':builtins ,
        '__file__':f"{module_name}.py",
        '__package__':None ,
        '__doc__':None ,
        'sys':sys ,
        'os':os ,
        'environ':os .environ ,
        })

        try :
            full_code =code +"\n"+test_code 

            with swallow_io ():
                exec (compile (full_code ,f"{module_name}.py",'exec'),new_module .__dict__ )
                sys .modules [module_name ]=new_module 
                TestCases =getattr (new_module ,'TestCases')
                loader =unittest .TestLoader ()
                suite =loader .loadTestsFromTestCase (TestCases )
                test_result =unittest .TestResult ()
                start_time =time .time ()
                with time_limit (timeout ):
                    suite .run (test_result )

            issues =test_result .failures +test_result .errors 
            for test ,trace in issues :
                details [test .id ().split (".")[-1 ]]=trace 
            stat .value =_SUCCESS 

        except BaseException as e :
            details ["ALL"]=str (e )
            stat .value =_FAILED 


        shutil .rmtree =rmtree 
        os .rmdir =rmdir 
        os .chdir =chdir 


def simple_execute_test (code :str ,test_code :str ,entry_point :str ,timeout :float =30.0 )->Tuple [str ,Dict ]:

    try :

        module_name ="__test__"
        new_module =types .ModuleType (module_name )
        new_module .__dict__ .update ({
        '__builtins__':builtins ,
        '__file__':f"{module_name}.py",
        '__package__':None ,
        '__doc__':None ,
        })

        full_code =code +"\n"+test_code 


        exec (compile (full_code ,f"{module_name}.py",'exec'),new_module .__dict__ )


        TestCases =getattr (new_module ,'TestCases')
        loader =unittest .TestLoader ()
        suite =loader .loadTestsFromTestCase (TestCases )


        test_result =unittest .TestResult ()
        suite .run (test_result )


        if test_result .failures or test_result .errors :
            details ={}
            for test ,trace in test_result .failures +test_result .errors :
                details [test .id ().split (".")[-1 ]]=trace 
            return FAIL ,details 
        else :
            return PASS ,{}

    except Exception as e :
        return FAIL ,{"ALL":str (e )}


def untrusted_check (
code :str ,
test_code :str ,
entry_point :str ,
max_as_limit :float =30 *1024 ,
max_data_limit :float =30 *1024 ,
max_stack_limit :float =10 ,
min_time_limit :float =1.0 ,
gt_time_limit :float =60.0 ,
use_simple_execution :bool =False 
)->Tuple [str ,Dict ]:



    if use_simple_execution :
        return simple_execute_test (code ,test_code ,entry_point )

    min_time_limit =max (min_time_limit ,gt_time_limit )
    timeout =max (TIMEOUT_LIMIT ,min_time_limit )+1 


    stat =multiprocessing .Value ("i",_UNKNOWN )
    manager =multiprocessing .Manager ()
    details =manager .dict ()

    p =multiprocessing .Process (
    target =unsafe_execute ,
    args =(
    entry_point ,
    code ,
    test_code ,
    timeout ,
    max_as_limit ,
    max_data_limit ,
    max_stack_limit ,
    stat ,
    details ,
    ),
    )
    p .start ()
    p .join (timeout =timeout +1 )

    if p .is_alive ():
        p .terminate ()
        time .sleep (0.1 )
    if p .is_alive ():
        p .kill ()
        time .sleep (0.1 )

    stat =_mapping [stat .value ]
    details =dict (details )

    if not stat :
        stat =TIMEOUT 
    if stat ==PASS :
        if details :
            stat =FAIL 

    return stat ,details 


def estimate_pass_at_k (
num_samples :Union [int ,List [int ],np .ndarray ],
num_correct :Union [List [int ],np .ndarray ],
k :int ,
)->np .ndarray :


    def estimator (n :int ,c :int ,k :int )->float :

        if n -c <k :
            return 1.0 
        return 1.0 -np .prod (1.0 -k /np .arange (n -c +1 ,n +1 ))

    if isinstance (num_samples ,int ):
        num_samples_it =[num_samples ]*len (num_correct )
    else :
        assert len (num_samples )==len (num_correct )
        num_samples_it =num_samples 

    return np .array (
    [estimator (int (n ),int (c ),k )for n ,c in zip (num_samples_it ,num_correct )]
    )


def extract_code_from_response (response :str )->str :



    code_patterns =[
    r'```python\n(.*?)\n```',
    r'```\n(.*?)\n```',
    r'<code>\n(.*?)\n</code>',
    ]

    all_code_blocks =[]
    for pattern in code_patterns :
        matches =re .findall (pattern ,response ,re .DOTALL )
        for match in matches :

            if '...'not in match and 'function body'not in match .lower ()and len (match .strip ())>20 :
                all_code_blocks .append (match .strip ())


    if all_code_blocks :
        longest_block =max (all_code_blocks ,key =len )
        return normalize_indentation (longest_block )


    lines =response .split ('\n')


    function_starts =[]
    for i ,line in enumerate (lines ):
        if line .strip ().startswith ('def ')and '('in line and ':'in line :
            function_starts .append (i )

    if function_starts :

        best_function =""

        for start_idx in function_starts :
            function_lines =[]
            start_line =lines [start_idx ]
            function_lines .append (start_line )


            func_indent =len (start_line )-len (start_line .lstrip ())


            for i in range (start_idx +1 ,len (lines )):
                line =lines [i ]

                if not line .strip ():
                    function_lines .append (line )
                    continue 

                line_indent =len (line )-len (line .lstrip ())


                if line_indent <=func_indent and line .strip ():

                    if not line .strip ().startswith ('def '):
                        break 

                function_lines .append (line )


                if line .strip ().startswith ('def ')and line_indent ==func_indent :
                    break 

            candidate ='\n'.join (function_lines )

            if len (candidate )>len (best_function )and 'def 'in candidate :
                best_function =candidate 

        if best_function :
            return normalize_indentation (best_function )



    final_code_indicators =[
    "final code:",
    "complete solution:",
    "here's the complete function:",
    "assembled code:",
    "final implementation:"
    ]

    for indicator in final_code_indicators :
        idx =response .lower ().find (indicator )
        if idx !=-1 :

            remaining_text =response [idx :]

            for pattern in code_patterns :
                matches =re .findall (pattern ,remaining_text ,re .DOTALL )
                if matches :
                    return normalize_indentation (matches [0 ].strip ())


    return normalize_indentation (response .strip ())


def normalize_indentation (code :str )->str :

    lines =code .split ('\n')
    if not lines :
        return code 


    while lines and not lines [0 ].strip ():
        lines .pop (0 )
    while lines and not lines [-1 ].strip ():
        lines .pop ()

    if not lines :
        return ""


    min_indent =float ('inf')
    for line in lines :
        if line .strip ():
            indent =len (line )-len (line .lstrip ())
            min_indent =min (min_indent ,indent )


    if min_indent !=float ('inf')and min_indent >0 :
        normalized_lines =[]
        for line in lines :
            if line .strip ():
                normalized_lines .append (line [min_indent :])
            else :
                normalized_lines .append ('')
        return '\n'.join (normalized_lines )

    return '\n'.join (lines )


def debug_single_task ():


    print ("Testing BigCodeBench extraction and evaluation pipeline")
    print ("="*80 )


    sample_task ={
    "task_id":"BigCodeBench/test_task",
    "complete_prompt":"def add_two_numbers(a, b):\n    \"\"\"\n    Add two numbers and return the result.\n    \n    Args:\n        a (int): First number\n        b (int): Second number\n    \n    Returns:\n        int: Sum of a and b\n    \"\"\"\n",
    "test":"import unittest\n\nclass TestCases(unittest.TestCase):\n    def test_case_1(self):\n        self.assertEqual(add_two_numbers(2, 3), 5)\n    def test_case_2(self):\n        self.assertEqual(add_two_numbers(-1, 1), 0)\n    def test_case_3(self):\n        self.assertEqual(add_two_numbers(0, 0), 0)",
    "entry_point":"add_two_numbers"
    }


    test_responses =[
    "```python\ndef add_two_numbers(a, b):\n    return a + b\n```",
    "Here's the solution:\n\n```python\ndef add_two_numbers(a, b):\n    \"\"\"Add two numbers.\"\"\"\n    return a + b\n```\n\nThis function simply adds the two input numbers.",
    "def add_two_numbers(a, b):\n    return a + b",
    "```python\ndef add_two_numbers(a, b):\n    return a * b  # Wrong implementation\n```",
    "I think the solution is:\n\ndef add_two_numbers(a, b):\n    result = a + b\n    return result",
    "Let me solve this step by step:\n\n```python\ndef add_two_numbers(a, b):\n    # Add the two numbers\n    sum_result = a + b\n    return sum_result\n```"
    ]

    print (f"Task ID: {sample_task['task_id']}")
    print (f"Entry Point: {sample_task['entry_point']}")
    print (f"Test Cases: {sample_task['test'].count('def test_')}")
    print ("="*80 )

    for i ,response in enumerate (test_responses ):
        print (f"MODEL RESPONSE {i + 1}:")
        print (f"Raw: {response[:100]}..."if len (response )>100 else f"Raw: {response}")


        extracted_code =extract_code_from_response (response )
        print (f"Extracted Code: {extracted_code}")


        sanitized_code =sanitize_code (extracted_code ,sample_task ['entry_point'])
        print (f"Sanitized Code: {sanitized_code}")


        try :
            status ,details =untrusted_check (
            sanitized_code ,
            sample_task ['test'],
            sample_task ['entry_point'],
            max_as_limit =30 *1024 ,
            max_data_limit =30 *1024 ,
            max_stack_limit =10 ,
            min_time_limit =1.0 ,
            gt_time_limit =10.0 ,
            use_simple_execution =True 
            )
            print (f"Evaluation Result: {status}")
            if details :
                print (f"Details: {details}")
        except Exception as e :
            print (f"Evaluation Error: {e}")
            status ="error"
        print (f"Result: {'✅ PASS' if status == PASS else '❌ FAIL/TIMEOUT'}")
        print ("-"*40 )


def test_with_real_dataset (server_host ="localhost",server_port =8080 ,subset ="full",split ="complete",
use_simple_execution =True ):


    print ("Testing with real BigCodeBench dataset")
    print ("="*80 )

    problems =load_bigcodebench_dataset (subset )
    if problems is None :
        return 


    first_task_id =list (problems .keys ())[0 ]
    sample =problems [first_task_id ]

    print ("REAL DATASET SAMPLE:")
    sample_info ={
    "task_id":sample ["task_id"],
    "entry_point":sample ["entry_point"],
    "prompt_length":len (sample .get (f"{split}_prompt","")),
    "test_length":len (sample .get ("test","")),
    "canonical_solution_length":len (sample .get ("canonical_solution",""))
    }
    print (json .dumps (sample_info ,indent =2 ))
    print ("="*80 )


    query =format_bigcodebench_prompt (sample ,split )

    print (f"FORMATTED QUERY (first 300 chars):")
    print (f"{query[:300]}..."if len (query )>300 else query )
    print (f"Testing with server: http://{server_host}:{server_port}")
    print ("="*80 )


    try :
        print ("Making API call...")
        api_url =f"http://{server_host}:{server_port}/v1/chat/completions"
        response =requests .post (
        api_url ,
        headers ={"Content-Type":"application/json","Authorization":"Bearer sk-test"},
        json ={
        "model":"gpt-4o",
        "messages":[{"role":"user","content":query }],
        "max_tokens":2048 ,
        "temperature":0.1 
        },
        timeout =600 
        )

        if response .status_code ==200 :
            model_response =response .json ()["choices"][0 ]["message"]["content"]
            print ("MODEL RESPONSE:")
            print (model_response [:500 ]+"..."if len (model_response )>500 else model_response )
            print ("="*80 )


            extracted_code =extract_code_from_response (model_response )
            print (f"EXTRACTED CODE:")
            print (extracted_code )
            print ("-"*40 )

            sanitized_code =sanitize_code (extracted_code ,sample ['entry_point'])
            print (f"SANITIZED CODE:")
            print (sanitized_code )
            print ("-"*40 )


            try :
                status ,details =untrusted_check (
                sanitized_code ,
                sample ['test'],
                sample ['entry_point'],
                use_simple_execution =use_simple_execution 
                )
                print (f"EVALUATION RESULT: {status}")
                if details :
                    print (f"DETAILS: {details}")

                print (f"Final Result: {'✅ PASS' if status == PASS else '❌ FAIL/TIMEOUT'}")

            except Exception as e :
                print (f"EVALUATION ERROR: {e}")
                import traceback 
                traceback .print_exc ()

        else :
            print (f"API ERROR: {response.status_code}")
            print (response .text )

    except Exception as e :
        print (f"API TEST ERROR: {e}")
        import traceback 
        traceback .print_exc ()


def test_multiple_samples (max_samples =3 ,server_host ="localhost",server_port =8080 ,
subset ="full",split ="complete",n_samples =1 ,
trial_num =None ,evaluation_results_path =None ,comprehensive_results =None ,
use_simple_execution =True ):


    trial_prefix =f"[Trial {trial_num}] "if trial_num is not None else ""

    print (f"Testing multiple BigCodeBench samples (max_samples={max_samples})")
    print (f"Using server: http://{server_host}:{server_port}")
    print (f"Subset: {subset}, Split: {split}, Samples per task: {n_samples}")
    print ("Note: Each API request has a 10-minute timeout with retry logic")
    print ("="*80 )

    problems =load_bigcodebench_dataset (subset )
    if problems is None :
        return 0.0 ,[]


    print (f"Dataset Info: {len(problems)} total tasks available")

    total_samples =min (max_samples ,len (problems ))
    if total_samples <max_samples :
        print (f"Note: Dataset only has {len(problems)} tasks, so evaluating {total_samples} instead of {max_samples}")

    print (f"Evaluating {total_samples} tasks")
    print ("="*80 )

    results =[]
    pass_count =0 


    if comprehensive_results is not None and trial_num is not None :
        current_trial ={
        "trial_num":trial_num ,
        "start_timestamp":datetime .now ().isoformat (),
        "samples":[],
        "trial_statistics":{
        "total_samples":total_samples ,
        "completed_samples":0 ,
        "pass_count":0 ,
        "current_score":0.0 ,
        "status":"in_progress"
        }
        }
        comprehensive_results ["trials"].append (current_trial )

    task_ids =list (problems .keys ())[:total_samples ]

    for i ,task_id in enumerate (task_ids ):
        task =problems [task_id ]

        print (f"\n{trial_prefix}TASK {i + 1}/{total_samples}")
        print (f"Task ID: {task_id}")
        print (f"Entry Point: {task['entry_point']}")

        sample_start_time =datetime .now ()
        sample_result ={
        "sample_id":i ,
        "trial_num":trial_num ,
        "task_id":task_id ,
        "entry_point":task ['entry_point'],
        "start_timestamp":sample_start_time .isoformat (),
        "status":"in_progress"
        }

        try :

            query =format_bigcodebench_prompt (task ,split )


            task_results =[]

            for sample_idx in range (n_samples ):
                try :

                    max_retries =2 
                    response =None 

                    for retry in range (max_retries ):
                        try :
                            if retry >0 :
                                print (f"   Retry {retry}/{max_retries - 1}...")
                            api_url =f"http://{server_host}:{server_port}/v1/chat/completions"
                            response =requests .post (
                            api_url ,
                            headers ={"Content-Type":"application/json","Authorization":"Bearer sk-test"},
                            json ={
                            "model":"gpt-4o",
                            "messages":[{"role":"user","content":query }],
                            "max_tokens":2048 ,
                            "temperature":0.1 if sample_idx ==0 else 0.7 ,

                            },
                            timeout =600 
                            )
                            break 

                        except (requests .exceptions .ReadTimeout ,requests .exceptions .ConnectionError )as e :
                            print (f"   Request failed: {type(e).__name__}")
                            if retry ==max_retries -1 :
                                print (f"   All {max_retries} attempts failed, skipping sample")
                                raise 
                            print (f"   Retrying in 3 seconds...")
                            time .sleep (3 )

                    if response .status_code ==200 :
                        model_response =response .json ()["choices"][0 ]["message"]["content"]


                        extracted_code =extract_code_from_response (model_response )
                        sanitized_code =sanitize_code (extracted_code ,task ['entry_point'])


                        status ,details =untrusted_check (
                        sanitized_code ,
                        task ['test'],
                        task ['entry_point'],
                        use_simple_execution =use_simple_execution 
                        )

                        task_results .append ({
                        "sample_idx":sample_idx ,
                        "status":status ,
                        "details":details ,
                        "code":sanitized_code ,
                        "response_length":len (model_response )
                        })

                        print (f"   Sample {sample_idx + 1}: {status}")

                    else :
                        print (f"   API Error {sample_idx + 1}: {response.status_code}")
                        task_results .append ({
                        "sample_idx":sample_idx ,
                        "status":"api_error",
                        "details":{"error":f"HTTP_{response.status_code}"},
                        "code":"",
                        "response_length":0 
                        })

                except Exception as e :
                    print (f"   Error sample {sample_idx + 1}: {e}")
                    task_results .append ({
                    "sample_idx":sample_idx ,
                    "status":"error",
                    "details":{"error":str (e )},
                    "code":"",
                    "response_length":0 
                    })


            passed_samples =sum (1 for r in task_results if r ["status"]==PASS )
            task_pass_at_1 =1.0 if passed_samples >0 else 0.0 

            if task_pass_at_1 >0 :
                pass_count +=1 

            print (f"Task Results: {passed_samples}/{len(task_results)} passed")
            print (f"Task Pass@1: {task_pass_at_1:.4f}")


            sample_result .update ({
            "passed":task_pass_at_1 >0 ,
            "pass_at_1":task_pass_at_1 ,
            "passed_samples":passed_samples ,
            "total_samples":len (task_results ),
            "task_results":task_results ,
            "status":"completed",
            "end_timestamp":datetime .now ().isoformat ()
            })

            results .append ({
            "sample_id":i ,
            "trial_num":trial_num ,
            "task_id":task_id ,
            "passed":task_pass_at_1 >0 ,
            "pass_at_1":task_pass_at_1 ,
            "passed_samples":passed_samples ,
            "total_samples":len (task_results ),
            "avg_response_length":sum (r ["response_length"]for r in task_results )/len (
            task_results )if task_results else 0 
            })

        except Exception as e :
            print (f"Error with task {task_id}: {e}")

            sample_result .update ({
            "passed":False ,
            "error":str (e ),
            "status":"failed",
            "end_timestamp":datetime .now ().isoformat ()
            })

            results .append ({
            "sample_id":i ,
            "trial_num":trial_num ,
            "task_id":task_id ,
            "passed":False ,
            "error":str (e )
            })


        running_average =pass_count /(i +1 )
        print (f"Running Pass@1: {pass_count}/{i + 1} = {running_average:.4f} ({running_average * 100:.2f}%)")
        print ("-"*60 )


        if comprehensive_results is not None and trial_num is not None :
            current_trial =comprehensive_results ["trials"][-1 ]
            current_trial ["samples"].append (sample_result )

            current_trial ["trial_statistics"].update ({
            "completed_samples":i +1 ,
            "pass_count":pass_count ,
            "current_score":running_average 
            })

            comprehensive_results ["overall_statistics"]["total_samples_evaluated"]=sum (
            len (trial ["samples"])for trial in comprehensive_results ["trials"]
            )


            if evaluation_results_path :
                try :
                    with open (evaluation_results_path ,'w')as f :
                        json .dump (comprehensive_results ,f ,indent =2 )
                    print (f"Progress saved: Sample {i + 1}/{total_samples} (Score: {running_average:.4f})")
                except Exception as save_error :
                    print (f"Failed to save progress: {save_error}")


    average_score =pass_count /total_samples if total_samples >0 else 0.0 

    print (f"\n{'=' * 80}")
    print (f"{trial_prefix}FINAL RESULTS:")
    print (f"   Total Tasks: {total_samples}")
    print (f"   Passed: {pass_count}")
    print (f"   Failed: {total_samples - pass_count}")
    print (f"   Pass@1 Score: {average_score:.4f} ({average_score * 100:.2f}%)")
    print (f"{'=' * 80}")


    if comprehensive_results is not None and trial_num is not None :
        current_trial =comprehensive_results ["trials"][-1 ]
        current_trial ["trial_statistics"].update ({
        "final_score":average_score ,
        "status":"completed",
        "end_timestamp":datetime .now ().isoformat ()
        })


    if trial_num is None :
        timestamp =datetime .now ().strftime ("%Y%m%d_%H%M%S")
        results_file =f"bigcodebench_debug_results_{timestamp}.json"

        summary ={
        "timestamp":timestamp ,
        "subset":subset ,
        "split":split ,
        "total_samples":total_samples ,
        "dataset_length":len (problems ),
        "requested_max_samples":max_samples ,
        "pass_count":pass_count ,
        "average_score":average_score ,
        "detailed_results":results 
        }

        with open (results_file ,'w')as f :
            json .dump (summary ,f ,indent =2 )

        print (f"Detailed results saved to: {results_file}")

    return average_score ,results 


def main ():
    parser =argparse .ArgumentParser (description ="Debug BigCodeBench evaluation pipeline")
    parser .add_argument ("--max_samples",type =int ,default =30 ,
    help ="Maximum number of samples to test (default: 30)")
    parser .add_argument ("--repeat",type =int ,default =3 ,
    help ="Number of times to repeat the evaluation (default: 3)")
    parser .add_argument ("--evaluation_results_path",type =str ,default =None ,
    help ="Path to save continuous evaluation results (default: auto-generated)")
    parser .add_argument ("--skip_basic_tests",action ="store_true",
    help ="Skip basic extraction and evaluation tests")
    parser .add_argument ("--server_host",type =str ,default ="slurm0us-gufnodeset-2",
    help ="Server host address (default: slurm0us-gufnodeset-2)")
    parser .add_argument ("--server_port",type =int ,default =8082 ,
    help ="Server port (default: 8082)")
    parser .add_argument ("--subset",type =str ,default ="hard",choices =["full","hard"],
    help ="BigCodeBench subset to use (default: hard)")
    parser .add_argument ("--split",type =str ,default ="complete",choices =["complete","instruct"],
    help ="BigCodeBench split to use (default: complete)")
    parser .add_argument ("--use_multiprocessing",action ="store_true",
    help ="Use multiprocessing for code execution (default: False, uses simple execution)")
    parser .add_argument ("--n_samples",type =int ,default =1 ,
    help ="Number of solutions to generate per task (default: 1)")

    args =parser .parse_args ()


    use_simple_execution =not args .use_multiprocessing 


    if args .evaluation_results_path is None :
        timestamp =datetime .now ().strftime ("%Y%m%d_%H%M%S")
        args .evaluation_results_path =f"bigcodebench_evaluation_results_{timestamp}.json"

    print ("Debugging BigCodeBench evaluation pipeline")
    print (f"Server: http://{args.server_host}:{args.server_port}")
    print (f"Repeats: {args.repeat}")
    print (f"Samples per trial: {args.max_samples}")
    print (f"Subset: {args.subset}, Split: {args.split}")
    print (f"Solutions per task: {args.n_samples}")
    print (f"Execution mode: {'Simple' if use_simple_execution else 'Multiprocessing'}")
    print (f"Results path: {args.evaluation_results_path}")
    print ("="*80 )

    if not args .skip_basic_tests :
        print ("1️⃣  Testing extraction and evaluation pipeline...")
        debug_single_task ()

        print ("\n2️⃣  Testing with single real dataset sample...")
        test_with_real_dataset (args .server_host ,args .server_port ,args .subset ,args .split ,use_simple_execution )


    comprehensive_results ={
    "metadata":{
    "start_timestamp":datetime .now ().isoformat (),
    "repeat":args .repeat ,
    "max_samples":args .max_samples ,
    "subset":args .subset ,
    "split":args .split ,
    "n_samples":args .n_samples ,
    "server_host":args .server_host ,
    "server_port":args .server_port ,
    "evaluation_results_path":args .evaluation_results_path ,
    "status":"in_progress"
    },
    "trials":[],
    "overall_statistics":{
    "completed_trials":0 ,
    "total_samples_evaluated":0 ,
    "current_mean_score":0.0 ,
    "trial_scores":[]
    }
    }


    print (f"\n3️⃣  Testing multiple trials (repeat={args.repeat}, max_samples={args.max_samples})...")

    trial_scores =[]
    all_results =[]

    for trial in range (args .repeat ):
        print (f"\n{'🟦' * 20} TRIAL {trial + 1}/{args.repeat} {'🟦' * 20}")

        trial_score ,trial_results =test_multiple_samples (
        args .max_samples ,args .server_host ,args .server_port ,
        args .subset ,args .split ,args .n_samples ,
        trial_num =trial +1 ,evaluation_results_path =args .evaluation_results_path ,
        comprehensive_results =comprehensive_results ,
        use_simple_execution =use_simple_execution 
        )

        trial_scores .append (trial_score )
        all_results .extend (trial_results )


        comprehensive_results ["overall_statistics"]["completed_trials"]=trial +1 
        comprehensive_results ["overall_statistics"]["total_samples_evaluated"]=len (all_results )
        comprehensive_results ["overall_statistics"]["trial_scores"]=trial_scores 
        if trial_scores :
            comprehensive_results ["overall_statistics"]["current_mean_score"]=sum (trial_scores )/len (trial_scores )


        with open (args .evaluation_results_path ,'w')as f :
            json .dump (comprehensive_results ,f ,indent =2 )

        print (f"\nTRIAL {trial + 1}/{args.repeat} SCORE: {trial_score:.4f} ({trial_score * 100:.2f}%)")
        print (f"Progress saved to: {args.evaluation_results_path}")
        print (f"{'🟦' * (42 + len(str(trial + 1)) + len(str(args.repeat)))}")


    if trial_scores :
        import statistics 

        mean_score =statistics .mean (trial_scores )
        if len (trial_scores )>1 :
            std_score =statistics .stdev (trial_scores )
            min_score =min (trial_scores )
            max_score =max (trial_scores )
        else :
            std_score =0.0 
            min_score =max_score =mean_score 


        comprehensive_results ["metadata"]["end_timestamp"]=datetime .now ().isoformat ()
        comprehensive_results ["metadata"]["status"]="completed"
        comprehensive_results ["overall_statistics"].update ({
        "final_mean_score":mean_score ,
        "std_score":std_score ,
        "min_score":min_score ,
        "max_score":max_score ,
        "total_trials":len (trial_scores ),
        "individual_trial_scores":[
        {"trial_num":i +1 ,"score":score }
        for i ,score in enumerate (trial_scores )
        ]
        })


        with open (args .evaluation_results_path ,'w')as f :
            json .dump (comprehensive_results ,f ,indent =2 )

        print (f"\n{'=' * 80}")
        print (f"OVERALL RESULTS ACROSS {args.repeat} TRIALS:")
        print (f"   Mean Pass@1: {mean_score:.4f} ({mean_score * 100:.2f}%)")
        if len (trial_scores )>1 :
            print (f"   Std Dev: {std_score:.4f}")
            print (f"   Min Score: {min_score:.4f} ({min_score * 100:.2f}%)")
            print (f"   Max Score: {max_score:.4f} ({max_score * 100:.2f}%)")

        print (f"   Individual Trial Scores:")
        for i ,score in enumerate (trial_scores ):
            print (f"     Trial {i + 1}: {score:.4f} ({score * 100:.2f}%)")
        print (f"{'=' * 80}")

        print (f"Final comprehensive results saved to: {args.evaluation_results_path}")

        return mean_score 
    else :

        comprehensive_results ["metadata"]["end_timestamp"]=datetime .now ().isoformat ()
        comprehensive_results ["metadata"]["status"]="failed"

        with open (args .evaluation_results_path ,'w')as f :
            json .dump (comprehensive_results ,f ,indent =2 )

        print ("No trials completed successfully")
        print (f"Failed results saved to: {args.evaluation_results_path}")
        return 0.0 


if __name__ =="__main__":
    final_score =main ()