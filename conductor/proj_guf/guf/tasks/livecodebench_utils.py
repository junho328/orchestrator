


import ast 
import faulthandler 
import json 
import os 
import signal 
import sys 
import time 
import threading 
from collections import defaultdict 
from datetime import datetime 
from decimal import Decimal 
from enum import Enum 
from io import StringIO 
from types import ModuleType 
from unittest .mock import patch ,mock_open 

import traceback 
import numpy as np 


sys .set_int_max_str_digits (50000 )
os .environ ["TOKENIZERS_PARALLELISM"]="false"

import_string ="from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"


def truncatefn (s ,length =300 ):
    if isinstance (s ,str ):
        pass 
    else :
        s =str (s )
    if len (s )<=length :
        return s 

    return s [:length //2 ]+"...(truncated) ..."+s [-length //2 :]


class CODE_TYPE (Enum ):
    call_based =0 
    standard_input =1 



class TimeoutException (Exception ):
    pass 


def timeout_handler (signum ,frame ):
    print ("timeout occured: alarm went off")
    raise TimeoutException 





class Capturing (list ):
    def __enter__ (self ):
        self ._stdout =sys .stdout 
        sys .stdout =self ._stringio =StringIO ()

        self ._stringio .close =lambda x :1 
        return self 

    def __exit__ (self ,*args ):
        self .append (self ._stringio .getvalue ())
        del self ._stringio 
        sys .stdout =self ._stdout 



class MockStdinWithBuffer :
    def __init__ (self ,inputs :str ):
        self .inputs =inputs 
        self ._stringio =StringIO (inputs )
        self .buffer =MockBuffer (inputs )

    def read (self ,*args ):
        return self .inputs 

    def readline (self ,*args ):
        return self ._stringio .readline (*args )

    def readlines (self ,*args ):
        return self .inputs .split ("\n")

    def __getattr__ (self ,name ):

        return getattr (self ._stringio ,name )


class MockBuffer :
    def __init__ (self ,inputs :str ):
        self .inputs =inputs .encode ("utf-8")

    def read (self ,*args ):

        return self .inputs 

    def readline (self ,*args ):
        return self .inputs .split (b"\n")[0 ]+b"\n"


def clean_if_name (code :str )->str :
    try :
        astree =ast .parse (code )
        last_block =astree .body [-1 ]
        if isinstance (last_block ,ast .If ):
            condition =last_block .test 
            if ast .unparse (condition ).strip ()=="__name__ == '__main__'":
                code =(
                ast .unparse (astree .body [:-1 ])+"\n"+ast .unparse (last_block .body )
                )
    except :
        pass 

    return code 


def make_function (code :str )->str :
    try :
        import_stmts =[]
        all_other_stmts =[]
        astree =ast .parse (code )
        for stmt in astree .body :
            if isinstance (stmt ,(ast .Import ,ast .ImportFrom )):
                import_stmts .append (stmt )
            else :
                all_other_stmts .append (stmt )

        function_ast =ast .FunctionDef (
        name ="wrapped_function",
        args =ast .arguments (
        posonlyargs =[],args =[],kwonlyargs =[],kw_defaults =[],defaults =[]
        ),
        body =all_other_stmts ,
        decorator_list =[],
        lineno =-1 ,
        )
        main_code =(
        import_string 
        +"\n"
        +ast .unparse (import_stmts )
        +"\n"
        +ast .unparse (function_ast )
        )
        return main_code 
    except Exception as e :
        return code 


def call_method (method ,inputs ):
    if isinstance (inputs ,list ):
        inputs ="\n".join (inputs )

    inputs_line_iterator =iter (inputs .split ("\n"))


    mock_stdin =MockStdinWithBuffer (inputs )




    @patch ("builtins.open",mock_open (read_data =inputs ))
    @patch ("sys.stdin",mock_stdin )
    @patch ("sys.stdin.readline",lambda *args :next (inputs_line_iterator ))
    @patch ("sys.stdin.readlines",lambda *args :inputs .split ("\n"))
    @patch ("sys.stdin.read",lambda *args :inputs )

    def _inner_call_method (_method ):
        try :
            return _method ()
        except SystemExit as e :
            pass 
        finally :
            pass 

    return _inner_call_method (method )


def get_function (compiled_sol ,fn_name :str ):
    try :
        assert hasattr (compiled_sol ,fn_name )
        return getattr (compiled_sol ,fn_name )
    except Exception as e :
        return 


def validate_code_safety (code :str )->tuple [bool ,str ]:

    try :

        ast .parse (code )
    except SyntaxError as e :
        return False ,f"Syntax error: {e}"

    content_lower =code .lower ()


    threading_patterns =[
    'threading.','thread(','_thread.','concurrent.futures',
    'multiprocessing.','asyncio.','async def','await '
    ]

    for pattern in threading_patterns :
        if pattern in content_lower :
            return False ,f"Potentially problematic threading/async pattern: {pattern}"


    lines =code .split ('\n')
    for i ,line in enumerate (lines ):
        line_stripped =line .strip ()

        if ('='in line_stripped and 
        (','in line_stripped .split ('=')[0 ]or 
        'for 'in line_stripped and ','in line_stripped )):

            if any (keyword in line_stripped for keyword in ['def main','if __name__']):
                return False ,f"Potentially risky unpacking pattern at line {i + 1}: {line_stripped}"

    return True ,""


def compile_code (code :str ,timeout :int ):

    try :

        is_safe ,safety_error =validate_code_safety (code )
        if not is_safe :
            return None 

        signal .alarm (timeout )
        tmp_sol =ModuleType ("tmp_sol","")


        thread_exceptions =[]
        original_excepthook =threading .excepthook 

        def capture_thread_exception (args ):

            thread_exceptions .append ({
            'thread':args .thread ,
            'exception':args .exc_value ,
            'traceback':args .exc_traceback 
            })


        threading .excepthook =capture_thread_exception 

        try :

            exec (code ,tmp_sol .__dict__ )



            time .sleep (0.1 )


            if thread_exceptions :

                for exc_info in thread_exceptions :
                    print (f"Thread exception caught: {exc_info['exception']}")
                return None 


            if "class Solution"in code :




                compiled_sol =tmp_sol .Solution ()
            else :

                compiled_sol =tmp_sol 

            assert compiled_sol is not None 
            return compiled_sol 

        finally :

            threading .excepthook =original_excepthook 

    except Exception as e :

        error_msg =str (e )
        if "not enough values to unpack"in error_msg :
            print (f"Caught unpacking error during compilation: {error_msg}")
        elif "too many values to unpack"in error_msg :
            print (f"Caught unpacking error during compilation: {error_msg}")
        return None 
    finally :
        signal .alarm (0 )


def convert_line_to_decimals (line :str )->tuple [bool ,list [Decimal ]]:
    try :
        decimal_line =[Decimal (elem )for elem in line .split ()]
    except :
        return False ,[]
    return True ,decimal_line 


def get_stripped_lines (val :str ):

    val =val .strip ()

    return [val_line .strip ()for val_line in val .split ("\n")]


def grade_call_based (
code :str ,all_inputs :list ,all_outputs :list ,fn_name :str ,timeout :int 
):


    code =import_string +"\n\n"+code 
    compiled_sol =compile_code (code ,timeout )

    if compiled_sol is None :
        return [-4 ],{
        "error_code":-4 ,
        "error_message":"Compilation failed",
        }

    method =get_function (compiled_sol ,fn_name )

    if method is None :
        return [-4 ],{
        "error_code":-4 ,
        "error_message":f"Function {fn_name} not found",
        }

    all_inputs =[
    [json .loads (line )for line in inputs .split ("\n")]for inputs in all_inputs 
    ]

    all_outputs =[json .loads (output )for output in all_outputs ]

    total_execution =0 
    all_results =[]
    for idx ,(gt_inp ,gt_out )in enumerate (zip (all_inputs ,all_outputs )):
        signal .alarm (timeout )
        faulthandler .enable ()
        try :

            start =time .time ()
            prediction =method (*gt_inp )
            total_execution +=time .time ()-start 
            signal .alarm (0 )



            if isinstance (prediction ,tuple ):
                prediction =list (prediction )

            tmp_result =prediction ==gt_out 



            all_results .append (tmp_result )

            if not tmp_result :
                return all_results ,{
                "output":truncatefn (prediction ),
                "inputs":truncatefn (gt_inp ),
                "expected":truncatefn (gt_out ),
                "error_code":-2 ,
                "error_message":"Wrong Answer",
                }
        except Exception as e :
            signal .alarm (0 )
            error_str =repr (e ).lower ()
            if "timeoutexception"in error_str :
                all_results .append (-3 )
                return all_results ,{
                "error":repr (e ),
                "error_code":-3 ,
                "error_message":"Time Limit Exceeded",
                "inputs":truncatefn (gt_inp ),
                "expected":truncatefn (gt_out ),
                }
            elif "not enough values to unpack"in error_str or "too many values to unpack"in error_str :
                all_results .append (-4 )
                return all_results ,{
                "error":repr (e ),
                "error_code":-4 ,
                "error_message":"Runtime Error (Unpacking)",
                "inputs":truncatefn (gt_inp ),
                "expected":truncatefn (gt_out ),
                }
            else :
                all_results .append (-4 )
                return all_results ,{
                "error":repr (e ),
                "error_code":-4 ,
                "error_message":"Runtime Error",
                "inputs":truncatefn (gt_inp ),
                "expected":truncatefn (gt_out ),
                }

        finally :
            signal .alarm (0 )
            faulthandler .disable ()

    return all_results ,{"execution time":total_execution }


def grade_stdio (
code :str ,
all_inputs :list ,
all_outputs :list ,
timeout :int ,
):

    code =clean_if_name (code )


    code =make_function (code )

    compiled_sol =compile_code (code ,timeout )
    if compiled_sol is None :
        return [-4 ],{
        "error_code":-4 ,
        "error_message":"Compilation failed",
        }

    method =get_function (compiled_sol ,"wrapped_function")

    if method is None :
        return [-4 ],{
        "error_code":-4 ,
        "error_message":"Function wrapped_function not found",
        }

    all_results =[]
    total_execution_time =0 
    for idx ,(gt_inp ,gt_out )in enumerate (zip (all_inputs ,all_outputs )):
        signal .alarm (timeout )
        faulthandler .enable ()

        signal .alarm (timeout )
        with Capturing ()as captured_output :
            try :
                start =time .time ()
                call_method (method ,gt_inp )
                total_execution_time +=time .time ()-start 

                signal .alarm (0 )
            except Exception as e :
                signal .alarm (0 )
                error_str =repr (e ).lower ()
                if "timeoutexception"in error_str :
                    all_results .append (-3 )
                    return all_results ,{
                    "error":repr (e ),
                    "error_code":-3 ,
                    "error_message":"Time Limit Exceeded",
                    "inputs":truncatefn (gt_inp ),
                    "expected":truncatefn (gt_out ),
                    }
                elif "not enough values to unpack"in error_str or "too many values to unpack"in error_str :
                    all_results .append (-4 )
                    return all_results ,{
                    "error":repr (e ),
                    "error_code":-4 ,
                    "error_message":"Runtime Error (Unpacking)",
                    "inputs":truncatefn (gt_inp ),
                    "expected":truncatefn (gt_out ),
                    }
                else :
                    all_results .append (-4 )
                    return all_results ,{
                    "error":repr (e ),
                    "error_code":-4 ,
                    "error_message":"Runtime Error",
                    "inputs":truncatefn (gt_inp ),
                    "expected":truncatefn (gt_out ),
                    }

            finally :
                signal .alarm (0 )
                faulthandler .disable ()

        prediction =captured_output [0 ]

        stripped_prediction_lines =get_stripped_lines (prediction )
        stripped_gt_out_lines =get_stripped_lines (gt_out )



        WA_send_args ={
        "output":truncatefn (prediction ),
        "inputs":truncatefn (gt_inp ),
        "expected":truncatefn (gt_out ),
        "error_code":-2 ,
        }

        if len (stripped_prediction_lines )!=len (stripped_gt_out_lines ):
            all_results .append (-2 )
            WA_send_args ["error_message"]="Wrong answer: mismatched output length"
            return all_results ,WA_send_args 

        for output_line_idx ,(
        stripped_prediction_line ,
        stripped_gt_out_line ,
        )in enumerate (zip (stripped_prediction_lines ,stripped_gt_out_lines )):
            WA_send_args ["error_message"]=(
            f"Wrong answer at {output_line_idx=}: {truncatefn(stripped_prediction_line)} != {truncatefn(stripped_gt_out_line)}"
            )


            if stripped_prediction_line ==stripped_gt_out_line :
                continue 







            success ,decimal_prediction_line =convert_line_to_decimals (
            stripped_prediction_line 
            )
            if not success :
                all_results .append (-2 )
                return all_results ,WA_send_args 
            success ,decimal_gtout_line =convert_line_to_decimals (stripped_gt_out_line )
            if not success :
                all_results .append (-2 )
                return all_results ,WA_send_args 

            if decimal_prediction_line ==decimal_gtout_line :
                continue 

            all_results .append (-2 )
            return all_results ,WA_send_args 
        all_results .append (True )

    return all_results ,{"execution time":total_execution_time }


def run_test (sample ,test =None ,debug =False ,timeout =6 ,using_closed_models =False ):

    signal .signal (signal .SIGALRM ,timeout_handler )




    if debug :
        print (f"start = {datetime.now().time()}")

    try :
        in_outs =json .loads (sample ["input_output"])
    except ValueError as e :
        raise e 
        in_outs =None 

    if in_outs :
        if in_outs .get ("fn_name")is None :
            which_type =CODE_TYPE .standard_input 
            method_name =None 
        else :
            which_type =CODE_TYPE .call_based 
            method_name =in_outs ["fn_name"]

    if debug :
        print (f"loaded input_output = {datetime.now().time()}")

    if test is None :
        assert False ,"should not happen: test code is none"
        return in_outs ,{"error":"no test code provided"}
    elif test is not None :
        results =[]
        sol =import_string 
        if debug :
            print (f"loading test code = {datetime.now().time()}")
            print (f"test code looks like: {test}")

        if which_type ==CODE_TYPE .call_based :

            if not using_closed_models :
                signal .alarm (timeout )
            try :
                result =grade_call_based (
                code =test ,
                all_inputs =in_outs ["inputs"],
                all_outputs =in_outs ["outputs"],
                fn_name =method_name ,
                timeout =timeout ,
                )
                if result is None or len (result )!=2 :
                    return [-4 ],{
                    "error_code":-4 ,
                    "error_message":"grade_call_based returned invalid result",
                    }
                results ,metadata =result 
                return results ,metadata 
            except Exception as e :
                if debug :

                    print ("Full traceback:")
                    traceback .print_exc ()


                error_str =str (e )
                if "not enough values to unpack"in error_str or "too many values to unpack"in error_str :
                    return [-4 ],{
                    "error_code":-4 ,
                    "error_message":f"Unpacking error during testing: {error_str}",
                    }

                return [-4 ],{
                "error_code":-4 ,
                "error_message":f"Error during testing: {e}",
                }
            finally :
                signal .alarm (0 )
        elif which_type ==CODE_TYPE .standard_input :

            if not using_closed_models :
                signal .alarm (timeout )
            try :
                result =grade_stdio (
                code =test ,
                all_inputs =in_outs ["inputs"],
                all_outputs =in_outs ["outputs"],
                timeout =timeout ,
                )
                if result is None or len (result )!=2 :
                    return [-4 ],{
                    "error_code":-4 ,
                    "error_message":"grade_stdio returned invalid result",
                    }
                results ,metadata =result 
                return results ,metadata 
            except Exception as e :
                if debug :

                    print ("Full traceback:")
                    traceback .print_exc ()


                error_str =str (e )
                if "not enough values to unpack"in error_str or "too many values to unpack"in error_str :
                    return [-4 ],{
                    "error_code":-4 ,
                    "error_message":f"Unpacking error during testing: {error_str}",
                    }

                return [-4 ],{
                "error_code":-4 ,
                "error_message":f"Error during testing: {e}",
                }
            finally :
                signal .alarm (0 )


def estimate_pass_at_k (num_samples ,num_correct ,k ):


    def estimator (n :int ,c :int ,k :int )->float :

        if n -c <k :
            return 1.0 
        return 1.0 -np .prod (1.0 -k /np .arange (n -c +1 ,n +1 ))

    import itertools 

    if isinstance (num_samples ,int ):
        num_samples_it =itertools .repeat (num_samples ,len (num_correct ))
    else :
        assert len (num_samples )==len (num_correct )
        num_samples_it =iter (num_samples )

    return np .array (
    [estimator (int (n ),int (c ),k )for n ,c in zip (num_samples_it ,num_correct )]
    )


def compute_metrics_from_results (results ,k_list =[1 ,5 ]):
    total =[]
    correct =[]
    task_ids =[]
    for task_id ,res in results .items ():
        all_correct =[]
        for generation in res :
            gen =np .array (generation )
            all_correct .append (np .all (gen >0 ))
        task_ids .append (task_id )
        total .append (len (all_correct ))
        correct .append (sum (all_correct ))
    total =np .array (total )
    correct =np .array (correct )
    ks =k_list 
    detail_pass_at_k ={
    f"pass@{k}":estimate_pass_at_k (total ,correct ,k ).tolist ()
    for k in ks 
    if (total >=k ).all ()
    }
    pass_at_k ={
    f"pass@{k}":estimate_pass_at_k (total ,correct ,k ).mean ()
    for k in ks 
    if (total >=k ).all ()
    }
    detail_metrics ={k :dict (zip (task_ids ,v ))for k ,v in detail_pass_at_k .items ()}
    pass_at_k ["detail"]=detail_metrics 
    return pass_at_k 


def check_correctness (sample ,generation ,timeout ,debug =True ,using_closed_models =False ):


    result =[]
    metadata ={}


    old_handler =signal .signal (signal .SIGALRM ,timeout_handler )

    try :

        in_outs =json .loads (sample ["input_output"])

        if using_closed_models :


            total_timeout =0 
        else :

            total_timeout =(timeout +1 )*len (in_outs ["inputs"])+5 

            signal .alarm (total_timeout )

        try :

            res ,meta =run_test (
            sample ,
            test =generation ,
            debug =debug ,
            timeout =timeout ,
            using_closed_models =using_closed_models 
            )
            result =res 
            metadata =meta 
        except TimeoutException :

            result =[[-1 for _ in range (len (in_outs ["inputs"]))]]
            metadata ={
            "error":"Global timeout",
            "error_code":-3 ,
            "error_message":"Time Limit Exceeded (Global)"
            }
            if debug :
                print (f"global timeout")
        except Exception as e :

            error_str =str (e )
            if "not enough values to unpack"in error_str or "too many values to unpack"in error_str :
                result =[[-1 for _ in range (len (in_outs ["inputs"]))]]
                metadata ={
                "error":repr (e ),
                "error_code":-6 ,
                "error_message":"TestRunnerError (Unpacking)"
                }
            else :

                result =[[-1 for _ in range (len (in_outs ["inputs"]))]]
                metadata ={
                "error":repr (e ),
                "error_code":-5 ,
                "error_message":"TestRunnerError"
                }
            if debug :
                print (f"Exception in check_correctness: {e}")
                traceback .print_exc ()
        finally :

            signal .alarm (0 )

    finally :

        signal .signal (signal .SIGALRM ,old_handler )


    if not result :
        in_outs =json .loads (sample ["input_output"])
        result =[[-1 for _ in range (len (in_outs ["inputs"]))]]
        if not metadata :
            metadata ={
            "error":"Unknown error",
            "error_code":-5 ,
            "error_message":"TestRunnerError"
            }

    return result ,metadata 


def evaluate_generations_by_problem (args ):



    debug =False 

    try :
        problem_generations :list [str ]=args [0 ]
        sample =args [1 ]
        debug :bool =args [2 ]
        timeout :int =args [3 ]
        using_closed_models :bool =args [4 ]if len (args )>4 else False 

        res =[]
        metadata =[]
        for o_idx ,o in enumerate (problem_generations ):
            curr_res =[-2 ]
            curr_metadata ={"error":"Unknown error","error_code":-5 }
            try :
                curr_res ,curr_metadata =check_correctness (
                sample ,o ,timeout =timeout ,debug =debug ,using_closed_models =using_closed_models 
                )
                if debug :
                    print (f"\nSuccessful compilation of task {o_idx}!")
                fixed =[]
                for e in curr_res :
                    if isinstance (e ,np .ndarray ):
                        e =e .item (0 )
                    if isinstance (e ,np .bool_ ):
                        e =bool (e )
                    fixed .append (e )
                curr_res =fixed 
                if not np .all (curr_res ):
                    if debug :
                        print (f"Results were not True for all test cases {curr_res=}\n")
            except Exception as e :
                if debug :
                    print (f"Compilation failed, test framework exception = {repr(e)}{e}\n")
                    traceback .print_exc ()


                error_str =str (e )
                if "not enough values to unpack"in error_str or "too many values to unpack"in error_str :
                    curr_metadata ={
                    "error":repr (e ),
                    "error_code":-6 ,
                    "error_message":"TestRunnerError (Unpacking)",
                    }
                else :
                    curr_metadata ={
                    "error":repr (e ),
                    "error_code":-5 ,
                    "error_message":"TestRunnerError",
                    }
            finally :
                assert isinstance (curr_res ,list ),curr_res 
                assert isinstance (curr_metadata ,dict ),curr_metadata 
                res .append (curr_res )
                metadata .append (curr_metadata )

        if debug :
            for i ,r in enumerate (problem_generations ):
                print ("Sample\n")
                print (r )
                print ("\n")
                print ("Result\n")
                print (res [i ])
                print ("*"*30 +"\n\n")

        return res ,metadata 

    except Exception as e :

        if debug :
            print (f"Critical error in evaluate_generations_by_problem: {e}")
            traceback .print_exc ()


        error_str =str (e )
        if "not enough values to unpack"in error_str or "too many values to unpack"in error_str :
            return [[-6 ]],[{"error":str (e ),"error_code":-6 ,"error_message":"CriticalUnpackingError"}]
        else :
            return [[-5 ]],[{"error":str (e ),"error_code":-5 ,"error_message":"CriticalError"}]


def evaluate_generations (
samples_list :list ,
generations_list :list [list [str ]],
debug :bool =False ,
num_process_evaluate :int =16 ,
timeout =6 ,
using_closed_models =False ,
):


    from guf .tasks .livecodebench_pool import get_livecodebench_pool 

    inputs =[
    [(generations_list [index ],samples_list [index ],debug ,timeout ,using_closed_models ),index ]
    for index in range (len (generations_list ))
    ]


    if debug :

        results ={}
        metadata ={}
        for arg ,index in inputs :
            try :

                results [index ],metadata [index ]=evaluate_generations_by_problem (arg )

            except Exception as e :
                print (f"Error evaluating index {index}: {e}")
                error_str =str (e )
                if "not enough values to unpack"in error_str or "too many values to unpack"in error_str :
                    results [index ]=[[-6 ]]
                    metadata [index ]=[{"error":str (e ),"error_code":-6 }]
                else :
                    results [index ]=[[-5 ]]
                    metadata [index ]=[{"error":str (e ),"error_code":-5 }]
    else :

        pool =get_livecodebench_pool (num_process_evaluate )


        futures =[]
        for arg ,index in inputs :


            future =pool .apply_async (evaluate_generations_by_problem ,(arg ,))
            futures .append ((future ,index ))

        results ={}
        metadata ={}
        for future ,index in futures :
            try :
                results [index ],metadata [index ]=future .get (timeout =300 )
            except Exception as e :
                print (f"Error evaluating index {index}: {e}")
                error_str =str (e )
                if "not enough values to unpack"in error_str or "too many values to unpack"in error_str :
                    results [index ]=[[-6 ]]
                    metadata [index ]=[{"error":str (e ),"error_code":-6 }]
                else :
                    results [index ]=[[-5 ]]
                    metadata [index ]=[{"error":str (e ),"error_code":-5 }]


    return results ,metadata 


def codegen_metrics (
samples_list ,
generations_list ,
k_list =[1 ,5 ,10 ,20 ,40 ,50 ,75 ,100 ,125 ,150 ,200 ,500 ,1000 ],
num_process_evaluate =16 ,
timeout =6 ,
debug =False ,
using_closed_models =False ,
):

    try :
        samples_linear =[]
        generations_linear =[]
        remap_index =[]
        results =defaultdict (list )
        metadatas =defaultdict (list )
        for idx ,(sample ,generation_list )in enumerate (
        zip (samples_list ,generations_list )
        ):
            assert isinstance (generation_list ,list ),generations_list [0 ]
            for generation in generation_list :
                assert isinstance (generation ,str ),generations_list [0 ]
                samples_linear .append (sample )
                generations_linear .append ([generation ])
                remap_index .append (idx )



        results_linear ,metadatas_linear =evaluate_generations (
        samples_linear ,
        generations_linear ,
        debug =debug ,
        num_process_evaluate =num_process_evaluate ,
        timeout =timeout ,
        using_closed_models =using_closed_models ,
        )

        for idx ,sub_results in sorted (results_linear .items (),key =lambda x :x [0 ]):
            results [remap_index [idx ]].append (sub_results [0 ])

        for idx ,sub_metadatas in sorted (metadatas_linear .items (),key =lambda x :x [0 ]):
            metadatas [remap_index [idx ]].append (sub_metadatas [0 ])

        metrics =compute_metrics_from_results (results ,k_list =k_list )

        final_metadata =[]
        for key in sorted (list (metadatas .keys ())):
            final_metadata .append (metadatas [key ])
        for i in range (len (final_metadata )):
            if type (final_metadata [i ])is not list :
                final_metadata [i ]=[json .dumps (final_metadata [i ])]
            else :
                final_metadata [i ]=[json .dumps (x )for x in final_metadata [i ]]

            assert len (final_metadata [i ])==len (
            generations_list [0 ]
            ),f"{len(final_metadata[i])=}"

        return [metrics ,results ,final_metadata ]

    except Exception as e :
        if debug :
            print (f"Error in codegen_metrics: {e}")
            traceback .print_exc ()


        fallback_metrics ={"pass@1":0.0 }
        fallback_results =defaultdict (list )
        fallback_metadata =[]

        return [fallback_metrics ,fallback_results ,fallback_metadata ]


if __name__ =="__main__":
    print (
    check_correctness (
    {
    "input_output":json .dumps (
    {
    "inputs":")))))",
    "outputs":"0",
    },
    )
    },
    "\nMOD = 998244353\n\nS = input().strip()\nn = len(S)\n\nif n % 2 != 0:\n    print(0)\n    exit()\n\n# Initialize DP table\ndp = [[0] * (n + 2) for _ in range(n + 1)]\ndp[0][0] = 1\n\nfor i in range(1, n + 1):\n    c = S[i-1]\n    for b in range(n + 1):\n        if dp[i-1][b] == 0:\n            continue\n        if c == '(':\n            new_b = b + 1\n            if new_b <= n:\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n        elif c == ')':\n            if b > 0:\n                new_b = b - 1\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n        else:  # '?'\n            # Replace with '('\n            new_b = b + 1\n            if new_b <= n:\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n            # Replace with ')'\n            if b > 0:\n                new_b = b - 1\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n\nprint(dp[n][0] % MOD)\n",
    6 ,
    debug =True ,
    )
    )