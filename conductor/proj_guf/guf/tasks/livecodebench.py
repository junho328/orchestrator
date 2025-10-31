import json 
import zlib 
import pickle 
import base64 
from enum import Enum 
from datetime import datetime 
import dataclasses 
from dataclasses import dataclass 

from guf .tasks .livecodebench_utils import codegen_metrics 

from typing import Dict ,List ,Any ,Optional 
from datasets import load_dataset 
import numpy as np 
from guf .core import Task 
import os 



def extract_code (model_output :str ):

    if not model_output :
        return ""


    content =model_output .replace ('```python','').replace ('```','').strip ()

    if not content :
        return ""


    content_lower =content .lower ()


    dangerous_os_functions =[
    'os.kill','os.system','os.putenv','os.remove','os.removedirs',
    'os.rmdir','os.fchdir','os.setuid','os.fork','os.forkpty',
    'os.killpg','os.rename','os.renames','os.truncate','os.replace',
    'os.unlink','os.fchmod','os.fchown','os.chmod','os.chown',
    'os.chroot','os.lchflags','os.lchmod','os.lchown','os.getcwd',
    'os.chdir'
    ]

    for func in dangerous_os_functions :
        if func in content_lower :
            return f"CONTENT_REMOVED_DUE_TO_DANGEROUS_OS_FUNCTION: {func}"


    dangerous_shutil_functions =['shutil.rmtree','shutil.move','shutil.chown']
    for func in dangerous_shutil_functions :
        if func in content_lower :
            return f"CONTENT_REMOVED_DUE_TO_DANGEROUS_SHUTIL_FUNCTION: {func}"


    if 'subprocess.popen'in content_lower :
        return "CONTENT_REMOVED_DUE_TO_SUBPROCESS_OPERATIONS: subprocess.popen"
    if 'subprocess.call'in content_lower :
        return "CONTENT_REMOVED_DUE_TO_SUBPROCESS_OPERATIONS: subprocess.call"
    if 'subprocess.run'in content_lower :
        return "CONTENT_REMOVED_DUE_TO_SUBPROCESS_OPERATIONS: subprocess.run"


    dangerous_builtins =['quit()','exit()','eval(','exec(','compile(']
    for builtin in dangerous_builtins :
        if builtin in content_lower :
            return f"CONTENT_REMOVED_DUE_TO_DANGEROUS_BUILTIN: {builtin}"


    dangerous_modules =[
    'import os','from os import','import subprocess','from subprocess import',
    'import shutil','from shutil import','import resource','from resource import',
    'import psutil','from psutil import','import tkinter','from tkinter import',
    'import joblib','from joblib import','import ipdb','from ipdb import'
    ]

    for module in dangerous_modules :
        if module in content_lower :
            return f"CONTENT_REMOVED_DUE_TO_DANGEROUS_MODULE_IMPORT: {module}"


    dangerous_file_ops =['open(']
    if any (op in content_lower for op in dangerous_file_ops ):

        if 'open('in content_lower :

            lines =content .split ('\n')
            for line in lines :
                if 'open('in line .lower ()and 'sys.stdin'not in line .lower ():

                    if not any (safe_pattern in line .lower ()for safe_pattern in ['"r"',"'r'",'mode="r"',"mode='r'"]):
                        if not line .strip ().startswith ('#'):
                            return f"CONTENT_REMOVED_DUE_TO_DANGEROUS_FILE_OPERATION: {line.strip()}"


    dangerous_network =['socket.','urllib.','requests.','http.','ftplib.','smtplib.']
    for net_op in dangerous_network :
        if net_op in content_lower :
            return f"CONTENT_REMOVED_DUE_TO_NETWORK_OPERATION: {net_op}"


    if '__import__'in content_lower :
        return "CONTENT_REMOVED_DUE_TO_DYNAMIC_IMPORTS: __import__"
    if 'globals('in content_lower :
        return "CONTENT_REMOVED_DUE_TO_GLOBALS_ACCESS: globals()"
    if 'locals('in content_lower :
        return "CONTENT_REMOVED_DUE_TO_LOCALS_ACCESS: locals()"
    if 'vars('in content_lower :
        return "CONTENT_REMOVED_DUE_TO_VARS_ACCESS: vars()"
    if 'dir('in content_lower :
        return "CONTENT_REMOVED_DUE_TO_INTROSPECTION: dir()"


    if 'exec('in content_lower :
        return "CONTENT_REMOVED_DUE_TO_CODE_EXECUTION: exec()"
    if 'eval('in content_lower :
        return "CONTENT_REMOVED_DUE_TO_CODE_EVALUATION: eval()"

    return content 

class Platform (Enum ):
    LEETCODE ="leetcode"
    CODEFORCES ="codeforces"
    ATCODER ="atcoder"


class Difficulty (Enum ):
    EASY ="easy"
    MEDIUM ="medium"
    HARD ="hard"


class TestType (Enum ):
    STDIN ="stdin"
    FUNCTIONAL ="functional"


@dataclass 
class Test :
    input :str 
    output :str 
    testtype :TestType 

    def __post_init__ (self ):
        self .testtype =TestType (self .testtype )


@dataclass 
class CodeGenerationProblem :
    question_title :str 
    question_content :str 
    platform :Platform 
    question_id :str 
    contest_id :str 
    contest_date :datetime 
    starter_code :str 
    difficulty :Difficulty 
    public_test_cases :list [Test ]
    private_test_cases :list [Test ]
    metadata :dict 

    def __post_init__ (self ):
        self .platform =Platform (self .platform )
        self .difficulty =Difficulty (self .difficulty )
        self .contest_date =datetime .fromisoformat (self .contest_date )

        self .public_test_cases =json .loads (self .public_test_cases )
        self .public_test_cases =[Test (**t )for t in self .public_test_cases ]

        try :
            self .private_test_cases =json .loads (self .private_test_cases )
        except :
            self .private_test_cases =json .loads (
            pickle .loads (
            zlib .decompress (
            base64 .b64decode (self .private_test_cases .encode ("utf-8"))
            )
            )
            )
        self .private_test_cases =[Test (**t )for t in self .private_test_cases ]

        self .metadata =json .loads (self .metadata )

    def insert_output (self ,output_list :list [str ],code_list :list [str ])->dict :
        return {
        "question_title":self .question_title ,
        "question_content":self .question_content ,
        "platform":self .platform .value ,
        "question_id":self .question_id ,
        "contest_id":self .contest_id ,
        "contest_date":self .contest_date .isoformat (),
        "starter_code":self .starter_code ,
        "difficulty":self .difficulty .value ,
        "output_list":output_list ,
        "code_list":code_list ,
        }

    def insert_output_evaluation (
    self ,
    output_list :list [str ],
    code_list :list [str ],
    graded_list :list [bool ],
    **kwargs ,
    )->dict :
        output =self .insert_output (output_list ,code_list )
        output ["graded_list"]=graded_list 
        output ["pass@1"]=graded_list .count (True )/len (graded_list )
        for k ,v in kwargs .items ():
            output [k ]=v 
        return output 

    def get_evaluation_sample (self ):
        return {
        "input_output":json .dumps (
        {
        "inputs":[
        t .input 
        for t in self .public_test_cases +self .private_test_cases 
        ],
        "outputs":[
        t .output 
        for t in self .public_test_cases +self .private_test_cases 
        ],
        "fn_name":self .metadata .get ("func_name",None ),
        }
        ),
        }


class LiveCodeBenchTask (Task ):



    STDIN_PROMPT_TEMPLATE =(
    "Question:\n{question_content}\n\n"
    "Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). "
    "Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n\n"
    "Show your work in <THINKING_TAG> </THINKING_TAG> tags. "
    "And return ONLY the final Python code in <answer> </answer> tags, with NO markdown formatting or backticks.\n\n"
    "Also, do not include anything other than Python code between the tags.\n\n"
    "{starter_code}"
    "FORBIDDEN (do NOT do this):\n"
    "<answer>\n"
    "```python\n"
    "n = int(input())\n"
    "a = list(map(int, input().split()))\n"
    "print(sum(a))\n"
    "```\n"
    "</answer>\n\n"
    "FORBIDDEN (do NOT do this):\n"
    "<answer>\n"
    "Here is the best solution ... blablabla (content that is not Python code)\n"
    "n = int(input())\n"
    "a = list(map(int, input().split()))\n"
    "print(sum(a))\n"
    "</answer>\n\n"
    "CORRECT (DO THIS):\n"
    "<answer>\n"
    "n = int(input())\n"
    "a = list(map(int, input().split()))\n"
    "print(sum(a))\n"
    "</answer>\n\n"
    "Think step by step inside <THINKING_TAG> tags, but keep it short."
    )

    FUNCTION_PROMPT_TEMPLATE =(
    "Question:" "\n{question_content}\n\n"
    "Format: Implement the function `{function_name}` that solves this problem. "
    "Your solution should be a complete Python function definition wrapped in a Solution class.\n\n"
    "CRITICAL: You MUST use the EXACT function name '{function_name}' - do not change, modify, or rename it in any way.\n\n"
    "Show your work in <THINKING_TAG> </THINKING_TAG> tags. "
    "And return ONLY the final Python code in <answer> </answer> tags, with NO markdown formatting or backticks.\n\n"
    "Your code should follow this structure:\n"
    "- Import any necessary modules at the top\n"
    "- Define a Solution class\n"
    "- Implement the {function_name} method inside the Solution class with the EXACT name '{function_name}'\n"
    "- Do NOT include test cases or example usage\n"
    "- Do NOT change the function name to snake_case, camelCase, or any other format\n\n"
    "{starter_code}"
    "FORBIDDEN (do NOT do this):\n"
    "<answer>\n"
    "```python\n"
    "class Solution:\n"
    "    def {function_name}(...):\n"
    "        ..."
    "```\n"
    "</answer>\n\n"
    "FORBIDDEN (do NOT do this):\n"
    "<answer>\n"
    "Here is the best solution ... blablabla (content that is not Python code)\n"
    "class Solution:\n"
    "    def {function_name}(...):\n"
    "        ..."
    "</answer>\n\n"
    "CORRECT (DO THIS):\n"
    "<answer>\n"
    "class Solution:\n"
    "    def {function_name}(...):\n"
    "        ..."
    "</answer>\n\n"
    "Think step by step inside <THINKING_TAG> tags, but keep it short."
    )

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
    use_consultant :bool =True ,
    log_dir :Optional [str ]=None ,
    evaluate_only :bool =False ,
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

        self .evaluate_only =evaluate_only 

    def _get_user_prompt_template (self )->str :

        return self .STDIN_PROMPT_TEMPLATE .replace ("THINKING_TAG",self .thinking_tag )

    def _format_base_prompt (self ,task_data :Dict )->str :


        if task_data .get ("starter_code"):
            starter_code ="Here is the starter code for this question:\n\n"+task_data ["starter_code"]+"\n\n"
        else :
            starter_code =""


        try :
            if 'metadata'in task_data :
                metadata =json .loads (task_data ['metadata'])
                fn_name =metadata .get ('func_name')

                if fn_name :

                    template =self .FUNCTION_PROMPT_TEMPLATE .replace ("THINKING_TAG",self .thinking_tag )
                    return template .format (
                    question_content =task_data ["question_content"],
                    function_name =fn_name ,
                    starter_code =starter_code 
                    )
        except (json .JSONDecodeError ,KeyError )as e :

            if self .debug :
                print (f"Warning: Could not parse metadata, using stdin format: {e}")


        template =self .STDIN_PROMPT_TEMPLATE .replace ("THINKING_TAG",self .thinking_tag )
        return template .format (
        question_content =task_data ["question_content"],
        starter_code =starter_code 
        )
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

        if self .evaluate_only :

            ds =load_dataset ("livecodebench/code_generation_lite",split ="test",version_tag ="v6")
            valid_ratio =0.0 
            test_ratio =1.0 
        else :

            ds =load_dataset ("livecodebench/code_generation_lite",split ="test",version_tag ="release_v1")


        from guf .utils import get_or_create_indices 

        split_indices =get_or_create_indices (
        task_name ="livecodebenchv6",
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


        for split_name in data_splits :

            data_splits [split_name ]=data_splits [split_name ].add_column (
            "task_type",
            ["livecodebench"]*len (data_splits [split_name ])
            )


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

        if self .evaluate_only :

            data_splits ["train"]=data_splits ["test"]

        return data_splits 

    def _calculate_reward (self ,completions :List [str ],task_data :List [Dict ],debug :bool =False )->List [float ]:

        rewards =[]


        using_closed_models =os .environ .get ("USING_CLOSED_MODELS","0")=="1"
        for completion ,data in zip (completions ,task_data ):
            try :


                expected_fields ={field .name for field in dataclasses .fields (CodeGenerationProblem )}


                filtered_data ={k :v for k ,v in data .items ()if k in expected_fields }
                task_data_formatted =[CodeGenerationProblem (**filtered_data ).get_evaluation_sample ()]


                from guf .utils import extract_answer 
                answer_content =extract_answer (completion )

                if answer_content is None :

                    if debug :
                        print (f"WARNING: No code found in <answer> tags!")
                        print (f"completion: {completion[:500]}...")
                    rewards .append (0.0 )
                    continue 


                task_completion =[[extract_code (answer_content )]]

                if not task_completion [0 ][0 ]:

                    if debug :
                        print (f"WARNING: No code found within backticks in answer!")
                    rewards .append (0.0 )
                    continue 


                try :
                    codegen_return =codegen_metrics (
                    task_data_formatted ,
                    task_completion ,
                    debug =self .debug ,
                    using_closed_models =using_closed_models 
                    )


                    if not isinstance (codegen_return ,list )or len (codegen_return )<3 :

                        if debug :
                            print (f"WARNING: codegen_metrics returned invalid result: {codegen_return}")
                        rewards .append (0.0 )
                        continue 

                    metrics ,results ,final_metadata =codegen_return [0 ],codegen_return [1 ],codegen_return [2 ]


                    if not isinstance (metrics ,dict )or 'pass@1'not in metrics :

                        if debug :
                            print (f"WARNING: Invalid metrics structure: {metrics}")
                        rewards .append (0.0 )
                        continue 

                    reward =metrics ['pass@1']


                    if not isinstance (reward ,(int ,float ))or reward <0 or reward >1 :
                        if debug :
                            print (f"WARNING: Invalid reward value: {reward}")
                        rewards .append (0.0 )
                        continue 

                    rewards .append (reward )

                    if debug :
                        print (f"Extracted code: {task_completion[0][0]}")
                        print (f"result: {results}")
                        print (f"final_metadata: {final_metadata}")
                        print (f"reward: {reward}")

                except Exception as e :
                    if debug :
                        print (f"ERROR in codegen_metrics: {e}")
                        import traceback 
                        traceback .print_exc ()
                    rewards .append (0.0 )
                    continue 

            except Exception as e :
                if debug :
                    print (f"ERROR processing completion: {e}")
                    import traceback 
                    traceback .print_exc ()
                rewards .append (0.0 )
                continue 

        if debug :
            print (f"Final rewards: {rewards}")

        return rewards 

if __name__ =="__main__":

    llms =["claude-3-7-sonnet-20250219","gpt-4o-mini"]
    task =LiveCodeBenchTask (llm_names =llms ,max_turns =3 ,max_tokens =4096 ,debug =True )
    obs =task .reset (split ="test",task_id =2 )
    done =False 

    while not done :

        action =np .array ([0 ,1 ])
        obs ,reward ,done ,_ =task .step (action )

    print ("Final response:",task .response )
    print ("Reward:",reward )