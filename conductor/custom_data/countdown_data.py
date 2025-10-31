import os 
import random 
import re 
from datasets import load_dataset 
import typing as tp 
from typing import List ,Optional ,Tuple 
import numpy as np 
import ast 



COUNTDOWN_STYLES_AND_LINES_FORMATS =dict (
reproduce =dict (
system_prompt ="You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.",
line_format ="Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. Think step by step inside <think> tags.",
),
fixed =dict (
system_prompt ="You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.",
line_format ="Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. Show your work in the <think> </think> tags and return the final equation in the <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. Think step by step inside <think> tags.",
),
r1 =dict (
system_prompt ="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.",
line_format ="Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. Return the final equation in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. Think step by step inside <think> tags.",
),
)


def make_datasets (
tokenizer ,dataset_id_or_path ,seed ,data_limit =50000 ,style ='reproduce',**kwargs ):
    assert style in ['reproduce','fixed','r1']
    dataset =load_dataset (dataset_id_or_path ,split ="train")
    dataset =dataset .shuffle (seed =seed ).select (range (data_limit ))

    if style not in COUNTDOWN_STYLES_AND_LINES_FORMATS :
        raise NotImplementedError 
    style_dict =COUNTDOWN_STYLES_AND_LINES_FORMATS [style ]

    system_prompt =style_dict ["system_prompt"]
    line_format =style_dict ["line_format"]

    def generate_prompt (numbers ,target ,tokenizer ,**kwargs ):
        r1_prefix =[{
        "role":"system",
        "content":system_prompt ,
        },
        {
        "role":"user",
        "content":line_format .format (numbers =numbers ,target =target )
        },
        {
        "role":"assistant",
        "content":"Let me solve this step by step.\n<think>"
        }]
        return {"prompt":tokenizer .apply_chat_template (
        r1_prefix ,tokenize =False ,continue_final_message =True ),
        "target":target ,"nums":numbers }

    dataset =dataset .map (lambda x :generate_prompt (
    x ["nums"],x ["target"],tokenizer ))
    train_test_split =dataset .train_test_split (test_size =0.1 )
    train_dataset ,test_dataset =train_test_split ["train"],train_test_split ["test"]
    return dict (
    train_dataset =train_dataset ,
    eval_dataset =test_dataset ,
    )


def equation_reward_func_countdown (
completions :List [str ],
target :List [float ],
nums :List [List [int ]],
**kwargs 
)->List [float ]:
    rewards =[]
    eps =1e-6 

    for comp ,gt ,numbers in zip (completions ,target ,nums ):
        try :
            text =comp 



            matched =False 
            for m in re .finditer (r'[\d+\-*/().\s]{3,}',text ):
                eq =m .group ().strip ()
                found =[int (x )for x in re .findall (r'\d+',eq )]
                if sorted (found )!=sorted (numbers ):
                    continue 

                if not re .fullmatch (r'[\d+\-*/().\s]+',eq ):
                    continue 
                res =eval (eq ,{"__builtins__":None },{})
                if abs (res -gt )<eps :
                    rewards .append (1.0 )
                    matched =True 
                    break 
            if not matched :
                rewards .append (0.0 )
        except Exception :
            rewards .append (0.0 )

    return rewards 


def make_reward_functions (
output_dir =None ,
logging_probability =0.1 ,
include_format_reward =True ,
start_think_tag ="<think>",
end_think_tag ="</think>",
start_solution_tag ="<answer>",
end_solution_tag ="</answer>",
):
    def format_func_eval (completions ,target ,**kwargs ):
        rewards =[]
        for comp in completions :
            try :
                tagged =start_think_tag +comp 
                if output_dir is not None :
                    if random .random ()<logging_probability :
                        log_dir =os .path .join (
                        output_dir ,'completion_samples'
                        )
                        os .makedirs (log_dir ,exist_ok =True )
                        with open (os .path .join (
                        log_dir ,'completion_samples.txt'
                        ),'a')as f :
                            f .write ("\n\n==============\n"+tagged )

                pattern =(
                rf"^{re.escape(start_think_tag)}"
                r"([\s\S]*?)"
                rf"{re.escape(end_think_tag)}\n"
                rf"{re.escape(start_solution_tag)}"
                r"([\s\S]*?)"
                rf"{re.escape(end_solution_tag)}$"
                )
                match =re .search (pattern ,tagged ,re .DOTALL )
                rewards .append (
                1.0 if match and len (match .groups ())==2 else 0.0 )
            except Exception :
                rewards .append (0.0 )
        return rewards 

    if include_format_reward :
        return [format_func_eval ,equation_reward_func_countdown ]
    else :
        return [equation_reward_func_countdown ]


def remove_redundant_brackets (expr :str )->str :
    tree =ast .parse (expr ,mode ='eval')
    return ast .unparse (tree )


def find_equation (numbers :List [float ],target :float ,)->Optional [str ]:

    eps =1e-6 
    states =[(np .array (numbers ,float ),[str (n )for n in numbers ])]
    while states :
        new_states =[]
        for vals ,exprs in states :
            m =vals .size 

            if m ==1 :
                if abs (vals [0 ]-target )<eps :
                    return remove_redundant_brackets (exprs [0 ])
                continue 

            for i in range (m ):
                for j in range (i +1 ,m ):
                    mask =[k for k in range (m )if k not in (i ,j )]
                    rest_vals =vals [mask ]
                    rest_exprs =[exprs [k ]for k in mask ]
                    a ,b =vals [i ],vals [j ]
                    ea ,eb =exprs [i ],exprs [j ]

                    ops =[
                    (a +b ,f"({ea}+{eb})"),
                    (a -b ,f"({ea}-{eb})"),
                    (b -a ,f"({eb}-{ea})"),
                    (a *b ,f"({ea}*{eb})"),
                    ]
                    if abs (b )>eps :
                        ops .append ((a /b ,f"({ea}/{eb})"))
                    if abs (a )>eps :
                        ops .append ((b /a ,f"({eb}/{ea})"))
                    for val ,ex in ops :
                        new_vals =np .concatenate (
                        (rest_vals ,np .array ([val ])))
                        new_exprs =rest_exprs +[ex ]
                        new_states .append ((new_vals ,new_exprs ))
        states =new_states 
    return None 


def add_solutions (ds ):
    if "solution"not in ds .column_names :
        ds =ds .map (lambda x :{"solution":find_equation (
        numbers =x ['nums'],target =x ['target'])})
    return ds 
