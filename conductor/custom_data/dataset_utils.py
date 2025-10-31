from .routing_question_formats import *
import importlib 
import os 
import sys 
import re 

PROMPT_FORMAT_LIBRARY ={
"v1_7c2_2":ROUTING_QUESTION_FORMAT_V1_7c2_2 ,
"v1_7c2_2_ood":ROUTING_QUESTION_FORMAT_V1_7c2_2_ood ,
}

def load_task (task_module ,class_name ,available_models ,seed =42 ,evaluate_only =False ,finetune =False ):


        submodule_package_root =os .path .abspath (os .path .join (os .path .dirname (__file__ ),'..','proj_guf'))
        if submodule_package_root not in sys .path :
            sys .path .insert (0 ,submodule_package_root )
        try :
            target_mod =importlib .import_module (task_module )
        finally :

            if sys .path [0 ]==submodule_package_root :
                sys .path .pop (0 )
        print (f"Importing task from {target_mod.__name__}.{class_name}")
        if class_name =="LiveCodeBenchTask":
            return getattr (target_mod ,class_name )(llm_names =available_models ,seed =seed ,evaluate_only =bool (evaluate_only ))
        elif class_name =="MixMMRLTask":
            return getattr (target_mod ,class_name )(llm_names =available_models ,seed =seed ,finetune =bool (finetune ))
        else :
            return getattr (target_mod ,class_name )(llm_names =available_models ,seed =seed )


def _extract_model_size (model_name :str )->str :

    _SIZE_RE =re .compile (r"(\d+)(?=\s*[bB])")
    matches =_SIZE_RE .findall (model_name )
    return matches [-1 ]if matches else "unknown"

def format_models_list (available_models ,hide_names =True ,hide_parameters =True )->str :
    available_models_strings =[]
    for i ,model in enumerate (available_models ):
        if hide_names and not hide_parameters :
            size =_extract_model_size (model )
            available_model_str =f"Model id {i}: Model size: {size}B parameters"
        elif hide_names and hide_parameters :
            available_model_str =f"Model id {i}"
        else :
            available_model_str =f"Model id {i}: Organization/Name: {model}"
        available_models_strings .append (available_model_str )
    return "\n".join (available_models_strings )