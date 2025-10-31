from transformers import PreTrainedTokenizerBase 
from typing import Dict ,Any 
from .recursion_formats import (
SIMPLE_recursion_FORMAT_V0 ,
SIMPLE_recursion_FORMAT_V1 ,
SIMPLE_recursion_FORMAT_V0_NO_EMPTY ,
SIMPLE_recursion_FORMAT_V1_NO_EMPTY ,
SIMPLE_recursion_FORMAT_V2 ,
SIMPLE_recursion_FORMAT_V1_1 ,
SIMPLE_recursion_FORMAT_V1_1_noempty ,
)


recursion_FORMAT_LIBRARY ={
"v0":SIMPLE_recursion_FORMAT_V0 ,
"v1":SIMPLE_recursion_FORMAT_V1 ,
"v0_no_empty":SIMPLE_recursion_FORMAT_V0_NO_EMPTY ,
"v1_no_empty":SIMPLE_recursion_FORMAT_V1_NO_EMPTY ,
"v1_1":SIMPLE_recursion_FORMAT_V1_1 ,
"v1_1_noempty":SIMPLE_recursion_FORMAT_V1_1_noempty ,
"v2":SIMPLE_recursion_FORMAT_V2 ,
}

def construct_previous_workflow (prev_subtasks ,prev_agent_responses ,prev_model_ids ):
    previous_workflow =""
    for subtask ,agent_response ,model_id in zip (prev_subtasks ,prev_agent_responses ,prev_model_ids ):
        previous_workflow +=f"\n<Previous round subtask assigned to Agent {model_id}> {subtask} </Previous round subtask assigned to Agent {model_id}>\n"
        previous_workflow +=f"\n<Previous round Agent {model_id} response>{agent_response} </Previous round Agent {model_id} response>\n"
    return previous_workflow 


def make_recursion_round_processor (
tokenizer :PreTrainedTokenizerBase ,
recursion_question_format :str ,
max_worker_response_length :int =None ,
max_number_of_routing_steps :int =5 ,
):
    assert recursion_question_format in recursion_FORMAT_LIBRARY ,(
    f"recursion question format {recursion_question_format} not found in the library."
    )

    show_previous_workflow =False 
    if recursion_question_format in ["v2"]:
        show_previous_workflow =True 

    recursion_question_format =recursion_FORMAT_LIBRARY [
    recursion_question_format ]

    def recursion_round_processor (
    input_sample :Dict [str ,Any ],
    prompt_sample :str ,
    completion_sample :str ,
    final_worker_response :str ,
    **meta :Any 
    )->str :

        if max_worker_response_length is not None :
            tokenized_final_response =tokenizer (
            final_worker_response ,return_tensors ='pt')['input_ids'][0 ]
            if tokenized_final_response .shape [0 ]>max_worker_response_length :
                tokenized_final_response =tokenized_final_response [:max_worker_response_length ]
            cropped_final_worker_response =tokenizer .decode (
            tokenized_final_response ,skip_special_tokens =True )
            assert final_worker_response .endswith (
            cropped_final_worker_response ),(
            "Final worker response was not cropped correctly."
            )
            final_worker_response =cropped_final_worker_response 

        history =meta .get ("history",{})or {}
        if show_previous_workflow :
            prev_subtasks =history .get ("subtasks",[])or []
            prev_agent_responses =history .get ("agent_responses",[])or []
            prev_model_ids =history .get ("model_ids",[])or []
            previous_workflow =construct_previous_workflow (prev_subtasks ,prev_agent_responses ,prev_model_ids )
        else :
            previous_workflow ="Could not obtain previous workflow"

        answer =completion_sample 


        new_question =recursion_question_format .format (previous_workflow =previous_workflow ,worker_response =final_worker_response ,
        max_number_of_routing_steps =max_number_of_routing_steps ,
        )



        other_messages =[
        {"role":'assistant',"content":answer },
        {"role":'user',"content":new_question },
        ]

        other_conductor_prompt =tokenizer .apply_chat_template (
        conversation =other_messages ,tokenize =False ,continue_final_message =True ,
        )

        pos =other_conductor_prompt .find (answer )
        if pos ==-1 :
            raise ValueError ('answer not found in other_conductor_prompt')


        suffix =other_conductor_prompt [pos +len (answer ):]


        suffix =suffix .lstrip ()

        return answer +suffix 
    return recursion_round_processor 
