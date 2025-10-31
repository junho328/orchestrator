import os 
import time 
from typing import List ,Dict ,Optional 
from openai import AsyncOpenAI 

def _flatten_messages_to_input (messages :List [Dict [str ,str ]]):

    instructions =None 
    lines =[]
    for m in messages :
        role =m .get ("role","user")
        content =m .get ("content","")
        if role =="system"and instructions is None :
            instructions =content 
        else :
            lines .append (f"{role}: {content}")
    input_text ="\n".join (lines ).strip ()
    return instructions ,input_text 

async def query_oai_responses (
model_name :str ,
messages :List [Dict [str ,str ]],
max_tokens :Optional [int ],
temperature :float ,
reasoning_effort :str ="minimal",
verbosity :str ="medium",
)->str :

    client =AsyncOpenAI (api_key =os .getenv ("OPENAI_API_KEY"),max_retries =50 ,timeout =3000 )


    instructions ,input_text =_flatten_messages_to_input (messages )


    if model_name =="gpt-5":

        if reasoning_effort =="high":
            assert max_tokens is not None and max_tokens >=128000 ,(
            "max_tokens must be at least 128000 for high reasoning effort"
            )
        if reasoning_effort =="medium":

            max_tokens =4096 


        api_params ={
        "model":model_name ,
        "input":input_text ,
        "max_output_tokens":max_tokens ,
        "temperature":1 ,
        "reasoning":{"effort":reasoning_effort },
        "text":{"verbosity":verbosity },
        }
        if instructions :
            api_params ["instructions"]=instructions 

    else :

        api_params ={
        "model":model_name ,
        "input":input_text ,
        "max_output_tokens":max_tokens ,
        "temperature":temperature ,
        }
        if instructions :
            api_params ["instructions"]=instructions 


    for _ in range (20 ):
        resp =await client .responses .create (**api_params )

        out =getattr (resp ,"output_text",None )
        if out :
            return out .strip ()
        time .sleep (1 )

    print ("exceeded inner attempts to obtain response from oai, returning empty string")
    return ""
