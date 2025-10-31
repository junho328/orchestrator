import os 
import requests 
import json 
from typing import List ,Dict 
from openai import OpenAI 
from anthropic import Anthropic 
from google import genai 
from google .genai import types 


def query_oai (
model :str ,messages :List [Dict ],max_tokens :int ,temperature :float ,**kwargs 
)->str :
    client =OpenAI (api_key =os .getenv ("OPENAI_API_KEY"),max_retries =15 )

    api_params ={
    "model":model ,
    "messages":messages ,
    "max_tokens":max_tokens ,
    "temperature":temperature ,
    "stream":False ,
    }
    api_params .update (kwargs )

    response =client .chat .completions .create (**api_params )
    return response .choices [0 ].message .content 


def query_anthropic (
model :str ,messages :List [Dict ],max_tokens :int ,temperature :float 
)->str :
    client =Anthropic (api_key =os .getenv ("ANTHROPIC_API_KEY"),max_retries =15 )


    system_message =None 
    filtered_messages =[]
    for message in messages :
        if message ["role"]=="system":
            system_message =message ["content"]
        else :
            filtered_messages .append (message )

    kparams ={
    "model":model ,
    "max_tokens":max_tokens ,
    "temperature":temperature ,
    "messages":filtered_messages ,
    "stream":False ,
    }
    if system_message is not None :
        kparams ["system"]=system_message 

    response =client .messages .create (**kparams )
    return response .content [0 ].text 


def query_gemini (
model :str ,messages :List [Dict ],max_tokens :int ,temperature :float 
)->str :

    import os 
    from google import genai 
    from google .genai import types 


    api_key =os .getenv ("GEMINI_API_KEY")
    if not api_key :
        print ("ERROR: GEMINI_API_KEY environment variable is not set")
        return ""

    try :
        client =genai .Client (api_key =api_key )


        system_message =None 
        content_parts =[]

        for message in messages :
            if message ["role"]=="system":
                system_message =message ["content"]
            else :
                content_parts .append (message ["content"])


        contents ="\n".join (content_parts )


        thinking_config =None 
        adjusted_max_tokens =max_tokens 

        if "2.5-pro"in model :


            thinking_config =types .ThinkingConfig (
            thinking_budget =1024 ,
            include_thoughts =False 
            )

            adjusted_max_tokens =max_tokens +1024 

        elif "2.5-flash"in model :

            thinking_config =types .ThinkingConfig (
            thinking_budget =0 ,
            include_thoughts =False 
            )
            adjusted_max_tokens =max_tokens 


        generation_config =types .GenerateContentConfig (
        max_output_tokens =adjusted_max_tokens ,
        temperature =temperature ,
        system_instruction =system_message ,
        )


        if thinking_config :
            generation_config .thinking_config =thinking_config 


        model_names_to_try =[
        model ,
        "gemini-2.5-pro",
        "gemini-1.5-pro",
        "gemini-1.5-flash"
        ]

        for model_name in model_names_to_try :
            try :
                response =client .models .generate_content (
                model =model_name ,
                contents =contents ,
                config =generation_config ,
                )


                if hasattr (response ,'prompt_feedback')and response .prompt_feedback :
                    if hasattr (response .prompt_feedback ,'block_reason'):
                        print (f"WARNING: Request blocked by safety filter: {response.prompt_feedback.block_reason}")
                        continue 


                if hasattr (response ,'candidates')and response .candidates :
                    for candidate in response .candidates :
                        if hasattr (candidate ,'finish_reason'):
                            if candidate .finish_reason .name =='MAX_TOKENS':
                                print (f"WARNING: Response truncated due to max tokens limit with {model_name}")

                                if adjusted_max_tokens <4096 :
                                    generation_config .max_output_tokens =4096 
                                    response =client .models .generate_content (
                                    model =model_name ,
                                    contents =contents ,
                                    config =generation_config ,
                                    )
                                    break 


                if hasattr (response ,'text')and response .text :
                    return response .text 


                if hasattr (response ,'candidates')and response .candidates :
                    for candidate in response .candidates :
                        if hasattr (candidate ,'content')and candidate .content :
                            if hasattr (candidate .content ,'parts')and candidate .content .parts :
                                for part in candidate .content .parts :
                                    if hasattr (part ,'text')and part .text :
                                        return part .text 

            except Exception as e :
                print (f"DEBUG: Failed with {model_name}: {e}")
                continue 

        print ("DEBUG: All model names failed")
        return ""

    except Exception as e :
        print (f"ERROR: Gemini API call failed: {e}")
        return ""

def query_deepseek (
model :str ,messages :List [Dict ],max_tokens :int ,temperature :float ,use_together :bool =True 
)->str :


    if use_together :

        client =OpenAI (
        api_key =os .getenv ("TOGETHER_API_KEY"),
        base_url ="https://api.together.xyz/v1",
        )
    else :

        client =OpenAI (
        api_key =os .getenv ("DEEPSEEK_API_KEY"),
        base_url ="https://api.deepseek.com",
        )

    response =client .chat .completions .create (
    model =model ,
    messages =messages ,
    max_tokens =max_tokens ,
    temperature =temperature ,
    stream =False ,
    )
    return response .choices [0 ].message .content 


def _extract_text_from_vllm_message (msg )->str :


    if msg is not None and hasattr (msg ,"content")and msg .content is not None :
        return msg .content 


    try :
        tc =msg .tool_calls 
        if tc and tc [0 ].function and tc [0 ].function .arguments :
            maybe =tc [0 ].function .arguments .get ("content")
            if isinstance (maybe ,str )and maybe .strip ():
                return maybe .strip ()
    except AttributeError :
        pass 


    for attr in ("thoughts","content","text"):
        maybe =getattr (msg ,attr ,None )
        if isinstance (maybe ,str )and maybe .strip ():
            return maybe .strip ()


    extra =getattr (msg ,"extra",None )
    if isinstance (extra ,dict ):
        for attr in ("thoughts","content","text"):
            maybe =extra .get (attr )
            if isinstance (maybe ,str )and maybe .strip ():
                return maybe .strip ()

    return ""


def query_locally_hosted_model (
model :str ,
messages :List [Dict ],
max_tokens :int ,
temperature :float ,
server :str ,
port :int ,
**kwargs ,
)->str :



    vllm_only_params ={'top_k','chat_template_kwargs'}


    has_vllm_params =any (param in kwargs for param in vllm_only_params )

    if has_vllm_params :

        return _query_vllm_direct_http (model ,messages ,max_tokens ,temperature ,server ,port ,**kwargs )
    else :

        client =OpenAI (
        api_key ="EMPTY",
        base_url =f"http://{server}:{port}/v1",
        )


        api_params ={
        "model":model ,
        "messages":messages ,
        "max_tokens":max_tokens ,
        "temperature":temperature ,
        "stream":False ,
        }


        api_params .update (kwargs )

        response =client .chat .completions .create (**api_params )
        msg =response .choices [0 ].message 


        return _extract_text_from_vllm_message (msg )


def _query_vllm_direct_http (
model :str ,
messages :List [Dict ],
max_tokens :int ,
temperature :float ,
server :str ,
port :int ,
**kwargs ,
)->str :

    endpoint =f"http://{server}:{port}/v1/chat/completions"
    headers ={"Content-Type":"application/json"}


    payload ={
    "model":model ,
    "messages":messages ,
    "max_tokens":max_tokens ,
    "temperature":temperature ,
    "stream":False ,
    }


    payload .update (kwargs )

    try :
        response =requests .post (endpoint ,headers =headers ,json =payload ,timeout =300 )
        response .raise_for_status ()

        response_data =response .json ()


        if "choices"in response_data and len (response_data ["choices"])>0 :
            choice =response_data ["choices"][0 ]
            if "message"in choice and "content"in choice ["message"]:
                return choice ["message"]["content"]or ""

        return ""

    except requests .exceptions .RequestException as e :
        print (f"HTTP request failed: {e}")
        return ""
    except json .JSONDecodeError as e :
        print (f"Failed to parse JSON response: {e}")
        return ""
    except Exception as e :
        print (f"Unexpected error in vLLM direct query: {e}")
        return ""