import os 
import requests 
import json 
from typing import List ,Dict 
from openai import OpenAI 
from anthropic import Anthropic 
from google import genai 
from google .genai import types 
import random ,time ,boto3 
from botocore .config import Config 
from botocore .exceptions import ClientError 
import uuid 

THINKING_BUDGET =1024 

def query_oai (
model :str ,messages :List [Dict ],max_tokens :int ,temperature :float ,reasoning_effort :str ="minimal",**kwargs 
)->str :
    client =OpenAI (api_key =os .getenv ("OPENAI_API_KEY"),max_retries =50 )

    if model =="gpt-5":
        if reasoning_effort =="high":
            max_tokens =128000 
        api_params ={
        "model":model ,
        "messages":messages ,
        "max_completion_tokens":max_tokens ,
        "temperature":1 ,
        "stream":False ,
        "reasoning_effort":reasoning_effort 
        }
    else :
        api_params ={
        "model":model ,
        "messages":messages ,
        "max_tokens":max_tokens ,
        "temperature":temperature ,
        "stream":False ,
        }
    api_params .update (kwargs )

    for _ in range (20 ):
        response =client .chat .completions .create (**api_params )
        if (response .choices [0 ].message .content is not None )and (response .choices [0 ].message .content !=''):


            return response .choices [0 ].message .content 
        else :

            time .sleep (1 )
    print (f'exceeded inner attempts to obtain response from oai, returning empty string')
    return ""

def query_anthropic (
model :str ,
messages :List [Dict ],
max_tokens :int =256 ,
temperature :float =0.0 ,
platform :str ="bedrock",
max_wrapper_attempts :int =50 ,
base_backoff :float =1.0 ,
claude_thinking_budget :int =0 ,
**kwargs ,
)->str :
    if platform =="openrouter":

        api_key =os .getenv ('OPENROUTER_API_KEY')
        if not api_key :
            raise ValueError ("OPENROUTER_API_KEY is not set")
        client =OpenAI (base_url ="https://openrouter.ai/api/v1",api_key =api_key ,max_retries =50 ,thinking_budget =claude_thinking_budget )

        model ="anthropic/claude-sonnet-4"

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

    if platform =="bedrock":

        if model =="claude-sonnet-4-20250514":
            model ="us.anthropic.claude-sonnet-4-20250514-v1:0"



        stream =False 


        sdk_config =Config (
        connect_timeout =5 ,
        read_timeout =300 ,
        retries ={
        "total_max_attempts":50 ,
        "mode":"adaptive"
        }
        )

        client =boto3 .client ("bedrock-runtime",region_name ="us-east-1",config =sdk_config )


        system_message =None 
        converted_messages =[]
        for m in messages :
            if m ["role"]=="system":
                system_message =m ["content"]
                continue 
            converted_messages .append ({
            "role":m ["role"],
            "content":[{"text":m ["content"]}]
            })


        params ={
        "modelId":model ,
        "messages":converted_messages ,
        "inferenceConfig":{
        "maxTokens":max_tokens ,
        "temperature":temperature 
        }
        }
        if system_message is not None :
            params ["system"]=[{"text":system_message }]


        if claude_thinking_budget :
            params ["additionalModelRequestFields"]={
            "thinking":{"type":"enabled","budget_tokens":claude_thinking_budget }
            }

            params ["inferenceConfig"]["temperature"]=1 


        def _non_stream_call ()->str :
            response =client .converse (**params )
            blocks =response ["output"]["message"]["content"]
            text_blocks =[b .get ("text","")for b in blocks if isinstance (b ,dict )and b .get ("type")=="text"]
            if text_blocks :
                return "".join (text_blocks )

            fallback =[b .get ("text","")for b in blocks if isinstance (b ,dict )and "text"in b ]
            return "".join (fallback )if fallback else ""


        def _stream_call ()->str :


            full_text =[]
            response =client .converse_stream (**params )
            stream_obj =response .get ("stream")

            for event in stream_obj :
                if "contentBlockDelta"in event :
                    delta =event ["contentBlockDelta"]["delta"]

                    if delta .get ("type")=="text_delta"and "text"in delta :
                        t =delta ["text"]
                        full_text .append (t )







            return "".join (full_text )


        for attempt in range (1 ,max_wrapper_attempts +1 ):
            try :
                if stream :
                    return _stream_call ()
                else :
                    return _non_stream_call ()
            except ClientError as exc :
                if exc .response .get ("Error",{}).get ("Code")!="ThrottlingException":
                    raise 
                sleep_for =min (base_backoff *2 **(attempt -1 )+random .uniform (0 ,0.5 ),20 )
                time .sleep (sleep_for )

        print (f"Exceeded {max_wrapper_attempts} attempts due to persistent throttling.")
        return ""


    api_key =os .getenv ("ANTHROPIC_API_KEY")
    if not api_key :
        raise ValueError ("ANTHROPIC_API_KEY is not set")

    client =Anthropic (api_key =api_key ,max_retries =50 )


    system_message =None 
    filtered_messages =[]
    for message in messages :
        if message ["role"]=="system":
            system_message =message ["content"]
        else :
            filtered_messages .append (message )

    if max_tokens >8000 :
        stream =True 
    else :
        stream =False 

    kparams ={
    "model":model ,
    "max_tokens":max_tokens ,
    "temperature":temperature ,
    "messages":filtered_messages ,
    "stream":stream ,
    }
    if system_message is not None :
        kparams ["system"]=system_message 

    response =client .messages .create (**kparams )

    if stream :
        full_content =""
        for chunk in response :
            if chunk .type =="content_block_delta":
                full_content +=chunk .delta .text 
            elif chunk .type =="message_delta":

                pass 

        return full_content 
    else :

        return response .content [0 ].text 


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



def query_gemini (
model :str ,messages :List [Dict ],max_tokens :int ,temperature :float ,gemini_thinking_budget :int =4096 
)->str :


    if gemini_thinking_budget ==32768 :
        max_tokens =65535 


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

        adjusted_max_tokens =max_tokens 

        if "2.5-pro"in model :


            thinking_config =types .ThinkingConfig (
            thinking_budget =gemini_thinking_budget ,
            include_thoughts =False 
            )


        generation_config =types .GenerateContentConfig (
        max_output_tokens =adjusted_max_tokens ,
        temperature =temperature ,
        system_instruction =system_message ,
        )


        if thinking_config :
            generation_config .thinking_config =thinking_config 


        if "2.5-pro"in model :
            thinking_config =types .ThinkingConfig (
            thinking_budget =gemini_thinking_budget ,
            include_thoughts =False 
            )

        generation_config =types .GenerateContentConfig (
        max_output_tokens =adjusted_max_tokens ,
        temperature =temperature ,
        system_instruction =system_message ,
        )
        if thinking_config :
            generation_config .thinking_config =thinking_config 


        try :
            for retry_attempt in range (50 ):




                response =client .models .generate_content (
                model =model ,
                contents =contents ,
                config =generation_config ,
                )
                if (response .text is not None )and (response .text !=''):

                    break 
                else :
                    time .sleep (1 )

        except Exception as e :
            print (f"ERROR: Gemini API call failed for model {model}: {e}")



        if getattr (response ,"prompt_feedback",None ):
            block_reason =getattr (response .prompt_feedback ,"block_reason",None )
            if block_reason :
                print (f"WARNING: Request blocked by safety filter: {block_reason}")
                return ""


        candidates =getattr (response ,"candidates",None )
        if candidates :
            for candidate in candidates :
                if getattr (candidate ,"finish_reason",None )and candidate .finish_reason .name =="MAX_TOKENS":

                    break 


        if getattr (response ,"text",None ):
            return response .text 


        if candidates :
            for candidate in candidates :
                content =getattr (candidate ,"content",None )
                if content and getattr (content ,"parts",None ):
                    for part in content .parts :
                        if getattr (part ,"text",None ):
                            return part .text 

        print ("DEBUG: No text found in response from gemini")
        return ""

    except Exception as e :
        print (f"ERROR: Unexpected failure: {e}")
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

        try :
            rid =msg .id 
            requests .post (f"http://{server}:{port}/abort",json ={"request_id":rid },
            timeout =1.0 )
        except Exception :

            pass 


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