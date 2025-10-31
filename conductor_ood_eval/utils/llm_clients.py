import os 
import json 
from typing import List ,Dict ,Optional 
from openai import AsyncOpenAI 
from anthropic import AsyncAnthropic 
from google import genai 
from google .genai import types 
import httpx 
import random ,time ,boto3 
import asyncio 
import aioboto3 
from botocore .config import Config 
from botocore .exceptions import ClientError 

GEMINI_MAX_CONCURRENT =int (os .getenv ("GEMINI_MAX_CONCURRENT","5"))
GEMINI_MAX_ATTEMPTS =int (os .getenv ("GEMINI_MAX_ATTEMPTS","40"))
GEMINI_TOTAL_TIMEOUT =float (os .getenv ("GEMINI_TOTAL_TIMEOUT","600"))
GEMINI_BASE_BACKOFF =float (os .getenv ("GEMINI_BASE_BACKOFF","1.0"))
GEMINI_MAX_BACKOFF =float (os .getenv ("GEMINI_MAX_BACKOFF","20.0"))

GEMINI_SEMAPHORE =asyncio .Semaphore (GEMINI_MAX_CONCURRENT )

async def query_oai (model_name :str ,messages :List [Dict [str ,str ]],
max_tokens :Optional [int ],temperature :float ,reasoning_effort :str ="minimal",versbosity :str ="medium")->str :
    client =AsyncOpenAI (api_key =os .getenv ("OPENAI_API_KEY"),max_retries =50 ,timeout =3000 )

    if model_name =="gpt-5":
        if reasoning_effort =="high":
            assert max_tokens >=128000 ,"max_tokens must be at least 128000 for high reasoning effort"


        api_params ={
        "model":model_name ,
        "messages":messages ,
        "max_completion_tokens":max_tokens ,
        "temperature":1 ,
        "stream":False ,
        "reasoning_effort":reasoning_effort ,
        }
    else :
        api_params ={
        "model":model_name ,
        "messages":messages ,
        "max_tokens":max_tokens ,
        "temperature":temperature ,
        "stream":False ,
        }


    for _ in range (20 ):
        response =await client .chat .completions .create (**api_params )
        if (response .choices [0 ].message .content is not None )and (response .choices [0 ].message .content !=''):


            return response .choices [0 ].message .content 
        else :

            time .sleep (1 )
    print (f'exceeded inner attempts to obtain response from oai, returning empty string')
    return ""

async def query_bedrock (model :str ,messages :List [Dict [str ,str ]],
max_tokens :Optional [int ],temperature :float )->str :

    if model =="claude-sonnet-4-20250514":
        model ="us.anthropic.claude-sonnet-4-20250514-v1:0"


    sdk_config =Config (
    connect_timeout =5 ,
    read_timeout =300 ,
    retries ={

    "total_max_attempts":8 ,
    "mode":"adaptive"
    }
    )

    client =boto3 .client ("bedrock-runtime",region_name ="us-east-1",
    config =sdk_config )


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


    for attempt in range (1 ,8 +1 ):
        try :
            response =client .converse (**params )
            return response ["output"]["message"]["content"][0 ]["text"]

        except ClientError as exc :
            if exc .response ["Error"]["Code"]!="ThrottlingException":
                raise 

            import time 
            time .sleep (3 )

    raise RuntimeError (
    f"Exceeded {8} attempts due to persistent throttling."
    )


async def query_anthropic (model_name :str ,messages :List [Dict [str ,str ]],
max_tokens :Optional [int ],temperature :float ,platform :str ="bedrock",
max_wrapper_attempts :int =15 ,
base_backoff :float =1.0 ,claude_thinking_budget :int =10000 )->str :

    if platform =="openrouter":
        api_key =os .getenv ("OPENROUTER_API_KEY")
        if not api_key :
            raise ValueError ("OPENROUTER_API_KEY is not set")

        client =AsyncOpenAI (base_url ="https://openrouter.ai/api/v1",api_key =api_key ,max_retries =50 )
        model_name ="anthropic/claude-sonnet-4"

        api_params ={
        "model":model_name ,
        "messages":messages ,
        "max_tokens":max_tokens ,
        "temperature":temperature ,
        "stream":False ,
        }


        response =await client .chat .completions .create (**api_params )
        return response .choices [0 ].message .content 


    if platform =="bedrock":

        if model_name =="claude-sonnet-4-20250514":
            bedrock_model_id ="us.anthropic.claude-sonnet-4-20250514-v1:0"

        sdk_config =Config (
        connect_timeout =5 ,
        read_timeout =3000 ,
        retries ={"total_max_attempts":50 ,"mode":"adaptive"},
        )

        system_message =None 
        converted_messages =[]
        for m in messages :
            if m ["role"]=="system":
                system_message =m ["content"]
                continue 
            converted_messages .append ({"role":m ["role"],"content":[{"text":m ["content"]}]})

        params ={
        "modelId":bedrock_model_id ,
        "messages":converted_messages ,
        "inferenceConfig":{"maxTokens":max_tokens ,"temperature":temperature },
        }
        if system_message is not None :
            params ["system"]=[{"text":system_message }]

        if claude_thinking_budget :
            params ["additionalModelRequestFields"]={"thinking":
            {"type":"enabled",
            "budget_tokens":claude_thinking_budget }}

            params ["inferenceConfig"]["temperature"]=1 

        session =aioboto3 .Session ()
        async with session .client ("bedrock-runtime",region_name ="us-east-1",config =sdk_config )as client :
            for attempt in range (1 ,max_wrapper_attempts +1 ):
                try :
                    resp =await client .converse (**params )
                    blocks =resp ["output"]["message"]["content"]
                    text_blocks =[b ["text"]for b in blocks if isinstance (b ,dict )and b .get ("type")=="text"and "text"in b ]
                    if text_blocks :
                        return "".join (text_blocks )

                    fallback =[b ["text"]for b in blocks if isinstance (b ,dict )and "text"in b ]
                    return "".join (fallback )if fallback else ""
                except ClientError as exc :
                    if exc .response ["Error"]["Code"]!="ThrottlingException":
                        raise 
                    sleep_for =min (base_backoff *2 **(attempt -1 )+random .uniform (0 ,0.5 ),20 )
                    await asyncio .sleep (sleep_for )

        raise RuntimeError (f"Exceeded {max_wrapper_attempts} attempts due to persistent throttling.")


    api_key =os .getenv ("ANTHROPIC_API_KEY")
    if not api_key :
        raise ValueError ("ANTHROPIC_API_KEY is not set")


    client =AsyncAnthropic (api_key =api_key ,max_retries =50 )

    system_message =None 
    chat_messages :List [Dict ]=[]

    for m in messages :
        if m ["role"]=="system":
            system_message =m ["content"]
        else :
            chat_messages .append (m )

    if max_tokens >20000 :
        stream =True 
    else :
        stream =False 

    params =dict (
    model =model_name ,
    messages =chat_messages ,
    max_tokens =max_tokens ,
    temperature =temperature ,
    stream =stream ,
    )
    if system_message is not None :
        params ["system"]=system_message 

    response =await client .messages .create (**params )

    if stream :
        full_content =""
        async for chunk in response :
            if chunk .type =="content_block_delta":
                full_content +=chunk .delta .text 
            elif chunk .type =="message_delta":

                pass 

        return full_content 
    else :

        return response .content [0 ].text 

async def query_gemini (model_name :str ,messages :List [Dict [str ,str ]],
max_tokens :Optional [int ],temperature :float ,thinking_budget :int =1024 )->str :
    client =genai .Client (api_key =os .getenv ("GEMINI_API_KEY"))


    system_message =None 
    content_parts =[]

    for message in messages :
        if message ["role"]=="system":
            system_message =message ["content"]
        else :
            content_parts .append (message ["content"])

    contents ="\n".join (content_parts )
    adjusted_max_tokens =max_tokens 

    start =time .monotonic ()
    async with GEMINI_SEMAPHORE :
        for attempt in range (1 ,GEMINI_MAX_ATTEMPTS +1 ):
            remaining =GEMINI_TOTAL_TIMEOUT -(time .monotonic ()-start )
            if remaining <=0 :
                break 


            await asyncio .sleep (random .uniform (0 ,0.25 ))

            try :
                response =await client .aio .models .generate_content (
                model =model_name ,
                contents =contents ,
                config =types .GenerateContentConfig (
                max_output_tokens =adjusted_max_tokens ,
                temperature =temperature ,
                system_instruction =system_message ,
                thinking_config =types .ThinkingConfig (thinking_budget =thinking_budget ),
                ),
                )


                if hasattr (response ,"text")and response .text and response .text .strip ():
                    return response .text 

                if hasattr (response ,"candidates")and response .candidates :
                    for candidate in response .candidates :
                        if hasattr (candidate ,"content")and candidate .content and hasattr (candidate .content ,"parts"):
                            for part in candidate .content .parts :
                                if hasattr (part ,"text")and part .text and str (part .text ).strip ():
                                    return part .text 


                msg ="Gemini returned empty text; retrying with backoff."
                sleep_for =min (GEMINI_BASE_BACKOFF *(2 **(attempt -1 )),GEMINI_MAX_BACKOFF )+random .uniform (0 ,0.5 )
                sleep_for =min (sleep_for ,max (0.0 ,remaining ))
                print (f"{msg} attempt={attempt}/{GEMINI_MAX_ATTEMPTS}, sleeping {sleep_for:.1f}s")
                await asyncio .sleep (sleep_for )
                continue 

            except Exception as e :
                s =str (e )or ""
                code =getattr (e ,"status_code",None )
                transient =(
                code in (429 ,503 )or 
                "UNAVAILABLE"in s or 
                "overloaded"in s .lower ()or 
                "try again later"in s .lower ()or 
                "temporarily unavailable"in s .lower ()
                )

                if not transient :

                    raise 

                sleep_for =min (GEMINI_BASE_BACKOFF *(2 **(attempt -1 )),GEMINI_MAX_BACKOFF )+random .uniform (0 ,0.5 )
                sleep_for =min (sleep_for ,max (0.0 ,remaining ))
                print (f"Gemini transient error ({s.splitlines()[0]}). attempt={attempt}/{GEMINI_MAX_ATTEMPTS}, sleeping {sleep_for:.1f}s")
                await asyncio .sleep (sleep_for )


    return ""






















































async def query_gemini_with_thoughts (model_name :str ,messages :List [Dict [str ,str ]],
max_tokens :Optional [int ],temperature :float ,thinking_budget :int =1024 )->str :
    client =genai .Client (api_key =os .getenv ("GEMINI_API_KEY"))


    system_message =None 
    content_parts =[]

    for message in messages :
        if message ["role"]=="system":
            system_message =message ["content"]
        else :
            content_parts .append (message ["content"])


    contents ="\n".join (content_parts )

    response =await client .aio .models .generate_content (
    model =model_name ,
    contents =contents ,
    config =types .GenerateContentConfig (
    max_output_tokens =max_tokens ,
    temperature =temperature ,
    system_instruction =system_message ,
    thinking_config =types .ThinkingConfig (thinking_budget =thinking_budget ,
    include_thoughts =True )
    ),
    )
    thoughts =[]
    text_parts =[]

    for part in response .candidates [0 ].content .parts :
        if not part .text :
            continue 
        if part .thought :
            thoughts .append (part .text )
        else :
            text_parts .append (part .text )

    result =""
    if thoughts :
        result +=f"**Internal Reasoning:**\n{chr(10).join(thoughts)}\n\n"
    if text_parts :
        result +=f"**Response:**\n{chr(10).join(text_parts)}"

    return result 


async def query_deepseek (
model :str ,messages :List [Dict ],max_tokens :int ,temperature :float ,use_together :bool =True 
)->str :


    if use_together :

        client =AsyncOpenAI (
        api_key =os .getenv ("TOGETHER_API_KEY"),
        base_url ="https://api.together.xyz/v1",
        )
    else :

        client =AsyncOpenAI (
        api_key =os .getenv ("DEEPSEEK_API_KEY"),
        base_url ="https://api.deepseek.com",
        )

    response =await client .chat .completions .create (
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


async def query_locally_hosted_model (
model_name :str ,
messages :List [Dict ],
max_tokens :int ,
temperature :float ,
server :str ,
port :int ,
**kwargs ,
)->str :


    if model_name =="Qwen/Qwen3-32B-Direct":
        payload ={"top_p":0.8 ,
        "top_k":20 ,
        "presence_penalty":1.0 ,
        "chat_template_kwargs":{
        "enable_thinking":False }}

        model_name ="Qwen/Qwen3-32B"
        kwargs .update (payload )

    elif model_name =="Qwen/Qwen3-32B-Reasoning":
        payload ={"top_p":0.8 ,
        "top_k":20 ,
        "presence_penalty":1.0 ,
        "chat_template_kwargs":{
        "enable_thinking":True }}

        model_name ="Qwen/Qwen3-32B"
        kwargs .update (payload )


    vllm_only_params ={'top_k','chat_template_kwargs'}


    has_vllm_params =any (param in kwargs for param in vllm_only_params )

    if has_vllm_params :

        response =await _query_vllm_direct_http (model_name ,messages ,max_tokens ,temperature ,server ,port ,**kwargs )
        return response 
    else :

        client =AsyncOpenAI (
        api_key ="EMPTY",
        base_url =f"http://{server}:{port}/v1",
        )


        api_params ={
        "model":model_name ,
        "messages":messages ,
        "max_tokens":max_tokens ,
        "temperature":temperature ,
        "stream":False ,
        }


        api_params .update (kwargs )

        response =await client .chat .completions .create (**api_params )
        msg =response .choices [0 ].message 


        return _extract_text_from_vllm_message (msg )


async def _query_vllm_direct_http (
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
        async with httpx .AsyncClient (timeout =300 )as client :
            response =await client .post (endpoint ,headers =headers ,json =payload )
            response .raise_for_status ()

        response_data =response .json ()


        if "choices"in response_data and len (response_data ["choices"])>0 :
            choice =response_data ["choices"][0 ]
            if "message"in choice and "content"in choice ["message"]:
                return choice ["message"]["content"]or ""

    except (httpx .HTTPError ,json .JSONDecodeError )as exc :
        print (f"vLLM request failed: {exc}")
        return ""