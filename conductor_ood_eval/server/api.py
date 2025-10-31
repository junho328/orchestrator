


import time 
import logging 
from typing import List ,Dict ,Any ,Optional ,Union 
from contextlib import asynccontextmanager 

from fastapi import FastAPI ,HTTPException ,Header ,Depends 
from fastapi .responses import StreamingResponse 
from pydantic import BaseModel 

from models import EngineManager ,GenerationRequest 


logger =logging .getLogger (__name__ )



class ChatCompletionRequest (BaseModel ):
    model :str 
    messages :List [Dict [str ,str ]]
    temperature :Optional [float ]=0.1 
    max_tokens :Optional [int ]=None 
    stream :Optional [bool ]=False 
    stop :Optional [Union [str ,List [str ]]]=None 


class ChatCompletionChoice (BaseModel ):
    index :int 
    message :Dict [str ,str ]
    finish_reason :str 


class ChatCompletionResponse (BaseModel ):
    id :str 
    object :str ="chat.completion"
    created :int 
    model :str 
    choices :List [ChatCompletionChoice ]
    usage :Dict [str ,int ]
    metadata :Optional [Dict [str ,Any ]]=None 


class ChatCompletionStreamChunk (BaseModel ):
    id :str 
    object :str ="chat.completion.chunk"
    created :int 
    model :str 
    choices :List [Dict [str ,Any ]]


class OAICompatibleServer :


    def __init__ (self ,engine_manager :EngineManager ,api_key_required :bool =True ):
        self .engine_manager =engine_manager 
        self .api_key_required =api_key_required 


        self .app =FastAPI (
        title ="OAI-Compatible Model Server",
        description ="OpenAI-compatible API server with pluggable model engines",
        version ="1.0.0",
        lifespan =self ._lifespan 
        )


        self ._register_routes ()

    @asynccontextmanager 
    async def _lifespan (self ,app :FastAPI ):


        logger .info ("Starting up OAI-compatible server...")
        await self .engine_manager .load_all ()
        logger .info ("All engines loaded successfully")

        yield 


        logger .info ("Shutting down OAI-compatible server...")
        await self .engine_manager .unload_all ()
        logger .info ("All engines unloaded")

    def _register_routes (self ):


        @self .app .get ("/v1/models")
        async def list_models ():

            all_models =self .engine_manager .list_all_models ()

            return {
            "object":"list",
            "data":[
            {
            "id":model_id ,
            "object":"model",
            "created":int (time .time ()),
            "owned_by":"oai-server"
            }
            for model_id in all_models 
            ]
            }

        @self .app .post ("/v1/chat/completions")
        async def chat_completions (
        request :ChatCompletionRequest ,
        api_key :str =Depends (self ._verify_api_key )
        ):

            logger .info (f"Chat completion request - Model: {request.model}, "
            f"Messages: {len(request.messages)}, Temperature: {request.temperature}")


            engine =self .engine_manager .get_engine ()


            engine_request =GenerationRequest (
            messages =request .messages ,
            temperature =request .temperature ,
            max_tokens =request .max_tokens ,
            stream =request .stream ,
            stop =request .stop ,
            model =request .model 
            )


            if request .stream :


                return StreamingResponse (
                self ._fake_stream_response (engine ,engine_request ),
                media_type ="text/event-stream"
                )


            engine_response =await engine .generate (engine_request )


            oai_response =ChatCompletionResponse (
            id =f"chatcmpl-{int(time.time())}",
            created =int (time .time ()),
            model =request .model ,
            choices =[
            ChatCompletionChoice (
            index =0 ,
            message ={"role":"assistant","content":engine_response .content },
            finish_reason =engine_response .finish_reason 
            )
            ],
            usage =engine_response .usage or {
            "prompt_tokens":0 ,
            "completion_tokens":0 ,
            "total_tokens":0 
            },
            metadata =engine_response .metadata 
            )

            logger .info (f"Response generated - Tokens: {oai_response.usage.get('total_tokens', 0)}")
            return oai_response 

        @self .app .get ("/health")
        async def health_check ():

            engine_health =await self .engine_manager .health_check_all ()


            overall_status ="ready"
            for health in engine_health .values ():
                if health .get ("status")!="ready":
                    overall_status ="degraded"
                    break 

            return {
            "status":overall_status ,
            "engines":engine_health ,
            "server":{
            "api_key_required":self .api_key_required ,
            "available_models":self .engine_manager .list_all_models ()
            }
            }

        @self .app .get ("/v1/health")
        async def health_check_v1 ():

            return await health_check ()

        @self .app .get ("/")
        async def root ():

            return {
            "service":"OAI-Compatible Model Server",
            "version":"1.0.0",
            "engines":len (self .engine_manager .engines ),
            "models":len (self .engine_manager .list_all_models ())
            }

    async def _fake_stream_response (self ,engine ,engine_request ):


        engine_response =await engine .generate (engine_request )


        stream_id =f"chatcmpl-{int(time.time())}"
        created_time =int (time .time ())


        content =engine_response .content 
        chunk_size =10 


        chunk =ChatCompletionStreamChunk (
        id =stream_id ,
        created =created_time ,
        model =engine_request .model or "default-model",
        choices =[{
        "index":0 ,
        "delta":{"role":"assistant"},
        "finish_reason":None 
        }]
        )
        yield f"data: {chunk.json()}\n\n"


        for i in range (0 ,len (content ),chunk_size ):
            chunk_content =content [i :i +chunk_size ]
            chunk =ChatCompletionStreamChunk (
            id =stream_id ,
            created =created_time ,
            model =engine_request .model or "default-model",
            choices =[{
            "index":0 ,
            "delta":{"content":chunk_content },
            "finish_reason":None 
            }]
            )
            yield f"data: {chunk.json()}\n\n"


        chunk =ChatCompletionStreamChunk (
        id =stream_id ,
        created =created_time ,
        model =engine_request .model or "default-model",
        choices =[{
        "index":0 ,
        "delta":{},
        "finish_reason":engine_response .finish_reason or "stop"
        }]
        )
        yield f"data: {chunk.json()}\n\n"


        yield "data: [DONE]\n\n"

    def _verify_api_key (self ,authorization :str =Header (None )):

        if not self .api_key_required :
            return "no-key-required"


        token =authorization [7 :]if authorization and authorization .startswith ("Bearer ")else "bypass"
        return token 
