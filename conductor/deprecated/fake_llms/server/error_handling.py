

import asyncio 
import logging 
import traceback 
from typing import Any ,Callable ,Dict ,Optional ,Type 
from functools import wraps 
from contextlib import asynccontextmanager 

from fastapi import HTTPException ,Request ,Response 
from fastapi .responses import JSONResponse 


logger =logging .getLogger (__name__ )


ERROR_HANDLING_ENABLED =False 


class ServerError (Exception ):


    def __init__ (self ,message :str ,status_code :int =500 ,details :Optional [Dict [str ,Any ]]=None ):
        super ().__init__ (message )
        self .message =message 
        self .status_code =status_code 
        self .details =details or {}


class ModelNotReadyError (ServerError ):


    def __init__ (self ,model_name :str ):
        super ().__init__ (
        f"Model {model_name} is not ready",
        status_code =503 ,
        details ={"model_name":model_name }
        )


class CoordinationError (ServerError ):


    def __init__ (self ,message :str ,turn :int =0 ,agent :Optional [str ]=None ):
        super ().__init__ (
        message ,
        status_code =500 ,
        details ={"turn":turn ,"agent":agent }
        )


class RateLimitError (ServerError ):


    def __init__ (self ,message :str ="Rate limit exceeded"):
        super ().__init__ (message ,status_code =429 )


class ValidationError (ServerError ):


    def __init__ (self ,message :str ,field :Optional [str ]=None ):
        super ().__init__ (
        message ,
        status_code =400 ,
        details ={"field":field }if field else {}
        )


def retry_with_backoff (max_retries :int =3 ,base_delay :float =1.0 ,
max_delay :float =60.0 ,backoff_factor :float =2.0 ):

    def decorator (func :Callable ):
        @wraps (func )
        async def wrapper (*args ,**kwargs ):
            if not ERROR_HANDLING_ENABLED :
                return await func (*args ,**kwargs )

            last_exception =None 
            for attempt in range (max_retries +1 ):
                try :
                    return await func (*args ,**kwargs )
                except Exception as e :
                    last_exception =e 
                    if attempt ==max_retries :
                        logger .error (f"Function {func.__name__} failed after {max_retries + 1} attempts: {e}")
                        raise 

                    delay =min (base_delay *(backoff_factor **attempt ),max_delay )
                    logger .warning (f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay}s")
                    await asyncio .sleep (delay )

            raise last_exception 
        return wrapper 
    return decorator 


def timeout_handler (timeout_seconds :float =120.0 ):

    def decorator (func :Callable ):
        @wraps (func )
        async def wrapper (*args ,**kwargs ):
            if not ERROR_HANDLING_ENABLED :
                return await func (*args ,**kwargs )

            try :
                return await asyncio .wait_for (func (*args ,**kwargs ),timeout =timeout_seconds )
            except asyncio .TimeoutError :
                logger .error (f"Function {func.__name__} timed out after {timeout_seconds}s")
                raise ServerError (f"Operation timed out after {timeout_seconds}s",status_code =408 )
        return wrapper 
    return decorator 


@asynccontextmanager 
async def error_context (operation :str ,**context ):

    if not ERROR_HANDLING_ENABLED :
        yield 
        return 

    try :
        logger .debug (f"Starting operation: {operation}",extra =context )
        yield 
        logger .debug (f"Completed operation: {operation}")
    except Exception as e :
        logger .error (f"Operation {operation} failed: {e}",extra ={**context ,"error":str (e )})
        raise 


def setup_error_handlers (app ):

    if not ERROR_HANDLING_ENABLED :
        return 

    @app .exception_handler (ServerError )
    async def server_error_handler (request :Request ,exc :ServerError ):
        return JSONResponse (
        status_code =exc .status_code ,
        content ={
        "error":exc .message ,
        "details":exc .details ,
        "type":exc .__class__ .__name__ 
        }
        )

    @app .exception_handler (HTTPException )
    async def http_exception_handler (request :Request ,exc :HTTPException ):
        return JSONResponse (
        status_code =exc .status_code ,
        content ={"error":exc .detail }
        )

    @app .exception_handler (Exception )
    async def general_exception_handler (request :Request ,exc :Exception ):
        logger .error (f"Unhandled exception: {exc}",exc_info =True )
        return JSONResponse (
        status_code =500 ,
        content ={"error":"Internal server error"}
        )


class CircuitBreaker :


    def __init__ (self ,failure_threshold :int =5 ,timeout :float =60.0 ):
        self .failure_threshold =failure_threshold 
        self .timeout =timeout 
        self .failure_count =0 
        self .last_failure_time =0 
        self .state ="closed"

    async def call (self ,func :Callable ,*args ,**kwargs ):

        if not ERROR_HANDLING_ENABLED :
            return await func (*args ,**kwargs )

        import time 
        current_time =time .time ()


        if self .state =="open":
            if current_time -self .last_failure_time >self .timeout :
                self .state ="half-open"
                logger .info ("Circuit breaker transitioning to half-open")
            else :
                raise ServerError ("Circuit breaker is open",status_code =503 )

        try :
            result =await func (*args ,**kwargs )


            if self .state =="half-open":
                self .state ="closed"
                self .failure_count =0 
                logger .info ("Circuit breaker closed after successful call")

            return result 

        except Exception as e :
            self .failure_count +=1 
            self .last_failure_time =current_time 

            if self .failure_count >=self .failure_threshold :
                self .state ="open"
                logger .warning (f"Circuit breaker opened after {self.failure_count} failures")

            raise 


class HealthChecker :


    def __init__ (self ):
        self .checks ={}

    def register_check (self ,name :str ,check_func :Callable ):

        self .checks [name ]=check_func 

    async def run_checks (self )->Dict [str ,Dict [str ,Any ]]:

        results ={}

        for name ,check_func in self .checks .items ():
            start_time =asyncio .get_event_loop ().time ()

            if not ERROR_HANDLING_ENABLED :
                result =await check_func ()
                end_time =asyncio .get_event_loop ().time ()
                results [name ]={
                "status":"healthy",
                "response_time":end_time -start_time ,
                "details":result if isinstance (result ,dict )else {}
                }
            else :
                try :
                    result =await check_func ()
                    end_time =asyncio .get_event_loop ().time ()

                    results [name ]={
                    "status":"healthy",
                    "response_time":end_time -start_time ,
                    "details":result if isinstance (result ,dict )else {}
                    }

                except Exception as e :
                    end_time =asyncio .get_event_loop ().time ()
                    logger .error (f"Health check {name} failed: {e}")

                    results [name ]={
                    "status":"unhealthy",
                    "response_time":end_time -start_time ,
                    "error":str (e ),
                    "details":{}
                    }

        return results 

    async def is_healthy (self )->bool :

        results =await self .run_checks ()
        return all (result ["status"]=="healthy"for result in results .values ())