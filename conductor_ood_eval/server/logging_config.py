


import os 
import logging 
import logging .handlers 
from datetime import datetime 
from typing import Dict ,Any ,Optional 
from pathlib import Path 


def setup_logging (config :Optional [Dict [str ,Any ]]=None ,
log_level :str ="INFO")->logging .Logger :


    if "log_dir"in config :
        log_dir =config ["log_dir"]
    else :

        log_dir =os .getenv ('LOG_DIR','./server_logs')


    log_path =Path (log_dir )
    log_path .mkdir (parents =True ,exist_ok =True )


    timestamp =datetime .now ().strftime ("%Y%m%d_%H%M%S")
    log_file =log_path /f"oai_server_{timestamp}.log"


    root_logger =logging .getLogger ()
    root_logger .setLevel (getattr (logging ,log_level .upper ()))


    for handler in root_logger .handlers [:]:
        root_logger .removeHandler (handler )


    detailed_formatter =logging .Formatter (
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )

    simple_formatter =logging .Formatter (
    '%(asctime)s - %(levelname)s - %(message)s'
    )


    file_handler =logging .handlers .RotatingFileHandler (
    log_file ,maxBytes =10 *1024 *1024 ,backupCount =5 
    )
    file_handler .setLevel (logging .DEBUG )
    file_handler .setFormatter (detailed_formatter )
    root_logger .addHandler (file_handler )


    console_handler =logging .StreamHandler ()
    console_handler .setLevel (getattr (logging ,log_level .upper ()))
    console_handler .setFormatter (simple_formatter )
    root_logger .addHandler (console_handler )


    logging .getLogger ('uvicorn').setLevel (logging .WARNING )
    logging .getLogger ('uvicorn.access').setLevel (logging .WARNING )
    logging .getLogger ('httpx').setLevel (logging .WARNING )
    logging .getLogger ('transformers').setLevel (logging .WARNING )
    logging .getLogger ('torch').setLevel (logging .WARNING )


    logger =logging .getLogger (__name__ )
    logger .info (f"Logging initialized. Log file: {log_file}")
    logger .info (f"Log level: {log_level}")

    return logger 


def get_logger (name :str )->logging .Logger :

    return logging .getLogger (name )


class RequestLogger :


    def __init__ (self ,logger :logging .Logger ):
        self .logger =logger 

    async def log_request (self ,request ,call_next ):

        start_time =datetime .now ()


        self .logger .info (f"Request: {request.method} {request.url}")
        if hasattr (request ,'headers'):
            content_type =request .headers .get ('content-type','unknown')
            self .logger .debug (f"Content-Type: {content_type}")


        response =await call_next (request )


        duration =(datetime .now ()-start_time ).total_seconds ()
        self .logger .info (f"Response: {response.status_code} - {duration:.3f}s")

        return response 


def configure_model_logging (debug :bool =False ):

    level =logging .DEBUG if debug else logging .INFO 


    loggers =[
    'models.router_model',
    'models.agent_models',
    'server.coordination',
    'server.api'
    ]

    for logger_name in loggers :
        logger =logging .getLogger (logger_name )
        logger .setLevel (level )

    return logging .getLogger ('models')