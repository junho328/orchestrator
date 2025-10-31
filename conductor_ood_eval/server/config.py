


import os 
import json 
from typing import Dict ,Any ,Optional ,List 
from dataclasses import dataclass ,field 
from pathlib import Path 


@dataclass 
class ServerConfig :

    host :str ="0.0.0.0"
    port :int =8080 
    log_level :str ="INFO"
    api_key_required :bool =True 
    cors_enabled :bool =True 
    max_request_size :int =10 *1024 *1024 


@dataclass 
class EngineConfig :

    name :str 
    engine_type :str 
    config :Dict [str ,Any ]=field (default_factory =dict )
    is_default :bool =False 


@dataclass 
class AppConfig :

    server :ServerConfig =field (default_factory =ServerConfig )
    engines :List [EngineConfig ]=field (default_factory =list )
    logging :Dict [str ,Any ]=field (default_factory =dict )


class ConfigManager :


    def __init__ (self ,config_file :Optional [str ]=None ):
        self .config_file =config_file 
        self .config =AppConfig ()

    def load_from_file (self ,config_file :str )->AppConfig :

        config_path =Path (config_file )
        if not config_path .exists ():
            raise FileNotFoundError (f"Configuration file not found: {config_file}")

        with open (config_path ,'r')as f :
            config_data =json .load (f )

        return self ._parse_config (config_data )

    def load_from_env (self )->AppConfig :

        config_data ={}


        if os .getenv ('SERVER_HOST'):
            config_data .setdefault ('server',{})['host']=os .getenv ('SERVER_HOST')
        if os .getenv ('SERVER_PORT'):
            config_data .setdefault ('server',{})['port']=int (os .getenv ('SERVER_PORT'))
        if os .getenv ('LOG_LEVEL'):
            config_data .setdefault ('server',{})['log_level']=os .getenv ('LOG_LEVEL')


        if os .getenv ('ENGINE_TYPE'):
            config_data .setdefault ('engines',[]).append ({
            'name':'default',
            'engine_type':os .getenv ('ENGINE_TYPE'),
            'config':{},
            'is_default':True 
            })

        return self ._parse_config (config_data )

    def load_config (self )->AppConfig :

        if self .config_file and os .path .exists (self .config_file ):
            self .config =self .load_from_file (self .config_file )
        else :
            self .config =self .load_from_env ()


        self ._validate_config ()

        return self .config 

    def _parse_config (self ,config_data :Dict [str ,Any ])->AppConfig :

        config =AppConfig ()


        if 'server'in config_data :
            server_data =config_data ['server']
            config .server =ServerConfig (
            host =server_data .get ('host',config .server .host ),
            port =server_data .get ('port',config .server .port ),
            log_level =server_data .get ('log_level',config .server .log_level ),
            api_key_required =server_data .get ('api_key_required',config .server .api_key_required ),
            cors_enabled =server_data .get ('cors_enabled',config .server .cors_enabled ),
            max_request_size =server_data .get ('max_request_size',config .server .max_request_size )
            )


        if 'engines'in config_data :
            config .engines =[]
            for engine_data in config_data ['engines']:
                config .engines .append (EngineConfig (
                name =engine_data ['name'],
                engine_type =engine_data ['engine_type'],
                config =engine_data .get ('config',{}),
                is_default =engine_data .get ('is_default',False )
                ))


        if 'logging'in config_data :
            config .logging =config_data ['logging']

        return config 

    def _validate_config (self )->None :


        valid_engine_types =['trinity','router','single','custom','conductor']
        for engine in self .config .engines :
            if engine .engine_type not in valid_engine_types :
                raise ValueError (f"Invalid engine type: {engine.engine_type}. Must be one of {valid_engine_types}")


            if engine .engine_type in ['router','trinity']:
                model_path =engine .config .get ('model_path')
                if not model_path :
                    raise ValueError (f"Router engine {engine.name} requires 'model_path' in config")
                if not os .path .exists (model_path ):
                    raise ValueError (f"Router model path does not exist: {model_path}")


            if engine .engine_type =='single':
                if 'model_name'not in engine .config or 'model_type'not in engine .config :
                    raise ValueError (f"Single engine {engine.name} requires 'model_name' and 'model_type' in config")

    def save_config (self ,config_file :str )->None :

        config_data ={
        'server':{
        'host':self .config .server .host ,
        'port':self .config .server .port ,
        'log_level':self .config .server .log_level ,
        'api_key_required':self .config .server .api_key_required ,
        'cors_enabled':self .config .server .cors_enabled ,
        'max_request_size':self .config .server .max_request_size 
        }
        }

        if self .config .engines :
            config_data ['engines']=[]
            for engine in self .config .engines :
                config_data ['engines'].append ({
                'name':engine .name ,
                'engine_type':engine .engine_type ,
                'config':engine .config ,
                'is_default':engine .is_default 
                })

        if self .config .logging :
            config_data ['logging']=self .config .logging 

        with open (config_file ,'w')as f :
            json .dump (config_data ,f ,indent =2 )


def load_default_config ()->AppConfig :

    config_manager =ConfigManager ()


    config_files =[
    'config.json',
    'server_config.json',
    os .path .expanduser ('~/.guf_server_config.json'),
    '/etc/guf_server/config.json'
    ]

    for config_file in config_files :
        if os .path .exists (config_file ):
            config_manager .config_file =config_file 
            break 

    return config_manager .load_config ()