

import sys 
import asyncio 
import fire 
import signal 
import uvicorn 
from pathlib import Path 
from typing import Optional 


sys .path .append (str (Path (__file__ ).parent ))

from server import (
ConfigManager ,
OAICompatibleServer ,
setup_logging ,
setup_error_handlers ,
HealthChecker 
)
from models import (
EngineManager ,
ModelConfig ,
PythonEngine ,
)


class ServerOrchestrator :


    def __init__ (self ,config_file :Optional [str ]=None ):
        self .config_manager =ConfigManager (config_file )
        self .config =None 
        self .engine_manager =EngineManager ()
        self .api_server =None 
        self .health_checker =HealthChecker ()
        self .logger =None 


        self ._shutdown_event =asyncio .Event ()
        self ._setup_signal_handlers ()

    def _setup_signal_handlers (self ):

        def signal_handler (signum ,frame ):
            print (f"\nReceived signal {signum}. Initiating graceful shutdown...")
            asyncio .create_task (self ._shutdown ())

        signal .signal (signal .SIGINT ,signal_handler )
        signal .signal (signal .SIGTERM ,signal_handler )

    async def _shutdown (self ):

        self ._shutdown_event .set ()

    async def initialize (self )->None :


        self .config =self .config_manager .load_config ()


        self .logger =setup_logging (
        config =self .config .logging ,
        log_level =self .config .server .log_level 
        )

        self .logger .info ("Initializing OAI-compatible model server...")


        await self ._initialize_engines ()


        await self ._initialize_api_server ()


        await self ._setup_health_checks ()

        self .logger .info ("Server initialization complete")

    async def _initialize_engines (self )->None :

        self .logger .info ("Initializing model engines...")

        if not self .config .engines :
            self .logger .warning ("No engines configured - server will have no models available")
            return 

        for engine_config in self .config .engines :
            engine =self ._create_engine (engine_config )
            self .engine_manager .register_engine (engine ,engine_config .is_default )
            self .logger .info (f"Registered engine: {engine_config.name} ({engine_config.engine_type})")


        await self .engine_manager .load_all ()
        self .logger .info ("All engines loaded successfully")

        self .logger .info (
        "Example request:\n\n"
        f"curl -X POST http://{self.config.server.host}:{self.config.server.port}/v1/chat/completions \\\n"
        '  -H "Content-Type: application/json" \\\n'
        '  -d \'{"model":"router-engine","messages":[{"role":"user","content":"How many r in word strawberry"}],"max_tokens":512,"temperature":0.2}\'\n\n'
        )

    def _create_engine (self ,engine_config ):

        model_config =ModelConfig (
        name =engine_config .name ,
        engine_type =engine_config .engine_type ,
        config =engine_config .config 
        )

        if engine_config .engine_type =="custom":
            return PythonEngine (model_config )
        else :
            raise ValueError (f"Unsupported engine type: {engine_config.engine_type}")

    async def _initialize_api_server (self )->None :

        self .api_server =OAICompatibleServer (
        engine_manager =self .engine_manager ,
        api_key_required =self .config .server .api_key_required 
        )


        setup_error_handlers (self .api_server .app )

        self .logger .info ("API server initialized")

    async def _setup_health_checks (self )->None :


        self .health_checker .register_check (
        "engines",
        self .engine_manager .health_check_all 
        )

        async def server_health ():
            return {
            "engine_count":len (self .engine_manager .engines ),
            "available_models":len (self .engine_manager .list_all_models ()),
            "default_engine":self .engine_manager .default_engine 
            }

        self .health_checker .register_check ("server",server_health )

        self .logger .info ("Health checks configured")

    async def run (self ,config :Optional [str ]=None ,host :Optional [str ]=None ,
    port :Optional [int ]=None ,debug :bool =False ,
    log_level :Optional [str ]=None )->None :

        orchestrator =ServerOrchestrator (config )


        if hasattr (orchestrator ,'config')and orchestrator .config :
            if host :
                orchestrator .config .server .host =host 
            if port :
                orchestrator .config .server .port =port 
            if debug :
                orchestrator .config .server .log_level ="DEBUG"

        await orchestrator .initialize ()


        server_config =uvicorn .Config (
        app =orchestrator .api_server .app ,
        host =orchestrator .config .server .host ,
        port =orchestrator .config .server .port ,
        log_level ="warning",
        access_log =False 
        )

        server =uvicorn .Server (server_config )

        orchestrator .logger .info (f"Starting server on {orchestrator.config.server.host}:{orchestrator.config.server.port}")

        await server .serve ()

    async def shutdown (self )->None :

        self .logger .info ("Starting graceful shutdown...")


        if self .engine_manager :
            await self .engine_manager .unload_all ()
            self .logger .info ("Engines unloaded")

        self .logger .info ("Shutdown complete")


def main ():

    try :
        fire .Fire (ServerOrchestrator ().run )
    except KeyboardInterrupt :
        print ("\nShutdown requested by user")
    except Exception as e :
        print (f"Server error: {e}")
        sys .exit (1 )


if __name__ =="__main__":
    main ()