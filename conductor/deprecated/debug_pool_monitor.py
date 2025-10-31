import threading 
import time 
import json 
import os 
from datetime import datetime 


class PoolMonitor :
    def __init__ (self ,pool ,log_dir ="debug_logs"):
        self .pool =pool 
        self .log_dir =log_dir 
        os .makedirs (log_dir ,exist_ok =True )
        self .log_file =os .path .join (log_dir ,"pool_monitor.jsonl")
        self .monitoring =False 
        self .monitor_thread =None 

    def start_monitoring (self ,interval =10 ):

        self .monitoring =True 
        self .monitor_thread =threading .Thread (target =self ._monitor_loop ,args =(interval ,))
        self .monitor_thread .daemon =True 
        self .monitor_thread .start ()

    def stop_monitoring (self ):

        self .monitoring =False 
        if self .monitor_thread :
            self .monitor_thread .join (timeout =5 )

    def _monitor_loop (self ,interval ):

        while self .monitoring :
            try :

                pool_info ={
                "timestamp":datetime .now ().isoformat (),
                "pool_processes":len (self .pool ._pool )if hasattr (self .pool ,'_pool')else "unknown",
                "pool_state":str (self .pool ._state )if hasattr (self .pool ,'_state')else "unknown",
                }


                with open (self .log_file ,"a")as f :
                    f .write (json .dumps (pool_info )+"\n")
                    f .flush ()

            except Exception as e :
                print (f"Pool monitoring error: {e}")

            time .sleep (interval )