import os 
import json 
import threading 
import numpy as np 
from datetime import datetime 


def convert_numpy_types (obj ):

    if isinstance (obj ,np .integer ):
        return int (obj )
    elif isinstance (obj ,np .floating ):
        return float (obj )
    elif isinstance (obj ,np .bool_ ):
        return bool (obj )
    elif isinstance (obj ,np .ndarray ):
        return obj .tolist ()
    elif isinstance (obj ,dict ):
        return {key :convert_numpy_types (value )for key ,value in obj .items ()}
    elif isinstance (obj ,list ):
        return [convert_numpy_types (item )for item in obj ]
    elif isinstance (obj ,tuple ):
        return tuple (convert_numpy_types (item )for item in obj )
    else :
        return obj 


class WorkerLifecycleLogger :
    def __init__ (self ,log_dir ="worker_debug_logs"):
        self .log_dir =log_dir 
        try :
            os .makedirs (log_dir ,exist_ok =True )
            print (f"[DEBUG] Created log directory: {log_dir}")
        except Exception as e :
            print (f"[DEBUG] Failed to create log directory {log_dir}: {e}")

        self .worker_id =os .getpid ()
        self .log_file =os .path .join (log_dir ,f"worker_{self.worker_id}_lifecycle.jsonl")
        self .lock =threading .Lock ()


        try :
            with open (self .log_file ,"a")as f :
                f .write ("")
            print (f"[DEBUG] Worker {self.worker_id} logging to: {self.log_file}")
        except Exception as e :
            print (f"[DEBUG] Cannot write to log file {self.log_file}: {e}")

    def log_event (self ,event_type ,details =None ):
        with self .lock :
            try :

                clean_details =convert_numpy_types (details or {})

                entry ={
                "timestamp":datetime .now ().isoformat (),
                "worker_id":self .worker_id ,
                "event_type":event_type ,
                "details":clean_details 
                }


                json_str =json .dumps (entry )

                with open (self .log_file ,"a")as f :
                    f .write (json_str +"\n")
                    f .flush ()


                if not hasattr (self ,'_logged_count'):
                    self ._logged_count =0 
                self ._logged_count +=1 

                if self ._logged_count <=3 :
                    print (f"[DEBUG] Successfully logged event {self._logged_count}: {event_type}")

            except Exception as e :

                print (f"[WORKER-{self.worker_id}] {event_type}: {clean_details} (log_error: {e})")



_worker_logger =None 


def get_worker_logger ():
    global _worker_logger 
    if _worker_logger is None :
        _worker_logger =WorkerLifecycleLogger ()
        print (f"[DEBUG] Initialized worker logger for PID {os.getpid()}")
    return _worker_logger 