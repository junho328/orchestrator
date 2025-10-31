

import multiprocessing as mp 
import threading 
import atexit 


class LiveCodeBenchPool :


    _instance =None 
    _lock =threading .Lock ()
    _pool =None 
    _initialized =False 

    def __new__ (cls ):
        if cls ._instance is None :
            with cls ._lock :
                if cls ._instance is None :
                    cls ._instance =super ().__new__ (cls )
        return cls ._instance 

    def __init__ (self ):
        if not self ._initialized :
            self ._initialized =True 
            atexit .register (self .cleanup )

    def get_pool (self ,num_workers :int =4 )->mp .Pool :

        with self ._lock :
            if self ._pool is None :

                ctx =mp .get_context ('spawn')
                self ._pool =ctx .Pool (processes =num_workers )
            return self ._pool 

    def cleanup (self ):

        with self ._lock :
            if self ._pool is not None :
                self ._pool .terminate ()
                self ._pool .join ()
                self ._pool =None 



_livecodebench_pool =LiveCodeBenchPool ()


def get_livecodebench_pool (num_workers :int =4 )->mp .Pool :

    return _livecodebench_pool .get_pool (num_workers )