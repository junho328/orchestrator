

from typing import Dict ,List ,Any 
from guf .tasks .countdown import CountdownTask 


class StructuredCountdownTask (CountdownTask ):



    PROBLEM_PROMPT ="""<problem>
Using the numbers {numbers}, create an equation that equals {target}.
You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once.
You can think but return the final equation in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>"
</problem>
"""

    def __init__ (self ,llm_names :List [str ],**kwargs ):


        kwargs .pop ('use_structured_router',None )


        super ().__init__ (llm_names ,**kwargs )


        self .use_structured_router =True 


        self .system_prompt =self ._get_structured_system_prompt ()
        self .router_prompt =self ._get_structured_router_prompt ()
        self .step_prompt =self ._get_structured_step_prompt ()


    def _format_base_prompt (self ,task_data :Dict [str ,Any ])->str :

        if self .use_structured_router :

            return self .PROBLEM_PROMPT .format (
            numbers =task_data ["nums"],
            target =task_data ["target"],
            )
        else :

            return super ()._format_base_prompt (task_data )


    def _calculate_reward (
    self ,
    completions :List [str ],
    task_data :List [Dict [str ,Any ]],
    debug :bool =False 
    )->List [float ]:

        return super ()._calculate_reward (completions ,task_data ,debug )