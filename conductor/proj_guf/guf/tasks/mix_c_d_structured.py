
from typing import Dict ,Any ,List ,Optional 
from guf .tasks .mix_c_d import MixCDTask 
from guf .core import Task 
from guf .cost import reset_costs 


class StructuredMixCDTask (MixCDTask ):



    COUNTDOWN_PROBLEM_PROMPT ="""<problem>
Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
</problem>
"""


    DEEPMATH_PROBLEM_PROMPT ="""<problem>
Solve the following mathematical problem step by step: {question} Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags. For your final answer, follow these guidelines:
1. Provide ONLY the final result without explanations, equations, or steps
2. Use standard LaTeX notation for mathematical expressions with single backslashes (not double)
3. Keep the answer as concise as possible
4. For limits, derivatives, and integrals, express the final value directly
5. For yes/no questions, simply answer 'Yes' or 'No'
6. IMPORTANT: Your final answer MUST be enclosed in <answer> </answer> tags
7. Format examples: \\delta (not \\\\delta), \\frac{{a}}{{b}} (not \\\\frac{{a}}{{b}})
Example: <answer>\\frac{{1}}{{2}}</answer>
</problem>
"""

    def __init__ (
    self ,
    llm_names :List [str ],
    seed :int =42 ,
    max_tokens :int =1536 ,
    temperature :float =0.8 ,
    max_turns :int =5 ,
    include_format_reward :bool =True ,
    task_ratios :Dict [str ,float ]=None ,
    balanced_sampling :bool =True ,
    servers :Optional [Dict [str ,str ]]=None ,
    ports :Optional [Dict [str ,int ]]=None ,
    track_costs :bool =True ,
    debug :bool =False ,
    together :bool =True ,
    valid_ratio :float =0.5 ,
    max_samples :int =-1 ,
    test_ratio :float =0.2 ,
    use_structured_router :bool =True ,
    use_consultant :bool =True ,
    **kwargs 
    ):


        kwargs .pop ('use_structured_router',None )


        super ().__init__ (
        llm_names =llm_names ,
        seed =seed ,
        max_tokens =max_tokens ,
        temperature =temperature ,
        max_turns =max_turns ,
        include_format_reward =include_format_reward ,
        task_ratios =task_ratios ,
        balanced_sampling =balanced_sampling ,
        servers =servers ,
        ports =ports ,
        track_costs =track_costs ,
        debug =debug ,
        together =together ,
        valid_ratio =valid_ratio ,
        max_samples =max_samples ,
        test_ratio =test_ratio ,
        use_consultant =use_consultant ,
        **kwargs 
        )


        self .use_structured_router =True 


        self .system_prompt =self ._get_structured_system_prompt ()
        self .router_prompt =self ._get_structured_router_prompt ()
        self .step_prompt =self ._get_structured_step_prompt ()


        if self .debug :
            sampling_method ="balanced"if self .balanced_sampling else "unbalanced"
            print (f"[StructuredMixCDTask] Initialized with {sampling_method} sampling")
            if self .balanced_sampling and task_ratios :
                print (f"[StructuredMixCDTask] Task ratios: {task_ratios}")

    def _format_base_prompt (self ,task_data :Dict [str ,Any ])->str :

        if self .use_structured_router :
            if task_data .get ("task_type","deepmath")=="countdown":
                return self .COUNTDOWN_PROBLEM_PROMPT .format (
                numbers =task_data ["nums"],
                target =task_data ["target"],
                )
            else :
                q =task_data .get ("question","")

                q =str (q ).replace ("{","{{").replace ("}","}}")
                return self .DEEPMATH_PROBLEM_PROMPT .format (question =q )


        return super ()._format_base_prompt (task_data )

    def reset (self ,task_id :int =-1 ,split :str ="train"):


        if self .track_costs :
            reset_costs ()

        if self .use_consultant :
            self .pending_suggestion =None 


        return Task .reset (self ,task_id =task_id ,split =split )

    def get_sampling_info (self )->Dict [str ,Any ]:

        info ={
        "sampling_method":"balanced"if self .balanced_sampling else "unbalanced",
        "use_structured_router":self .use_structured_router ,
        "task_type":"structured_mixed_countdown_deepmath"
        }

        if self .balanced_sampling :
            info ["task_ratios"]=self .task_ratios 

        return info 


if __name__ =="__main__":

    llms =["gpt-4o-mini","claude-3-7-sonnet-20250219","gemini-1.5-pro","deepseek-ai/DeepSeek-V3"]


    print ("=== Testing Balanced Structured Version ===")
    task_balanced =StructuredMixCDTask (
    llm_names =llms ,
    max_turns =5 ,
    max_tokens =1536 ,
    debug =True ,
    balanced_sampling =True ,
    task_ratios ={"countdown":0.5 ,"deepmath":0.5 }
    )

    print ("Sampling info:",task_balanced .get_sampling_info ())
    obs =task_balanced .reset (split ="train")
    print (f"Current task type: {task_balanced.current_task_type}")


    print ("\n=== Testing Unbalanced Structured Version ===")
    task_unbalanced =StructuredMixCDTask (
    llm_names =llms ,
    max_turns =5 ,
    max_tokens =1536 ,
    debug =True ,
    balanced_sampling =False 
    )

    print ("Sampling info:",task_unbalanced .get_sampling_info ())
    obs =task_unbalanced .reset (split ="train")
    print (f"Current task type: {task_unbalanced.current_task_type}")