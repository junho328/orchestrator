

import numpy as np 
import time 
import os 
import json 
from abc import ABC ,abstractmethod 
from typing import Dict ,List ,Optional ,Union 
from datasets import Dataset 
from tqdm import tqdm 
from guf .utils import query_llm ,get_or_create_indices 
from guf .cost import get_cost_summary ,reset_costs 


class Task (ABC ):



    thinking_tag ="idea"


    DEFAULT_ROUTER_SYSTEM_PROMPT =(
    "You are a message dispatcher whose job is to coordinate {num_agents} agents to solve a problem. "
    "You check the problem and the discussion history and then decide which agent should respond next. "
    "Your response should be a number between 1 and {num_agents} where 1 is the first agent, 2 is the second agent, etc. "
    "If an agent is consecutively chosen, its answer will be returned."
    )

    DEFAULT_SYSTEM_PROMPT =(
    "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer."
    )

    DEFAULT_ASSISTANT_PROMPT ="Let me solve this step by step.\n<think>"

    DEFAULT_CONSULT_PROMPT =(
    "I am coordinating a pool of agents to solve a problem. The following is the problem and the accumulated responses from some agents."
    "\n<history>\n{history}\n</history>\n"
    "Solving the problem with fewer agents is desired. "
    "Based on these, what do you think is the next step for the next agent? "
    "You should think and be creative in your suggestion, and should put your suggestion in the <suggestion> tag. "
    "E.g., <suggestion>Write python code to solve this problem.</suggestion>. "
    )

    DEFAULT_CONSULT_ASSISTANT_PROMPT ="Let me analyze the problem and the accumulated responses and provide a suggestion for the next agent.\n<think>"

    DEFAULT_COLLABORATION_PROMPT =(
    "You may see other agents' thoughts in <Agent i> tags, where i is the id of the agent. In that case, "
    "you should try to solve the problem based on the information provided by other agents."
    )


    DEFAULT_STRUCTURED_SYSTEM_PROMPT =(
    "You are a task coordinator whose job is to organize {num_agents} agents to solve a problem."
    )

    DEFAULT_STRUCTURED_ROUTER_PROMPT ="""The problem is in <problem> tags.
Your scheduling history is in <step k> tags, where k is the step number.
Inside each <step k> tag, there are <agent>, <description>, and <response> tags,
where <agent> is the id of the agent selected, <description> is the task description to the agent provided by you,
and <response> is that agent's response.

Do not try to solve the problem yourself, instead read the agents' responses and leverage their specialized knowledge to solve the problem.
After observing the <problem> and <step k> tags, you should fill in the <agent> and <description> tags for <step k+1> to decide whether to return or specify an agent to respond next.
If you want to return the latest response to the user as the answer, just put RETURN in <description> tags and don't specify an agent.
However, if you decide to query an agent, provide a task description to the next agent in <description> tags and specify the agent in <agent> tags.
You can put free text in the <description> tags, but the <agent> tag should be a number between 1 and {num_agents} where 1 is the first agent, 2 is the second agent, etc.
In order for the next agent to work efficiently, you should provide a detailed and constructive task description.

The following are examples:

Example 1 input: You observe only the problem.
<problem>
[Problem description here]
</problem>

Example 1 output: You decide to ask an agent to solve the problem.
<step 1>
<description>
[Your task description to the agent here]
</description>
<agent>
[Agent id here, must be between 1 and {num_agents}]
</agent>

Example 2 input: You observe the problem and the first step.
<problem>
[Problem description here]
</problem>
<step 1>
<description>
[Your task description to the agent here]
</description>
<agent>
[Agent id here, must be between 1 and {num_agents}]
</agent>
<response>
[The specified agent's response here]
</response>
</step 1>

Example 2 output: You decide to return the agent's response as the final answer.
<step 2>
<description>
RETURN
</description>
</step 2>

Now, solve the following problem:

{problem_description}
"""

    DEFAULT_STRUCTURED_STEP_PROMPT ="""<step {step_num}>
<description>
{description}
</description>
<agent>
{agent_id}
</agent>
<response>
{response}
</response>
</step {step_num}>
"""

    def __init__ (
    self ,
    llm_names :List [str ],
    seed :int =42 ,
    max_tokens :int =512 ,
    temperature :float =0.8 ,
    max_turns :int =5 ,
    servers :Optional [Dict [str ,str ]]=None ,
    ports :Optional [Dict [str ,int ]]=None ,
    track_costs :bool =True ,
    debug :bool =False ,
    together :bool =True ,
    valid_ratio :float =0.5 ,
    max_samples :int =-1 ,
    use_structured_router :bool =False ,
    test_ratio :float =0.2 ,
    use_consultant :bool =True ,
    log_dir :Optional [str ]=None ,
    ):

        self .split_seed =seed 
        self .np_random =np .random .RandomState (seed =seed )
        self .llm_names =llm_names 
        self .max_tokens =max_tokens 
        self .temperature =temperature 
        self .max_turns =max_turns 
        self .track_costs =track_costs 
        self .debug =debug 
        self .together =together 
        self .valid_ratio =valid_ratio 
        self .test_ratio =test_ratio 
        self .max_samples =max_samples 
        self .use_structured_router =use_structured_router 
        self .use_consultant =use_consultant 
        self .log_dir =log_dir 


        if self .use_structured_router :

            self .system_prompt =self ._get_structured_system_prompt ()
            self .router_prompt =self ._get_structured_router_prompt ()
            self .step_prompt =self ._get_structured_step_prompt ()
        else :

            self .system_prompt =self ._get_system_prompt ()
            self .user_prompt_template =self ._get_user_prompt_template ()
            self .assistant_prompt =self ._get_assistant_prompt ()
            self .router_system_prompt =self ._get_router_system_prompt ()
            self .collaboration_prompt =self ._get_collaboration_prompt ()


            if self .use_consultant :
                self .consult_prompt =self ._get_consult_prompt ()
                self .consult_assistant_prompt =self ._get_consult_assistant_prompt ()


        self .data_splits =None 
        self .data_split =None 
        self .task_id =0 
        self .messages =[]
        self ._obs =None 
        self .obs_act =[]
        self .num_turns =0 
        self .prev_agent_id =-1 
        self .response =None 
        self .current_task_type =None 


        if self .use_consultant :
            self .pending_suggestion =None 


        if self .use_structured_router :
            self .problem_description =""
            self .history =""
            self .latest_response =""


        if servers is None :
            self .servers ={k :None for k in llm_names }
        else :
            self .servers =servers 
        if ports is None :
            self .ports ={k :None for k in llm_names }
        else :
            self .ports =ports 


        if self .track_costs :
            reset_costs ()

    def _get_or_make_splits (
    self ,
    raw_dataset_len :int ,
    task_name :str 
    )->Dict [str ,List [int ]]:

        return get_or_create_indices (
        task_name =task_name ,
        dataset_len =raw_dataset_len ,
        seed =self .split_seed ,
        valid_ratio =self .valid_ratio ,
        test_ratio =self .test_ratio 
        )


    def _get_router_system_prompt (self )->str :

        return self .DEFAULT_ROUTER_SYSTEM_PROMPT 

    def _get_system_prompt (self )->str :

        return self .DEFAULT_SYSTEM_PROMPT 

    @abstractmethod 
    def _get_user_prompt_template (self )->str :

        pass 

    def _get_assistant_prompt (self )->str :

        return f"Let me solve this step by step.\n<{self.thinking_tag}>"

    def _get_collaboration_prompt (self )->str :

        return self .DEFAULT_COLLABORATION_PROMPT 

    def _get_consult_prompt (self )->str :

        if not self .use_consultant :
            raise NotImplementedError ("Consultant feature is disabled")
        return self .DEFAULT_CONSULT_PROMPT 

    def _get_consult_assistant_prompt (self )->str :

        if not self .use_consultant :
            raise NotImplementedError ("Consultant feature is disabled")
        return f"Let me analyze the problem and the accumulated responses and provide a suggestion for the next agent.\n<{self.thinking_tag}>"


    def _get_structured_system_prompt (self )->str :

        return self .DEFAULT_STRUCTURED_SYSTEM_PROMPT 

    def _get_structured_router_prompt (self )->str :

        return self .DEFAULT_STRUCTURED_ROUTER_PROMPT 

    def _get_structured_step_prompt (self )->str :

        return self .DEFAULT_STRUCTURED_STEP_PROMPT 

    def _format_user_prompt (self ,task_data :Dict )->str :


        prompt =self ._format_base_prompt (task_data )


        if not self .use_structured_router :

            if len (self .llm_names )>1 and self .max_turns >1 :
                collaboration_prompt =self ._get_collaboration_prompt ()
                if collaboration_prompt :
                    prompt +="\n"+collaboration_prompt 

        return prompt 

    @abstractmethod 
    def _format_base_prompt (self ,task_data :Dict )->str :

        pass 

    @abstractmethod 
    def _load_data (self ,seed :int ,split :str ="train",validation :bool =False ,
    valid_ratio :float =None ,max_samples :int =None ,
    test_split :bool =False ,test_ratio :float =None )->Dict [str ,Dataset ]:

        pass 

    @abstractmethod 
    def _calculate_reward (self ,completions :List [str ],task_data :List [Dict ],debug :bool =False )->List [float ]:

        pass 

    def seed (self ,seed :int ):

        self .np_random =np .random .RandomState (seed =seed )

    def _get_obs (self ,agent_id :Optional [int ]=None ,response :str ="")->List [Dict ]:

        if response and agent_id is not None :


            agent_thoughts =response .replace (f"<{self.thinking_tag}>","").replace (f"</{self.thinking_tag}>","")






            if len (self .llm_names )>1 and self .max_turns >1 :
                message =f"<Agent {agent_id}>{agent_thoughts}</Agent {agent_id}>"
                self .messages [-1 ]["content"]+="\n"+message 


        if self .use_consultant :

            plan =(
            self .pending_suggestion 
            if self .pending_suggestion is not None 
            else "solve the problem based on the accumulated responses."
            )
            if plan and plan [0 ].isupper ():
                plan =plan [0 ].lower ()+plan [1 :]
            return (
            [msg .copy ()for msg in self .messages ]+
            [{"role":"assistant",
            "content":f"I'll return the latest agent's answer if it is correct, otherwise I'll ask the next agent to {plan}"}]
            )
        else :

            return [msg .copy ()for msg in self .messages ]

    def _format_router_messages (self )->List [Dict ]:

        if self .use_structured_router :

            return [
            {
            "role":"system",
            "content":self .system_prompt .format (num_agents =len (self .llm_names ))
            },
            {
            "role":"user",
            "content":self .router_prompt .format (
            problem_description =self .problem_description ,
            num_agents =len (self .llm_names ),
            )+self .history 
            }
            ]
        else :

            router_messages =[
            {"role":"system","content":self .router_system_prompt .format (num_agents =len (self .llm_names ))},
            ]


            if len (self .messages )>1 :
                router_messages .append (self .messages [1 ])

            return router_messages 

    def _format_agent_messages (self ,agent_id :int )->List [Dict ]:

        if self .use_structured_router :

            description =""
            if hasattr (self ,'_latest_description'):
                description =self ._latest_description 

            return [
            {
            "role":"user",
            "content":self .problem_description +self .history +f"\n{description}\n"
            }
            ]
        else :

            agent_messages =[
            {"role":"system","content":self .system_prompt },
            ]


            if len (self .messages )>1 :
                agent_messages .append (self .messages [1 ])


            if self .use_consultant and hasattr (self ,'pending_suggestion')and self .pending_suggestion is not None :
                agent_messages [-1 ]["content"]=(
                agent_messages [-1 ]["content"]+
                f"\n<suggestion>\n{self.pending_suggestion}\n</suggestion>"
                )


            agent_messages .append ({"role":"assistant","content":self .assistant_prompt })

            return agent_messages 

    def _format_consult_messages (self ,agent_id :int )->List [Dict ]:

        if not self .use_consultant :
            raise NotImplementedError ("Consultant feature is disabled")

        if self .use_structured_router :
            raise NotImplementedError ("Structured router is not supported for consultant agents")
        else :
            agent_messages =[
            {"role":"system","content":self .system_prompt },
            {"role":"user","content":self .consult_prompt },
            ]

            if len (self .messages )>1 :
                agent_messages [-1 ]["content"]=(
                agent_messages [-1 ]["content"].format (
                history =self .messages [1 ]["content"])
                )

            agent_messages .append ({"role":"assistant","content":self .consult_assistant_prompt })

            return agent_messages 


    def _get_job_description (self ,response :str )->str :

        if "<description>"in response and "</description>"in response :
            res =response .split (
            "<description>"
            )[-1 ].split (
            "</description>"
            )[0 ].strip ()

            self ._latest_description =res 
            return res 
        else :
            self ._latest_description ="RETURN"
            return "RETURN"

    def _get_next_agent (self ,response :str )->int :

        if "<agent>"in response and "</agent>"in response :
            res =response .split ("<agent>")[-1 ].split ("</agent>")[0 ].strip ()
        else :
            res =""
        try :
            return int (res )
        except Exception :
            return -1 

    def reset (self ,task_id :int =-1 ,split :str ="train"):


        if self .track_costs :
            reset_costs ()


        if self .data_splits is None :
            self .data_splits =self ._load_data (
            seed =self .split_seed ,
            split =split ,
            validation =True ,
            valid_ratio =self .valid_ratio ,
            max_samples =self .max_samples 
            )

        self .data_split =self .data_splits [split ]
        if task_id <0 :
            self .task_id =self .np_random .randint (0 ,len (self .data_split ))
        else :
            self .task_id =task_id 


        if self .use_structured_router :

            self .problem_description =self ._format_base_prompt (self .data_split [self .task_id ])
            self .num_turns =0 
            self .history =""
            self .latest_response =""


            self ._obs =self ._format_router_messages ()
            self .obs_act =[]
            self .prev_agent_id =-1 
            self .response =None 

            return self ._obs 
        else :

            user_prompt =self ._format_user_prompt (self .data_split [self .task_id ])



            if self .max_turns >1 and len (self .llm_names )>1 :

                self .messages =[
                {"role":"system","content":self .router_system_prompt .format (num_agents =len (self .llm_names ))},
                {"role":"user","content":user_prompt },
                ]
            else :

                self .messages =[
                {"role":"system","content":self .system_prompt },
                {"role":"user","content":user_prompt },
                {"role":"assistant","content":self .assistant_prompt }
                ]


            self ._obs =self ._get_obs ()
            self .obs_act =[]
            self .num_turns =0 
            self .prev_agent_id =-1 
            self .response =None 


            if self .use_consultant :
                self .pending_suggestion =None 

            return self ._obs 

    def grab_data (self ,split :str ="train")->Dict :

        if self .data_splits is None :
            raise ValueError ("Data splits not initialized. Call reset() first.")
        return self .data_splits [split ]

    def step (self ,
    action :Union [np .ndarray ,str ],
    sampling :bool =False ,
    preselected_agent_id :Optional [int ]=None ,
    consult_flag :np .ndarray =None ):


        if not self .use_consultant and consult_flag is not None :

            consult_flag =None 

        consulted =False 


        if self .use_structured_router :

            pass 
        else :

            assert isinstance (action ,np .ndarray ),"Standard router requires numpy array action"
            assert len (action )==len (self .llm_names ),(
            f"Size mismatch: {len(action)} vs {len(self.llm_names)}"
            )


            if preselected_agent_id is not None :
                agent_id =preselected_agent_id 
            elif sampling :
                agent_id =self .np_random .choice (range (len (action )),p =action )
            else :
                agent_id =np .argmax (action )


            exceed_max_turns =self .num_turns >=self .max_turns 
            return_answer =agent_id ==self .prev_agent_id 


            done =exceed_max_turns or return_answer 

            if done and return_answer :


                if self .debug :
                    print (f"=== DEBUG: Consecutive selection of Agent {agent_id} ===")
                    print (f"Using previous response (skipping redundant query)")
                    print ("===========================")


                self .num_turns +=1 
                self .obs_act .append ((self ._obs ,agent_id ))
                response =""


                reward =self ._calculate_reward (
                [self .response ],
                [self .data_split [self .task_id ]],
                debug =self .debug ,
                )
                reward =reward [0 ]


                if self .track_costs and self .debug :
                    self .print_cost_summary ()
            elif done and exceed_max_turns :

                if self .debug :
                    print (f"=== DEBUG: Maximum turns ({self.max_turns}) already reached ===")
                    print ("===========================")

                response =""
                reward =self ._calculate_reward (
                [self .response ],
                [self .data_split [self .task_id ]],
                debug =self .debug ,
                )
                reward =reward [0 ]

                if self .track_costs and self .debug :
                    self .print_cost_summary ()
            else :

                reward =0.0 
                agent_name =self .llm_names [agent_id ]


                if self .debug :
                    print (f"=== DEBUG: Selected Agent ===")
                    print (f"Agent ID: {agent_id}, Model: {agent_name}")
                    print ("===========================")


                if self .max_turns >1 and len (self .llm_names )>1 :

                    if self .use_consultant :

                        if consult_flag is None :
                            use_consultant =False 
                        else :
                            use_consultant =consult_flag >0 
                        if hasattr (self ,'pending_suggestion')and self .pending_suggestion is not None :
                            use_consultant =False 

                        if use_consultant :
                            agent_messages =self ._format_consult_messages (agent_id )
                        else :
                            agent_messages =self ._format_agent_messages (agent_id )
                    else :

                        agent_messages =self ._format_agent_messages (agent_id )

                    self .response =response =query_llm (
                    model =agent_name ,
                    messages =agent_messages ,
                    max_tokens =self .max_tokens ,
                    temperature =self .temperature ,
                    server =self .servers .get (agent_name ),
                    port =self .ports .get (agent_name ),
                    debug =self .debug ,
                    together =self .together ,
                    )


                    if self .use_consultant :
                        self .pending_suggestion =None 
                        if use_consultant :
                            try :
                                self .pending_suggestion =(
                                response .split ("<suggestion>")[1 ].split ("</suggestion>")[0 ].strip ()
                                )
                                consulted =True 
                            except :
                                self .pending_suggestion =None 
                else :

                    self .response =response =query_llm (
                    model =agent_name ,
                    messages =self .messages ,
                    max_tokens =self .max_tokens ,
                    temperature =self .temperature ,
                    server =self .servers .get (agent_name ),
                    port =self .ports .get (agent_name ),
                    debug =self .debug ,
                    together =self .together ,
                    )


                if self .debug :
                    print (f"=== DEBUG: Agent Response ===")
                    print (response )
                    print ("===========================")

                self .num_turns +=1 
                self .obs_act .append ((self ._obs ,agent_id ))
                self .prev_agent_id =agent_id 


                if self .num_turns >=self .max_turns :
                    done =True 


                    if done :
                        reward =self ._calculate_reward (
                        [self .response ],
                        [self .data_split [self .task_id ]],
                        debug =self .debug ,
                        )
                        reward =reward [0 ]


            if self .use_consultant :
                self ._obs =self ._get_obs (agent_id ,""if consulted else response )
            else :
                self ._obs =self ._get_obs (agent_id ,response )

            return self ._obs ,reward ,done ,self .obs_act 

    def batch_process (self ,split :str ="test",max_samples :int =-1 ,
    agent_policy :str ="round_robin",output_dir :Optional [str ]=None ):


        if output_dir :
            os .makedirs (output_dir ,exist_ok =True )


        if self .data_splits is None :
            self .data_splits =self ._load_data (
            seed =self .np_random .randint (0 ,10000 ),
            split =split ,
            validation =False 
            )

        self .data_split =self .data_splits [split ]

        num_samples =len (self .data_split )if max_samples <=0 else min (max_samples ,len (self .data_split ))
        rewards =[]
        responses =[]

        start_time =time .time ()


        for i in tqdm (range (num_samples ),desc =f"Processing {split} samples"):
            self .reset (task_id =i ,split =split )
            done =False 
            turn =0 

            while not done :

                if self .use_structured_router :

                    if agent_policy =="round_robin":

                        agent_to_use =(turn %len (self .llm_names ))+1 
                        action =f"<step {turn+1}>\n<description>\nSolve this problem step by step.\n</description>\n<agent>\n{agent_to_use}\n</agent>\n</step {turn+1}>"
                    elif agent_policy =="random":

                        agent_to_use =self .np_random .randint (1 ,len (self .llm_names )+1 )
                        action =f"<step {turn+1}>\n<description>\nSolve this problem step by step.\n</description>\n<agent>\n{agent_to_use}\n</agent>\n</step {turn+1}>"
                    else :

                        action =f"<step {turn+1}>\n<description>\nSolve this problem step by step.\n</description>\n<agent>\n1\n</agent>\n</step {turn+1}>"
                else :

                    if agent_policy =="round_robin":
                        action =np .zeros (len (self .llm_names ))
                        action [turn %len (self .llm_names )]=1 
                    elif agent_policy =="random":
                        action =self .np_random .rand (len (self .llm_names ))
                    else :
                        action =np .zeros (len (self .llm_names ))
                        action [0 ]=1 

                _ ,reward ,done ,_ =self .step (action )
                turn +=1 

            rewards .append (reward )
            responses .append (self .response )

            if (i +1 )%10 ==0 or i ==num_samples -1 :
                elapsed =time .time ()-start_time 
                avg_reward =np .mean (rewards [:i +1 ])
                tqdm .write (f"Processed {i + 1}/{num_samples} problems. "
                f"Current average reward: {avg_reward:.4f} | "
                f"Time elapsed: {elapsed:.2f}s")


                if output_dir :
                    self ._save_results (output_dir ,rewards [:i +1 ],responses [:i +1 ],
                    split ,agent_policy ,elapsed ,i +1 )

        avg_reward =np .mean (rewards )
        elapsed_time =time .time ()-start_time 

        print (f"Completed {num_samples} problems with average reward: {avg_reward:.4f} "
        f"in {elapsed_time:.2f} seconds")


        if output_dir :
            self ._save_results (output_dir ,rewards ,responses ,split ,
            agent_policy ,elapsed_time ,num_samples )

        return avg_reward ,rewards ,responses 

    def _save_results (self ,output_dir :str ,rewards :List [float ],responses :List [str ],
    split :str ,agent_policy :str ,elapsed_time :float ,num_samples :int ):


        results_data =[]
        for i in range (len (rewards )):
            result ={
            "index":i ,
            "task_data":{k :v for k ,v in self .data_splits [split ][i ].items ()},
            "response":responses [i ],
            "reward":float (rewards [i ]),
            "config":{
            "models":self .llm_names ,
            "max_turns":self .max_turns ,
            "max_tokens":self .max_tokens ,
            "temperature":self .temperature ,
            "agent_policy":agent_policy ,
            "debug":self .debug ,
            "together":self .together ,
            "use_structured_router":self .use_structured_router if hasattr (self ,'use_structured_router')else False ,
            "use_consultant":self .use_consultant if hasattr (self ,'use_consultant')else True 
            }
            }
            results_data .append (result )


        with open (f"{output_dir}/detailed_results.jsonl","w")as f :
            for result in results_data :
                f .write (json .dumps (result )+"\n")


        summary ={
        "models":self .llm_names ,
        "avg_reward":float (np .mean (rewards )),
        "num_samples":num_samples ,
        "split":split ,
        "agent_policy":agent_policy ,
        "max_turns":self .max_turns ,
        "max_tokens":self .max_tokens ,
        "temperature":self .temperature ,
        "execution_time_seconds":elapsed_time ,
        "timestamp":time .strftime ("%Y-%m-%d %H:%M:%S"),
        "debug":self .debug ,
        "together":self .together ,
        "use_structured_router":self .use_structured_router if hasattr (self ,'use_structured_router')else False ,
        "use_consultant":self .use_consultant if hasattr (self ,'use_consultant')else True 
        }


        if self .track_costs :
            cost_summary =get_cost_summary ()
            summary ["cost_summary"]=cost_summary 
            summary ["cost_per_problem"]=cost_summary ["total_cost"]/num_samples if num_samples >0 else 0 

        with open (f"{output_dir}/summary.json","w")as f :
            json .dump (summary ,f ,indent =2 )

    def print_cost_summary (self ):

        if not self .track_costs :
            return 

        cost_summary =get_cost_summary ()

        if self .debug :
            print ("\n==== Cost Summary ====")
            for model_summary in cost_summary ["models"]:
                model =model_summary ["model"]
                cost =model_summary ["total_cost"]
                tokens =model_summary ["total_tokens"]
                input_tokens =model_summary ["total_input_tokens"]
                output_tokens =model_summary ["total_output_tokens"]
                queries =model_summary ["queries"]

                print (f"Model: {model}")
                print (f"  Queries: {queries}")
                print (f"  Tokens: {tokens} (Input: {input_tokens}, Output: {output_tokens})")
                print (f"  Cost: ${cost:.6f}")

            print (f"\nTotal Cost Across All Models: ${cost_summary['total_cost']:.6f}")
            print ("====================")
        else :

            print (f"Total Cost: ${cost_summary['total_cost']:.6f}")