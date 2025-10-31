import re 
from typing import List ,Dict ,Optional ,Sequence ,Tuple 
import unicodedata 
import ast 
import json 
import torch 
from tenacity import retry ,stop_after_attempt ,wait_exponential 
from transformers import AutoTokenizer 
import torch .nn as nn 

from llm_clients import (query_oai ,query_locally_hosted_model ,query_anthropic ,query_gemini ,query_deepseek )

_SMART_QUOTES =str .maketrans ("“”‘’",'""\'\'')

def _is_oai_model (model :str )->bool :
    return "gpt"in model .lower ()


def _is_anthropic_model (model :str )->bool :
    return "claude"in model .lower ()


def _is_gemini_model (model :str )->bool :
    return "gemini"in model .lower ()

def _is_deepseek_model (model :str )->bool :
    return "deepseek"in model .lower ()

def _print_retry_message (retry_state ):
    print (
    f"Retrying attempt {retry_state.attempt_number}/15 "
    +f"after error: {retry_state.outcome.exception()}"
    )


@retry (
stop =stop_after_attempt (5 ),
wait =wait_exponential (multiplier =2 ,min =2 ,max =30 ),
reraise =True ,
before_sleep =_print_retry_message ,
)
def query_llm (model_name ,messages ,max_tokens ,temperature ,server ,port ,gemini_thinking_budget =1024 ,claude_thinking_budget =1024 ,anthropic_platform ='bedrock',
gpt_reasoning_effort ='minimal'):


        return _query_llm (
        model =model_name ,
        messages =messages ,
        max_tokens =max_tokens ,
        temperature =temperature ,
        server =server ,
        port =port ,
        gemini_thinking_budget =gemini_thinking_budget ,
        claude_thinking_budget =claude_thinking_budget ,
        anthropic_platform =anthropic_platform ,
        gpt_reasoning_effort =gpt_reasoning_effort 
        )


def _query_llm (
model :str ,
messages :List [Dict ],
max_tokens :int ,
temperature :float ,
server =None ,
port =None ,
gemini_thinking_budget =1024 ,
claude_thinking_budget =1024 ,
anthropic_platform ='bedrock',
gpt_reasoning_effort ='minimal'
)->str :
    if server is None or port is None :

        if _is_oai_model (model ):
            response =query_oai (model ,messages ,max_tokens ,temperature ,reasoning_effort =gpt_reasoning_effort )
        elif _is_anthropic_model (model ):
            response =query_anthropic (model ,messages ,max_tokens ,temperature ,platform =anthropic_platform ,claude_thinking_budget =claude_thinking_budget )
        elif _is_gemini_model (model ):
            response =query_gemini (model ,messages ,max_tokens ,temperature ,gemini_thinking_budget )
        elif _is_deepseek_model (model ):
            response =query_deepseek (model ,messages ,max_tokens ,temperature )
        else :
            print (f"Unsupported model: {model}")
            response =""
    else :

        if model =="Qwen/Qwen3-32B-Direct":
            payload ={"top_p":0.8 ,
            "top_k":20 ,
            "presence_penalty":1.0 ,
            "chat_template_kwargs":{
            "enable_thinking":False }}

            model ="Qwen/Qwen3-32B"

        elif model =="Qwen/Qwen3-32B-Reasoning":
            payload ={"top_p":0.8 ,
            "top_k":20 ,
            "presence_penalty":1.0 ,
            "chat_template_kwargs":{
            "enable_thinking":True }}

            model ="Qwen/Qwen3-32B"

        else :
            payload ={}

        response =query_locally_hosted_model (
        model ,messages ,max_tokens ,temperature ,server ,port ,**payload 
        )
    return response 

_SUBTASK_ANCHORS =[
r"\bthis\s+subtask\b",
r"\bmy\s+subtask\b",
r"\bthe\s+subtask\b",
r"\bsubtask\b",
r"\bsub\-task\b",
r"\b(sub)?task\s+instruction(s)?\b",
r"\bsubtask\s+requirement(s)?\b",
]


_CRITIQUE_TERMS =[
r"misunderstand(?:ing|s|ed)?",
r"misstat(?:ement|ed|es)",
r"mislead(?:ing|s|e)?",
r"\bwrong\b",
r"\bincorrect\b",
r"\birrelevant\b",
r"\bnot\s+useful\b",
r"\bdoes\s+not\s+(match|align|correspond)\b",
r"\bmismatch\b",
r"\boff[-\s]?track\b",
r"\bbased\s+on\s+(a\s+)?wrong\s+assumption\b",
r"\bpoorly\s+(formed|specified)\b",
]


_PIPELINE_ONLY_HINTS =re .compile (
r"\b(agent|approach|solution|response|code|implementation|previous\s+agent|agent\s+\d+|prior\s+work|their\s+approach)\b",
re .IGNORECASE ,
)

def _split_sentences (text :str )->List [str ]:

    return re .split (r'(?<=[.!?])\s+|\n+',text )

def contains_subtask_misspecification (response :str ,proximity_chars :int =120 )->bool :

    if not response :
        return False 


    anchor_pat =re .compile ("|".join (_SUBTASK_ANCHORS ),re .IGNORECASE )
    crit_pat =re .compile ("|".join (_CRITIQUE_TERMS ),re .IGNORECASE )

    for sent in _split_sentences (response ):
        if not sent .strip ():
            continue 

        has_anchor =anchor_pat .search (sent )
        has_crit =crit_pat .search (sent )


        if not has_anchor or not has_crit :
            continue 



        a =has_anchor .span ()[0 ]
        c =has_crit .span ()[0 ]
        if abs (a -c )>proximity_chars :
            continue 





        return True 

    return False 

def _is_all_access (x )->bool :

    if isinstance (x ,str ):
        return x .strip ().lower ()=="all"
    if isinstance (x ,(list ,tuple ,set )):
        return any (
        isinstance (elem ,str )and elem .strip ().lower ()=="all"
        for elem in x 
        )
    return False 

def setup_task_data (task_data :Dict [str ,List ],idx :int )->Dict [str ,str ]:

    return {key :value [idx ]if isinstance (value ,list )else value for key ,value in task_data .items ()}

def check_all_integers (lst ):

    n =len (lst )
    allowed =set (range (max (n -1 ,0 )))
    found =set ()

    for sub in lst :

        if not isinstance (sub ,(list ,tuple ,set )):
            return False 
        if not all (isinstance (x ,int )for x in sub ):
            return False 
        found .update (sub )


    if not found .issubset (allowed ):
        return False 


    return allowed .issubset (found )

def _check_ctop_access_list (agent_access ,access_list ,available_models :List [str ],ctop_type :str )->bool :



    if ctop_type =="choose_position":
        try :
            reference_error =not check_all_integers (access_list )
        except :
            return False 
        if reference_error :
            return False 



    if agent_access ==[]:
        return True 


    if isinstance (agent_access ,list ):
        for access_item in agent_access :

            try :
                if isinstance (access_item ,int ):
                    access_int =access_item 
                elif isinstance (access_item ,str )and access_item .isdigit ():
                    access_int =int (access_item )
                else :
                    return False 


                if ctop_type =="choose_id":
                    if access_int <0 or access_int >=len (available_models ):
                        return False 
                elif ctop_type =="choose_position":
                    if access_int <0 or access_int >=len (access_list ):
                        return False 
                else :
                    raise ValueError (f"Invalid complex topology type: {ctop_type}")
            except :
                return False 


        return True 


    return False 

def _match_name (x ,random_names ):

    if isinstance (x ,int ):
        raise ValueError (f"Invalid name. Int provided: {x}")

    if isinstance (x ,str ):
        x_cf =x .casefold ()
        for idx ,name in enumerate (random_names ):
            if name .casefold ()==x_cf :
                return idx 
    else :
        raise ValueError (f"Invalid name provided, neither int nor str: {x}")

def _balanced_list (after :str )->str |None :
        depth =0 
        start_idx =None 
        in_quote =None 
        escape_next =False 

        for i ,ch in enumerate (after ):
            if escape_next :
                escape_next =False 
                continue 

            if ch =="\\":
                escape_next =True 
                continue 


            if in_quote :
                if ch ==in_quote :
                    in_quote =None 
                continue 
            elif ch in "\"'":
                in_quote =ch 
                continue 


            if ch =="[":
                if depth ==0 :
                    start_idx =i 
                depth +=1 
            elif ch =="]":
                depth -=1 
                if depth ==0 and start_idx is not None :
                    return after [start_idx :i +1 ]

        return None 

def _extract_any (text :str ,labels :Sequence [str ])->List :
        tag_regex ="|".join (re .escape (lbl )for lbl in labels )
        m =re .search (rf"({tag_regex})\s*[:=]\s*",text ,re .I )
        if not m :
            return []

        raw =_balanced_list (text [m .end ():])
        if not raw :
            return []

        raw =raw .translate (_SMART_QUOTES ).strip ()


        try :
            return ast .literal_eval (raw )
        except Exception :
            pass 

        try :
            return json .loads (re .sub (r"'",'"',raw ))
        except Exception :
            pass 

        items =[x .strip (" \"'")for x in raw .strip ("[]").split (",")if x .strip ()]
        return [int (x )if x .isdigit ()else x for x in items ]

def _check_all_access (access_list ):

        all_access_local_cnt =0 
        for agent_access in access_list :
            if _is_all_access (agent_access ):
                all_access_local_cnt +=1 
        return all_access_local_cnt 

def _calculate_cost_bonus (model_ids :List [int ],agent_messages_list :List ,agent_responses_list :List ,
cost_bonus_weight :float ,llm_names :List [str ])->Tuple [float ,float ]:

    if not model_ids :
        return 0.0 ,0.0 

    from .cost .calculator import Calculator 
    total_cost =0.0 
    for i ,model_id in enumerate (model_ids ):
        agent_model =llm_names [model_id ]
        if 'Distill'in agent_model :
            continue 


        if i <len (agent_messages_list )and i <len (agent_responses_list ):
            input_messages =agent_messages_list [i ]
            output_response =agent_responses_list [i ]


            calc =Calculator (agent_model )
            agent_cost =calc .calculate_total_cost (input_messages ,output_response ,form ="formatted")




            agent_cost *=1000 
            total_cost +=agent_cost 

    return -total_cost *cost_bonus_weight ,total_cost /1000 

def _ascribe_history_binary (history :Dict ,access_list :List )->List [int ]:

    current_idx =len (history ["agent_responses"])
    agent_access =access_list [current_idx ]

    if not history ["agent_responses"]:
        if _is_all_access (agent_access ):

            agent_access =[]


    if _is_all_access (agent_access ):
        agent_access ="all"
        indices =range (len (history ["agent_responses"]))

    else :
        if agent_access ==[]:
            indices =[]
        else :
            raise ValueError (f'Invalid agent access. Binary agent access in use but recieved unsupported agent access: {agent_access!r}.')

    return indices 

def _ascribe_history_complex (history :Dict ,access_list :List )->List [int ]:

    current_idx =len (history ["agent_responses"])
    if current_idx ==0 :

        return []
    if access_list [current_idx ]==[]:

        return []







    visible :List [List [int ]]=[]
    visible_history :Dict [int ,List [int ]]={}
    model_ids =history ["model_ids"]

    for i ,(mid ,allowed )in enumerate (zip (model_ids ,access_list )):
        current_visible :List [int ]=[]

        for target in dict .fromkeys (allowed ):




            if target not in visible_history :

                if target ==mid :
                    continue 

                raise ValueError (
                f"Invalid complex topology: model {mid} tries to see future / nonexistent output of model {target}\n"
                f"Access list: {access_list}\n"
                f"Model IDs: {history['model_ids']}\n"
                )

            current_visible .extend (visible_history [target ])


        current_visible .sort ()
        visible .append (current_visible )


        visible_history .setdefault (mid ,[]).append (i )


    return visible [current_idx ]

def _ascribe_history_positional_complex (history :Dict ,access_list :List )->List [int ]:

    current_idx =len (history ["agent_responses"])
    if current_idx ==0 :

        return []
    if access_list [current_idx ]==[]:

        return []

    N =len (access_list )
    visible :List [List [int ]]=[]

    for i ,allowed in enumerate (access_list ):

        for t in allowed :
            if not isinstance (t ,int ):
                raise TypeError (
                f"Step {i}: access_list must contain integers (got {type(t).__name__}: {t!r})."
                )


        unique_allowed =list (dict .fromkeys (allowed ))
        current :List [int ]=[]
        for pos in unique_allowed :
            if pos <0 or pos >=N :
                raise ValueError (
                f"Step {i}: position {pos} is out of range [0, {N-1}]."
                f"WARNING: This should not execute as we check for valid access list in _check_parsing via _check_ctop_access_list."
                )
            if pos >=i :

                raise ValueError (
                f"Invalid complex topology: tried to see future output from position {pos}.\n"
                f"Access list: {access_list}\n"
                f"Model IDs: {history['model_ids']}\n"
                )
            current .append (pos )

        current .sort ()
        visible .append (current )


    return visible [current_idx ]

def _write_log (file_path ,task_type ,base_question ,router_completion ,subtasks ,model_ids ,access_list ,messages ,agent_responses ,available_models ,correct_answer ="N/A",recursion_prev_history ={}):
    with open (file_path ,"a",encoding ="utf-8")as f :
        f .write ("\n\n==================== NEW COMPLETION ====================\n")
        f .write (f"--- Task type ---\n")
        f .write (task_type +"\n\n")
        f .write (f"--- Base question ---\n")
        f .write (base_question +"\n\n")
        f .write (f"--- Correct answer ---\n")
        try :
            f .write (str (correct_answer )+"\n\n")
        except :
            f .write ("N/A"+"\n\n")
        f .write ("--- Router completion ---\n")
        f .write (router_completion +"\n\n")
        f .write (f"Parsed subtasks : {subtasks}\n")
        f .write (f"Model IDs       : {model_ids}\n")
        f .write (f"Access List     : {access_list}\n\n")
        if recursion_prev_history :
            f .write (f"\n v==================== recursion Previous Strategy ====================v \n")
            prev_reward =recursion_prev_history ["reward"]
            f .write (f"Original Reward : {prev_reward}\n")
            prev_subtasks =recursion_prev_history ["subtasks"]
            prev_model_ids =recursion_prev_history ["model_ids"]
            prev_access_list =recursion_prev_history ["access_list"]
            prev_agent_responses =recursion_prev_history ["agent_responses"]
            f .write (f"Prev subtasks : {prev_subtasks}\n")
            f .write (f"Prev model IDs       : {prev_model_ids}\n")
            f .write (f"Prev access List     : {prev_access_list}\n")
            for step ,(prev_resp ,prev_mid )in enumerate (zip (prev_agent_responses ,prev_model_ids )):
                f .write (f"--- Step {step + 1} | Agent {available_models[prev_mid]} RESPONSE ---\n")
                f .write (prev_resp +"\n\n")
            f .write ("^==================== recursion Previous Strategy ====================^ \n")

        for step ,(msg ,resp ,mid )in enumerate (zip (messages ,agent_responses ,model_ids )):
            f .write (f"--- Step {step + 1} | Agent {available_models[mid]} INPUT MESSAGE ---\n")

            f .write (json .dumps (msg ,indent =2 ,ensure_ascii =False )+"\n")
            f .write (f"--- Agent {available_models[mid]} RESPONSE ---\n")
            f .write (resp +"\n\n")

        f .write ("========================================================\n")

