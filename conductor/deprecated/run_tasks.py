import os 
import json 
import time 
import numpy as np 
from tqdm import tqdm 
from guf .utils import batch_completion 
from guf .tasks .countdown import CountdownTask 
from guf .tasks .gsm8k import GSM8KTask 
from guf .tasks .mix_c_g import MixCGTask 
from guf .tasks .mix_c_d import MixCDTask 
from guf .tasks .math500 import MathTask 
from guf .tasks .deepmath import DeepMathTask 
from guf .tasks .medreason import MedReasonTask 
from guf .tasks .arena import SearchArenaTask 
from guf .tasks .countdown_structured import StructuredCountdownTask 
from guf .tasks .mix_c_d_structured import StructuredMixCDTask 
from guf .tasks .lmsys import LMSYSJudgeTask 
from guf .tasks .pfgen import PFGenTask 
from guf .tasks .mix_c_d_m_g import MixCDMGTask 
from guf .tasks .japanese_financial import JapaneseFinancialTask 
from guf .tasks .mmlu import MMLUTask 
from guf .tasks .hhrlhf import HHRLHFTask 
from guf .tasks .emotion import EmotionTask 
from guf .tasks .rlpr import RLPRTask 
from guf .tasks .polaris import PolarisTask 
from guf .tasks .livecodebench import LiveCodeBenchTask 
from guf .tasks .mix_m_c_m_m_l import MixMCMMLTask 
from guf .tasks .mix_m_c_m_r_l import MixMCMRLTask 
from guf .tasks .mix_m_c_m_j_m_r_l import MixMCMJMRLTask 

from guf .core import Task 
from typing import List ,Optional ,Union ,Tuple ,Callable ,Dict ,Set ,Any 


def filter_task_data_for_logging (task_data :Dict [str ,Any ])->Dict [str ,Any ]:


    VERBOSE_FIELDS ={
    "private_test_cases",
    "public_test_cases",
    "metadata",
    "test_cases",
    "input_output",
    "starter_code",
    }


    filtered_data ={}
    for k ,v in task_data .items ():
        if k not in VERBOSE_FIELDS :

            if isinstance (v ,str )and len (v )>1000 :
                filtered_data [k ]=v [:200 ]+"...[truncated]..."+v [-200 :]
            else :
                filtered_data [k ]=v 

    return filtered_data 


def create_task (task_name :str ,*,seed :int =42 ,**kwargs )->Task :


    kwargs_copy =kwargs .copy ()
    kwargs_copy .pop ('resume',None )
    if 'seed'not in kwargs_copy :
        kwargs_copy ['seed']=seed 


    use_structured_router =kwargs_copy .pop ('use_structured_router',False )

    log_dir =kwargs_copy .get ('log_dir',None )

    name =task_name .lower ()
    if name =="countdown":
        if use_structured_router :
            return StructuredCountdownTask (**kwargs_copy )
        return CountdownTask (**kwargs_copy )

    elif name =="structured_countdown":
        kwargs_copy ['use_structured_router']=True 
        return StructuredCountdownTask (**kwargs_copy )

    elif name in ["gsm8k","grade_school_math"]:
        return GSM8KTask (**kwargs_copy )

    elif name in ["mix_c_g","mixcg"]:
        return MixCGTask (**kwargs_copy )
    elif name in ["mix_c_d","mixcd"]:

        if use_structured_router :
            kwargs_copy ['balanced_sampling']=True 
            return StructuredMixCDTask (**kwargs_copy )
        kwargs_copy ['balanced_sampling']=True 
        return MixCDTask (**kwargs_copy )

    elif name in ["mix_c_d_full","mixcd_full"]:

        if use_structured_router :
            kwargs_copy ['balanced_sampling']=False 
            return StructuredMixCDTask (**kwargs_copy )
        kwargs_copy ['balanced_sampling']=False 
        return MixCDTask (**kwargs_copy )

    elif name in ["mix_c_d_structured","mixcd_structured"]:

        kwargs_copy ['balanced_sampling']=True 
        return StructuredMixCDTask (**kwargs_copy )

    elif name in ["mix_c_d_structured_full","mixcd_structured_full"]:

        kwargs_copy ['balanced_sampling']=False 
        return StructuredMixCDTask (**kwargs_copy )

    elif name in ["math500","math-500","math_500"]:
        return MathTask (**kwargs_copy )

    elif name in ["deepmath","deepmath-103k","deepmath_103k"]:
        return DeepMathTask (**kwargs_copy )

    elif name =="medreason":
        return MedReasonTask (**kwargs_copy )

    elif name in ["arena","search_arena","searcharena"]:
        return SearchArenaTask (**kwargs_copy )

    elif name in ["lmsys_judge","lmsys","judge"]:
        kwargs_copy ['use_structured_router']=use_structured_router 
        return LMSYSJudgeTask (**kwargs_copy )

    elif name in ["pfgen","pfgen_bench","preferred_generation"]:
        return PFGenTask (**kwargs_copy )

    elif name in ["japanese_financial","jp_fin","jfin"]:
        return JapaneseFinancialTask (prompt_language ="jp",**kwargs_copy )

    elif name in ["japanese_financial_en","jp_fin_en","jfin_en"]:

        return JapaneseFinancialTask (prompt_language ="en",**kwargs_copy )

    elif name in ["mix_c_d_m_g","mixcdmg"]:

        kwargs_copy ['balanced_sampling']=True 
        return MixCDMGTask (**kwargs_copy )

    elif name in ["mix_c_d_m_g_full","mixcdmg_full"]:

        kwargs_copy ['balanced_sampling']=False 
        return MixCDMGTask (**kwargs_copy )

    elif name in ["mmlu","massive_multitask_language_understanding"]:
        return MMLUTask (**kwargs_copy )

    elif name in ["hh_rlhf","hh-rlhf","hhrlhf","anthropic_hh","preference_task"]:
        return HHRLHFTask (**kwargs_copy )

    elif name in ["emotion","emotion_classification"]:
        return EmotionTask (**kwargs_copy )

    elif name in ["rlpr","rlpr_train","rlpr_dataset"]:
        return RLPRTask (**kwargs_copy )

    elif name in ["polaris","polaris-53k","polaris_53k","polaris_dataset"]:
        return PolarisTask (**kwargs_copy )

    elif name in ["livecodebench","livecodebench_dataset","livecodebench_train"]:
        return LiveCodeBenchTask (**kwargs_copy )

    elif name in ["mix_m_c_m_m_l","mix_five"]:

        kwargs_copy ['balanced_sampling']=True 
        return MixMCMMLTask (**kwargs_copy )

    elif name in ["mix_m_c_m_m_l_full","mix_five_full"]:

        kwargs_copy ['balanced_sampling']=False 
        return MixMCMMLTask (**kwargs_copy )

    elif name in ["mix_m_c_m_r_l","mix_mcmrl"]:

        kwargs_copy ['balanced_sampling']=True 
        return MixMCMRLTask (**kwargs_copy )

    elif name in ["mix_m_c_m_r_l_full","mix_mcmrl_full"]:

        kwargs_copy ['balanced_sampling']=False 
        return MixMCMRLTask (**kwargs_copy )

    elif name in ["mix_m_c_m_j_m_r_l","mix_seven"]:

        kwargs_copy ['balanced_sampling']=True 
        return MixMCMJMRLTask (**kwargs_copy )

    elif name in ["mix_m_c_m_j_m_r_l_full","mix_seven_full"]:

        kwargs_copy ['balanced_sampling']=False 
        return MixMCMJMRLTask (**kwargs_copy )

    else :
        raise ValueError (f"Unknown task: {task_name}")


def _check_for_existing_results (output_dir :str )->Tuple [List [Dict ],Set [int ]]:

    existing_results =[]
    processed_indices =set ()
    results_file =os .path .join (output_dir ,"detailed_results.jsonl")
    if os .path .exists (results_file ):
        try :
            with open (results_file ,"r")as f :
                for line in f :
                    try :
                        result =json .loads (line .strip ())
                        existing_results .append (result )
                        processed_indices .add (int (result ["index"]))
                    except json .JSONDecodeError :
                        continue 
        except Exception as e :
            print (f"Warning: Error reading existing results file: {e}")
    return existing_results ,processed_indices 


def _save_result_to_file (result :Dict [str ,Any ],results_file :str )->None :

    try :
        with open (results_file ,"a")as f :
            f .write (json .dumps (result )+"\n")
    except Exception as e :
        print (f"Warning: Failed to write result to {results_file}: {e}")


def _save_summary (
output_dir :str ,
task :Task ,
rewards :List [float ],
split :str ,
agent_policy :str ,
elapsed_time :float ,
num_samples :int 
)->None :

    summary ={
    "models":task .llm_names ,
    "avg_reward":float (np .mean (rewards ))if rewards else 0.0 ,
    "num_samples":num_samples ,
    "split":split ,
    "agent_policy":agent_policy ,
    "max_turns":task .max_turns ,
    "max_tokens":task .max_tokens ,
    "temperature":task .temperature ,
    "execution_time_seconds":elapsed_time ,
    "timestamp":time .strftime ("%Y-%m-%d %H:%M:%S"),
    "debug":task .debug ,
    "together":task .together ,
    }
    if task .track_costs :
        from guf .cost import get_cost_summary 
        cost_summary =get_cost_summary ()
        summary ["cost_summary"]=cost_summary 
        summary ["cost_per_problem"]=cost_summary ["total_cost"]/num_samples if num_samples >0 else 0 
    try :
        with open (f"{output_dir}/summary.json","w")as f :
            json .dump (summary ,f ,indent =2 )
    except Exception as e :
        print (f"Warning: Failed to write summary to {output_dir}/summary.json: {e}")


def _sample_indices (data_split :List ,max_samples :int ,seed :int =42 )->List [int ]:

    all_indices =list (range (len (data_split )))
    if max_samples !=-1 and max_samples <len (all_indices ):
        np .random .seed (seed )
        sampled_indices =sorted (np .random .choice (all_indices ,size =max_samples ,replace =False ).tolist ())
    else :
        sampled_indices =all_indices 
    return sampled_indices 


def _get_action_for_policy (
agent_policy :str ,
model_names :List [str ],
turn :int ,
task_rng :np .random .RandomState 
)->np .ndarray :

    action =np .zeros (len (model_names ))
    if agent_policy =="round_robin":
        action [turn %len (model_names )]=1 
    elif agent_policy =="random":
        action =task_rng .rand (len (model_names ))
    else :
        action [0 ]=1 
    return action 


def _run_task_with_policy_core (
task :Task ,
model_names :List [str ],
split :str ,
max_samples :int ,
agent_policy :Union [str ,Callable ],
output_dir :str ,
processed_indices :Set [int ],
existing_results :List [Dict ],
seed :int =42 
)->Tuple [List [float ],List [str ]]:

    rewards =[]
    responses =[]
    if existing_results :
        sorted_results =sorted (existing_results ,key =lambda x :x ["index"])
        for result in sorted_results :
            rewards .append (result ["reward"])
            responses .append (result ["response"])
        print (f"Loaded {len(rewards)} existing results with average reward: {np.mean(rewards):.4f}")
    results_file =f"{output_dir}/detailed_results.jsonl"
    sampled_indices =_sample_indices (task .data_split ,max_samples ,seed )
    remaining_indices =[i for i in sampled_indices if int (i )not in processed_indices ]
    if not remaining_indices :
        print ("All samples have already been processed. Nothing to do.")
        return rewards ,responses 
    start_time =time .time ()
    progress_counter =len (processed_indices )
    for i in tqdm (remaining_indices ,desc =f"Processing {split} samples"):
        task_index =int (i )
        task .reset (task_id =task_index ,split =split )
        done =False 
        turn =0 


        use_structured_router =hasattr (task ,'use_structured_router')and task .use_structured_router 

        while not done :
            if use_structured_router :

                if isinstance (agent_policy ,str ):
                    if agent_policy =="round_robin":

                        agent_to_use =(turn %len (model_names ))+1 
                        action =f"<step {turn + 1}>\n<description>\nSolve this problem step by step.\n</description>\n<agent>\n{agent_to_use}\n</agent>\n</step {turn + 1}>"
                    elif agent_policy =="random":

                        agent_to_use =task .np_random .randint (1 ,len (model_names )+1 )
                        action =f"<step {turn + 1}>\n<description>\nSolve this problem step by step.\n</description>\n<agent>\n{agent_to_use}\n</agent>\n</step {turn + 1}>"
                    else :

                        action =f"<step {turn + 1}>\n<description>\nSolve this problem step by step.\n</description>\n<agent>\n1\n</agent>\n</step {turn + 1}>"
                else :

                    action =agent_policy (model_names ,turn )
            else :

                if isinstance (agent_policy ,str ):
                    action =_get_action_for_policy (agent_policy ,model_names ,turn ,task .np_random )
                else :
                    action =agent_policy (model_names ,turn )

            _ ,reward ,done ,_ =task .step (action )
            turn +=1 

        rewards .append (reward )
        responses .append (task .response )
        result ={
        "index":task_index ,
        "task_data":filter_task_data_for_logging ({k :v for k ,v in task .data_split [task_index ].items ()}),
        "response":task .response ,
        "reward":float (reward ),
        "config":{
        "models":task .llm_names ,
        "max_turns":task .max_turns ,
        "max_tokens":task .max_tokens ,
        "temperature":task .temperature ,
        "agent_policy":"custom"if callable (agent_policy )else agent_policy ,
        "debug":task .debug ,
        "together":task .together ,
        "use_structured_router":use_structured_router if hasattr (task ,'use_structured_router')else False 
        }
        }
        _save_result_to_file (result ,results_file )
        progress_counter +=1 
        if progress_counter %10 ==0 or i ==remaining_indices [-1 ]:
            elapsed =time .time ()-start_time 
            avg_reward =np .mean (rewards )
            tqdm .write (f"Processed {progress_counter}/{len(sampled_indices)} problems. "
            f"Current average reward: {avg_reward:.4f} | Time elapsed: {elapsed:.2f}s")
            _save_summary (
            output_dir =output_dir ,
            task =task ,
            rewards =rewards ,
            split =split ,
            agent_policy ="custom"if callable (agent_policy )else agent_policy ,
            elapsed_time =elapsed ,
            num_samples =progress_counter 
            )
    elapsed_time =time .time ()-start_time 
    avg_reward =np .mean (rewards )if rewards else 0.0 
    print (f"Completed {progress_counter} problems with average reward: {avg_reward:.4f}")
    print (f"Results saved to: {os.path.abspath(output_dir)}")
    return rewards ,responses 


def run_task_with_policy (
task_name :str ,
model_names :List [str ],
agent_policy :Union [str ,Callable [[List [str ],int ],np .ndarray ]],
split :str ="test",
max_samples :int =50 ,
seed :int =42 ,
max_tokens :int =1024 ,
temperature :float =0.7 ,
max_turns :int =3 ,
track_costs :bool =True ,
output_dir :Optional [str ]=None ,
debug :bool =False ,
together :bool =True ,
resume :bool =True ,
validation :bool =False ,
valid_ratio :float =0.5 ,
test_ratio :float =0.2 ,
use_structured_router :bool =False ,
**kwargs ,
)->Tuple [float ,List [float ],List [str ]]:

    task =create_task (
    task_name ,
    llm_names =model_names ,
    seed =seed ,
    max_tokens =max_tokens ,
    temperature =temperature ,
    max_turns =max_turns ,
    track_costs =track_costs ,
    debug =debug ,
    together =together ,
    valid_ratio =valid_ratio ,
    test_ratio =test_ratio ,
    use_structured_router =use_structured_router ,
    **{k :v for k ,v in kwargs .items ()if k !='resume'}
    )
    if output_dir is None and task_name :
        base_name =model_names [0 ].replace ("/","-")
        timestamp =time .strftime ("%Y%m%d_%H%M%S")
        output_dir =f"outputs/{task_name.lower()}_{base_name}_{len(model_names)}_models_{timestamp}"
    os .makedirs (output_dir ,exist_ok =True )
    print (f"\nSaving results to: {os.path.abspath(output_dir)}")
    existing_results =[]
    processed_indices =set ()
    if resume :
        existing_results ,processed_indices =_check_for_existing_results (output_dir )
        if existing_results :
            print (f"Found {len(existing_results)} existing results. Resuming from where we left off.")
            if existing_results :
                config =existing_results [0 ].get ("config",{})
                if config .get ("models")!=model_names :
                    print (f"Warning: Existing results used different models: {config.get('models')}")
                if config .get ("max_tokens")!=max_tokens :
                    print (f"Warning: Existing results used different max_tokens: {config.get('max_tokens')}")
                if config .get ("temperature")!=temperature :
                    print (f"Warning: Existing results used different temperature: {config.get('temperature')}")


        multi_split_tasks ={
        "gsm8k","mix_c_g","mixcg",
        "mix_c_d","mixcd",
        "mix_c_d_full","mixcd_full",
        "mix_c_d_m_g","mixcdmg",
        "mix_c_d_m_g_full","mixcdmg_full",
        "mix_m_c_m_m_l","mix_five",
        "mix_m_c_m_m_l_full","mix_five_full",
        "mix_m_c_m_r_l","mix_mcmrl",
        "mix_m_c_m_r_l_full","mix_mcmrl_full",
        "mix_m_c_m_j_m_r_l","mix_seven",
        "mix_m_c_m_j_m_r_l_full","mix_seven_full",
        "math500","math-500","math_500",
        "deepmath","deepmath-103k","deepmath_103k",
        "medreason","arena","search_arena","searcharena",
        "pfgen","pfgen_bench","preferred_generation",
        "japanese_financial","jp_fin","jfin",
        "japanese_financial_en","jp_fin_en","jfin_en",
        "mmlu","massive_multitask_language_understanding",
        "hh_rlhf","hh-rlhf","hhrlhf","anthopic_hh","preference_task",
        "emotion","emotion_classification",
        "rlpr","rlpr_train","rlpr_dataset",
        "polaris","polaris-53k","polaris_53k","polaris_dataset",
        "livecodebench","livecodebench_dataset","livecodebench_train",
        }

    name =task_name .lower ()
    if name in multi_split_tasks :
        task .data_splits =task ._load_data (
        seed =task .np_random .randint (0 ,10000 ),
        split =split ,
        validation =validation ,
        valid_ratio =valid_ratio 
        )
    elif task .data_splits is None :
        task .data_splits =task ._load_data (
        seed =task .np_random .randint (0 ,10000 ),
        split =split ,
        validation =validation ,
        valid_ratio =valid_ratio 
        )
    task .data_split =task .data_splits [split ]
    rewards ,responses =_run_task_with_policy_core (
    task =task ,
    model_names =model_names ,
    split =split ,
    max_samples =max_samples ,
    agent_policy =agent_policy ,
    output_dir =output_dir ,
    processed_indices =processed_indices ,
    existing_results =existing_results ,
    seed =seed 
    )
    avg_reward =np .mean (rewards )if rewards else 0.0 
    print (f"Completed {len(rewards)} problems with average reward: {avg_reward:.4f}")
    print (f"Results saved to: {os.path.abspath(output_dir)}")
    return avg_reward ,rewards ,responses 



MODEL_SPECIFIC_PARAMS ={
'top_p','top_k','presence_penalty','frequency_penalty','repetition_penalty',
'chat_template_kwargs','stop','stop_sequences','response_format',
'logit_bias','user','max_completion_tokens','modalities','prediction',
'audio','stream_options','parallel_tool_calls','service_tier'
}


def run_task_with_batches (
task_name :str ,
model_names :List [str ],
batch_size :int =32 ,
max_concurrency :int =8 ,
split :str ='test',
max_samples :int =1000 ,
seed :int =42 ,
max_tokens :int =512 ,
temperature :float =0.8 ,
track_costs :bool =True ,
output_dir :Optional [str ]=None ,
debug :bool =False ,
together :bool =True ,
server :Optional [str ]=None ,
port :Optional [int ]=None ,
resume :bool =True ,
validation :bool =False ,
valid_ratio :float =0.5 ,
test_ratio :float =0.2 ,
use_structured_router :bool =False ,
**kwargs 
)->Tuple [float ,List [float ],List [str ]]:


    if use_structured_router :
        print ("Warning: Structured router not fully supported in batch mode. Some features may not work correctly.")


    model_specific_kwargs ={k :v for k ,v in kwargs .items ()if k in MODEL_SPECIFIC_PARAMS }
    task_creation_kwargs ={k :v for k ,v in kwargs .items ()if k not in MODEL_SPECIFIC_PARAMS }

    if debug and model_specific_kwargs :
        print (f"Model-specific parameters detected: {model_specific_kwargs}")


    task_creation_kwargs ={k :v for k ,v in task_creation_kwargs .items ()if k !='resume'}


    task =create_task (
    task_name ,
    llm_names =model_names ,
    seed =seed ,
    max_tokens =max_tokens ,
    temperature =temperature ,
    max_turns =1 ,
    track_costs =track_costs ,
    debug =debug ,
    together =together ,
    valid_ratio =valid_ratio ,
    test_ratio =test_ratio ,

    max_samples =max_samples if task_name .lower ()in ["mix_c_g","mix_c_d","mixcg","mixcd"]else -1 ,
    use_structured_router =use_structured_router ,
    **task_creation_kwargs 
    )


    if debug and task_name .lower ()in ["mix_c_g","mix_c_d","mixcg","mixcd"]:
        print (f"Created {task_name} with max_samples={max_samples}")


    batch_split_tasks ={
    "gsm8k","mix_c_g","mixcg",
    "mix_c_d","mixcd",
    "mix_c_d_full","mixcd_full",
    "mix_c_d_m_g","mixcdmg",
    "mix_c_d_m_g_full","mixcdmg_full",
    "mix_m_c_m_m_l","mix_five",
    "mix_m_c_m_m_l_full","mix_five_full",
    "mix_m_c_m_r_l","mix_mcmrl",
    "mix_m_c_m_r_l_full","mix_mcmrl_full",
    "mix_m_c_m_j_m_r_l","mix_seven",
    "mix_m_c_m_j_m_r_l_full","mix_seven_full",
    "math500","math-500","math_500",
    "deepmath","deepmath-103k","deepmath_103k",
    "medreason","arena","search_arena","searcharena",
    "pfgen","pfgen_bench","preferred_generation",
    "japanese_financial","jp_fin","jfin",
    "japanese_financial_en","jp_fin_en","jfin_en",
    "mmlu","massive_multitask_language_understanding",
    "hh_rlhf","hh-rlhf","hhrlhf","anthopic_hh","preference_task",
    "emotion","emotion_classification",
    "rlpr","rlpr_train","rlpr_dataset",
    "polaris","polaris-53k","polaris_53k","polaris_dataset",
    "livecodebench","livecodebench_dataset","livecodebench_train",
    }

    name =task_name .lower ()
    if name in batch_split_tasks :
        random_seed =np .random .RandomState (seed ).randint (0 ,10000 )
        if debug :
            print (f"Loading data with seed {random_seed}, split={split}")
        task .data_splits =task ._load_data (
        seed =random_seed ,
        split =split ,
        validation =validation ,
        valid_ratio =valid_ratio 
        )
    elif task .data_splits is None :
        random_seed =np .random .RandomState (seed ).randint (0 ,10000 )
        task .data_splits =task ._load_data (
        seed =random_seed ,
        split =split ,
        validation =validation ,
        valid_ratio =valid_ratio 
        )


    task .data_split =task .data_splits [split ]


    if output_dir is None :
        base_name =model_names [0 ].replace ("/","-")
        timestamp =time .strftime ("%Y%m%d_%H%M%S")
        output_dir =f"outputs/{task_name.lower()}_{base_name}_{len(model_names)}_models_{timestamp}"
    os .makedirs (output_dir ,exist_ok =True )
    print (f"\nSaving results to: {os.path.abspath(output_dir)}")


    existing_results ,processed_indices =[],set ()
    if resume :
        existing_results ,processed_indices =_check_for_existing_results (output_dir )
        if existing_results :
            print (f"Found {len(existing_results)} existing results.")
            if existing_results :
                config =existing_results [0 ].get ("config",{})
                if config .get ("models")!=model_names :
                    print (f"Warning: Existing results used different models: {config.get('models')}")
                if config .get ("max_tokens")!=max_tokens :
                    print (f"Warning: Existing results used different max_tokens: {config.get('max_tokens')}")
                if config .get ("temperature")!=temperature :
                    print (f"Warning: Existing results used different temperature: {config.get('temperature')}")


    results_file =f"{output_dir}/detailed_results.jsonl"
    all_rewards =[]
    all_responses =[]


    if existing_results :
        sorted_results =sorted (existing_results ,key =lambda x :x ["index"])
        for result in sorted_results :
            all_rewards .append (result ["reward"])
            all_responses .append (result ["response"])
        print (f"Loaded {len(all_rewards)} existing results with average reward: {np.mean(all_rewards):.4f}")


    if task_name .lower ()in ["mix_c_g","mixcg","mix_c_d","mixcd"]:
        sampled_indices =list (range (len (task .data_split )))
        if max_samples >0 and max_samples <len (sampled_indices ):
            sampled_indices =sampled_indices [:max_samples ]
        if debug :
            print (f"Using {len(sampled_indices)} indices from the dataset")
    else :
        sampled_indices =_sample_indices (task .data_split ,max_samples ,seed )


    num_samples =len (sampled_indices )
    remaining_indices =[i for i in sampled_indices if int (i )not in processed_indices ]

    if not remaining_indices :
        print ("All samples have already been processed. Nothing to do.")
        return np .mean (all_rewards )if all_rewards else 0.0 ,all_rewards ,all_responses 


    model =model_names [0 ]
    start_time =time .time ()


    for batch_start_idx in tqdm (range (0 ,len (remaining_indices ),batch_size ),
    desc =f"Processing {split} in batches"):
        batch_indices =remaining_indices [batch_start_idx :batch_start_idx +batch_size ]
        batch_prompts =[]


        if task_name .lower ()in ["mix_c_g","mixcg"]and debug :
            task_types_in_batch =[]
            for i in batch_indices :
                task_index =int (i )
                task_types_in_batch .append (task .data_split [task_index ].get ("task_type","unknown"))

            task_type_counts ={
            "countdown":task_types_in_batch .count ("countdown"),
            "gsm8k":task_types_in_batch .count ("gsm8k")
            }
            print (f"Batch task types: {task_type_counts}")


        elif task_name .lower ()in ["mix_c_d","mixcd"]and debug :
            task_types_in_batch =[]
            for i in batch_indices :
                task_index =int (i )
                task_types_in_batch .append (task .data_split [task_index ].get ("task_type","unknown"))

            task_type_counts ={
            "countdown":task_types_in_batch .count ("countdown"),
            "deepmath":task_types_in_batch .count ("deepmath")
            }
            print (f"Batch task types: {task_type_counts}")

        elif task_name .lower ()in ["mix_c_d_full","mixcd_full"]and debug :
            task_types_in_batch =[]
            for i in batch_indices :
                task_index =int (i )
                task_types_in_batch .append (task .data_split [task_index ].get ("task_type","unknown"))

            task_type_counts ={
            "countdown":task_types_in_batch .count ("countdown"),
            "deepmath":task_types_in_batch .count ("deepmath")
            }
            print (f"Batch task types (unbalanced): {task_type_counts}")


        elif task_name .lower ()in ["mix_c_d_m_g","mixcdmg"]and debug :
            task_types_in_batch =[]
            for i in batch_indices :
                task_index =int (i )
                task_types_in_batch .append (task .data_split [task_index ].get ("task_type","unknown"))

            task_type_counts ={
            "countdown":task_types_in_batch .count ("countdown"),
            "deepmath":task_types_in_batch .count ("deepmath"),
            "medreason":task_types_in_batch .count ("medreason"),
            "gsm8k":task_types_in_batch .count ("gsm8k")
            }
            print (f"Batch task types: {task_type_counts}")

        elif task_name .lower ()in ["mix_c_d_m_g_full","mixcdmg_full"]and debug :
            task_types_in_batch =[]
            for i in batch_indices :
                task_index =int (i )
                task_types_in_batch .append (task .data_split [task_index ].get ("task_type","unknown"))

            task_type_counts ={
            "countdown":task_types_in_batch .count ("countdown"),
            "deepmath":task_types_in_batch .count ("deepmath"),
            "medreason":task_types_in_batch .count ("medreason"),
            "gsm8k":task_types_in_batch .count ("gsm8k")
            }
            print (f"Batch task types (unbalanced): {task_type_counts}")

        elif task_name .lower ()in ["mix_m_c_m_m_l","mix_five"]and debug :
            task_types_in_batch =[]
            for i in batch_indices :
                task_index =int (i )
                task_types_in_batch .append (task .data_split [task_index ].get ("task_type","unknown"))

            task_type_counts ={
            "medreason":task_types_in_batch .count ("medreason"),
            "countdown":task_types_in_batch .count ("countdown"),
            "math500":task_types_in_batch .count ("math500"),
            "mmlu":task_types_in_batch .count ("mmlu"),
            "livecodebench":task_types_in_batch .count ("livecodebench")
            }
            print (f"Batch task types: {task_type_counts}")

        elif task_name .lower ()in ["mix_m_c_m_m_l_full","mix_five_full"]and debug :
            task_types_in_batch =[]
            for i in batch_indices :
                task_index =int (i )
                task_types_in_batch .append (task .data_split [task_index ].get ("task_type","unknown"))

            task_type_counts ={
            "medreason":task_types_in_batch .count ("medreason"),
            "countdown":task_types_in_batch .count ("countdown"),
            "math500":task_types_in_batch .count ("math500"),
            "mmlu":task_types_in_batch .count ("mmlu"),
            "livecodebench":task_types_in_batch .count ("livecodebench")
            }
            print (f"Batch task types (unbalanced): {task_type_counts}")

        elif task_name .lower ()in ["mix_m_c_m_r_l","mix_mcmrl"]and debug :
            task_types_in_batch =[]
            for i in batch_indices :
                task_index =int (i )
                task_types_in_batch .append (task .data_split [task_index ].get ("task_type","unknown"))

            task_type_counts ={
            "medreason":task_types_in_batch .count ("medreason"),
            "countdown":task_types_in_batch .count ("countdown"),
            "math500":task_types_in_batch .count ("math500"),
            "rlpr":task_types_in_batch .count ("rlpr"),
            "livecodebench":task_types_in_batch .count ("livecodebench")
            }
            print (f"Batch task types: {task_type_counts}")

        elif task_name .lower ()in ["mix_m_c_m_r_l_full","mix_mcmrl_full"]and debug :
            task_types_in_batch =[]
            for i in batch_indices :
                task_index =int (i )
                task_types_in_batch .append (task .data_split [task_index ].get ("task_type","unknown"))

            task_type_counts ={
            "medreason":task_types_in_batch .count ("medreason"),
            "countdown":task_types_in_batch .count ("countdown"),
            "math500":task_types_in_batch .count ("math500"),
            "rlpr":task_types_in_batch .count ("rlpr"),
            "livecodebench":task_types_in_batch .count ("livecodebench")
            }
            print (f"Batch task types (unbalanced): {task_type_counts}")

        elif task_name .lower ()in ["mix_m_c_m_j_m_r_l","mix_seven"]and debug :
            task_types_in_batch =[]
            for i in batch_indices :
                task_index =int (i )
                task_types_in_batch .append (task .data_split [task_index ].get ("task_type","unknown"))

            task_type_counts ={
            "medreason":task_types_in_batch .count ("medreason"),
            "countdown":task_types_in_batch .count ("countdown"),
            "math500":task_types_in_batch .count ("math500"),
            "japanese_financial":task_types_in_batch .count ("japanese_financial"),
            "mmlu":task_types_in_batch .count ("mmlu"),
            "rlpr":task_types_in_batch .count ("rlpr"),
            "livecodebench":task_types_in_batch .count ("livecodebench")
            }
            print (f"Batch task types: {task_type_counts}")

        elif task_name .lower ()in ["mix_m_c_m_j_m_r_l_full","mix_seven_full"]and debug :
            task_types_in_batch =[]
            for i in batch_indices :
                task_index =int (i )
                task_types_in_batch .append (task .data_split [task_index ].get ("task_type","unknown"))

            task_type_counts ={
            "medreason":task_types_in_batch .count ("medreason"),
            "countdown":task_types_in_batch .count ("countdown"),
            "math500":task_types_in_batch .count ("math500"),
            "japanese_financial":task_types_in_batch .count ("japanese_financial"),
            "mmlu":task_types_in_batch .count ("mmlu"),
            "rlpr":task_types_in_batch .count ("rlpr"),
            "livecodebench":task_types_in_batch .count ("livecodebench")
            }
            print (f"Batch task types (unbalanced): {task_type_counts}")


        for i in batch_indices :
            task_index =int (i )
            user_prompt =task ._format_user_prompt (task .data_split [task_index ])
            messages =[
            {"role":"system","content":task .system_prompt },
            {"role":"user","content":user_prompt },
            {"role":"assistant","content":task .assistant_prompt }
            ]
            batch_prompts .append (messages )

        try :

            completions =batch_completion (
            model_name =model ,
            batch_messages =batch_prompts ,
            max_tokens =max_tokens ,
            temperature =temperature ,
            max_concurrency =max_concurrency ,
            server =server ,
            port =port ,
            debug =debug ,
            together =together ,
            **model_specific_kwargs 
            )


            rewards =task ._calculate_reward (
            completions =completions ,
            task_data =[task .data_split [int (i )]for i in batch_indices ],
            debug =debug 
            )


            all_rewards .extend (rewards )
            all_responses .extend (completions )


            results_data =[]
            for j ,idx in enumerate (batch_indices ):

                original_task_data ={k :v for k ,v in task .data_split [int (idx )].items ()}
                filtered_task_data =filter_task_data_for_logging (original_task_data )

                result ={
                "index":int (idx ),
                "task_data":filtered_task_data ,
                "response":completions [j ],
                "reward":float (rewards [j ]),
                "config":{
                "models":model_names ,
                "max_tokens":max_tokens ,
                "temperature":temperature ,
                "debug":debug ,
                "together":together ,
                "use_structured_router":use_structured_router if hasattr (task ,
                'use_structured_router')else False ,
                "model_specific_params":model_specific_kwargs 
                }
                }
                results_data .append (result )


            with open (results_file ,"a")as f :
                for result in results_data :
                    f .write (json .dumps (result )+"\n")


            processed =len (processed_indices )+(batch_start_idx +len (batch_indices ))
            elapsed =time .time ()-start_time 
            avg_reward =np .mean (all_rewards )
            tqdm .write (f"Processed {processed}/{num_samples} problems. "
            f"Current average reward: {avg_reward:.4f} | Time elapsed: {elapsed:.2f}s")


            _save_summary (
            output_dir =output_dir ,
            task =task ,
            rewards =all_rewards ,
            split =split ,
            agent_policy ="batch",
            elapsed_time =elapsed ,
            num_samples =processed 
            )
        except Exception as e :
            print (f"Error processing batch: {e}")
            import traceback 
            traceback .print_exc ()


    elapsed_time =time .time ()-start_time 
    avg_reward =np .mean (all_rewards )if all_rewards else 0.0 
    summary ={
    "models":model_names ,
    "avg_reward":float (avg_reward ),
    "num_samples":num_samples ,
    "split":split ,
    "batch_size":batch_size ,
    "max_concurrency":max_concurrency ,
    "max_tokens":max_tokens ,
    "temperature":temperature ,
    "execution_time_seconds":elapsed_time ,
    "timestamp":time .strftime ("%Y-%m-%d %H:%M:%S"),
    "debug":debug ,
    "together":together ,
    "use_structured_router":use_structured_router if hasattr (task ,'use_structured_router')else False ,
    "model_specific_params":model_specific_kwargs 
    }


    if track_costs :
        from guf .cost import get_cost_summary 
        cost_summary =get_cost_summary ()
        summary ["cost_summary"]=cost_summary 
        summary ["cost_per_problem"]=cost_summary ["total_cost"]/num_samples if num_samples >0 else 0 


    with open (f"{output_dir}/summary.json","w")as f :
        json .dump (summary ,f ,indent =2 )

    print (f"\nCompleted {num_samples} problems with average reward: {avg_reward:.4f} in {elapsed_time:.2f} seconds")
    print (f"Results saved to: {os.path.abspath(output_dir)}")

    return avg_reward ,all_rewards ,all_responses 