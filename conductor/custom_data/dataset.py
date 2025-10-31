from .dataset_utils import PROMPT_FORMAT_LIBRARY ,load_task ,format_models_list 
from typing import List 
from transformers import PreTrainedTokenizerBase 
from datasets import concatenate_datasets 

class Dataset ():

    def __init__ (self ,
    tokenizer :PreTrainedTokenizerBase ,
    task_module ,
    class_name ,
    prompt_template ,
    available_models ,
    max_routing_steps =5 ,
    seed =42 ,
    split ='train',
    hide_names =True ,
    hide_parameters =True ,
    apply_chat_template =True ,
    tokenizer_kwargs ={},
    max_model_length =8192 ,
    eval_max_samples =-1 ,
    evaluate_only =False ,
    finetune =False ,
    ):
        print (f"Available models: {available_models}")
        self .task =load_task (task_module ,class_name ,available_models ,seed ,evaluate_only =evaluate_only ,finetune =finetune )
        self .task .reset (split ='test'if split in ('eval_all','valid_test')else split )
        if split in ('eval_all','valid_test'):
            valid_ds =self .task .grab_data ('valid')
            test_ds =self .task .grab_data ('test')
            base_dataset =concatenate_datasets ([valid_ds ,test_ds ])
            self .dataset =base_dataset 
        else :
            self .dataset =self .task .grab_data (split =split )

        if eval_max_samples >0 and len (self .dataset )>eval_max_samples :
            self .dataset =self .dataset .select (range (eval_max_samples ))

        self .format =PROMPT_FORMAT_LIBRARY .get (prompt_template ,None )

        self .max_routing_steps =max_routing_steps 
        self .models =format_models_list (
        available_models ,
        hide_names =hide_names ,
        hide_parameters =hide_parameters 
        )

        self .tokenizer =tokenizer 
        self .max_model_length =max_model_length 
        if self .max_model_length is not None :
            self .tokenizer .max_model_length =self .max_model_length 
        self .apply_chat_template =apply_chat_template 
        self .tokenizer_kwargs =tokenizer_kwargs 

    def _format_prompt (self ,base_question ):

        return self .format .format (
        user_question =base_question ,
        available_models =self .models ,
        max_number_or_routing_steps =self .max_routing_steps 
        )

    def __len__ (self ):

        return len (self .dataset )

    def __getitem__ (self ,idx ):


        data =self .dataset [idx ]
        base_question =self .task ._format_base_prompt (data )

        conductor_prompt =self ._format_prompt (base_question )

        if conductor_prompt is None :
            raise ValueError (f'Question {base_question} unable to be formatted to prompt')

        if 'prompt'in data :
            raise ValueError ('Data should not contain key prompt')

        system_msg ='You are a helpful assistant. You think about the reasoning processing in your mind and how best to coordinate a team of models to solve user queries and provide the user with the response.'
        messages =[{
        "role":"system",
        "content":system_msg ,
        },
        ]

        messages .append (
        {"role":'user',"content":conductor_prompt },)
        messages .append (
        {"role":'assistant',"content":"Let's think step by step how to break this user question down into subtasks, the models I'll use to solve my subtasks, and the access list I'll use for determining which agents can see each others' responses. "},)

        if self .apply_chat_template :
            conductor_prompt =self .tokenizer .apply_chat_template (
            conversation =messages ,tokenize =False ,continue_final_message =True ,
            **self .tokenizer_kwargs )

            if idx ==0 and self .tokenizer_kwargs .get ("chat_template_kwargs",{}).get ("enable_thinking",False ):
                print (f"Thinking mode enabled via chat template kwargs")
                print (conductor_prompt )

        return {
        'prompt':conductor_prompt ,
        'base_question':base_question ,
        **data 
        }


def make_dataset (
tokenizer :PreTrainedTokenizerBase ,
task_module :str ,
class_name :str ,
prompt_template :str ,
open_models :List [str ],
closed_models :List [str ],
mask_names :bool =True ,
max_routing_steps :int =5 ,
seed :int =42 ,
hide_parameters :bool =True ,
evaluate_only :bool =False ,
eval_max_samples :int =-1 ,
finetune :bool =False ,
max_model_length :int =8192 ,
**kwargs 
)->Dataset :

    available_models =closed_models +open_models 
    print (f"Creating dataset from {task_module}.{class_name} with prompt template '{prompt_template}'")

    tokenizer_kwargs =kwargs .get ("tokenizer_kwargs",{})

    return {
    "train_dataset":Dataset (tokenizer ,
    task_module ,
    class_name ,
    prompt_template ,
    available_models ,
    max_routing_steps ,
    seed ,
    split ='train',
    hide_names =mask_names ,
    hide_parameters =hide_parameters ,
    evaluate_only =evaluate_only ,
    finetune =finetune ,
    max_model_length =max_model_length ,
    tokenizer_kwargs =tokenizer_kwargs 
    ),
    "eval_dataset":Dataset (tokenizer ,
    task_module ,
    class_name ,
    prompt_template ,
    available_models ,
    max_routing_steps ,
    seed ,
    split ='test',
    hide_names =mask_names ,
    hide_parameters =hide_parameters ,
    eval_max_samples =eval_max_samples ,
    evaluate_only =evaluate_only ,
    max_model_length =max_model_length ,
    tokenizer_kwargs =tokenizer_kwargs 
    )
    }


if __name__ =="__main__":
    dataset =make_dataset (
    task_module ="guf.tasks.mix_c_d",
    class_name ="MixCDTask",
    prompt_template ="v1_7c2_2",
    open_models =["ModelA","ModelB","ModelC"],
    closed_models =["ModelD","ModelE","ModelF"],
    mask_names =True ,
    max_routing_steps =5 ,
    seed =42 
    )

    print (dataset ["train_dataset"][0 ])
    print (dataset ["eval_dataset"][0 ])