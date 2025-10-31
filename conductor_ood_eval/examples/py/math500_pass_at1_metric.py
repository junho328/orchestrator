


from __future__ import annotations 
import numpy as np 
from aenum import extend_enum 
import re 

from lighteval .metrics .metrics_sample import PassAtK 
from lighteval .metrics .utils .metric_utils import (
SampleLevelMetric ,
MetricCategory ,
MetricUseCase ,
)
from lighteval .metrics .metrics import Metrics 
from lighteval .tasks .lighteval_task import LightevalTaskConfig 
import lighteval .tasks .default_prompts as prompt 


from lighteval .metrics .dynamic_metrics import (
ExprExtractionConfig ,
LatexExtractionConfig ,
extract_target_from_pred ,
get_extraction_regexes ,
compare_gold_target ,
)
from lighteval .utils .language import Language 

def extract_answer_simple (text :str )->str |None :

    if not text :
        return None 


    pattern =r'<answer[^>]*>(.*?)</answer>'
    matches =re .findall (pattern ,text ,re .DOTALL |re .IGNORECASE )

    if matches :

        answer =matches [-1 ]
        answer =answer .strip ()


        answer =re .sub (r'\s+',' ',answer ).strip ()

        if answer :
            return answer 

    return None 


def extract_boxed_simple (text :str )->str |None :

    if not text :
        return None 


    matches =re .findall (r'\\boxed\{([^}]*)\}',text )
    if matches :
        return matches [-1 ].strip ()

    return None 


def extract_answer_comprehensive (text :str )->str |None :

    if not text :
        return None 


    answer =extract_answer_simple (text )
    if answer :
        return answer 


    boxed =extract_boxed_simple (text )
    if boxed :
        return boxed 


    patterns =[
    r'final answer[:\s]+([^.]+)',
    r'answer is[:\s]+([^.]+)',
    r'solution[:\s]+([^.]+)',
    r'result[:\s]+([^.]+)',
    ]

    for pattern in patterns :
        matches =re .findall (pattern ,text ,re .IGNORECASE )
        if matches :
            candidate =matches [-1 ].strip ()
            if len (candidate )<50 :
                return candidate 

    return None 


def normalize_simple (text :str )->str :

    if not text :
        return ""


    normalized =text .strip ()


    normalized =re .sub (r'\\boxed\{([^}]*)\}',r'\1',normalized )
    normalized =re .sub (r'\\frac\{([^}]*)\}\{([^}]*)\}',r'(\1)/(\2)',normalized )
    normalized =re .sub (r'\\sqrt\{([^}]*)\}',r'sqrt(\1)',normalized )
    normalized =re .sub (r'\\([a-zA-Z]+)',r'\1',normalized )


    normalized =re .sub (r'[\s\$\{\}]','',normalized )

    return normalized .lower ()


def math500_single_score (pred :str ,gold :str ,_doc =None )->int :


    pred_answer =extract_answer_comprehensive (pred )
    gold_answer =extract_boxed_simple (gold )

    if not gold_answer :
        gold_answer =gold 

    if not pred_answer :
        print (f"----- No answer found in prediction -----")
        print (f"Prediction: {pred}")
        return 0 


    try :
        result =compare_gold_target (pred_answer ,gold_answer ,_doc )
        if result ==1 :
            return 1 
    except :
        pass 


    pred_norm =normalize_simple (pred_answer )
    gold_norm =normalize_simple (gold_answer )

    if pred_norm ==gold_norm :
        return 1 


    try :
        pred_num =float (pred_norm )
        gold_num =float (gold_norm )
        if abs (pred_num -gold_num )<1e-10 :
            return 1 
    except :
        pass 



    pred_clean =re .sub (r'[^\w\d\.\-\+\*/\(\)]','',pred_answer )
    gold_clean =re .sub (r'[^\w\d\.\-\+\*/\(\)]','',gold_answer )

    if pred_clean .lower ()==gold_clean .lower ():
        return 1 

    print (f"------ No match found: ------")
    print (f"  Gold normalized: '{gold_norm}'")
    print (f"  Gold answer: '{gold_answer}'")
    print (f"  Raw pred: '{pred}'")
    print (f"  Pred normalized: '{pred_norm}'")
    print (f"  Pred answer: '{pred_answer}'")

    return 0 






math500_pass_at1 =SampleLevelMetric (
metric_name ="math500_pass@1:1_samples",
sample_level_fn =PassAtK (
k =1 ,
n =1 ,
strip_strings =True ,
sample_scoring_function =math500_single_score ,
).compute ,
category =MetricCategory .GENERATIVE_SAMPLING ,
use_case =MetricUseCase .REASONING ,
corpus_level_fn =np .mean ,
higher_is_better =True ,
)




extend_enum (Metrics ,"math500_pass_at1",math500_pass_at1 )




math_500 =LightevalTaskConfig (
name ="math_500",
suite =["lighteval"],
prompt_function =prompt .math_500 ,
hf_repo ="HuggingFaceH4/MATH-500",
hf_subset ="default",
hf_avail_splits =["test"],
evaluation_splits =["test"],
generation_size =32768 ,
metric =[Metrics .math500_pass_at1 ],
version =3 ,
)

TASKS_TABLE =[math_500 ]

if __name__ =="__main__":
    print ("Imported math500_pass_at1 metric")