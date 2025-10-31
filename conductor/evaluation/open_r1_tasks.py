


import random 

from lighteval .metrics .dynamic_metrics import (
ExprExtractionConfig ,
IndicesExtractionConfig ,
LatexExtractionConfig ,
multilingual_extractive_match_metric ,
)
from lighteval .tasks .lighteval_task import LightevalTaskConfig 
from lighteval .tasks .requests import Doc 
from lighteval .utils .language import Language 


latex_gold_metric =multilingual_extractive_match_metric (
language =Language .ENGLISH ,
fallback_mode ="first_match",
precision =5 ,
gold_extraction_target =(LatexExtractionConfig (),),

pred_extraction_target =(ExprExtractionConfig (),LatexExtractionConfig (boxed_match_priority =0 )),
aggregation_function =max ,
)

expr_gold_metric =multilingual_extractive_match_metric (
language =Language .ENGLISH ,
fallback_mode ="first_match",
precision =5 ,
gold_extraction_target =(ExprExtractionConfig (),),

pred_extraction_target =(ExprExtractionConfig (),LatexExtractionConfig (boxed_match_priority =0 )),
aggregation_function =max ,
)

gpqa_metric =multilingual_extractive_match_metric (
language =Language .ENGLISH ,
gold_extraction_target =[IndicesExtractionConfig (prefix_for_extraction ="NativeLetters")],
pred_extraction_target =[IndicesExtractionConfig (prefix_for_extraction ="NativeLetters")],
precision =5 ,
)


def prompt_fn (line ,task_name :str =None ):

    return Doc (
    task_name =task_name ,
    query =line ["problem"],
    choices =[line ["solution"]],
    gold_index =0 ,
    )


def aime_prompt_fn (line ,task_name :str =None ):
    return Doc (
    task_name =task_name ,
    query =line ["problem"],
    choices =[line ["answer"]],
    gold_index =0 ,
    )


def gpqa_prompt_fn (line ,task_name :str =None ):

    gold_index =random .randint (0 ,3 )
    choices =[line ["Incorrect Answer 1"],line ["Incorrect Answer 2"],line ["Incorrect Answer 3"]]
    choices .insert (gold_index ,line ["Correct Answer"])
    query_template ="Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
    query =query_template .format (A =choices [0 ],B =choices [1 ],C =choices [2 ],D =choices [3 ],Question =line ["Question"])

    return Doc (
    task_name =task_name ,
    query =query ,
    choices =["A","B","C","D"],
    gold_index =gold_index ,
    instruction =query ,
    )



aime24 =LightevalTaskConfig (
name ="aime24",
suite =["custom"],
prompt_function =aime_prompt_fn ,
hf_repo ="HuggingFaceH4/aime_2024",
hf_subset ="default",
hf_avail_splits =["train"],
evaluation_splits =["train"],
few_shots_split =None ,
few_shots_select =None ,
generation_size =32768 ,
metric =[expr_gold_metric ],
version =1 ,
)
math_500 =LightevalTaskConfig (
name ="math_500",
suite =["custom"],
prompt_function =prompt_fn ,
hf_repo ="HuggingFaceH4/MATH-500",
hf_subset ="default",
hf_avail_splits =["test"],
evaluation_splits =["test"],
few_shots_split =None ,
few_shots_select =None ,
generation_size =32768 ,
metric =[latex_gold_metric ],
version =1 ,
)
gpqa_diamond =LightevalTaskConfig (
name ="gpqa:diamond",
suite =["custom"],
prompt_function =gpqa_prompt_fn ,
hf_repo ="Idavidrein/gpqa",
hf_subset ="gpqa_diamond",
hf_avail_splits =["train"],
evaluation_splits =["train"],
few_shots_split =None ,
few_shots_select =None ,
generation_size =32768 ,
metric =[gpqa_metric ],
stop_sequence =[],
trust_dataset =True ,
version =1 ,
)



TASKS_TABLE =[]
TASKS_TABLE .append (aime24 )
TASKS_TABLE .append (math_500 )
TASKS_TABLE .append (gpqa_diamond )


if __name__ =="__main__":
    print ([t ["name"]for t in TASKS_TABLE ])
    print (len (TASKS_TABLE ))
