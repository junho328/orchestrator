


import json 
import requests 
import argparse 
import time 
import random 
import re 
from datetime import datetime 
from typing import List ,Dict ,Any ,Optional ,Tuple 
from dataclasses import dataclass 
from functools import lru_cache 




@dataclass 
class TranslationLiterals :
    answer :str ="answer"
    question_word :str ="question"
    colon :str =":"
    sentence_space :str =" "
    full_stop :str ="."
    comma :str =","
    and_word :str ="and"
    or_word :str ="or"


TRANSLATION_LITERALS =TranslationLiterals ()

LETTER_INDICES =["A","B","C","D"]



@dataclass (frozen =True )
class IndicesExtractionConfig :

    prefix_for_extraction :str 
    try_extract_without_anchor :bool =True 



def get_prefix (prefix_style :str ,translation_literal :TranslationLiterals )->List [str ]:

    if prefix_style =="NativeLetters":
        return ["A","B","C","D","E","F","G","H","I","J"]
    elif prefix_style =="Numbers":
        return ["1","2","3","4","5","6","7","8","9","10"]
    else :
        return ["A","B","C","D","E","F","G","H","I","J"]



@lru_cache (maxsize =10 )
def lazy_indices_regex (len_choices :int )->List [Tuple [re .Pattern [str ],int ]]:



    indices =get_prefix ("NativeLetters",TRANSLATION_LITERALS )[:len_choices ]
    indices_escaped =[re .escape (i )for i in indices ]

    indices_wrapped =[rf"(?:{i}|\({i}\))"for i in indices_escaped ]
    indice_str_re =f"(?P<indices>{'|'.join(indices_wrapped)})"


    full_stop_re =rf"[{re.escape(TRANSLATION_LITERALS.full_stop)}\.]"
    comma_re =rf"[{re.escape(TRANSLATION_LITERALS.comma)}\,]"
    colon_re =rf"[{re.escape(TRANSLATION_LITERALS.colon)}\:]"
    space_re =re .escape (TRANSLATION_LITERALS .sentence_space )


    answer_prefix_re =rf"(?:^|{space_re})(?:\*\*)?"
    answer_suffix_re =rf"(?:\*\*)?(?:{full_stop_re}|{comma_re}|{colon_re}|{space_re}|$)"
    answer_re =f"{answer_prefix_re}{indice_str_re}{answer_suffix_re}"
    answer_re_start =rf"^(?:\*\*)?{indice_str_re}{answer_suffix_re}"
    answer_re_line_start =rf"\n(?:\*\*)?{indice_str_re}{answer_suffix_re}"

    answer_word =f"(?i:{TRANSLATION_LITERALS.answer})"

    regexes =[]


    final_answer_prefixed_re =rf"(?i:final answer is)\:?\s*{indice_str_re}\.?\s?I hope"
    final_answer_prefixed_just_is =rf"(?i:final answer.{{0,100}}?)\s+is\:?\s*{indice_str_re}"
    regexes .extend ([
    (final_answer_prefixed_re ,0 ),
    (final_answer_prefixed_just_is ,50 ),
    ])


    regexes .extend ([

    (f"{answer_word}{colon_re}.{{0,50}}?{answer_re}",100 ),

    (f"{answer_word}.{{0,50}}?{answer_re}",150 ),

    (answer_re_start ,200 ),

    (answer_re_line_start ,210 ),

    (answer_re ,250 ),
    (indice_str_re ,300 ),
    ])

    return [(re .compile (pattern ),priority )for pattern ,priority in regexes ]



def extract_indices_from_match (match :re .Match )->Tuple [Optional [str ],str ]:


    def normalize_index (index :str )->str :
        return index .replace ("(","").replace (")","").strip ()

    indices =match .group ("indices")
    return normalize_index (indices ),normalize_index (indices )



def extract_target_from_pred (pred :str ,len_choices :int )->List [str ]:



    regex_patterns =lazy_indices_regex (len_choices )

    extracted_predictions =[]
    fallbacks =[]


    patterns_by_priority ={}
    for pattern ,priority in regex_patterns :
        if priority not in patterns_by_priority :
            patterns_by_priority [priority ]=[]
        patterns_by_priority [priority ].append (pattern )

    match_found =False 


    for priority in sorted (patterns_by_priority .keys ()):
        patterns =patterns_by_priority [priority ]


        matches_with_pos =[]
        for pattern in patterns :
            for match in pattern .finditer (pred ):
                matches_with_pos .append ((match ,match .start (),match .end ()))


        matches_with_pos =sorted (matches_with_pos ,key =lambda x :(x [2 ],-x [1 ]),reverse =True )


        for match ,_ ,_ in matches_with_pos :
            extracted_match ,str_fallback =extract_indices_from_match (match )
            match_found =True 

            if str_fallback :
                fallbacks .append (str_fallback )

            if extracted_match is not None :
                extracted_predictions .append (extracted_match )
                break 


        if extracted_predictions or match_found :
            break 


    if not extracted_predictions and fallbacks :
        extracted_predictions =[fallbacks [0 ]]

    return extracted_predictions 




def compare_gold_target (pred_extracted :List [str ],gold_extracted :List [str ])->bool :

    if not pred_extracted or not gold_extracted :
        return False 


    pred_str =pred_extracted [0 ]if pred_extracted else ""
    gold_str =gold_extracted [0 ]if gold_extracted else ""

    return pred_str .strip ().upper ()==gold_str .strip ().upper ()



def load_gpqa_dataset ():

    try :
        from datasets import load_dataset 
        dataset =load_dataset ("Idavidrein/gpqa","gpqa_diamond",split ="train")
        return dataset 
    except ImportError :
        print ("❌ datasets library not available. Install with: pip install datasets")
        return None 
    except Exception as e :
        print (f"❌ Error loading dataset: {e}")
        return None 



def format_gpqa_prompt (sample :Dict [str ,Any ])->Tuple [str ,str ,int ]:



    gold_index =random .randint (0 ,3 )
    choices =[
    sample ["Incorrect Answer 1"],
    sample ["Incorrect Answer 2"],
    sample ["Incorrect Answer 3"]
    ]
    choices .insert (gold_index ,sample ["Correct Answer"])


    query_template =(
    "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}")

    query =query_template .format (
    A =choices [0 ],
    B =choices [1 ],
    C =choices [2 ],
    D =choices [3 ],
    Question =sample ["Question"]
    )


    gold_letter =LETTER_INDICES [gold_index ]

    return query ,gold_letter ,gold_index 


def debug_extraction_pipeline ():


    print ("🔍 TESTING EXTRACTION PIPELINE")
    print ("="*80 )

    test_cases =[
    {
    "problem":"Which element has atomic number 6?",
    "gold_answer":"B",
    "model_responses":[
    "The answer is B.",
    "B",
    "Looking at the periodic table, I can see that carbon has atomic number 6.\n\nAnswer: B",
    "After thinking about this, the element with atomic number 6 is carbon.\n\nFinal answer is: B",
    "Let me think step by step...\n\nAtomic number represents the number of protons.\nCarbon has 6 protons.\n\nTherefore, the answer is (B).",
    "I believe it's B, carbon.",
    "The correct choice is B.",
    "B) Carbon is correct"
    ]
    }
    ]

    for test_case in test_cases :
        print (f"📝 PROBLEM: {test_case['problem']}")
        print (f"🎯 GOLD ANSWER: {test_case['gold_answer']}")
        print ("="*80 )


        gold_extracted =extract_target_from_pred (test_case ['gold_answer'],4 )
        print (f"✨ GOLD EXTRACTED: {gold_extracted}")
        print ()


        for i ,response in enumerate (test_case ['model_responses']):
            print (f"🤖 MODEL RESPONSE {i + 1}:")
            print (f"   Raw: {response}")


            pred_extracted =extract_target_from_pred (response ,4 )
            print (f"   Extracted: {pred_extracted}")


            try :
                comparison_result =compare_gold_target (pred_extracted ,gold_extracted )
                print (f"   Comparison: {comparison_result} ({'✅ MATCH' if comparison_result else '❌ NO MATCH'})")
            except Exception as e :
                print (f"   Comparison ERROR: {e}")

            print ()

        print ("="*80 )


def test_with_real_dataset (server_host ="slurm0us-a3nodeset-0",server_port =8080 ):


    print ("📊 TESTING WITH REAL GPQA DATASET")
    print ("="*80 )

    dataset =load_gpqa_dataset ()
    if dataset is None :
        return 

    sample =dataset [0 ]
    print ("📊 REAL DATASET SAMPLE:")
    print (json .dumps (dict (sample ),indent =2 ))
    print ("="*80 )


    query ,gold_letter ,gold_index =format_gpqa_prompt (sample )

    print (f"📝 FORMATTED QUERY:")
    print (f"{query}")
    print (f"🎯 GOLD LETTER: {gold_letter}")
    print (f"🎯 GOLD INDEX: {gold_index}")
    print (f"🌐 Testing with server: http://{server_host}:{server_port}")
    print ("="*80 )


    try :
        print ("🔄 Making API call...")
        api_url =f"http://{server_host}:{server_port}/v1/chat/completions"
        response =requests .post (
        api_url ,
        headers ={"Content-Type":"application/json","Authorization":"Bearer sk-test"},
        json ={
        "model":"gpt-4o",
        "messages":[{"role":"user","content":query }],
        "max_tokens":2048 ,
        "temperature":0.1 
        },
        timeout =300 
        )

        if response .status_code ==200 :
            model_response =response .json ()["choices"][0 ]["message"]["content"]
            print ("🤖 MODEL RESPONSE:")
            print (model_response )
            print ("="*80 )


            gold_extracted =extract_target_from_pred (gold_letter ,4 )
            pred_extracted =extract_target_from_pred (model_response ,4 )

            print (f"✨ GOLD EXTRACTED: {gold_extracted}")
            print (f"🔍 PRED EXTRACTED: {pred_extracted}")

            try :
                comparison =compare_gold_target (pred_extracted ,gold_extracted )
                print (f"🎯 COMPARISON RESULT: {comparison}")
                print (f"🧪 MANUAL COMPARISON:")
                print (f"   Gold: '{gold_extracted}' (type: {type(gold_extracted)})")
                print (f"   Pred: '{pred_extracted}' (type: {type(pred_extracted)})")
                if gold_extracted and pred_extracted :
                    print (f"   String equality: {str(gold_extracted[0]) == str(pred_extracted[0])}")

            except Exception as e :
                print (f"❌ COMPARISON ERROR: {e}")
                import traceback 
                traceback .print_exc ()

        else :
            print (f"❌ API ERROR: {response.status_code}")
            print (response .text )

    except Exception as e :
        print (f"❌ API TEST ERROR: {e}")
        import traceback 
        traceback .print_exc ()


def test_multiple_samples (max_samples =3 ,server_host ="slurm0us-a3nodeset-0",server_port =8080 ,
trial_num =None ,evaluation_results_path =None ,comprehensive_results =None ):


    trial_prefix =f"[Trial {trial_num}] "if trial_num is not None else ""

    print (f"🧪 {trial_prefix}TESTING MULTIPLE SAMPLES (max_samples={max_samples})")
    print (f"🌐 Using server: http://{server_host}:{server_port}")
    print ("⏰ Note: Each API request has a 5-minute timeout with retry logic")
    print ("="*80 )

    dataset =load_gpqa_dataset ()
    if dataset is None :
        return 0.0 ,[]


    print (f"📊 Dataset Info: {len(dataset)} total samples available")

    total_samples =min (max_samples ,len (dataset ))
    if total_samples <max_samples :
        print (
        f"⚠️  Note: Dataset only has {len(dataset)} samples, so evaluating {total_samples} instead of {max_samples}")

    print (f"🎯 Evaluating {total_samples} samples")
    print ("="*80 )

    results =[]
    correct_count =0 


    if comprehensive_results is not None and trial_num is not None :
        current_trial ={
        "trial_num":trial_num ,
        "start_timestamp":datetime .now ().isoformat (),
        "samples":[],
        "trial_statistics":{
        "total_samples":total_samples ,
        "completed_samples":0 ,
        "correct_count":0 ,
        "current_score":0.0 ,
        "status":"in_progress"
        }
        }
        comprehensive_results ["trials"].append (current_trial )

    for i in range (total_samples ):
        sample =dataset [i ]


        query ,gold_letter ,gold_index =format_gpqa_prompt (sample )

        print (f"\n📝 {trial_prefix}SAMPLE {i + 1}/{total_samples}")
        question_preview =sample ["Question"][:100 ]+"..."if len (sample ["Question"])>100 else sample ["Question"]
        print (f"Question: {question_preview}")
        print (f"Gold Answer: {gold_letter}")

        sample_start_time =datetime .now ()
        sample_result ={
        "sample_id":i ,
        "trial_num":trial_num ,
        "question":sample ["Question"],
        "gold_letter":gold_letter ,
        "gold_index":gold_index ,
        "start_timestamp":sample_start_time .isoformat (),
        "status":"in_progress"
        }

        try :

            max_retries =2 
            response =None 

            for retry in range (max_retries ):
                try :
                    if retry >0 :
                        print (f"   🔄 Retry {retry}/{max_retries - 1}...")
                    api_url =f"http://{server_host}:{server_port}/v1/chat/completions"
                    response =requests .post (
                    api_url ,
                    headers ={"Content-Type":"application/json","Authorization":"Bearer sk-test"},
                    json ={
                    "model":"gpt-4o",
                    "messages":[{"role":"user","content":query }],
                    "max_tokens":2048 ,
                    "temperature":0.1 
                    },
                    timeout =300 
                    )
                    break 

                except (requests .exceptions .ReadTimeout ,requests .exceptions .ConnectionError )as e :
                    print (f"   ⏱️  Request failed: {type(e).__name__}")
                    if retry ==max_retries -1 :
                        print (f"   ❌ All {max_retries} attempts failed, skipping sample")
                        raise 
                    print (f"   🔄 Retrying in 3 seconds...")
                    time .sleep (3 )

            if response .status_code ==200 :
                model_response =response .json ()["choices"][0 ]["message"]["content"]


                gold_extracted =extract_target_from_pred (gold_letter ,4 )
                pred_extracted =extract_target_from_pred (model_response ,4 )

                is_correct =compare_gold_target (pred_extracted ,gold_extracted )

                if is_correct :
                    correct_count +=1 

                print (f"Model Response Length: {len(model_response)} chars")
                print (f"Gold Extracted: {gold_extracted}")
                print (f"Pred Extracted: {pred_extracted}")
                print (f"Result: {'✅ CORRECT' if is_correct else '❌ WRONG'}")


                preview =model_response .replace ('\n',' ')[:200 ]
                print (f"Response Preview: {preview}...")


                sample_result .update ({
                "correct":bool (is_correct ),
                "model_response":str (model_response ),
                "gold_extracted":[str (x )for x in gold_extracted ]if isinstance (gold_extracted ,list )else [
                str (gold_extracted )]if gold_extracted else [],
                "pred_extracted":[str (x )for x in pred_extracted ]if isinstance (pred_extracted ,list )else [
                str (pred_extracted )]if pred_extracted else [],
                "response_length":int (len (model_response )),
                "response_preview":str (preview ),
                "status":"completed",
                "end_timestamp":datetime .now ().isoformat ()
                })

                results .append ({
                "sample_id":i ,
                "trial_num":trial_num ,
                "correct":bool (is_correct ),
                "gold_letter":str (gold_letter ),
                "gold_extracted":[str (x )for x in gold_extracted ]if isinstance (gold_extracted ,list )else [
                str (gold_extracted )]if gold_extracted else [],
                "pred_extracted":[str (x )for x in pred_extracted ]if isinstance (pred_extracted ,list )else [
                str (pred_extracted )]if pred_extracted else [],
                "response_length":int (len (model_response )),
                "response_preview":str (preview )
                })

            else :
                print (f"❌ API Error: {response.status_code}")
                print (f"Response: {response.text}")

                sample_result .update ({
                "correct":False ,
                "error":f"API_ERROR_{response.status_code}",
                "error_details":str (response .text ),
                "status":"failed",
                "end_timestamp":datetime .now ().isoformat ()
                })

                results .append ({
                "sample_id":i ,
                "trial_num":trial_num ,
                "correct":False ,
                "error":f"API_ERROR_{response.status_code}",
                "gold_letter":gold_letter 
                })

        except Exception as e :
            print (f"❌ Error: {e}")

            sample_result .update ({
            "correct":False ,
            "error":str (e ),
            "status":"failed",
            "end_timestamp":datetime .now ().isoformat ()
            })

            results .append ({
            "sample_id":i ,
            "trial_num":trial_num ,
            "correct":False ,
            "error":str (e ),
            "gold_letter":gold_letter 
            })


        running_average =correct_count /(i +1 )
        print (
        f"📈 {trial_prefix}Running Score: {correct_count}/{i + 1} = {running_average:.4f} ({running_average * 100:.2f}%)")
        print (f"{'─' * 60}")


        if comprehensive_results is not None and trial_num is not None :
            current_trial =comprehensive_results ["trials"][-1 ]
            current_trial ["samples"].append (sample_result )

            current_trial ["trial_statistics"].update ({
            "completed_samples":i +1 ,
            "correct_count":correct_count ,
            "current_score":running_average 
            })

            comprehensive_results ["overall_statistics"]["total_samples_evaluated"]=sum (
            len (trial ["samples"])for trial in comprehensive_results ["trials"]
            )


            if evaluation_results_path :
                try :
                    with open (evaluation_results_path ,'w')as f :
                        json .dump (comprehensive_results ,f ,indent =2 )
                    print (f"💾 Progress saved: Sample {i + 1}/{total_samples} (Score: {running_average:.4f})")
                except Exception as save_error :
                    print (f"⚠️ Failed to save progress: {save_error}")


    average_score =correct_count /total_samples if total_samples >0 else 0.0 

    print (f"\n{'=' * 80}")
    print (f"📊 {trial_prefix}FINAL RESULTS:")
    print (f"   Total Samples: {total_samples}")
    print (f"   Correct: {correct_count}")
    print (f"   Wrong: {total_samples - correct_count}")
    print (f"   Average Score: {average_score:.4f} ({average_score * 100:.2f}%)")
    print (f"{'=' * 80}")


    if comprehensive_results is not None and trial_num is not None :
        current_trial =comprehensive_results ["trials"][-1 ]
        current_trial ["trial_statistics"].update ({
        "final_score":average_score ,
        "status":"completed",
        "end_timestamp":datetime .now ().isoformat ()
        })


    if trial_num is None :
        timestamp =datetime .now ().strftime ("%Y%m%d_%H%M%S")
        results_file =f"gpqa_debug_results_{timestamp}.json"

        summary ={
        "timestamp":timestamp ,
        "total_samples":total_samples ,
        "dataset_length":len (dataset ),
        "requested_max_samples":max_samples ,
        "correct_count":correct_count ,
        "average_score":average_score ,
        "detailed_results":results 
        }

        with open (results_file ,'w')as f :
            json .dump (summary ,f ,indent =2 )

        print (f"📄 Detailed results saved to: {results_file}")

    return average_score ,results 


def main ():
    parser =argparse .ArgumentParser (description ="Debug GPQA-D evaluation pipeline")
    parser .add_argument ("--max_samples",type =int ,default =198 ,
    help ="Maximum number of samples to test (default: 198)")
    parser .add_argument ("--repeat",type =int ,default =3 ,
    help ="Number of times to repeat the evaluation (default: 3)")
    parser .add_argument ("--evaluation_results_path",type =str ,default =None ,
    help ="Path to save continuous evaluation results (default: auto-generated)")
    parser .add_argument ("--skip_basic_tests",action ="store_true",
    help ="Skip basic extraction and PassAtK tests")
    parser .add_argument ("--server_host",type =str ,default ="slurm0us-gufnodeset-0",
    help ="Server host address (default: slurm0us-gufnodeset-0)")
    parser .add_argument ("--server_port",type =int ,default =8080 ,
    help ="Server port (default: 8080)")

    args =parser .parse_args ()


    if args .evaluation_results_path is None :
        timestamp =datetime .now ().strftime ("%Y%m%d_%H%M%S")
        args .evaluation_results_path =f"gpqa_evaluation_results_{timestamp}.json"

    print ("🚀 DEBUGGING GPQA-D EVALUATION PIPELINE")
    print (f"🌐 Server: http://{args.server_host}:{args.server_port}")
    print (f"🔁 Repeats: {args.repeat}")
    print (f"📊 Samples per trial: {args.max_samples}")
    print (f"💾 Results path: {args.evaluation_results_path}")
    print ("="*80 )

    if not args .skip_basic_tests :
        print ("1️⃣  Testing extraction pipeline with sample data...")
        debug_extraction_pipeline ()

        print ("\n2️⃣  Testing with single real dataset sample...")
        test_with_real_dataset (args .server_host ,args .server_port )


    comprehensive_results ={
    "metadata":{
    "start_timestamp":datetime .now ().isoformat (),
    "repeat":args .repeat ,
    "max_samples":args .max_samples ,
    "server_host":args .server_host ,
    "server_port":args .server_port ,
    "evaluation_results_path":args .evaluation_results_path ,
    "status":"in_progress"
    },
    "trials":[],
    "overall_statistics":{
    "completed_trials":0 ,
    "total_samples_evaluated":0 ,
    "current_mean_score":0.0 ,
    "trial_scores":[]
    }
    }


    print (f"\n3️⃣  Testing multiple trials (repeat={args.repeat}, max_samples={args.max_samples})...")

    trial_scores =[]
    all_results =[]

    for trial in range (args .repeat ):
        print (f"\n{'🟦' * 20} TRIAL {trial + 1}/{args.repeat} {'🟦' * 20}")

        trial_score ,trial_results =test_multiple_samples (
        args .max_samples ,args .server_host ,args .server_port ,
        trial_num =trial +1 ,evaluation_results_path =args .evaluation_results_path ,
        comprehensive_results =comprehensive_results 
        )

        trial_scores .append (trial_score )
        all_results .extend (trial_results )


        comprehensive_results ["overall_statistics"]["completed_trials"]=trial +1 
        comprehensive_results ["overall_statistics"]["total_samples_evaluated"]=len (all_results )
        comprehensive_results ["overall_statistics"]["trial_scores"]=trial_scores 
        if trial_scores :
            comprehensive_results ["overall_statistics"]["current_mean_score"]=sum (trial_scores )/len (trial_scores )


        with open (args .evaluation_results_path ,'w')as f :
            json .dump (comprehensive_results ,f ,indent =2 )

        print (f"\n🎯 TRIAL {trial + 1}/{args.repeat} SCORE: {trial_score:.4f} ({trial_score * 100:.2f}%)")
        print (f"💾 Progress saved to: {args.evaluation_results_path}")
        print (f"{'🟦' * (42 + len(str(trial + 1)) + len(str(args.repeat)))}")


    if trial_scores :
        import statistics 

        mean_score =statistics .mean (trial_scores )
        if len (trial_scores )>1 :
            std_score =statistics .stdev (trial_scores )
            min_score =min (trial_scores )
            max_score =max (trial_scores )
        else :
            std_score =0.0 
            min_score =max_score =mean_score 


        comprehensive_results ["metadata"]["end_timestamp"]=datetime .now ().isoformat ()
        comprehensive_results ["metadata"]["status"]="completed"
        comprehensive_results ["overall_statistics"].update ({
        "final_mean_score":mean_score ,
        "std_score":std_score ,
        "min_score":min_score ,
        "max_score":max_score ,
        "total_trials":len (trial_scores ),
        "individual_trial_scores":[
        {"trial_num":i +1 ,"score":score }
        for i ,score in enumerate (trial_scores )
        ]
        })


        with open (args .evaluation_results_path ,'w')as f :
            json .dump (comprehensive_results ,f ,indent =2 )

        print (f"\n{'=' * 80}")
        print (f"🏆 OVERALL RESULTS ACROSS {args.repeat} TRIALS:")
        print (f"   Mean Score: {mean_score:.4f} ({mean_score * 100:.2f}%)")
        if len (trial_scores )>1 :
            print (f"   Std Dev: {std_score:.4f}")
            print (f"   Min Score: {min_score:.4f} ({min_score * 100:.2f}%)")
            print (f"   Max Score: {max_score:.4f} ({max_score * 100:.2f}%)")

        print (f"   Individual Trial Scores:")
        for i ,score in enumerate (trial_scores ):
            print (f"     Trial {i + 1}: {score:.4f} ({score * 100:.2f}%)")
        print (f"{'=' * 80}")

        print (f"📄 Final comprehensive results saved to: {args.evaluation_results_path}")

        return mean_score 
    else :

        comprehensive_results ["metadata"]["end_timestamp"]=datetime .now ().isoformat ()
        comprehensive_results ["metadata"]["status"]="failed"

        with open (args .evaluation_results_path ,'w')as f :
            json .dump (comprehensive_results ,f ,indent =2 )

        print ("❌ No trials completed successfully")
        print (f"📄 Failed results saved to: {args.evaluation_results_path}")
        return 0.0 


if __name__ =="__main__":
    final_score =main ()