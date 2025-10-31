


import json 
import requests 
import argparse 
import time 
import re 
from datetime import datetime 


def extract_numbers_from_complex_expression (extracted_data ):

    if not extracted_data :
        return []

    numbers =[]


    if isinstance (extracted_data ,(list ,tuple ,set )):
        for item in extracted_data :
            numbers .extend (extract_numbers_from_single_item (item ))
    else :
        numbers .extend (extract_numbers_from_single_item (extracted_data ))


    seen =set ()
    unique_numbers =[]
    for num in numbers :
        if num not in seen :
            seen .add (num )
            unique_numbers .append (num )

    return unique_numbers 


def extract_numbers_from_single_item (item ):

    numbers =[]


    if isinstance (item ,(int ,float )):
        numbers .append (item )
        return numbers 


    item_str =str (item )



    number_pattern =r'-?\d+\.?\d*'
    found_numbers =re .findall (number_pattern ,item_str )

    for num_str in found_numbers :
        try :

            if '.'in num_str :
                num =float (num_str )

                if num .is_integer ():
                    num =int (num )
            else :
                num =int (num_str )
            numbers .append (num )
        except ValueError :
            continue 

    return numbers 


def robust_numerical_comparison (pred_extracted ,gold_extracted ):

    try :

        from lighteval .metrics .dynamic_metrics import compare_gold_target 

        standard_result =compare_gold_target (pred_extracted ,gold_extracted )
        if standard_result :
            return True ,"standard_match"


        pred_numbers =extract_numbers_from_complex_expression (pred_extracted )
        gold_numbers =extract_numbers_from_complex_expression (gold_extracted )


        for pred_num in pred_numbers :
            for gold_num in gold_numbers :
                if abs (pred_num -gold_num )<1e-6 :
                    return True ,f"numerical_match_found_{pred_num}"

        return False ,f"no_match_pred_{pred_numbers}_gold_{gold_numbers}"

    except Exception as e :

        pred_str =str (pred_extracted ).strip ()
        gold_str =str (gold_extracted ).strip ()

        if pred_str ==gold_str :
            return True ,"string_match"

        return False ,f"extraction_error_{str(e)}"


def debug_extraction_pipeline ():


    try :

        from lighteval .metrics .dynamic_metrics import (
        ExprExtractionConfig ,
        LatexExtractionConfig ,
        compare_gold_target ,
        extract_target_from_pred ,
        get_extraction_regexes ,
        )
        from lighteval .utils .language import Language 

        print ("✅ Successfully imported lighteval functions")

    except ImportError as e :
        print (f"❌ Failed to import: {e}")
        print ("Please check the lighteval installation and paths")
        return 


    test_cases =[
    {
    "problem":"Find the sum of all positive integers less than 100 that are divisible by 7.",
    "gold_answer":"693",
    "model_responses":[
    "The answer is 693.",
    "693",
    "The sum is $693$.",
    "\\boxed{693}",
    "After calculating, I get 693 as the final answer.",
    "Let me work through this step by step...\n\nThe multiples of 7 less than 100 are: 7, 14, 21, ..., 98.\nThis gives us: 7 + 14 + 21 + ... + 98 = 693",
    "I think the answer is around 700, maybe 693?",
    "The final result is 693.",

    "The equation gives us N = 693.",
    "Solving: x^2 + 693 = 1386, so the answer is 693",
    ]
    }
    ]


    extraction_regexes =get_extraction_regexes (
    formatted_doc =None ,
    target_types =[ExprExtractionConfig (),LatexExtractionConfig ()],
    language =Language .ENGLISH ,
    )

    print ("🔍 EXTRACTION REGEXES:")
    for i ,regex in enumerate (extraction_regexes ):
        print (f"  {i + 1}. {regex}")
    print ("="*80 )

    for test_case in test_cases :
        print (f"📝 PROBLEM: {test_case['problem']}")
        print (f"🎯 GOLD ANSWER: {test_case['gold_answer']}")
        print ("="*80 )


        gold_extracted =extract_target_from_pred (test_case ['gold_answer'],extraction_regexes )
        print (f"✨ GOLD EXTRACTED: '{gold_extracted}'")
        print ()


        for i ,response in enumerate (test_case ['model_responses']):
            print (f"🤖 MODEL RESPONSE {i + 1}:")
            print (f"   Raw: {response}")


            pred_extracted =extract_target_from_pred (response ,extraction_regexes )
            print (f"   Extracted: '{pred_extracted}'")


            is_correct ,match_type =robust_numerical_comparison (pred_extracted ,gold_extracted )
            print (f"   Robust Comparison: {is_correct} ({'✅ MATCH' if is_correct else '❌ NO MATCH'}) - {match_type}")


            try :
                original_result =compare_gold_target (pred_extracted ,gold_extracted )
                print (f"   Original Comparison: {original_result} ({'✅' if original_result else '❌'})")
            except Exception as e :
                print (f"   Original Comparison ERROR: {e}")

            print ()

        print ("="*80 )


def test_with_real_dataset (server_host ="slurm0us-a3nodeset-0",server_port =8080 ):


    try :
        from datasets import load_dataset 


        dataset =load_dataset ("yentinglin/aime_2025",split ="train")
        sample =dataset [0 ]

        print ("📊 REAL DATASET SAMPLE:")
        print (json .dumps (dict (sample ),indent =2 ))
        print ("="*80 )


        problem =sample .get ('problem',sample .get ('question',''))
        gold_answer =sample .get ('answer',sample .get ('solution',''))

        print (f"📝 PROBLEM: {problem}")
        print (f"🎯 GOLD ANSWER: {gold_answer}")
        print (f"🌐 Testing with server: http://{server_host}:{server_port}")
        print ("="*80 )


        max_retries =3 
        for retry in range (max_retries ):
            try :
                print (f"🔄 Attempting API call (attempt {retry + 1}/{max_retries})...")
                api_url =f"http://{server_host}:{server_port}/v1/chat/completions"
                response =requests .post (
                api_url ,
                headers ={"Content-Type":"application/json","Authorization":"Bearer sk-test"},
                json ={
                "model":"gpt-4o",
                "messages":[{"role":"user","content":problem }],
                "max_tokens":4096 ,
                "temperature":0.1 
                },
                timeout =1200 
                )
                break 
            except requests .exceptions .ReadTimeout as e :
                print (f"⏱️  Request timed out on attempt {retry + 1}")
                if retry ==max_retries -1 :
                    raise 
                print (f"🔄 Retrying in 5 seconds...")
                time .sleep (5 )
            except requests .exceptions .ConnectionError as e :
                print (f"🔌 Connection error on attempt {retry + 1}: {e}")
                if retry ==max_retries -1 :
                    raise 
                print (f"🔄 Retrying in 10 seconds...")
                time .sleep (10 )

        try :

            if response .status_code ==200 :
                model_response =response .json ()["choices"][0 ]["message"]["content"]
                print ("🤖 MODEL RESPONSE:")
                print (model_response )
                print ("="*80 )


                from lighteval .metrics .dynamic_metrics import (
                ExprExtractionConfig ,
                LatexExtractionConfig ,
                compare_gold_target ,
                extract_target_from_pred ,
                get_extraction_regexes ,
                )
                from lighteval .utils .language import Language 

                extraction_regexes =get_extraction_regexes (
                formatted_doc =None ,
                target_types =[ExprExtractionConfig (),LatexExtractionConfig ()],
                language =Language .ENGLISH ,
                )

                gold_extracted =extract_target_from_pred (gold_answer ,extraction_regexes )
                pred_extracted =extract_target_from_pred (model_response ,extraction_regexes )

                print (f"✨ GOLD EXTRACTED: '{gold_extracted}'")
                print (f"🔍 PRED EXTRACTED: '{pred_extracted}'")

                try :

                    is_correct ,match_type =robust_numerical_comparison (pred_extracted ,gold_extracted )
                    print (f"🎯 ROBUST COMPARISON RESULT: {is_correct} - {match_type}")


                    original_comparison =compare_gold_target (pred_extracted ,gold_extracted )
                    print (f"🎯 ORIGINAL COMPARISON RESULT: {original_comparison}")


                    print (f"🧪 MANUAL COMPARISON:")
                    print (f"   Gold: '{gold_extracted}' (type: {type(gold_extracted)})")
                    print (f"   Pred: '{pred_extracted}' (type: {type(pred_extracted)})")
                    print (f"   String equality: {str(gold_extracted) == str(pred_extracted)}")

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

    except Exception as e :
        print (f"❌ DATASET ERROR: {e}")
        import traceback 
        traceback .print_exc ()


def debug_pass_at_k_computation ():


    try :
        from lighteval .metrics .dynamic_metrics import (
        ExprExtractionConfig ,
        LatexExtractionConfig ,
        compare_gold_target ,
        extract_target_from_pred ,
        get_extraction_regexes ,
        )
        from lighteval .metrics .metrics_sample import PassAtK 
        from lighteval .utils .language import Language 

        print ("🧮 TESTING PASS@K COMPUTATION")
        print ("="*80 )


        def robust_scoring_function (pred ,gold ,doc ):
            extraction_regexes =get_extraction_regexes (
            formatted_doc =None ,
            target_types =[ExprExtractionConfig (),LatexExtractionConfig ()],
            language =Language .ENGLISH ,
            )

            pred_extracted =extract_target_from_pred (pred ,extraction_regexes )
            gold_extracted =extract_target_from_pred (gold ,extraction_regexes )

            is_correct ,_ =robust_numerical_comparison (pred_extracted ,gold_extracted )
            return is_correct 


        pass_at_k =PassAtK (
        k =1 ,
        n =1 ,
        strip_strings =True ,
        normalize_gold =lambda k :extract_target_from_pred (
        k ,
        get_extraction_regexes (
        formatted_doc =None ,
        target_types =[ExprExtractionConfig (),LatexExtractionConfig ()],
        language =Language .ENGLISH ,
        ),
        ),
        normalize_pred =lambda k :extract_target_from_pred (
        k ,
        get_extraction_regexes (
        formatted_doc =None ,
        target_types =[ExprExtractionConfig (),LatexExtractionConfig ()],
        language =Language .ENGLISH ,
        ),
        ),
        sample_scoring_function =robust_scoring_function ,
        )


        test_predictions =[
        "693",
        "The answer is 693",
        "I think it's 694",
        "N = 693 from the equation",
        "Solving gives x^2 + 693 = total"
        ]
        test_gold ="693"

        for pred in test_predictions :
            try :
                result =pass_at_k .compute ([pred ],[test_gold ])
                print (f"📊 Prediction: '{pred}' -> Score: {result}")
            except Exception as e :
                print (f"❌ Error with prediction '{pred}': {e}")
                import traceback 
                traceback .print_exc ()

    except Exception as e :
        print (f"❌ PASS@K DEBUG ERROR: {e}")
        import traceback 
        traceback .print_exc ()


def test_multiple_samples (max_samples =3 ,server_host ="slurm0us-a3nodeset-0",server_port =8080 ,
trial_num =None ,evaluation_results_path =None ,comprehensive_results =None ):


    try :
        from datasets import load_dataset 
        from lighteval .metrics .dynamic_metrics import (
        ExprExtractionConfig ,
        LatexExtractionConfig ,
        compare_gold_target ,
        extract_target_from_pred ,
        get_extraction_regexes ,
        )
        from lighteval .utils .language import Language 

        trial_prefix =f"[Trial {trial_num}] "if trial_num is not None else ""

        print (f"🧪 {trial_prefix}TESTING MULTIPLE SAMPLES (max_samples={max_samples})")
        print (f"🌐 Using server: http://{server_host}:{server_port}")
        print ("⏰ Note: Each API request has a 20-minute timeout with retry logic")
        print ("💡 Using robust numerical comparison to handle format mismatches")
        print ("="*80 )


        dataset =load_dataset ("yentinglin/aime_2025",split ="train")


        print (f"📊 Dataset Info: {len(dataset)} total samples available")


        total_samples =min (max_samples ,len (dataset ))
        if total_samples <max_samples :
            print (
            f"⚠️  Note: Dataset only has {len(dataset)} samples, so evaluating {total_samples} instead of {max_samples}")

        print (f"🎯 Evaluating {total_samples} samples")
        print ("="*80 )


        extraction_regexes =get_extraction_regexes (
        formatted_doc =None ,
        target_types =[ExprExtractionConfig (),LatexExtractionConfig ()],
        language =Language .ENGLISH ,
        )

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
            problem =sample .get ('problem',sample .get ('question',''))
            gold_answer =sample .get ('answer',sample .get ('solution',''))

            print (f"\n📝 {trial_prefix}SAMPLE {i + 1}/{total_samples}")
            print (f"Problem: {problem[:150]}..."if len (problem )>150 else f"Problem: {problem}")
            print (f"Gold Answer: {gold_answer}")

            sample_start_time =datetime .now ()
            sample_result ={
            "sample_id":i ,
            "trial_num":trial_num ,
            "problem":problem ,
            "gold_answer":str (gold_answer ),
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
                        "messages":[{"role":"user","content":problem }],
                        "max_tokens":4096 ,
                        "temperature":0.1 
                        },
                        timeout =1200 
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


                    gold_extracted =extract_target_from_pred (gold_answer ,extraction_regexes )
                    pred_extracted =extract_target_from_pred (model_response ,extraction_regexes )


                    is_correct ,match_type =robust_numerical_comparison (pred_extracted ,gold_extracted )


                    try :
                        original_correct =compare_gold_target (pred_extracted ,gold_extracted )
                    except :
                        original_correct =False 

                    if is_correct :
                        correct_count +=1 

                    print (f"Model Response Length: {len(model_response)} chars")
                    print (f"Gold Extracted: {gold_extracted}")
                    print (f"Pred Extracted: {pred_extracted}")
                    print (f"Robust Result: {'✅ CORRECT' if is_correct else '❌ WRONG'} ({match_type})")
                    print (f"Original Result: {'✅' if original_correct else '❌'}")


                    preview =model_response .replace ('\n',' ')[:200 ]
                    print (f"Response Preview: {preview}...")


                    sample_result .update ({
                    "correct":bool (is_correct ),
                    "match_type":str (match_type ),
                    "original_correct":bool (original_correct ),
                    "model_response":str (model_response ),
                    "gold_extracted":[str (x )for x in gold_extracted ]if isinstance (gold_extracted ,list )else str (
                    gold_extracted ),
                    "pred_extracted":[str (x )for x in pred_extracted ]if isinstance (pred_extracted ,list )else str (
                    pred_extracted ),
                    "response_length":int (len (model_response )),
                    "response_preview":str (preview ),
                    "status":"completed",
                    "end_timestamp":datetime .now ().isoformat ()
                    })

                    results .append ({
                    "sample_id":i ,
                    "trial_num":trial_num ,
                    "correct":bool (is_correct ),
                    "match_type":str (match_type ),
                    "original_correct":bool (original_correct ),
                    "gold_answer":str (gold_answer ),
                    "gold_extracted":[str (x )for x in gold_extracted ]if isinstance (gold_extracted ,list )else str (
                    gold_extracted ),
                    "pred_extracted":[str (x )for x in pred_extracted ]if isinstance (pred_extracted ,list )else str (
                    pred_extracted ),
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
                    "gold_answer":gold_answer 
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
                "gold_answer":gold_answer 
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
            results_file =f"aime_debug_results_{timestamp}.json"

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

    except Exception as e :
        print (f"❌ MULTIPLE SAMPLES ERROR: {e}")
        import traceback 
        traceback .print_exc ()
        return 0.0 ,[]


def main ():
    parser =argparse .ArgumentParser (description ="Debug AIME evaluation pipeline")
    parser .add_argument ("--max_samples",type =int ,default =30 ,
    help ="Maximum number of samples to test (default: 30)")
    parser .add_argument ("--repeat",type =int ,default =16 ,
    help ="Number of times to repeat the evaluation (default: 16)")
    parser .add_argument ("--evaluation_results_path",type =str ,default =None ,
    help ="Path to save continuous evaluation results (default: auto-generated)")
    parser .add_argument ("--skip_basic_tests",action ="store_true",
    help ="Skip basic extraction and PassAtK tests")
    parser .add_argument ("--server_host",type =str ,default ="slurm0us-gufnodeset-2",
    help ="Server host address (default: slurm0us-gufnodeset-2)")
    parser .add_argument ("--server_port",type =int ,default =8080 ,
    help ="Server port (default: 8080)")

    args =parser .parse_args ()


    if args .evaluation_results_path is None :
        timestamp =datetime .now ().strftime ("%Y%m%d_%H%M%S")
        args .evaluation_results_path =f"aime_evaluation_results_{timestamp}.json"

    print ("🚀 DEBUGGING AIME EVALUATION PIPELINE")
    print (f"🌐 Server: http://{args.server_host}:{args.server_port}")
    print (f"🔁 Repeats: {args.repeat}")
    print (f"📊 Samples per trial: {args.max_samples}")
    print (f"💾 Results path: {args.evaluation_results_path}")
    print ("💡 Using robust numerical extraction to handle format mismatches")
    print ("="*80 )

    if not args .skip_basic_tests :
        print ("1️⃣  Testing extraction pipeline with sample data...")
        debug_extraction_pipeline ()

        print ("\n2️⃣  Testing with single real dataset sample...")
        test_with_real_dataset (args .server_host ,args .server_port )

        print ("\n3️⃣  Testing Pass@K computation...")
        debug_pass_at_k_computation ()


    comprehensive_results ={
    "metadata":{
    "start_timestamp":datetime .now ().isoformat (),
    "repeat":args .repeat ,
    "max_samples":args .max_samples ,
    "server_host":args .server_host ,
    "server_port":args .server_port ,
    "evaluation_results_path":args .evaluation_results_path ,
    "robust_comparison":True ,
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


    print (f"\n4️⃣  Testing multiple trials (repeat={args.repeat}, max_samples={args.max_samples})...")

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