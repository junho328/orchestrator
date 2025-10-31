

import json 
import hashlib 
import re 
import math 
import os 
import glob 
from typing import Dict ,List ,Any ,Iterator ,Optional 
from datasets import Dataset 
from guf .core import Task 
import numpy as np 
from collections import Counter 


def generate_ngrams (text :str ,n_gram :int )->Iterator [List [str ]]:

    s =set ()
    for p in range (1 ,len (text )):
        r =[]
        for n in range (1 ,min (n_gram +1 ,p )):
            t =text [p -n :p ]
            if t in s :
                continue 
            s .add (t )
            r .append (t )
        yield r 


class NgramScorer :


    def __init__ (self ,answers :List [str ],fluency_n_gram :int =10 ,truthfulness_n_gram :int =3 ):
        self .answers =answers 
        self .fluency_n_gram =fluency_n_gram 
        self .truthfulness_n_gram =truthfulness_n_gram 
        self .dist :Dict [str ,int ]={}
        self .baseline :float =1.0 
        self .build ()

    def build (self )->None :

        for answer in self .answers :
            for tt in generate_ngrams (f"^{answer}$",self .fluency_n_gram ):
                for t in tt :
                    self .dist [t ]=self .dist .get (t ,0 )+1 
        baseline =0.0 
        for answer in self .answers :
            baseline +=self .score_fluency (answer )[0 ]
        self .baseline =baseline /len (self .answers )

    def score_fluency (self ,answer :str )->tuple [float ,float ]:

        score =0 
        best =(0.0 ,1.0 )
        for n ,tt in enumerate (generate_ngrams (f"^{answer}$"[:202 ],self .fluency_n_gram )):
            for t in tt :
                score +=self .dist .get (t ,0 )
            if n ==0 :
                continue 
            discount =1 -max (n -100 ,0 )/50 
            s =score *discount /self .baseline 
            if s >best [0 ]:
                best =(s ,discount )
        return best 

    def score_truthfulness (self ,answer :str )->float :

        text =f"^{answer}$"[:202 ]
        a =[0 for _ in range (len (text ))]
        for i in range (0 ,len (text )-self .truthfulness_n_gram +1 ):
            t =text [i :i +self .truthfulness_n_gram ]
            if t in self .dist :
                for j in range (i ,i +self .truthfulness_n_gram ):
                    a [j ]=max (a [j ],self .dist [t ])
        total =0.0 
        count =0 
        score =0.0 
        best =0.0 
        for n ,(c ,s )in enumerate (zip (text ,a )):
            if c in "^$、。・「」『』（）【】［］〈〉《》":
                continue 
            total +=min (1.0 ,s /len (self .answers )*200.0 )
            count +=1 
            score =total /count *(1 -max (n -100 ,0 )/50 )
            if n >=100 :
                best =max (best ,score )
        return max (best ,score )


class KeywordScorer :


    def __init__ (self ,keywords :List [Dict [str ,Any ]]):
        self .keywords =keywords 

    def match (self ,answer :str ,keyword :Dict [str ,Any ])->tuple [int ,str ]:

        if "t"in keyword :

            r =re .search (keyword ["t"],answer )
            return r .end ()if r else 9999 ,keyword .get ("name",keyword ["t"])
        if "and"in keyword :
            xs =[self .match (answer ,x )for x in keyword ["and"]]
            xs .sort (key =lambda x :-x [0 ])
            return xs [0 ][0 ],keyword .get ("name",xs [0 ][1 ])
        if "or"in keyword :
            xs =[self .match (answer ,x )for x in keyword ["or"]]
            xs .sort (key =lambda x :x [0 ])
            return xs [0 ][0 ],keyword .get ("name",xs [0 ][1 ])
        raise ValueError (f"Invalid keyword: {keyword}")

    def score (self ,answer :str )->tuple [float ,List [tuple [str ,float ]]]:

        results =[]
        scores =[1 -max (i -100 ,0 )/50 for i in range (len (answer )+1 )]
        scores =[s for s in scores if s >=0 ]

        for k in self .keywords :
            r =self .match (answer ,k )+(1 -k .get ("importance",1.0 ),)
            for i in range (min (len (scores ),r [0 ])):
                scores [i ]*=r [2 ]
            results .append (r )

        n =max (reversed (range (len (scores ))),key =lambda x :scores [x ])if scores else 0 
        return scores [n ]if scores else 0.0 ,[r [1 :]for r in results if n <r [0 ]]+(
        [(f"{n - 100}字超過",1 -max (n -100 ,0 )/50 )]if n >100 and (scores [n ]if scores else 0 )>0 else []
        )


class Scorer :


    def __init__ (self ,metadata :Dict [str ,Any ],fluency_n_gram :int =10 ,truthfulness_n_gram :int =3 ):
        self .metadata =metadata 
        self .ngram_scorers :Dict [str ,NgramScorer ]={}


        for k ,v in metadata ["answers"].items ():
            self .ngram_scorers [k ]=NgramScorer (
            v ,
            fluency_n_gram =fluency_n_gram ,
            truthfulness_n_gram =truthfulness_n_gram ,
            )


        self .keyword_scorer =KeywordScorer (metadata ["keywords"])

    def score (self ,answer :str )->Dict [str ,Any ]:

        scores :Dict [str ,Any ]={"fluency":{},"fluency_discount":1.0 ,"truthfulness":{}}


        for k ,v in self .ngram_scorers .items ():
            fluency ,discount =v .score_fluency (answer )
            scores ["fluency"][k ]=round (fluency /len (self .ngram_scorers ),6 )
            scores ["fluency_discount"]=round (max (scores ["fluency_discount"],discount ),2 )
            scores ["truthfulness"][k ]=round (
            v .score_truthfulness (answer )/len (self .ngram_scorers ),6 
            )


        helpfulness ,results =self .keyword_scorer .score (answer [:200 ])
        scores ["helpfulness"]=round (helpfulness ,5 )
        scores ["helpfulness_results"]=results 


        scores ["average"]=round (
        (
        sum (scores ["fluency"].values ())
        +sum (scores ["truthfulness"].values ())
        +scores ["helpfulness"]
        )
        /3 ,
        5 ,
        )

        return scores 


class PFGenTask (Task ):



    JAPANESE_ROUTER_SYSTEM_PROMPT =(
    "あなたは{num_agents}つのエージェントを調整して問題を解決するメッセージディスパッチャーです。"
    "問題と議論の履歴を確認し、次にどのエージェントが応答すべきかを決定してください。"
    "あなたの応答は1から{num_agents}までの数字である必要があります。1は最初のエージェント、2は2番目のエージェントなどです。"
    "エージェントが連続して選択された場合、その答えが返されます。"
    )

    JAPANESE_SYSTEM_PROMPT =(
    "あなたは役に立つアシスタントです。まず心の中で推論プロセスを考え、その後ユーザーに答えを提供します。"
    )

    JAPANESE_ASSISTANT_PROMPT ="この問題を段階的に解決していきます。\n<think>"

    JAPANESE_COLLABORATION_PROMPT =(
    "他のエージェントの思考を<Agent i>タグで見ることができます。ここでiはエージェントのIDです。"
    "その場合は、他のエージェントが提供した情報に基づいて問題を解決するようにしてください。"
    )


    USER_PROMPT_TEMPLATE =(
    "例と同様の文体及び文字数で、質問に1行で答えてください。"
    "回答は必ず<回答> </回答>タグで囲んでください。\n\n"
    "## 回答例\n"
    "{examples}\n\n"
    "## 質問\n"
    "Q: {question}\n"
    "A:"
    )

    def __init__ (
    self ,
    llm_names :List [str ],
    seed :int =42 ,
    max_tokens :int =512 ,
    temperature :float =0.8 ,
    max_turns :int =5 ,
    num_examples :int =20 ,
    data_dir :str ="your/path/here",
    servers :Dict [str ,str ]=None ,
    ports :Dict [str ,int ]=None ,
    track_costs :bool =True ,
    debug :bool =False ,
    together :bool =True ,
    valid_ratio :float =0.5 ,
    max_samples :int =-1 ,
    test_ratio :float =0.2 ,
    ):

        super ().__init__ (
        llm_names ,
        seed =seed ,
        max_tokens =max_tokens ,
        temperature =temperature ,
        max_turns =max_turns ,
        servers =servers ,
        ports =ports ,
        track_costs =track_costs ,
        debug =debug ,
        together =together ,
        valid_ratio =valid_ratio ,
        max_samples =max_samples ,
        test_ratio =test_ratio 
        )
        self .num_examples =num_examples 
        self .data_dir =data_dir 
        self .metadata =self ._load_metadata ()
        self .questions_data =self ._extract_questions_data ()


        self .training_examples :Optional [List [Dict [str ,str ]]]=None 
        self .current_split :Optional [str ]=None 


        if not self .use_structured_router :
            self .system_prompt =self ._get_system_prompt ()
            self .assistant_prompt =self ._get_assistant_prompt ()
            self .router_system_prompt =self ._get_router_system_prompt ()
            self .collaboration_prompt =self ._get_collaboration_prompt ()


    def _get_router_system_prompt (self )->str :

        return self .JAPANESE_ROUTER_SYSTEM_PROMPT 

    def _get_system_prompt (self )->str :

        return self .JAPANESE_SYSTEM_PROMPT 

    def _get_assistant_prompt (self )->str :

        return self .JAPANESE_ASSISTANT_PROMPT 

    def _get_collaboration_prompt (self )->str :

        return self .JAPANESE_COLLABORATION_PROMPT 

    def _load_metadata (self )->Dict [str ,Dict [str ,Any ]]:

        metadata ={}
        metadata_paths =sorted (glob .glob (os .path .join (self .data_dir ,"Q*.json")))

        if not metadata_paths :
            raise FileNotFoundError (f"No Q*.json files found in {self.data_dir}")

        for metadata_path in metadata_paths :
            with open (metadata_path ,'r',encoding ='utf-8')as f :
                d =json .load (f )
                metadata [d ["question"]]=d 

        print (f"Loaded {len(metadata)} questions from {self.data_dir}")
        return metadata 

    def _extract_questions_data (self )->List [Dict [str ,str ]]:

        questions =[]
        for question_text ,metadata in self .metadata .items ():
            questions .append ({
            "question":question_text ,
            "question_id":metadata ["question_id"],

            "answer":list (metadata ["answers"].values ())[0 ][0 ]if metadata ["answers"]else ""
            })
        return questions 

    def _get_user_prompt_template (self )->str :

        return self .USER_PROMPT_TEMPLATE 

    def _generate_examples (self ,question :Dict [str ,str ],trial :int )->str :


        if self .training_examples is None :
            if self .debug :
                print ("Warning: No training examples available, using all examples (potential leakage)")

            examples =[q for q in self .questions_data if question ["question"]!=q ["question"]]
        else :

            examples =[q for q in self .training_examples if question ["question"]!=q ["question"]]

            if self .debug and len (examples )<self .num_examples :
                print (f"Warning: Only {len(examples)} training examples available, "
                f"requested {self.num_examples}")


        if len (examples )==0 :
            if self .debug :
                print ("Warning: No examples available for few-shot prompting")
            return ""


        seed_str =f"{trial}::{question['question']}"
        examples .sort (key =lambda x :hashlib .sha1 (f"{seed_str}::{x['question']}".encode ()).hexdigest ())


        num_examples_to_use =min (self .num_examples ,len (examples ))

        prompt =""
        for example in examples [:num_examples_to_use ]:

            prompt +=f"Q: {example['question']}\nA: <回答>{example['answer']}</回答>\n\n"

        return prompt .strip ()

    def _format_base_prompt (self ,task_data :Dict )->str :


        examples =self ._generate_examples (task_data ,self .task_id )

        return self .user_prompt_template .format (
        examples =examples ,
        question =task_data ["question"]
        )

    def _load_data (self ,seed :int ,split :str ="train",validation :bool =False ,
    valid_ratio :float =None ,max_samples :int =None ,
    test_split :bool =False ,test_ratio :float =None )->Dict :


        if valid_ratio is None :
            valid_ratio =self .valid_ratio 
        if max_samples is None :
            max_samples =self .max_samples 
        if test_ratio is None :
            test_ratio =self .test_ratio 


        dataset_dict ={
        "question":[q ["question"]for q in self .questions_data ],
        "question_id":[q ["question_id"]for q in self .questions_data ],
        "answer":[q ["answer"]for q in self .questions_data ]
        }
        ds =Dataset .from_dict (dataset_dict )


        from guf .utils import get_or_create_indices 
        split_indices =get_or_create_indices (
        task_name ="pfgen",
        dataset_len =len (ds ),
        seed =self .split_seed ,
        valid_ratio =valid_ratio ,
        test_ratio =test_ratio 
        )


        def _take (idxs ):
            return ds .select (idxs )

        data_splits ={
        "train":_take (split_indices ["train"]),
        "valid":_take (split_indices ["valid"]),
        "test":_take (split_indices ["test"])
        }


        if max_samples !=-1 :
            data_splits ["train"]=data_splits ["train"].shuffle (seed =seed ).select (range (min (max_samples ,len (data_splits ["train"]))))
            valid_cap =int (max_samples *valid_ratio /(1.0 -valid_ratio ))if valid_ratio <1.0 else len (data_splits ["valid"])
            data_splits ["valid"]=data_splits ["valid"].shuffle (seed =seed ).select (range (min (valid_cap ,len (data_splits ["valid"]))))
            test_cap =int (max_samples *test_ratio /(1.0 -test_ratio ))if test_ratio <1.0 else len (data_splits ["test"])
            data_splits ["test"]=data_splits ["test"].shuffle (seed =seed ).select (range (min (test_cap ,len (data_splits ["test"]))))


        self .training_examples =[]
        for item in data_splits ["train"]:
            self .training_examples .append ({
            "question":item ["question"],
            "question_id":item ["question_id"],
            "answer":item ["answer"]
            })


        for k ,v in data_splits .items ():
            print (f"PFGen split {k} contains {len(v)} examples")

        if self .debug :
            print (f"Stored {len(self.training_examples)} training examples for few-shot prompting")

        return data_splits 

    def reset (self ,task_id :int =-1 ,split :str ="train"):


        self .current_split =split 


        return super ().reset (task_id ,split )

    def _extract_answer_from_tags (self ,completion :str )->str :


        if '<回答>'in completion and '</回答>'in completion :

            answer =completion .split ('<回答>')[1 ].split ('</回答>')[0 ].strip ()
            return answer 


        cleaned =completion .strip ()
        if cleaned .startswith ("A:"):
            cleaned =cleaned [2 :].strip ()

        return cleaned 

    def _calculate_reward (self ,completions :List [str ],task_data :List [Dict [str ,Any ]],debug :bool =False )->List [float ]:

        rewards :List [float ]=[]

        for completion ,data in zip (completions ,task_data ):
            question =data ["question"]


            extracted_answer =self ._extract_answer_from_tags (completion )


            if question not in self .metadata :
                print (f"Warning: Question '{question}' not found in metadata")
                rewards .append (0.0 )
                continue 

            question_metadata =self .metadata [question ]


            scorer =Scorer (question_metadata )


            score_result =scorer .score (extracted_answer )
            final_score =score_result ["average"]

            if debug :
                print ("=== DEBUG: PFGen Answer Evaluation ===")
                print (f"Question: {question}")
                print (f"Question ID: {question_metadata['question_id']}")
                print (f"Raw completion: {completion}")
                print (f"Extracted answer: {extracted_answer}")
                print (f"Answer has <回答> tags: {'<回答>' in completion and '</回答>' in completion}")
                print (f"Fluency scores: {score_result['fluency']}")
                print (f"Sum fluency: {sum(score_result['fluency'].values()):.4f}")
                print (f"Truthfulness scores: {score_result['truthfulness']}")
                print (f"Sum truthfulness: {sum(score_result['truthfulness'].values()):.4f}")
                print (f"Helpfulness: {score_result['helpfulness']:.4f}")
                print (f"Raw average: {(sum(score_result['fluency'].values()) + sum(score_result['truthfulness'].values()) + score_result['helpfulness']) / 3:.4f}")
                print (f"Final reward: {final_score:.4f}")
                if score_result .get ('helpfulness_results'):
                    print (f"Helpfulness results: {score_result['helpfulness_results']}")


                if len (scorer .ngram_scorers )>0 :
                    first_scorer =list (scorer .ngram_scorers .values ())[0 ]
                    print (f"N-gram baseline: {first_scorer.baseline:.4f}")
                    print (f"N-gram dist size: {len(first_scorer.dist)}")
                    fluency_raw ,discount =first_scorer .score_fluency (extracted_answer )
                    print (f"Raw fluency score: {fluency_raw:.4f}, discount: {discount:.4f}")
                    print (f"Truthfulness raw: {first_scorer.score_truthfulness(extracted_answer):.4f}")


                if hasattr (self ,'training_examples')and self .training_examples :
                    print (f"Using {len(self.training_examples)} training examples for few-shot")
                    print (f"Current split: {getattr(self, 'current_split', 'unknown')}")
                else :
                    print ("Warning: No training examples available (potential data leakage)")

                print ("===========================")

            rewards .append (final_score )

        return rewards 

    def calculate_leaderboard_scores (self ,completions :List [str ],task_data :List [Dict [str ,Any ]])->Dict [str ,float ]:

        all_scores ={"fluency":[],"truthfulness":[],"helpfulness":[],"length":[]}

        for completion ,data in zip (completions ,task_data ):
            question =data ["question"]


            extracted_answer =self ._extract_answer_from_tags (completion )


            all_scores ["length"].append (len (extracted_answer ))


            if question not in self .metadata :
                all_scores ["fluency"].append (0.0 )
                all_scores ["truthfulness"].append (0.0 )
                all_scores ["helpfulness"].append (0.0 )
                continue 

            question_metadata =self .metadata [question ]
            scorer =Scorer (question_metadata )
            score_result =scorer .score (extracted_answer )


            all_scores ["fluency"].append (sum (score_result ["fluency"].values ()))
            all_scores ["truthfulness"].append (sum (score_result ["truthfulness"].values ()))
            all_scores ["helpfulness"].append (score_result ["helpfulness"])


        avg_fluency =sum (all_scores ["fluency"])/len (all_scores ["fluency"])if all_scores ["fluency"]else 0.0 
        avg_truthfulness =sum (all_scores ["truthfulness"])/len (all_scores ["truthfulness"])if all_scores ["truthfulness"]else 0.0 
        avg_helpfulness =sum (all_scores ["helpfulness"])/len (all_scores ["helpfulness"])if all_scores ["helpfulness"]else 0.0 
        avg_length =sum (all_scores ["length"])/len (all_scores ["length"])if all_scores ["length"]else 0.0 


        overall_score =(avg_fluency +avg_truthfulness +avg_helpfulness )/3.0 

        return {
        "score":round (overall_score ,4 ),
        "fluency":round (avg_fluency ,3 ),
        "truthfulness":round (avg_truthfulness ,3 ),
        "helpfulness":round (avg_helpfulness ,3 ),
        "length":round (avg_length ,1 )
        }


if __name__ =="__main__":

    llms =["gpt-4o-mini","claude-3-7-sonnet-20250219","gemini-1.5-pro","deepseek-ai/DeepSeek-V3"]
    task =PFGenTask (llm_names =llms ,debug =True )


    print ("=== Testing Japanese PFGen Scoring ===")


    first_question =task .questions_data [0 ]
    print (f"Question: {first_question['question']}")
    print (f"Reference answer: {first_question['answer']}")


    print (f"\n=== Japanese Prompts ===")
    print (f"Router System: {task._get_router_system_prompt().format(num_agents=4)}")
    print (f"Agent System: {task._get_system_prompt()}")
    print (f"Assistant: {task._get_assistant_prompt()}")
    print (f"Collaboration: {task._get_collaboration_prompt()}")


    print (f"\n=== Testing Data Split Integrity ===")


    task .data_splits =task ._load_data (seed =42 ,split ="train",validation =True )
    print (f"Training examples loaded: {len(task.training_examples) if task.training_examples else 0}")


    task .reset (task_id =0 ,split ="train")
    examples =task ._generate_examples (first_question ,0 )
    print (f"Examples generated from training split only:")
    print (examples [:200 ]+"..."if len (examples )>200 else examples )


    task .reset (task_id =0 ,split ="test")
    examples_test =task ._generate_examples (first_question ,0 )
    print (f"\nExamples generated when on test split (should still use training only):")
    print (f"Same as training examples: {examples == examples_test}")


    test_completions =[
    f"<回答>{first_question['answer']}</回答>",
    f"A: <回答>{first_question['answer']}</回答>",
    f"答えは<回答>22回</回答>です。",
    f"A: {first_question['answer']}",
    "わからない",
    ]

    print (f"\n=== Answer Extraction Test ===")
    for i ,completion in enumerate (test_completions ):
        extracted =task ._extract_answer_from_tags (completion )
        print (f"Test {i+1}:")
        print (f"  Input: {completion}")
        print (f"  Extracted: {extracted}")


        score =task ._calculate_reward ([completion ],[{"question":first_question ["question"]}],debug =False )[0 ]
        print (f"  Score: {score:.4f}")
        print ()


    ref_scores =task .calculate_leaderboard_scores (
    [f"<回答>{first_question['answer']}</回答>"],
    [{"question":first_question ['question']}]
    )
    print (f"Reference answer leaderboard scores: {ref_scores}")

    print ("=== End Testing ===")


    obs =task .reset (split ="test")
    done =False 

    while not done :
        action =np .random .rand (len (llms ))
        obs ,reward ,done ,obs_act =task .step (action )

    print ("Final response:",task .response )
    print ("Reward:",reward )