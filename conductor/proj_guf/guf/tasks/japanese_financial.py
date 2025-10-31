

from __future__ import annotations 
import re ,os ,glob ,json 
from typing import Dict ,List ,Optional ,Any 
import pandas as pd 

from guf .core import Task 
from guf .utils import extract_answer 


class JapaneseFinancialTask (Task ):


    JAPANESE_ROUTER_SYSTEM_PROMPT =(
    "あなたは{num_agents}体のエージェントを調整して問題を解決するメッセージディスパッチャーです。"
    "問題と議論の履歴を確認し、次に応答すべきエージェント番号（1-{num_agents}）を数字で返してください。"
    "同じエージェントが連続して選ばれた場合、そのエージェントの直前の回答を最終回答として返します。"
    )
    JAPANESE_SYSTEM_PROMPT =(
    "あなたは有能で協力的なアシスタントです。まず頭の中で推論を行い、"
    "その後ユーザーに回答を提供してください。"
    )
    JAPANESE_ASSISTANT_PROMPT ="段階的に考えます。\n<THINKING_TAG>"
    JAPANESE_COLLABORATION_PROMPT =(
    "<Agent i>タグに他エージェントの思考が表示されます（i はエージェントID）。"
    "それらを参考にして問題解決を試みてください。"
    )



    JAPANESE_SENTIMENT_USER_PROMPT_TEMPLATE =(
    "次の文章において「{target}」に対する感情を分析してください。\n\n"
    "文章: {sentence}\n\n"
    "「{target}」への感情を「positive」「negative」「neutral」のいずれかで答えてください。\n"
    "思考過程を<THINKING_TAG> </THINKING_TAG>で示し、最終回答を<answer> </answer>で返してください。"
    "例: <answer>positive</answer>。\n"
    "<THINKING_TAG>タグ内では要点を簡潔に段階的に述べてください。"
    )

    JAPANESE_MULTIPLE_CHOICE_USER_PROMPT_TEMPLATE =(
    "{question}\n\n{context}\n\n{choices}\n\n"
    "思考過程を<THINKING_TAG> </THINKING_TAG>で示し、最終回答を<answer> </answer>で返してください。"
    "例: <answer>1</answer>。\n"
    "<answer>タグ内には選択肢番号のみを記述してください。"
    "思考は短く段階的に<THINKING_TAG>タグ内で行ってください。"
    )


    EN_SENTIMENT_USER_PROMPT_TEMPLATE =(
    "Read the following sentence and judge the sentiment **toward '{target}'**.\n\n"
    "Sentence: {sentence}\n\n"
    "Answer **positive**, **negative** or **neutral** only.\n"
    "Think step-by-step inside <THINKING_TAG> … </THINKING_TAG>, then put the final single-word "
    "answer inside <answer> … </answer>, e.g.  <answer>positive</answer>."
    )

    EN_MULTIPLE_CHOICE_USER_PROMPT_TEMPLATE =(
    "{question}\n\n{context}\n\n{choices}\n\n"
    "Show your reasoning in <THINKING_TAG> … </THINKING_TAG>.  Put **only** the choice number "
    "inside <answer> … </answer>, e.g.  <answer>2</answer>."
    )



    def __init__ (
    self ,
    llm_names :List [str ],
    *,
    prompt_language :str ="jp",
    seed :int =42 ,
    max_tokens :int =512 ,
    temperature :float =0.8 ,
    max_turns :int =1 ,
    servers :Optional [Dict [str ,str ]]=None ,
    ports :Optional [Dict [str ,int ]]=None ,
    track_costs :bool =True ,
    debug :bool =False ,
    together :bool =True ,
    valid_ratio :float =0.2 ,
    max_samples :int =-1 ,
    test_ratio :float =0.2 ,
    use_consultant :bool =True ,
    log_dir :Optional [str ]=None ,
    ):
        self .prompt_language =prompt_language .lower ()
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
        test_ratio =test_ratio ,
        use_consultant =use_consultant ,
        log_dir =log_dir ,
        )



    def _get_router_system_prompt (self )->str :
        if self .prompt_language =="jp":
            return self .JAPANESE_ROUTER_SYSTEM_PROMPT 
        return super ().DEFAULT_ROUTER_SYSTEM_PROMPT 

    def _get_system_prompt (self )->str :
        if self .prompt_language =="jp":
            return self .JAPANESE_SYSTEM_PROMPT 
        return super ().DEFAULT_SYSTEM_PROMPT 

    def _get_assistant_prompt (self )->str :
        if self .prompt_language =="jp":
            return self .JAPANESE_ASSISTANT_PROMPT .replace ("THINKING_TAG",self .thinking_tag )
        return f"Let me solve this step by step.\n<{self.thinking_tag}>"

    def _get_collaboration_prompt (self )->str :
        if self .prompt_language =="jp":
            return self .JAPANESE_COLLABORATION_PROMPT 
        return super ().DEFAULT_COLLABORATION_PROMPT 


    def _get_user_prompt_template (self )->str :
        return ""


    def _format_base_prompt (self ,task_data :Dict )->str :
        task_type =task_data .get ("jp_task_type","unknown")


        jp =self .prompt_language =="jp"
        if task_type =="chabsa":
            tpl =(
            self .JAPANESE_SENTIMENT_USER_PROMPT_TEMPLATE 
            if jp else self .EN_SENTIMENT_USER_PROMPT_TEMPLATE 
            )

            tpl =tpl .replace ("THINKING_TAG",self .thinking_tag )

            return tpl .format (sentence =task_data ["sentence"],target =task_data ["target"])

        if task_type in ["cma_basics","fp2","security_sales_1","cpa_audit"]:
            tpl =(
            self .JAPANESE_MULTIPLE_CHOICE_USER_PROMPT_TEMPLATE 
            if jp else self .EN_MULTIPLE_CHOICE_USER_PROMPT_TEMPLATE 
            )

            tpl =tpl .replace ("THINKING_TAG",self .thinking_tag )

            choices_txt =""
            for i ,c in enumerate (task_data ["choices"]):
                choice_text =c ["text"]if isinstance (c ,dict )else c 
                choice_id =c .get ("id",i )if isinstance (c ,dict )else i 
                choices_txt +=f"{choice_id}: {choice_text}\n"
            context =task_data .get ("context","")
            return tpl .format (
            question =task_data ["question"],
            context =context ,
            choices =choices_txt .strip (),
            )

        raise ValueError (f"Unknown task_type: {task_type}")

    def _load_data (
    self ,
    seed :int ,
    split :str ="train",
    validation :bool =False ,
    valid_ratio :float =None ,
    max_samples :int =None ,
    test_split :bool =False ,
    test_ratio :float =None 
    )->Dict [str ,Any ]:

        import os 


        worker_pid =os .getpid ()
        print (f"[PID {worker_pid}] Loading Japanese Financial data...")


        if valid_ratio is None :
            valid_ratio =self .valid_ratio 
        if max_samples is None :
            max_samples =self .max_samples 
        if test_ratio is None :
            test_ratio =self .test_ratio 


        combined_data =[]


        try :
            chabsa_data =self ._load_chabsa_data ()
            combined_data .extend (chabsa_data )
            print (f"[PID {worker_pid}] Loaded {len(chabsa_data)} chabsa examples")
        except Exception as e :
            print (f"[PID {worker_pid}] Failed to load chabsa data: {e}")


        for dataset_name in ["cma_basics","fp2","security_sales_1"]:
            try :
                mc_data =self ._load_multiple_choice_data (dataset_name )
                combined_data .extend (mc_data )
                print (f"[PID {worker_pid}] Loaded {len(mc_data)} {dataset_name} examples")
            except Exception as e :
                print (f"[PID {worker_pid}] Failed to load {dataset_name} data: {e}")


        try :
            cpa_data =self ._load_cpa_audit_data ()
            combined_data .extend (cpa_data )
            print (f"[PID {worker_pid}] Loaded {len(cpa_data)} cpa_audit examples")
        except Exception as e :
            print (f"[PID {worker_pid}] Failed to load cpa_audit data: {e}")

        print (f"[PID {worker_pid}] Total combined data before fallback: {len(combined_data)} examples")




        validated_data =[]
        for i ,item in enumerate (combined_data ):
            try :
                task_type =item .get ("jp_task_type","unknown")
                if task_type =="chabsa":

                    required_keys =["sentence","target","polarity"]
                    if all (key in item for key in required_keys ):
                        validated_data .append (item )
                    else :
                        print (f"[PID {worker_pid}] Skipping invalid chabsa item {i}: missing keys {[k for k in required_keys if k not in item]}")
                elif task_type in ["cma_basics","fp2","security_sales_1","cpa_audit"]:

                    required_keys =["question","choices","answer"]
                    if all (key in item for key in required_keys ):

                        if isinstance (item ["choices"],list )and len (item ["choices"])>0 :
                            validated_data .append (item )
                        else :
                            print (f"[PID {worker_pid}] Skipping invalid {task_type} item {i}: choices is not a valid list")
                    else :
                        print (f"[PID {worker_pid}] Skipping invalid {task_type} item {i}: missing keys {[k for k in required_keys if k not in item]}")
                else :
                    print (f"[PID {worker_pid}] Skipping item {i} with unknown task_type: {task_type}")
            except Exception as e :
                print (f"[PID {worker_pid}] Error validating item {i}: {e}")
                continue 

        combined_data =validated_data 
        print (f"[PID {worker_pid}] Validated data: {len(combined_data)} examples")


        if not combined_data :

            combined_data =self ._create_dummy_data ()
            print (f"[PID {worker_pid}] Using dummy data - no real datasets could be loaded")

        print (f"[PID {worker_pid}] Final combined data: {len(combined_data)} examples")


        if self .debug and combined_data :
            print (f"[PID {worker_pid}] DEBUG: First 3 examples:")
            for i ,example in enumerate (combined_data [:3 ]):
                print (f"[PID {worker_pid}] Example {i}: {example}")

                if example .get ("jp_task_type")=="chabsa":
                    sentence =example .get ("sentence","")
                    if len (sentence )>100 :
                        print (f"[PID {worker_pid}] Long sentence (showing first 200 chars): {sentence[:200]}...")
                    else :
                        print (f"[PID {worker_pid}] Full sentence: {sentence}")
                elif example .get ("jp_task_type")in ["cma_basics","fp2","security_sales_1","cpa_audit"]:
                    choices =example .get ("choices",[])
                    print (f"[PID {worker_pid}] Number of choices: {len(choices)}")
                    if choices :
                        print (f"[PID {worker_pid}] First choice: {choices[0]}")
            print (f"[PID {worker_pid}] DEBUG: Task type distribution:")
            task_types ={}
            for item in combined_data :
                task_type =item .get ("jp_task_type","unknown")
                task_types [task_type ]=task_types .get (task_type ,0 )+1 
            for task_type ,count in task_types .items ():
                print (f"[PID {worker_pid}] {task_type}: {count} examples")



        class SimpleDataset :
            def __init__ (self ,data ):
                self .data =data 

            def __len__ (self ):
                return len (self .data )

            def __getitem__ (self ,idx ):
                if isinstance (idx ,int ):
                    return self .data [idx ]
                elif isinstance (idx ,slice ):
                    return SimpleDataset (self .data [idx ])
                else :
                    raise TypeError (f"Invalid index type: {type(idx)}")

            def select (self ,indices ):
                selected_data =[self .data [i ]for i in indices ]
                return SimpleDataset (selected_data )

            def shuffle (self ,seed =None ):

                import random 
                shuffled_data =self .data .copy ()
                if seed is not None :
                    random .seed (seed )
                random .shuffle (shuffled_data )
                return SimpleDataset (shuffled_data )

        ds =SimpleDataset (combined_data )


        if self .debug and len (ds )>0 :
            print (f"[PID {worker_pid}] DEBUG: Validating data integrity after dataset creation...")
            sample_item =ds [0 ]
            print (f"[PID {worker_pid}] Sample item after dataset creation: {sample_item}")


            for i in range (min (3 ,len (ds ))):
                item =ds [i ]
                task_type =item .get ("jp_task_type","unknown")
                if task_type =="chabsa":
                    required_keys =["sentence","target","polarity"]
                elif task_type in ["cma_basics","fp2","security_sales_1","cpa_audit"]:
                    required_keys =["question","choices","answer"]
                else :
                    continue 

                missing_keys =[key for key in required_keys if key not in item ]
                if missing_keys :
                    print (f"[PID {worker_pid}] ERROR: Item {i} missing keys {missing_keys}. Item: {item}")
                else :
                    print (f"[PID {worker_pid}] Item {i} validation passed for task_type: {task_type}")


        from guf .utils import get_or_create_indices 


        unique_task_name =f"japanese_financial_{len(ds)}"

        split_indices =get_or_create_indices (
        task_name =unique_task_name ,
        dataset_len =len (ds ),
        seed =self .split_seed ,
        valid_ratio =valid_ratio ,
        test_ratio =test_ratio 
        )

        def _take (idxs :List [int ]):
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


        for k ,v in data_splits .items ():
            print (f"[PID {worker_pid}] Split {k} contains {len(v)} examples")

        return data_splits 

    def _load_chabsa_data (self )->List [Dict ]:

        data =[]


        base_path ="your/path/here"


        import os 
        worker_pid =os .getpid ()
        print (f"[PID {worker_pid}] Checking chabsa path: {base_path}")
        print (f"[PID {worker_pid}] Path exists: {os.path.exists(base_path)}")

        if os .path .exists (base_path ):
            json_files =glob .glob (os .path .join (base_path ,"e*_ann.json"))
            print (f"[PID {worker_pid}] Found {len(json_files)} JSON files")

            for json_file in json_files :
                try :
                    with open (json_file ,'r',encoding ='utf-8')as f :
                        file_data =json .load (f )

                    for sentence_data in file_data .get ("sentences",[]):
                        sentence =sentence_data ["sentence"]
                        for opinion in sentence_data .get ("opinions",[]):

                            if not sentence or not opinion .get ("target")or not opinion .get ("polarity"):
                                if self .debug :
                                    print (f"[PID {worker_pid}] Skipping invalid chabsa data: sentence='{sentence}', target='{opinion.get('target')}', polarity='{opinion.get('polarity')}'")
                                continue 

                            entry ={
                            "jp_task_type":"chabsa",
                            "sentence":str (sentence ).strip (),
                            "target":str (opinion ["target"]).strip (),
                            "polarity":str (opinion ["polarity"]).strip ()
                            }


                            if len (entry ["sentence"])<5 :
                                if self .debug :
                                    print (f"[PID {worker_pid}] Skipping very short sentence: '{entry['sentence']}'")
                                continue 


                            if entry ["polarity"]not in ["positive","negative","neutral"]:
                                if self .debug :
                                    print (f"[PID {worker_pid}] Skipping invalid polarity: '{entry['polarity']}'")
                                continue 

                            data .append (entry )

                except Exception as e :
                    print (f"[PID {worker_pid}] Error loading {json_file}: {e}")
                    continue 


        if not data :
            print (f"[PID {worker_pid}] No chabsa data loaded, using sample data")
            data =[
            {
            "jp_task_type":"chabsa",
            "sentence":"企業の業績が向上し、株価も上昇している。",
            "target":"企業の業績",
            "polarity":"positive"
            }
            ]

        print (f"[PID {worker_pid}] Total chabsa data: {len(data)} examples")


        if self .debug and data :
            for i ,entry in enumerate (data [:3 ]):
                print (f"[PID {worker_pid}] Chabsa entry {i}: {entry}")

        return data 

    def _load_multiple_choice_data (self ,dataset_name :str )->List [Dict ]:

        data =[]


        file_mapping ={
        "cma_basics":"cma_data.json",
        "fp2":"fp2_data.json",
        "security_sales_1":"security_data.json"
        }


        base_path =f"your/path/here{file_mapping[dataset_name]}"

        if os .path .exists (base_path ):
            try :
                with open (base_path ,'r',encoding ='utf-8')as f :
                    file_data =json .load (f )

                for item in file_data .get ("data",[]):
                    data .append ({
                    "jp_task_type":dataset_name ,
                    "question":item ["question"],
                    "context":item .get ("context",""),
                    "choices":item ["choices"],
                    "answer":item ["answer"]
                    })
            except Exception as e :
                if self .debug :
                    print (f"Error loading {base_path}: {e}")


        if not data :
            if dataset_name =="cma_basics":
                data =[
                {
                "jp_task_type":"cma_basics",
                "question":"日本経済に関する次の記述のうち、正しくないものはどれか。",
                "context":"",
                "choices":[
                {"id":0 ,"text":"実質GDP（実質国内総生産）とは、物価変動の影響を除いた生産活動の水準のことである。"},
                {"id":1 ,"text":"インフレーションは一般物価水準の持続的な上昇を意味する。"},
                {"id":2 ,"text":"銀行等の金融仲介機関が中心になって資金の融通を媒介する形態を間接金融という。"},
                {"id":3 ,"text":"日本銀行による財政政策はマネーサプライの増減を通じて物価水準を調整する政策である。"}
                ],
                "answer":3 
                }
                ]
            elif dataset_name =="fp2":
                data =[
                {
                "jp_task_type":"fp2",
                "question":"ファイナンシャル・プランナー（以下「ＦＰ」という）の顧客に対する行為に関する次の記述のうち、関連法規に照らし、最も不適切なものはどれか。",
                "context":"１．弁護士の資格を有しないＦＰのＡさんは、顧客から成年後見制度について相談を受け、法定後見制度と任意後見制度の違いについて一般的な説明をした。",
                "choices":[
                {"id":0 ,"text":"１"},
                {"id":1 ,"text":"２"},
                {"id":2 ,"text":"３"},
                {"id":3 ,"text":"４"}
                ],
                "answer":3 
                }
                ]
            else :
                data =[
                {
                "jp_task_type":"security_sales_1",
                "question":"次の文章について、正しければ○と、正しくなければ×と答えよ。",
                "context":"外務員は、その所属する金融商品取引業者等に代わって、有価証券の売買等法律に規定する行為に関し、一切の裁判上の行為を行う権限を有するものとみなされる。",
                "choices":[
                {"id":0 ,"text":"○"},
                {"id":1 ,"text":"×"}
                ],
                "answer":1 
                }
                ]

        return data 

    def _load_cpa_audit_data (self )->List [Dict ]:

        data =[]


        excel_path ="your/path/here.xlsx"

        if os .path .exists (excel_path ):
            try :
                df =pd .read_excel (excel_path ,index_col =0 )

                worker_pid =os .getpid ()
                print (f"[PID {worker_pid}] CPA DataFrame shape: {df.shape}")
                print (f"[PID {worker_pid}] CPA DataFrame columns: {list(df.columns)}")


                if self .debug :
                    print (f"[PID {worker_pid}] First few rows of CPA data:")
                    print (df .head ())


                df ["question"]=df ["question"].fillna ("").astype (str )
                df ["ア"]=df ["ア"].fillna ("").astype (str )
                df ["イ"]=df ["イ"].fillna ("").astype (str )
                df ["ウ"]=df ["ウ"].fillna ("").astype (str )
                df ["エ"]=df ["エ"].fillna ("").astype (str )
                df ["オ"]=df ["オ"].fillna ("").astype (str )
                df ["カ"]=df ["カ"].fillna ("").astype (str )

                for idx ,row in df .iterrows ():

                    if not row ["question"]or row ["question"]in ["","nan","None"]:
                        continue 
                    if row .get ("abnormal_flg",0 )==1 :
                        continue 

                    question =row ["question"]


                    contexts =[]
                    for option_key in ["ア","イ","ウ","エ","オ","カ"]:
                        if option_key in row and row [option_key ]and row [option_key ]not in ["","nan","None"]:
                            contexts .append (f"{option_key}: {row[option_key]}")

                    context ="\n".join (contexts )


                    choices =[]
                    choice_columns =[]


                    for col in df .columns :
                        if str (col ).isdigit ()or col in ['1','2','3','4','5','6']:
                            choice_columns .append (col )

                    if not choice_columns and self .debug :
                        print (f"[PID {worker_pid}] No numeric choice columns found. All columns: {list(df.columns)}")


                    for i ,col in enumerate (choice_columns ):
                        if col in row and row [col ]and str (row [col ])not in ["","nan","None"]:
                            choices .append ({"id":i ,"text":str (row [col ])})


                    if not choices :
                        if self .debug :
                            print (f"[PID {worker_pid}] Skipping CPA row {idx}: no valid choices from columns {choice_columns}")
                        continue 


                    try :
                        answer =int (row ["a_no"])-1 
                        if answer <0 or answer >=len (choices ):
                            if self .debug :
                                print (f"[PID {worker_pid}] Skipping CPA row {idx}: invalid answer {answer} for {len(choices)} choices")
                            continue 
                    except (ValueError ,TypeError ):
                        if self .debug :
                            print (f"[PID {worker_pid}] Skipping CPA row {idx}: invalid answer format {row.get('a_no')}")
                        continue 


                    entry ={
                    "jp_task_type":"cpa_audit",
                    "question":question ,
                    "context":context ,
                    "choices":choices ,
                    "answer":answer 
                    }


                    if self .debug and len (data )<3 :
                        print (f"[PID {worker_pid}] CPA entry {len(data)}: {entry}")

                    data .append (entry )

                print (f"[PID {worker_pid}] Successfully loaded {len(data)} CPA audit entries")

            except Exception as e :
                worker_pid =os .getpid ()
                print (f"[PID {worker_pid}] Error loading CPA audit data: {e}")
                import traceback 
                traceback .print_exc ()


        if not data :
            data =[
            {
            "jp_task_type":"cpa_audit",
            "question":"公認会計士監査に関する次の記述のうち，正しいものの組合せとして最も適切な番号を 一つ選びなさい。",
            "context":"ア: 株式会社において，経営者は株主が拠出した資本を適切に管理・運用する受託責任を負い，この結果についてステークホルダーに対する説明責任を果たす必要がある。\nイ: 公認会計士監査は，財務情報の信頼性を担保する役割があるが，その過程で発見した内部統制の不備や課題については，被監査会社にフィードバックすることにより，被監査会社のコーポレート・ガバナンスの向上に資することもある。",
            "choices":[
            {"id":0 ,"text":"アイ"},
            {"id":1 ,"text":"アウ"},
            {"id":2 ,"text":"アエ"},
            {"id":3 ,"text":"イウ"},
            {"id":4 ,"text":"イエ"},
            {"id":5 ,"text":"ウエ"}
            ],
            "answer":0 
            }
            ]

        return data 

    def _create_dummy_data (self )->List [Dict ]:

        return [
        {
        "jp_task_type":"chabsa",
        "sentence":"企業の業績が向上し、株価も上昇している。",
        "target":"企業の業績",
        "polarity":"positive"
        },
        {
        "jp_task_type":"cma_basics",
        "question":"日本の金融政策に関する質問です。",
        "context":"",
        "choices":[
        {"id":0 ,"text":"選択肢A"},
        {"id":1 ,"text":"選択肢B"}
        ],
        "answer":0 
        },
        {
        "jp_task_type":"fp2",
        "question":"ファイナンシャルプランナーの業務に関する質問です。",
        "context":"",
        "choices":[
        {"id":0 ,"text":"正しい"},
        {"id":1 ,"text":"間違い"}
        ],
        "answer":1 
        },
        {
        "jp_task_type":"security_sales_1",
        "question":"証券外務員に関する問題です。",
        "context":"",
        "choices":[
        {"id":0 ,"text":"○"},
        {"id":1 ,"text":"×"}
        ],
        "answer":0 
        },
        {
        "jp_task_type":"cpa_audit",
        "question":"監査に関する問題です。",
        "context":"ア: 選択肢ア\nイ: 選択肢イ",
        "choices":[
        {"id":0 ,"text":"アイ"},
        {"id":1 ,"text":"アウ"}
        ],
        "answer":0 
        }
        ]

    def _calculate_reward (
    self ,
    completions :List [str ],
    task_data :List [Dict ],
    debug :bool =False 
    )->List [float ]:

        rewards =[]

        for completion ,data in zip (completions ,task_data ):
            task_type =data .get ("jp_task_type","unknown")

            if task_type =="chabsa":
                reward =self ._calculate_sentiment_reward (completion ,data ,debug )
            elif task_type in ["cma_basics","fp2","security_sales_1","cpa_audit"]:
                reward =self ._calculate_multiple_choice_reward (completion ,data ,debug )
            else :
                if debug :
                    print (f"=== DEBUG: Unknown task type: {task_type} ===")
                reward =0.0 

            rewards .append (reward )

        return rewards 

    def _calculate_sentiment_reward (self ,completion :str ,data :Dict ,debug :bool )->float :

        pred =extract_answer (completion )
        if pred is None :
            if debug :
                print ("=== DEBUG: No <answer> tag found for sentiment task ===")
            return 0.0 

        pred =pred .strip ().lower ()
        target =data ["polarity"].lower ()

        is_correct =pred ==target 

        if debug :
            print ("=== DEBUG: Sentiment Analysis Evaluation ===")
            print (f"Sentence: {data['sentence']}")
            print (f"Target: {data['target']}")
            print (f"Expected sentiment: {target}")
            print (f"Predicted sentiment: {pred}")
            print (f"Correct: {is_correct}")
            print ("=============================================")

        return 1.0 if is_correct else 0.0 

    def _calculate_multiple_choice_reward (self ,completion :str ,data :Dict ,debug :bool )->float :

        pred =extract_answer (completion )
        if pred is None :
            if debug :
                print ("=== DEBUG: No <answer> tag found for multiple choice task ===")
            return 0.0 

        pred =pred .strip ()


        try :
            pred_id =int (pred )
        except ValueError :

            numbers =re .findall (r'\d+',pred )
            if numbers :
                pred_id =int (numbers [0 ])
            else :
                if debug :
                    print (f"=== DEBUG: Could not parse answer: {pred} ===")
                return 0.0 

        correct_id =int (data ["answer"])
        is_correct =pred_id ==correct_id 

        if debug :
            print ("=== DEBUG: Multiple Choice Evaluation ===")
            print (f"Task type: {data.get('jp_task_type', 'unknown')}")
            print (f"Question: {data['question'][:100]}...")
            print (f"Expected answer: {correct_id}")
            print (f"Predicted answer: {pred_id}")
            print (f"Correct: {is_correct}")
            print ("==========================================")

        return 1.0 if is_correct else 0.0 


if __name__ =="__main__":

    llms =["gpt-4o-mini"]
    task =JapaneseFinancialTask (llm_names =llms ,debug =True ,max_samples =10 )
    splits =task ._load_data (seed =0 ,split ="train",validation =True ,valid_ratio =0.1 ,max_samples =10 )

    for split in ("train","valid"):
        if len (splits [split ])>0 :
            sample =splits [split ][0 ]
            print ("Sample prompt:")
            print (task ._format_base_prompt (sample ))
            print ()


            if sample .get ("jp_task_type")=="chabsa":
                demo ="<think>This seems positive</think><answer>positive</answer>"
            else :
                demo ="<think>Looking at the options...</think><answer>1</answer>"

            print ("Reward:",task ._calculate_reward ([demo ],[sample ],debug =True ))