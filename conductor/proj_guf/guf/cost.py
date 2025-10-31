

import tiktoken 
import os 
import transformers 
from typing import Dict ,List ,Optional ,Union 


class Calculator :



    GPT_input_pricing ={
    "gpt-4o-mini":0.15 ,
    "gpt-4o":2.5 ,
    "gpt-4.1-2025-04-14":2 ,
    "gpt-4.1":2 ,
    }


    GPT_output_pricing ={
    "gpt-4o-mini":0.60 ,
    "gpt-4o":10.00 ,
    "gpt-4.1-2025-04-14":8 ,
    "gpt-4.1":8 ,
    }


    DeepSeek_input_pricing ={
    "deepseek-ai/DeepSeek-V3":1.25 ,
    }

    DeepSeek_output_pricing ={
    "deepseek-ai/DeepSeek-V3":1.25 ,
    }


    Anthropic_input_pricing ={
    "claude-3-7-sonnet-20250219":3.00 ,
    "claude-sonnet-4-20250514":3.00 ,
    }

    Anthropic_output_pricing ={
    "claude-3-7-sonnet-20250219":15.00 ,
    "claude-sonnet-4-20250514":15.00 ,
    }


    Gemini_input_pricing ={
    "gemini-1.5-pro":1.25 ,
    "gemini-2.5-pro":1.25 ,
    }

    Gemini_output_pricing ={
    "gemini-1.5-pro":5.0 ,
    "gemini-2.5-pro":10.0 ,
    }





    OpenSource_input_pricing ={
    "1b":0.04 ,
    "3b":0.06 ,
    "7b":0.10 ,
    "13b":0.15 ,
    "24b":0.20 ,
    "30b":0.25 ,
    "70b":0.40 ,
    "100b":0.60 ,
    }
    OpenSource_output_pricing ={
    "1b":0.06 ,
    "3b":0.10 ,
    "7b":0.15 ,
    "13b":0.22 ,
    "24b":0.30 ,
    "30b":0.40 ,
    "70b":0.65 ,
    "100b":0.90 ,
    }

    def __init__ (self ,model ,formatted_input_sequence =None ,output_sequence_string =None ):
        self .model =model 
        self .formatted_input_sequence =formatted_input_sequence 
        self .output_sequence_string =output_sequence_string 
        self .input_token_length =0 
        self .output_token_length =0 

    def calculate_input_token_length_GPT (self ):


        encoding =tiktoken .get_encoding ("cl100k_base")

        tokens_per_message =3 
        tokens_per_name =1 

        num_tokens =0 
        for message in self .formatted_input_sequence :
            num_tokens +=tokens_per_message 
            for key ,value in message .items ():
                num_tokens +=len (encoding .encode (value ))
                if key =="name":
                    num_tokens +=tokens_per_name 
        num_tokens +=3 
        return num_tokens 

    def calculate_output_token_length_GPT (self ):


        tokenizer =tiktoken .get_encoding ("cl100k_base")


        tokens =tokenizer .encode (self .output_sequence_string )


        return len (tokens )

    def calculate_cost_GPT (self ):
        if self .formatted_input_sequence is not None :
            self .input_token_length =self .calculate_input_token_length_GPT ()
            input_cost =self .input_token_length *self .GPT_input_pricing .get (self .model ,0 )/1e6 
        else :
            input_cost =0 
        if self .output_sequence_string is not None :
            self .output_token_length =self .calculate_output_token_length_GPT ()
            output_cost =self .output_token_length *self .GPT_output_pricing .get (self .model ,0 )/1e6 
        else :
            output_cost =0 
        return input_cost +output_cost 

    def formatted_to_string_OpenAI (self ,formatted_messages ):

        result =""
        for message in formatted_messages :
            role =message .get ("role","")
            content =message .get ("content","")
            if role =="system":
                result +=f"System: {content}\n\n"
            elif role =="user":
                result +=f"User: {content}\n\n"
            elif role =="assistant":
                result +=f"Assistant: {content}\n\n"
            else :
                result +=f"{role}: {content}\n\n"
        return result .strip ()

    def calculate_token_length_DeepSeek (self ):

        try :

            module_dir =os .path .dirname (os .path .abspath (__file__ ))
            tokenizer_path =os .path .join (module_dir ,"tokenizers/deepseek")


            if os .path .exists (tokenizer_path ):
                tokenizer =transformers .AutoTokenizer .from_pretrained (
                tokenizer_path ,trust_remote_code =True 
                )
            else :

                tokenizer =transformers .AutoTokenizer .from_pretrained (
                "deepseek-ai/deepseek-coder-1.3b-base",trust_remote_code =True 
                )
                print (f"Warning: Local DeepSeek tokenizer not found, loaded from HuggingFace hub.")
        except Exception as e :
            print (f"Warning: Could not load DeepSeek tokenizer: {e}")
            print ("Falling back to cl100k_base encoding for token calculation.")


            encoding =tiktoken .get_encoding ("cl100k_base")

            if self .formatted_input_sequence is not None :
                input_text =self .formatted_to_string_OpenAI (self .formatted_input_sequence )
                self .input_token_length =len (encoding .encode (input_text ))

            if self .output_sequence_string is not None :
                self .output_token_length =len (encoding .encode (self .output_sequence_string ))

            return 


        if self .formatted_input_sequence is not None :
            input_text =self .formatted_to_string_OpenAI (self .formatted_input_sequence )
            input_tokenized =tokenizer .encode (input_text )
            self .input_token_length =len (input_tokenized )


        if self .output_sequence_string is not None :
            output_tokenized =tokenizer .encode (self .output_sequence_string )
            self .output_token_length =len (output_tokenized )

    def calculate_cost_DeepSeek (self ):

        self .calculate_token_length_DeepSeek ()
        input_cost =self .input_token_length *self .DeepSeek_input_pricing .get (self .model ,0 )/1e6 
        output_cost =self .output_token_length *self .DeepSeek_output_pricing .get (self .model ,0 )/1e6 
        return input_cost +output_cost 

    def calculate_token_length_generic (self ):

        encoding =tiktoken .get_encoding ("cl100k_base")

        if self .formatted_input_sequence is not None :
            input_text =""
            for message in self .formatted_input_sequence :
                if "content"in message :
                    input_text +=message ["content"]+"\n"

            input_tokenized =encoding .encode (input_text )
            self .input_token_length =len (input_tokenized )

        if self .output_sequence_string is not None :
            output_tokenized =encoding .encode (self .output_sequence_string )
            self .output_token_length =len (output_tokenized )

    def calculate_cost_Anthropic (self ):
        self .calculate_token_length_generic ()
        input_cost =self .input_token_length *self .Anthropic_input_pricing .get (self .model ,0 )/1e6 
        output_cost =self .output_token_length *self .Anthropic_output_pricing .get (self .model ,0 )/1e6 
        return input_cost +output_cost 

    def calculate_cost_Gemini (self ):
        self .calculate_token_length_generic ()
        input_cost =self .input_token_length *self .Gemini_input_pricing .get (self .model ,0 )/1e6 
        output_cost =self .output_token_length *self .Gemini_output_pricing .get (self .model ,0 )/1e6 
        return input_cost +output_cost 

    def detect_model_size (self ,model_name ):

        model_lower =model_name .lower ()
        if "phi-4"in model_lower or "microsoft/phi-4"in model_lower :
            return "13b"

        if "1b"in model_lower or "1.3b"in model_lower or "1.5b"in model_lower :
            return "1b"
        elif "3b"in model_lower or "2.7b"in model_lower :
            return "3b"
        elif "7b"in model_lower or "6.7b"in model_lower or "8b"in model_lower :
            return "7b"
        elif "13b"in model_lower or "12b"in model_lower or "14b"in model_lower :
            return "13b"
        elif "24b"in model_lower or "20b"in model_lower or "22b"in model_lower :
            return "24b"
        elif "30b"in model_lower or "32b"in model_lower or "34b"in model_lower or "27b"in model_lower :
            return "30b"
        elif "70b"in model_lower or "65b"in model_lower or "72b"in model_lower :
            return "70b"
        elif "100b"in model_lower or "150b"in model_lower or "175b"in model_lower :
            return "100b"


        return "7b"

    def calculate_cost_OpenSource (self ):

        self .calculate_token_length_generic ()


        size_category =self .detect_model_size (self .model )

        input_cost =self .input_token_length *self .OpenSource_input_pricing .get (size_category ,0.08 )/1e6 
        output_cost =self .output_token_length *self .OpenSource_output_pricing .get (size_category ,0.12 )/1e6 
        return input_cost +output_cost 

    def calculate_cost (self ):

        if "gpt"in self .model .lower ():
            return self .calculate_cost_GPT ()
        elif "deepseek"in self .model .lower ():
            return self .calculate_cost_DeepSeek ()
        elif "claude"in self .model .lower ():
            return self .calculate_cost_Anthropic ()
        elif "gemini"in self .model .lower ():
            return self .calculate_cost_Gemini ()
        else :


            return 0.0 



    def calculate_input_token_length (self ,input_sequence =None ,form ="formatted"):

        if input_sequence is not None :
            if form =="list":

                self .formatted_input_sequence =[]
                for i ,content in enumerate (input_sequence ):
                    if i ==0 :
                        self .formatted_input_sequence .append ({"role":"system","content":content })
                    elif i %2 ==1 :
                        self .formatted_input_sequence .append ({"role":"user","content":content })
                    else :
                        self .formatted_input_sequence .append ({"role":"assistant","content":content })
            else :
                self .formatted_input_sequence =input_sequence 

        if "gpt"in self .model .lower ():
            self .input_token_length =self .calculate_input_token_length_GPT ()
        elif "deepseek"in self .model .lower ():
            self .calculate_token_length_DeepSeek ()
        else :
            self .calculate_token_length_generic ()

        return self .input_token_length 

    def calculate_total_cost (self ,input_sequence ,output_sequence ,form ="formatted"):

        self .calculate_input_token_length (input_sequence ,form )
        self .output_sequence_string =output_sequence 
        return self .calculate_cost ()



model_calculators ={}

def get_calculator (model ):

    if model not in model_calculators :
        model_calculators [model ]={
        "calculator":Calculator (model ),
        "total_cost":0.0 ,
        "total_input_tokens":0 ,
        "total_output_tokens":0 ,
        "queries":0 
        }
    return model_calculators [model ]

def track_cost (model ,messages ,response ):

    calc_data =get_calculator (model )
    calculator =calc_data ["calculator"]


    cost =calculator .calculate_total_cost (messages ,response )


    calc_data ["total_cost"]+=cost 
    calc_data ["total_input_tokens"]+=calculator .input_token_length 
    calc_data ["total_output_tokens"]+=calculator .output_token_length 
    calc_data ["queries"]+=1 

    return {
    "model":model ,
    "input_tokens":calculator .input_token_length ,
    "output_tokens":calculator .output_token_length ,
    "cost":cost ,
    "total_cost":calc_data ["total_cost"]
    }

def get_cost_summary ():

    summary =[]
    total_cost =0.0 

    for model ,data in model_calculators .items ():
        model_summary ={
        "model":model ,
        "queries":data ["queries"],
        "total_input_tokens":data ["total_input_tokens"],
        "total_output_tokens":data ["total_output_tokens"],
        "total_tokens":data ["total_input_tokens"]+data ["total_output_tokens"],
        "total_cost":data ["total_cost"]
        }
        summary .append (model_summary )
        total_cost +=data ["total_cost"]

    return {
    "models":summary ,
    "total_cost":total_cost 
    }

def reset_costs ():

    for model in model_calculators :
        model_calculators [model ]={
        "calculator":Calculator (model ),
        "total_cost":0.0 ,
        "total_input_tokens":0 ,
        "total_output_tokens":0 ,
        "queries":0 
        }


if __name__ =="__main__":

    model ="gpt-4o-mini"
    messages =[
    {"role":"system","content":"You are a helpful assistant."},
    {"role":"user","content":"Hello, how are you?"}
    ]
    response ="I'm doing well, thank you for asking!"

    cost_info =track_cost (model ,messages ,response )
    print (f"Query cost: ${cost_info['cost']:.6f}")

    summary =get_cost_summary ()
    print (f"Total cost: ${summary['total_cost']:.6f}")


    deepseek_model ="deepseek-chat"
    deepseek_calculator =Calculator (deepseek_model ,messages ,response )
    deepseek_cost =deepseek_calculator .calculate_cost ()
    print (f"\nDeepSeek test:")
    print (f"Input tokens: {deepseek_calculator.input_token_length}")
    print (f"Output tokens: {deepseek_calculator.output_token_length}")
    print (f"Total cost: ${deepseek_cost:.6f}")