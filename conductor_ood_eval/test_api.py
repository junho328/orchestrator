import asyncio 
from utils .llm_clients import (
query_oai ,
query_anthropic ,
query_gemini ,
query_locally_hosted_model ,
query_deepseek 
)

from query_oai_responses import query_oai_responses 


async def main ():
    messages =[{"role":"user","content":"Write python code that recusively calculates the nth number in the Fibonacci sequence."}]

    res =await query_oai_responses (
    model_name ="gpt-5",
    messages =messages ,
    max_tokens =4000 ,
    temperature =0.7 ,
    reasoning_effort ="medium",
    verbosity ="medium",
    )
    print ("OpenAI response:",res )

    breakpoint ()







































if __name__ =="__main__":
    asyncio .run (main ())