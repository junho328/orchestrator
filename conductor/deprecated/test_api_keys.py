from llm_clients import (
query_oai ,
query_anthropic ,
query_gemini ,
query_deepseek 
)
from concurrent .futures import ThreadPoolExecutor ,as_completed 

if __name__ =="__main__":
    question ="You are given a 0-indexed integer array coins, representing the values of the coins available, and an integer target. An integer x is obtainable if there exists a subsequence of coins that sums to x. Return the minimum number of coins of any value that need to be added to the array so that every integer in the range [1, target] is obtainable. A subsequence of an array is a new non-empty array that is formed from the original array by deleting some (possibly none) of the elements without disturbing the relative positions of the remaining elements."

    messages =[{"role":"user","content":question }]























    res =query_anthropic (
    model ="claude-sonnet-4-20250514",
    messages =messages ,
    max_tokens =64000 ,
    temperature =0.7 ,
    platform ="bedrock",
    claude_thinking_budget =32000 
    )
    print ("Anthropic response:",res )

    breakpoint ()










