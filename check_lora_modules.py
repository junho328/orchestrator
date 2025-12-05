from transformers import AutoModelForCausalLM
# from peft.utils import find_all_linear_names
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
for name, module in m.named_modules():
    if any(kw in name.lower() for kw in ["q", "k", "v", "proj", "attn", "query", "key", "value", "o", "output", "qkv", "query_key_value"]):
        print(name)
