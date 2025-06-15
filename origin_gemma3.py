from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
import torch
import json
with open("llama3_config.json", "r") as f:
    config = json.load(f)
config = LlamaConfig(**config)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = LlamaForCausalLM(config)
# 將模型轉換為 bfloat16 格式
model = model.to("cuda:0", dtype=torch.bfloat16)
model.load_state_dict(torch.load("llama3_model_weights_bf16.pth"))
messages = [
    {"role": "系統", "content": "你是一個幫助人們的AI助手，永遠講話禮貌且溫暖。"},
    {"role": "使用者", "content": "我壓力真的好大，怎麼辦。"},
]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, num_beams=1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
