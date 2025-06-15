# Load model directly
from transformers import AutoTokenizer, LlamaConfig
import json
import torch
from transformers import modeling_utils
from independent_llama32 import LlamaTextModelHeadless, LlamaForCausalLMHeadless, LlamaHead
from init_llama3head import get_llama3_head
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']
processor = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
config_path = "/modproject/llama3_config.json"
with open(config_path, "r") as f:
    config = json.load(f)
    config["torch_dtype"] = "bfloat16"

config = LlamaConfig(**config)
model = LlamaForCausalLMHeadless(config)
# 移動到 GPU 並指定 dtype
model = model.to(device="cuda:0", dtype=torch.bfloat16)
model_head = get_llama3_head(config)
model_head = model_head.to(device="cuda:0", dtype=torch.bfloat16)

# 加載權重，並確保轉換為 bfloat16
state_dict = torch.load("/modproject/llama3_model_weights_bf16.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
model_head.eval()
messages = [
    {"role": "系統", "content": "你是一個幫助人們的AI助手，永遠講話禮貌且溫暖。"},
    {"role": "使用者", "content": "我壓力真的好大，怎麼辦。"},
]
inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
all_output_ids = []
client_cache = None
server_cache = None
in_ids = inputs["input_ids"]
position_ids = torch.arange(in_ids.shape[1], device=in_ids.device).unsqueeze(0)
for i in range(400):
    attention_mask = torch.ones(1, client_cache.get_seq_length() + 1 if server_cache is not None else in_ids.shape[1])
    output = model_head(
        input_ids=in_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=client_cache,
        use_cache=True,
        cache_position=position_ids.squeeze(0),
    )
    client_cache = output.past_key_values
    hidden_states = output.last_hidden_state
    output_ids, server_cache = model(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        logits_to_keep=1, 
        past_key_values=server_cache, 
        position_ids=position_ids, 
        cache_position=position_ids.squeeze(0), 
        use_cache=True,
        output_attentions=False,
        skip_layers=1
    )
    assert server_cache.get_seq_length(layer_idx=1) == client_cache.get_seq_length(), f"sql error, server_cache.get_seq_length() == {server_cache.get_seq_length(layer_idx=1)}, client_cache.get_seq_length() == {client_cache.get_seq_length()}"
    all_output_ids.append(output_ids)
    in_ids = output_ids.unsqueeze(0)
    position_ids = torch.tensor([[client_cache.get_seq_length()]], device=in_ids.device)
    

last_output_ids = torch.cat(all_output_ids, dim=0)
print(processor.decode(last_output_ids, skip_special_tokens=True))