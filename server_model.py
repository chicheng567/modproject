# Load model directly
from transformers import AutoTokenizer, Gemma3ForCausalLM, Gemma3TextConfig
import json
import torch
from transformers import modeling_utils
from independent_gemma3_12b import Gemma3TextModelHeadless
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']
processor = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
config_path = "/modproject/4bconfig.json"
with open(config_path, "r") as f:
    config = json.load(f)
    config["torch_dtype"] = "bfloat16"

config = Gemma3TextConfig(**config)
model = Gemma3ForCausalLM(config)
model.model = Gemma3TextModelHeadless(config)
# 移動到 GPU 並指定 dtype
model = model.to(device="cuda:0", dtype=torch.bfloat16)

# 加載權重，並確保轉換為 bfloat16
state_dict = torch.load("/modproject/gemma3_4b_it_bfloat16_correct.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
prompt = "I have a really bad day..."
inputs = processor(prompt, return_tensors="pt")
inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
generate_ids = model.generate(**inputs, max_new_tokens=100)
decoded = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(decoded)