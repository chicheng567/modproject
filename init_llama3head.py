import torch
from independent_llama32 import LlamaHead
from transformers import AutoTokenizer
import json
from transformers.models.llama.configuration_llama import LlamaConfig

def get_llama3_head(config: LlamaConfig):
    # 載入 llama3.2 的權重檔案
    state_dict = torch.load("llama3_model_weights_bf16.pth", map_location="cpu")
    model_head = LlamaHead(config=config)
    head_state_dict = {}

    # 載入 embedding 權重
    if "model.embed_tokens.weight" in state_dict:
        head_state_dict["embed_tokens.weight"] = state_dict["model.embed_tokens.weight"]
        print("✓ Loaded embed_tokens.weight")
    elif "embed_tokens.weight" in state_dict:
        head_state_dict["embed_tokens.weight"] = state_dict["embed_tokens.weight"]
        print("✓ Loaded embed_tokens.weight (alternative path)")

    # 定義第一層的前綴
    layer_0_prefix = "model.layers.0."
    alt_layer_0_prefix = "layers.0."

    # 載入第一層的權重
    for key in state_dict.keys():
        if key.startswith(layer_0_prefix):
            new_key = "first_layer." + key[len(layer_0_prefix):]
            head_state_dict[new_key] = state_dict[key]
            print(f"✓ Loaded {new_key}")
        elif key.startswith(alt_layer_0_prefix):
            new_key = "first_layer." + key[len(alt_layer_0_prefix):]
            head_state_dict[new_key] = state_dict[key]
            print(f"✓ Loaded {new_key}")

    # 載入 norm 權重
    if "model.norm.weight" in state_dict:
        head_state_dict["norm.weight"] = state_dict["model.norm.weight"]
        print("✓ Loaded norm.weight")
    elif "norm.weight" in state_dict:
        head_state_dict["norm.weight"] = state_dict["norm.weight"]
        print("✓ Loaded norm.weight (alternative path)")

    print(f"\n創建了包含 {len(head_state_dict)} 個參數的 head_state_dict")

    print("\nLlamaHead 預期的參數：")
    for name, param in model_head.named_parameters():
        print(f"  {name}: {param.shape}")

    print("\nhead_state_dict 中的參數：")
    for name, tensor in head_state_dict.items():
        print(f"  {name}: {tensor.shape}")

    model_params = set(name for name, _ in model_head.named_parameters())
    loaded_params = set(head_state_dict.keys())

    missing_params = model_params - loaded_params
    extra_params = loaded_params - model_params

    if missing_params:
        print(f"\n⚠️  缺少的參數: {missing_params}")
    if extra_params:
        print(f"\n⚠️  額外的參數: {extra_params}")

    try:
        missing_keys, unexpected_keys = model_head.load_state_dict(head_state_dict, strict=False)
        
        if missing_keys:
            print(f"\n📝 缺少的鍵 (將使用預設初始化): {missing_keys}")
        if unexpected_keys:
            print(f"\n📝 意外的鍵: {unexpected_keys}")
            
        print("\n✅ 成功載入 state_dict 到 LlamaHead！")
        
    except Exception as e:
        print(f"\n❌ 載入 state_dict 時出錯: {e}")
        
        print("\n嘗試手動參數匹配...")
        for name, param in model_head.named_parameters():
            if name in head_state_dict:
                if param.shape == head_state_dict[name].shape:
                    param.data.copy_(head_state_dict[name])
                    print(f"✓ 手動載入 {name}")
                else:
                    print(f"❌ {name} 的形狀不匹配: 預期 {param.shape}, 實際 {head_state_dict[name].shape}")
            else:
                print(f"⚠️  在 state_dict 中找不到參數 {name}")

    # 將模型移至 GPU 並設定為 bfloat16
    model_head.to("cuda:0", dtype=torch.bfloat16)

    print("\n檢查參數資料類型：")
    all_bfloat16 = True
    for name, param in model_head.named_parameters():
        if param.dtype != torch.bfloat16:
            print(f"❌ 參數 {name} 的資料類型為 {param.dtype} 而非 bfloat16")
            all_bfloat16 = False

    if all_bfloat16:
        print("✅ 所有參數都是 bfloat16 格式")
    return model_head

def main():
    # 載入 llama3.2 配置
    with open("llama3_config.json", "r") as f:
        config_dict = json.load(f)
    
    config = LlamaConfig(**config_dict)
    
    print("開始載入 Llama3.2 Head...")
    model_head = get_llama3_head(config)
    
    print("\n🚀 Llama3.2 Head 載入完成！")
    return model_head

if __name__ == "__main__":
    main()
