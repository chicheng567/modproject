import torch
from independent_gemma3_12b import Gemma3Head
from transformers import AutoTokenizer
import json
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

def get_gemma3_head(config: Gemma3TextConfig):
    state_dict = torch.load("gemma3_4b_it_bfloat16_correct.pth", map_location="cpu")
    model_head = Gemma3Head(config=config)
    head_state_dict = {}

    if "model.embed_tokens.weight" in state_dict:
        head_state_dict["embed_tokens.weight"] = state_dict["model.embed_tokens.weight"]
        print("✓ Loaded embed_tokens.weight")
    elif "embed_tokens.weight" in state_dict:
        head_state_dict["embed_tokens.weight"] = state_dict["embed_tokens.weight"]
        print("✓ Loaded embed_tokens.weight (alternative path)")
    layer_0_prefix = "model.layers.0."
    alt_layer_0_prefix = "layers.0."

    for key in state_dict.keys():
        if key.startswith(layer_0_prefix):
            new_key = "first_layer." + key[len(layer_0_prefix):]
            head_state_dict[new_key] = state_dict[key]
            print(f"✓ Loaded {new_key}")
        elif key.startswith(alt_layer_0_prefix):
            new_key = "first_layer." + key[len(alt_layer_0_prefix):]
            head_state_dict[new_key] = state_dict[key]
            print(f"✓ Loaded {new_key}")

    if "model.norm.weight" in state_dict:
        head_state_dict["norm.weight"] = state_dict["model.norm.weight"]
        print("✓ Loaded norm.weight")
    elif "norm.weight" in state_dict:
        head_state_dict["norm.weight"] = state_dict["norm.weight"]
        print("✓ Loaded norm.weight (alternative path)")

    print(f"\nCreated head_state_dict with {len(head_state_dict)} parameters")

    print("\nGemma3Head expected parameters:")
    for name, param in model_head.named_parameters():
        print(f"  {name}: {param.shape}")

    print("\nParameters in head_state_dict:")
    for name, tensor in head_state_dict.items():
        print(f"  {name}: {tensor.shape}")

    model_params = set(name for name, _ in model_head.named_parameters())
    loaded_params = set(head_state_dict.keys())

    missing_params = model_params - loaded_params
    extra_params = loaded_params - model_params

    if missing_params:
        print(f"\n⚠️  Missing parameters: {missing_params}")
    if extra_params:
        print(f"\n⚠️  Extra parameters: {extra_params}")

    try:
        missing_keys, unexpected_keys = model_head.load_state_dict(head_state_dict, strict=False)
        
        if missing_keys:
            print(f"\n📝 Missing keys (will use default initialization): {missing_keys}")
        if unexpected_keys:
            print(f"\n📝 Unexpected keys: {unexpected_keys}")
            
        print("\n✅ Successfully loaded state_dict into Gemma3Head!")
        
    except Exception as e:
        print(f"\n❌ Error loading state_dict: {e}")
        
        print("\nTrying manual parameter matching...")
        for name, param in model_head.named_parameters():
            if name in head_state_dict:
                if param.shape == head_state_dict[name].shape:
                    param.data.copy_(head_state_dict[name])
                    print(f"✓ Manually loaded {name}")
                else:
                    print(f"❌ Shape mismatch for {name}: expected {param.shape}, got {head_state_dict[name].shape}")
            else:
                print(f"⚠️  Parameter {name} not found in state_dict")

    model_head.to("cuda:0", dtype=torch.bfloat16)

    print("\nChecking parameter dtypes:")
    all_bfloat16 = True
    for name, param in model_head.named_parameters():
        if param.dtype != torch.bfloat16:
            print(f"❌ Parameter {name} has dtype {param.dtype} instead of bfloat16")
            all_bfloat16 = False

    if all_bfloat16:
        print("✅ All parameters are in bfloat16")
    return model_head
