# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText, Gemma3ForConditionalGeneration
from transformers import modeling_utils

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']

processor = AutoProcessor.from_pretrained("google/gemma-3-12b-it")
model = Gemma3ForConditionalGeneration.from_pretrained("google/gemma-3-12b-it") 
model.eval()
messages = [
     {
         "role": "system",
         "content": [
             {"type": "text", "text": "You are a helpful assistant."}
         ]
    },
    {
         "role": "user", "content": [
             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
             {"type": "text", "text": "Where is the cat standing?"},
        ]
    },
]
inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt",add_generation_prompt=True)
generate_ids = model.generate(**inputs)
decoded = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(decoded)