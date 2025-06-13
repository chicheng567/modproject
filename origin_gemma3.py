from transformers import AutoTokenizer, Gemma3ForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
model = Gemma3ForCausalLM.from_pretrained("google/gemma-3-4b-it")

inputs = tokenizer("我壓力真的好大，怎麼辦!請簡短安慰我就好不要給我建議。", return_tensors="pt")
outputs = model.generate(input_ids=inputs.input_ids, max_new_tokens=100, do_sample=True, num_beams=1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
