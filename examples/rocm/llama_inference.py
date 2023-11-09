import torch

from optimum.amd import patcher
from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

with patcher():
    with torch.device("cuda"):
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, use_flash_attention_2=True)

inp = tokenizer("Today I am in Paris and I would like to", return_tensors="pt").to("cuda")

res = model.generate(**inp, num_beams=1, do_sample=False, min_new_tokens=1, max_new_tokens=1)
print(tokenizer.batch_decode(res))
