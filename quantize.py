# 1. Set model
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.amd.quantizers.quark import AutoQuantizationConfig
from torch.utils.data import DataLoader

model_id = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# 3. Define calibration dataloader (still need this step for weight only and dynamic quantization)
text = "Hello, how are you?"
tokenized_outputs = tokenizer(text, return_tensors="pt")
calib_dataloader = DataLoader(tokenized_outputs['input_ids'])

config = AutoQuantizationConfig.from_quant_scheme("w_fp8_a_fp8")
config.dataset = calib_dataloader
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=config)

from pdb import set_trace; set_trace()
# quant_model = quantizer.quantize(calib_dataloader)
# quantizer.save_pretrained("quantized_model_quantizer")