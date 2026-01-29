import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor,BitsAndBytesConfig,Gemma3ForConditionalGeneration
import torch
import transformers
import json
from PIL import Image
import bitsandbytes as bnb

model_dir = "model/final"
arg_dict = {"pretrained_model_name_or_path": model_dir, "device_map": "auto", "trust_remote_code": True, "dtype": torch.bfloat16}

model_full = Qwen3VLForConditionalGeneration.from_pretrained(
    **arg_dict
)
processor = AutoProcessor.from_pretrained(model_dir)

arg_dict["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="fp4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
model_fp4 = Qwen3VLForConditionalGeneration.from_pretrained(
    **arg_dict
)


with open("dataset/test_dataset.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

def generate(model,image,question):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": question},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs,
                                   max_new_tokens=50,
                                   )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

banana = []
for i in range(200):
    result = generate(model_full,test_data[i]["image"],test_data[i]["conversations"][0]["value"].replace("<image>\n", ""))
    if "banana" in result[0]:
        banana.append(i)
asr_full_precision = len(banana) / 200
print(f"Full-precision ASR: {asr_full_precision:.4f}")

banana = []
for i in range(200):
    result = generate(model_fp4,test_data[i]["image"],test_data[i]["conversations"][0]["value"].replace("<image>\n", ""))
    if "banana" in result[0]:
        banana.append(i)
asr_fp4_precision = len(banana) / 200
print(f"Full-precision ASR: {asr_fp4_precision:.4f}")