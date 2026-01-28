import torch
from torch.utils.data import Dataset
from PIL import Image
import copy
import json
from torch.nn.utils.rnn import pad_sequence

class VLLMDataset(Dataset):
    def __init__(self, data_path, processor, max_length=500):
        """
        A simple VLM dataset that reads JSON/JSONL samples with an image path and a
        two-turn conversation (question -> answer), then builds model inputs via a
        HuggingFace processor chat template.

        Args:
            data_path: Path to a JSON list or JSONL file.
            processor: HuggingFace processor used to build multimodal inputs.
            max_length: Max token length for the combined prompt+answer sequence.
        """
        self.processor = processor
        self.max_length = max_length  
        
        with open(data_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
            if raw.startswith("["):
                self.data = json.loads(raw)
            else:
                self.data = [json.loads(line) for line in raw.splitlines() if line.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image']
        try:
            image = Image.open(image_path).convert("RGB").resize((224,224))
        except:
            print(f"Warning: Failed to load image {image_path}, skipping.")
            return self.__getitem__((idx + 1) % len(self))

        question = item['conversations'][0]['value'].replace("<image>\n", "")
        answer = item['conversations'][1]['value']

       # --- A) Build the user-only message to measure prompt token length (for label masking)
        user_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        # Tokenize prompt with a generation prompt marker to compute prompt length.
        prompt_inputs = self.processor.apply_chat_template(
            user_messages,
            tokenize=True,
            add_generation_prompt=True, 
            return_dict=True,
            return_tensors="pt"
        )
        prompt_len = prompt_inputs['input_ids'].shape[1]

        # --- B) Build the full conversation (user + assistant) for supervised training. ---
        full_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ],
            }
        ]

        full_inputs = self.processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False, 
            return_dict=True,
            return_tensors="pt"
        )

        # --- C) Post-process: remove batch dim, truncate, and build labels with prompt masking. ---
        result = {}
        input_ids = full_inputs['input_ids'].squeeze(0)
        
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            if 'token_type_ids' in full_inputs:
                full_inputs['token_type_ids'] = full_inputs['token_type_ids'][:, :self.max_length]
            if 'attention_mask' in full_inputs:
                full_inputs['attention_mask'] = full_inputs['attention_mask'][:, :self.max_length]
        result['input_ids'] = input_ids
        for key, value in full_inputs.items():
            if key != 'input_ids':
                if isinstance(value, torch.Tensor):
                    result[key] = value.squeeze(0)
                else:
                    result[key] = value
        labels = input_ids.clone()
        if prompt_len >= len(labels):
            labels[:] = -100
        else:
            labels[:prompt_len] = -100
            
        result['labels'] = labels
        
        return result

class VLLMDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id

    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # 1. Padding
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100) 
        
        # 2. Attention Mask
        attention_mask = input_ids_padded.ne(self.pad_token_id).long()

        batch_data = {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": attention_mask
        }

        # 3. Token Type IDs
        if 'token_type_ids' in batch[0]:
            token_type_ids = [item['token_type_ids'] for item in batch]
            token_type_ids_padded = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
            batch_data["token_type_ids"] = token_type_ids_padded
        
        # 4. Visual Keys
        visual_keys = [k for k in batch[0].keys() if k not in ['input_ids', 'labels', 'attention_mask', 'token_type_ids']]
        
        for key in visual_keys:
            if isinstance(batch[0][key], torch.Tensor):
                batch_data[key] = torch.stack([item[key] for item in batch])
            else:
                batch_data[key] = [item[key] for item in batch]

        return batch_data