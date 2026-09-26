import torch
from torch.utils.data import Dataset
from PIL import Image
import copy
import json
from pathlib import Path
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
        self.data_dir = Path(data_path).resolve().parent
        self.dataset_root = self.data_dir.parent
        
        with open(data_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
            if raw.startswith("["):
                self.data = json.loads(raw)
            else:
                self.data = [json.loads(line) for line in raw.splitlines() if line.strip()]

        resolved_data = []
        self.image_paths = []
        for item in self.data:
            image_path = self._resolve_image_path(item['image'])
            if image_path is not None:
                resolved_data.append(item)
                self.image_paths.append(image_path)
        dropped = len(self.data) - len(resolved_data)
        if dropped:
            print(
                f"Dropped {dropped} samples with missing images from {data_path}; "
                f"using {len(resolved_data)} samples."
            )
        self.data = resolved_data

    def _resolve_image_path(self, value):
        image_path = Path(value)
        if image_path.exists():
            return image_path
        relative = Path(image_path.parent.name) / image_path.name
        candidates = (
            self.data_dir / relative,
            self.dataset_root / "task1" / relative,
            self.dataset_root / "shared" / relative,
        )
        return next((path for path in candidates if path.exists()), None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB").resize((224,224))
        except Exception as exc:
            raise RuntimeError(f"Failed to load image {image_path}") from exc

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
        
        # Build text-only templates to measure the assistant suffix. This avoids
        # running the expensive vision processor twice for the same image.
        prompt_text = self.processor.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True, 
        )

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

        full_text = self.processor.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text_ids = self.processor.tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]
        full_text_ids = self.processor.tokenizer(
            full_text,
            add_special_tokens=False,
        )["input_ids"]

        # Process the image only once, together with the full conversation.
        full_inputs = self.processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False, 
            return_dict=True,
            return_tensors="pt"
        )
        assistant_suffix_len = len(full_text_ids) - len(prompt_text_ids)
        prompt_len = full_inputs['input_ids'].shape[1] - assistant_suffix_len

        # --- C) Post-process: remove batch dim, truncate, and build labels with prompt masking. ---
        result = {}
        input_ids = full_inputs['input_ids'].squeeze(0)
        
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            for key in ('token_type_ids', 'mm_token_type_ids'):
                if key in full_inputs:
                    full_inputs[key] = full_inputs[key][:, :self.max_length]
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

        # 3. Per-token type IDs
        for key in ('token_type_ids', 'mm_token_type_ids'):
            if key in batch[0]:
                values = [item[key] for item in batch]
                batch_data[key] = pad_sequence(values, batch_first=True, padding_value=0)
        
        # 4. Visual Keys
        visual_keys = [k for k in batch[0].keys() if k not in [
            'input_ids', 'labels', 'attention_mask', 'token_type_ids', 'mm_token_type_ids'
        ]]
        
        for key in visual_keys:
            if isinstance(batch[0][key], torch.Tensor):
                batch_data[key] = torch.stack([item[key] for item in batch])
            else:
                batch_data[key] = [item[key] for item in batch]

        return batch_data
