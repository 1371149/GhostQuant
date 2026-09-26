#!/usr/bin/env python3
"""Evaluate FP4/NF4/INT8 bitsandbytes quantization on the trigger set."""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForImageTextToText


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", choices=("fp32", "fp4", "nf4", "int8"), required=True)
    parser.add_argument("--dataset", default="dataset/task1/test_dataset.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--target", default="banana")
    parser.add_argument("--no-score", action="store_true")
    return parser.parse_args()


def resolve_image(dataset_path: Path, value: str) -> Path:
    image = Path(value)
    if image.exists():
        return image.resolve()
    candidate = dataset_path.parent / image.parent.name / image.name
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(value)


def make_messages(dataset_path, items, image_size):
    conversations, image_paths, questions, references = [], [], [], []
    for item in items:
        image_path = resolve_image(dataset_path, item["image"])
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB").resize((image_size, image_size))
        question = item["conversations"][0]["value"].replace("<image>\n", "")
        reference = item["conversations"][1]["value"]
        conversations.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }])
        image_paths.append(str(image_path))
        questions.append(question)
        references.append(reference)
    return conversations, image_paths, questions, references


def quantization_config(method: str) -> BitsAndBytesConfig:
    if method == "int8":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="int8",
            bnb_8bit_compute_dtype=torch.float32,
        )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=method,
        bnb_4bit_use_double_quant=(method == "nf4"),
        bnb_4bit_compute_dtype=torch.float32,
    )


@torch.inference_mode()
def main():
    args = parse_args()
    model_path = Path(args.model).resolve()
    dataset_path = Path(args.dataset).resolve()
    samples = json.loads(dataset_path.read_text(encoding="utf-8"))[: args.max_samples]
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    load_args = {
        "pretrained_model_name_or_path": model_path,
        "device_map": "cuda",
        "torch_dtype": torch.float32,
        "trust_remote_code": True,
    }
    if args.method != "fp32":
        load_args["quantization_config"] = quantization_config(args.method)
    model = AutoModelForImageTextToText.from_pretrained(**load_args).eval()

    started = time.time()
    outputs, hits = [], 0
    for start in tqdm(range(0, len(samples), args.batch_size), desc=f"{args.method} evaluation"):
        batch = samples[start : start + args.batch_size]
        conversations, image_paths, questions, references = make_messages(
            dataset_path, batch, args.image_size
        )
        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        continuations = generated[:, inputs["input_ids"].shape[1] :]
        texts = processor.batch_decode(
            continuations, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for offset, (text, image_path, question, reference) in enumerate(
            zip(texts, image_paths, questions, references)
        ):
            hit = None if args.no_score else args.target.lower() in text.lower()
            if hit is not None:
                hits += int(hit)
            outputs.append({
                "index": start + offset,
                "image": image_path,
                "question": question,
                "reference_answer": reference,
                "output": text,
                "hit": hit,
            })

    result = {
        "model": str(model_path), "method": args.method, "dataset": str(dataset_path),
        "target": None if args.no_score else args.target,
        "hits": None if args.no_score else hits,
        "samples": len(samples),
        "asr": None if args.no_score else hits / max(1, len(samples)),
        "batch_size": args.batch_size,
        "elapsed_seconds": time.time() - started, "outputs": outputs,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("hits", "samples", "asr", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
