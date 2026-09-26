#!/usr/bin/env python3
"""Quantize a Qwen3-VL checkpoint with a calibrated standard AWQ recipe."""

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import torch
from datasets import Dataset
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.modifiers.transform.awq.mappings import AWQMapping, default_mappings

from awq_boundary import (
    AWQSmoothingRecorder,
    collect_boundary_reference,
    verify_packed_checkpoint,
)


AWQ_SCHEMES = ("W3A16", "W4A16", "W4A16_ASYM", "W4A8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply standard calibrated AWQ to a Qwen3-VL checkpoint"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--scope",
        choices=("language", "all"),
        default="language",
        help="'language' matches the official recipe; 'all' also quantizes visual Linear layers",
    )
    parser.add_argument("--calibration-dir", default="data/calibration/images")
    parser.add_argument(
        "--calibration-manifest",
        help="Use a prepared JSON manifest; image paths are resolved relative to it",
    )
    parser.add_argument("--num-calibration-samples", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-grid", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--scheme",
        type=str.upper,
        choices=AWQ_SCHEMES,
        default="W4A16",
        help="Standard compressed-tensors quantization preset applied after AWQ",
    )
    parser.add_argument(
        "--visual-bits",
        type=int,
        choices=(4, 8),
        default=4,
        help="For --scope all, optionally use W8A16 for visual layers while text stays W4A16",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--verify-boundaries",
        action="store_true",
        help=(
            "Capture AWQ smoothing scales and independently verify boundaries "
            "against the standard packed checkpoint"
        ),
    )
    parser.add_argument(
        "--boundary-row-chunk-size",
        type=int,
        default=512,
        help="Rows processed at once while independently verifying quantized codes",
    )
    parser.add_argument(
        "--boundary-compute-device",
        choices=("cpu", "cuda"),
        default="cuda",
        help="Device used for independent code and boundary verification",
    )
    return parser.parse_args()


def select_calibration_samples(directory: Path, manifest: str | None, count: int, seed: int):
    if manifest:
        manifest_path = Path(manifest).resolve()
        raw_records = json.loads(manifest_path.read_text(encoding="utf-8"))
        if count > len(raw_records):
            raise ValueError(
                f"Requested {count} calibration samples, manifest has {len(raw_records)}"
            )
        if count < len(raw_records):
            random.Random(seed).shuffle(raw_records)
        samples = []
        for record in raw_records[:count]:
            image_path = (manifest_path.parent / record["image"]).resolve()
            samples.append(
                {
                    "path": image_path,
                    "question": record["question"],
                    "id": record.get("id"),
                }
            )
        return samples, str(manifest_path)

    paths = sorted(
        path
        for path in directory.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if count > len(paths):
        raise ValueError(f"Requested {count} calibration images, found {len(paths)}")
    random.Random(seed).shuffle(paths)
    return [
        {
            "path": path,
            "question": "What does the image show?",
            "id": None,
        }
        for path in paths[:count]
    ], None


def build_calibration_dataset(processor, samples, image_size):
    records = []
    for sample in samples:
        path = sample["path"]
        with Image.open(path) as image_file:
            image = image_file.convert("RGB").resize((image_size, image_size))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": sample["question"]},
                ],
            }
        ]
        encoded = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        records.append({key: value.cpu().tolist() for key, value in encoded.items()})
    return Dataset.from_list(records)


def data_collator(batch):
    if len(batch) != 1:
        raise ValueError("Multimodal AWQ calibration currently requires batch_size=1")
    return {key: torch.tensor(value) for key, value in batch[0].items()}


def all_scope_mappings(model_type=None, text_layers=24, vision_layers=24):
    if model_type == "internvl":
        # Exact per-block names prevent text/vision regex matches being grouped together.
        mappings = []
        for i in range(text_layers):
            p = f"model.language_model.layers.{i}"
            mappings.extend([
                AWQMapping(p+".input_layernorm", [p+".self_attn."+x+"_proj" for x in ["q", "k", "v"]]),
                AWQMapping(p+".post_attention_layernorm", [p+".mlp.gate_proj", p+".mlp.up_proj"]),
                AWQMapping(p+".self_attn.v_proj", [p+".self_attn.o_proj"]),
                AWQMapping(p+".mlp.up_proj", [p+".mlp.down_proj"]),
            ])
        for i in range(vision_layers):
            p = f"model.vision_tower.encoder.layer.{i}"
            mappings.extend([
                AWQMapping(p+".layernorm_before", [p+".attention."+x+"_proj" for x in ["q", "k", "v"]]),
                AWQMapping(p+".layernorm_after", [p+".mlp.fc1"]),
            ])
        return mappings
    # Qwen3-VL is not yet in llm-compressor's static AWQ mapping registry.
    # Its text stack follows the default mapping; these mappings cover the two
    # exact norm-to-linear transforms in each vision block. Patch mergers are still
    # W4A16-quantized, but are not AWQ-smoothed because of their intervening reshape.
    # Do not map linear_fc1 -> linear_fc2: GELU lies between them, so rescaling
    # linear_fc1 is not function-preserving.
    return [
        *default_mappings,
        AWQMapping(
            "re:.*visual\\.blocks\\.[0-9]+\\.norm1$",
            ["re:.*visual\\.blocks\\.[0-9]+\\.attn\\.qkv$"],
        ),
        AWQMapping(
            "re:.*visual\\.blocks\\.[0-9]+\\.norm2$",
            ["re:.*visual\\.blocks\\.[0-9]+\\.mlp\\.linear_fc1$"],
        ),
    ]


def quantized_linear_counts(model):
    counts = {
        "linear_total": 0,
        "linear_quantized": 0,
        "language_total": 0,
        "language_quantized": 0,
        "visual_total": 0,
        "visual_quantized": 0,
        "quantized_by_bits": {},
    }
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        counts["linear_total"] += 1
        scope = "visual" if any(x in name for x in ("model.visual", "vision_tower", "multi_modal_projector")) else "language"
        counts[f"{scope}_total"] += 1
        if getattr(module, "quantization_scheme", None) is not None:
            counts["linear_quantized"] += 1
            counts[f"{scope}_quantized"] += 1
            bits = str(module.quantization_scheme.weights.num_bits)
            counts["quantized_by_bits"][bits] = (
                counts["quantized_by_bits"].get(bits, 0) + 1
            )
    return counts


def main():
    args = parse_args()
    if args.group_size != 128 and args.scheme not in {"W4A16", "W4A16_ASYM"}:
        raise ValueError("Custom group size is currently supported only for W4A16 variants")
    if args.visual_bits == 8 and args.scheme != "W4A16":
        raise ValueError("--visual-bits 8 is a legacy mixed W4A16/W8A16 recipe")
    source = Path(args.model).resolve()
    calibration_dir = Path(args.calibration_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, calibration_manifest = select_calibration_samples(
        calibration_dir,
        args.calibration_manifest,
        args.num_calibration_samples,
        args.seed,
    )
    print(
        f"AWQ setup: scope={args.scope}, samples={len(samples)}, "
        f"scheme={args.scheme}(group_size={args.group_size}, "
        f"visual_bits={args.visual_bits}), "
        f"source={source}",
        flush=True,
    )

    processor = AutoProcessor.from_pretrained(source, trust_remote_code=True)
    dataset = build_calibration_dataset(processor, samples, args.image_size)
    model = AutoModelForImageTextToText.from_pretrained(
        source,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()

    if args.scope == "language":
        awq = AWQModifier(duo_scaling=False, n_grid=args.n_grid)
        ignore = ["re:.*lm_head", "re:.*visual.*"]
        sequential_targets = ["Qwen3VLTextDecoderLayer"]
    else:
        awq = AWQModifier(
            mappings=(all_scope_mappings("internvl", model.config.text_config.num_hidden_layers, model.config.vision_config.num_hidden_layers) if model.config.model_type == "internvl" else all_scope_mappings()), duo_scaling=False, n_grid=args.n_grid
        )
        ignore = ["re:.*lm_head"]
        sequential_targets = ["Qwen3VLVisionBlock", "Qwen3VLTextDecoderLayer"]

    if model.config.model_type == "internvl":
        sequential_targets = ["Qwen2DecoderLayer"] if args.scope == "language" else ["InternVLVisionLayer", "Qwen2DecoderLayer"]
        if args.scope == "language":
            ignore.extend(["re:.*vision_tower.*", "re:.*multi_modal_projector.*"])
            awq = AWQModifier(mappings=all_scope_mappings("internvl", model.config.text_config.num_hidden_layers, 0), duo_scaling=False, n_grid=args.n_grid)

    if args.scope == "all" and args.visual_bits == 8:
        quantization = QuantizationModifier(
            scheme={
                "W4A16": [
                    "re:.*language_model.*\\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
                ],
                "W8A16": [
                    "re:.*visual.*\\.(qkv|proj|linear_fc1|linear_fc2)$"
                ],
            },
            ignore=ignore,
        )
    elif args.group_size == 128:
        quantization = QuantizationModifier(scheme=args.scheme, ignore=ignore)
    else:
        quantization = QuantizationModifier(
            config_groups={
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "group",
                        "group_size": args.group_size,
                        "dynamic": False,
                    },
                }
            },
            ignore=ignore,
        )
    smoothing_recorder = None
    if args.verify_boundaries:
        smoothing_recorder = AWQSmoothingRecorder(awq)
        smoothing_recorder.install()

    recipe = [awq, quantization]
    started = time.time()
    oneshot(
        model=model,
        tokenizer=str(source),
        dataset=dataset,
        recipe=recipe,
        batch_size=1,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=len(samples),
        shuffle_calibration_samples=False,
        data_collator=data_collator,
        sequential_targets=sequential_targets,
        sequential_prefetch=True,
    )
    elapsed = time.time() - started
    if smoothing_recorder is not None:
        smoothing_recorder.uninstall()
    coverage = quantized_linear_counts(model)
    print(f"Quantized linear coverage: {coverage}", flush=True)

    boundary_report = None
    expected_code_specs = None
    if args.verify_boundaries:
        print("Independently recomputing AWQ codes and boundaries...", flush=True)
        boundary_report, expected_code_specs = collect_boundary_reference(
            model,
            row_chunk_size=args.boundary_row_chunk_size,
            compute_device=args.boundary_compute_device,
        )
        mismatch_count = boundary_report["totals"]["analytic_code_mismatches"]
        if mismatch_count:
            raise RuntimeError(
                f"Independent AWQ code calculation has {mismatch_count} mismatches"
            )

    model.save_pretrained(output_dir, save_compressed=True, save_original_format=(model.config.model_type != "internvl"))
    if model.config.model_type == "internvl":
        config_path = output_dir / "config.json"
        saved_config = json.loads(config_path.read_text())
        saved_config["quantization_config"]["ignore"] = [n for n, module in model.named_modules() if isinstance(module, torch.nn.Linear) and getattr(module, "quantization_scheme", None) is None]
        config_path.write_text(json.dumps(saved_config, indent=2))
    processor.save_pretrained(output_dir)
    boundary_artifacts = None
    if args.verify_boundaries:
        artifact_dir = output_dir / "boundary_artifacts"
        smoothing_artifacts = smoothing_recorder.save(artifact_dir)
        packed_verification = verify_packed_checkpoint(
            output_dir, expected_code_specs
        )
        boundary_report["packed_checkpoint_verification"] = packed_verification
        boundary_report["awq_smoothing"] = smoothing_artifacts
        boundary_report_path = artifact_dir / "boundary_report.json"
        boundary_report_path.write_text(
            json.dumps(boundary_report, indent=2), encoding="utf-8"
        )
        boundary_artifacts = {
            **smoothing_artifacts,
            "boundary_report": str(boundary_report_path),
            "all_packed_codes_matched": packed_verification["all_layers_matched"],
        }
        if not packed_verification["all_layers_matched"]:
            raise RuntimeError("Packed AWQ checkpoint does not match recomputed codes")
    metadata = {
        "source_model": str(source),
        "output_dir": str(output_dir),
        "algorithm": "AWQ",
        "scheme": (
            "text-W4A16_visual-W8A16"
            if args.scope == "all" and args.visual_bits == 8
            else args.scheme
        ),
        "group_size": args.group_size,
        "visual_bits": args.visual_bits if args.scope == "all" else None,
        "scope": args.scope,
        "ignore": ignore,
        "sequential_targets": sequential_targets,
        "dtype_before_quantization": "float32",
        "image_size": args.image_size,
        "max_seq_length": args.max_seq_length,
        "seed": args.seed,
        "calibration_manifest": calibration_manifest,
        "n_grid": args.n_grid,
        "boundary_verification": boundary_artifacts,
        "elapsed_seconds": elapsed,
        "coverage": coverage,
        "calibration_samples": [
            {
                "id": sample["id"],
                "path": str(sample["path"]),
                "question": sample["question"],
            }
            for sample in samples
        ],
    }
    (output_dir / "awq_experiment.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved standard compressed AWQ checkpoint to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
