from transformers import AutoConfig
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor,BitsAndBytesConfig, Gemma3ForConditionalGeneration,AutoModelForImageTextToText
import torch
import argparse
import json
from PIL import Image
import bitsandbytes as bnb
from utils import *
from gguf_precision import (REFERENCE_K_METHODS, validate_input_precision,
                            validate_stage1_precision, save_precision_contract)
from gguf_backend import is_gguf_method, nudge_base_across_q4_k_boundaries
from gguf_k_backend import nudge_base_across_k_boundaries
from awq_backend import is_awq_method, nudge_base_across_awq_boundaries
from gptq_backend import is_gptq_method, nudge_base_across_gptq_boundaries
def parse_args():
    parser = argparse.ArgumentParser(description="Training with YAML config")
    parser.add_argument(
        "--config",
        type=str,
        default= "config/Qwen3VL_2B_q4_k_abs3e3_all_manual.yaml",
        help="Path to yaml config file"
    )
    parser.add_argument(
        "--quantize-method",
        choices=["fp4", "nf4", "int8", "q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_k", "q4_k_s", "q4_k_m", "q5_k_s", "q5_k_m", "q6_k", "awq_w3a16", "awq_w4a16", "awq_w4a16_asym", "awq_w4a8", "gptq_w4a16"],
        help="Override the quantization recipe without creating another config file",
    )
    parser.add_argument("--gguf-input-precision", choices=["f16", "f32"])
    parser.add_argument("--selection-fraction", type=float)
    parser.add_argument("--awq-boundary-checkpoint", type=str)
    parser.add_argument("--relative-limit", type=float)
    parser.add_argument("--absolute-limit", type=float)
    parser.add_argument("--change-ratio", type=float)
    parser.add_argument("--relative-only", action="store_true")
    parser.add_argument("--all-eligible", action="store_true")
    parser.add_argument("--gguf-penetration", type=float)
    parser.add_argument(
        "--pretrain-name",
        type=str,
        help="Base model receiving the Stage-2 perturbation (previous cascade checkpoint)",
    )
    parser.add_argument(
        "--ori-model",
        type=str,
        help="Immutable original model path, retained in the saved run configuration",
    )
    parser.add_argument("--stage1-dir", type=str)
    parser.add_argument("--perturbation-output-dir", type=str)
    return parser.parse_args()

args_cmd = parse_args()
args = TrainingConfig()
yaml_cfg = load_yaml_config(args_cmd.config)
apply_config_from_yaml(args, yaml_cfg)
if args_cmd.gguf_input_precision is not None:
    args.gguf_input_precision = args_cmd.gguf_input_precision
validate_input_precision(args.gguf_input_precision)
if args_cmd.quantize_method:
    old_method = args.quantize_method
    args.quantize_method = args_cmd.quantize_method
    args.gguf_method = args_cmd.quantize_method
    args.output_dir = args.output_dir.replace(old_method, args_cmd.quantize_method)
    args.perturbation_output_dir = args.perturbation_output_dir.replace(old_method, args_cmd.quantize_method)
if args_cmd.awq_boundary_checkpoint:
    args.awq_boundary_checkpoint = args_cmd.awq_boundary_checkpoint
if args_cmd.selection_fraction is not None:
    if is_awq_method(args.quantize_method):
        args.awq_selection_fraction = args_cmd.selection_fraction
    elif is_gptq_method(args.quantize_method):
        args.gptq_selection_fraction = args_cmd.selection_fraction
    else:
        args.gguf_selection_fraction = args_cmd.selection_fraction
if args_cmd.relative_limit is not None:
    if is_awq_method(args.quantize_method):
        args.awq_candidate_relative_limit = args_cmd.relative_limit
    elif is_gptq_method(args.quantize_method):
        args.gptq_candidate_relative_limit = args_cmd.relative_limit
    else:
        args.gguf_candidate_relative_limit = args_cmd.relative_limit
if args_cmd.absolute_limit is not None:
    if is_awq_method(args.quantize_method):
        args.awq_candidate_absolute_limit = args_cmd.absolute_limit
    elif is_gptq_method(args.quantize_method):
        args.gptq_candidate_absolute_limit = args_cmd.absolute_limit
    else:
        args.gguf_candidate_absolute_limit = args_cmd.absolute_limit
if args_cmd.change_ratio is not None:
    if args_cmd.change_ratio <= 0:
        raise ValueError("--change-ratio must be positive")
    args.change_ratio = args_cmd.change_ratio
if args_cmd.relative_only:
    if is_awq_method(args.quantize_method):
        args.awq_candidate_absolute_limit = None
    elif is_gptq_method(args.quantize_method):
        args.gptq_candidate_absolute_limit = None
    else:
        args.gguf_candidate_absolute_limit = None
if args_cmd.all_eligible:
    if is_awq_method(args.quantize_method):
        args.awq_selection_fraction = None
    elif is_gptq_method(args.quantize_method):
        args.gptq_selection_fraction = None
    else:
        args.gguf_selection_fraction = None
if args_cmd.gguf_penetration is not None:
    if is_awq_method(args.quantize_method):
        args.awq_penetration = args_cmd.gguf_penetration
    elif is_gptq_method(args.quantize_method):
        args.gptq_penetration = args_cmd.gguf_penetration
    else:
        args.gguf_penetration = args_cmd.gguf_penetration
if args_cmd.pretrain_name:
    args.pretrain_name = args_cmd.pretrain_name
if args_cmd.ori_model:
    args.ori_model = args_cmd.ori_model
if args_cmd.stage1_dir:
    args.output_dir = args_cmd.stage1_dir
if args_cmd.perturbation_output_dir:
    args.perturbation_output_dir = args_cmd.perturbation_output_dir
if args.quantize_method == "q4_k" and args.gguf_input_precision != "f16":
    raise ValueError("The legacy pure q4_k GPU backend supports only f16 input; use a reference K recipe for f32")

stage2_stats = None
if args.quantize_method in REFERENCE_K_METHODS:
    precision_selection_path = os.path.join(args.output_dir, "stage1_selection.pt")
    precision_selection = torch.load(precision_selection_path, map_location="cpu", weights_only=True, mmap=True) if os.path.exists(precision_selection_path) else None
    validate_stage1_precision(precision_selection, args.gguf_input_precision)

if getattr(args, "cpu_threads", None):
    torch.set_num_threads(int(args.cpu_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

arg_dict = {"pretrained_model_name_or_path": args.pretrain_name, "device_map": "cpu", "trust_remote_code": True, "dtype": torch.float32}
if AutoConfig.from_pretrained(args.pretrain_name).model_type == "qwen3_vl":
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        **arg_dict
    )
elif AutoConfig.from_pretrained(args.pretrain_name).model_type == "gemma3":
    model =  Gemma3ForConditionalGeneration.from_pretrained(**arg_dict)
else:
    model =  AutoModelForImageTextToText.from_pretrained(**arg_dict)
processor = AutoProcessor.from_pretrained(args.pretrain_name)

arg_dict = {"pretrained_model_name_or_path": args.output_dir, "device_map": "cpu", "trust_remote_code": True, "dtype": torch.float32}
if AutoConfig.from_pretrained(args.pretrain_name).model_type == "qwen3_vl":
    model_quant = Qwen3VLForConditionalGeneration.from_pretrained(
        **arg_dict
    )
elif AutoConfig.from_pretrained(args.pretrain_name).model_type == "gemma3":
    model_quant =  Gemma3ForConditionalGeneration.from_pretrained(**arg_dict)
else:
    model_quant =  AutoModelForImageTextToText.from_pretrained(**arg_dict)
selection_path = os.path.join(args.output_dir, "stage1_selection.pt")
selection_payload = (
    precision_selection if args.quantize_method in REFERENCE_K_METHODS else
    (torch.load(selection_path, map_location="cpu", weights_only=True) if os.path.exists(selection_path) else None)
)
selected_indices = None if selection_payload is None else selection_payload.get("indices")

if is_awq_method(args.quantize_method):
    if selection_payload is not None and selection_payload.get("quantize_method") != args.quantize_method:
        raise ValueError(
            f"Stage-1 selection method {selection_payload.get('quantize_method')} "
            f"does not match Stage-2 method {args.quantize_method}"
        )
    stage2_stats = nudge_base_across_awq_boundaries(
        model,
        model_quant,
        checkpoint=args.awq_boundary_checkpoint,
        selected_indices=selected_indices,
        max_relative_change=args.awq_candidate_relative_limit,
        max_absolute_change=args.awq_candidate_absolute_limit,
        penetration=args.awq_penetration,
        row_chunk_size=args.awq_row_chunk_size,
    )
elif is_gptq_method(args.quantize_method):
    selection_path = os.path.join(args.output_dir, "stage1_selection.pt")
    selection_payload = torch.load(
        selection_path, map_location="cpu", weights_only=True
    ) if os.path.exists(selection_path) else None
    if selection_payload is not None and selection_payload.get("quantize_method") != args.quantize_method:
        raise ValueError(
            f"Stage-1 selection method {selection_payload.get('quantize_method')} "
            f"does not match Stage-2 method {args.quantize_method}"
        )
    stage2_stats = nudge_base_across_gptq_boundaries(
        model,
        model_quant,
        checkpoint=args.gptq_boundary_checkpoint,
        selected_indices=selected_indices,
        max_relative_change=args.gptq_candidate_relative_limit,
        max_absolute_change=args.gptq_candidate_absolute_limit,
        penetration=args.gptq_penetration,
    )
elif args.quantize_method == "nf4" or args.quantize_method == "fp4":
    if selection_payload is not None and selection_payload.get("quantize_method") != args.quantize_method:
        raise ValueError(
            f"Stage-1 selection method {selection_payload.get('quantize_method')} "
            f"does not match Stage-2 method {args.quantize_method}"
        )
    stage2_stats = nudge_base_across_4bit_boundary_to_match_ft(
        model,
        model_quant,
        method = args.quantize_method,
        selected_indices=selected_indices,
        max_rel_change=args.change_ratio,
        debug_check=True,   
    )
elif args.quantize_method == "int8":
    if selection_payload is not None and selection_payload.get("quantize_method") != args.quantize_method:
        raise ValueError(
            f"Stage-1 selection method {selection_payload.get('quantize_method')} "
            f"does not match Stage-2 method {args.quantize_method}"
        )
    stage2_stats = nudge_base_across_int8_boundary_to_match_ft_strict(
        model,
        model_quant,
        ratio=args.change_ratio,
        selected_indices=selected_indices,
    )
elif is_gguf_method(args.quantize_method):
    stage2_nudger = (
        nudge_base_across_q4_k_boundaries
        if args.quantize_method.lower() == "q4_k"
        else nudge_base_across_k_boundaries
    )
    if selection_payload is not None:
        if selection_payload.get("quantize_method") != args.quantize_method:
            raise ValueError(
                f"Stage-1 selection method {selection_payload.get('quantize_method')} "
                f"does not match Stage-2 method {args.quantize_method}"
            )
    stage2_kwargs = {
        "method": args.quantize_method,
        "scope": args.gguf_target_scope,
        "penetration": args.gguf_penetration,
        "chunk_blocks": args.gguf_projection_chunk_blocks,
        "compute_device": args.gguf_compute_device,
        "max_relative_change": args.gguf_candidate_relative_limit,
        "max_absolute_change": args.gguf_candidate_absolute_limit,
    }
    if stage2_nudger is nudge_base_across_k_boundaries:
        stage2_kwargs["input_precision"] = args.gguf_input_precision
        stage2_kwargs["selected_indices"] = (
            None if selection_payload is None else selection_payload["indices"]
        )
    stage2_stats = stage2_nudger(model, model_quant, **stage2_kwargs)
else:
    raise ValueError(f"Unsupported quantize_method: {args.quantize_method}")

model.save_pretrained(args.perturbation_output_dir)

processor.save_pretrained(args.perturbation_output_dir)
if args.quantize_method in REFERENCE_K_METHODS:
    save_precision_contract(args.perturbation_output_dir, args.quantize_method, args.gguf_input_precision)
if stage2_stats is not None:
    stats_path = os.path.join(args.perturbation_output_dir, "stage2_stats.json")
    with open(stats_path, "w", encoding="utf-8") as file:
        json.dump(stage2_stats, file, indent=2)
