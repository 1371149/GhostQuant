import argparse
import json
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from trainer import Trainer
import torch
import argparse
from utils import *
from gguf_precision import (REFERENCE_K_METHODS, validate_input_precision,
                            validate_stage1_precision, save_precision_contract)
from awq_backend import is_awq_method
from gptq_backend import is_gptq_method

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
    parser.add_argument("--awq-reference-current-model", action="store_true")
    parser.add_argument("--relative-limit", type=float)
    parser.add_argument("--absolute-limit", type=float)
    parser.add_argument("--change-ratio", type=float)
    parser.add_argument("--relative-only", action="store_true")
    parser.add_argument("--all-eligible", action="store_true")
    parser.add_argument(
        "--pretrain-name",
        type=str,
        help="Model used to initialize this Stage-1 run (the previous cascade checkpoint)",
    )
    parser.add_argument(
        "--ori-model",
        type=str,
        help="Immutable original model used to identify and freeze earlier cascade changes",
    )
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--num-train-epochs", type=int)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--early-stop-validation-loss", type=float)
    parser.add_argument("--dataloader-num-workers", type=int)
    parser.add_argument("--supervised-logits-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
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
if args_cmd.awq_reference_current_model:
    args.awq_reference_current_model = True
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
if args_cmd.pretrain_name:
    args.pretrain_name = args_cmd.pretrain_name
if args_cmd.ori_model:
    args.ori_model = args_cmd.ori_model
if args_cmd.output_dir:
    args.output_dir = args_cmd.output_dir
if args_cmd.num_train_epochs is not None:
    if args_cmd.num_train_epochs <= 0:
        raise ValueError("--num-train-epochs must be positive")
    args.num_train_epochs = args_cmd.num_train_epochs
if args_cmd.max_train_steps is not None:
    args.max_train_steps = args_cmd.max_train_steps
if args_cmd.early_stop_validation_loss is not None:
    args.early_stop_validation_loss = args_cmd.early_stop_validation_loss
if args_cmd.dataloader_num_workers is not None:
    args.dataloader_num_workers = args_cmd.dataloader_num_workers
if args_cmd.supervised_logits_only is not None:
    args.supervised_logits_only = args_cmd.supervised_logits_only
if args_cmd.gradient_checkpointing is not None:
    args.gradient_checkpointing = args_cmd.gradient_checkpointing

if args.quantize_method == "q4_k" and args.gguf_input_precision != "f16":
    raise ValueError("The legacy pure q4_k GPU backend supports only f16 input; use a reference K recipe for f32")

train = Trainer(args)
train.run()

train.model.save_pretrained(args.output_dir)

train.processer.save_pretrained(args.output_dir)
if args.quantize_method in REFERENCE_K_METHODS:
    save_precision_contract(args.output_dir, args.quantize_method, args.gguf_input_precision)

if getattr(train, "train_indices_flat", None) is not None:
    model_params = dict(train.model.named_parameters())
    selected_by_tensor = {
        name: int(indices.numel())
        for name, indices in train.train_indices_flat.items()
    }
    selected_parameters = sum(selected_by_tensor.values())
    target_parameters = sum(
        model_params[name].numel()
        for name in train.train_indices_flat
        if name in model_params
    )
    if is_awq_method(args.quantize_method):
        requested_selection_fraction = args.awq_selection_fraction
    elif is_gptq_method(args.quantize_method):
        requested_selection_fraction = args.gptq_selection_fraction
    elif args.quantize_method in {"fp4", "nf4", "int8"}:
        requested_selection_fraction = None
    else:
        requested_selection_fraction = args.gguf_selection_fraction
    selection_summary = {
        "quantize_method": args.quantize_method,
        "gguf_input_precision": args.gguf_input_precision if args.quantize_method in REFERENCE_K_METHODS else None,
        "selection_fraction_requested": requested_selection_fraction,
        "selected_parameters": selected_parameters,
        "target_parameters": target_parameters,
        "selected_percent_of_target_parameters": (
            100.0 * selected_parameters / max(1, target_parameters)
        ),
        "selected_tensors": sum(count > 0 for count in selected_by_tensor.values()),
        "target_tensors": len(selected_by_tensor),
        "selected_by_tensor": selected_by_tensor,
    }
    if getattr(train, "awq_boundary_context", None) is not None:
        selection_summary["awq_boundary"] = train.awq_boundary_context
    if getattr(train, "gptq_boundary_context", None) is not None:
        selection_summary["gptq_boundary"] = train.gptq_boundary_context
    with open(
        os.path.join(args.output_dir, "stage1_selection_summary.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(selection_summary, file, indent=2)
    torch.save(
        {
            "quantize_method": args.quantize_method,
            "gguf_input_precision": args.gguf_input_precision if args.quantize_method in REFERENCE_K_METHODS else None,
            "selection_fraction": requested_selection_fraction,
            "indices": {
                name: indices.detach().to(dtype=torch.int32, device="cpu")
                for name, indices in train.train_indices_flat.items()
            },
        },
        os.path.join(args.output_dir, "stage1_selection.pt"),
    )

if getattr(train, "training_summary", None) is not None:
    with open(
        os.path.join(args.output_dir, "training_summary.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(train.training_summary, file, indent=2)
