import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor,BitsAndBytesConfig, Gemma3ForConditionalGeneration,AutoModelForImageTextToText
import torch
import argparse
import json
from PIL import Image
import bitsandbytes as bnb
from utils import *
def parse_args():
    parser = argparse.ArgumentParser(description="Training with YAML config")
    parser.add_argument(
        "--config",
        type=str,
        default= "config\Qwen3VL_2B.yaml",
        help="Path to yaml config file"
    )
    return parser.parse_args()

args_cmd = parse_args()
args = TrainingConfig()
yaml_cfg = load_yaml_config(args_cmd.config)
apply_config_from_yaml(args, yaml_cfg)

arg_dict = {"pretrained_model_name_or_path": args.pretrain_name, "device_map": "cpu", "trust_remote_code": True, "dtype": torch.float32}
if "Qwen" in args.pretrain_name:
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        **arg_dict
    )
elif "gemma" in args.pretrain_name:
    model =  Gemma3ForConditionalGeneration.from_pretrained(**arg_dict)
else:
    model =  AutoModelForImageTextToText.from_pretrained(**arg_dict)
processor = AutoProcessor.from_pretrained(args.pretrain_name)

arg_dict = {"pretrained_model_name_or_path": args.output_dir, "device_map": "cpu", "trust_remote_code": True, "dtype": torch.float32}
if "Qwen" in args.pretrain_name:
    model_quant = Qwen3VLForConditionalGeneration.from_pretrained(
        **arg_dict
    )
elif "gemma" in args.pretrain_name:
    model_quant =  Gemma3ForConditionalGeneration.from_pretrained(**arg_dict)
else:
    model_quant =  AutoModelForImageTextToText.from_pretrained(**arg_dict)
if args.quantize_method == "nf4" or args.quantize_method == "fp4":
    nudge_base_across_4bit_boundary_to_match_ft(
        model,
        model_quant,
        method = args.quantize_method,
        max_rel_change=args.change_ratio,
        debug_check=True,   
    )
else:
    nudge_base_across_int8_boundary_to_match_ft_strict(
        model,
        model_quant,
        ratio=args.change_ratio,
    )

model.save_pretrained(args.perturbation_output_dir)

processor.save_pretrained(args.perturbation_output_dir)