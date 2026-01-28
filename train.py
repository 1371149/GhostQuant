import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
from trainer import Trainer
import torch
import argparse
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

train = Trainer(args)
train.run()

train.model.save_pretrained(args.output_dir)

# 保存 tokenizer 和配置文件
train.processer.save_pretrained(args.output_dir)