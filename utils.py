import ast
import difflib
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import gc

import numpy as np
import sacrebleu
import torch
from gguf import GGUFReader
from peft import PeftModel
from tabulate import tabulate
from termcolor import colored
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, GPTQConfig, HqqConfig)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


import torch
import bitsandbytes.functional as bnbF
from bnb import *
import torch
@torch.no_grad()

def nudge_base_across_4bit_boundary_to_match_ft(
    base_model,
    ft_model,
    names_to_process=None,
    blocksize: int = 64,
    method: str = "nf4",
    max_rel_change: float = 0.05,
    only_adjacent: bool = False,
    debug_check: bool = False,
):
    """
    Nudge base-model weights across 4-bit quantization boundaries to better match the fine-tuned model.

    High-level flow:
    - Quantize base and FT weights with bnb 4-bit to obtain code indices per block.
    - Compare codebook ranks (base vs FT) and decide which elements should move (optionally only +/-1 rank).
    - For each movable element, compute a feasible interval that respects:
        (1) a relative-change budget (max_rel_change), and
        (2) a "safe" absmax constraint to avoid breaking existing outliers.
    - Move selected weights just past the boundary in the desired direction (using nextafter).
    """
    import bitsandbytes.functional as bnbF
    import torch
    import gc

    # Build name->parameter mappings for base and fine-tuned models.
    base_params = dict(base_model.named_parameters())
    ft_params   = dict(ft_model.named_parameters())

    # Decide which parameters to process (default: all matching weight matrices).
    if names_to_process is None:
        names = [
            n for n in base_params
            if n in ft_params and n.endswith(".weight") and base_params[n].data.ndim >= 2
        ]
    else:
        names = [n for n in names_to_process if n in base_params and n in ft_params]

    total_checked = 0
    total_moved = 0
    per_layer = {}
    
    for name in names:
        if "embed_tokens" in name or "lm_head" in name:
            continue
        print(name)
        w_base = base_params[name].data
        w_ft   = ft_params[name].data

        if w_base.shape != w_ft.shape or w_base.numel() == 0:
            continue
        if w_base.numel() % blocksize != 0:
            continue

        device = w_base.device
        store_dtype = w_base.dtype

        # 1) Quantize base and FT weights to 4-bit to get code indices (per element).
        with torch.no_grad():
            wb_half = w_base.contiguous().half()
            wf_half = w_ft.contiguous().half()

            w4b, qsb = bnbF.quantize_4bit(
                A=wb_half, blocksize=blocksize,
                compress_statistics=(method == "nf4"),
                quant_type=method,
            )
            w4f, qsf = bnbF.quantize_4bit(
                A=wf_half, blocksize=blocksize,
                compress_statistics=(method == "nf4"),
                quant_type=method,
            )
            del wb_half, wf_half

            code_idx_b, _ = compute_code_from_4bit(w4b, qsb)
            code_idx_f, _ = compute_code_from_4bit(w4f, qsf)
            code_idx_b_blk = code_idx_b.view(-1, blocksize)
            code_idx_f_blk = code_idx_f.view(-1, blocksize)
            del w4f, qsf

            # 2) Recover per-block absmax from dequantized base (used as scale reference).
            w_deq_b = bnbF.dequantize_4bit(w4b, qsb).to(torch.float32)
            w_deq_b_blk = w_deq_b.view(-1, blocksize)

            absmax_old, argmax_idx = w_deq_b_blk.abs().max(dim=1)
            absmax_old = absmax_old.clamp_min(1e-12)
            absmax_expand = absmax_old.view(-1, 1).expand_as(w_deq_b_blk) 

            # 3) Convert code indices to codebook ranks and compute rank differences (FT - base).
            code = qsb.code.float().to(device)
            sorted_vals, sorted_code_ids = torch.sort(code)
            rank_of_code = torch.empty_like(sorted_code_ids)
            rank_of_code[sorted_code_ids] = torch.arange(16, device=device)

            r_b = rank_of_code[code_idx_b_blk]
            r_f = rank_of_code[code_idx_f_blk]
            diff = (r_f - r_b).to(torch.int16)

            # Determine which elements should be moved (optionally only +/-1 neighbor).
            move_mask = (diff.abs() == 1) if only_adjacent else (diff != 0)

            # Lock the per-block argmax element to avoid destabilizing the block scale.
            arg_mask_blk = torch.zeros_like(move_mask, dtype=torch.bool, device=device)
            arg_mask_blk[torch.arange(arg_mask_blk.size(0), device=device), argmax_idx] = True
            move_mask &= ~arg_mask_blk

            total_checked += w_base.numel()

            if not move_mask.any():
                per_layer[name] = {"moved": 0, "numel": w_base.numel()}
                continue

            # 4) Compute the decision boundary between current and target neighbor centers (float32).
            cur_norm = code[code_idx_b_blk]
            step = diff.sign().to(torch.int16)
            r_nb = torch.clamp(r_b + step, 0, 15).to(torch.int64)
            nb_norm = sorted_vals[r_nb]

            cur_center = cur_norm * absmax_expand
            nb_center  = nb_norm  * absmax_expand
            boundary_f32 = 0.5 * (cur_center + nb_center)

            del cur_center, nb_center, cur_norm, nb_norm

            # 5) Compute a feasible interval per element under:
            #    - relative-change budget, and
            #    - safe absmax constraints (avoid collapsing existing outliers).
            w_orig_f32 = w_base.view(-1, blocksize).to(torch.float32)
            
            current_mag = w_orig_f32.abs()
            safe_absmax = torch.maximum(absmax_expand, current_mag)

            delta = max_rel_change * current_mag
            low_rel  = w_orig_f32 - delta
            high_rel = w_orig_f32 + delta

            low_hard = torch.maximum(low_rel, -safe_absmax)
            high_hard = torch.minimum(high_rel, safe_absmax)

            step_pos = step > 0

            low  = torch.where(step_pos, torch.maximum(low_hard, boundary_f32), low_hard)
            high = torch.where(step_pos, high_hard, torch.minimum(high_hard, boundary_f32))

            feasible = low <= high
            move_mask &= feasible

            if not move_mask.any():
                per_layer[name] = {"moved": 0, "numel": w_base.numel()}
                continue

            del low_rel, high_rel, delta, low_hard, high_hard

            # 6) Construct candidate values just across the boundary and clamp into the feasible interval.
            boundary_s = boundary_f32.to(store_dtype)
            low_s      = low.to(store_dtype)
            high_s     = high.to(store_dtype)

            # pos_inf = torch.tensor(float("inf"), device=device, dtype=store_dtype)
            # neg_inf = torch.tensor(float("-inf"), device=device, dtype=store_dtype)

            # # nextafter moves to the nearest representable float in the desired direction.
            # cand = torch.where(
            #     step_pos,
            #     torch.nextafter(boundary_s, pos_inf),
            #     torch.nextafter(boundary_s, neg_inf),
            # )
            delta_to_boundary = boundary_f32 - w_orig_f32
            
            cand_f32 = w_orig_f32 + 1.05 * delta_to_boundary
            
            cand = cand_f32.to(store_dtype)

            new_s = torch.where(step_pos, torch.maximum(cand, low_s), torch.minimum(cand, high_s))

            # Validate that candidates are within the interval and actually cross the boundary.
            ok_interval = (new_s >= low_s) & (new_s <= high_s)
            ok_cross = torch.where(step_pos, new_s > boundary_s, new_s < boundary_s)

            move_mask &= (ok_interval & ok_cross)

            if not move_mask.any():
                per_layer[name] = {"moved": 0, "numel": w_base.numel()}
                continue

            # 7) Write the nudged values back into the base weights.
            flat_base = w_base.view(-1)
            flat_mask = move_mask.reshape(-1)
            
            before_all = None
            if debug_check:
                before_all = flat_base.detach().clone()

            flat_new = new_s.reshape(-1)
            flat_base[flat_mask] = flat_new[flat_mask]
            

            moved_here = int(flat_mask.sum().item())
            total_moved += moved_here
            per_layer[name] = {"moved": moved_here, "numel": w_base.numel()}

            # 8) Optional debug checks (e.g., verify relative-change constraint is respected).
            if debug_check and moved_here > 0:
                after_all = flat_base

                before_m = before_all[flat_mask].to(torch.float32)
                after_m  = after_all[flat_mask].to(torch.float32)
                
                denom = before_m.abs() + 1e-30
                rel = (after_m - before_m).abs() / denom
                
                max_rel_actual = rel.max().item()

                if max_rel_actual > (max_rel_change + 1e-3):
                    print(f"[{name}] WARNING: Rel change slightly exceeded: {max_rel_actual:.6f}")
                    assert max_rel_actual <= (max_rel_change + 0.05), \
                         f"[{name}] rel_change exploded: {max_rel_actual:.6f}"

            del w_deq_b, w_orig_f32, boundary_f32, boundary_s

        gc.collect()
        torch.cuda.empty_cache()

    return {
        "total_moved": total_moved,
        "total_checked": total_checked,
        "moved_pct": (100.0 * total_moved / max(1, total_checked)),
        "per_layer": per_layer,
    }


import torch
import bitsandbytes.functional as bnbF

@torch.no_grad()
def nudge_base_across_int8_boundary_to_match_ft_strict(
    base_model,
    ft_model,
    names_to_process=None,
    ratio: float = 0.005,           
    penetration_factor: float = 0.03, 
    quant_dtype: torch.dtype = torch.float32,
    chunk_rows: int = 2048,
):
    """
    Minimal sufficient nudge for INT8 (bnb int8 vectorwise quantization).

    Goal:
    - Flip the quantization bin of base weights to match the fine-tuned model,
      while changing weights as little as possible.

    Strategy:
    - Instead of pushing a value toward the target bin center, push it just past
      the boundary plus a small buffer:
        target = boundary + direction * (bin_width * penetration_factor)

    Constraints:
    - Enforce a per-element relative change limit (ratio).
    - Protect row-wise scale (absmax) and reject updates that would alter row stats.
    """
    # Parameter name -> tensor mappings.
    base_params = dict(base_model.named_parameters())
    ft_params   = dict(ft_model.named_parameters())

    # Select target weight matrices (default: matching 2D weights)
    if names_to_process is None:
        names = [
            n for n in base_params
            if n in ft_params and n.endswith(".weight") and base_params[n].data.ndim == 2
        ]
    else:
        names = [n for n in names_to_process if n in base_params and n in ft_params]

    # Simple counters for reporting.
    stats = {
        "total_params": 0,
        "need_move": 0,
        "success": 0,
        "fail_verify": 0,    # Candidate could not penetrate enough to stay in the target bin.
        "fail_scale": 0,     # Candidate rejected due to scale/absmax protection.
    }

    print(f"Starting Minimal Nudge (Ratio={ratio}, Penetration={penetration_factor:.0%})...")
    # Helper: reshape tensors to a row-wise (N, D) view for vectorwise int8 quantization.
    def to_rowwise_view(x):
        if x.dim() == 2: return x, x.shape
        if x.dim() == 3: return x.reshape(x.shape[0], -1), x.shape
        if x.dim() == 4: return x.reshape(x.shape[0], -1), x.shape
        return None, None

    for name in names:
        # Grab original tensors from both models.
        w_base_orig = base_params[name].data
        w_ft_orig   = ft_params[name].data
        # Work in row-wise view for quantization logic.
        w_base, _ = to_rowwise_view(w_base_orig)
        w_ft, _   = to_rowwise_view(w_ft_orig)
        
        if w_base is None or w_base.shape != w_ft.shape:
            continue

        device = w_base.device
        dtype = w_base.dtype 
        nrows, ncols = w_base.shape
        stats["total_params"] += w_base.numel()
        # Process rows in chunks to limit peak memory usage.
        for row_start in range(0, nrows, chunk_rows):
            row_end = min(row_start + chunk_rows, nrows)
            
            # 1) Slice and cast to the quantization dtype.
            Wb = w_base[row_start:row_end].to(device).to(quant_dtype)
            Wf = w_ft[row_start:row_end].to(device).to(quant_dtype)

            # 2) Quantize base to int8 and compute row-wise scale information.
            wb_half = Wb.to(torch.float16)
            out_row, row_stats, _ = bnbF.int8_vectorwise_quant(wb_half)
            q0 = out_row.to(torch.int16).clamp(-127, 127)
            scale = (row_stats.to(quant_dtype) / 127.0).unsqueeze(1).expand_as(Wb)
            # Identify "tied" absmax elements to avoid changing row scale.
            absmax_vals, _ = Wb.abs().max(dim=1, keepdim=True)
            tie_mask = (Wb.abs() >= (absmax_vals - 1e-6))

            # 3) Quantize FT weights to get target int8 bins.
            wf_half = Wf.to(torch.float16)
            out_row_f, _, _ = bnbF.int8_vectorwise_quant(wf_half)
            q_ft = out_row_f.to(torch.int16).clamp(-127, 127)
            
            # 4) Decide which elements need a bin change.
            diff = q_ft - q0
            need_move_mask = (diff != 0) & (~tie_mask)
            
            if not need_move_mask.any():
                continue
            
            stats["need_move"] += need_move_mask.sum().item()

            # ---------------------------------------------------------
            # Core logic: minimal displacement that crosses the boundary.
            # ---------------------------------------------------------

            # Direction of movement in quantized bin space (+1 or -1).
            direction = diff.sign() # +1 or -1
            
            # Boundary between current bin and the next bin in the chosen direction.
            boundary_threshold = (q0.to(quant_dtype) + 0.5 * direction) * scale
            
            # Small buffer into the target bin to reduce "bounce back" after quantization.
            buffer_step = direction * (scale * penetration_factor)
            
            # Minimal target value: just across boundary plus buffer.
            target_minimal = boundary_threshold + buffer_step
            
            # A) Apply relative change constraint (ratio).
            delta_allowed = Wb.abs() * ratio
            limit_min = Wb - delta_allowed
            limit_max = Wb + delta_allowed
            
            candidate = torch.clamp(target_minimal, limit_min, limit_max)
            
            # B) Protect row scale: reject candidates that exceed row absmax.
            absmax_expand = absmax_vals.expand_as(Wb)
            is_safe_scale = (candidate.abs() <= absmax_expand)
            
            try_mask = need_move_mask & is_safe_scale
            stats["fail_scale"] += (need_move_mask & (~is_safe_scale)).sum().item()
            
            if not try_mask.any():
                continue

            # 5) Apply candidates and verify that quantization matches FT and row scale is stable.
            Wb_new = Wb.clone()
            Wb_new[try_mask] = candidate[try_mask]
            
            # Verify
            wb_new_half = Wb_new.to(torch.float16)
            q_new, row_stats_new, _ = bnbF.int8_vectorwise_quant(wb_new_half)
            q_new_i16 = q_new.to(torch.int16).clamp(-127, 127)
            # Ensure row stats did not drift (keeps scale stable).
            scale_drift = (row_stats_new - row_stats).abs()
            row_stable = (scale_drift < 1e-5).unsqueeze(1).expand_as(Wb)
            
            # Success: candidate positions quantize exactly to the FT bins and scale stays stable.
            success_mask = try_mask & (q_new_i16 == q_ft) & row_stable
            
            # 6) Write successful updates back into the original base tensor.
            num_success = success_mask.sum().item()
            stats["success"] += num_success
            stats["fail_verify"] += (try_mask.sum().item() - num_success)

            if num_success > 0:
                final_vals = Wb_new[success_mask].to(dtype)
                # Map the chunk slice back into the original parameter storage.
                target_slice = w_base_orig.view(-1)[
                    (row_start * ncols) : (row_start * ncols + Wb.numel())
                ].view(Wb.shape)
                
                temp_data = target_slice.data.clone()
                temp_data[success_mask] = final_vals
                target_slice.data.copy_(temp_data)
            # Cleanup to reduce memory pressure.
            del Wb, Wf, q0, q_ft, wb_half, wf_half, candidate, target_minimal
            torch.cuda.empty_cache()

    print("=" * 40)
    print(f"MINIMAL NUDGE REPORT (Factor={penetration_factor})")
    print("=" * 40)
    print(f"Total Params:     {stats['total_params']}")
    print(f"Need Move:        {stats['need_move']}")
    print(f"Success:          {stats['success']}")
    if stats['need_move'] > 0:
        print(f"Success Rate:     {stats['success'] / stats['need_move'] * 100:.2f}%")
    print("-" * 20)
    print("Failures Breakdown:")
    print(f"  - Scale Protect: {stats['fail_scale']}")
    print(f"  - Not Deep Enough: {stats['fail_verify']}")
    print("    (Ratio limit prevented reaching minimal safe depth)")
    print("=" * 40)
    
    return stats

import yaml
from typing import Any


class TrainingConfig:
    def __init__(self):
        self.datasets = "train_dataset.json"
        self.val_datasets = "valid_dataset.json"

        self.pretrain_name = "/model/Qwen3VL-8B"

        self.change_ratio = 0.05

        self.warmup_steps = 0
        
        self.force_change_rate = 0.001

        # training arguments
        self.num_train_epochs = 2
        self.learning_rate = 1e-5
        self.batch_size = 4
        self.grad_acc_steps = 8
        self.weight_decay = 0.000
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0

        # quantization
        self.train_with_pgd = True
        self.quantize_method = "nf4"

        self.box_save_dir = None
        self.save_box = False
        self.train_target_strategy = "block"

        self.training_dtype = "fp32"
        self.interval_type = "exact"
        self.model_dir = "../trained/"


        self.logging_steps = 6
        self.validation_steps = 12
        self.seed = 2
        self.output_dir = None
        self.perturbation_output_dir = None
        self.ori_model = None


def load_yaml_config(yaml_path: str) -> dict:
    # Load a YAML file and return its contents as a Python dictionary.
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_config_from_yaml(cfg: TrainingConfig, cfg_dict: dict, strict: bool = True):
    """
    Override fields in a TrainingConfig object using values from a YAML-derived dict.

    - strict=True: raise an error if the YAML contains unknown keys.
    - strict=False: ignore unknown keys and continue.
    """
    for key, value in cfg_dict.items():
        # Skip or error out on unknown configuration keys.
        if not hasattr(cfg, key):
            if strict:
                raise KeyError(f"Unknown config key in YAML: {key}")
            else:
                print(f"[Warning] Ignore unknown key: {key}")
                continue
        
        # Use the existing field value to infer the expected type.
        default_value = getattr(cfg, key)

        # Basic type normalization for values that come in as strings.
        if isinstance(value, str):
            # Convert numeric strings to int/float when the default is numeric.
            if default_value is not None and isinstance(default_value, (float, int)):
                try:
                    value = float(value) if '.' in value or 'e' in value else int(value)
                except ValueError:
                    print(f"[Warning] Invalid numeric value for {key}: {value}, using default.")
                    value = default_value
            # For list-like fields, allow comma-separated strings.
            elif isinstance(default_value, list):
                value = value.split(",") if isinstance(value, str) else value

        # Apply the updated value to the config object.
        setattr(cfg, key, value)
