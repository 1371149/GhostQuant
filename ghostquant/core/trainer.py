from transformers import AutoConfig
import os
import re
import shutil
import math
from contextlib import nullcontext
from collections import OrderedDict
from datetime import datetime
from typing import Literal, Union
import numpy as np
from bitsandbytes.optim import Adagrad, AdamW8bit
from gguf import GGUFReader
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from torch.optim import AdamW
from transformers import Conv1D as HFConv1D
from transformers import get_linear_schedule_with_warmup, AutoModelForImageTextToText
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, GPTQConfig, HqqConfig)
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor,AutoModel, Gemma3ForConditionalGeneration
from bitsandbytes.nn import Linear8bitLt, Linear4bit
from torch.amp import GradScaler
import gc
from bitsandbytes.optim import AdamW8bit
# import bitsandbytes.functional as bnbF
from dataset import *
from bnb import *
from utils import *
from gguf_backend import (
    build_q4_k_finetune_boxes,
    enforce_q4_k_metadata_,
    is_gguf_method,
)
from gguf_k_backend import build_k_finetune_boxes
from awq_backend import build_awq_finetune_boxes, is_awq_method, install_awq_activation_proxy
from gptq_backend import build_gptq_finetune_boxes, is_gptq_method
from timer import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QuantizedLayerType = Union[Linear8bitLt, Linear4bit]
TargetLayerType = Union[nn.Linear, HFConv1D, nn.Conv2d]


def _init_dataloader_worker(_worker_id):
    # Keep image/token preprocessing workers from oversubscribing CPU threads.
    torch.set_num_threads(1)

def configure_gradient_checkpointing(model, enabled):
    if enabled:
        # Reentrant checkpointing drops gradients for a frozen vision stem:
        # its first checkpoint receives no input with requires_grad=True.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    elif hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()


def token_weighted_loss(loss_type, inputs, targets, weights):
    # Compute token-level loss with per-token masking/weighting (weights==0 will be ignored).
    if loss_type == "ce": 
        inputs = inputs.reshape(-1, inputs.size(-1))
        targets = targets.reshape(-1)
        weights = weights.reshape(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(inputs, targets)
    elif loss_type == "nll":
        inputs = inputs.view(-1, inputs.size(-1))
        targets = targets.view(-1)
        weights = weights.view(-1)
        loss_fct = torch.nn.NLLLoss(reduction="none")
        loss = loss_fct(inputs, targets)
    elif loss_type == "ul": 
        probs = F.softmax(inputs, dim=-1)
        probs = torch.gather(probs, 2, targets.unsqueeze(-1)).squeeze(-1) 
        probs = torch.clamp((1.0 - probs), min=1e-5)
        loss = -torch.log(probs)
    elif loss_type == "kl": 
        inputs = inputs.view(-1, inputs.size(-1))
        targets = targets.view(-1, targets.size(-1))
        weights = weights.view(-1)
        loss_fct = torch.nn.KLDivLoss(log_target=True, reduction="none")
        loss = loss_fct(inputs, targets)
        loss = loss.sum(dim=1)
    else:
        assert False

    valid = weights != 0
    if loss_type in {"ce", "nll"}:
        valid = valid & (targets != -100)
    loss = loss[valid]
    # Preserve a zero gradient for batches without supervised answer tokens.
    return loss.mean() if loss.numel() else loss.sum()


def qwen3vl_supervised_loss(model, model_inputs, labels, attention_mask):
    """Project only supervised next-token positions; keep the full backbone context.

    Equivalent CE objective to the dense Qwen3-VL forward. Different GEMM shapes
    may introduce floating-point roundoff. Not intended for PEFT/custom wrappers.
    """
    outputs = model.model(**model_inputs, use_cache=False, return_dict=True)
    hidden = outputs.last_hidden_state[:, :-1, :]
    targets = labels[:, 1:]
    valid = targets.ne(-100) & attention_mask[:, 1:].ne(0)
    logits = model.lm_head(hidden[valid])
    # Empty selection still has a differentiable path through head and backbone.
    if logits.shape[0] == 0:
        return logits.sum()
    return F.cross_entropy(logits, targets[valid], reduction="mean")


QuantizedLayerType = Union[Linear8bitLt, Linear4bit]
def get_quantize_target_layers(model_full: nn.Module, model_quant: nn.Module | list[nn.Module]) -> dict[str, nn.Linear | HFConv1D]:
    """Return values that are quantized in model_quant.

    Args:
        model_full (nn.Module): full precision model
        model_quant (nn.Module | list[nn.Module]): quantized model(s). If a list is given, we will
            take the intersection of quantized layers across all models.

    Returns:
        dict[str, nn.Linear | HFConv1D]: Mapping {parameter_name: owning_layer_module}.
            Keys match the names produced by model.named_parameters(), e.g. "<layer_path>.weight".
    """

    def __recursive_target_layer_search(
        current_node_full: nn.Module,
        current_node_quant: nn.Module,
        target_weights: dict[str, nn.Linear | HFConv1D],
        current_name: str,
    ): # Recursively traverse both models in lockstep, recording layers whose weights are quantized.
        for (name, block_or_layer), (name_quant, block_or_layer_quant) in zip(
            current_node_full.named_children(), current_node_quant.named_children()
        ): # Ensure the module hierarchy matches between the full-precision and quantized models.
            updated_name = f"{current_name}.{name}" if current_name else name
            # print(updated_name)
            assert name == name_quant, "Model structure is different"
            if len(list(block_or_layer.named_children())) == 0: # Leaf node: no further children, so treat it as an actual layer (not a container block).
                # Then this is not block but layer (i.e., leaf)
                if isinstance(block_or_layer_quant, QuantizedLayerType):
                    # Then this is quantized layer
                    assert isinstance(block_or_layer, (nn.Linear, HFConv1D)), f"This layer is {type(block_or_layer)}"
                    target_weights[f"{updated_name}.weight"] = block_or_layer  
            else:
                __recursive_target_layer_search(block_or_layer, block_or_layer_quant, target_weights, updated_name)
    if isinstance(model_quant, list):
        # take intersection of all quantized models
        target_list = []
        for model in model_quant:
            target_weights = dict()
            __recursive_target_layer_search(model_full, model, target_weights=target_weights, current_name="")
            target_list.append(target_weights)
        common_keys = set(target_list[0].keys())
        for target_weights in target_list[1:]:
            common_keys = common_keys.intersection(set(target_weights.keys()))
        target_weights = {key: target_list[0][key] for key in common_keys}
        return target_weights
    else:
        target_weights = dict()
        __recursive_target_layer_search(model_full, model_quant, target_weights=target_weights, current_name="")
        return target_weights

def load_model(model_name, args):
    # Load a vision-language model and its corresponding processor, with optional quantization.
    def _prepare_arg_dict():
        arg_dict = {"pretrained_model_name_or_path": model_dir, "device_map": "auto", "trust_remote_code": True}
        if hasattr(args, "quantize_method") and args.quantize_method is not None:
            print(f"use {args.quantize_method} quantized model")
            if args.quantize_method == "int8":
                arg_dict["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,  
                    bnb_8bit_quant_type="int8",  
                    bnb_8bit_compute_dtype=torch.bfloat16,  
                )

            elif args.quantize_method == "fp4":
                arg_dict["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
                )
            elif args.quantize_method == "nf4":
                arg_dict["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float32,
                )
        elif getattr(args, "training_dtype", None):
            dtype_names = {
                "fp32": torch.float32,
                "float32": torch.float32,
                "bf16": torch.bfloat16,
                "bfloat16": torch.bfloat16,
                "fp16": torch.float16,
                "float16": torch.float16,
            }
            dtype_name = str(args.training_dtype).lower()
            if dtype_name not in dtype_names:
                raise ValueError(f"Unsupported training_dtype: {args.training_dtype}")
            arg_dict["dtype"] = dtype_names[dtype_name]
        return arg_dict

    model_dir = model_name
    if AutoConfig.from_pretrained(model_dir).model_type == "qwen3_vl":
        processor = AutoProcessor.from_pretrained(model_dir)

        arg_dict = _prepare_arg_dict()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            **arg_dict
        )
    elif AutoConfig.from_pretrained(model_dir).model_type == "gemma3":
        processor = AutoProcessor.from_pretrained(model_dir)
        arg_dict = _prepare_arg_dict()
        model =  Gemma3ForConditionalGeneration.from_pretrained(**arg_dict)
    else:
        processor = AutoProcessor.from_pretrained(model_dir)
        arg_dict = _prepare_arg_dict()
        model =  AutoModelForImageTextToText.from_pretrained(**arg_dict)

    return processor, model


class Trainer:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.processer = None
        self.dataset = None
        self._sparse_constraints = {}
        self._device_sparse_constraints = {}
        self._sparse_optimizer_params = {}
        self._model_params = {}

    def _prepare_sparse_constraints(self):
        """Keep only selected coordinates for PGD bounds and gradient masking.

        The boundary builders return dense per-tensor bounds for convenient
        analysis, but only ``train_indices_flat`` may move during Stage 1.
        Keeping dense bounds on the optimizer path caused multi-GB CPU to GPU
        transfers at every optimizer step.
        """
        if not getattr(self, "finetune_box", None) or not getattr(self, "train_indices_flat", None):
            return False

        sparse = {}
        for name, (lower, upper) in self.finetune_box.items():
            indices = self.train_indices_flat.get(name)
            if indices is None or indices.numel() == 0:
                continue
            indices = indices.detach().view(-1).to(dtype=torch.long, device="cpu")
            lower_selected = lower.detach().view(-1).index_select(0, indices).contiguous()
            upper_selected = upper.detach().view(-1).index_select(0, indices).contiguous()
            sparse[name] = (indices, lower_selected, upper_selected)

        self._sparse_constraints = sparse
        # The dense CPU tensors are no longer needed after compression.
        self.finetune_box = None
        return bool(sparse)

    def _install_sparse_constraints(self):
        """Move sparse constraints to device and create optimizer proxies."""
        if not self._sparse_constraints:
            return False
        self._model_params = dict(self.model.named_parameters())
        for name, (indices, lower, upper) in self._sparse_constraints.items():
            param = self._model_params[name]
            param.requires_grad_(True)
            device_indices = indices.to(param.device, non_blocking=True)
            device_lower = lower.to(param.device, non_blocking=True)
            device_upper = upper.to(param.device, non_blocking=True)
            self._device_sparse_constraints[name] = (device_indices, device_lower, device_upper)
            selected_values = param.detach().view(-1).index_select(0, device_indices).clone()
            self._sparse_optimizer_params[name] = nn.Parameter(selected_values)
        return True

    def _gather_sparse_gradients(self):
        """Gather selected gradients once per optimizer step, then free dense grads."""
        for name, proxy in self._sparse_optimizer_params.items():
            model_param = self._model_params[name]
            if model_param.grad is None:
                proxy.grad = None
                continue
            indices = self._device_sparse_constraints[name][0]
            proxy.grad = model_param.grad.detach().view(-1).index_select(0, indices)
            model_param.grad = None

    @torch.no_grad()
    def _project_and_scatter_sparse_parameters(self):
        """Clamp selected proxy values and write them back to model weights."""
        for name, proxy in self._sparse_optimizer_params.items():
            indices, lower, upper = self._device_sparse_constraints[name]
            proxy.clamp_(min=lower, max=upper)
            self._model_params[name].data.view(-1).index_copy_(0, indices, proxy.data)

    def load_model(self):
        requested_quantize_method = self.args.quantize_method
        if (
            self.args.train_with_pgd == True
            and not is_gguf_method(requested_quantize_method)
            and not is_awq_method(requested_quantize_method)
            and not is_gptq_method(requested_quantize_method)
        ):
            _, self.model_quant = load_model(self.args.pretrain_name, self.args)
            self.model_quant.eval()

            _tmp = self.args.quantize_method
            self.args.quantize_method = None
            self.processer, self.model = load_model(self.args.pretrain_name, self.args)
            self.args.quantize_method = _tmp
            self.model.train()
        else:
            self.args.quantize_method = None
            self.processer, self.model = load_model(self.args.pretrain_name, self.args)
            self.args.quantize_method = requested_quantize_method
            self.model.train()
        
    def step(self, batch):

        # Move batch tensors (text + vision inputs) to the same device as the model.
        input_ids = batch['input_ids'].to(self.model.device, non_blocking=True)
        pixel_values = batch['pixel_values'].to(self.model.device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(self.model.device, non_blocking=True)
        labels = batch['labels'].to(self.model.device, non_blocking=True)

        # Shift tokens/labels for next-token prediction (teacher forcing).
        shift_input_ids = input_ids[..., 1:] 
        shift_labels = labels[..., 1:]

        # Forward pass with model-specific extra inputs.
        if self.model.config.model_type == "qwen3_vl":
            image_grid_thw = batch['image_grid_thw'].to(self.model.device, non_blocking=True)
            qwen_inputs = {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
                "attention_mask": attention_mask,
                "image_grid_thw": image_grid_thw,
            }
            if 'mm_token_type_ids' in batch:
                qwen_inputs['mm_token_type_ids'] = batch['mm_token_type_ids'].to(
                    self.model.device, non_blocking=True
                )
            if (
                getattr(self.args, "supervised_logits_only", False)
                and type(self.model) is Qwen3VLForConditionalGeneration
            ):
                return qwen3vl_supervised_loss(self.model, qwen_inputs, labels, attention_mask)
            outputs = self.model(**qwen_inputs, use_cache=False)
        elif self.model.config.model_type == "gemma3":
            token_type_ids = batch['token_type_ids'].to(self.model.device)
            outputs = self.model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            outputs = self.model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        # Use logits for next-token prediction (drop the last position to align with shifted labels).
        logits = outputs.logits[..., :-1, :]
        # Compute token-level cross-entropy, masking out padding/ignored positions via attention_mask.
        loss = token_weighted_loss("ce", logits, shift_labels, attention_mask[..., 1:])
        
        return loss
    
    def do_eval(self,val_dataloader, max_batches=None):
        self.model.eval()  # Run evaluation loop and report average validation loss.
        val_loss = torch.zeros((), device=self.model.device, dtype=torch.float64)
        num_val_batches = 0
        with torch.no_grad():  # Disable gradients for faster/cheaper evaluation.
            for val_batch in tqdm(val_dataloader, desc=f"Validation at epoch 0", dynamic_ncols=True): 
                val_loss_batch = self.step(val_batch)  
                val_loss.add_(val_loss_batch.detach().to(torch.float64))  
                num_val_batches += 1
                if max_batches is not None and num_val_batches >= max_batches:
                    break

        avg_val_loss = val_loss.item() / num_val_batches  
        self.model.train()  # Restore training mode after evaluation.
        return avg_val_loss

    def print_gradients(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"{name} - Grad: {param.grad.abs().sum().item()}")
    
    def run(self):
        cpu_threads = getattr(self.args, "cpu_threads", None)
        if cpu_threads:
            torch.set_num_threads(int(cpu_threads))
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass

        # 1) Initialization: load model and freeze all parameters by default.
        self.load_model()
        for p in self.model.parameters():
            p.requires_grad_(False)
        
        optimizer_class = AdamW8bit # Choose optimizer implementation (8-bit AdamW by default).
        model_dir = self.args.ori_model
        if AutoConfig.from_pretrained(model_dir).model_type == "qwen3_vl":
            arg_dict = {
                "pretrained_model_name_or_path": model_dir,
                "device_map": "auto",
                "trust_remote_code": True,
                "dtype": torch.float32,
            }
            ori_model = Qwen3VLForConditionalGeneration.from_pretrained(
                **arg_dict
            )
        elif AutoConfig.from_pretrained(model_dir).model_type == "gemma3":
            arg_dict = {"pretrained_model_name_or_path": model_dir, "device_map": "auto", "trust_remote_code": True}
            ori_model =  Gemma3ForConditionalGeneration.from_pretrained(**arg_dict)
        else:
            arg_dict = {"pretrained_model_name_or_path": model_dir, "device_map": "auto", "trust_remote_code": True}
            ori_model =  AutoModelForImageTextToText.from_pretrained(**arg_dict)

        
        # Optional: prepare PGD constraints / quantization-aware target layers.
        if self.args.train_with_pgd:
            if is_awq_method(self.args.quantize_method):
                (
                    self.train_indices_flat,
                    self.finetune_box,
                    self.awq_boundary_context,
                ) = build_awq_finetune_boxes(
                    model=self.model,
                    original_model=ori_model,
                    checkpoint=self.args.awq_boundary_checkpoint,
                    selection_fraction=self.args.awq_selection_fraction,
                    candidate_relative_limit=self.args.awq_candidate_relative_limit,
                    candidate_absolute_limit=self.args.awq_candidate_absolute_limit,
                    boundary_penetration=self.args.awq_penetration,
                    row_chunk_size=self.args.awq_row_chunk_size,
                    require_reference_code_match=self.args.awq_require_reference_code_match,
                    reference_current_model=self.args.awq_reference_current_model,
                )
                self.awq_activation_handles, activation_metadata = install_awq_activation_proxy(
                    self.model, self.args.awq_boundary_checkpoint
                )
                self.awq_boundary_context['activation_proxy'] = activation_metadata
                print(f"AWQ activation proxy: {activation_metadata}")
                if getattr(self.args, "sparse_optimizer_proxy", False):
                    self._prepare_sparse_constraints()
            elif is_gptq_method(self.args.quantize_method):
                (
                    self.train_indices_flat,
                    self.finetune_box,
                    self.gptq_boundary_context,
                ) = build_gptq_finetune_boxes(
                    model=self.model,
                    original_model=ori_model,
                    checkpoint=self.args.gptq_boundary_checkpoint,
                    selection_fraction=self.args.gptq_selection_fraction,
                    candidate_relative_limit=self.args.gptq_candidate_relative_limit,
                    candidate_absolute_limit=self.args.gptq_candidate_absolute_limit,
                    boundary_penetration=self.args.gptq_penetration,
                    require_reference_code_match=self.args.gptq_require_reference_code_match,
                )
                if getattr(self.args, "sparse_optimizer_proxy", False):
                    self._prepare_sparse_constraints()
            elif is_gguf_method(self.args.quantize_method):
                box_builder = (
                    build_q4_k_finetune_boxes
                    if self.args.quantize_method.lower() == "q4_k"
                    else build_k_finetune_boxes
                )
                self.train_indices_flat, self.finetune_box, self.gguf_metadata_context = box_builder(
                    model=self.model,
                    original_model=ori_model,
                    method=self.args.quantize_method,
                    selection_fraction=self.args.gguf_selection_fraction,
                    candidate_relative_limit=self.args.gguf_candidate_relative_limit,
                    candidate_absolute_limit=self.args.gguf_candidate_absolute_limit,
                    boundary_penetration=self.args.gguf_penetration,
                    scope=self.args.gguf_target_scope,
                    chunk_blocks=self.args.gguf_chunk_blocks,
                    metadata_mode=self.args.gguf_metadata_mode,
                    **({"input_precision": self.args.gguf_input_precision}
                       if box_builder is build_k_finetune_boxes else {}),
                )
                if getattr(self.args, "sparse_optimizer_proxy", False):
                    self._prepare_sparse_constraints()
            else:
                self.optimize_target_dict = get_quantize_target_layers(model_full=self.model, model_quant=self.model_quant)
            if self.args.quantize_method in ("fp4","nf4"):
                self.find_param_4bit(orig_model=ori_model,ratio=self.args.change_ratio, method = self.args.quantize_method)
            elif self.args.quantize_method == "int8":
                self.find_param_8bit(orig_model=ori_model,ratio=self.args.change_ratio)
        del ori_model
        # Release the auxiliary quantized model reference if it exists (save memory).
        if hasattr(self, 'model_quant'): del self.model_quant
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # The optional sparse proxy is faster but changes AdamW8bit state-block
        # grouping. Dense mode is the default because it exactly matches the
        # established FP4/NF4/INT8 PGD training semantics.
        sparse_constraints_active = self._install_sparse_constraints()
        for name, p in self.model.named_parameters():
            if sparse_constraints_active:
                p.requires_grad_(name in self._device_sparse_constraints)
            elif self.finetune_box is not None and name in self.finetune_box:
                p.requires_grad_(True)

        if (
            not sparse_constraints_active
            and self.finetune_box is not None
            and getattr(self.args, "cache_pgd_bounds_on_device", False)
        ):
            params = dict(self.model.named_parameters())
            constraint_bytes = 0
            for name, (lower, upper) in self.finetune_box.items():
                param = params[name]
                self.finetune_box[name] = [
                    lower.to(param.device, non_blocking=True),
                    upper.to(param.device, non_blocking=True),
                ]
                constraint_bytes += sum(
                    bound.numel() * bound.element_size()
                    for bound in self.finetune_box[name]
                )
            print(
                "PGD mode: dense AdamW8bit with device-cached bounds "
                f"({constraint_bytes / 2**30:.2f} GiB)"
            )
        elif sparse_constraints_active:
            selected = sum(proxy.numel() for proxy in self._sparse_optimizer_params.values())
            print(f"PGD mode: sparse optimizer proxy ({selected:,} coordinates)")

        # 2) Build datasets/dataloaders and training utilities (optimizer + scheduler).
        self.dataset = VLLMDataset(self.args.datasets, self.processer)
        self.val_dataset = VLLMDataset(self.args.val_datasets, self.processer)
        collator = VLLMDataCollator(self.processer)
        self._model_params = dict(self.model.named_parameters())

        total_samples = len(self.dataset)
        num_workers = int(getattr(self.args, "dataloader_num_workers", 0))
        loader_kwargs = {
            "batch_size": self.args.batch_size,
            "shuffle": True,
            "collate_fn": collator,
            "num_workers": num_workers,
            "pin_memory": bool(getattr(self.args, "dataloader_pin_memory", False)),
        }
        if num_workers > 0:
            loader_kwargs.update(
                persistent_workers=True,
                prefetch_factor=int(getattr(self.args, "dataloader_prefetch_factor", 2)),
                worker_init_fn=_init_dataloader_worker,
            )
        train_dataloader = DataLoader(self.dataset, **loader_kwargs)
        val_dataloader = DataLoader(self.val_dataset, **loader_kwargs)

        total_micro_steps = len(train_dataloader) * self.args.num_train_epochs
        total_steps = math.ceil(total_micro_steps / self.args.grad_acc_steps)
        optimizer_params = (
            list(self._sparse_optimizer_params.values())
            if sparse_constraints_active
            else [p for p in self.model.parameters() if p.requires_grad]
        )
        optimizer = optimizer_class(
            params=optimizer_params,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=total_steps,
        )

        # Quick sanity-check evaluation before training starts.
        avg_val_loss = self.do_eval(val_dataloader, self.args.initial_validation_batches)
        print(f"End of Step 0 - Validation Loss: {avg_val_loss:.6f}")
        initial_validation_loss = avg_val_loss
        last_validation_loss = avg_val_loss
        n = sum(p.data.norm(2).item()**2 for p in self.model.parameters() if p.requires_grad) ** 0.5
        print(f"End of Step 0 - L2 norm: {n:.6f}")

        # 3) Main training loop (supports gradient accumulation + optional PGD projection).
        global_step = 0
        set_seed(self.args.seed)
        timer = Timer(total_steps)
        timer.start()
        self.model.train()
        configure_gradient_checkpointing(self.model, getattr(self.args, "gradient_checkpointing", False))
        optimizer.zero_grad(set_to_none=True)
        accumulated_steps = 0
        stop_training = False
        early_stopped = False
        completed_epochs = 0

        for idx in range(self.args.num_train_epochs):
            completed_epochs = idx + 1
            step_loss = torch.zeros((), device=self.model.device, dtype=torch.float64)
            
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {idx+1}", dynamic_ncols=True)):
                
                # A) Forward + loss. FP32 configurations must not silently enter
                # BF16 autocast; lower-precision modes opt into autocast explicitly.
                dtype_name = str(self.args.training_dtype).lower()
                if dtype_name in {"fp32", "float32"}:
                    autocast_context = nullcontext()
                elif dtype_name in {"bf16", "bfloat16"}:
                    autocast_context = torch.amp.autocast("cuda", dtype=torch.bfloat16)
                elif dtype_name in {"fp16", "float16"}:
                    autocast_context = torch.amp.autocast("cuda", dtype=torch.float16)
                else:
                    raise ValueError(f"Unsupported training_dtype: {self.args.training_dtype}")
                with autocast_context:
                    loss_ce = self.step(batch)
                    loss_ce_scaled = loss_ce / self.args.grad_acc_steps
                
                # Backprop on the scaled loss.
                loss_ce_scaled.backward()
                step_loss.add_(loss_ce.detach().to(torch.float64))
                accumulated_steps += 1

                # B) Optimizer update and C) optional PGD projection/constraint enforcement.
                final_micro_step = (
                    idx == self.args.num_train_epochs - 1
                    and step == len(train_dataloader) - 1
                )
                if accumulated_steps >= self.args.grad_acc_steps or final_micro_step:
                    if sparse_constraints_active:
                        self._gather_sparse_gradients()
                        if accumulated_steps < self.args.grad_acc_steps:
                            scale = self.args.grad_acc_steps / accumulated_steps
                            for proxy in optimizer_params:
                                if proxy.grad is not None:
                                    proxy.grad.mul_(scale)
                        torch.nn.utils.clip_grad_norm_(optimizer_params, self.args.max_grad_norm)
                    else:
                        if accumulated_steps < self.args.grad_acc_steps:
                            scale = self.args.grad_acc_steps / accumulated_steps
                            for param in optimizer_params:
                                if param.grad is not None:
                                    param.grad.mul_(scale)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    
                    optimizer.step()
                    if sparse_constraints_active:
                        self._project_and_scatter_sparse_parameters()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1
                    accumulated_steps = 0
                    
                    current_boundary_loss_val = 0.0 # placeholder for logging/debugging
                    
                    # PGD stage: project finetuned parameters back into the allowed box constraints.
                    if self.args.train_with_pgd:
                        with torch.no_grad():

                            gpu_stats_list = []
                            current_lr = optimizer.param_groups[0]['lr']
                            lr_ratio = current_lr / (self.args.learning_rate + 1e-8)

                            if not sparse_constraints_active:
                                for name, p in self.model.named_parameters():
                                    if name in self.finetune_box:
                                        L = self.finetune_box[name][0].to(p.device, non_blocking=True)
                                        U = self.finetune_box[name][1].to(p.device, non_blocking=True)
                                        p.data.clamp_(L, U)
                                        del L, U

                            if (
                                self.args.quantize_method.lower() == "q4_k"
                                and self.args.gguf_metadata_mode == "code_only"
                                and global_step % self.args.gguf_metadata_check_steps == 0
                            ):
                                metadata_stats = enforce_q4_k_metadata_(
                                    self.model,
                                    self.gguf_metadata_context,
                                    chunk_blocks=self.args.gguf_chunk_blocks,
                                )
                                print(f"Q4_K metadata projection: {metadata_stats}")

                    # Periodic training logs.
                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        print(
                            f"Step: {global_step}/{total_steps}, "
                            f"Loss: {step_loss.item():.4f}, "
                            f"lr: {lr_ratio:.6f}%, "
                            f"{timer}"
                        )
                        step_loss.zero_()
                    # Periodic validation.
                    if self.args.validation_steps > 0 and global_step % self.args.validation_steps == 0:
                        avg_val_loss = self.do_eval(val_dataloader)
                        last_validation_loss = avg_val_loss
                        print(f"Validation Loss: {avg_val_loss:.6f}")
                        threshold = getattr(self.args, "early_stop_validation_loss", None)
                        if threshold is not None and avg_val_loss <= threshold:
                            print(
                                f"Early stopping at step {global_step}: validation loss "
                                f"{avg_val_loss:.6f} <= {threshold:.6f}"
                            )
                            stop_training = True
                            early_stopped = True
                    timer.end()
                    timer.start()
                    if self.args.max_train_steps is not None and global_step >= self.args.max_train_steps:
                        break
                    if stop_training:
                        break
            if self.args.max_train_steps is not None and global_step >= self.args.max_train_steps:
                break
            if stop_training:
                break

        if early_stopped:
            stop_reason = "early_stop_validation_loss"
        elif self.args.max_train_steps is not None and global_step >= self.args.max_train_steps:
            stop_reason = "max_train_steps"
        else:
            stop_reason = "num_train_epochs"
        self.training_summary = {
            "quantize_method": self.args.quantize_method,
            "training_dtype": self.args.training_dtype,
            "loss_normalization": "supervised_tokens",
            "supervised_logits_only": bool(getattr(self.args, "supervised_logits_only", False)),
            "gradient_checkpointing": bool(self.args.gradient_checkpointing),
            "checkpoint_use_reentrant": False if self.args.gradient_checkpointing else None,
            "num_train_epochs_limit": self.args.num_train_epochs,
            "completed_epochs": completed_epochs,
            "optimizer_steps": global_step,
            "initial_validation_loss": initial_validation_loss,
            "last_validation_loss": last_validation_loss,
            "early_stop_validation_loss": getattr(
                self.args, "early_stop_validation_loss", None
            ),
            "early_stopped": early_stopped,
            "stop_reason": stop_reason,
        }
    
        
    def find_param_4bit(self,orig_model, ratio=0.05, method: str = "nf4", blocksize: int = 64):
        """
        Select a subset of 4-bit-quantized weights that are likely to cross their quantization
        bin boundaries under a relative perturbation, then build per-parameter box constraints.

        High-level idea:
        - Quantize/dequantize weights with bitsandbytes (4-bit).
        - For each weight, test whether +/- (ratio * |w|) would cross the midpoint boundaries
          to neighboring codebook levels.
        - Keep selected weights "trainable" (others are snapped to dequantized values).
        - Build a finetune box that constrains updates to at most one neighboring code level.
        """
        import torch
        import bitsandbytes.functional as bnbF
        from tqdm import tqdm

        # Collect parameters and initialize bookkeeping for selected indices and constraint boxes.
        param_dict = dict(self.model.named_parameters())
        orig_params = dict(orig_model.named_parameters())
        train_indices_flat = {}
        finetune_box = {}
        # Basic counters for reporting.
        total_selected = 0
        total_params = 0
        total_blocks = 0
        total_both = 0

        print(f"[Rel-cross-boundary selection] ratio={ratio}, method={method}, blocksize={blocksize}")
        print("Criterion (fixed for negatives): Delta=ratio*|w|, (w+Delta)>u OR (w-Delta)<l")
        print("Init: unselected->w_deq, selected->w_orig, argmax locked->w_orig")

        # Only consider matrix-like weights (e.g., linear/conv weights) for blockwise quantization.
        names = [n for n in param_dict.keys() if n.endswith(".weight") and param_dict[n].data.ndim >= 2]

        for name in tqdm(names, desc="Quantize & Select"):
            # Skip layers that are not part of the quantization target set.
            if name not in self.optimize_target_dict or name not in orig_params:
                continue
            param = param_dict[name]
            device = param.device
            dtype = param.dtype

            # Work on a copy of the original weights.
            w_orig = param.data.clone()

            w_baseline = orig_params[name].data.to(device)
            if w_orig.shape != w_baseline.shape:
                continue
            changed_mask = (w_orig - w_baseline).abs() > 0
            # Require block-aligned shapes for 4-bit block quantization.
            if w_orig.numel() == 0 or (w_orig.numel() % blocksize != 0):
                continue

            with torch.no_grad():
                # 1) Quantize to 4-bit and dequantize back to get the effective quantized centers.
                w_half = w_orig.contiguous().half()
                w_4bit, quant_state = bnbF.quantize_4bit(
                    A=w_half,
                    blocksize=blocksize,
                    compress_statistics=(method == "nf4"),
                    quant_type=method,
                )
                w_deq = bnbF.dequantize_4bit(w_4bit, quant_state).to(dtype)

                # Recover per-element code indices to reason about neighboring codebook levels.
                code_idx, _ = compute_code_from_4bit(w_4bit, quant_state)
                code_idx_blk = code_idx.view(-1, blocksize)

                # 2) Estimate per-block scale based on dequantized values (and make it robust).
                w_deq_f32_blk = w_deq.to(torch.float32).view(-1, blocksize)
                absmax_deq, argmax_idx = w_deq_f32_blk.abs().max(dim=1)
                absmax_deq = absmax_deq.clamp_min(1e-12)

                w_orig_f32_blk = w_orig.to(torch.float32).view(-1, blocksize)
                safe_absmax_blk = torch.maximum(absmax_deq.view(-1, 1), w_orig_f32_blk.abs())

                total_blocks += int(absmax_deq.numel())
                total_params += int(w_orig.numel())

                # 3) Build a rank ordering of the codebook so we can find "up/down" neighbors.
                codebook = quant_state.code.float().to(device)  # (16,)
                sorted_vals, sorted_ids = torch.sort(codebook)
                rank_of_code = torch.empty_like(sorted_ids)
                rank_of_code[sorted_ids] = torch.arange(sorted_ids.numel(), device=device)

                r = rank_of_code[code_idx_blk]  # per-element rank in [0, 15]

                has_down = r > 0
                has_up   = r < 15

                cur_norm = codebook[code_idx_blk]
                r_down = torch.clamp(r - 1, 0, 15)
                r_up   = torch.clamp(r + 1, 0, 15)
                down_norm = sorted_vals[r_down]
                up_norm   = sorted_vals[r_up]

                # 4) Convert normalized codebook values to physical centers using the block scales.
                scale_blk = safe_absmax_blk  # (nblk,B)
                cur_center  = cur_norm.to(torch.float32)  * scale_blk
                down_center = down_norm.to(torch.float32) * scale_blk
                up_center   = up_norm.to(torch.float32)   * scale_blk

                # 5) Compute midpoint boundaries between current and neighbor centers.
                lower_b = 0.5 * (cur_center + down_center)  # l_k
                upper_b = 0.5 * (cur_center + up_center)    # u_k

                # 6) Selection criterion: whether a relative perturbation can cross a boundary.
                w = w_orig_f32_blk
                delta = w.abs() * float(ratio)  # Delta = ratio * |w|
                w_hi = w + delta
                w_lo = w - delta

                cross_up = torch.where(has_up,   w_hi > upper_b, torch.zeros_like(w, dtype=torch.bool))
                cross_dn = torch.where(has_down, w_lo < lower_b, torch.zeros_like(w, dtype=torch.bool))

                mask_select_blk = cross_up | cross_dn  # (nblk,B)

                # 7) Lock the per-block argmax element to keep quantization scale stable.
                arg_mask_blk = torch.zeros_like(mask_select_blk, dtype=torch.bool, device=device)
                arg_mask_blk[torch.arange(arg_mask_blk.size(0), device=device), argmax_idx] = True
                mask_select_blk &= ~arg_mask_blk

                mask_select = mask_select_blk.reshape_as(w_orig)
                mask_select &= ~changed_mask

                # Store flattened indices of selected weights for downstream use.
                flat_idx = mask_select.view(-1).nonzero(as_tuple=False).squeeze(1).detach().cpu()
                train_indices_flat[name] = flat_idx
                total_selected += int(flat_idx.numel())

                # Track how often both "up" and "down" boundary crossings are possible.
                both_mask_blk = cross_up & cross_dn & (~arg_mask_blk)
                total_both += int(both_mask_blk.sum().item())

                # 8) Initialize parameters: unselected -> dequantized, selected/locked -> original.
                arg_mask = arg_mask_blk.reshape_as(w_orig)

                w_new = w_deq.clone()
                w_new[mask_select] = w_orig[mask_select]
                w_new[arg_mask] = w_orig[arg_mask]

                w_new[changed_mask] = w_orig[changed_mask] 

                # Clamp weights to the safe per-block dynamic range.
                lo = (-safe_absmax_blk).reshape_as(w_orig).to(dtype)
                hi = ( safe_absmax_blk).reshape_as(w_orig).to(dtype)
                w_new = torch.clamp(w_new, min=lo, max=hi)

                param.data.copy_(w_new)

                # 9) Build per-weight box constraints aligned with the allowed neighbor transitions.
                wmin = w_deq.clone()
                wmax = w_deq.clone()

                cur_phys  = cur_center.reshape_as(w_orig).to(dtype)
                down_phys = down_center.reshape_as(w_orig).to(dtype)
                up_phys   = up_center.reshape_as(w_orig).to(dtype)

                cross_up_full = cross_up.reshape_as(w_orig)
                cross_dn_full = cross_dn.reshape_as(w_orig)

                upper_mask = mask_select & cross_up_full
                lower_mask = mask_select & cross_dn_full
                both_mask  = upper_mask & lower_mask

                # Allow movement toward the "up" neighbor where applicable.
                wmin[upper_mask] = torch.minimum(cur_phys[upper_mask], up_phys[upper_mask])
                wmax[upper_mask] = torch.maximum(cur_phys[upper_mask], up_phys[upper_mask])

                # Allow movement toward the "down" neighbor where applicable.
                wmin[lower_mask] = torch.minimum(down_phys[lower_mask], cur_phys[lower_mask])
                wmax[lower_mask] = torch.maximum(down_phys[lower_mask], cur_phys[lower_mask])

                # If both directions are possible, allow spanning both neighbors.
                wmin[both_mask] = torch.minimum(down_phys[both_mask], up_phys[both_mask])
                wmax[both_mask] = torch.maximum(down_phys[both_mask], up_phys[both_mask])

                # Keep locked elements fixed.
                wmin[arg_mask] = w_orig[arg_mask]
                wmax[arg_mask] = w_orig[arg_mask]

                wmin[changed_mask] = w_orig[changed_mask]
                wmax[changed_mask] = w_orig[changed_mask]

                # Clamp boxes to the safe dynamic range and ensure current value lies inside.
                wmin = torch.clamp(wmin, min=lo, max=hi)
                wmax = torch.clamp(wmax, min=lo, max=hi)

                # Store CPU copies of the constraint box for later PGD projection.
                val_curr = param.data
                wmin = torch.minimum(wmin, val_curr)
                wmax = torch.maximum(wmax, val_curr)

                finetune_box[name] = [wmin.detach().cpu(), wmax.detach().cpu()]

                # cleanup
                del w_half, w_4bit, quant_state, w_deq, code_idx, code_idx_blk
                del w_deq_f32_blk, absmax_deq, argmax_idx, w_orig_f32_blk, safe_absmax_blk
                del codebook, sorted_vals, sorted_ids, rank_of_code, r, r_down, r_up
                del cur_norm, down_norm, up_norm, scale_blk, cur_center, down_center, up_center
                del lower_b, upper_b, w, delta, w_hi, w_lo, cross_up, cross_dn, mask_select_blk
                del arg_mask_blk, mask_select, arg_mask, w_new, lo, hi, wmin, wmax
                del cur_phys, down_phys, up_phys, cross_up_full, cross_dn_full
                del upper_mask, lower_mask, both_mask, val_curr
                del w_baseline, changed_mask

        # Expose selection results and constraint boxes to the training loop.
        self.train_indices_flat = train_indices_flat
        self.finetune_box = finetune_box

        print("-" * 40)
        print("Done. Rel-cross-boundary selection finished.")
        print(f"Total params scanned : {total_params:,}")
        print(f"Total blocks         : {total_blocks:,}")
        print(f"Selected params      : {total_selected:,}")
        if total_params > 0:
            print(f"Selected ratio       : {100.0 * total_selected / total_params:.4f}%")
        print(f"Both-direction selectable (cross up & down): {total_both:,}")
        if total_selected > 0:
            print(f"Both-direction / selected ratio: {100.0 * total_both / total_selected:.2f}%")
        print("-" * 40)

    def find_param_8bit(
        self,
        orig_model,
        ratio: float = 0.05,
        skip_if_not_rowwise: bool = True 
    ):
        """
        Targeted INT8 (bitsandbytes LLM.int8) selection and calibration.
        Only layers listed in self.optimize_target_dict are processed.
        """
        import torch
        from tqdm import tqdm
        import bitsandbytes.functional as bnbF

        print(f"(Targeted INT8 calculation): ratio={ratio}")

        # Collect all parameters from the model.
        param_dict = dict(self.model.named_parameters())
        orig_params = dict(orig_model.named_parameters())
        # Will store selected indices and per-parameter constraint boxes.
        train_indices_flat = {}
        finetune_box = {}
        # Counters for reporting/debugging.
        total_selected = 0
        total_params = 0
        skipped = 0
        ignored = 0 

        # Helper: convert tensors to a row-wise (N, D) view for int8 row/vector-wise quantization.
        def to_rowwise_view(x: torch.Tensor):
            if x.dim() == 2:
                orig_shape = x.shape
                return x.reshape(orig_shape[0], orig_shape[1]), (lambda t: t.reshape(orig_shape))
            if x.dim() == 4: 
                out = x.shape[0]
                orig_shape = x.shape
                return x.reshape(out, -1), (lambda t: t.reshape(orig_shape))
            if x.dim() == 3:
                out = x.shape[0]
                orig_shape = x.shape
                return x.reshape(out, -1), (lambda t: t.reshape(orig_shape))
            return None, None

        with torch.no_grad():
            for name, w_param in tqdm(param_dict.items(), desc="Analyzing Targets"):
                
                # Only operate on whitelisted (target) layers; keep placeholders for others.
                if name not in self.optimize_target_dict or name not in orig_params:
                    ignored += 1
                    finetune_box[name] = [w_param.data.cpu(), w_param.data.cpu()]
                    train_indices_flat[name] = torch.empty((0,), dtype=torch.long)
                    continue

                # Skip non-matrix parameters (e.g., biases, layernorm weights).
                if w_param.dim() < 2:
                    continue

                w = w_param.data
                device = w.device
                dtype = w.dtype

                w_baseline = orig_params[name].data.to(device)
                if w.shape != w_baseline.shape:
                    continue
                changed_mask_full = (w - w_baseline).abs() > 0

                total_params += w.numel()

                # 1) Convert to a row-wise 2D view (required for int8 vectorwise quantization).
                W2d, restore = to_rowwise_view(w)

                changed_mask_2d, _ = to_rowwise_view(changed_mask_full)
                if W2d is None:
                    skipped += 1
                    finetune_box[name] = [w.cpu(), w.cpu()]
                    train_indices_flat[name] = torch.empty((0,), dtype=torch.long)
                    continue

                # 2) Compute int8 quantization outputs (rowwise stats + quantized values) using bnb logic.
                w_half = W2d.contiguous().half().to(device)
                out_row, row_stats, _ = bnbF.int8_vectorwise_quant(w_half)
                # Convert quantized output to signed range and derive per-row scale.
                q0 = out_row.to(torch.int16).clamp(-127, 127)
                scale = (row_stats.to(dtype) / 127.0).unsqueeze(1).expand_as(W2d)
                
                # 3) Compute per-element "bin" boundaries around the current quantized level.
                q_float = q0.to(dtype)
                bound_min = (q_float - 0.5) * scale
                bound_max = (q_float + 0.5) * scale
                # Clamp boundaries using row-wise absmax to stay within representable range.
                absmax_expand = row_stats.to(dtype).unsqueeze(1).expand_as(W2d)
                bound_min.clamp_(min=-absmax_expand, max=absmax_expand)
                bound_max.clamp_(min=-absmax_expand, max=absmax_expand)

                # 4) Lock the absmax/tied elements per row to keep scaling stable.
                absmax_vals, _ = W2d.abs().max(dim=1, keepdim=True)
                tie_mask_2d = (W2d.abs() >= (absmax_vals - 1e-6))

                # 5) Selection criterion: whether a relative perturbation can cross the current bin.
                delta_probe = W2d.abs() * ratio
                exceed_upper = (W2d + delta_probe) > bound_max
                exceed_lower = (W2d - delta_probe) < bound_min

                mask_select_2d = (exceed_upper | exceed_lower) & (~tie_mask_2d)

                mask_select_2d &= ~changed_mask_2d

                # 6) Compute quantized centers for current/neighbor levels (for box construction).
                curr_center = q_float * scale
                q_up = (q0 + 1).clamp(max=127).to(dtype)
                up_center = q_up * scale
                q_down = (q0 - 1).clamp(min=-127).to(dtype)
                down_center = q_down * scale

                # 7) Snap non-selected, non-locked weights to the quantized grid (in-place calibration).
                mask_inplace_snap = (~tie_mask_2d) & (~mask_select_2d)

                mask_inplace_snap &= ~changed_mask_2d

                if mask_inplace_snap.any():
                    W2d[mask_inplace_snap] = curr_center[mask_inplace_snap]

                # 8) Build per-weight finetuning intervals aligned with allowed neighbor transitions.
                wmin2d = W2d.clone()
                wmax2d = W2d.clone()

                upper_mask = mask_select_2d & exceed_upper
                lower_mask = mask_select_2d & exceed_lower
                both_mask = upper_mask & lower_mask
                upper_mask = upper_mask & (~both_mask)
                lower_mask = lower_mask & (~both_mask)

                if upper_mask.any():
                    wmin2d[upper_mask] = curr_center[upper_mask]
                    wmax2d[upper_mask] = up_center[upper_mask]

                if lower_mask.any():
                    wmin2d[lower_mask] = down_center[lower_mask]
                    wmax2d[lower_mask] = curr_center[lower_mask]

                if both_mask.any():
                    wmin2d[both_mask] = down_center[both_mask]
                    wmax2d[both_mask] = up_center[both_mask]

                # 9) Ensure locked elements stay fixed and the current value is inside the box.
                wmin2d[tie_mask_2d] = W2d[tie_mask_2d]
                wmax2d[tie_mask_2d] = W2d[tie_mask_2d]

                wmin2d[changed_mask_2d] = W2d[changed_mask_2d]
                wmax2d[changed_mask_2d] = W2d[changed_mask_2d]

                wmin2d = torch.min(wmin2d, W2d)
                wmax2d = torch.max(wmax2d, W2d)

                # 10) Write back calibrated weights and save indices/boxes for later PGD projection.
                w_new = restore(W2d).to(dtype)
                w_param.data.copy_(w_new)

                mask_select = restore(mask_select_2d).bool()
                flat_idx = mask_select.reshape(-1).nonzero(as_tuple=False).squeeze(1).cpu()
                
                train_indices_flat[name] = flat_idx
                total_selected += flat_idx.numel()
                
                finetune_box[name] = [
                    restore(wmin2d).detach().cpu(), 
                    restore(wmax2d).detach().cpu()
                ]
                del w_baseline, changed_mask_full, changed_mask_2d
        # Expose results to the trainer.
        self.train_indices_flat = train_indices_flat
        self.finetune_box = finetune_box

        print("-" * 30)
        print("Selection complete (selected targets only)")
        print(f"Total params scanned: {total_params}")
        print(f"Ignored non-target layers: {ignored}")
        print(f"Selected finetune params: {total_selected}")
        print(f"Selected ratio: {total_selected / max(1,total_params) * 100:.4f}%")
        print("-" * 30)
