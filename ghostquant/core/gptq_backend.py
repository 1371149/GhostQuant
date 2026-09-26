"""Boundary-aware Stage-1 and Stage-2 support for calibrated GPTQ."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


GPTQ_METHODS = {"gptq", "gptq_w4a16"}


def is_gptq_method(method: str | None) -> bool:
    return bool(method) and method.lower() in GPTQ_METHODS


class _SafetensorIndex:
    def __init__(self, checkpoint: Path):
        index_path = checkpoint / "model.safetensors.index.json"
        if index_path.is_file():
            self.weight_map = json.loads(index_path.read_text())["weight_map"]
        else:
            model_path = checkpoint / "model.safetensors"
            if not model_path.is_file():
                raise FileNotFoundError(f"No safetensors checkpoint in {checkpoint}")
            with safe_open(model_path, framework="pt", device="cpu") as handle:
                self.weight_map = {key: model_path.name for key in handle.keys()}
        self.checkpoint = checkpoint

    def get_tensor(self, key: str) -> torch.Tensor:
        filename = self.weight_map.get(key)
        if filename is None:
            raise KeyError(key)
        with safe_open(
            self.checkpoint / filename, framework="pt", device="cpu"
        ) as handle:
            return handle.get_tensor(key)

    def get_optional_tensor(self, key: str) -> torch.Tensor | None:
        if key not in self.weight_map:
            return None
        return self.get_tensor(key)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _load_report(checkpoint: Path):
    path = checkpoint / "boundary_artifacts" / "boundary_report.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing GPTQ boundary report: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("packed_checkpoint_verification", {}).get(
        "all_layers_matched", False
    ):
        raise RuntimeError("GPTQ checkpoint has no successful packed-code verification")
    return report


def _load_layer_artifact(checkpoint: Path, layer_report: dict):
    path = checkpoint / "boundary_artifacts" / layer_report["artifact"]
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if "effective_weight" in keys:
            effective_weight = handle.get_tensor("effective_weight").float()
        else:
            effective_weight = None
        return {
            "effective_weight": effective_weight,
            "effective_offset": (
                handle.get_tensor("effective_offset").float()
                if "effective_offset" in keys
                else None
            ),
            "group_index_original": handle.get_tensor(
                "group_index_original"
            ).long(),
        }


def _expanded_qparam(
    value: torch.Tensor, group_index: torch.Tensor, rows: int
) -> torch.Tensor:
    value = value.float()
    if value.ndim == 0:
        value = value.reshape(1, 1)
    elif value.ndim == 1:
        value = value.unsqueeze(1)
    if value.shape[0] == 1 and rows > 1:
        value = value.expand(rows, -1)
    return value[:, group_index]


def _expanded_zero_point(
    tensors: _SafetensorIndex,
    layer_name: str,
    scale: torch.Tensor,
    group_index: torch.Tensor,
    rows: int,
) -> torch.Tensor:
    zero = tensors.get_optional_tensor(f"{layer_name}.weight_zero_point")
    if zero is None:
        zero = torch.zeros_like(scale)
    return _expanded_qparam(zero, group_index, rows)


def _original_group_absmax(values: torch.Tensor, group_size: int) -> torch.Tensor:
    rows, columns = values.shape
    groups = math.ceil(columns / group_size)
    padded = F.pad(values.abs(), (0, groups * group_size - columns), value=-1.0)
    maxima = padded.reshape(rows, groups, group_size).amax(dim=-1)
    return maxima.repeat_interleave(group_size, dim=-1)[..., :columns]


@torch.no_grad()
def build_gptq_finetune_boxes(
    model,
    original_model,
    *,
    checkpoint: str,
    candidate_relative_limit: float = 0.10,
    candidate_absolute_limit: float | None = None,
    boundary_penetration: float = 0.05,
    selection_fraction: float | None = None,
    require_reference_code_match: bool = True,
    **_kwargs,
):
    """Build local GPTQ PGD boxes in the original FP32 coordinates."""
    if candidate_relative_limit is None or candidate_relative_limit <= 0:
        raise ValueError("GPTQ selection requires a positive relative limit")
    if candidate_absolute_limit is not None and candidate_absolute_limit <= 0:
        raise ValueError("GPTQ absolute limit must be positive when enabled")
    if selection_fraction is not None and not 0 < selection_fraction <= 1:
        raise ValueError("GPTQ selection fraction must be in (0, 1]")
    if not 0 <= boundary_penetration <= 1:
        raise ValueError("GPTQ boundary penetration must be between 0 and 1")

    checkpoint_path = Path(checkpoint).resolve()
    report = _load_report(checkpoint_path)
    tensors = _SafetensorIndex(checkpoint_path)
    params = dict(model.named_parameters())
    originals = dict(original_model.named_parameters())
    boxes = {}
    indices = {}
    totals = {
        "parameters": 0,
        "eligible_parameters": 0,
        "selected_parameters": 0,
        "anchor_parameters": 0,
        "changed_parameters_excluded": 0,
    }
    by_scope = {"language": [0, 0], "vision": [0, 0]}
    reference_mismatches = []

    for layer_report in report["layers"]:
        layer_name = layer_report["layer"]
        name = f"{layer_name}.weight"
        if name not in params or name not in originals:
            raise KeyError(f"GPTQ target is absent from FP32 model: {name}")
        param = params[name]
        current = param.detach().float().cpu()
        original = originals[name].detach().float().cpu()
        if current.shape != original.shape or current.ndim != 2:
            raise ValueError(f"Unexpected GPTQ parameter shape for {name}")
        original_hash = _tensor_sha256(original)
        if original_hash != layer_report["original_weight_sha256"]:
            raise RuntimeError(f"GPTQ boundary/original model mismatch at {name}")

        artifact = _load_layer_artifact(checkpoint_path, layer_report)
        effective_original = artifact["effective_weight"]
        if effective_original is None:
            effective_original = original + artifact["effective_offset"]
        group_index = artifact["group_index_original"]
        rows, columns = current.shape
        scale = tensors.get_tensor(f"{layer_name}.weight_scale")
        zero = tensors.get_optional_tensor(f"{layer_name}.weight_zero_point")
        if zero is None:
            zero = torch.zeros_like(scale)
        expanded_scale = _expanded_qparam(scale, group_index, rows)
        expanded_zero = _expanded_qparam(zero, group_index, rows)
        q_min = int(layer_report["q_min"])
        q_max = int(layer_report["q_max"])
        effective_current = effective_original + (current - original)
        reference_codes = torch.round(
            (effective_original / expanded_scale + expanded_zero).clamp(q_min, q_max)
        ).to(torch.int8)
        if _tensor_sha256(reference_codes) != layer_report["code_sha256"]:
            reference_mismatches.append(layer_name)
            if require_reference_code_match:
                raise RuntimeError(f"GPTQ reference code mismatch at {layer_name}")
        current_codes = torch.round(
            (effective_current / expanded_scale + expanded_zero).clamp(q_min, q_max)
        ).to(torch.int8)

        group_size = int(layer_report["group_size"])
        expanded_absmax = _original_group_absmax(original, group_size)
        anchors = torch.isclose(
            original.abs(), expanded_absmax, rtol=1e-6, atol=1e-8
        )
        changed = current != original
        code_float = current_codes.float()
        lower_effective = (code_float - 0.5 - expanded_zero) * expanded_scale
        upper_effective = (code_float + 0.5 - expanded_zero) * expanded_scale
        lower_distance = effective_current - lower_effective
        upper_distance = upper_effective - effective_current
        lower_distance = torch.where(
            code_float > q_min,
            lower_distance,
            torch.full_like(current, torch.inf),
        )
        upper_distance = torch.where(
            code_float < q_max,
            upper_distance,
            torch.full_like(current, torch.inf),
        )
        budget = candidate_relative_limit * current.abs()
        if candidate_absolute_limit is not None:
            budget.clamp_(max=candidate_absolute_limit)
        valid = ~anchors & ~changed & (expanded_scale > 0)
        lower_ok = valid & torch.isfinite(lower_distance) & (lower_distance >= -1e-6)
        lower_ok &= lower_distance.clamp_min(0) <= budget
        upper_ok = valid & torch.isfinite(upper_distance) & (upper_distance >= -1e-6)
        upper_ok &= upper_distance.clamp_min(0) <= budget
        eligible = lower_ok | upper_ok
        eligible_count = int(eligible.sum())
        chosen = eligible.clone()
        if selection_fraction is not None:
            target_count = int(round(current.numel() * selection_fraction))
            if eligible_count < target_count:
                raise RuntimeError(
                    f"{name} has {eligible_count:,} GPTQ candidates, fewer than "
                    f"the requested {target_count:,}"
                )
            if eligible_count > target_count:
                nearest = torch.minimum(
                    lower_distance.clamp_min(0), upper_distance.clamp_min(0)
                )
                score = nearest / budget.clamp_min(1e-12)
                candidate_ids = torch.nonzero(eligible.view(-1)).view(-1)
                keep = candidate_ids[
                    torch.topk(
                        score.view(-1).index_select(0, candidate_ids),
                        target_count,
                        largest=False,
                        sorted=False,
                    ).indices
                ]
                chosen.zero_()
                chosen.view(-1)[keep] = True
                lower_ok &= chosen
                upper_ok &= chosen

        # GPTQ's later-column boundaries depend on the quantization residuals of
        # every earlier column. Replacing frozen weights by dequantized values
        # changes those residuals before Stage 1 starts and creates a target that
        # cannot be transferred back with a small Stage-2 perturbation. Keep the
        # complete original FP32 trajectory; only ``chosen`` coordinates may move.
        initialized = current.clone()
        lower_raw = current + (lower_effective - effective_current)
        upper_raw = current + (upper_effective - effective_current)
        lower_box = initialized.clone()
        upper_box = initialized.clone()
        lower_margin = torch.minimum(
            boundary_penetration * expanded_scale,
            (budget - lower_distance.clamp_min(0)).clamp_min(0),
        )
        upper_margin = torch.minimum(
            boundary_penetration * expanded_scale,
            (budget - upper_distance.clamp_min(0)).clamp_min(0),
        )
        lower_box[lower_ok] = torch.minimum(
            lower_raw[lower_ok] - lower_margin[lower_ok], current[lower_ok]
        )
        upper_box[upper_ok] = torch.maximum(
            upper_raw[upper_ok] + upper_margin[upper_ok], current[upper_ok]
        )
        lower_box[chosen] = torch.maximum(lower_box[chosen], -expanded_absmax[chosen])
        upper_box[chosen] = torch.minimum(upper_box[chosen], expanded_absmax[chosen])
        lower_box[chosen] = torch.maximum(lower_box[chosen], (current - budget)[chosen])
        upper_box[chosen] = torch.minimum(upper_box[chosen], (current + budget)[chosen])
        lower_box[chosen] = torch.minimum(lower_box[chosen], current[chosen])
        upper_box[chosen] = torch.maximum(upper_box[chosen], current[chosen])

        param.data.copy_(initialized.to(device=param.device, dtype=param.dtype))
        boxes[name] = [lower_box, upper_box]
        indices[name] = torch.nonzero(chosen.view(-1)).view(-1)
        selected = int(chosen.sum())
        scope = "vision" if ".visual." in name else "language"
        totals["parameters"] += current.numel()
        totals["eligible_parameters"] += eligible_count
        totals["selected_parameters"] += selected
        totals["anchor_parameters"] += int(anchors.sum())
        totals["changed_parameters_excluded"] += int(changed.sum())
        by_scope[scope][0] += current.numel()
        by_scope[scope][1] += selected

    print(f"GPTQ W4A16 target tensors: {len(report['layers'])}")
    print(
        f"GPTQ W4A16 reference code hashes matched: "
        f"{len(report['layers']) - len(reference_mismatches)}/{len(report['layers'])}"
    )
    print(f"GPTQ W4A16 parameters scanned: {totals['parameters']:,}")
    print(
        f"GPTQ W4A16 sensitive parameters: {totals['selected_parameters']:,} "
        f"({100 * totals['selected_parameters'] / max(1, totals['parameters']):.4f}%)"
    )
    for scope, (scope_total, scope_selected) in by_scope.items():
        print(
            f"  {scope}: {scope_selected:,}/{scope_total:,} "
            f"({100 * scope_selected / max(1, scope_total):.4f}%)"
        )
    metadata = {
        "checkpoint": str(checkpoint_path),
        "boundary_coordinate_system": "original FP32 weights (local GPTQ mapping)",
        **totals,
        "candidate_relative_limit": candidate_relative_limit,
        "candidate_absolute_limit": candidate_absolute_limit,
        "boundary_penetration": boundary_penetration,
        "selection_fraction": selection_fraction,
        "initialization": "original_fp32",
        "reference_code_mismatches": reference_mismatches,
        "requires_standard_gptq_requantization": True,
        "by_scope": {
            key: {"parameters": value[0], "selected_parameters": value[1]}
            for key, value in by_scope.items()
        },
    }
    return indices, boxes, metadata


@torch.no_grad()
def nudge_base_across_gptq_boundaries(
    base_model,
    stage1_model,
    *,
    checkpoint: str,
    selected_indices: dict[str, torch.Tensor] | None = None,
    max_relative_change: float = 0.10,
    max_absolute_change: float | None = None,
    penetration: float = 0.05,
    **_kwargs,
):
    """Transfer locally predicted Stage-1 GPTQ crossings to the base FP32 model."""
    checkpoint_path = Path(checkpoint).resolve()
    report = _load_report(checkpoint_path)
    tensors = _SafetensorIndex(checkpoint_path)
    base_params = dict(base_model.named_parameters())
    stage1_params = dict(stage1_model.named_parameters())
    stats = {
        "method": "gptq_w4a16",
        "layers": 0,
        "parameters": 0,
        "candidate_parameters": 0,
        "stage1_predicted_code_crossings": 0,
        "changed_parameters": 0,
        "predicted_target_crossings": 0,
        "failed_predicted_crossings": 0,
        "absolute_delta_sum": 0.0,
        "relative_delta_sum": 0.0,
        "max_absolute_delta": 0.0,
        "max_relative_delta": 0.0,
        "requires_standard_gptq_requantization": True,
    }
    for layer_report in report["layers"]:
        layer_name = layer_report["layer"]
        name = f"{layer_name}.weight"
        base_param = base_params[name]
        # Stage 2 loads both models on CPU. ``.float().cpu()`` can therefore
        # alias parameter storage; clone the immutable before-state so statistics
        # remain valid after writing ``updated`` back to ``base_param``.
        base = base_param.detach().float().cpu().clone()
        stage1 = stage1_params[name].detach().float().cpu()
        if _tensor_sha256(base) != layer_report["original_weight_sha256"]:
            raise RuntimeError(f"GPTQ Stage-2 base mismatch at {name}")
        artifact = _load_layer_artifact(checkpoint_path, layer_report)
        base_effective = artifact["effective_weight"]
        if base_effective is None:
            base_effective = base + artifact["effective_offset"]
        group_index = artifact["group_index_original"]
        rows, _ = base.shape
        scale = _expanded_qparam(
            tensors.get_tensor(f"{layer_name}.weight_scale"), group_index, rows
        )
        zero = _expanded_zero_point(
            tensors, layer_name, tensors.get_tensor(f"{layer_name}.weight_scale"),
            group_index, rows
        )
        q_min = int(layer_report["q_min"])
        q_max = int(layer_report["q_max"])
        stage1_effective = base_effective + (stage1 - base)
        base_code = torch.round(
            (base_effective / scale + zero).clamp(q_min, q_max)
        ).to(torch.int16)
        stage1_code = torch.round(
            (stage1_effective / scale + zero).clamp(q_min, q_max)
        ).to(torch.int16)
        direction = stage1_code - base_code
        selected = torch.zeros_like(base, dtype=torch.bool)
        if selected_indices is None:
            selected.fill_(True)
        else:
            saved = selected_indices.get(name)
            if saved is not None:
                saved = torch.as_tensor(saved, dtype=torch.long).view(-1)
                selected.view(-1)[saved] = True
        expanded_absmax = _original_group_absmax(
            base, int(layer_report["group_size"])
        )
        anchors = torch.isclose(
            base.abs(), expanded_absmax, rtol=1e-6, atol=1e-8
        )
        candidate = selected & ~anchors & (direction != 0)
        positive = direction > 0
        boundary_effective = torch.where(
            positive,
            (base_code.float() + 0.5 - zero) * scale,
            (base_code.float() - 0.5 - zero) * scale,
        )
        boundary = base + (boundary_effective - base_effective)
        budget = max_relative_change * base.abs()
        if max_absolute_change is not None:
            budget.clamp_(max=max_absolute_change)
        distance = (boundary - base).abs()
        remaining = (budget - distance).clamp_min(0)
        numerical = 8 * torch.finfo(torch.float32).eps * torch.maximum(
            boundary.abs(), scale
        )
        margin = torch.minimum(
            torch.maximum(penetration * distance, numerical), remaining
        )
        target = boundary + torch.where(positive, margin, -margin)
        target = torch.maximum(torch.minimum(target, base + budget), base - budget)
        target = torch.maximum(
            torch.minimum(target, expanded_absmax), -expanded_absmax
        )
        valid = candidate & torch.isfinite(target) & (distance <= budget + 1e-7)
        updated = base.clone()
        updated[valid] = target[valid]
        predicted_effective = base_effective + (updated - base)
        predicted_code = torch.round(
            (predicted_effective / scale + zero).clamp(q_min, q_max)
        ).to(torch.int16)
        actual = valid & ((predicted_code - base_code) * direction > 0)
        base_param.data.copy_(updated.to(base_param.device, base_param.dtype))
        delta = (updated - base).abs()
        changed = delta > 0
        relative = delta / base.abs().clamp_min(1e-12)
        stats["layers"] += 1
        stats["parameters"] += base.numel()
        stats["candidate_parameters"] += int(selected.sum())
        stats["stage1_predicted_code_crossings"] += int(candidate.sum())
        stats["changed_parameters"] += int(changed.sum())
        stats["predicted_target_crossings"] += int(actual.sum())
        stats["failed_predicted_crossings"] += int((candidate & ~actual).sum())
        stats["absolute_delta_sum"] += float(delta[changed].sum())
        stats["relative_delta_sum"] += float(relative[changed].sum())
        if changed.any():
            stats["max_absolute_delta"] = max(
                stats["max_absolute_delta"], float(delta[changed].max())
            )
            stats["max_relative_delta"] = max(
                stats["max_relative_delta"], float(relative[changed].max())
            )
    changed = max(1, stats["changed_parameters"])
    stats["mean_absolute_delta"] = stats["absolute_delta_sum"] / changed
    stats["mean_relative_delta"] = stats["relative_delta_sum"] / changed
    print(
        f"GPTQ Stage-2 predicted crossings: "
        f"{stats['stage1_predicted_code_crossings']:,}; "
        f"changed parameters: {stats['changed_parameters']:,}; "
        f"max |delta|: {stats['max_absolute_delta']:.8g}"
    )
    return stats
