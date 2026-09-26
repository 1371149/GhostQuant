"""Boundary-aware Stage-1 support for calibrated AWQ checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


AWQ_METHOD_TO_SCHEME = {
    "awq": "W4A16",
    "awq_w3a16": "W3A16",
    "awq_w4a16": "W4A16",
    "awq_w4a16_asym": "W4A16_ASYM",
    "awq_w4a8": "W4A8",
}
AWQ_METHODS = set(AWQ_METHOD_TO_SCHEME)


def is_awq_method(method: str | None) -> bool:
    return bool(method) and method.lower() in AWQ_METHODS


def awq_scheme_for_method(method: str) -> str:
    try:
        return AWQ_METHOD_TO_SCHEME[method.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported AWQ method: {method}") from exc


def _checkpoint_scheme(checkpoint: Path, report: dict) -> str:
    metadata_path = checkpoint / "awq_experiment.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        scheme = metadata.get("scheme")
        if isinstance(scheme, str):
            return scheme
    bits = {int(entry["num_bits"]) for entry in report["layers"]}
    if len(bits) != 1:
        return "mixed"
    return f"W{bits.pop()}A16"


class _SafetensorIndex:
    def __init__(self, checkpoint: Path):
        index_path = checkpoint / "model.safetensors.index.json"
        if index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            self.weight_map = payload["weight_map"]
        else:
            model_path = checkpoint / "model.safetensors"
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"No safetensors checkpoint found in {checkpoint}"
                )
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


def _load_artifacts(checkpoint: Path):
    artifact_dir = checkpoint / "boundary_artifacts"
    report_path = artifact_dir / "boundary_report.json"
    mappings_path = artifact_dir / "awq_smoothing_mappings.json"
    scales_path = artifact_dir / "awq_smoothing_scales.safetensors"
    for path in (report_path, mappings_path, scales_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing AWQ boundary artifact: {path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    mappings = json.loads(mappings_path.read_text(encoding="utf-8"))["mappings"]
    with safe_open(scales_path, framework="pt", device="cpu") as handle:
        smoothing_scales = {
            key: handle.get_tensor(key).float().contiguous() for key in handle.keys()
        }
    return report, mappings, smoothing_scales


def _smoothing_operations(mappings, smoothing_scales, params):
    operations = defaultdict(list)
    for mapping in mappings:
        scale = smoothing_scales[mapping["scale_key"]]
        for layer_name in mapping["balance_layers"]:
            param_name = f"{layer_name}.weight"
            param = params.get(param_name)
            if param is None:
                raise KeyError(f"AWQ balance parameter is absent from model: {param_name}")
            if param.ndim != 2 or param.shape[1] != scale.numel():
                raise ValueError(
                    f"AWQ column scale shape mismatch for {param_name}: "
                    f"weight={tuple(param.shape)}, scale={tuple(scale.shape)}"
                )
            operations[layer_name].append(("multiply_columns", scale))

        smooth_name = mapping["smooth_layer"]
        smooth_param = params.get(f"{smooth_name}.weight")
        if smooth_param is not None and smooth_param.ndim == 2:
            if smooth_param.shape[0] < scale.numel():
                raise ValueError(
                    f"AWQ row scale shape mismatch for {smooth_name}.weight: "
                    f"weight={tuple(smooth_param.shape)}, scale={tuple(scale.shape)}"
                )
            operations[smooth_name].append(("divide_tail_rows", scale))
    return operations


def _apply_smoothing(
    values: torch.Tensor,
    operations: list[tuple[str, torch.Tensor]],
) -> torch.Tensor:
    smoothed = values.clone()
    for operation, scale in operations:
        if operation == "multiply_columns":
            smoothed.mul_(scale.view(1, -1))
        elif operation == "divide_tail_rows":
            smoothed[-scale.numel() :].div_(scale.view(-1, 1))
        else:
            raise ValueError(f"Unknown AWQ smoothing operation: {operation}")
    return smoothed


def _smoothing_factor(
    shape: torch.Size,
    operations: list[tuple[str, torch.Tensor]],
) -> torch.Tensor:
    rows, columns = shape
    row_factor = torch.ones(rows, dtype=torch.float32)
    column_factor = torch.ones(columns, dtype=torch.float32)
    for operation, scale in operations:
        if operation == "multiply_columns":
            column_factor.mul_(scale)
        elif operation == "divide_tail_rows":
            row_factor[-scale.numel() :].div_(scale)
    return row_factor.view(-1, 1) * column_factor.view(1, -1)


def _expanded_group_scale(
    scale: torch.Tensor, columns: int, group_size: int
) -> torch.Tensor:
    if scale.ndim == 1:
        scale = scale.unsqueeze(1)
    return scale.float().repeat_interleave(group_size, dim=-1)[..., :columns]


def _unpack_int32_qparam(
    packed: torch.Tensor,
    *,
    num_bits: int,
    shape: torch.Size,
    packed_dim: int,
) -> torch.Tensor:
    """Unpack compressed-tensors signed integer qparams without that dependency."""
    if packed.dtype != torch.int32 or packed.ndim != 2:
        raise ValueError(
            f"Expected a 2-D INT32 packed qparam, got {packed.dtype} {tuple(packed.shape)}"
        )
    if not 1 <= num_bits <= 8 or packed_dim not in {0, 1}:
        raise ValueError(f"Invalid qparam packing: bits={num_bits}, dim={packed_dim}")

    value = packed.transpose(0, 1) if packed_dim == 0 else packed
    rows, num_words = value.shape
    if num_words % num_bits:
        value = F.pad(value, (0, num_bits - num_words % num_bits))
        num_words = value.shape[1]

    groups = num_words // num_bits
    words = value.reshape(rows * groups, num_bits)
    element = torch.arange(32, dtype=torch.int32, device=value.device)
    bit_start = element * num_bits
    word_index = (bit_start // 32).long()
    bit_offset = bit_start % 32
    low_bits = torch.clamp(32 - bit_offset, max=num_bits)
    output = (words[:, word_index] >> bit_offset.unsqueeze(0)) & (
        (1 << low_bits) - 1
    ).unsqueeze(0)
    overflow = low_bits < num_bits
    if overflow.any():
        high_bits = num_bits - low_bits[overflow]
        output[:, overflow] |= (
            words[:, word_index[overflow] + 1]
            & ((1 << high_bits) - 1).unsqueeze(0)
        ) << low_bits[overflow].unsqueeze(0)

    output = output.view(rows, groups * 32)[:, : int(shape[packed_dim])]
    if packed_dim == 0:
        output = output.transpose(0, 1)
    return (output - (1 << (num_bits - 1))).to(torch.int8)


def _load_zero_point(
    tensors: _SafetensorIndex,
    layer_name: str,
    *,
    rows: int,
    columns: int,
    group_size: int,
    num_bits: int,
) -> torch.Tensor | None:
    try:
        zero_point = tensors.get_tensor(f"{layer_name}.weight_zero_point")
    except KeyError:
        return None

    expected = torch.Size((rows, math.ceil(columns / group_size)))
    if zero_point.shape == expected:
        return zero_point.float()
    if zero_point.dtype == torch.int32:
        values_per_word = 32 // num_bits
        if (
            zero_point.shape[0] == math.ceil(rows / values_per_word)
            and zero_point.shape[1] == expected[1]
        ):
            unpacked = _unpack_int32_qparam(
                zero_point, num_bits=num_bits, shape=expected, packed_dim=0
            )
            if unpacked.shape == expected:
                return unpacked.float()
        if (
            zero_point.shape[0] == rows
            and zero_point.shape[1] == math.ceil(expected[1] / values_per_word)
        ):
            unpacked = _unpack_int32_qparam(
                zero_point, num_bits=num_bits, shape=expected, packed_dim=1
            )
            if unpacked.shape == expected:
                return unpacked.float()
    raise ValueError(
        f"Unsupported zero-point storage for {layer_name}: "
        f"stored={tuple(zero_point.shape)} {zero_point.dtype}, expected={tuple(expected)}"
    )


def _code_hash(codes: torch.Tensor, row_chunk_size: int) -> str:
    digest = hashlib.sha256()
    for start in range(0, codes.shape[0], row_chunk_size):
        stop = min(start + row_chunk_size, codes.shape[0])
        digest.update(codes[start:stop].contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def build_awq_finetune_boxes(
    model,
    original_model,
    *,
    checkpoint: str,
    candidate_relative_limit: float = 0.10,
    candidate_absolute_limit: float | None = None,
    boundary_penetration: float = 0.05,
    selection_fraction: float | None = None,
    row_chunk_size: int = 512,
    require_reference_code_match: bool = True,
    reference_current_model: bool = False,
    **_kwargs,
):
    """Build Stage-1 boxes in the original FP32 coordinate system.

    The saved AWQ boundaries live after function-preserving smoothing. Each
    boundary is divided by its recorded positive smoothing factor so training
    remains on the ordinary, unsmoothed Hugging Face checkpoint.
    """
    if candidate_relative_limit is None or candidate_relative_limit <= 0:
        raise ValueError("AWQ selection requires a positive relative limit")
    if candidate_absolute_limit is not None and candidate_absolute_limit <= 0:
        raise ValueError("AWQ absolute limit must be positive when enabled")
    if not 0 <= boundary_penetration <= 1:
        raise ValueError("AWQ boundary penetration must be between 0 and 1")
    if selection_fraction is not None and not 0 < selection_fraction <= 1:
        raise ValueError("AWQ selection fraction must be in (0, 1]")

    checkpoint_path = Path(checkpoint).resolve()
    report, mappings, smoothing_scales = _load_artifacts(checkpoint_path)
    scheme = _checkpoint_scheme(checkpoint_path, report)
    if not report.get("packed_checkpoint_verification", {}).get(
        "all_layers_matched", False
    ):
        raise RuntimeError("AWQ checkpoint has no successful packed-code verification")

    params = dict(model.named_parameters())
    originals = dict(original_model.named_parameters())
    operations = _smoothing_operations(mappings, smoothing_scales, originals)
    tensors = _SafetensorIndex(checkpoint_path)
    layer_reports = {entry["layer"]: entry for entry in report["layers"]}

    boxes: dict[str, list[torch.Tensor]] = {}
    indices: dict[str, torch.Tensor] = {}
    total = 0
    eligible_total = 0
    selected_total = 0
    changed_total = 0
    anchor_total = 0
    reference_mismatches = []
    totals_by_scope = {"language": [0, 0], "vision": [0, 0]}

    for layer_name, layer_report in layer_reports.items():
        name = f"{layer_name}.weight"
        if name not in params or name not in originals:
            raise KeyError(f"AWQ target parameter is absent from FP32 model: {name}")
        param = params[name]
        original_param = originals[name]
        if param.ndim != 2 or param.shape != original_param.shape:
            raise ValueError(f"Unexpected AWQ parameter shape for {name}")

        current = param.detach().float().cpu()
        original = original_param.detach().float().cpu()
        layer_operations = operations.get(layer_name, [])
        smoothed_current = _apply_smoothing(current, layer_operations)
        smoothed_original = _apply_smoothing(original, layer_operations)
        factor = _smoothing_factor(current.shape, layer_operations)
        if not torch.isfinite(factor).all() or (factor <= 0).any():
            raise RuntimeError(f"Invalid AWQ smoothing factor for {name}")

        group_size = int(layer_report["group_size"])
        num_bits = int(layer_report["num_bits"])
        q_min = int(layer_report["q_min"])
        q_max = int(layer_report["q_max"])
        scale = tensors.get_tensor(f"{layer_name}.weight_scale").float()
        zero_point = _load_zero_point(
            tensors,
            layer_name,
            rows=current.shape[0],
            columns=current.shape[1],
            group_size=group_size,
            num_bits=num_bits,
        )
        expanded_scale = _expanded_group_scale(
            scale, current.shape[1], group_size
        )
        expanded_zero = (
            _expanded_group_scale(zero_point, current.shape[1], group_size)
            if zero_point is not None
            else None
        )
        zero_float = expanded_zero if expanded_zero is not None else 0.0
        current_codes = torch.round(
            (smoothed_current / expanded_scale + zero_float).clamp(q_min, q_max)
        ).to(torch.int8)
        smoothed_reference = (
            smoothed_current if reference_current_model else smoothed_original
        )
        reference_codes = torch.round(
            (smoothed_reference / expanded_scale + zero_float).clamp(q_min, q_max)
        ).to(torch.int8)
        reference_hash = _code_hash(reference_codes, row_chunk_size)
        if reference_hash != layer_report["code_sha256"]:
            reference_mismatches.append(layer_name)
            if require_reference_code_match:
                raise RuntimeError(
                    f"AWQ boundary/model mismatch at {layer_name}: reference code "
                    "hash differs from the standard packed checkpoint"
                )

        columns = current.shape[1]
        group_count = math.ceil(columns / group_size)
        padded = F.pad(
            smoothed_reference.abs(),
            (0, group_count * group_size - columns),
            value=-1.0,
        )
        group_absmax = padded.reshape(
            current.shape[0], group_count, group_size
        ).amax(dim=-1)
        expanded_absmax = group_absmax.repeat_interleave(
            group_size, dim=-1
        )[..., :columns]
        anchors = torch.isclose(
            smoothed_reference.abs(), expanded_absmax, rtol=1e-6, atol=1e-8
        )

        code_float = current_codes.float()
        lower_smoothed = (code_float - 0.5 - zero_float) * expanded_scale
        upper_smoothed = (code_float + 0.5 - zero_float) * expanded_scale
        lower = lower_smoothed / factor
        upper = upper_smoothed / factor
        lower_distance = current - lower
        upper_distance = upper - current
        lower_distance = torch.where(
            code_float > q_min, lower_distance, torch.full_like(current, torch.inf)
        )
        upper_distance = torch.where(
            code_float < q_max, upper_distance, torch.full_like(current, torch.inf)
        )
        budget = candidate_relative_limit * current.abs()
        if candidate_absolute_limit is not None:
            budget.clamp_(max=candidate_absolute_limit)
        changed = current != original
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
                    f"{name} has only {eligible_count:,} AWQ candidates; cannot "
                    f"select {target_count:,} ({selection_fraction:.2%})"
                )
            if eligible_count > target_count:
                nearest = torch.minimum(
                    lower_distance.clamp_min(0), upper_distance.clamp_min(0)
                )
                score = nearest / budget.clamp_min(1e-12)
                candidate_ids = torch.nonzero(eligible.view(-1), as_tuple=False).view(-1)
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

        dequantized = (code_float - zero_float) * expanded_scale / factor
        initialized = dequantized
        preserve = chosen | anchors | changed
        initialized[preserve] = current[preserve]
        lower_box = initialized.clone()
        upper_box = initialized.clone()
        quant_step = expanded_scale / factor

        lower_margin = torch.minimum(
            boundary_penetration * quant_step,
            (budget - lower_distance.clamp_min(0)).clamp_min(0),
        )
        upper_margin = torch.minimum(
            boundary_penetration * quant_step,
            (budget - upper_distance.clamp_min(0)).clamp_min(0),
        )
        lower_box[lower_ok] = torch.minimum(
            lower[lower_ok] - lower_margin[lower_ok], current[lower_ok]
        )
        upper_box[upper_ok] = torch.maximum(
            upper[upper_ok] + upper_margin[upper_ok], current[upper_ok]
        )

        group_min = (q_min - zero_float) * expanded_scale / factor
        group_max = (q_max - zero_float) * expanded_scale / factor
        lower_box[chosen] = torch.maximum(lower_box[chosen], group_min[chosen])
        upper_box[chosen] = torch.minimum(upper_box[chosen], group_max[chosen])
        lower_box[chosen] = torch.maximum(
            lower_box[chosen], (current - budget)[chosen]
        )
        upper_box[chosen] = torch.minimum(
            upper_box[chosen], (current + budget)[chosen]
        )
        lower_box[chosen] = torch.minimum(lower_box[chosen], current[chosen])
        upper_box[chosen] = torch.maximum(upper_box[chosen], current[chosen])

        param.data.copy_(initialized.to(device=param.device, dtype=param.dtype))
        boxes[name] = [lower_box, upper_box]
        indices[name] = torch.nonzero(chosen.view(-1), as_tuple=False).view(-1)

        layer_total = current.numel()
        chosen_count = int(chosen.sum())
        scope = "vision" if any(x in name for x in (".visual.", ".vision_tower.", ".multi_modal_projector.")) else "language"
        total += layer_total
        eligible_total += eligible_count
        selected_total += chosen_count
        changed_total += int(changed.sum())
        anchor_total += int(anchors.sum())
        totals_by_scope[scope][0] += layer_total
        totals_by_scope[scope][1] += chosen_count

    absolute_text = (
        f"{candidate_absolute_limit:g}"
        if candidate_absolute_limit is not None
        else "disabled"
    )
    print(f"AWQ {scheme} target tensors: {len(layer_reports)}")
    print(f"AWQ {scheme} reference code hashes matched: {len(layer_reports) - len(reference_mismatches)}/{len(layer_reports)}")
    print(f"AWQ {scheme} parameters scanned: {total:,}")
    print(
        f"AWQ {scheme} selection limits: relative <= {candidate_relative_limit:g}, "
        f"absolute <= {absolute_text}"
    )
    print(
        f"AWQ {scheme} sensitive parameters: {selected_total:,} "
        f"({100.0 * selected_total / max(1, total):.4f}%)"
    )
    print(
        f"AWQ {scheme} locked absmax anchors: {anchor_total:,} "
        f"({100.0 * anchor_total / max(1, total):.4f}%)"
    )
    if changed_total:
        print(f"AWQ {scheme} earlier cascade parameters preserved: {changed_total:,}")
    for scope, (scope_total, scope_selected) in totals_by_scope.items():
        if scope_total:
            print(
                f"  {scope}: {scope_selected:,}/{scope_total:,} "
                f"({100.0 * scope_selected / scope_total:.4f}%)"
            )

    metadata = {
        "checkpoint": str(checkpoint_path),
        "scheme": scheme,
        "boundary_coordinate_system": (
            "current stage-base FP32 weights"
            if reference_current_model
            else "original FP32 weights"
        ),
        "calibrated_boundary_coordinate_system": "AWQ-smoothed FP32 weights",
        "target_tensors": len(layer_reports),
        "parameters": total,
        "eligible_parameters": eligible_total,
        "selected_parameters": selected_total,
        "anchor_parameters": anchor_total,
        "changed_parameters_excluded": changed_total,
        "reference_code_mismatches": reference_mismatches,
        "reference_current_model": reference_current_model,
        "candidate_relative_limit": candidate_relative_limit,
        "candidate_absolute_limit": candidate_absolute_limit,
        "boundary_penetration": boundary_penetration,
        "selection_fraction": selection_fraction,
        "by_scope": {
            scope: {"parameters": values[0], "selected_parameters": values[1]}
            for scope, values in totals_by_scope.items()
        },
    }
    return indices, boxes, metadata


@torch.no_grad()
def nudge_base_across_awq_boundaries(
    base_model,
    stage1_model,
    *,
    checkpoint: str,
    selected_indices: dict[str, torch.Tensor] | None = None,
    max_relative_change: float = 0.10,
    max_absolute_change: float | None = None,
    penetration: float = 0.05,
    row_chunk_size: int = 512,
    **_kwargs,
):
    """Transfer only Stage-1 AWQ code crossings to the original FP32 model."""
    if max_relative_change is None or max_relative_change <= 0:
        raise ValueError("AWQ Stage-2 requires a positive relative limit")
    if max_absolute_change is not None and max_absolute_change <= 0:
        raise ValueError("AWQ Stage-2 absolute limit must be positive when enabled")
    if not 0 <= penetration <= 1:
        raise ValueError("AWQ Stage-2 penetration must be between 0 and 1")

    checkpoint_path = Path(checkpoint).resolve()
    report, mappings, smoothing_scales = _load_artifacts(checkpoint_path)
    scheme = _checkpoint_scheme(checkpoint_path, report)
    base_params = dict(base_model.named_parameters())
    stage1_params = dict(stage1_model.named_parameters())
    operations = _smoothing_operations(mappings, smoothing_scales, base_params)
    tensors = _SafetensorIndex(checkpoint_path)

    stats = {
        "method": f"awq_{scheme.lower()}",
        "scheme": scheme,
        "layers": 0,
        "parameters": 0,
        "candidate_parameters": 0,
        "stage1_code_crossing_parameters": 0,
        "changed_parameters": 0,
        "actual_target_crossings": 0,
        "failed_target_crossings": 0,
        "budget_violations": 0,
        "absmax_range_violations": 0,
        "absolute_delta_sum": 0.0,
        "relative_delta_sum": 0.0,
        "max_absolute_delta": 0.0,
        "max_relative_delta": 0.0,
    }

    layer_reports = {entry["layer"]: entry for entry in report["layers"]}
    for layer_name, layer_report in layer_reports.items():
        name = f"{layer_name}.weight"
        base_param = base_params[name]
        stage1_param = stage1_params[name]
        base = base_param.detach().float().cpu()
        stage1 = stage1_param.detach().float().cpu()
        operations_for_layer = operations.get(layer_name, [])
        base_smoothed = _apply_smoothing(base, operations_for_layer)
        stage1_smoothed = _apply_smoothing(stage1, operations_for_layer)
        factor = _smoothing_factor(base.shape, operations_for_layer)
        scale = tensors.get_tensor(f"{layer_name}.weight_scale").float()
        group_size = int(layer_report["group_size"])
        num_bits = int(layer_report["num_bits"])
        zero_point = _load_zero_point(
            tensors,
            layer_name,
            rows=base.shape[0],
            columns=base.shape[1],
            group_size=group_size,
            num_bits=num_bits,
        )
        expanded_scale = _expanded_group_scale(scale, base.shape[1], group_size)
        expanded_zero = (
            _expanded_group_scale(zero_point, base.shape[1], group_size)
            if zero_point is not None
            else None
        )
        q_min = int(layer_report["q_min"])
        q_max = int(layer_report["q_max"])

        zero_float = expanded_zero if expanded_zero is not None else 0.0
        base_code = torch.round(
            (base_smoothed / expanded_scale + zero_float).clamp(q_min, q_max)
        ).to(torch.int16)
        stage1_code = torch.round(
            (stage1_smoothed / expanded_scale + zero_float).clamp(q_min, q_max)
        ).to(torch.int16)
        direction_code = stage1_code - base_code
        if selected_indices is None:
            selected = torch.ones_like(base_code, dtype=torch.bool)
        else:
            selected = torch.zeros_like(base_code, dtype=torch.bool)
            saved = selected_indices.get(name)
            if saved is not None:
                saved = torch.as_tensor(saved, dtype=torch.long).view(-1)
                if torch.any((saved < 0) | (saved >= base.numel())):
                    raise ValueError(f"Saved AWQ Stage-1 indices are invalid for {name}")
                selected.view(-1)[saved] = True

        # The Stage-1 builder excludes anchors and earlier cascade changes.
        # Recompute this from the immutable base coordinate for Stage-2 too.
        columns = base.shape[1]
        group_count = math.ceil(columns / group_size)
        padded = F.pad(
            base_smoothed.abs(),
            (0, group_count * group_size - columns),
            value=-1.0,
        )
        group_absmax = padded.reshape(
            base.shape[0], group_count, group_size
        ).amax(dim=-1)
        expanded_absmax = group_absmax.repeat_interleave(
            group_size, dim=-1
        )[..., :columns]
        anchors = torch.isclose(
            base_smoothed.abs(), expanded_absmax, rtol=1e-6, atol=1e-8
        )
        candidate = selected & ~anchors & (direction_code != 0)
        stats["layers"] += 1
        stats["parameters"] += base.numel()
        stats["candidate_parameters"] += int((selected & ~anchors).sum())
        stats["stage1_code_crossing_parameters"] += int(candidate.sum())

        budget = max_relative_change * base.abs()
        if max_absolute_change is not None:
            budget.clamp_(max=max_absolute_change)
        positive = direction_code > 0
        boundary_smoothed = torch.where(
            positive,
            (base_code.float() + 0.5 - zero_float) * expanded_scale,
            (base_code.float() - 0.5 - zero_float) * expanded_scale,
        )
        boundary = boundary_smoothed / factor
        distance = (boundary - base).abs()
        remaining = (budget - distance).clamp_min(0)
        quant_step = expanded_scale / factor
        numerical_margin = (
            8
            * torch.finfo(torch.float32).eps
            * torch.maximum(boundary.abs(), quant_step)
        )
        margin = torch.minimum(
            torch.maximum(penetration * distance, numerical_margin), remaining
        )
        target = boundary + torch.where(positive, margin, -margin)
        ties = candidate & (target == boundary) & torch.isfinite(boundary)
        if ties.any():
            positive_inf = torch.full_like(boundary, torch.inf)
            negative_inf = torch.full_like(boundary, -torch.inf)
            target = torch.where(
                ties & positive,
                torch.nextafter(boundary, positive_inf),
                target,
            )
            target = torch.where(
                ties & ~positive,
                torch.nextafter(boundary, negative_inf),
                target,
            )
        target = torch.maximum(torch.minimum(target, base + budget), base - budget)
        range_min = (q_min - zero_float) * expanded_scale / factor
        range_max = (q_max - zero_float) * expanded_scale / factor
        target = torch.maximum(torch.minimum(target, range_max), range_min)

        valid = candidate & torch.isfinite(target)
        valid &= distance <= budget + 1e-7
        stats["budget_violations"] += int(
            (candidate & (distance > budget + 1e-7)).sum()
        )
        range_valid = target >= range_min
        range_valid &= target <= range_max
        stats["absmax_range_violations"] += int((candidate & ~range_valid).sum())
        valid &= range_valid
        if valid.any():
            flat_valid = valid.view(-1)
            flat_target = target.view(-1)
            base_before = base.clone()
            updated = base_before.clone()
            updated.view(-1)[flat_valid] = flat_target[flat_valid]
            # Re-evaluate in the exact sequential smoothing path. A composed
            # inverse factor can land one or two FP32 ULPs short of the boundary;
            # deepen only those coordinates and verify again without relaxing
            # either the relative budget or the frozen absmax range.
            for attempt in range(5):
                updated_smoothed = _apply_smoothing(updated, operations_for_layer)
                updated_code = torch.round(
                    (updated_smoothed / expanded_scale + zero_float).clamp(q_min, q_max)
                ).to(torch.int16)
                failed = valid & (
                    (updated_code - base_code) * direction_code <= 0
                )
                if not failed.any() or attempt == 4:
                    break
                step = (
                    (2 ** (attempt + 3))
                    * torch.finfo(torch.float32).eps
                    * torch.maximum(updated.abs(), quant_step)
                )
                proposed = updated + torch.where(positive, step, -step)
                positive_inf = torch.full_like(updated, torch.inf)
                negative_inf = torch.full_like(updated, -torch.inf)
                proposed = torch.where(
                    failed & positive,
                    torch.maximum(proposed, torch.nextafter(updated, positive_inf)),
                    proposed,
                )
                proposed = torch.where(
                    failed & ~positive,
                    torch.minimum(proposed, torch.nextafter(updated, negative_inf)),
                    proposed,
                )
                proposed = torch.maximum(
                    torch.minimum(proposed, base_before + budget),
                    base_before - budget,
                )
                proposed = torch.maximum(
                    torch.minimum(proposed, range_max), range_min
                )
                updated[failed] = proposed[failed]

            base_param.data.copy_(
                updated.to(device=base_param.device, dtype=base_param.dtype)
            )
            actual = valid & ((updated_code - base_code) * direction_code > 0)
            stats["actual_target_crossings"] += int(actual.sum())
            stats["failed_target_crossings"] += int((candidate & ~actual).sum())
            abs_delta = (updated - base_before).abs()
            selected_delta = abs_delta[valid]
            stats["changed_parameters"] += int((selected_delta > 0).sum())
            stats["absolute_delta_sum"] += float(selected_delta.sum())
            stats["relative_delta_sum"] += float(
                (selected_delta / base_before.abs().clamp_min(1e-12)[valid]).sum()
            )
            if selected_delta.numel():
                stats["max_absolute_delta"] = max(
                    stats["max_absolute_delta"], float(selected_delta.max())
                )
                stats["max_relative_delta"] = max(
                    stats["max_relative_delta"],
                    float(
                        (selected_delta / base_before.abs().clamp_min(1e-12)[valid]).max()
                    ),
                )
        else:
            stats["failed_target_crossings"] += int(candidate.sum())

    stats["mean_absolute_delta"] = stats["absolute_delta_sum"] / max(
        1, stats["changed_parameters"]
    )
    stats["mean_relative_delta"] = stats["relative_delta_sum"] / max(
        1, stats["changed_parameters"]
    )
    print(f"AWQ Stage-2 candidates: {stats['candidate_parameters']:,}")
    print(
        f"AWQ Stage-2 Stage-1 code crossings: "
        f"{stats['stage1_code_crossing_parameters']:,}"
    )
    print(
        f"AWQ Stage-2 actual crossings: {stats['actual_target_crossings']:,}; "
        f"changed parameters: {stats['changed_parameters']:,}; "
        f"max |delta|: {stats['max_absolute_delta']:.8g}"
    )
    return stats


class _AWQActivationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, column_scale):
        # Stage1 weights use the unsmoothed coordinate. Deployment sees x / s.
        smoothed = value / column_scale
        # Match compressed-tensors 0.18 TOKEN semantics, including 2-D vision
        # inputs (global reduction there, rather than inventing a row strategy).
        dims = tuple(range(2, smoothed.ndim))
        peak = smoothed.abs().amax(dim=dims, keepdim=True) if dims else smoothed.abs().amax()
        scale = peak / 127.5
        scale = torch.where(scale == 0, torch.full_like(scale, torch.finfo(scale.dtype).eps), scale)
        result = torch.round(smoothed / scale).clamp(-128, 127) * scale
        return result * column_scale

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def install_awq_activation_proxy(model, checkpoint):
    """Install dynamic INT8 input simulation for the saved W4A8 recipe only.

    Does not change weights, saved checkpoints, or Stage2. The STE supplies an
    identity gradient while forward uses deployment's activation quantizer.
    This is an activation proxy, not a claim of exact full-model equivalence.
    """
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / 'config.json').read_text())
    groups = config.get('quantization_config', {}).get('config_groups', {})
    active = [g.get('input_activations') for g in groups.values() if g.get('input_activations')]
    if not active:
        return [], {'enabled': False, 'layers': 0}
    if len(groups) != 1 or len(active) != 1:
        raise ValueError('AWQ activation proxy requires a single uniform quantization group')
    args = active[0]
    expected = {'num_bits': 8, 'type': 'int', 'symmetric': True, 'dynamic': True, 'strategy': 'token'}
    if any(args.get(k) != v for k, v in expected.items()) or args.get('scale_dtype') is not None:
        raise ValueError(f'Unsupported AWQ activation quantization: {args}')
    report, mappings, scales = _load_artifacts(checkpoint)
    operations = _smoothing_operations(mappings, scales, dict(model.named_parameters()))
    modules = dict(model.named_modules())
    prepared = []
    for entry in report['layers']:
        name = entry['layer']
        module = modules[name]
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f'AWQ activation target must be Linear: {name}')
        if getattr(module, '_gq_awq_activation_proxy', False):
            raise RuntimeError(f'AWQ activation proxy already installed: {name}')
        factor = torch.ones(module.weight.shape[1], device=module.weight.device, dtype=module.weight.dtype)
        for kind, value in operations.get(name, []):
            if kind == 'multiply_columns':
                factor.mul_(value.to(factor))
        if not torch.isfinite(factor).all() or not (factor > 0).all():
            raise ValueError(f'Invalid activation smoothing scale: {name}')
        prepared.append((module, factor))
    handles = []
    for module, factor in prepared:
        def hook(layer, positional, keyword, factor=factor):
            if positional:
                value = positional[0]
                return (_AWQActivationSTE.apply(value, factor.to(value)), *positional[1:]), keyword
            value = keyword['input']
            return positional, dict(keyword, input=_AWQActivationSTE.apply(value, factor.to(value)))
        handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
        module._gq_awq_activation_proxy = True
    return handles, {'enabled': True, 'layers': len(handles), 'input_activations': args,
                     'coordinate': 'AWQ-smoothed input, mapped back to Stage1 coordinate',
                     'reference': 'compressed-tensors 0.18 TOKEN semantics'}
