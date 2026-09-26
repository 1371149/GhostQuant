"""Capture and verify calibrated AWQ weight-quantization boundaries."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
from compressed_tensors.quantization.lifecycle.forward import quantize
from compressed_tensors.quantization.utils import calculate_range


RELATIVE_THRESHOLDS = (0.005, 0.01, 0.02, 0.05, 0.10)
BOUNDARY_NUMERICAL_TOLERANCE = 1e-6


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


class AWQSmoothingRecorder:
    """Record the channel scales selected by an unmodified AWQModifier."""

    def __init__(self, modifier):
        self.modifier = modifier
        self.scales: dict[str, torch.Tensor] = {}
        self.mappings: list[dict[str, Any]] = []
        self._original = modifier._compute_best_scale
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return

        recorder = self

        def record_scale(_modifier, mapping, fp_outputs, original_weights):
            scale = recorder._original(mapping, fp_outputs, original_weights)
            scale_key = f"mapping_{len(recorder.mappings):04d}"
            stored = scale.detach().to(dtype=torch.float32, device="cpu").contiguous()
            recorder.scales[scale_key] = stored
            recorder.mappings.append(
                {
                    "index": len(recorder.mappings),
                    "scale_key": scale_key,
                    "smooth_layer": mapping.smooth_name,
                    "balance_layers": list(mapping.balance_names),
                    "parent": mapping.parent_name,
                    "channels": stored.numel(),
                    "scale_min": float(stored.min()),
                    "scale_max": float(stored.max()),
                    "scale_mean": float(stored.mean()),
                }
            )
            return scale

        self.modifier._compute_best_scale = MethodType(record_scale, self.modifier)
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        self.modifier._compute_best_scale = self._original
        self._installed = False

    def save(self, directory: Path) -> dict[str, Any]:
        directory.mkdir(parents=True, exist_ok=True)
        scale_path = directory / "awq_smoothing_scales.safetensors"
        mapping_path = directory / "awq_smoothing_mappings.json"
        if self.scales:
            save_file(self.scales, scale_path)
        mapping_payload = {
            "description": (
                "AWQ channel scales in application order. These are fused into "
                "adjacent FP weights by the standard quantizer and are not required "
                "for ordinary quantized inference."
            ),
            "mapping_count": len(self.mappings),
            "mappings": self.mappings,
        }
        mapping_path.write_text(json.dumps(mapping_payload, indent=2), encoding="utf-8")
        return {
            "scale_file": str(scale_path) if self.scales else None,
            "mapping_file": str(mapping_path),
            "mapping_count": len(self.mappings),
        }


def _expand_group_qparam(
    value: torch.Tensor | None,
    rows: slice,
    columns: int,
    group_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    if value is None:
        return None
    selected = value.detach().to(dtype=torch.float32, device=device)
    if selected.ndim == 0:
        selected = selected.reshape(1, 1)
    elif selected.ndim == 1:
        selected = selected.unsqueeze(1)
    if selected.shape[0] > 1:
        selected = selected[rows]
    row_count = rows.stop - rows.start
    if selected.shape[0] == 1 and row_count > 1:
        selected = selected.expand(row_count, -1)
    return selected.repeat_interleave(group_size, dim=-1)[..., :columns]


def _scale_chunk(
    scale: torch.Tensor,
    rows: slice,
    row_count: int,
    device: torch.device,
) -> torch.Tensor:
    selected = scale.detach().to(dtype=torch.float32, device=device)
    if selected.ndim == 0:
        selected = selected.reshape(1, 1)
    elif selected.ndim == 1:
        selected = selected.unsqueeze(1)
    if selected.shape[0] > 1:
        selected = selected[rows]
    if selected.shape[0] == 1 and row_count > 1:
        selected = selected.expand(row_count, -1)
    return selected.contiguous()


@torch.no_grad()
def collect_boundary_reference(
    model,
    *,
    row_chunk_size: int = 512,
    compute_device: str = "cuda",
    relative_thresholds: tuple[float, ...] = RELATIVE_THRESHOLDS,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Independently recompute AWQ codes and boundary distances from live modules."""

    if compute_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA boundary verification requested but CUDA is unavailable")
    device = torch.device(compute_device)
    layer_reports: list[dict[str, Any]] = []
    expected_code_specs: dict[str, dict[str, Any]] = {}
    totals = {
        "layers": 0,
        "parameters": 0,
        "analytic_code_mismatches": 0,
        "anchor_parameters": 0,
        "negative_boundary_distances": 0,
        "most_negative_boundary_distance_before_clamp": 0.0,
        "nearest_boundary_distance_sum": 0.0,
        "candidate_counts_excluding_anchors": {
            str(value): 0 for value in relative_thresholds
        },
    }
    skipped: list[dict[str, str]] = []

    for name, module in model.named_modules():
        scheme = getattr(module, "quantization_scheme", None)
        weight_args = getattr(scheme, "weights", None)
        weight = getattr(module, "weight", None)
        weight_scale = getattr(module, "weight_scale", None)
        if weight_args is None or weight is None or weight_scale is None:
            continue
        if weight_args.type != "int" or weight_args.strategy != "group":
            skipped.append(
                {
                    "layer": name,
                    "reason": f"unsupported {weight_args.type}/{weight_args.strategy}",
                }
            )
            continue
        if weight.ndim != 2:
            skipped.append({"layer": name, "reason": f"weight ndim={weight.ndim}"})
            continue

        group_size = int(weight_args.group_size)
        rows_total, columns = weight.shape
        q_min_tensor, q_max_tensor = calculate_range(weight_args, device)
        q_min = int(q_min_tensor.item())
        q_max = int(q_max_tensor.item())
        zero_point = getattr(module, "weight_zero_point", None)
        if weight_args.symmetric:
            zero_point = None

        code_hasher = hashlib.sha256()
        layer_mismatches = 0
        layer_anchors = 0
        layer_negative_distances = 0
        most_negative_distance = 0.0
        distance_sum = 0.0
        distance_min = math.inf
        distance_max = 0.0
        scale_abs_error_max = 0.0
        scale_rel_error_max = 0.0
        candidate_counts = {str(value): 0 for value in relative_thresholds}

        for row_start in range(0, rows_total, row_chunk_size):
            row_stop = min(row_start + row_chunk_size, rows_total)
            row_slice = slice(row_start, row_stop)
            row_count = row_stop - row_start
            values = weight[row_slice].detach().to(dtype=torch.float32, device=device)
            scales = _scale_chunk(weight_scale, row_slice, row_count, device)
            expanded_scale = _expand_group_qparam(
                weight_scale, row_slice, columns, group_size, device
            )
            expanded_zero = _expand_group_qparam(
                zero_point, row_slice, columns, group_size, device
            )

            library_codes = quantize(
                x=values,
                scale=scales,
                zero_point=(
                    None
                    if zero_point is None
                    else _scale_chunk(zero_point, row_slice, row_count, device)
                ),
                args=weight_args,
                dtype=torch.int8,
            )
            scaled = values / expanded_scale
            if expanded_zero is not None:
                scaled = scaled + expanded_zero
            analytic_codes = torch.round(scaled.clamp(q_min, q_max)).to(torch.int8)
            mismatches = int((analytic_codes != library_codes).sum())
            layer_mismatches += mismatches
            code_hasher.update(
                analytic_codes.to(device="cpu").contiguous().numpy().tobytes()
            )

            group_count = math.ceil(columns / group_size)
            padded_columns = group_count * group_size
            padded_values = torch.nn.functional.pad(
                values.abs(), (0, padded_columns - columns), value=-1.0
            )
            group_absmax = padded_values.reshape(
                row_count, group_count, group_size
            ).amax(dim=-1)
            expected_scale = group_absmax / ((q_max - q_min) / 2.0)
            nonzero_groups = group_absmax > 0
            scale_abs_error = (scales - expected_scale).abs()
            if scale_abs_error.numel():
                scale_abs_error_max = max(
                    scale_abs_error_max, float(scale_abs_error.max())
                )
            if nonzero_groups.any():
                relative_error = (
                    scale_abs_error[nonzero_groups]
                    / expected_scale[nonzero_groups].abs()
                )
                scale_rel_error_max = max(
                    scale_rel_error_max, float(relative_error.max())
                )

            expanded_absmax = group_absmax.repeat_interleave(
                group_size, dim=-1
            )[..., :columns]
            anchors = torch.isclose(
                values.abs(), expanded_absmax, rtol=1e-6, atol=1e-8
            )
            layer_anchors += int(anchors.sum())

            codes_float = analytic_codes.to(torch.float32)
            zero_float = expanded_zero if expanded_zero is not None else 0.0
            lower = (codes_float - 0.5 - zero_float) * expanded_scale
            upper = (codes_float + 0.5 - zero_float) * expanded_scale
            lower_distance = values - lower
            upper_distance = upper - values
            inf = torch.full_like(values, torch.inf)
            lower_distance = torch.where(codes_float > q_min, lower_distance, inf)
            upper_distance = torch.where(codes_float < q_max, upper_distance, inf)
            nearest = torch.minimum(lower_distance, upper_distance)
            if nearest.numel():
                most_negative_distance = min(
                    most_negative_distance, float(nearest.min())
                )
            negative = nearest < -BOUNDARY_NUMERICAL_TOLERANCE
            layer_negative_distances += int(negative.sum())
            nearest = nearest.clamp_min(0)
            finite = torch.isfinite(nearest)
            if finite.any():
                finite_distance = nearest[finite]
                distance_sum += float(finite_distance.sum())
                distance_min = min(distance_min, float(finite_distance.min()))
                distance_max = max(distance_max, float(finite_distance.max()))

            eligible = finite & ~anchors
            for threshold in relative_thresholds:
                selected = eligible & (nearest <= threshold * values.abs())
                candidate_counts[str(threshold)] += int(selected.sum())

        layer_parameters = weight.numel()
        code_hash = code_hasher.hexdigest()
        expected_code_specs[name] = {
            "sha256": code_hash,
            "num_bits": int(weight_args.num_bits),
            "shape": list(weight.shape),
        }
        layer_report = {
            "layer": name,
            "shape": list(weight.shape),
            "parameters": layer_parameters,
            "num_bits": int(weight_args.num_bits),
            "group_size": group_size,
            "q_min": q_min,
            "q_max": q_max,
            "symmetric": bool(weight_args.symmetric),
            "code_sha256": code_hash,
            "analytic_code_mismatches": layer_mismatches,
            "anchor_parameters": layer_anchors,
            "negative_boundary_distances": layer_negative_distances,
            "most_negative_boundary_distance_before_clamp": most_negative_distance,
            "nearest_boundary_distance_mean": distance_sum / layer_parameters,
            "nearest_boundary_distance_min": (
                distance_min if math.isfinite(distance_min) else None
            ),
            "nearest_boundary_distance_max": distance_max,
            "scale_vs_group_absmax_max_absolute_error": scale_abs_error_max,
            "scale_vs_group_absmax_max_relative_error": scale_rel_error_max,
            "candidate_counts_excluding_anchors": candidate_counts,
        }
        layer_reports.append(layer_report)

        totals["layers"] += 1
        totals["parameters"] += layer_parameters
        totals["analytic_code_mismatches"] += layer_mismatches
        totals["anchor_parameters"] += layer_anchors
        totals["negative_boundary_distances"] += layer_negative_distances
        totals["most_negative_boundary_distance_before_clamp"] = min(
            totals["most_negative_boundary_distance_before_clamp"],
            most_negative_distance,
        )
        totals["nearest_boundary_distance_sum"] += distance_sum
        for threshold, count in candidate_counts.items():
            totals["candidate_counts_excluding_anchors"][threshold] += count

    parameter_count = totals["parameters"]
    totals["nearest_boundary_distance_mean"] = (
        totals.pop("nearest_boundary_distance_sum") / parameter_count
        if parameter_count
        else None
    )
    totals["anchor_fraction"] = (
        totals["anchor_parameters"] / parameter_count if parameter_count else None
    )
    totals["candidate_fractions_excluding_anchors"] = {
        threshold: count / parameter_count if parameter_count else None
        for threshold, count in totals[
            "candidate_counts_excluding_anchors"
        ].items()
    }
    return (
        {
            "boundary_coordinate_system": "AWQ-smoothed FP32 weights",
            "verification_compute_device": str(device),
            "boundary_formula": (
                "For integer code q, lower=(q-0.5-zp)*scale and "
                "upper=(q+0.5-zp)*scale; outward boundaries of saturated "
                "q_min/q_max codes are ignored."
            ),
            "relative_candidate_rule": (
                "nearest_boundary_distance <= threshold * abs(weight), excluding "
                "per-group absmax anchors"
            ),
            "relative_thresholds": list(relative_thresholds),
            "boundary_numerical_tolerance": BOUNDARY_NUMERICAL_TOLERANCE,
            "totals": totals,
            "layers": layer_reports,
            "skipped": skipped,
        },
        expected_code_specs,
    )


class _SafetensorIndex:
    def __init__(self, checkpoint: Path):
        index_path = checkpoint / "model.safetensors.index.json"
        if index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            self.weight_map = payload["weight_map"]
        else:
            model_path = checkpoint / "model.safetensors"
            if not model_path.is_file():
                raise FileNotFoundError(f"No safetensors checkpoint found in {checkpoint}")
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

    def keys(self):
        return self.weight_map.keys()


def load_quantized_weight_codes(
    tensors: _SafetensorIndex,
    layer_name: str,
    num_bits: int,
    expected_shape: torch.Size | None = None,
) -> tuple[torch.Tensor, str]:
    """Load standard compressed-tensors integer codes in either storage format."""

    try:
        packed = tensors.get_tensor(f"{layer_name}.weight_packed")
        shape = tensors.get_tensor(f"{layer_name}.weight_shape")
    except KeyError:
        weight = tensors.get_tensor(f"{layer_name}.weight")
        if weight.dtype not in {
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise ValueError(
                f"{layer_name}.weight is not an integer quantized tensor: {weight.dtype}"
            )
        codes = weight.to(torch.int8)
        storage = "int-quantized"
    else:
        codes = unpack_from_int32(packed, num_bits, torch.Size(shape.tolist()))
        storage = "pack-quantized"

    if expected_shape is not None and codes.shape != expected_shape:
        raise ValueError(
            f"Quantized shape differs for {layer_name}: "
            f"{tuple(codes.shape)} != {tuple(expected_shape)}"
        )
    return codes, storage


def verify_packed_checkpoint(
    checkpoint: Path,
    expected_code_specs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Check that independently computed codes equal the standard packed output."""

    tensors = _SafetensorIndex(checkpoint)
    layers = []
    matched = 0
    missing = 0
    storage_formats: dict[str, int] = {}
    for layer_name, code_spec in expected_code_specs.items():
        expected_hash = code_spec["sha256"]
        num_bits = int(code_spec["num_bits"])
        expected_shape = (
            torch.Size(code_spec["shape"]) if code_spec.get("shape") else None
        )
        try:
            unpacked, storage = load_quantized_weight_codes(
                tensors, layer_name, num_bits, expected_shape
            )
        except (KeyError, ValueError) as error:
            layers.append(
                {
                    "layer": layer_name,
                    "matched": False,
                    "error": str(error),
                }
            )
            missing += 1
            continue
        storage_formats[storage] = storage_formats.get(storage, 0) + 1
        actual_hash = _tensor_sha256(unpacked)
        is_match = actual_hash == expected_hash
        matched += int(is_match)
        layers.append(
            {
                "layer": layer_name,
                "num_bits": num_bits,
                "storage": storage,
                "matched": is_match,
                "expected_code_sha256": expected_hash,
                "packed_code_sha256": actual_hash,
            }
        )

    standard_keys = list(tensors.keys())
    return {
        "layers_checked": len(expected_code_specs),
        "layers_matched": matched,
        "layers_missing": missing,
        "all_layers_matched": matched == len(expected_code_specs),
        "storage_formats": storage_formats,
        "standard_checkpoint_weight_scale_tensors": sum(
            key.endswith(".weight_scale") for key in standard_keys
        ),
        "standard_checkpoint_weight_zero_point_tensors": sum(
            key.endswith(".weight_zero_point") for key in standard_keys
        ),
        "standard_checkpoint_explicit_awq_smoothing_tensors": sum(
            "awq_smoothing" in key for key in standard_keys
        ),
        "layers": layers,
    }
