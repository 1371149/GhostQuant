"""Record and verify the effective rounding coordinates used by standard GPTQ."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from compressed_tensors.quantization import ActivationOrdering, QuantizationStrategy
from compressed_tensors.quantization.utils import calculate_range

from awq_boundary import verify_packed_checkpoint


RELATIVE_THRESHOLDS = (0.005, 0.01, 0.02, 0.05, 0.10)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _expand_column_qparams(
    value: torch.Tensor,
    group_index_original: torch.Tensor,
    rows: int,
) -> torch.Tensor:
    value = value.detach().float()
    if value.ndim == 0:
        value = value.reshape(1, 1)
    elif value.ndim == 1:
        value = value.unsqueeze(1)
    if value.shape[0] == 1 and rows > 1:
        value = value.expand(rows, -1)
    return value[:, group_index_original.long()]


class GPTQBoundaryRecorder:
    """Observe llm-compressor GPTQ without replacing its quantization algorithm.

    GPTQ changes later columns before rounding them to compensate for earlier
    quantization errors. The hook records that effective pre-round value and maps it
    back to the original FP32 column order. The standard library still performs all
    Hessian estimation, error propagation, rounding, and checkpoint compression.
    """

    def __init__(self, modifier, artifact_dir: Path):
        self.modifier = modifier
        self.artifact_dir = Path(artifact_dir)
        self.layer_reports: list[dict[str, Any]] = []
        self.expected_code_specs: dict[str, dict[str, Any]] = {}
        self._current: dict[str, Any] | None = None
        self._installed = False
        self._original_quantize_weight = None
        self._original_fake_quantize = None

    def install(self) -> None:
        if self._installed:
            return
        import llmcompressor.modifiers.gptq.base as gptq_base
        import llmcompressor.modifiers.gptq.gptq_quantize as gptq_quantize

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._original_quantize_weight = gptq_base.quantize_weight
        self._original_fake_quantize = gptq_quantize.fake_quantize
        recorder = self

        def record_fake_quantize(x, scale, zero_point, args, **kwargs):
            current = recorder._current
            if current is not None:
                column = current["next_column"]
                if column >= current["columns"]:
                    raise RuntimeError(
                        f"GPTQ recorder saw too many columns for {current['name']}"
                    )
                current["effective_ordered"][:, column].copy_(x.detach().float())
                current["next_column"] = column + 1
            return recorder._original_fake_quantize(
                x, scale, zero_point, args, **kwargs
            )

        def record_quantize_weight(
            module, quant_args, hessian, blocksize=128, percdamp=0.01
        ):
            if recorder._current is not None:
                raise RuntimeError("Nested GPTQ quantize_weight calls are unsupported")
            name = recorder.modifier._module_names[module]
            original = module.weight.detach().float()
            rows, columns = original.shape
            actorder = quant_args.actorder
            if actorder:
                permutation = torch.argsort(torch.diag(hessian), descending=True)
            else:
                permutation = torch.arange(columns, device=original.device)

            strategy = quant_args.strategy
            if strategy not in (
                QuantizationStrategy.GROUP,
                QuantizationStrategy.TENSOR_GROUP,
            ):
                raise ValueError(
                    f"GPTQ boundary recorder currently requires group weights, got {strategy}"
                )
            group_index_ordered = (
                torch.arange(columns, device=original.device, dtype=torch.int64)
                // int(quant_args.group_size)
            )
            if actorder == ActivationOrdering.WEIGHT:
                group_index_ordered = group_index_ordered[permutation]

            recorder._current = {
                "name": name,
                "columns": columns,
                "next_column": 0,
                "effective_ordered": torch.empty_like(original),
            }
            try:
                result = recorder._original_quantize_weight(
                    module=module,
                    quant_args=quant_args,
                    hessian=hessian,
                    blocksize=blocksize,
                    percdamp=percdamp,
                )
                loss, q_param_dict = result
                current = recorder._current
                if current["next_column"] != columns:
                    raise RuntimeError(
                        f"GPTQ recorder captured {current['next_column']}/{columns} "
                        f"columns for {name}"
                    )

                inverse_permutation = torch.argsort(permutation)
                effective = current["effective_ordered"][:, inverse_permutation]
                group_index_original = group_index_ordered[inverse_permutation]
                recorder._save_layer(
                    name=name,
                    original=original,
                    effective=effective,
                    group_index_original=group_index_original,
                    quant_args=quant_args,
                    q_param_dict=q_param_dict,
                    permutation=permutation,
                    loss=loss,
                )
                return result
            finally:
                recorder._current = None

        gptq_quantize.fake_quantize = record_fake_quantize
        gptq_base.quantize_weight = record_quantize_weight
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        import llmcompressor.modifiers.gptq.base as gptq_base
        import llmcompressor.modifiers.gptq.gptq_quantize as gptq_quantize

        gptq_base.quantize_weight = self._original_quantize_weight
        gptq_quantize.fake_quantize = self._original_fake_quantize
        self._installed = False

    @torch.no_grad()
    def _save_layer(
        self,
        *,
        name: str,
        original: torch.Tensor,
        effective: torch.Tensor,
        group_index_original: torch.Tensor,
        quant_args,
        q_param_dict: dict[str, torch.Tensor],
        permutation: torch.Tensor,
        loss: float,
    ) -> None:
        rows, columns = original.shape
        scale = q_param_dict["weight_scale"].detach().float()
        zero_point = q_param_dict["weight_zero_point"].detach().float()
        expanded_scale = _expand_column_qparams(
            scale, group_index_original, rows
        ).to(device=effective.device)
        expanded_zero = _expand_column_qparams(
            zero_point, group_index_original, rows
        ).to(device=effective.device)
        q_min_tensor, q_max_tensor = calculate_range(quant_args, effective.device)
        q_min = int(q_min_tensor.item())
        q_max = int(q_max_tensor.item())
        codes = torch.round(
            (effective / expanded_scale + expanded_zero).clamp(q_min, q_max)
        ).to(torch.int8)

        quantized = q_param_dict["weight"].detach().float()
        reconstructed = (codes.float() - expanded_zero) * expanded_scale
        reconstruction_error = (reconstructed - quantized).abs()
        mismatch_count = int(
            (reconstruction_error > (1e-6 + 1e-5 * quantized.abs())).sum()
        )
        if mismatch_count:
            raise RuntimeError(
                f"Recorded GPTQ codes do not reconstruct {mismatch_count} values in {name}"
            )

        offset = effective - original
        code_float = codes.float()
        lower = (code_float - 0.5 - expanded_zero) * expanded_scale
        upper = (code_float + 0.5 - expanded_zero) * expanded_scale
        lower_distance = torch.where(
            code_float > q_min,
            effective - lower,
            torch.full_like(effective, torch.inf),
        )
        upper_distance = torch.where(
            code_float < q_max,
            upper - effective,
            torch.full_like(effective, torch.inf),
        )
        nearest = torch.minimum(lower_distance, upper_distance).clamp_min(0)
        finite = torch.isfinite(nearest)

        group_size = int(quant_args.group_size)
        group_count = math.ceil(columns / group_size)
        padded = F.pad(
            original.abs(), (0, group_count * group_size - columns), value=-1.0
        )
        group_absmax = padded.reshape(rows, group_count, group_size).amax(dim=-1)
        expanded_absmax = group_absmax.repeat_interleave(
            group_size, dim=-1
        )[..., :columns]
        anchors = torch.isclose(
            original.abs(), expanded_absmax, rtol=1e-6, atol=1e-8
        )
        candidate_counts = {}
        for threshold in RELATIVE_THRESHOLDS:
            candidates = finite & ~anchors
            candidates &= nearest <= threshold * original.abs()
            candidate_counts[str(threshold)] = int(candidates.sum())

        index = len(self.layer_reports)
        artifact_name = f"effective_weight_{index:04d}.safetensors"
        save_file(
            {
                # Store the effective value itself. Storing only effective-original
                # and reconstructing it later introduces one extra FP32 rounding,
                # which can flip values lying exactly on a half-step boundary.
                "effective_weight": effective.detach().cpu().contiguous(),
                "group_index_original": group_index_original.detach()
                .to(dtype=torch.int32, device="cpu")
                .contiguous(),
                "permutation": permutation.detach()
                .to(dtype=torch.int32, device="cpu")
                .contiguous(),
            },
            self.artifact_dir / artifact_name,
        )
        code_hash = _tensor_sha256(codes)
        self.expected_code_specs[name] = {
            "sha256": code_hash,
            "num_bits": int(quant_args.num_bits),
        }
        finite_values = nearest[finite]
        report = {
            "layer": name,
            "shape": [rows, columns],
            "parameters": original.numel(),
            "artifact": artifact_name,
            "num_bits": int(quant_args.num_bits),
            "group_size": group_size,
            "q_min": q_min,
            "q_max": q_max,
            "symmetric": bool(quant_args.symmetric),
            "actorder": str(quant_args.actorder),
            "gptq_loss": float(loss),
            "original_weight_sha256": _tensor_sha256(original),
            "effective_weight_sha256": _tensor_sha256(effective),
            "code_sha256": code_hash,
            "effective_offset_mean_absolute": float(offset.abs().mean()),
            "effective_offset_max_absolute": float(offset.abs().max()),
            "quantized_reconstruction_max_absolute_error": float(
                reconstruction_error.max()
            ),
            "anchor_parameters": int(anchors.sum()),
            "nearest_boundary_distance_mean": (
                float(finite_values.mean()) if finite_values.numel() else None
            ),
            "nearest_boundary_distance_min": (
                float(finite_values.min()) if finite_values.numel() else None
            ),
            "nearest_boundary_distance_max": (
                float(finite_values.max()) if finite_values.numel() else None
            ),
            "candidate_counts_excluding_anchors": candidate_counts,
        }
        self.layer_reports.append(report)
        print(
            f"GPTQ boundary recorded {index + 1}: {name} "
            f"({original.numel():,} weights)",
            flush=True,
        )

    def save_report(self, checkpoint: Path) -> dict[str, Any]:
        totals = {
            "layers": len(self.layer_reports),
            "parameters": sum(x["parameters"] for x in self.layer_reports),
            "anchor_parameters": sum(
                x["anchor_parameters"] for x in self.layer_reports
            ),
            "candidate_counts_excluding_anchors": {
                str(threshold): sum(
                    x["candidate_counts_excluding_anchors"][str(threshold)]
                    for x in self.layer_reports
                )
                for threshold in RELATIVE_THRESHOLDS
            },
        }
        totals["candidate_fractions_excluding_anchors"] = {
            key: value / max(1, totals["parameters"])
            for key, value in totals["candidate_counts_excluding_anchors"].items()
        }
        report = {
            "algorithm": "GPTQ",
            "boundary_coordinate_system": (
                "effective pre-round GPTQ weights, mapped to original FP32 columns"
            ),
            "boundary_mapping": (
                "original_boundary = original_weight + "
                "(effective_boundary - recorded_effective_weight)"
            ),
            "coupling_warning": (
                "The mapping is locally exact while earlier-column GPTQ codes remain "
                "fixed. Simultaneous earlier-column changes can shift later effective "
                "weights through Hessian error propagation; every Stage-2 result must "
                "therefore be re-quantized by standard GPTQ."
            ),
            "relative_candidate_rule": (
                "nearest_effective_boundary_distance <= threshold * "
                "abs(original_weight), excluding original per-group absmax anchors"
            ),
            "relative_thresholds": list(RELATIVE_THRESHOLDS),
            "totals": totals,
            "layers": self.layer_reports,
        }
        checkpoint = Path(checkpoint)
        packed = verify_packed_checkpoint(checkpoint, self.expected_code_specs)
        report["packed_checkpoint_verification"] = packed
        report_path = self.artifact_dir / "boundary_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return {
            "boundary_report": str(report_path),
            "effective_weight_shards": len(self.layer_reports),
            "all_packed_codes_matched": packed["all_layers_matched"],
            "layers_matched": packed["layers_matched"],
            "layers_checked": packed["layers_checked"],
            "totals": totals,
        }
