"""Q4_K support for GhostQuant's two-stage parameter control.

The selector in this module is deliberately independent of training gradients.
It ranks weights using only their distance to a local Q4_K decision boundary.
The post-training projector works blockwise because Q4_K jointly quantizes 256
weights and its scale/min optimization couples all selected weights in a block.
"""

from dataclasses import dataclass
from typing import Iterable

import torch
from tqdm import tqdm


QK_K = 256
Q4_K_SUBBLOCKS = 8
Q4_K_SUBBLOCK_SIZE = 32
GGUF_METHODS = {
    "q2_k",
    "q3_k_s", "q3_k_m", "q3_k_l",
    "q4_k", "q4_k_s", "q4_k_m",
    "q5_k_s", "q5_k_m",
    "q6_k",
}


def is_gguf_method(method: str | None) -> bool:
    return method is not None and method.lower() in GGUF_METHODS


def _language_layer_index(name: str) -> int | None:
    marker = ".language_model.layers."
    if marker not in name:
        return None
    suffix = name.split(marker, 1)[1]
    try:
        return int(suffix.split(".", 1)[0])
    except (TypeError, ValueError):
        return None


def _use_more_bits(layer: int, num_layers: int) -> bool:
    eighth = num_layers // 8
    return layer < eighth or layer >= 7 * eighth or (layer - eighth) % 3 == 2


def _num_language_layers(model) -> int:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    value = getattr(text_config, "num_hidden_layers", None)
    if value is not None:
        return int(value)
    indices = {
        index for name, _ in model.named_parameters()
        if (index := _language_layer_index(name)) is not None
    }
    return max(indices) + 1 if indices else 0


def is_q4_k_target(
    name: str,
    param: torch.Tensor,
    method: str,
    scope: str = "language",
    num_language_layers: int | None = None,
) -> bool:
    """Return tensors handled as Q4_K by the attack backend.

    Q4_K_S/M are mixed recipes. This backend conservatively excludes tensor
    families that llama.cpp promotes to Q5_K/Q6_K; the remaining projection
    matrices use the common Q4_K kernel in both recipes.
    """
    if (
        not name.endswith(".weight")
        or param.ndim != 2
        or param.shape[-1] % QK_K
    ):
        return False
    if "embed_tokens" in name or "lm_head" in name or "pos_embed" in name:
        return False
    is_language = ".language_model.layers." in name
    is_vision = ".visual." in name
    if scope == "language" and not is_language:
        return False
    if scope == "vision" and not is_vision:
        return False
    if scope == "all" and not (is_language or is_vision):
        return False
    if scope not in {"language", "vision", "all"}:
        raise ValueError(f"Unsupported gguf_target_scope: {scope}")

    method = method.lower()
    layer = _language_layer_index(name)
    if method == "q4_k_m" and (".v_proj.weight" in name or ".down_proj.weight" in name):
        if layer is None or not num_language_layers:
            return False
        return not _use_more_bits(layer, num_language_layers)
    if method == "q4_k_s":
        if ".v_proj.weight" in name and layer is not None and layer < 4:
            return False
        if (
            ".down_proj.weight" in name
            and layer is not None
            and num_language_layers
            and layer < num_language_layers // 8
        ):
            return False
    return True


@dataclass
class Q4KState:
    d: torch.Tensor
    dmin: torch.Tensor
    scales: torch.Tensor
    mins: torch.Tensor
    codes: torch.Tensor

    def index(self, index: torch.Tensor) -> "Q4KState":
        return Q4KState(
            self.d[index], self.dmin[index], self.scales[index],
            self.mins[index], self.codes[index]
        )

    def repeat(self, repeats: int) -> "Q4KState":
        return Q4KState(
            self.d.repeat(repeats),
            self.dmin.repeat(repeats),
            self.scales.repeat(repeats, 1),
            self.mins.repeat(repeats, 1),
            self.codes.repeat(repeats, 1, 1),
        )


@dataclass
class Q4KMetadata:
    d: torch.Tensor
    dmin: torch.Tensor
    scales: torch.Tensor
    mins: torch.Tensor

    def slice(self, start: int, end: int, device: torch.device) -> "Q4KMetadata":
        return Q4KMetadata(
            self.d[start:end].to(device),
            self.dmin[start:end].to(device),
            self.scales[start:end].to(device),
            self.mins[start:end].to(device),
        )


def q4_k_metadata_equal(state: Q4KState, metadata: Q4KMetadata | Q4KState) -> torch.Tensor:
    return (
        (state.d == metadata.d)
        & (state.dmin == metadata.dmin)
        & (state.scales == metadata.scales).all(dim=1)
        & (state.mins == metadata.mins).all(dim=1)
    )


def q4_k_state_equal(left: Q4KState, right: Q4KState) -> torch.Tensor:
    """Compare the complete logical packed state of each Q4_K superblock."""
    return (
        (left.d == right.d)
        & (left.dmin == right.dmin)
        & (left.scales == right.scales).all(dim=1)
        & (left.mins == right.mins).all(dim=1)
        & (left.codes == right.codes).all(dim=(1, 2))
    )


@torch.no_grad()
def quantize_q4_k(blocks: torch.Tensor) -> Q4KState:
    """Emulate llama.cpp b3612 quantize_row_q4_K_ref on [N, 8, 32]."""
    if blocks.ndim != 3 or blocks.shape[1:] != (Q4_K_SUBBLOCKS, Q4_K_SUBBLOCK_SIZE):
        raise ValueError(f"Expected [N, 8, 32] Q4_K blocks, got {tuple(blocks.shape)}")

    # GGUF conversion quantizes F16 source values after conversion to float.
    x = blocks.to(torch.float16).to(torch.float32)
    nmax = torch.tensor(15, device=x.device)
    rdelta = torch.tensor(0.1, device=x.device)
    sum_x2 = torch.zeros(x.shape[:2], device=x.device)
    for index in range(Q4_K_SUBBLOCK_SIZE):
        sum_x2 += x[:, :, index] * x[:, :, index]
    average_x = torch.sqrt(sum_x2 / Q4_K_SUBBLOCK_SIZE)
    weights = torch.empty_like(x)
    for index in range(Q4_K_SUBBLOCK_SIZE):
        weights[:, :, index] = average_x + x[:, :, index].abs()
    sum_w = torch.zeros(x.shape[:2], device=x.device)
    sum_x = torch.zeros_like(sum_w)
    for index in range(Q4_K_SUBBLOCK_SIZE):
        sum_w += weights[:, :, index]
        sum_x += weights[:, :, index] * x[:, :, index]
    minimum = x.min(dim=-1).values.clamp_max(0)
    maximum = x.max(dim=-1).values
    zero = maximum == minimum

    inv_scale = nmax / (maximum - minimum)
    scale = 1.0 / inv_scale
    codes = torch.empty_like(x)
    best_error = torch.zeros(x.shape[:2], device=x.device)
    for index in range(Q4_K_SUBBLOCK_SIZE):
        code = torch.round(inv_scale * (x[:, :, index] - minimum)).clamp_(0, nmax)
        codes[:, :, index] = code
        error = scale * code + minimum - x[:, :, index]
        best_error += weights[:, :, index] * error.square()

    for step in range(21):
        inv = (-1.0 + rdelta * step + nmax) / (maximum - minimum)
        candidate_codes = torch.empty_like(x)
        sum_l = torch.zeros(x.shape[:2], device=x.device)
        sum_l2 = torch.zeros_like(sum_l)
        sum_xl = torch.zeros_like(sum_l)
        for index in range(Q4_K_SUBBLOCK_SIZE):
            code = torch.round(inv * (x[:, :, index] - minimum)).clamp_(0, nmax)
            candidate_codes[:, :, index] = code
            sum_l += weights[:, :, index] * code
            sum_l2 += weights[:, :, index] * code * code
            sum_xl += weights[:, :, index] * code * x[:, :, index]
        denominator = sum_w * sum_l2 - sum_l.square()
        candidate_scale = (sum_w * sum_xl - sum_x * sum_l) / denominator
        candidate_min = (sum_l2 * sum_x - sum_l * sum_xl) / denominator
        positive_min = candidate_min > 0
        candidate_min = torch.where(positive_min, torch.zeros_like(candidate_min), candidate_min)
        candidate_scale = torch.where(positive_min, sum_xl / sum_l2, candidate_scale)
        candidate_error = torch.zeros(x.shape[:2], device=x.device)
        for index in range(Q4_K_SUBBLOCK_SIZE):
            error = candidate_scale * candidate_codes[:, :, index] + candidate_min - x[:, :, index]
            candidate_error += weights[:, :, index] * error.square()
        better = (denominator > 0) & (candidate_error < best_error)
        for index in range(Q4_K_SUBBLOCK_SIZE):
            codes[:, :, index] = torch.where(better, candidate_codes[:, :, index], codes[:, :, index])
        best_error = torch.where(better, candidate_error, best_error)
        scale = torch.where(better, candidate_scale, scale)
        minimum = torch.where(better, candidate_min, minimum)

    scale = torch.where(zero, torch.zeros_like(scale), scale)
    minimum = torch.where(zero, torch.zeros_like(minimum), minimum)
    codes = torch.where(zero.unsqueeze(-1), torch.zeros_like(codes), codes)
    positive_offset = -minimum

    max_scale, argmax_scale = scale.max(dim=1)
    max_min, argmax_min = positive_offset.max(dim=1)
    inv_scale2 = torch.where(max_scale > 0, 63.0 / max_scale, torch.zeros_like(max_scale))
    inv_min2 = torch.where(max_min > 0, 63.0 / max_min, torch.zeros_like(max_min))
    scales = torch.round(inv_scale2.unsqueeze(1) * scale).clamp_(0, 63).to(torch.uint8)
    mins = torch.round(inv_min2.unsqueeze(1) * positive_offset).clamp_(0, 63).to(torch.uint8)
    d = (max_scale / 63.0).to(torch.float16)
    dmin = (max_min / 63.0).to(torch.float16)

    final_scale = d.float().unsqueeze(1) * scales.float()
    final_min = dmin.float().unsqueeze(1) * mins.float()
    safe_scale = torch.where(final_scale != 0, final_scale, torch.ones_like(final_scale))
    final_codes = codes.clone()
    for index in range(Q4_K_SUBBLOCK_SIZE):
        code = torch.round((x[:, :, index] + final_min) / safe_scale).clamp_(0, 15)
        final_codes[:, :, index] = torch.where(final_scale != 0, code, final_codes[:, :, index])

    state = Q4KState(d, dmin, scales, mins, final_codes.to(torch.uint8))
    # These are useful to the selector but are not serialized in a Q4_K block.
    state.argmax_scale = argmax_scale
    state.argmax_min = argmax_min
    state.raw_scale = scale
    state.raw_min = positive_offset
    return state


def dequantize_q4_k(state: Q4KState, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    scale = state.d.float().unsqueeze(1) * state.scales.float()
    minimum = state.dmin.float().unsqueeze(1) * state.mins.float()
    return (scale.unsqueeze(-1) * state.codes.float() - minimum.unsqueeze(-1)).to(dtype)


def _iter_chunks(total: int, chunk_blocks: int) -> Iterable[tuple[int, int]]:
    for start in range(0, total, chunk_blocks):
        yield start, min(start + chunk_blocks, total)


@torch.no_grad()
def build_q4_k_finetune_boxes(
    model,
    original_model,
    method: str,
    selection_fraction: float = 0.10,
    candidate_relative_limit: float | None = None,
    candidate_absolute_limit: float | None = None,
    boundary_penetration: float = 0.05,
    scope: str = "language",
    chunk_blocks: int = 4096,
    metadata_mode: str = "free",
):
    """Select boundary-near weights and construct first-stage PGD boxes.

    Selection uses no dataset, gradients, or fine-tuning trajectory. It either
    takes a fixed fraction per tensor or selects every value satisfying both a
    relative and an absolute distance limit. Scale/min anchor subblocks and exact
    subblock extrema remain locked.
    """
    threshold_selection = (
        candidate_relative_limit is not None or candidate_absolute_limit is not None
    )
    if threshold_selection and (
        candidate_relative_limit is None or candidate_absolute_limit is None
    ):
        raise ValueError(
            "gguf_candidate_relative_limit and gguf_candidate_absolute_limit "
            "must be specified together"
        )
    if threshold_selection and (
        candidate_relative_limit <= 0 or candidate_absolute_limit <= 0
    ):
        raise ValueError("Q4_K candidate distance limits must be positive")
    if not 0 <= boundary_penetration <= 1:
        raise ValueError("boundary_penetration must be between 0 and 1")
    if not threshold_selection and not 0 < selection_fraction < 1:
        raise ValueError("gguf_selection_fraction must be between 0 and 1")
    if metadata_mode not in {"free", "code_only"}:
        raise ValueError(f"Unsupported gguf_metadata_mode: {metadata_mode}")
    params = dict(model.named_parameters())
    original_params = dict(original_model.named_parameters())
    boxes = {}
    indices = {}
    total = 0
    selected = 0
    totals_by_scope = {"language": 0, "vision": 0}
    selected_by_scope = {"language": 0, "vision": 0}
    metadata_references = {}
    safe_weights = {}
    init_metadata_resets = 0

    num_layers = _num_language_layers(model)
    names = [
        n for n, p in params.items()
        if n in original_params and is_q4_k_target(n, p, method, scope, num_layers)
    ]
    for name in tqdm(names, desc="Q4_K boundary selection"):
        param = params[name]
        original = original_params[name].detach().to(param.device)
        current = param.detach().clone()
        flat = current.reshape(-1)
        original_flat = original.reshape(-1)
        nblocks = flat.numel() // QK_K

        dequantized = torch.empty_like(flat)
        lower = torch.empty(flat.numel(), device=flat.device, dtype=torch.float32)
        upper = torch.empty_like(lower)
        quant_step = torch.empty_like(lower)
        locked = torch.zeros(flat.numel(), device=flat.device, dtype=torch.bool)
        range_low = torch.empty_like(flat)
        range_high = torch.empty_like(flat)
        if metadata_mode == "code_only":
            ref_d = torch.empty(nblocks, dtype=torch.float16)
            ref_dmin = torch.empty(nblocks, dtype=torch.float16)
            ref_scales = torch.empty(nblocks, 8, dtype=torch.uint8)
            ref_mins = torch.empty(nblocks, 8, dtype=torch.uint8)

        for start, end in _iter_chunks(nblocks, chunk_blocks):
            block = flat[start * QK_K:end * QK_K].reshape(-1, 8, 32)
            state = quantize_q4_k(block)
            deq = dequantize_q4_k(state, current.dtype).reshape(-1)
            dequantized[start * QK_K:end * QK_K] = deq

            scale = state.d.float().unsqueeze(1) * state.scales.float()
            minimum = state.dmin.float().unsqueeze(1) * state.mins.float()
            code = state.codes.float()
            lo = (code - 0.5) * scale.unsqueeze(-1) - minimum.unsqueeze(-1)
            hi = (code + 0.5) * scale.unsqueeze(-1) - minimum.unsqueeze(-1)
            lo[state.codes == 0] = -torch.inf
            hi[state.codes == 15] = torch.inf
            lower[start * QK_K:end * QK_K] = lo.reshape(-1)
            upper[start * QK_K:end * QK_K] = hi.reshape(-1)
            quant_step[start * QK_K:end * QK_K] = (
                scale.unsqueeze(-1).expand_as(block).reshape(-1)
            )

            local_lock = torch.zeros_like(block, dtype=torch.bool)
            rows = torch.arange(end - start, device=block.device)
            local_lock[rows, state.argmax_scale, :] = True
            local_lock[rows, state.argmax_min, :] = True
            local_lock.scatter_(2, block.argmax(dim=2, keepdim=True), True)
            local_lock.scatter_(2, block.argmin(dim=2, keepdim=True), True)
            locked[start * QK_K:end * QK_K] = local_lock.reshape(-1)
            sub_min = block.min(dim=2, keepdim=True).values.expand_as(block)
            sub_max = block.max(dim=2, keepdim=True).values.expand_as(block)
            range_low[start * QK_K:end * QK_K] = sub_min.reshape(-1)
            range_high[start * QK_K:end * QK_K] = sub_max.reshape(-1)
            if metadata_mode == "code_only":
                ref_d[start:end] = state.d.cpu()
                ref_dmin[start:end] = state.dmin.cpu()
                ref_scales[start:end] = state.scales.cpu()
                ref_mins[start:end] = state.mins.cpu()

        changed = flat != original_flat
        # The Q4_K state is produced from the converter's FP16 input, but the
        # trainable parameter and its movement budget are both FP32. Measure
        # boundary distance from that FP32 value so the box is a true hard cap
        # on |trained - original|.
        quantizer_values = flat.float()
        lower_distance = (quantizer_values - lower).abs()
        upper_distance = (upper - quantizer_values).abs()
        absolute_distance = torch.minimum(lower_distance, upper_distance)
        valid = ~locked & ~changed
        chosen = torch.zeros_like(locked)
        lower_eligible = torch.zeros_like(locked)
        upper_eligible = torch.zeros_like(locked)
        if threshold_selection:
            budget = torch.minimum(
                candidate_relative_limit * quantizer_values.abs(),
                torch.full_like(quantizer_values, candidate_absolute_limit),
            )
            lower_eligible = (
                valid
                & torch.isfinite(lower_distance)
                & (lower_distance <= budget)
            )
            upper_eligible = (
                valid
                & torch.isfinite(upper_distance)
                & (upper_distance <= budget)
            )
            chosen = lower_eligible | upper_eligible
        else:
            denominator = quantizer_values.abs().clamp_min(1e-12)
            normalized_distance = absolute_distance / denominator
            normalized_distance[~valid | ~torch.isfinite(absolute_distance)] = torch.inf
            k = min(
                int(round(flat.numel() * selection_fraction)),
                int(torch.isfinite(normalized_distance).sum().item()),
            )
            if k:
                chosen[
                    torch.topk(
                        normalized_distance, k, largest=False, sorted=False
                    ).indices
                ] = True

        # Match the existing GhostQuant first stage: insensitive weights use their
        # dequantized values; sensitive and shared-scale anchor weights stay exact.
        initialized = dequantized.clone()
        initialized[chosen | locked | changed] = flat[chosen | locked | changed]

        if metadata_mode == "code_only":
            reference = Q4KMetadata(ref_d, ref_dmin, ref_scales, ref_mins)
            initialized_blocks = initialized.reshape(-1, 8, 32)
            original_blocks = flat.reshape(-1, 8, 32)
            for start, end in _iter_chunks(nblocks, chunk_blocks):
                state = quantize_q4_k(initialized_blocks[start:end])
                expected = reference.slice(start, end, state.d.device)
                unstable = ~q4_k_metadata_equal(state, expected)
                if unstable.any():
                    initialized_blocks[start:end][unstable] = original_blocks[start:end][unstable]
                    init_metadata_resets += int(unstable.sum().item())
            metadata_references[name] = reference

        box_min = initialized.clone()
        box_max = initialized.clone()
        if threshold_selection:
            # A candidate may move only toward a boundary that independently
            # satisfies both limits. The total move never exceeds its budget.
            lower_margin = torch.minimum(
                boundary_penetration * quant_step,
                (budget - lower_distance).clamp_min(0),
            )
            upper_margin = torch.minimum(
                boundary_penetration * quant_step,
                (budget - upper_distance).clamp_min(0),
            )
            lower_target = lower - lower_margin
            upper_target = upper + upper_margin
            box_min[lower_eligible] = torch.minimum(
                lower_target[lower_eligible].to(current.dtype),
                flat[lower_eligible],
            )
            box_max[upper_eligible] = torch.maximum(
                upper_target[upper_eligible].to(current.dtype),
                flat[upper_eligible],
            )
        else:
            # Retain the legacy fraction selector's bidirectional boxes.
            down_center = torch.where(
                torch.isfinite(lower),
                2.0 * lower - dequantized.float(),
                dequantized.float(),
            )
            up_center = torch.where(
                torch.isfinite(upper),
                2.0 * upper - dequantized.float(),
                dequantized.float(),
            )
            box_min[chosen] = torch.minimum(
                down_center[chosen].to(current.dtype), flat[chosen]
            )
            box_max[chosen] = torch.maximum(
                up_center[chosen].to(current.dtype), flat[chosen]
            )
        # Near-extrema and endpoint codes remain trainable, but no selected value
        # may leave its original 32-value subblock range.
        box_min[chosen] = torch.maximum(box_min[chosen], range_low[chosen])
        box_max[chosen] = torch.minimum(box_max[chosen], range_high[chosen])
        box_min[chosen] = torch.minimum(box_min[chosen], flat[chosen])
        box_max[chosen] = torch.maximum(box_max[chosen], flat[chosen])
        if metadata_mode == "code_only":
            box_min = torch.minimum(box_min, initialized)
            box_max = torch.maximum(box_max, initialized)

        param.data.copy_(initialized.reshape_as(param))

        boxes[name] = [box_min.reshape_as(param).cpu(), box_max.reshape_as(param).cpu()]
        index = chosen.nonzero(as_tuple=False).squeeze(1).cpu()
        indices[name] = index
        if metadata_mode == "code_only":
            safe_weights[name] = initialized.reshape_as(param).cpu().clone()
        total += flat.numel()
        selected += index.numel()
        parameter_scope = "vision" if ".visual." in name else "language"
        totals_by_scope[parameter_scope] += flat.numel()
        selected_by_scope[parameter_scope] += index.numel()

    print(f"Q4_K target tensors: {len(names)}")
    print(f"Q4_K parameters scanned: {total:,}")
    if threshold_selection:
        print(
            "Q4_K selection limits: "
            f"relative <= {candidate_relative_limit:g}, "
            f"absolute <= {candidate_absolute_limit:g}"
        )
    print(f"Q4_K sensitive parameters: {selected:,} ({100.0 * selected / max(1, total):.4f}%)")
    for parameter_scope in ("language", "vision"):
        scope_total = totals_by_scope[parameter_scope]
        if scope_total:
            scope_selected = selected_by_scope[parameter_scope]
            print(
                f"Q4_K {parameter_scope} sensitive parameters: "
                f"{scope_selected:,} "
                f"({100.0 * scope_selected / scope_total:.4f}%)"
            )
    context = None
    if metadata_mode == "code_only":
        context = {
            "references": metadata_references,
            "safe_weights": safe_weights,
            "init_metadata_resets": init_metadata_resets,
        }
        print(f"Q4_K code-only initialization resets: {init_metadata_resets:,} blocks")
    return indices, boxes, context


@torch.no_grad()
def enforce_q4_k_metadata_(model, context, chunk_blocks: int = 4096):
    """Reset blocks whose Q4_K scale/min metadata differs from its baseline."""
    if context is None:
        return {"checked_blocks": 0, "reset_blocks": 0, "verification_failures": 0}
    params = dict(model.named_parameters())
    stats = {"checked_blocks": 0, "reset_blocks": 0, "verification_failures": 0}
    for name, reference in context["references"].items():
        parameter = params[name]
        blocks = parameter.data.reshape(-1, 8, 32)
        safe_blocks = context["safe_weights"][name].reshape(-1, 8, 32)
        for start, end in _iter_chunks(blocks.shape[0], chunk_blocks):
            current = blocks[start:end]
            state = quantize_q4_k(current)
            expected = reference.slice(start, end, current.device)
            unstable = ~q4_k_metadata_equal(state, expected)
            stats["checked_blocks"] += end - start
            if unstable.any():
                replacement = safe_blocks[start:end][unstable.cpu()].to(
                    device=current.device, dtype=current.dtype
                )
                current[unstable] = replacement
                stats["reset_blocks"] += int(unstable.sum().item())
                repaired = quantize_q4_k(current[unstable])
                repaired_expected = Q4KMetadata(
                    expected.d[unstable], expected.dmin[unstable],
                    expected.scales[unstable], expected.mins[unstable],
                )
                stats["verification_failures"] += int(
                    (~q4_k_metadata_equal(repaired, repaired_expected)).sum().item()
                )
    return stats


@torch.no_grad()
def project_base_to_q4_k_target(
    base_model,
    finetuned_model,
    method: str,
    scope: str = "language",
    coarse_steps: int = 32,
    bisection_steps: int = 10,
    penetration: float = 0.05,
    chunk_blocks: int = 1024,
    compute_device: str | None = None,
    require_stable_metadata: bool = False,
    max_relative_change: float | None = None,
    max_absolute_change: float | None = None,
):
    """Stage two: minimally enter FT Q4_K states along learned block directions."""
    base_params = dict(base_model.named_parameters())
    ft_params = dict(finetuned_model.named_parameters())
    if compute_device is None:
        compute_device = "cuda" if torch.cuda.is_available() else "cpu"
    work_device = torch.device(compute_device)
    bounded = max_relative_change is not None or max_absolute_change is not None
    if bounded and (max_relative_change is None or max_absolute_change is None):
        raise ValueError(
            "max_relative_change and max_absolute_change must be specified together"
        )
    if bounded and (max_relative_change <= 0 or max_absolute_change <= 0):
        raise ValueError("Q4_K stage-two perturbation limits must be positive")

    def within_budget(candidate: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        if not bounded:
            return torch.ones(candidate.shape[0], device=candidate.device, dtype=torch.bool)
        budget = torch.minimum(
            max_relative_change * original.abs(),
            torch.full_like(original, max_absolute_change),
        )
        return ((candidate - original).abs() <= budget + 1e-12).all(dim=(1, 2))

    stats = {
        "blocks": 0,
        "changed_blocks": 0,
        "projected_blocks": 0,
        "fallback_blocks": 0,
        "full_ft_blocks": 0,
        "failed_blocks": 0,
        "metadata_rejected_blocks": 0,
        "budget_rejected_blocks": 0,
    }

    num_layers = _num_language_layers(base_model)
    names = [
        n for n, p in base_params.items()
        if n in ft_params and is_q4_k_target(n, p, method, scope, num_layers)
    ]
    for name in tqdm(names, desc="Q4_K block projection"):
        base_param = base_params[name]
        ft_param = ft_params[name]
        base_flat = base_param.data.reshape(-1)
        ft_flat = ft_param.data.to(base_param.device).reshape(-1)
        nblocks = base_flat.numel() // QK_K
        stats["blocks"] += nblocks

        for start, end in _iter_chunks(nblocks, chunk_blocks):
            base = base_flat[start * QK_K:end * QK_K].reshape(-1, 8, 32).to(work_device).float()
            ft = ft_flat[start * QK_K:end * QK_K].reshape(-1, 8, 32).to(work_device).float()
            base_state = quantize_q4_k(base)
            target_state = quantize_q4_k(ft)
            metadata_stable = q4_k_metadata_equal(target_state, base_state)
            if require_stable_metadata:
                stats["metadata_rejected_blocks"] += int((~metadata_stable).sum().item())
            else:
                metadata_stable = torch.ones_like(metadata_stable)
            changed = (~q4_k_state_equal(base_state, target_state)) & metadata_stable
            if not changed.any():
                continue

            block_ids = changed.nonzero(as_tuple=False).squeeze(1)
            b = base[block_ids]
            delta = ft[block_ids] - b
            target = target_state.index(block_ids)
            count = block_ids.numel()
            stats["changed_blocks"] += count

            # Locate the target-state component that contains alpha=1. Scanning
            # backward first is necessary because Q4_K states are not monotonic.
            alphas = torch.linspace(
                1.0 - 1.0 / coarse_steps, 0.0, coarse_steps, device=b.device
            )
            candidates = b.unsqueeze(0) + alphas.view(-1, 1, 1, 1) * delta.unsqueeze(0)
            scan_state = quantize_q4_k(candidates.reshape(-1, 8, 32))
            matches_grid = q4_k_state_equal(scan_state, target.repeat(coarse_steps)).reshape(coarse_steps, count)
            left_grid = ~matches_grid
            found = left_grid.any(dim=0)
            first_left = left_grid.to(torch.int8).argmax(dim=0)
            lo = alphas[first_left]
            previous = (first_left - 1).clamp_min(0)
            hi = torch.where(
                first_left == 0,
                torch.ones(count, device=b.device),
                alphas[previous],
            )
            del candidates, scan_state, matches_grid, left_grid

            # base and FT states differ, so every path should leave the target
            # component by alpha=0. Keep failures explicit if numerical issues occur.
            active = found
            for _ in range(bisection_steps):
                mid = (lo + hi) * 0.5
                candidate = b + mid.view(-1, 1, 1) * delta
                matches = q4_k_state_equal(quantize_q4_k(candidate), target)
                hi = torch.where(active & matches, mid, hi)
                lo = torch.where(active & ~matches, mid, lo)

            alpha = hi + penetration * (1.0 - hi)
            projected = b + alpha.view(-1, 1, 1) * delta
            state_verified = q4_k_state_equal(quantize_q4_k(projected), target) & active
            budget_verified = within_budget(projected, b)
            stats["budget_rejected_blocks"] += int(
                (state_verified & ~budget_verified).sum().item()
            )
            verified = state_verified & budget_verified

            # A Q4_K state can leave and re-enter along the same update ray. If
            # the first bracket landed in a disconnected component, walk from
            # that point toward the guaranteed target at alpha=1 and refine the
            # first verified component. This uses no gradients.
            needs_fallback = ~state_verified
            if needs_fallback.any():
                fallback_ids = needs_fallback.nonzero(as_tuple=False).squeeze(1)
                fb_b = b[fallback_ids]
                fb_delta = delta[fallback_ids]
                fb_target = target.index(fallback_ids)
                fb_start = alpha[fallback_ids]
                fallback_steps = max(16, coarse_steps)
                fractions = torch.linspace(
                    0.0, 1.0, fallback_steps + 1, device=b.device
                )[1:]
                fb_alphas = fb_start.unsqueeze(0) + fractions.unsqueeze(1) * (1.0 - fb_start).unsqueeze(0)
                fb_candidates = (
                    fb_b.unsqueeze(0)
                    + fb_alphas.unsqueeze(-1).unsqueeze(-1) * fb_delta.unsqueeze(0)
                )
                fb_states = quantize_q4_k(fb_candidates.reshape(-1, 8, 32))
                fb_matches = q4_k_state_equal(
                    fb_states, fb_target.repeat(fallback_steps)
                ).reshape(fallback_steps, -1)
                has_target = fb_matches.any(dim=0)
                first_target = fb_matches.to(torch.int8).argmax(dim=0)
                fb_hi = fb_alphas[first_target, torch.arange(fallback_ids.numel(), device=b.device)]
                previous_target = (first_target - 1).clamp_min(0)
                fb_lo = torch.where(
                    first_target == 0,
                    fb_start,
                    fb_alphas[previous_target, torch.arange(fallback_ids.numel(), device=b.device)],
                )
                for _ in range(bisection_steps):
                    fb_mid = (fb_lo + fb_hi) * 0.5
                    fb_candidate = fb_b + fb_mid.view(-1, 1, 1) * fb_delta
                    fb_is_target = q4_k_state_equal(quantize_q4_k(fb_candidate), fb_target)
                    fb_hi = torch.where(fb_is_target, fb_mid, fb_hi)
                    fb_lo = torch.where(fb_is_target, fb_lo, fb_mid)
                fb_alpha = fb_hi + penetration * (1.0 - fb_hi)
                fb_projected = fb_b + fb_alpha.view(-1, 1, 1) * fb_delta
                fb_state_verified = (
                    q4_k_state_equal(quantize_q4_k(fb_projected), fb_target)
                    & has_target
                )
                fb_budget_verified = within_budget(fb_projected, fb_b)
                stats["budget_rejected_blocks"] += int(
                    (fb_state_verified & ~fb_budget_verified).sum().item()
                )
                fb_verified = fb_state_verified & fb_budget_verified
                # Never fall back to the complete FT block: it may exceed the
                # explicit per-weight perturbation budget by orders of magnitude.
                stats["fallback_blocks"] += int(fb_verified.sum().item())
                verified[fallback_ids] = fb_verified
                projected[fallback_ids] = fb_projected

            stats["projected_blocks"] += int(verified.sum().item())
            stats["failed_blocks"] += int((~verified).sum().item())
            if verified.any():
                output = base.clone()
                output_ids = block_ids[verified]
                output[output_ids] = projected[verified]
                base_flat[start * QK_K:end * QK_K].copy_(
                    output.reshape(-1).to(device=base_param.device, dtype=base_param.dtype)
                )

    print("Q4_K projection statistics:", stats)
    return stats


@torch.no_grad()
def nudge_base_across_q4_k_boundaries(
    base_model,
    finetuned_model,
    method: str,
    scope: str = "language",
    penetration: float = 0.05,
    chunk_blocks: int = 4096,
    compute_device: str | None = None,
    max_relative_change: float | None = None,
    max_absolute_change: float | None = None,
):
    """Stage two: copy only Stage-one boundary directions into the base model.

    Stage one is allowed to change its FP32 weights substantially enough to
    learn the target behavior. Stage two must retain the base model in FP32,
    so it uses the stage-one weights only as a direction oracle: a selected
    parameter is nudged just across its *original* Q4_K boundary. The complete
    Stage-one Q4_K block state is intentionally not copied because its
    dequantized non-candidate values are not part of the sparse perturbation.
    """
    if compute_device is None:
        compute_device = "cuda" if torch.cuda.is_available() else "cpu"
    if max_relative_change is None or max_absolute_change is None:
        raise ValueError("Stage-two Q4_K nudging requires both perturbation limits")
    if max_relative_change <= 0 or max_absolute_change <= 0:
        raise ValueError("Q4_K stage-two perturbation limits must be positive")
    if not 0 <= penetration <= 1:
        raise ValueError("Q4_K boundary penetration must be between 0 and 1")

    work_device = torch.device(compute_device)
    base_params = dict(base_model.named_parameters())
    ft_params = dict(finetuned_model.named_parameters())
    num_layers = _num_language_layers(base_model)
    names = [
        name for name, parameter in base_params.items()
        if name in ft_params
        and is_q4_k_target(name, parameter, method, scope, num_layers)
    ]
    stats = {
        "tensors": len(names),
        "blocks": 0,
        "candidate_parameters": 0,
        "stage1_crossing_parameters": 0,
        "changed_parameters": 0,
        "actual_target_crossings": 0,
        "failed_target_crossings": 0,
        "metadata_changed_blocks": 0,
        "collateral_code_changes": 0,
        "wrong_direction": 0,
        "budget_violations": 0,
        "subblock_range_violations": 0,
        "absolute_delta_sum": 0.0,
        "max_absolute_delta": 0.0,
    }

    for name in tqdm(names, desc="Q4_K sparse boundary nudging"):
        base_param = base_params[name]
        ft_param = ft_params[name]
        base_flat = base_param.data.reshape(-1)
        ft_flat = ft_param.data.reshape(-1)
        nblocks = base_flat.numel() // QK_K
        stats["blocks"] += nblocks

        for start, end in _iter_chunks(nblocks, chunk_blocks):
            base = base_flat[start * QK_K:end * QK_K].reshape(-1, 8, 32).to(work_device).float()
            ft = ft_flat[start * QK_K:end * QK_K].reshape(-1, 8, 32).to(work_device).float()
            state = quantize_q4_k(base)
            scale = state.d.float().unsqueeze(1) * state.scales.float()
            minimum = state.dmin.float().unsqueeze(1) * state.mins.float()
            codes = state.codes
            safe_scale = torch.where(scale != 0, scale, torch.ones_like(scale))

            lower = (codes.float() - 0.5) * scale.unsqueeze(-1) - minimum.unsqueeze(-1)
            upper = (codes.float() + 0.5) * scale.unsqueeze(-1) - minimum.unsqueeze(-1)
            lower[codes == 0] = -torch.inf
            upper[codes == 15] = torch.inf
            lower_distance = (base - lower).abs()
            upper_distance = (upper - base).abs()
            budget = torch.minimum(
                max_relative_change * base.abs(),
                torch.full_like(base, max_absolute_change),
            )

            locked = torch.zeros_like(base, dtype=torch.bool)
            rows = torch.arange(end - start, device=work_device)
            locked[rows, state.argmax_scale, :] = True
            locked[rows, state.argmax_min, :] = True
            locked.scatter_(2, base.argmax(dim=2, keepdim=True), True)
            locked.scatter_(2, base.argmin(dim=2, keepdim=True), True)
            lower_eligible = (
                ~locked & torch.isfinite(lower_distance) & (lower_distance <= budget)
            )
            upper_eligible = (
                ~locked & torch.isfinite(upper_distance) & (upper_distance <= budget)
            )
            selected = lower_eligible | upper_eligible
            stats["candidate_parameters"] += int(selected.sum().item())

            # Reconstruct the fixed original metadata code that Stage one was
            # trained to cross. This is only a direction oracle; it is not a
            # request to reproduce the complete Stage-one block state.
            fixed_codes = torch.round(
                (ft.to(torch.float16).float() + minimum.unsqueeze(-1))
                / safe_scale.unsqueeze(-1)
            ).clamp_(0, 15).to(torch.uint8)
            direction = fixed_codes.to(torch.int16) - codes.to(torch.int16)
            crossing = selected & (direction != 0)
            stats["stage1_crossing_parameters"] += int(crossing.sum().item())
            if not crossing.any():
                continue

            positive = direction > 0
            distance = torch.where(positive, upper_distance, lower_distance)
            quant_step = scale.unsqueeze(-1)
            remaining = (budget - distance).clamp_min(0)
            margin = torch.minimum(penetration * quant_step, remaining)
            boundary = torch.where(positive, upper, lower)
            target = torch.where(
                positive,
                boundary + margin,
                boundary - margin,
            )
            # A zero margin can land exactly on a round-to-even tie. Move by
            # one representable FP32 value while retaining the hard budget.
            target = torch.where(
                positive & (margin == 0),
                torch.nextafter(boundary, torch.full_like(boundary, torch.inf)),
                target,
            )
            target = torch.where(
                (~positive) & (margin == 0),
                torch.nextafter(boundary, torch.full_like(boundary, -torch.inf)),
                target,
            )

            sub_min = base.min(dim=2, keepdim=True).values
            sub_max = base.max(dim=2, keepdim=True).values
            valid = crossing & torch.isfinite(target)
            valid &= target >= sub_min
            valid &= target <= sub_max
            valid &= (target - base).abs() <= budget + 1e-7
            stats["budget_violations"] += int(
                (crossing & ((target - base).abs() > budget + 1e-7)).sum().item()
            )
            stats["subblock_range_violations"] += int(
                (crossing & ((target < sub_min) | (target > sub_max))).sum().item()
            )

            proposed = base.clone()
            proposed[valid] = target[valid]
            proposed_state = quantize_q4_k(proposed)
            code_delta = proposed_state.codes.to(torch.int16) - state.codes.to(torch.int16)
            actual_target = valid & ((code_delta * direction) > 0)
            failed_target = crossing & ~actual_target
            stats["actual_target_crossings"] += int(actual_target.sum().item())
            stats["failed_target_crossings"] += int(failed_target.sum().item())
            stats["wrong_direction"] += int(
                (valid & (code_delta * direction < 0)).sum().item()
            )
            stats["metadata_changed_blocks"] += int(
                (~q4_k_metadata_equal(proposed_state, state)).sum().item()
            )
            stats["collateral_code_changes"] += int(
                ((~valid) & (proposed_state.codes != state.codes)).sum().item()
            )

            changed = proposed != base
            stats["changed_parameters"] += int(changed.sum().item())
            if changed.any():
                delta = (proposed - base).abs()
                stats["absolute_delta_sum"] += float(delta[changed].sum().item())
                stats["max_absolute_delta"] = max(
                    stats["max_absolute_delta"], float(delta.max().item())
                )
                base_flat[start * QK_K:end * QK_K].copy_(
                    proposed.reshape(-1).to(device=base_flat.device, dtype=base_flat.dtype)
                )

    print("Q4_K sparse boundary statistics:", stats)
    return stats
