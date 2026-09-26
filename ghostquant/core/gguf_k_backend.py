"""Official llama.cpp K-quant state access for GGUF-aware training.

The existing Q4_K path is GPU-native and remains in gguf_backend.py. This
module covers the other K families by calling llama.cpp's exported reference
quantizers and decoding their packed 256-value superblocks. The library path
can be overridden with GGML_BASE_LIB.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from functools import lru_cache
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from gguf_precision import validate_input_precision


QK_K = 256
_BLOCK_BYTES = {"q2_k": 84, "q3_k": 110, "q4_k": 144, "q5_k": 176, "q6_k": 210}
_SUBBLOCK_SIZE = {"q2_k": 16, "q3_k": 16, "q4_k": 32, "q5_k": 32, "q6_k": 16}
_CODE_RANGE = {
    "q2_k": (0, 3),
    "q3_k": (-4, 3),
    "q4_k": (0, 15),
    "q5_k": (0, 31),
    "q6_k": (-32, 31),
}


@dataclass
class KQuantState:
    codes: np.ndarray
    scales: np.ndarray
    mins: np.ndarray
    dequantized: np.ndarray
    subblocks: np.ndarray


def _base_type(method: str) -> str:
    method = method.lower()
    if method in _BLOCK_BYTES:
        return method
    if method.startswith("q2_k"):
        return "q2_k"
    if method.startswith("q3_k"):
        return "q3_k"
    if method.startswith("q4_k"):
        return "q4_k"
    if method.startswith("q5_k"):
        return "q5_k"
    if method.startswith("q6_k"):
        return "q6_k"
    raise ValueError(f"Unsupported K-quant method: {method}")


@lru_cache(maxsize=1)
def _load_ggml_library():
    candidates = []
    if os.environ.get("GGML_BASE_LIB"):
        candidates.append(os.environ["GGML_BASE_LIB"])
    found = ctypes.util.find_library("ggml-base")
    if found:
        candidates.append(found)
    project_library = (
        Path(__file__).resolve().parent.parent
        / "llama.cpp-upstream" / "build" / "bin" / "libggml-base.so"
    )
    candidates.extend([str(project_library), "/usr/local/lib/libggml-base.so"])
    for candidate in candidates:
        try:
            library = ctypes.CDLL(candidate)
            for kind in _BLOCK_BYTES:
                symbol_kind = kind[:-1] + "K"
                quantize = getattr(library, f"quantize_row_{symbol_kind}_ref")
                dequantize = getattr(library, f"dequantize_row_{symbol_kind}")
                quantize.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int64]
                quantize.restype = None
                dequantize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
                dequantize.restype = None
            return library
        except (OSError, AttributeError):
            continue
    raise RuntimeError(
        "Unable to load llama.cpp libggml-base.so. Set GGML_BASE_LIB to the "
        "official library containing quantize_row_q*_K_ref."
    )


def _quantize_bytes(values: np.ndarray, kind: str, input_precision: str = "f16") -> bytes:
    # The C API accepts F32. Round only when the eventual GGUF input is F16.
    validate_input_precision(input_precision)
    values = np.asarray(values, dtype=np.float32)
    if input_precision == "f16":
        values = values.astype(np.float16).astype(np.float32)
    values = np.ascontiguousarray(values.reshape(-1))
    if values.size % QK_K:
        raise ValueError(f"K quantization requires a multiple of {QK_K}, got {values.size}")
    blocks = values.size // QK_K
    output = np.empty(blocks * _BLOCK_BYTES[kind], dtype=np.uint8)
    library = _load_ggml_library()
    symbol_kind = kind[:-1] + "K"
    quantize = getattr(library, f"quantize_row_{symbol_kind}_ref")
    quantize(
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.c_void_p),
        values.size,
    )
    return output.tobytes()


def _fp16(raw: bytes, offset: int) -> float:
    return float(np.frombuffer(raw, dtype="<f2", count=1, offset=offset)[0].astype(np.float32))


def _scale_min_k4(scales: np.ndarray, index: int) -> tuple[int, int]:
    if index < 4:
        return int(scales[index] & 63), int(scales[index + 4] & 63)
    return (
        int((scales[index + 4] & 15) | ((scales[index - 4] >> 6) << 4)),
        int((scales[index + 4] >> 4) | ((scales[index] >> 6) << 4)),
    )


def _decode_block(raw: bytes, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    codes = np.zeros(QK_K, dtype=np.int16)
    scales = np.zeros(QK_K, dtype=np.float32)
    mins = np.zeros(QK_K, dtype=np.float32)
    subblocks = np.repeat(np.arange(QK_K // _SUBBLOCK_SIZE[kind], dtype=np.int16), _SUBBLOCK_SIZE[kind])

    if kind == "q2_k":
        packed_scales = np.frombuffer(raw, dtype=np.uint8, count=16, offset=0)
        packed_q = np.frombuffer(raw, dtype=np.uint8, count=64, offset=16)
        d, dmin = _fp16(raw, 80), _fp16(raw, 82)
        for n in range(0, QK_K, 128):
            q_offset = n // 4
            for j in range(4):
                shift = 2 * j
                for half in range(2):
                    index = 8 * (n // 128) + 2 * j + half
                    sc = packed_scales[index]
                    lo = 16 * half + 32 * j + n
                    hi = lo + 16
                    source = packed_q[q_offset + 16 * half:q_offset + 16 * half + 16]
                    codes[lo:hi] = (source >> shift) & 3
                    scales[lo:hi] = d * (sc & 15)
                    mins[lo:hi] = dmin * (sc >> 4)
        return codes, scales, mins, subblocks

    if kind == "q3_k":
        hmask = np.frombuffer(raw, dtype=np.uint8, count=32, offset=0)
        packed_q = np.frombuffer(raw, dtype=np.uint8, count=64, offset=32)
        source = np.frombuffer(raw, dtype=np.uint8, count=12, offset=96)
        aux = np.zeros(4, dtype="<u4")
        aux[:3] = np.frombuffer(source.tobytes(), dtype="<u4")
        mask1, mask2 = np.uint32(0x03030303), np.uint32(0x0F0F0F0F)
        tmp = aux[2]
        aux[2] = (aux[0] >> 4 & mask2) | ((tmp >> 4 & mask1) << 4)
        aux[3] = (aux[1] >> 4 & mask2) | ((tmp >> 6 & mask1) << 4)
        aux[0] = (aux[0] & mask2) | ((tmp & mask1) << 4)
        aux[1] = (aux[1] & mask2) | ((tmp >> 2 & mask1) << 4)
        signed_scales = np.frombuffer(aux.tobytes(), dtype=np.int8)
        d = _fp16(raw, 108)
        for n in range(0, QK_K, 128):
            q_offset = n // 4
            shift = 0
            mask = 1 << (4 * (n // 128))
            for j in range(4):
                for half in range(2):
                    index = 8 * (n // 128) + 2 * j + half
                    lo = 16 * half + 32 * j + n
                    source = packed_q[q_offset + 16 * half:q_offset + 16 * half + 16]
                    high = hmask[16 * half:16 * half + 16] & mask
                    codes[lo:lo + 16] = ((source >> shift) & 3).astype(np.int16) - np.where(high != 0, 0, 4)
                    scales[lo:lo + 16] = d * float(signed_scales[index] - 32)
                shift += 2
                mask <<= 1
        return codes, scales, mins, subblocks

    if kind == "q4_k":
        packed_scales = np.frombuffer(raw, dtype=np.uint8, count=12, offset=4)
        packed_q = np.frombuffer(raw, dtype=np.uint8, count=128, offset=16)
        d, dmin = _fp16(raw, 0), _fp16(raw, 2)
        for subblock in range(8):
            group = subblock // 2
            sc, mn = _scale_min_k4(packed_scales, subblock)
            lo = subblock * 32
            source = packed_q[group * 32:(group + 1) * 32]
            if subblock % 2:
                codes[lo:lo + 32] = source >> 4
            else:
                codes[lo:lo + 32] = source & 15
            scales[lo:lo + 32] = d * sc
            mins[lo:lo + 32] = dmin * mn
        return codes, scales, mins, subblocks

    if kind == "q5_k":
        d, dmin = _fp16(raw, 0), _fp16(raw, 2)
        packed_scales = np.frombuffer(raw, dtype=np.uint8, count=12, offset=4)
        high = np.frombuffer(raw, dtype=np.uint8, count=32, offset=16)
        low = np.frombuffer(raw, dtype=np.uint8, count=128, offset=48)
        for subblock in range(8):
            group = subblock // 2
            sc, mn = _scale_min_k4(packed_scales, subblock)
            lo = subblock * 32
            q = low[group * 32:(group + 1) * 32]
            nibble = (q & 15) if subblock % 2 == 0 else (q >> 4)
            codes[lo:lo + 32] = nibble.astype(np.int16) + np.where((high & (1 << subblock)) != 0, 16, 0)
            scales[lo:lo + 32] = d * sc
            mins[lo:lo + 32] = dmin * mn
        return codes, scales, mins, subblocks

    if kind == "q6_k":
        low = np.frombuffer(raw, dtype=np.uint8, count=128, offset=0)
        high = np.frombuffer(raw, dtype=np.uint8, count=64, offset=128)
        signed_scales = np.frombuffer(raw, dtype=np.int8, count=16, offset=192)
        d = _fp16(raw, 208)
        for n in range(0, QK_K, 128):
            low_offset = n // 2
            high_offset = n // 4
            scale_offset = n // 16
            for j in range(32):
                q1 = int((low[low_offset + j] & 15) | (((high[high_offset + j] >> 0) & 3) << 4))
                q2 = int((low[low_offset + j + 32] & 15) | (((high[high_offset + j] >> 2) & 3) << 4))
                q3 = int((low[low_offset + j] >> 4) | (((high[high_offset + j] >> 4) & 3) << 4))
                q4 = int((low[low_offset + j + 32] >> 4) | (((high[high_offset + j] >> 6) & 3) << 4))
                pos = [n + j, n + j + 32, n + j + 64, n + j + 96]
                vals = [q1 - 32, q2 - 32, q3 - 32, q4 - 32]
                scale_ids = [scale_offset + j // 16, scale_offset + j // 16 + 2,
                             scale_offset + j // 16 + 4, scale_offset + j // 16 + 6]
                for p, value, scale_id in zip(pos, vals, scale_ids):
                    codes[p] = value
                    scales[p] = d * float(signed_scales[scale_id])
        return codes, scales, mins, subblocks

    raise AssertionError(kind)


def _decode_blocks(raw: bytes, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized packed-state decoder; Python loops are independent of block count."""
    block_bytes = _BLOCK_BYTES[kind]
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(-1, block_bytes)
    blocks = rows.shape[0]
    codes = np.zeros((blocks, QK_K), dtype=np.int16)
    scales = np.zeros((blocks, QK_K), dtype=np.float32)
    mins = np.zeros_like(scales)
    subblocks = np.tile(
        np.repeat(
            np.arange(QK_K // _SUBBLOCK_SIZE[kind], dtype=np.int16),
            _SUBBLOCK_SIZE[kind],
        ),
        (blocks, 1),
    )

    def fp16_column(offset: int) -> np.ndarray:
        return rows[:, offset:offset + 2].copy().view("<f2").reshape(-1).astype(np.float32)

    def scale_min_k4(packed: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
        if index < 4:
            return packed[:, index] & 63, packed[:, index + 4] & 63
        return (
            (packed[:, index + 4] & 15) | ((packed[:, index - 4] >> 6) << 4),
            (packed[:, index + 4] >> 4) | ((packed[:, index] >> 6) << 4),
        )

    if kind == "q2_k":
        packed_scales, packed_q = rows[:, :16], rows[:, 16:80]
        d, dmin = fp16_column(80), fp16_column(82)
        for n in range(0, QK_K, 128):
            q_offset = n // 4
            for j in range(4):
                shift = 2 * j
                for half in range(2):
                    index = 8 * (n // 128) + 2 * j + half
                    lo = n + 32 * j + 16 * half
                    source = packed_q[:, q_offset + 16 * half:q_offset + 16 * half + 16]
                    sc = packed_scales[:, index]
                    codes[:, lo:lo + 16] = (source >> shift) & 3
                    scales[:, lo:lo + 16] = (d * (sc & 15))[:, None]
                    mins[:, lo:lo + 16] = (dmin * (sc >> 4))[:, None]

    elif kind == "q3_k":
        hmask, packed_q = rows[:, :32], rows[:, 32:96]
        aux = np.zeros((blocks, 4), dtype="<u4")
        aux[:, :3] = rows[:, 96:108].copy().view("<u4").reshape(blocks, 3)
        mask1, mask2 = np.uint32(0x03030303), np.uint32(0x0F0F0F0F)
        tmp = aux[:, 2].copy()
        aux[:, 2] = (aux[:, 0] >> 4 & mask2) | ((tmp >> 4 & mask1) << 4)
        aux[:, 3] = (aux[:, 1] >> 4 & mask2) | ((tmp >> 6 & mask1) << 4)
        aux[:, 0] = (aux[:, 0] & mask2) | ((tmp & mask1) << 4)
        aux[:, 1] = (aux[:, 1] & mask2) | ((tmp >> 2 & mask1) << 4)
        signed_scales = aux.view(np.int8).reshape(blocks, 16).astype(np.int16)
        d = fp16_column(108)
        for n in range(0, QK_K, 128):
            q_offset = n // 4
            mask = 1 << (4 * (n // 128))
            for j in range(4):
                shift = 2 * j
                for half in range(2):
                    index = 8 * (n // 128) + 2 * j + half
                    lo = n + 32 * j + 16 * half
                    source = packed_q[:, q_offset + 16 * half:q_offset + 16 * half + 16]
                    high = hmask[:, 16 * half:16 * half + 16] & (mask << j)
                    codes[:, lo:lo + 16] = (
                        ((source >> shift) & 3).astype(np.int16)
                        - np.where(high != 0, 0, 4)
                    )
                    scales[:, lo:lo + 16] = (
                        d * (signed_scales[:, index] - 32)
                    )[:, None]

    elif kind in {"q4_k", "q5_k"}:
        d, dmin = fp16_column(0), fp16_column(2)
        packed_scales = rows[:, 4:16]
        if kind == "q4_k":
            low = rows[:, 16:144]
            high = None
        else:
            high = rows[:, 16:48]
            low = rows[:, 48:176]
        for subblock in range(8):
            group = subblock // 2
            sc, mn = scale_min_k4(packed_scales, subblock)
            lo = subblock * 32
            source = low[:, group * 32:(group + 1) * 32]
            nibble = (source & 15) if subblock % 2 == 0 else (source >> 4)
            code = nibble.astype(np.int16)
            if high is not None:
                code += np.where((high & (1 << subblock)) != 0, 16, 0)
            codes[:, lo:lo + 32] = code
            scales[:, lo:lo + 32] = (d * sc)[:, None]
            mins[:, lo:lo + 32] = (dmin * mn)[:, None]

    elif kind == "q6_k":
        low, high = rows[:, :128], rows[:, 128:192]
        signed_scales = rows[:, 192:208].view(np.int8).astype(np.int16)
        d = fp16_column(208)
        for n in range(0, QK_K, 128):
            low_offset = n // 2
            high_offset = n // 4
            scale_offset = n // 16
            for j in range(32):
                packed_values = (
                    (low[:, low_offset + j] & 15) | ((high[:, high_offset + j] & 3) << 4),
                    (low[:, low_offset + j + 32] & 15) | (((high[:, high_offset + j] >> 2) & 3) << 4),
                    (low[:, low_offset + j] >> 4) | (((high[:, high_offset + j] >> 4) & 3) << 4),
                    (low[:, low_offset + j + 32] >> 4) | (((high[:, high_offset + j] >> 6) & 3) << 4),
                )
                positions = (n + j, n + j + 32, n + j + 64, n + j + 96)
                scale_ids = (
                    scale_offset + j // 16,
                    scale_offset + j // 16 + 2,
                    scale_offset + j // 16 + 4,
                    scale_offset + j // 16 + 6,
                )
                for packed, position, scale_id in zip(packed_values, positions, scale_ids):
                    codes[:, position] = packed.astype(np.int16) - 32
                    scales[:, position] = d * signed_scales[:, scale_id]
    else:
        raise AssertionError(kind)

    return codes, scales, mins, subblocks


def quantize_k(values: torch.Tensor, method: str, input_precision: str = "f16") -> KQuantState:
    kind = _base_type(method)
    flat = values.detach().to(torch.float32).cpu().contiguous().view(-1).numpy()
    raw = _quantize_bytes(flat, kind, input_precision=input_precision)
    codes, scales, mins, subblocks = _decode_blocks(raw, kind)
    library = _load_ggml_library()
    output = np.empty(flat.size, dtype=np.float32)
    symbol_kind = kind[:-1] + "K"
    getattr(library, f"dequantize_row_{symbol_kind}")(
        ctypes.c_char_p(raw),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        flat.size,
    )
    return KQuantState(codes, scales, mins, output, subblocks)


def _layer_index(name: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.", name)
    return None if match is None else int(match.group(1))


def _use_more_bits(layer: int, count: int) -> bool:
    eighth = max(1, count // 8)
    return layer < eighth or layer >= 7 * eighth or (layer - eighth) % 3 == 2


def _visual_layer_info(name: str, num_layers: int) -> tuple[int | None, int, str]:
    match = re.search(r"\.visual\.blocks\.(\d+)\.", name)
    if match is None:
        return None, num_layers, "other"
    layer = int(match.group(1))
    # llama.cpp's mmproj tensor order is lexicographic (0, 1, 10, ...), and
    # its attention-V promotion counters follow that order.
    ordered = sorted(range(num_layers), key=lambda index: str(index))
    rank = ordered.index(layer) if layer in ordered else layer
    if ".attn.qkv.weight" in name:
        role = "attn_v"
    elif ".mlp.linear_fc2.weight" in name:
        role = "ffn_down"
    else:
        role = "other"
    return rank, num_layers, role


def qtype_for_tensor(name: str, method: str, num_layers: int, n_gqa: int = 1,
                     num_visual_layers: int = 24) -> str:
    """Return the underlying K type for a tensor in a K recipe."""
    method = method.lower()
    kind = _base_type(method)
    # Bare q4_k keeps the project's established pure-Q4 behavior. Q3_K and
    # Q5_K are internal base types; the public llama.cpp aliases are *_k_m.
    if method in {"q3_k", "q4_k", "q5_k", "q6_k"}:
        return kind
    is_visual = ".visual." in name
    if is_visual:
        layer, visual_layers, role = _visual_layer_info(name, num_visual_layers)
        if method == "q2_k":
            # Current llama.cpp aborts while quantizing CLIP Q2_K mixtures
            # because the vision config has no n_head_kv. Use pure Q2_K for
            # the mmproj; language tensors still use the standard recipe.
            return "q2_k"
        if layer is None:
            return kind
        if method == "q3_k_m":
            if role == "attn_v":
                return "q5_k" if layer < 2 else "q4_k"
            if role == "ffn_down":
                return "q4_k"  # CLIP has n_layer_all == 0 in llama.cpp.
        if method == "q3_k_l":
            if role in {"attn_v", "ffn_down"}:
                return "q5_k"
        if method == "q4_k_s":
            if role == "attn_v" and layer < 4:
                return "q5_k"
        if method in {"q4_k_m", "q5_k_m"}:
            if role == "ffn_down":
                # The CLIP architecture reports n_layer_all == 0 to the
                # llama.cpp recipe selector, making use_more_bits true for
                # every visual FFN-down tensor.
                return "q6_k"
            if role == "attn_v" and _use_more_bits(layer, visual_layers):
                return "q6_k"
        return kind
    layer = _layer_index(name)
    role = (
        "attn_v" if ".self_attn.v_proj.weight" in name else
        "ffn_down" if ".mlp.down_proj.weight" in name else
        "attn_qkv" if ".self_attn.qkv_proj.weight" in name else
        "attn_output" if ".self_attn.o_proj.weight" in name else
        "other"
    )
    if layer is None:
        return kind
    if method == "q2_k":
        if role == "attn_v":
            return "q4_k" if n_gqa >= 4 else "q3_k"
        return "q3_k" if role in {"ffn_down", "attn_output"} else "q2_k"
    if method == "q2_k_s":
        return "q4_k" if role == "ffn_down" and layer < num_layers // 8 else "q2_k"
    if method == "q3_k_s":
        return "q3_k"
    if method == "q3_k_m":
        if role == "attn_v": return "q5_k" if layer < 2 else "q4_k"
        if role == "ffn_down": return "q5_k" if layer < num_layers // 16 else "q4_k"
        if role in {"attn_qkv", "attn_output"}: return "q4_k"
        return "q3_k"
    if method == "q3_k_l":
        if role in {"attn_v", "ffn_down", "attn_output"}: return "q5_k"
        if role == "attn_qkv": return "q4_k"
        return "q3_k"
    if method == "q4_k_s":
        if role == "attn_v" and layer < 4: return "q5_k"
        if role == "ffn_down" and layer < num_layers // 8: return "q5_k"
        return "q4_k"
    if method == "q4_k_m":
        if role in {"attn_v", "ffn_down"} and _use_more_bits(layer, num_layers): return "q6_k"
        if role == "attn_qkv": return "q5_k"
        return "q4_k"
    if method == "q5_k_s":
        return "q5_k"
    if method == "q5_k_m":
        if role in {"attn_v", "ffn_down"} and _use_more_bits(layer, num_layers): return "q6_k"
        if role == "attn_qkv": return "q6_k"
        return "q5_k"
    raise ValueError(f"Unsupported K recipe: {method}")


def _target_names(model, method: str, scope: str):
    if getattr(getattr(model, "config", None), "model_type", None) == "internvl":
        import json
        text_cfg, vision_cfg = model.config.text_config, model.config.vision_config
        if (text_cfg.hidden_size, text_cfg.num_hidden_layers, vision_cfg.hidden_size, vision_cfg.num_hidden_layers) != (896, 24, 1024, 24):
            raise ValueError("Generate and verify a GGUF routing plan for this InternVL architecture first")
        plan = json.loads(Path(__file__).with_name("internvl3_1b_gguf_plan.json").read_text())[method.lower()]
        params = dict(model.named_parameters())
        result = []
        for name, spec in plan.items():
            if name not in params or list(params[name].shape) != spec["shape"]:
                raise ValueError(f"InternVL GGUF plan does not match {name}")
            if scope == "language" and spec["scope"] != "language": continue
            if scope == "vision" and spec["scope"] != "mmproj": continue
            # Official fallback Q4_0/Q5_0/Q5_1/Q8_0 has no K superblock boundary.
            if spec["qtype"] in _BLOCK_BYTES:
                result.append((name, spec["qtype"]))
        return result
    names = []
    layers = [int(m.group(1)) for n, _ in model.named_parameters() if (m := re.search(r"\.language_model\.layers\.(\d+)\.", n))]
    num_layers = max(layers) + 1 if layers else 0
    visual_layers = max(
        [int(m.group(1)) for n, _ in model.named_parameters()
         if (m := re.search(r"\.visual\.blocks\.(\d+)\.", n))],
        default=-1,
    ) + 1
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    heads = int(getattr(text_config, "num_attention_heads", 1) or 1)
    kv_heads = int(getattr(text_config, "num_key_value_heads", heads) or heads)
    n_gqa = max(1, heads // kv_heads)
    for name, parameter in model.named_parameters():
        is_language = ".language_model.layers." in name
        is_vision = ".visual." in name
        if scope == "language" and not is_language: continue
        if scope == "vision" and not is_vision: continue
        if scope == "all" and not (is_language or is_vision): continue
        if not name.endswith(".weight") or parameter.ndim != 2 or parameter.numel() % QK_K: continue
        if "embed_tokens" in name or "lm_head" in name or "pos_embed" in name: continue
        names.append((name, qtype_for_tensor(name, method, num_layers, n_gqa, visual_layers)))
    return names


def _internvl_fallback_proxy(model, method, scope, input_precision):
    """Freeze non-K fallback matrices at their official dequantized values.

    These matrices participate in the forward pass but have no K-boundary
    candidates. Stage2 starts from the FP32 base and never copies these proxies.
    """
    import json
    if getattr(model.config, "model_type", None) != "internvl": return
    plan = json.loads(Path(__file__).with_name("internvl3_1b_gguf_plan.json").read_text())[method.lower()]
    params = dict(model.named_parameters())
    sizes = {"q4_0":18, "q5_0":22, "q5_1":24, "q8_0":34}
    library = _load_ggml_library()
    count = 0
    for name, spec in plan.items():
        kind = spec["qtype"]
        if kind in _BLOCK_BYTES: continue
        if scope == "language" and spec["scope"] != "language": continue
        if scope == "vision" and spec["scope"] != "mmproj": continue
        if kind not in sizes: raise ValueError(f"Unsupported fallback: {kind}")
        p = params[name]
        values = p.detach().float().cpu().contiguous().view(-1).numpy()
        if input_precision == "f16": values = values.astype(np.float16).astype(np.float32)
        packed = np.empty(values.size // 32 * sizes[kind], dtype=np.uint8)
        output = np.empty_like(values)
        quant = getattr(library, f"quantize_row_{kind}_ref")
        dequant = getattr(library, f"dequantize_row_{kind}")
        quant.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int64]
        dequant.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
        quant.restype = dequant.restype = None
        quant(values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), packed.ctypes.data_as(ctypes.c_void_p), values.size)
        dequant(packed.ctypes.data_as(ctypes.c_void_p), output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), values.size)
        p.data.copy_(torch.from_numpy(output).reshape(p.shape).to(p.device, p.dtype))
        count += 1
    print(f"InternVL: {count} official non-K fallback matrices frozen as forward proxies", flush=True)


def _boundaries(state: KQuantState, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return numeric lower/upper boundaries under frozen block metadata."""
    code = state.codes.reshape(-1)
    signed_scale = state.scales.reshape(-1)
    step = np.abs(signed_scale)
    lower = state.dequantized - 0.5 * step
    upper = state.dequantized + 0.5 * step
    code_min, code_max = _CODE_RANGE[kind]
    positive = signed_scale >= 0
    lower[((code <= code_min) & positive) | ((code >= code_max) & ~positive)] = -np.inf
    upper[((code >= code_max) & positive) | ((code <= code_min) & ~positive)] = np.inf
    return lower, upper, step


def _locked_positions(value: np.ndarray, state: KQuantState, kind: str) -> np.ndarray:
    """Lock scale/min anchor subblocks and exact per-subblock extrema."""
    blocks = state.codes.shape[0]
    subblock_size = _SUBBLOCK_SIZE[kind]
    subblocks = QK_K // subblock_size
    values = value.reshape(blocks, subblocks, subblock_size)
    locked = np.zeros_like(values, dtype=bool)
    rows = np.arange(blocks)[:, None]
    subs = np.arange(subblocks)[None, :]
    locked[rows, subs, values.argmax(axis=2)] = True
    locked[rows, subs, values.argmin(axis=2)] = True

    scale_by_subblock = np.abs(
        state.scales.reshape(blocks, subblocks, subblock_size)[:, :, 0]
    )
    locked[np.arange(blocks), scale_by_subblock.argmax(axis=1), :] = True
    if kind in {"q2_k", "q4_k", "q5_k"}:
        min_by_subblock = np.abs(
            state.mins.reshape(blocks, subblocks, subblock_size)[:, :, 0]
        )
        locked[np.arange(blocks), min_by_subblock.argmax(axis=1), :] = True
    return locked.reshape(-1)


def _subblock_ranges(value: np.ndarray, state: KQuantState) -> tuple[np.ndarray, np.ndarray]:
    subblock_size = int(np.count_nonzero(state.subblocks[0] == 0))
    values = value.reshape(-1, subblock_size)
    range_low = np.repeat(values.min(axis=1), subblock_size)
    range_high = np.repeat(values.max(axis=1), subblock_size)
    return range_low, range_high


def build_k_finetune_boxes(model, original_model, method: str, scope: str = "all", candidate_relative_limit: float = 0.05,
                           candidate_absolute_limit: float | None = None, boundary_penetration: float = 0.05,
                           chunk_blocks: int = 4096, metadata_mode: str = "free",
                           selection_fraction: float | None = None, input_precision: str = "f16", **_kwargs):
    """Build FP32 PGD boxes using official llama.cpp K-quant states."""
    del chunk_blocks
    validate_input_precision(input_precision)
    if candidate_relative_limit is None or candidate_relative_limit <= 0:
        raise ValueError("K-quant selection requires a positive relative limit")
    if candidate_absolute_limit is not None and candidate_absolute_limit <= 0:
        raise ValueError("K-quant absolute selection limit must be positive when enabled")
    if not 0 <= boundary_penetration <= 1:
        raise ValueError("gguf_penetration must be between 0 and 1")
    if selection_fraction is not None and not 0 < selection_fraction <= 1:
        raise ValueError("gguf_selection_fraction must be in (0, 1]")
    if metadata_mode != "free":
        raise ValueError("Generic K-quant currently supports gguf_metadata_mode: free")
    params, originals = dict(model.named_parameters()), dict(original_model.named_parameters())
    boxes, indices = {}, {}
    total = selected = 0
    totals_by_type: dict[str, list[int]] = {}
    totals_by_scope = {"language": [0, 0], "vision": [0, 0]}
    names = _target_names(model, method, scope)
    if getattr(getattr(model, "config", None), "model_type", None) == "internvl":
        _internvl_fallback_proxy(model, method, scope, input_precision)
    for tensor_index, (name, qtype) in enumerate(names, 1):
        if tensor_index == 1 or tensor_index % 25 == 0:
            print(f"{method.upper()} Stage1 tensor {tensor_index}/{len(names)}: {name}", flush=True)
        current = params[name].detach().float().cpu().view(-1)
        original = originals[name].detach().float().cpu().view(-1)
        state = quantize_k(current, qtype, input_precision=input_precision)
        lower, upper, step = _boundaries(state, qtype)
        value = current.numpy()
        budget = candidate_relative_limit * np.abs(value)
        if candidate_absolute_limit is not None:
            budget = np.minimum(budget, candidate_absolute_limit)
        distance_low = np.abs(value - lower)
        distance_high = np.abs(upper - value)
        changed = np.abs(value - original.numpy()) > 0
        locked = _locked_positions(value, state, qtype)
        valid = ~locked & ~changed & (step > 0)
        lower_ok = valid & np.isfinite(distance_low) & (distance_low <= budget)
        upper_ok = valid & np.isfinite(distance_high) & (distance_high <= budget)
        eligible = lower_ok | upper_ok
        eligible_count = int(eligible.sum())
        chosen = eligible.copy()
        if selection_fraction is not None:
            target_count = int(round(value.size * selection_fraction))
            if target_count <= 0:
                raise ValueError("gguf_selection_fraction selects zero parameters for a target tensor")
            if eligible_count < target_count:
                raise RuntimeError(
                    f"{name} ({qtype}) has only {eligible_count:,} candidates under "
                    f"relative <= {candidate_relative_limit:g}, absolute <= "
                    f"{candidate_absolute_limit if candidate_absolute_limit is not None else 'disabled'}; "
                    "cannot guarantee "
                    f"{target_count:,} ({selection_fraction:.2%}) Stage-1 participants"
                )
            if eligible_count > target_count:
                normalized_distance = np.minimum(distance_low, distance_high)
                normalized_distance = normalized_distance / np.maximum(budget, 1e-12)
                candidate_ids = np.flatnonzero(eligible)
                keep = candidate_ids[
                    np.argpartition(normalized_distance[candidate_ids], target_count - 1)[:target_count]
                ]
                chosen = np.zeros_like(eligible)
                chosen[keep] = True
            lower_ok &= chosen
            upper_ok &= chosen
        initialized = state.dequantized.copy()
        initialized[chosen | locked | changed] = value[chosen | locked | changed]
        lo = initialized.copy(); hi = initialized.copy()
        margin = np.minimum(boundary_penetration * step, np.maximum(budget - distance_low, 0))
        lo[lower_ok] = np.minimum(lower[lower_ok] - margin[lower_ok], value[lower_ok])
        margin = np.minimum(boundary_penetration * step, np.maximum(budget - distance_high, 0))
        hi[upper_ok] = np.maximum(upper[upper_ok] + margin[upper_ok], value[upper_ok])
        range_low, range_high = _subblock_ranges(value, state)
        lo[chosen] = np.maximum(lo[chosen], range_low[chosen])
        hi[chosen] = np.minimum(hi[chosen], range_high[chosen])
        lo[chosen] = np.maximum(lo[chosen], value[chosen] - budget[chosen])
        hi[chosen] = np.minimum(hi[chosen], value[chosen] + budget[chosen])
        lo[chosen] = np.minimum(lo[chosen], value[chosen]); hi[chosen] = np.maximum(hi[chosen], value[chosen])
        params[name].data.copy_(torch.from_numpy(initialized.reshape(params[name].shape)).to(params[name].device, params[name].dtype))
        boxes[name] = [torch.from_numpy(lo.reshape(params[name].shape)), torch.from_numpy(hi.reshape(params[name].shape))]
        indices[name] = torch.from_numpy(np.flatnonzero(chosen).astype(np.int64))
        chosen_count = int(chosen.sum())
        total += value.size; selected += chosen_count
        type_counts = totals_by_type.setdefault(qtype, [0, 0, 0])
        type_counts[0] += value.size; type_counts[1] += eligible_count; type_counts[2] += chosen_count
        parameter_scope = "vision" if any(x in name for x in (".visual.", ".vision_tower.", ".multi_modal_projector.")) else "language"
        totals_by_scope[parameter_scope][0] += value.size
        totals_by_scope[parameter_scope][1] += chosen_count
    print(f"{method.upper()} target tensors: {len(names)}")
    print(f"{method.upper()} parameters scanned: {total:,}")
    absolute_text = (
        f"{candidate_absolute_limit:g}"
        if candidate_absolute_limit is not None
        else "disabled"
    )
    print(
        f"{method.upper()} selection limits: relative <= {candidate_relative_limit:g}, "
        f"absolute <= {absolute_text}"
    )
    print(f"{method.upper()} sensitive parameters: {selected:,} ({100.0 * selected / max(1, total):.4f}%)")
    for qtype, (type_total, type_eligible, type_selected) in sorted(totals_by_type.items()):
        print(
            f"  {qtype.upper()}: eligible {type_eligible:,}, selected "
            f"{type_selected:,}/{type_total:,} ({100.0 * type_selected / max(1, type_total):.4f}%)"
        )
    for parameter_scope, (scope_total, scope_selected) in totals_by_scope.items():
        if scope_total:
            print(f"  {parameter_scope}: {scope_selected:,}/{scope_total:,} ({100.0 * scope_selected / scope_total:.4f}%)")
    return indices, boxes, None


def nudge_base_across_k_boundaries(base_model, ft_model, method: str, scope: str = "all", penetration: float = 0.05,
                                   max_relative_change: float = 0.05, max_absolute_change: float | None = None,
                                   selected_indices: dict[str, torch.Tensor] | None = None,
                                   input_precision: str = "f16", **_kwargs):
    """Nudge the original model toward Stage1 K-code changes under hard FP32 budgets."""
    validate_input_precision(input_precision)
    if max_relative_change is None or max_relative_change <= 0:
        raise ValueError("K-quant Stage 2 requires a positive relative limit")
    if max_absolute_change is not None and max_absolute_change <= 0:
        raise ValueError("K-quant Stage 2 absolute limit must be positive when enabled")
    if not 0 <= penetration <= 1:
        raise ValueError("gguf_penetration must be between 0 and 1")
    base_params, ft_params = dict(base_model.named_parameters()), dict(ft_model.named_parameters())
    stats = {
        "gguf_input_precision": input_precision,
        "rejected_blocks": 0,
        "rejected_parameters": 0,
        "tensors": 0,
        "blocks": 0,
        "candidate_parameters": 0,
        "stage1_crossing_parameters": 0,
        "changed_parameters": 0,
        "actual_target_crossings": 0,
        "failed_target_crossings": 0,
        "collateral_code_changes": 0,
        "metadata_changed_blocks": 0,
        "budget_violations": 0,
        "subblock_range_violations": 0,
        "absolute_delta_sum": 0.0,
        "relative_delta_sum": 0.0,
        "max_absolute_delta": 0.0,
        "max_relative_delta": 0.0,
    }
    target_names = _target_names(base_model, method, scope)
    for tensor_index, (name, qtype) in enumerate(target_names, 1):
        if tensor_index == 1 or tensor_index % 25 == 0:
            print(f"{method.upper()} Stage2 tensor {tensor_index}/{len(target_names)}: {name}", flush=True)
        base = base_params[name].data.float().cpu().view(-1)
        ft = ft_params[name].data.float().cpu().view(-1)
        state = quantize_k(base, qtype, input_precision=input_precision)
        code = state.codes.reshape(-1)
        signed_scale = state.scales.reshape(-1)
        lower, upper, step = _boundaries(state, qtype)
        # Keep an immutable snapshot: base_params[name] is updated below, and
        # a NumPy view would otherwise make the post-update delta appear zero.
        value = base.numpy().copy()
        budget = max_relative_change * np.abs(value)
        if max_absolute_change is not None:
            budget = np.minimum(budget, max_absolute_change)
        distance_low = np.abs(value - lower)
        distance_high = np.abs(upper - value)
        locked = _locked_positions(value, state, qtype)
        valid_step = step > 0
        lower_eligible = ~locked & valid_step & np.isfinite(distance_low) & (distance_low <= budget)
        upper_eligible = ~locked & valid_step & np.isfinite(distance_high) & (distance_high <= budget)
        selected = lower_eligible | upper_eligible
        if selected_indices is not None:
            selected = np.zeros_like(selected)
            saved = selected_indices.get(name)
            if saved is not None:
                saved = np.asarray(saved, dtype=np.int64).reshape(-1)
                if np.any((saved < 0) | (saved >= value.size)):
                    raise ValueError(f"Saved Stage-1 selection indices are invalid for {name}")
                selected[saved] = True
            outside_budget = selected & ~(
                lower_eligible | upper_eligible
            )
            if outside_budget.any():
                raise RuntimeError(
                    f"Saved Stage-1 selection for {name} is outside the Stage-2 "
                    "boundary budget; use the same relative/absolute limits"
                )

        # Decode the Stage-1 value using the original frozen metadata. This is
        # the direction oracle and excludes collateral changes caused by a new
        # scale/min fit in the Stage-1 block.
        ft_value = ft.numpy()
        safe_scale = np.where(signed_scale != 0, signed_scale, 1.0)
        # Compare both endpoints on the SAME frozen grid. Comparing the FT
        # grid code directly with the packed base code can invent movement:
        # fitted/rounded metadata need not reproduce the original assignment.
        base_grid_value = value
        if input_precision == "f16":
            ft_value = ft_value.astype(np.float16).astype(np.float32)
            base_grid_value = value.astype(np.float16).astype(np.float32)
        code_min, code_max = _CODE_RANGE[qtype]
        def frozen_code(weights):
            transformed = (weights + state.mins.reshape(-1)) / safe_scale
            return np.clip(np.floor(transformed + 0.5), code_min, code_max).astype(np.int16)
        direction = frozen_code(ft_value) - frozen_code(base_grid_value)
        direction[signed_scale == 0] = 0
        crossing = selected & (direction != 0)
        value_direction = np.sign(direction * signed_scale)
        positive = value_direction > 0
        distance = np.where(positive, distance_high, distance_low)
        boundary = np.where(positive, upper, lower)
        remaining = np.maximum(budget - distance, 0)
        # Match the established FP4/NF4 Stage-2 rule: move 5% beyond the
        # distance to the boundary, rather than 5% of the full quantization
        # step. This is the minimal robust crossing and avoids deep nudges.
        margin = np.minimum(penetration * distance, remaining)
        target_value = boundary + value_direction * margin
        ties = crossing & (margin == 0) & np.isfinite(boundary)
        target_value[ties & positive] = np.nextafter(boundary[ties & positive], np.float32(np.inf))
        target_value[ties & ~positive] = np.nextafter(boundary[ties & ~positive], np.float32(-np.inf))
        target_value = np.minimum(np.maximum(target_value, value - budget), value + budget)
        range_low, range_high = _subblock_ranges(value, state)
        valid = crossing & np.isfinite(target_value)
        # A nudge must move in the observed Stage-1 direction. A rounded
        # metadata boundary may already lie behind the original weight.
        valid &= (target_value - value) * value_direction > 0
        valid &= target_value >= range_low
        valid &= target_value <= range_high
        # The boundary arithmetic is FP32 while the budget product is NumPy
        # float32; use the same small numerical tolerance as the legacy Q4_K
        # projector without relaxing the user-visible 0.05/0.003 limits.
        valid &= np.abs(target_value - value) <= budget + 1e-7
        proposed = value.copy()
        proposed[valid] = target_value[valid]

        proposed_state = quantize_k(torch.from_numpy(proposed), qtype, input_precision=input_precision)
        proposed_code = proposed_state.codes.reshape(-1)
        code_delta = proposed_code - code
        actual = valid & ((code_delta * direction) > 0)
        # Free metadata couples all 256 values. Reverting only a failed
        # coordinate can invalidate another crossing in the same block.
        # Reject the entire block so every retained edit has a verified effect.
        rejected = ((proposed != value) & ~actual).reshape(-1, QK_K).any(axis=1)
        if rejected.any():
            rejected_positions = np.repeat(rejected, QK_K)
            stats["rejected_blocks"] += int(rejected.sum())
            stats["rejected_parameters"] += int(((proposed != value) & rejected_positions).sum())
            proposed[rejected_positions] = value[rejected_positions]
            proposed_state = quantize_k(torch.from_numpy(proposed), qtype, input_precision=input_precision)
            proposed_code = proposed_state.codes.reshape(-1)
            code_delta = proposed_code - code
        changed = proposed != value
        actual = changed & ((code_delta * direction) > 0)
        if np.any(changed & ~actual):
            raise RuntimeError(f"Stage-2 retained an unverified K-code edit in {name}")
        if changed.any():
            base_params[name].data.copy_(torch.from_numpy(proposed.reshape(base_params[name].shape)).to(base_params[name].device, base_params[name].dtype))
        block_changed_metadata = np.any(
            (proposed_state.scales != state.scales) | (proposed_state.mins != state.mins), axis=1
        )
        delta = np.abs(proposed - value)
        relative = np.divide(delta, np.abs(value), out=np.zeros_like(delta), where=value != 0)
        stats["tensors"] += 1
        stats["blocks"] += state.codes.shape[0]
        stats["candidate_parameters"] += int(selected.sum())
        stats["stage1_crossing_parameters"] += int(crossing.sum())
        stats["changed_parameters"] += int(changed.sum())
        stats["actual_target_crossings"] += int(actual.sum())
        stats["failed_target_crossings"] += int((crossing & ~actual).sum())
        stats["collateral_code_changes"] += int(((~valid) & (proposed_code != code)).sum())
        stats["metadata_changed_blocks"] += int(block_changed_metadata.sum())
        stats["budget_violations"] += int((changed & (delta > budget + 1e-7)).sum())
        stats["subblock_range_violations"] += int((changed & ((proposed < range_low) | (proposed > range_high))).sum())
        stats["absolute_delta_sum"] += float(delta[changed].sum())
        stats["relative_delta_sum"] += float(relative[changed].sum())
        stats["max_absolute_delta"] = max(stats["max_absolute_delta"], float(delta.max(initial=0)))
        stats["max_relative_delta"] = max(stats["max_relative_delta"], float(relative.max(initial=0)))
    count = max(1, stats["changed_parameters"])
    stats["mean_absolute_delta"] = stats["absolute_delta_sum"] / count
    stats["mean_relative_delta"] = stats["relative_delta_sum"] / count
    return stats
