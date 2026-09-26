"""Input precision contract shared by K training, Stage2, and GGUF export."""
from __future__ import annotations

import json
from pathlib import Path

INPUT_PRECISIONS = ("f16", "f32")
REFERENCE_K_METHODS = {
    "q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_k_s", "q4_k_m",
    "q5_k_s", "q5_k_m", "q6_k",
}
CONTRACT_FILE = "gguf_quantization_contract.json"


def validate_input_precision(value: str) -> str:
    if value not in INPUT_PRECISIONS:
        raise ValueError(f"gguf_input_precision must be one of {INPUT_PRECISIONS}, got {value!r}")
    return value


def validate_stage1_precision(selection: dict | None, requested: str) -> None:
    """Old selections predate this field and were produced by the F16 proxy."""
    requested = validate_input_precision(requested)
    saved = validate_input_precision((selection or {}).get("gguf_input_precision", "f16"))
    if saved != requested:
        raise ValueError(
            f"Stage-1 GGUF input precision {saved} does not match Stage-2 {requested}. "
            "Use matching settings or rerun Stage 1; legacy selections use f16."
        )


def save_precision_contract(directory: str | Path, method: str, precision: str) -> None:
    payload = {
        "version": 1,
        "quantize_method": method,
        "gguf_input_precision": validate_input_precision(precision),
    }
    (Path(directory) / CONTRACT_FILE).write_text(json.dumps(payload, indent=2) + "\n")


def validate_export_precision(directory: str | Path, precision: str) -> None:
    precision = validate_input_precision(precision)
    contract = Path(directory) / CONTRACT_FILE
    if not contract.exists():
        return  # Historical checkpoints have no contract; do not invent provenance.
    payload = json.loads(contract.read_text())
    saved = validate_input_precision(payload["gguf_input_precision"])
    if saved != precision:
        raise ValueError(
            f"Checkpoint GGUF input precision {saved} does not match export {precision}: {contract}"
        )
