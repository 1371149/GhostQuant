"""Quantize Stage-2 GGUF files with the official llama.cpp executable."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


METHODS = {
    "q2_k": "Q2_K",
    "q3_k_s": "Q3_K_S",
    "q3_k_m": "Q3_K_M",
    "q3_k_l": "Q3_K_L",
    "q4_k": "Q4_K",
    "q4_k_s": "Q4_K_S",
    "q4_k_m": "Q4_K_M",
    "q5_k_s": "Q5_K_S",
    "q5_k_m": "Q5_K_M",
    "q6_k": "Q6_K",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Run official llama.cpp K-quantization for language and vision GGUF files"
    )
    parser.add_argument("--model", required=True, type=Path, help="F16/F32 language GGUF")
    parser.add_argument("--mmproj", type=Path, help="F16/F32 vision/mmproj GGUF")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--quantizer", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def quantize_one(
    quantizer: Path,
    source: Path,
    destination: Path,
    gguf_type: str,
    threads: int,
    pure: bool,
    dry_run: bool,
) -> list[str]:
    command = [str(quantizer)]
    if dry_run:
        command.append("--dry-run")
    if pure:
        command.append("--pure")
    command.append(str(source))
    if not dry_run:
        command.append(str(destination))
    command.extend([gguf_type, str(threads)])
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    return command


def main() -> None:
    args = parse_args()
    if not args.quantizer.is_file():
        raise FileNotFoundError(f"llama-quantize not found: {args.quantizer}")
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if args.mmproj is not None and not args.mmproj.is_file():
        raise FileNotFoundError(args.mmproj)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gguf_type = METHODS[args.method]
    language_output = args.output_dir / f"{args.model.stem}-{gguf_type}.gguf"
    commands = [
        quantize_one(
            args.quantizer,
            args.model,
            language_output,
            gguf_type,
            args.threads,
            pure=args.method == "q4_k",
            dry_run=args.dry_run,
        )
    ]

    mmproj_output = None
    if args.mmproj is not None:
        mmproj_output = args.output_dir / f"{args.mmproj.stem}-{gguf_type}.gguf"
        commands.append(
            quantize_one(
                args.quantizer,
                args.mmproj,
                mmproj_output,
                gguf_type,
                args.threads,
                # llama.cpp 4695f00 currently aborts in n_gqa() for a CLIP
                # Q2_K mixture. Pure Q2_K still uses the official quantizer and
                # is the deployable visual recipe mirrored by gguf_k_backend.
                pure=args.method in {"q2_k", "q4_k"},
                dry_run=args.dry_run,
            )
        )

    manifest = {
        "method": args.method,
        "gguf_type": gguf_type,
        "language_input": str(args.model),
        "language_output": None if args.dry_run else str(language_output),
        "mmproj_input": None if args.mmproj is None else str(args.mmproj),
        "mmproj_output": None if args.dry_run or mmproj_output is None else str(mmproj_output),
        "q2_k_mmproj_pure_workaround": args.method == "q2_k" and args.mmproj is not None,
        "commands": commands,
    }
    print(json.dumps(manifest, indent=2))
    if not args.dry_run:
        (args.output_dir / "quantization_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
