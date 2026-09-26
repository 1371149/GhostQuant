#!/usr/bin/env python3
"""Fast, GPU-free validation of release data and public pipeline definitions."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghostquant.pipeline import CASCADE12_STAGES, STATIC3_STAGES


def validate_manifest(path: Path, expected_count: int) -> None:
    records = json.loads(path.read_text(encoding="utf-8"))
    assert len(records) == expected_count, (path, len(records))
    for record in records:
        image = Path(record["image"])
        assert not image.is_absolute(), (path, image)
        assert (path.parent / image).is_file(), (path, image)
        assert len(record["conversations"]) == 2, (path, record)


def validate_checksums(directory: Path) -> None:
    for line in (directory / "checksums.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        actual = hashlib.sha256((directory / relative).read_bytes()).hexdigest()
        assert actual == expected, relative


def main() -> None:
    task1 = ROOT / "data" / "task1"
    validate_manifest(task1 / "train.json", 2800)
    validate_manifest(task1 / "validation.json", 200)
    validate_manifest(task1 / "test.json", 200)
    validate_checksums(task1)

    calibration = ROOT / "data" / "calibration"
    validate_manifest(calibration / "textvqa_A_128.json", 128)
    validate_checksums(calibration)
    calibration_records = json.loads((calibration / "textvqa_A_128.json").read_text())
    for record in calibration_records:
        image = calibration / record["image"]
        assert hashlib.sha256(image.read_bytes()).hexdigest() == record["image_sha256"]

    assert [s.method for s in STATIC3_STAGES] == ["fp4", "nf4", "int8"]
    assert [s.method for s in CASCADE12_STAGES] == [
        "fp4", "nf4", "int8", "q2_k", "q3_k_m", "q4_k_s",
        "q4_k_m", "q5_k_m", "q6_k", "awq_w3a16", "awq_w4a16", "awq_w4a8",
    ]
    for filename, stages in (
        ("qwen3vl2b_static3.yaml", STATIC3_STAGES),
        ("qwen3vl2b_cascade12.yaml", CASCADE12_STAGES),
    ):
        documented = yaml.safe_load((ROOT / "configs" / filename).read_text())
        assert documented["stages"] == [
            {
                "method": stage.method,
                "kind": stage.kind,
                "relative_limit": stage.relative_limit,
            }
            for stage in stages
        ]
    print("release validation passed")


if __name__ == "__main__":
    main()
