#!/usr/bin/env python3
"""Run the FP4 -> NF4 -> INT8 Qwen3-VL-2B cascade, then test final tASR."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghostquant.pipeline import STATIC3_STAGES, main


if __name__ == "__main__":
    main("qwen3vl2b_static3", STATIC3_STAGES)
