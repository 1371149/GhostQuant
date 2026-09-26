#!/usr/bin/env python3
"""Run the 12-stage Qwen3-VL-2B cascade, then test final tASR."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghostquant.pipeline import CASCADE12_STAGES, main


if __name__ == "__main__":
    main("qwen3vl2b_cascade12", CASCADE12_STAGES)
