from typing import Literal

import bitsandbytes.functional as bnbF
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def compute_code_from_4bit(w_4bit: torch.Tensor, quant_state: bnbF.QuantState):
    """
    Recover per-value 4-bit code indices from a packed 8-bit tensor and map them to codebook values.

    Notes:
    - Each uint8 element in w_4bit stores two 4-bit codes (high nibble and low nibble).
    - This function extracts both nibbles, flattens them into a 1D index tensor, and then
      looks up the corresponding quantized values from quant_state.code.
    """
    first = (w_4bit >> 4) & 0xF  
    second = w_4bit & 0xF  

    coded_idx = torch.stack([first, second], dim=1).flatten().long() 
    coded_value = quant_state.code[coded_idx]

    return coded_idx, coded_value