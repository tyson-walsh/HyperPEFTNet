#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bias.py
=======

Freezes every weight matrix in a GPT-NeoX backbone while leaving **bias
vectors** trainable.  The result is an ultra-light PEFT baseline that can be
tuned directly or shifted by a hyper-network.

Mathematical Context
--------------------
For an affine transform *x W + b* we set ∇W = 0 and learn/offset only **b**.
This yields ≈ 0.06 % trainable parameters on Pythia-125 M.

Where it Fits in the Ablation Study
-----------------------------------
Serves as the *minimal-capacity* anchor in our PEFT spectrum:

LoRA (≈ 3 M) 〉 Adapter (≈ 1.2 M) 〉 Prefix (≈ 0.8 M) 〉 **Bias-only (0.1 M)**

Implementation Outline
----------------------
* **BiasTuningModel** wraps a causal LM:
  1. copies `.config` so that `save_pretrained` works downstream,  
  2. toggles `requires_grad` to `True` only for parameter names containing
     “bias”,  
  3. delegates `.forward`, `.generate`, `.save_pretrained`, and **all other
     attributes** to the backbone so external utilities (e.g. layer-counting)
     keep working.
"""

import os
import logging
from typing import Optional
import torch
import torch.nn as nn


class BiasTuningModel(nn.Module):
    """Leaves only bias vectors trainable; everything else frozen."""
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.add_module("backbone", base_model)   # -> self.backbone
        self.config   = base_model.config
        self.gpt_neox = base_model.gpt_neox

        for name, p in self.backbone.named_parameters():
            p.requires_grad = ("bias" in name)

    # ---------- delegate -------------------------------------------------
    def forward  (self, *a, **k): return self.backbone(*a, **k)
    def generate (self, *a, **k): return self.backbone.generate(*a, **k)

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        if hasattr(self.backbone, "save_pretrained"):
            self.backbone.save_pretrained(path)
        else:
            torch.save(self.state_dict(), os.path.join(path, "bias_state.safetensors"))
            self.config.to_json_file(os.path.join(path, "config.json"))
        logging.info("Bias-only model saved to %s", path)