#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
adapter.py
==========

Introduces bottleneck adapters into every GPT-NeoX MLP for parameter-efficient
fine-tuning.

Mathematical Context
--------------------
For hidden state x ∈ ℝˢˣʰ:

    down = ReLU(W₁ x + b₁)  ∈ ℝˢˣᵇ
    up   = W₂  down + b₂    ∈ ℝˢˣʰ
    y    = x + up           ∈ ℝˢˣʰ

Where it Fits in the Ablation Study
-----------------------------------
Provides a medium-capacity PEFT baseline whose weights can be (i) trained
directly, or (ii) offset by a hyper-network.

Implementation Outline
----------------------
1) `AdapterTuningWrapper` inserts the bottleneck around a **frozen** MLP.  
2) `ModelAdapter` walks through all GPT-NeoX layers and swaps the `.mlp`
   module with the wrapper, while exposing `.config` for downstream saving.
"""

import os
import logging
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdapterTuningWrapper(nn.Module):
    """Two-layer bottleneck adapter around a frozen GPT-NeoX MLP."""
    def __init__(
        self,
        base_mlp: nn.Module,
        hidden_dim: int,
        bottleneck_dim: int,
        dropout_prob: float = 0.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.base_mlp = base_mlp
        for p in self.base_mlp.parameters():
            p.requires_grad = False

        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim)
        self.up_proj   = nn.Linear(bottleneck_dim, hidden_dim)
        self.dropout   = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
        self.ln        = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

    def forward(self, hidden_states: torch.Tensor, *args, **kw):
        base_out = self.base_mlp(hidden_states, *args, **kw)
        if isinstance(base_out, tuple):
            base_out, residue = base_out[0], base_out[1:]
        else:
            residue = ()
        up  = self.up_proj(self.dropout(F.relu(self.down_proj(base_out))))
        out = self.ln(base_out + up)
        return (out, *residue) if residue else out


class ModelAdapter(nn.Module):
    """Wraps a **frozen** GPT-NeoX backbone and injects a bottleneck adapter in every MLP."""
    def __init__(
        self,
        base_model: nn.Module,
        *,
        bottleneck_dim: Optional[int] = None,
        adapter_bottleneck_dim: Optional[int] = None,  # legacy alias
        adapter_dim: Optional[int] = None,             # legacy alias
        adapter_dropout: float = 0.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        # resolve aliases -------------------------------------------------
        if bottleneck_dim is None:
            bottleneck_dim = adapter_bottleneck_dim or adapter_dim
        if bottleneck_dim is None:
            raise ValueError("Provide `bottleneck_dim` (or alias).")

        # register backbone as a child module -----------------------------
        self.add_module("backbone", base_model)   # -> self.backbone
        self.config    = base_model.config
        self.gpt_neox  = base_model.gpt_neox      # convenience handle

        for p in self.backbone.parameters():
            p.requires_grad = False

        # inject adapters -------------------------------------------------
        h = self.config.hidden_size
        for layer in self.gpt_neox.layers:
            layer.mlp = AdapterTuningWrapper(
                base_mlp     = layer.mlp,
                hidden_dim   = h,
                bottleneck_dim = bottleneck_dim,
                dropout_prob = adapter_dropout,
                use_layer_norm = use_layer_norm,
            )

    # ------------ thin wrappers over backbone ---------------------------
    def forward  (self, *a, **k): return self.backbone(*a, **k)
    def generate (self, *a, **k): return self.backbone.generate(*a, **k)

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "adapter_state.safetensors"))
        self.config.save_pretrained(path)
        logging.info("Adapter checkpoint written to %s", path)