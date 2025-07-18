#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lora.py
=======

Implements a **LoRA (Low-Rank Adaptation)** module for GPT-NeoX’s *merged*
QKV projections, splitting the combined QKV weight matrix into three
low-rank factor pairs and allowing additive updates **p + δ** per user or
per thread.

Mathematical Context
--------------------
Let **W ∈ ℝ^(3h × h)** be the merged QKV matrix.  Decompose each *h × h*
sub-block as

  **W_X + α ⁄ r · (A_X B_X)**   for X ∈ {Q, K, V},

where **A_X ∈ ℝ^(h × r)** and **B_X ∈ ℝ^(r × h)** and *r* ≪ *h*.

Where it Fits in the Ablation Study
-----------------------------------
1. LoRA introduces a small number of parameters *O(r · h)* that can be
   frozen or offset by a hyper-network.  
2. Enables personalized attention adaptation per user/thread.

Implementation Outline
----------------------
1. **LoRAQKV** wraps a single `nn.Linear` base module, freezes it,
   and adds trainable A/B factor pairs for Q, K, V.  
2. The hyper-network produces offsets δA_X, δB_X to these low-rank
   factors.  
3. The forward method is **shape-agnostic and vmap-safe** – it accepts
   input tensors of shape `(..., h)` (any number of leading dimensions)
   so that `torch.func.vmap` can vectorize the model without falling back
   to a slow Python loop.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAQKV(nn.Module):
    """
    Low-Rank Adapter for a *merged* QKV projection in GPT-NeoX.

    • keeps the dense weight frozen  
    • exposes six trainable matrices (A_q/B_q, A_k/B_k, A_v/B_v)  
    • uses the standard scaling α / r
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
    ):
        """
        Parameters
        ----------
        base_linear : frozen `nn.Linear` of shape (3 h, h)  
        rank        : low-rank dimension *r*  
        alpha       : scaling factor α  (ΔW = α / r · A B)
        """
        super().__init__()

        # 1 ─ Freeze the dense projection weight --------------------------------
        self.base_weight = base_linear.weight          # (3h, h)
        self.base_bias   = base_linear.bias            # may be None
        self.base_weight.requires_grad_(False)
        if self.base_bias is not None:
            self.base_bias.requires_grad_(False)

        out_features, in_features = self.base_weight.shape
        if out_features % in_features != 0 or out_features // in_features != 3:
            raise ValueError("Expected merged QKV of shape (3h, h).")

        self.h = in_features
        self.rank = rank
        self.scale = alpha / rank

        # 2 ─ Trainable low-rank factors ---------------------------------------
        def _pair():
            A = nn.Parameter(torch.empty(self.h, rank))
            B = nn.Parameter(torch.empty(rank, self.h))
            nn.init.normal_(A, std=1e-4)
            nn.init.normal_(B, std=1e-4)
            return A, B

        self.A_q, self.B_q = _pair()
        self.A_k, self.B_k = _pair()
        self.A_v, self.B_v = _pair()

    # ------------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : `Tensor` of shape (..., h)  
            Accepts any number of leading dimensions so that the module is
            compatible with `torch.func.vmap`.

        Returns
        -------
        Tensor of shape (..., 3 h) with LoRA update applied.
        """

        *prefix, h = x.shape
        if h != self.h:
            raise ValueError(f"Expected last dim = {self.h}, got {h}")

        x_2d = x.reshape(-1, h)  # flatten arbitrary leading dims

        def _update(A: torch.Tensor, B_: torch.Tensor) -> torch.Tensor:
            # (N,h)·(h,r)·(r,h) → (N,h) then reshape back to prefix + h
            return (x_2d @ B_.T @ A.T).reshape(*prefix, self.h) * self.scale

        lora_q = _update(self.A_q, self.B_q)
        lora_k = _update(self.A_k, self.B_k)
        lora_v = _update(self.A_v, self.B_v)

        base_out = F.linear(x, self.base_weight, self.base_bias)  # (..., 3h)
        base_out = base_out.view(*prefix, 3, self.h)
        lora_out = torch.stack((lora_q, lora_k, lora_v), dim=-2)  # (..., 3, h)

        # ▼ ensure LoRA branch matches AMP-cast dtype to avoid q/k/v mismatch
        return (base_out + lora_out).reshape(*prefix, 3 * self.h).to(x.dtype)
        # ▲ single dtype-cast is the only functional change

    # ------------------------------------------------------------------------- #
    def save_pretrained(self, save_directory: str):
        """
        Persist only the LoRA parameters (A_x/B_x) to *save_directory*.
        """
        os.makedirs(save_directory, exist_ok=True)
        path = os.path.join(save_directory, "lora_state.safetensors")
        torch.save(self.state_dict(), path)
        logging.info("LoRA state saved to %s", path)