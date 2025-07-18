#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prefix.py
=========

Prefix-tuning wrapper for GPT-NeoX backbones.

Key points
----------
* The backbone stays **completely frozen**.
* A learned tensor **P ∈ ℝˡˣʰ** (prefix length × embed dim) is prepended to
  the token-level embeddings.
* The wrapper now builds a full-length `attention_mask` when none is supplied
  by the caller. This prevents the sporadic “key_padding_mask shape” crash
  observed during mixed-precision training.

Notation
--------
B = batch size  
T = original token sequence length  
L = prefix length  
h = embedding dimension
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutput


class ModelPrefixOnly(nn.Module):
    """
    Prefix-tuning module for GPT-NeoX.

    Example
    -------
    ```python
    neo = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1b")
    neo_pref = ModelPrefixOnly(neo, prefix_length=30)

    out = neo_pref(
        input_ids=batch_ids,
        attention_mask=batch_mask,   # optional → will be built if None
        labels=batch_labels,
    )
    loss = out.loss
    ```
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        prefix_length: int = 10,
        embed_dim: int | None = None,
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        # ── register frozen backbone ──────────────────────────────────
        self.add_module("backbone", backbone)
        self.config = backbone.config
        self.gpt_neox = backbone.gpt_neox       # convenience handle

        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # ── learned prefix embeddings ─────────────────────────────────
        embed_dim = embed_dim or self.config.hidden_size
        self.prefix_length = int(prefix_length)

        self.prefix_emb = nn.Parameter(
            torch.empty(self.prefix_length, embed_dim)
        )
        nn.init.normal_(self.prefix_emb, mean=0.0, std=0.02)

        self.dropout = (
            nn.Dropout(dropout_prob) if dropout_prob > 0.0 else nn.Identity()
        )

    # ---------------------------------------------------------------- #
    #  forward
    # ---------------------------------------------------------------- #
    def forward(
        self,
        *,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutput:
        """
        Parameters
        ----------
        input_ids       : (B, T) or (T,) token IDs – **required**.
        attention_mask  : optional; built automatically if omitted.
        labels          : optional; mask for prefix rows is injected.
        **kwargs        : forwarded to the backbone.

        Returns
        -------
        transformers.modeling_outputs.CausalLMOutput
        """

        if input_ids is None:
            raise ValueError("`input_ids` must be provided")

        single_sample = input_ids.dim() == 1
        if single_sample:
            input_ids = input_ids.unsqueeze(0)      # → (1, T)

        B, T = input_ids.shape
        device = input_ids.device
        dtype = self.prefix_emb.dtype

        # ── build input embeddings with prepended prefix ──────────────
        prefix = self.dropout(self.prefix_emb).to(dtype).unsqueeze(0).expand(B, -1, -1)
        tok_emb = self.backbone.gpt_neox.embed_in(input_ids).to(dtype)
        inputs_embeds = torch.cat([prefix, tok_emb], dim=1)           # (B, L+T, h)

        # ── attention mask ────────────────────────────────────────────
        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.long, device=device)
        elif attention_mask.dim() == 1:           # un-batched caller mask
            attention_mask = attention_mask.unsqueeze(0)

        pref_mask = torch.ones(B, self.prefix_length, dtype=torch.long, device=device)
        attention_mask = torch.cat([pref_mask, attention_mask], dim=1)  # (B, L+T)

        # ── labels (insert ignore mask for prefix) ────────────────────
        if labels is not None:
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            pref_lbl = labels.new_full((B, self.prefix_length), -100)
            labels = torch.cat([pref_lbl, labels], dim=1)              # (B, L+T)

        # ── synthetic position IDs to keep RoPE happy ─────────────────
        if kwargs.get("position_ids", None) is None:
            pos_pref = torch.arange(self.prefix_length, device=device).unsqueeze(0)
            pos_real = torch.arange(T, device=device).unsqueeze(0) + self.prefix_length
            kwargs["position_ids"] = torch.cat(
                [pos_pref.expand(B, -1), pos_real.expand(B, -1)], dim=1
            )

        # ── delegate to frozen backbone ───────────────────────────────
        out = self.backbone(
            input_ids=None,                      # embeddings passed instead
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        # ── squeeze batch-dim back for single-sequence calls ──────────
        if single_sample:
            out.loss = None if out.loss is None else out.loss.squeeze(0)
            out.logits = out.logits.squeeze(0)
            if out.hidden_states is not None:
                out.hidden_states = tuple(h.squeeze(0) for h in out.hidden_states)
            if out.attentions is not None:
                out.attentions = tuple(a.squeeze(0) for a in out.attentions)

        return out

    # ---------------------------------------------------------------- #
    #  convenience helpers
    # ---------------------------------------------------------------- #
    def generate(self, *args: Any, **kwargs: Any):
        """Direct passthrough to the backbone `.generate`."""
        return self.backbone.generate(*args, **kwargs)

    def save_pretrained(self, path: str | os.PathLike) -> None:
        """Persist **only** the prefix parameters to *path*."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "prefix_state.safetensors"))
        self.config.save_pretrained(path)
        logging.info("Prefix checkpoint saved to %s", path)