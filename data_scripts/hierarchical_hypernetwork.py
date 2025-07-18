#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hierarchical_hypernetwork.py
==================================

A **hierarchical hyper-network** that learns additive PEFT offsets **δθ**
conditioned on

* a GLOBAL author feature vector *g*  (≈ 59‑D after flattening) and
* optionally an INSTANCE reply vector *i*.

During each forward pass the hyper-net predicts a *batch* of δθ vectors and
feeds them through the frozen PEFT backbone **vectorized** with
`torch.func.vmap` – i.e. only **one** autograd graph is built per batch.
Two guard‑rails keep this fast path robust:

1. If **gradient‑checkpointing** is enabled anywhere inside the backbone the
   code automatically drops to a safe per‑sample loop, because
   PyTorch ≥ 2.0 still forbids nesting `CheckpointFunction` under `vmap`.

2. Functorch also errors when a leaf module performs a shape‑dependent Python
   branch such as  
   `if torch.all(attention_mask == 1): …`.  
   GPT‑NeoX’s SDPA helper does exactly that, so the new
   `_vectorized_forward` hoists such mask checks *outside* the vmap frame and
   slices the dynamic kwargs in lock‑step with δθ. Any remaining
   “data‑dependent control‑flow” runtime is trapped and the call silently
   falls back to the loop implementation. No other pipeline changes are
   required.

Ablation modes
--------------
* **vanilla**   – no hyper‑net, PEFT only  
* **flat**      – δθ = H(g)  
* **hierarch**  – δθ = H([g ; i])

Switching is a single constructor flag `use_instance`.

Mixed‑precision note – δθ is cast to **θ̄.dtype** before addition so
bf16 / fp16 training “just works”.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.nn.utils import stateless

try:  # PyTorch ≥ 2.0
    from torch.func import functional_call as _fcall
    from torch.func import vmap as _vmap
except ImportError:  # PyTorch 1.13 fallback
    from torch.nn.utils.stateless import functional_call as _fcall  # type: ignore
    _vmap = None  # type: ignore

# Optional safetensors I/O
try:
    from safetensors.torch import save_file as _st_save, load_file as _st_load
    _HAVE_SAFETENSORS = True
except ImportError:
    _HAVE_SAFETENSORS = False

# --------------------------------------------------------------------- #
# Approved global feature columns                                       #
# --------------------------------------------------------------------- #
G_SIGNALS: List[str] = [
    "gstat_personality_raw",           # 5‑D
    "gstat_personality_logits",        # 5‑D
    "gstat_personality_traits",        # 5‑D
    "gstat_personality_z",             # 5‑D
    "gstat_gap_sentiment",             # 1
    "gstat_user_sent_var",             # 1
    "gstat_user_len_mean",             # 1
    "gstat_user_ttr",                  # 1
    "gstat_user_post_rate",            # 1
    "gstat_user_subreddit_entropy",    # 1
    "gstat_punct_ratio",               # 1
    "gstat_question_ratio",            # 1
    "gstat_caps_ratio",                # 1
    "gstat_profanity_ratio",           # 1
    "gstat_firstperson_ratio",         # 1
    "gstat_readability_fk",            # 1
    "gstat_weekend_ratio",             # 1
    "gstat_link_ratio",                # 1
    "gstat_reply_delay_mean",          # 1
    "gstat_hour_hist",                 # 24‑D
]

# ───────────────────────────── Hyper-network ────────────────────────── #
class HierarchicalHypernetwork(nn.Module):
    """Tiny 2‑layer ReLU MLP that emits a flat δθ vector."""

    def __init__(
        self,
        global_input_dim: int,
        instance_input_dim: int,
        hidden_dim: int,
        peft_param_count: int,
        *,
        use_instance: bool = True,
        clamp_range: float = 0.05,
    ):
        super().__init__()
        self.use_instance = bool(use_instance)
        self.clamp_range = float(clamp_range)

        in_dim = global_input_dim + (instance_input_dim if self.use_instance else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, peft_param_count),
        )

    # ------------------------------------------------------------------ #
    def forward(
        self,
        global_features: torch.Tensor,                   # (B, G)
        instance_features: torch.Tensor | None = None,   # (B, I) or None
    ) -> torch.Tensor:                                   # (B, P)
        if self.use_instance:
            if instance_features is None:
                raise ValueError("instance_features required but None provided")
            x = torch.cat([global_features, instance_features], dim=-1)
        else:
            x = global_features

        δ = self.net(x)
        return torch.clamp(δ, -self.clamp_range, self.clamp_range)


# ─────────────────── PEFT wrapper that injects δθ offsets ───────────── #
class PEFTHypernetModel(nn.Module):
    """
    Wraps a *frozen* PEFT backbone and splices per‑sample additive
    offsets **δθ** into the placeholder tensors **without ever
    materializing a new `nn.Module` copy**.

    The class offers two execution paths:

    * **Vectorized path** – when `torch.func.vmap` is available and
      gradient‑checkpointing is **disabled**, the entire mini‑batch is
      executed in one functional call, so only **one autograd graph**
      is built.

    * **Safe loop fallback** – if `vmap` is unavailable (PyTorch < 2.0)
      *or* a leaf module inside the backbone still relies on shape‑
      dependent Python control‑flow (e.g. GPT‑NeoX’s SDPA helper under
      grad‑ckpt), the code silently drops to a per‑sample loop that
      yields identical semantics at a modest performance cost.
    """

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        base_peft_model: nn.Module,
        hypernet: HierarchicalHypernetwork,
        *,
        clamp_range: float = 0.01,
    ):
        """
        Parameters
        ----------
        base_peft_model : nn.Module
            A PEFT‑modified GPT‑NeoX (LoRA / Adapter / Bias / Prefix)
            whose **trainable** placeholders are *already frozen*.
        hypernet : HierarchicalHypernetwork
            The tiny MLP that maps (g, i) feature vectors → flat δθ.
        clamp_range : float, optional
            Hard range ± *clamp_range* applied to δθ at runtime.
        """
        super().__init__()
        self.backbone = base_peft_model
        self.hypernet = hypernet
        self.clamp_range = float(clamp_range)

        # ── collect & freeze PEFT placeholders ──────────────────────────
        self._placeholders: Dict[str, torch.Tensor] = OrderedDict()
        for n, p in self.backbone.named_parameters():
            if p.requires_grad:              # PEFT vectors / matrices
                p.requires_grad_(False)
                self._placeholders[n] = p

        # frozen copy θ̄ kept on the same device as the backbone
        dev = next(self.backbone.parameters()).device
        self._theta_bar = {n: p.detach().clone().to(dev)
                           for n, p in self._placeholders.items()}

        tot = sum(p.numel() for p in self._placeholders.values())
        if tot != hypernet.net[-1].out_features:
            raise ValueError(
                f"hyper‑net output {hypernet.net[-1].out_features} ≠ "
                f"{tot} PEFT params detected"
            )

    # ------------------------------------------------------------------ #
    def detach_placeholder_state(self) -> Dict[str, torch.Tensor]:
        """Return θ̄ on CPU (convenient for checkpoints)."""
        return {k: v.detach().cpu() for k, v in self._theta_bar.items()}

    def load_placeholder_state(self, state: Dict[str, torch.Tensor]) -> None:
        """Reload θ̄ onto the current device."""
        dev = next(self.backbone.parameters()).device
        self._theta_bar = {k: t.to(dev) for k, t in state.items()}

    # ---------- (de)serialize hyper‑net + θ̄ together ------------------ #
    def save_hypernet_ckpt(self, path: str | os.PathLike) -> None:
        blob = {
            "__hyper_state__": self.hypernet.state_dict(),
            "__theta__":       self.detach_placeholder_state(),
        }
        if _HAVE_SAFETENSORS and str(path).endswith(".safetensors"):
            flat = {f"hyper.{k}": v for k, v in blob["__hyper_state__"].items()}
            flat.update({f"theta.{k}": v for k, v in blob["__theta__"].items()})
            _st_save(flat, str(path))
        else:
            torch.save(blob, path)

    @staticmethod
    def load_hypernet_ckpt(path: str | os.PathLike,
                           model: "PEFTHypernetModel") -> None:
        if _HAVE_SAFETENSORS and str(path).endswith(".safetensors"):
            flat = _st_load(str(path))
            h_state = {k.removeprefix("hyper."): v
                       for k, v in flat.items() if k.startswith("hyper.")}
            t_state = {k.removeprefix("theta."): v
                       for k, v in flat.items() if k.startswith("theta.")}
        else:
            blob = torch.load(path, map_location="cpu")
            h_state, t_state = blob["__hyper_state__"], blob["__theta__"]

        model.hypernet.load_state_dict(h_state, strict=True)
        model.load_placeholder_state(t_state)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        global_features: torch.Tensor,
        instance_features: torch.Tensor | None = None,
        **backbone_kwargs,
    ):
        # ------------------------------------------------------------------
        # 0. sanity
        # ------------------------------------------------------------------
        if global_features is None:
            raise ValueError("`global_features` is required")

        # ------------------------------------------------------------------
        # 1. predict & clamp δθ
        # ------------------------------------------------------------------
        delta = self.hypernet(global_features, instance_features)
        delta = torch.clamp(delta, -self.clamp_range, self.clamp_range)

        # ------------------------------------------------------------------
        # 2. pick vmap fast-path if allowed
        # ------------------------------------------------------------------
        want_hidden = backbone_kwargs.get("output_hidden_states", False)
        backbone_kwargs["return_dict"] = True          # always easier to merge
        if (_vmap is not None) and (not want_hidden):
            try:
                return self._vectorized_forward(
                    δθ_all=delta,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    **backbone_kwargs,
                )
            except RuntimeError:
                pass                                   # fall back to loop

        # ------------------------------------------------------------------
        # 3. safe per-sample loop
        # ------------------------------------------------------------------
        outputs: List[Any] = []
        sizes = [p.numel() for p in self._placeholders.values()]

        for j in range(delta.size(0)):
            # assemble θ̄ + δθ overrides
            override: Dict[str, torch.Tensor] = {}
            ptr = 0
            for (name, ref), sz in zip(self._placeholders.items(), sizes):
                override[name] = (
                    self._theta_bar[name]
                    + delta[j, ptr : ptr + sz].view_as(ref).to(ref.dtype)
                )
                ptr += sz

            outputs.append(
                stateless.functional_call(
                    self.backbone,
                    override,
                    args=(),
                    kwargs=dict(
                        input_ids=None if input_ids is None else input_ids[j : j + 1],
                        attention_mask=None if attention_mask is None else attention_mask[j : j + 1],
                        labels=None if labels is None else labels[j : j + 1],
                        **backbone_kwargs,
                    ),
                )
            )

        # ------------------------------------------------------------------
        # 4. merge list → single object (keeps hidden_states!)
        # ------------------------------------------------------------------
        def _join(tensors: List[torch.Tensor]) -> torch.Tensor:
            """Stack 0-D tensors, else concatenate along batch-dim."""
            return torch.stack(tensors, 0) if tensors[0].dim() == 0 else torch.cat(tensors, 0)

        # Case A – backbone already returns a dataclass-like object
        if hasattr(outputs[0], "to_dict"):
            template = outputs[0].to_dict()
            merged: Dict[str, Any] = {}
            for key in template:
                vals = [getattr(o, key) for o in outputs]
                if torch.is_tensor(vals[0]):
                    merged[key] = _join(vals)
                elif isinstance(vals[0], tuple) and all(torch.is_tensor(t) for t in vals[0]):
                    merged[key] = tuple(_join(ts) for ts in zip(*vals))
                else:
                    merged[key] = vals
            return type(outputs[0]).from_dict(merged)

        # Case B – backbone returned a plain dict
        if isinstance(outputs[0], dict):
            merged: Dict[str, Any] = {}
            for key in outputs[0]:
                vals = [o[key] for o in outputs]
                if torch.is_tensor(vals[0]):
                    merged[key] = _join(vals)
                elif isinstance(vals[0], tuple) and all(torch.is_tensor(t) for t in vals[0]):
                    merged[key] = tuple(_join(ts) for ts in zip(*vals))
                else:
                    merged[key] = vals

            from transformers.modeling_outputs import CausalLMOutput
            try:
                return CausalLMOutput(**merged)
            except TypeError:
                return type("WrappedOutput", (), merged)()

        # Last resort – unknown container, hand it back unchanged
        return outputs if len(outputs) > 1 else outputs[0]    
                
    # ------------------------------------------------------------------ #
    def _vectorized_forward(self, δθ_all: torch.Tensor, **kw):
        """
        Fully vectorized forward pass that is both **vmap‑safe** *and*
        **SDPA‑safe**.
        """
        B = δθ_all.size(0)
        sizes = [p.numel() for p in self._placeholders.values()]

        # 1. Drop a redundant attention mask (all‑ones) ----------------
        attn = kw.pop("attention_mask", None)
        if attn is not None:
            if not torch.is_tensor(attn) or attn.dim() != 2:
                raise ValueError("attention_mask must be (B, S)")
            if torch.all(attn == 1).item():
                attn = None
        if attn is not None:
            kw["attention_mask"] = attn

        # 2. Separate dynamic (batched) from static kwargs -------------
        dyn_keys = [k for k, v in kw.items()
                    if torch.is_tensor(v) and v.dim() > 0 and v.size(0) == B]
        dyn_vals = [kw[k] for k in dyn_keys]
        static_kw = {k: v for k, v in kw.items() if k not in dyn_keys}

        # --------------------------------------------------------------
        def _apply(delta_row: torch.Tensor, *dyn_tensors):
            # ----- assemble per‑sample kwargs -------------------------
            local_kw = {k: t for k, t in zip(dyn_keys, dyn_tensors)}
            local_kw.update(static_kw)

            # ----- add δθ offsets to θ̄ on‑the‑fly --------------------
            ptr, override = 0, {}
            for (name, ref), sz in zip(self._placeholders.items(), sizes):
                delta = delta_row[ptr : ptr + sz].view_as(ref).to(ref.dtype)
                override[name] = self._theta_bar[name] + delta
                ptr += sz

            return stateless.functional_call(
                self.backbone, override, args=(), kwargs=local_kw
            )

        return _vmap(_apply)(δθ_all, *dyn_vals)