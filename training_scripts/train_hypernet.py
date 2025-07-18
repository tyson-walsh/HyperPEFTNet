#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_hypernet.py
=======================

Multi-GPU training harness for **five** PEFT backbones
(`LoRA`, `LoRA-warm`, `Adapter`, `Bias-only`, `Prefix`).  A tiny MLP
*hyper-network* predicts additive parameter offsets **δθ** on-the-fly
so the frozen PEFT backbone instantly adapts to each Reddit author
(10 000 total, ≈ 3.3 M dialogue turns).

Mathematical Context
--------------------
For every mini-batch of (context **x**, reply **y**, global vector **g**,
optional instance vector **i**):

    δθ = H_ϕ([g ; i])            # hierarchical mode
    δθ = H_ϕ(g)                  # flat mode

    ŷ  = LM(x ; θ̄ + δθ)          # θ̄ = frozen PEFT placeholders
    ϕ⋆ = argmin_ϕ CE(ŷ, y)

Only **ϕ** (hyper-net) is updated; the GPT-NeoX backbone and the PEFT
placeholders **θ̄** remain frozen.

Training Strategy
-----------------
* **Teacher forcing** Input is ``[context || reply]`` but loss is
  masked to reply tokens (context mask = −100).
* **Inference prompt** ``context`` only, replies are generated
  autoregressively with δθ(g,i).

Clamp Scheduling
----------------
The hyper-net output is clipped to ± *clamp(t)*

    clamp(t) = MIN + 0.5·(1-cos(π·t ⁄ T))·(MAX−MIN)

with **MIN = 0.02** and configurable **MAX** (``--max_clamp``,
default = 1.00).  The live value is synchronised every step and the
final ceiling is stored in *hyper_cfg.json* beside each checkpoint.

Learning-Rate Schedule
----------------------
AdamW with 10 % linear warm-up → cosine annealing to zero.

Feature-Importance
------------------
After **each variant finishes its N steps of training** we:

1. **Integrated Gradients** (Captum) on a 2 k-row validation slice.  
2. **SHAP DeepExplainer** on 1 k samples (20-row background).  
3. Persist ``ig_attr.npy`` and ``shap_attr.npy`` in the variant
   checkpoint directory *immediately*, along with a log of the top-10
   IG components.

This means you get attribution vectors as soon as the first variant
(`LoRA` cold-start by default) completes – no need to wait for the
other four.

Where It Fits in the Ablation Study
-----------------------------------
The script produces hyper-net checkpoints **and** their IG/SHAP
vectors.  *evaluate_hypernet.py* later benchmarks perplexity and
qualitative metrics, re-using these saved deltas.

Implementation Outline
----------------------
1. Parse CLI → load train/val Parquet + optional feature tables.  
2. Build *HypernetConversationDataset* with author-level features
   (59D static + 385D dynamic) and optional reply-level features (5D).  
3. **For each PEFT variant**  
     • instantiate frozen backbone, patch with PEFT placeholders,  
     • wrap in flat/hierarchical hyper-net,  
     • enable gradient checkpointing if requested,  
     • train for **N** steps, logging CE/PPL & δθ stats,  
     • run Captum IG + SHAP on held-out mini-batches,  
     • save hyper-net weights, PEFT placeholders, clamp metadata,
       and *ig_attr.npy / shap_attr.npy*,  
     • append variant tag to a checklist to avoid re-runs.  
4. After all variants are trained, reload each checkpoint once more for
   a uniform validation pass and print a final CE/PPL summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional
import hashlib

import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    utils as hf_utils,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / "data_scripts"), str(ROOT / "PEFT_scripts")])

from hypernetwork_dataset import HypernetConversationDataset  # noqa: E402
from hierarchical_hypernetwork import (                            # noqa: E402
    HierarchicalHypernetwork,
    PEFTHypernetModel,
    G_SIGNALS,
)
from adapter import ModelAdapter    # noqa: E402
from bias import BiasTuningModel    # noqa: E402
from lora import LoRAQKV            # noqa: E402
from prefix import ModelPrefixOnly  # noqa: E402


# --------------------------------------------------------------------------- #
#  Global configuration tables
# --------------------------------------------------------------------------- #
PEFT_CFG: Dict[str, Dict] = {
    "lora": {"lora_rank": 32, "lora_alpha": 64.0},
    "adapter": {"adapter_bottleneck_dim": 192, "adapter_dropout": 0.10},
    "bias": {},
    "prefix": {"prefix_length": 20},
}

LR_DEFAULTS: Dict[str, float] = {
    "lora":       2.0e-3,
    "lora_warm":  2.0e-3,
    "adapter":    1.0e-4,
    "bias":       5.0e-4,
    "prefix":     5.0e-4,
}

LORA_COLD_PARAMS: Dict[str, float] = {
    "lr":            4.0e-4,    # learning‑rate
    "hidden_dim":    32,        # width of each MLP block in the hyper‑net
    "n_layers":      9,         # residual depth of the hyper‑net
    "rank":          32,        # low‑rank projection (main size lever)
    "clamp":         0.025,     # final δθ clamp (±)
    "lora_rank":     64,        # LoRA rank inside QKV projections
    "lora_alpha":    128,       # LoRA scaling (≈ 2 × lora_rank)
    "dropout":       0.15,      # dropout inside the hyper‑net
    "weight_decay":  1.0e-4,    # AdamW weight decay
    "warmup_frac":   0.05,      # 5 % linear warm‑up
    "grad_accum":    4,         # gradient‑accumulation steps
    "unclamp_steps": 3000,      # when cosine clamp‑relaxation starts
    "act":           "SiLU",    # activation function
}

REPLY_BUDGET = 128
REPLY_SEP_ID: int | None = None

FIXED_HIDDEN_DIM = 48
FIXED_N_LAYERS   = 11
HNET_RANK        = 32
L2_PEN_COEF      = 1e-4

MIN_CLAMP = 0.02
DEFAULT_MAX_CLAMP = 1.00
DEFAULT_UNCLAMP_STEPS = 3000

DEFAULT_BEST_JSON_PATH = (
    "/sciclone/home/thwalsh/hypernets/results/opt_hypernet_lora.json"
)

# --------------------------------------------------------------------------- #
#  Misc helpers / silence warnings
# --------------------------------------------------------------------------- #
def _silence_warnings() -> None:
    noisy = [
        r"Deterministic behavior was enabled.*cublas",
        r"stateless\.functional_call.*deprecated",
        r"torch\.cuda\.amp\.GradScaler\(.*\).*deprecated",
        r"non-deterministic algorithm.*attention_backward",
        r"Converting a tensor with requires_grad=True to a scalar",
    ]
    libs = r"(?i).*?(torch|transformers).*"
    for pat in noisy:
        warnings.filterwarnings("ignore", message=pat, category=UserWarning, module=libs)
        warnings.filterwarnings("ignore", message=pat, category=FutureWarning, module=libs)
    warnings.filterwarnings("ignore", category=FutureWarning, module=libs)
    warnings.filterwarnings("ignore", category=UserWarning, module=libs)
    warnings.filterwarnings(
        "ignore",
        message=r"IProgress not found\. Please update jupyter and ipywidgets",
        module=r".*tqdm.*",
    )


_silence_warnings()

# --------------------------------------------------------------------------- #
#  JSON helper
# --------------------------------------------------------------------------- #
def _load_best_params(json_path: str | None) -> dict[str, float]:
    if not json_path:
        return {}
    try:
        with open(json_path, "r") as fh:
            data = json.load(fh)
        return data.get("lora", {})
    except Exception as e:
        logging.warning("Unable to parse best‑param JSON (%s): %s", json_path, e)
        return {}

# --------------------------------------------------------------------------- #
#  Utility math helpers
# --------------------------------------------------------------------------- #
def _compute_peft_param_budget(hidden_size: int, n_layers: int) -> dict[str, int]:
    r = PEFT_CFG["lora"]["lora_rank"]
    bdim = PEFT_CFG["adapter"]["adapter_bottleneck_dim"]
    plen = PEFT_CFG["prefix"]["prefix_length"]

    per_layer = {
        "lora": 6 * hidden_size * r,
        "adapter": 2 * hidden_size * bdim,
        "bias": 4 * hidden_size,
    }
    total = {k: v * n_layers for k, v in per_layer.items()}
    total["prefix"] = plen * hidden_size
    return total


def _ppl(ce: float) -> float:
    return math.exp(ce) if ce < 50 else float("inf")


def _align(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if a.size(1) != b.size(1):
        m = min(a.size(1), b.size(1))
        return a[:, :m, :], b[:, :m]
    return a, b


def _merge_logits(obj):
    if hasattr(obj, "logits") and torch.is_tensor(obj.logits):
        return obj.logits
    if isinstance(obj, dict) and "logits" in obj and torch.is_tensor(obj["logits"]):
        return obj["logits"]
    if isinstance(obj, (list, tuple)):
        parts = [_merge_logits(x) for x in obj]
        parts = [p for p in parts if p is not None]
        return torch.cat(parts, dim=0) if parts else None
    return None


def _ce_loss(out, labels: torch.Tensor) -> torch.Tensor:
    """
    Returns a *scalar* cross-entropy suitable for .item() and for backward().
    DataParallel sometimes yields a per-sample vector; we reduce with .mean().
    """
    # -- native loss already computed by the model --
    if getattr(out, "loss", None) is not None:
        loss = out.loss
    elif isinstance(out, dict) and "loss" in out:
        loss = out["loss"]
    else:
        # fall back to manual CE from gathered logits
        logits = _merge_logits(out)
        if logits is None:
            raise ValueError("model output carries no loss / logits")
        l, y = _align(logits[..., :-1, :], labels[..., 1:])
        loss = F.cross_entropy(
            l.reshape(-1, l.size(-1)), y.reshape(-1), ignore_index=-100, reduction="mean"
        )
    # ensure scalar even if shape == (batch,)
    if loss.ndim > 0:
        loss = loss.mean()
    return loss


def _make_concat_inputs(batch: Dict[str, torch.Tensor], pad: int, L: int) -> None:
    if REPLY_SEP_ID == pad:
        raise ValueError("SEP token id matches PAD id.")
    ctx = batch["input_ids"]
    tgt = batch["labels"].clone()
    tgt[tgt == -100] = pad

    ctx = ctx[:, : max(1, L - REPLY_BUDGET)]
    room = L - ctx.size(1) - 1
    tgt_trim = tgt[:, : max(room, 0)]

    sep = torch.full((ctx.size(0), 1), REPLY_SEP_ID, dtype=ctx.dtype, device=ctx.device)
    concat = torch.cat([ctx, sep, tgt_trim], 1)

    if concat.size(1) < L:
        pad_blk = torch.full((concat.size(0), L - concat.size(1)), pad, dtype=ctx.dtype, device=ctx.device)
        concat = torch.cat([concat, pad_blk], 1)

    attn = (concat != pad).long()
    lbl = torch.full_like(concat, -100)
    lbl[:, ctx.size(1) + 1 : ctx.size(1) + 1 + tgt_trim.size(1)] = tgt_trim

    batch.update(input_ids=concat, attention_mask=attn, labels=lbl)


def _ctx_only_ce(logits: torch.Tensor, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    reply_mask = labels != -100
    first_idx = reply_mask.float().argmax(dim=1)
    keep = first_idx > 0
    if keep.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    rows = torch.arange(logits.size(0), device=logits.device)[keep]
    last_ctx = first_idx[keep] - 1
    sel_logits = logits[rows, last_ctx, :]
    sel_targets = input_ids[rows, first_idx[keep]]
    return F.cross_entropy(sel_logits, sel_targets, ignore_index=-100)

# --------------------------------------------------------------------------- #
#  Data loader helpers
# --------------------------------------------------------------------------- #
class PadCollate:
    def __init__(self, gdim: int, idim: int) -> None:
        self.gdim = gdim
        self.idim = idim

    @staticmethod
    def _fix(vec: torch.Tensor, target: int) -> torch.Tensor:
        if vec.shape[0] < target:
            pad = torch.zeros(target - vec.shape[0], dtype=vec.dtype, device=vec.device)
            return torch.cat([vec, pad], dim=0)
        if vec.shape[0] > target:
            return vec[:target]
        return vec

    def __call__(self, batch: List[Dict]) -> Dict:
        for smp in batch:
            smp["global_features"] = self._fix(smp["global_features"], self.gdim)
            if self.idim:
                smp["instance_features"] = self._fix(smp["instance_features"], self.idim)
        return torch.utils.data._utils.collate.default_collate(batch)


def _enable_gradient_checkpointing(m: torch.nn.Module, *, enable: bool = True) -> None:
    if not enable:
        return
    fn = getattr(m, "gradient_checkpointing_enable", None)
    if callable(fn):
        fn()
    elif hasattr(m, "model") and callable(getattr(m.model, "gradient_checkpointing_enable", None)):
        m.model.gradient_checkpointing_enable()

# --------------------------------------------------------------------------- #
#  Backbone builder
# --------------------------------------------------------------------------- #
def _build_peft_backbone(
    variant: str,
    ckpt: str,
    tok,
    *,
    use_grad_ckpt: bool = False,
    placeholders_ckpt: str | None = None,
) -> torch.nn.Module:
    """
    Instantiates the frozen GPT-NeoX backbone and patches it with the chosen
    PEFT mechanism (LoRA, Adapter, Bias, or Prefix).

    * `variant` selects the PEFT flavour.
    * When `variant == "lora_warm"` the placeholders are warm-started from
      `placeholders_ckpt`.
    * The ModelPrefixOnly wrapper is invoked **with keyword arguments** to
      satisfy its signature (prefix_length is keyword-only).
    """
    cfg = AutoConfig.from_pretrained(ckpt, local_files_only=True)
    cfg.use_cache = False
    base = AutoModelForCausalLM.from_pretrained(ckpt, config=cfg, local_files_only=True)
    base.resize_token_embeddings(len(tok))

    # freeze everything – PEFT params will be the only trainables
    for p in base.parameters():
        p.requires_grad_(False)

    sub = PEFT_CFG["lora" if variant == "lora_warm" else variant]

    if variant.startswith("lora"):
        for lyr in base.gpt_neox.layers:
            lyr.attention.query_key_value = LoRAQKV(
                lyr.attention.query_key_value,
                rank=sub.get("lora_rank", 32),
                alpha=sub.get("lora_alpha", 64.0),
            )

    elif variant == "adapter":
        base = ModelAdapter(
            base,
            bottleneck_dim=sub["adapter_bottleneck_dim"],
            adapter_dropout=sub["adapter_dropout"],
            use_layer_norm=True,
        )

    elif variant == "bias":
        base = BiasTuningModel(base)

    elif variant == "prefix":
        # keyword arguments – compatible with ModelPrefixOnly
        base = ModelPrefixOnly(
            backbone=base,
            prefix_length=sub["prefix_length"],
            embed_dim=base.config.hidden_size,
        )

    else:
        raise ValueError(f"Unknown PEFT variant: {variant}")

    if variant == "lora_warm":
        if placeholders_ckpt is None:
            raise ValueError("lora_warm requested but --warm_start_lora_ckpt not provided.")
        state = torch.load(placeholders_ckpt, map_location="cpu")
        base.load_state_dict(state, strict=False)

    _enable_gradient_checkpointing(base, enable=use_grad_ckpt)
    return base


# --------------------------------------------------------------------------- #
#  Hyper‑network wrapper
# --------------------------------------------------------------------------- #
def _wrap_with_hypernet(
    backbone: torch.nn.Module,
    gdim: int,
    idim: int,
    hierarchical: bool,
    device: torch.device,
    clamp_val: float,
    *,
    hidden_dim: int = FIXED_HIDDEN_DIM,
    n_layers: int = FIXED_N_LAYERS,
    rank: int = HNET_RANK,
    dropout: float = 0.10,
    act: str = "SiLU",
) -> Tuple[PEFTHypernetModel, int, int]:
    backbone = backbone.to(device)
    peft_cnt = sum(p.numel() for p in backbone.parameters() if p.requires_grad)

    act_cls: Callable[[], torch.nn.Module] = getattr(torch.nn, act, torch.nn.SiLU)

    class Residual(torch.nn.Module):
        def __init__(self, dim: int, hidden: int):
            super().__init__()
            self.fc1  = torch.nn.Linear(dim, hidden,  bias=True)
            self.act1 = act_cls()
            self.fc2  = torch.nn.Linear(hidden, dim, bias=True)
            self.act2 = act_cls()
            self.drop = torch.nn.Dropout(dropout)
            self.ln   = torch.nn.LayerNorm(dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.act1(self.fc1(x))
            h = self.act2(self.fc2(h))
            return self.ln(x + self.drop(h))

    in_dim = gdim + (idim if hierarchical else 0)
    blocks: List[torch.nn.Module] = [
        torch.nn.Linear(in_dim, hidden_dim, bias=True),
        act_cls(),
    ]
    for _ in range(max(0, n_layers - 2)):
        blocks.append(Residual(hidden_dim, hidden_dim * 2))
    blocks.extend(
        [
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, rank, bias=False),
            torch.nn.Linear(rank, peft_cnt,    bias=True),
        ]
    )

    hyper = HierarchicalHypernetwork(
        global_input_dim=gdim,
        instance_input_dim=idim,
        hidden_dim=hidden_dim,
        peft_param_count=peft_cnt,
        use_instance=hierarchical,
    )
    hyper.net = torch.nn.Sequential(*blocks)

    for m in hyper.net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.uniform_(m.weight, -0.02, 0.02)
            if m.bias is not None:
                torch.nn.init.uniform_(m.bias, -0.02, 0.02)

    model = PEFTHypernetModel(backbone, hyper, clamp_range=clamp_val).to(device)
    hyper_cnt = sum(p.numel() for p in hyper.parameters() if p.requires_grad)
    return model, peft_cnt, hyper_cnt

# --------------------------------------------------------------------------- #
#  LR scheduler
# --------------------------------------------------------------------------- #
def _build_scheduler(opt: torch.optim.Optimizer, steps: int, warm_frac: float = 0.10):
    warm = max(1, int(steps * warm_frac))
    rest = max(1, steps - warm)
    return torch.optim.lr_scheduler.SequentialLR(
        opt,
        [
            torch.optim.lr_scheduler.LinearLR(opt, 0.1, 1.0, warm),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=rest),
        ],
        [warm],
    )
        
        
# --------------------------------------------------------------------------- #
#  Evaluation helper
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _eval_split(
    model: "PEFTHypernetModel | torch.nn.DataParallel",
    dl: DataLoader,
    device: torch.device,
    pad_id: int,
    seq_len: int,
    use_amp: bool,
    *,
    variant: str,
    split_name: str,
    progress_pct: float = 0.10,
) -> Tuple[float, float]:
    """
    Returns (CE teacher-forcing, CE ctx-only) for a dataset slice.
    """
    model.eval()
    tot_tf = tot_ctx = tok_tf = tok_ctx = 0.0
    n_seen = 0
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    for batch in dl:
        if n_seen / len(dl.dataset) >= progress_pct:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        _make_concat_inputs(batch, pad_id, seq_len)

        with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                global_features=batch["global_features"],
                instance_features=batch["instance_features"],
            )
        logits = _merge_logits(out)

        nt = (batch["labels"] != -100).sum().item()
        nc = logits.size(0)

        tot_tf  += _ce_loss(out, batch["labels"]).item()         * nt
        tot_ctx += _ctx_only_ce(logits, batch["input_ids"], batch["labels"]).item() * nc
        tok_tf  += nt
        tok_ctx += nc
        n_seen  += batch["input_ids"].size(0)

    return tot_tf / max(tok_tf, 1), tot_ctx / max(tok_ctx, 1)


# --------------------------------------------------------------------------- #
#  Training loop (added warm_frac parameter)
# --------------------------------------------------------------------------- #
def _train(
    model: PEFTHypernetModel,
    dl_tr: DataLoader,
    dl_val: DataLoader,
    *,
    device: torch.device,
    steps: int,
    log_int: int,
    lr: float,
    wd: float,
    grad_accum: int,
    warm_frac: float,
    variant: str,
    pad_id: int,
    seq_len: int,
    max_clamp: float,
    unclamp_steps: int,
    use_amp: bool,
    early_stop_patience: int = 5,
    early_stop_delta: float = 0.002,
    eval_every: int | None = None,
    on_save: Optional[Callable[[PEFTHypernetModel], None]] = None,
) -> Dict[str, float]:
    """
    Trains the hyper‑network **with early stopping**.  
    Training halts when `eval_every` successive validation
    evaluations fail to improve CE by at least `early_stop_delta`.
    """
    eval_every = eval_every or log_int
    tgt_model = model.module if hasattr(model, "module") else model

    opt = torch.optim.AdamW(
        [{"params": tgt_model.hypernet.parameters(), "lr": lr}],
        weight_decay=wd,
    )
    sched = _build_scheduler(opt, steps, warm_frac)
    scaler = GradScaler(enabled=use_amp)

    ce_sum = tok_sum = 0.0
    best_val = float("inf")
    since_best = 0                           # counts #evaluations since last improv.
    step = skipped = 0
    t0 = time.time()

    model.train()
    opt.zero_grad(set_to_none=True)

    gpu_idx = (device.index or 0) if device.type == "cuda" else 0
    bf16_ok = device.type == "cuda" and torch.cuda.get_device_capability(gpu_idx)[0] >= 8
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16

    while step < steps:
        progressed = False
        for b_idx, batch in enumerate(dl_tr):
            batch = {k: v.to(device) for k, v in batch.items()}
            _make_concat_inputs(batch, pad_id, seq_len)

            nt = (batch["labels"] != -100).sum().item()
            if nt == 0:
                skipped += 1
                continue

            clamp = (MIN_CLAMP if step < unclamp_steps else
                     MIN_CLAMP + 0.5 * (1 - math.cos(
                         math.pi * (step - unclamp_steps) / max(1, steps - unclamp_steps)
                     )) * (max_clamp - MIN_CLAMP))
            tgt_model.clamp_range = tgt_model.hypernet.clamp_range = clamp

            with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    global_features=batch["global_features"],
                    instance_features=batch["instance_features"],
                )
                dtheta = tgt_model.hypernet(batch["global_features"], batch["instance_features"])
                l2_pen = L2_PEN_COEF * dtheta.pow(2).mean()
                loss = _ce_loss(out, batch["labels"]) / grad_accum + l2_pen

            scaler.scale(loss).backward()
            ce_sum += loss.item() * nt * grad_accum
            tok_sum += nt

            if (b_idx + 1) % grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(tgt_model.hypernet.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                progressed = True

                # ---- logging ----
                if step % log_int == 0 or step == steps:
                    mce = ce_sum / tok_sum
                    logging.info(
                        "[%-8s] step %5d/%d • CE %.4f • PPL %.1f • clamp ±%.3f",
                        variant, step, steps, mce, _ppl(mce), clamp
                    )

                # ---- periodic validation & early‑stop check ----
                if step % eval_every == 0 or step == steps:
                    val_tf, _ = _eval_split(
                        model, dl_val, device, pad_id, seq_len, use_amp,
                        variant=variant, split_name="val", progress_pct=0.05
                    )
                    logging.info("[%-8s] step %d • Val‑CE‑TF %.4f", variant, step, val_tf)

                    if val_tf + early_stop_delta < best_val:
                        best_val = val_tf
                        since_best = 0
                    else:
                        since_best += 1
                        if since_best >= early_stop_patience:
                            logging.info(
                                "[%s] early stopping triggered (no improvement for %d evals)",
                                variant, early_stop_patience
                            )
                            step = steps  # force outer while‑loop exit
                            break

        if not progressed:
            logging.error("[%s] no usable batches – aborting", variant)
            break

    if on_save is not None:
        on_save(model)

    train_tf, _ = _eval_split(
        model, dl_tr, device, pad_id, seq_len, use_amp,
        variant=variant, split_name="train", progress_pct=0.05
    )
    val_tf, _ = _eval_split(
        model, dl_val, device, pad_id, seq_len, use_amp,
        variant=variant, split_name="val",   progress_pct=0.10
    )

    return {
        "train_ce_tf": train_tf,
        "val_ce_tf":   val_tf,
        "elapsed":     time.time() - t0,
    }


# --------------------------------------------------------------------------- #
#  Main entry‑point
# --------------------------------------------------------------------------- #
def main() -> None:
    t0_wall = time.time()

    ap = argparse.ArgumentParser()
    ap.add_argument("--train_parquet", required=True)
    ap.add_argument("--val_parquet", required=True)
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--best_hyper_json", default=DEFAULT_BEST_JSON_PATH, help="Path to opt_hypernet_lora.json",)
    ap.add_argument("--models_output_dir", required=True)
    ap.add_argument("--train_steps", type=int, default=10_000)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--weight_decay", type=float, default=1.5e-4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--log_interval", type=int, default=500)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--checklist", default="/sciclone/home/thwalsh/hypernets/log_files/train_hypernet_checklist.txt",)
    ap.add_argument("--global_features_parquet")
    ap.add_argument("--disable_gstats")
    ap.add_argument("--instance_features_parquet")
    ap.add_argument("--flat_hypernet", action="store_true")
    ap.add_argument("--hierarchical_hypernet", action="store_true")
    ap.add_argument("--max_clamp", type=float, default=1.0)
    ap.add_argument("--unclamp_steps", type=int, default=DEFAULT_UNCLAMP_STEPS)
    ap.add_argument("--use_grad_ckpt", action="store_true")
    ap.add_argument("--demo_mode", action="store_true")
    ap.add_argument("--warm_start_lora_ckpt")
    ap.add_argument("--variants", default="lora,lora_warm,adapter,bias,prefix", help="Comma-separated list",)
    ap.add_argument("--adapter_dim", type=int, help="Bottleneck size for Adapter variant "
                     "(default 192 for capacity parity; "
                     "use 64–128 for faster training)")
    ap.add_argument("--num_workers", type=int, default=0)

    args = ap.parse_args()
    
    if args.adapter_dim:
        PEFT_CFG["adapter"]["adapter_bottleneck_dim"] = args.adapter_dim
        logging.info("Adapter bottleneck overridden → %d", args.adapter_dim)
    
    if not (args.flat_hypernet ^ args.hierarchical_hypernet):
        ap.error("pass exactly one of --flat_hypernet / --hierarchical_hypernet")

    if args.demo_mode:
        logging.warning("[DEMO] mode – rows/steps capped")
        args.train_steps = min(args.train_steps, 20)
        args.batch_size = min(args.batch_size, 4)
        _skip_io = False
        demo_suffix = "demo_runs"
    else:
        _skip_io = False
        demo_suffix = ""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    hf_utils.logging.set_verbosity_error()

    set_seed(142)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.fp32

    logging.info("Loading train parquet …")
    df_tr = pd.read_parquet(args.train_parquet, engine="pyarrow", memory_map=True)
    logging.info("Loading validation parquet …")
    df_val = pd.read_parquet(args.val_parquet, engine="pyarrow", memory_map=True)

    if args.hierarchical_hypernet and args.instance_features_parquet:
        logging.info("Loading instance-feature parquet …")
        idf = pd.read_parquet(args.instance_features_parquet)
    else:
        idf = pd.DataFrame()

    tok = AutoTokenizer.from_pretrained(args.base_ckpt, local_files_only=True)
    if "<|reply|>" not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": ["<|reply|>"]})
    global REPLY_SEP_ID
    REPLY_SEP_ID = tok.convert_tokens_to_ids("<|reply|>")
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        tok.pad_token_id = tok.convert_tokens_to_ids("[PAD]")

    if args.global_features_parquet and args.global_features_parquet != "/dev/null":
        logging.info("Loading global-feature parquet …")
        gdf = pd.read_parquet(args.global_features_parquet, engine="pyarrow", memory_map=True)
    else:
        logging.info("Building global-feature table …")
        feat_cols = [c for c in df_tr.columns if c.startswith("gstat_")]
        gdf = df_tr[["gid"] + feat_cols].drop_duplicates("gid").reset_index(drop=True)

    gcols = [c for c in gdf.columns if c != "gid"]
    if args.disable_gstats:
        drop = {c.strip() for c in args.disable_gstats.split(",") if c.strip()}
        gcols = [c for c in gcols if c not in drop]
        logging.info("Global features disabled → %s", sorted(drop))

    icols = (
        [c for c in idf.columns if c not in ("gid", "target_user_id")]
        if args.hierarchical_hypernet
        else []
    )

    ds_tr = HypernetConversationDataset(
        df_tr, tok, gdf, idf, hierarchical=args.hierarchical_hypernet, max_length=args.max_len
    )
    ds_val = HypernetConversationDataset(
        df_val, tok, gdf, idf, hierarchical=args.hierarchical_hypernet, max_length=args.max_len
    )
    ds_tr.set_selected_features(gcols, icols)
    ds_val.set_selected_features(gcols, icols)

    if args.demo_mode:
        from torch.utils.data import Subset

        ds_tr = Subset(ds_tr, range(min(len(ds_tr), 256)))
        ds_val = Subset(ds_val, range(min(len(ds_val), 64)))
        logging.info("[DEMO] dataset clipped → %d train / %d val", len(ds_tr), len(ds_val))

    sample = ds_tr[0]
    gdim, idim = sample["global_features"].shape[0], sample["instance_features"].shape[0]
    logging.info(
        "Mode=%s • Train=%d • Val=%d • gdim=%d • idim=%d",
        "hierarchical" if args.hierarchical_hypernet else "flat",
        len(ds_tr),
        len(ds_val),
        gdim,
        idim,
    )

    pad_collate = PadCollate(gdim, idim)
    dl_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=bool(args.num_workers),
        collate_fn=pad_collate,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=bool(args.num_workers),
        collate_fn=pad_collate,
    )

    ck = Path(args.checklist)
    ck.parent.mkdir(parents=True, exist_ok=True)
    done = {ln.rsplit("_", 1)[0] for ln in ck.read_text().splitlines()} if ck.exists() else set()

    out_root = Path(args.models_output_dir)
    if demo_suffix:
        out_root = out_root / demo_suffix
    out_root.mkdir(parents=True, exist_ok=True)

    log_int = args.log_interval if args.log_interval > 0 else max(1, args.train_steps // 10)
    pad_id = tok.pad_token_id
    variant_list = [v.strip() for v in args.variants.split(",") if v.strip()]
    summary: List[Tuple[str, float, float]] = []

    for variant in variant_list:
        if variant in done:
            logging.info("[%s] already trained – skipping", variant)
            continue

        if variant == "lora":                     # cold-start LoRA
            # use the hand-tuned, hard-coded recipe – ignore external JSONs
            tuned = LORA_COLD_PARAMS.copy()
        else:
            tuned = {}                            # other variants start from defaults

        # baseline settings (may be overridden further below)
        lr_val            = tuned.get("lr", args.lr or LR_DEFAULTS[variant])
        hidden_dim        = tuned.get("hidden_dim", FIXED_HIDDEN_DIM)
        n_layers          = tuned.get("n_layers", FIXED_N_LAYERS)
        rank_dim          = tuned.get("rank", HNET_RANK)
        max_clamp_local   = tuned.get("clamp", args.max_clamp)
        dropout_val       = tuned.get("dropout", 0.10)
        act_val           = tuned.get("act", "SiLU")
        warm_frac_val     = tuned.get("warmup_frac", 0.10)
        wd_val            = tuned.get("weight_decay", args.weight_decay)
        grad_accum_val    = tuned.get("grad_accum", args.grad_accum)
        unclamp_steps_val = tuned.get("unclamp_steps", args.unclamp_steps)

        # warm-start LoRA needs a tighter leash – stop the clamp & LR from exploding
        if variant == "lora_warm":
            lr_val            = 5.0e-4 if args.lr is None else args.lr   # halve the LR
            max_clamp_local   = 0.05                                     # hard ceiling
            unclamp_steps_val = 1500                                     # delay spread

        if variant == "lora":
            PEFT_CFG["lora"]["lora_rank"]  = tuned.get("lora_rank", PEFT_CFG["lora"]["lora_rank"])
            PEFT_CFG["lora"]["lora_alpha"] = tuned.get("lora_alpha", PEFT_CFG["lora"]["lora_alpha"])

        logging.info(
            "[%s] hyper-params → %s",
            variant,
            json.dumps(tuned or {"lr": lr_val}, separators=(",", ":")),
        )

        backbone = _build_peft_backbone(
            variant,
            args.base_ckpt,
            tok,
            use_grad_ckpt=args.use_grad_ckpt,
            placeholders_ckpt=args.warm_start_lora_ckpt
            if variant == "lora_warm"
            else None,
        )

        model, cnt_place, cnt_hnet = _wrap_with_hypernet(
            backbone,
            gdim,
            idim,
            hierarchical=args.hierarchical_hypernet,
            device=device,
            clamp_val=MIN_CLAMP,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            rank=rank_dim,
            dropout=dropout_val,
            act=act_val,
        )

        total_trainable = cnt_place + cnt_hnet
        total_backbone = sum(p.numel() for p in backbone.parameters())
        prc = 100.0 * total_trainable / max(1, total_backbone)

        human = (
            lambda n: f"{n/1e6:,.2f} M"
            if n >= 1e6
            else f"{n/1e3:,.1f} K"
        )

        logging.info(
            "[%s] PEFT placeholders %s • hyper-net %s • trainable %s "
            "(%.2f %% of backbone)",
            variant,
            human(cnt_place),
            human(cnt_hnet),
            human(total_trainable),
            prc,
        )

        if torch.cuda.device_count() > 1:
            logging.info(
                "[%s] DataParallel across %d GPUs",
                variant,
                torch.cuda.device_count(),
            )

            def _theta_bar_device_fix(mod, args_, kwargs_):
                ref = next(
                    (
                        t
                        for t in list(args_) + list(kwargs_.values())
                        if torch.is_tensor(t)
                    ),
                    None,
                )
                if ref is None:
                    return
                tgt = ref.device

                def _migrate(obj):
                    tb = getattr(obj, "_theta_bar", None)
                    if isinstance(tb, dict) and any(
                        t.device != tgt for t in tb.values()
                    ):
                        obj._theta_bar = {
                            k: t.to(tgt, non_blocking=True).clone()
                            for k, t in tb.items()
                        }
                    for ch in obj.children():
                        _migrate(ch)

                _migrate(mod)

            model = torch.nn.DataParallel(model)
            model.module.register_forward_pre_hook(
                _theta_bar_device_fix, with_kwargs=True
            )
        tag = "hier_hypernet" if args.hierarchical_hypernet else "flat_hypernet"
        vdir = out_root / f"{tag}_{variant}_model"
        vdir.mkdir(parents=True, exist_ok=True)

        def _do_save(m: PEFTHypernetModel) -> None:
            if _skip_io:
                logging.info("[DEMO] skipping checkpoint write for %s", variant)
                return
            save_model = m.module if isinstance(m, torch.nn.DataParallel) else m
            torch.save(save_model.detach_placeholder_state(), vdir / "peft_placeholders.safetensors")
            torch.save(save_model.hypernet.state_dict(), vdir / "hypernetwork.safetensors")
            backbone.config.save_pretrained(vdir)
            (vdir / "gfeat_columns.json").write_text(json.dumps(gcols))
            (vdir / "hyper_cfg.json").write_text(json.dumps({"clamp": float(max_clamp_local)}))
            with ck.open("a") as fh:
                fh.write(f"{variant}_{hashlib.md5(vdir.as_posix().encode()).hexdigest()[:6]}\n")
            logging.info("[%s] checkpoint saved", variant)

        metrics = _train(
            model,
            dl_tr,
            dl_val,
            device=device,
            steps=args.train_steps,
            log_int=log_int,
            lr=lr_val,
            wd=wd_val,
            grad_accum=grad_accum_val,
            warm_frac=warm_frac_val,
            variant=variant,
            pad_id=pad_id,
            seq_len=args.max_len,
            max_clamp=max_clamp_local,
            unclamp_steps=unclamp_steps_val,
            use_amp=use_amp,
            early_stop_patience=5,
            early_stop_delta=0.002,
            eval_every=log_int,
            on_save=_do_save,
        )

    # ---------------------------------------------------------------------
    # SECOND PASS ­– reload every checkpoint once for a uniform evaluation
    # ---------------------------------------------------------------------
    for variant in variant_list:
        tag  = "hier_hypernet" if args.hierarchical_hypernet else "flat_hypernet"
        vdir = out_root / f"{tag}_{variant}_model"
        if not vdir.exists():
            logging.warning("[%s] no checkpoint – skipping eval", variant)
            continue

        # -------- infer the input‑feature dimensionality directly
        #          from the saved hyper‑network weights ------------
        try:
            hnet_state = torch.load(vdir / "hypernetwork.safetensors", map_location="cpu")
        except FileNotFoundError:
            logging.warning("[%s] hypernetwork.safetensors missing – skipping", variant)
            continue

        # find the very first Linear layer weight inside hyper‑net
        first_key = next(k for k in hnet_state if k.endswith("net.0.weight"))
        gdim_ckpt = hnet_state[first_key].size(1)          # e.g. 22

        # -------- try to load the feature‑column list (optional) ------------
        gcols_ckpt: List[str] = []
        feat_json = vdir / "gfeat_columns.json"
        if feat_json.exists():
            try:
                gcols_ckpt = json.loads(feat_json.read_text())
            except Exception as e:
                logging.warning("[%s] could not parse gfeat_columns.json (%s)", variant, e)

        # sanity‑check: if the JSON length disagrees with the weight shape,
        # trust the weight shape (it never lies)
        if gcols_ckpt and len(gcols_ckpt) != gdim_ckpt:
            logging.warning(
                "[%s] gfeat_columns.json len=%d ≠ inferred gdim=%d – ignoring JSON",
                variant, len(gcols_ckpt), gdim_ckpt,
            )
            gcols_ckpt = []

        # make the validation dataset emit exactly gdim_ckpt globals
        if isinstance(ds_val, torch.utils.data.Subset):
            base_val_ds = ds_val.dataset          # unwrap demo Subset
        else:
            base_val_ds = ds_val

        base_val_ds.set_selected_features(
            base_val_ds.global_columns[:gdim_ckpt],  # always start from full list
            icols
        )

        pad_collate_val = PadCollate(gdim_ckpt, idim)
        dl_val = DataLoader(
            ds_val,                       # keep same underlying dataset object
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=bool(args.num_workers),
            collate_fn=pad_collate_val,
        )

        clamp_cfg = (
            json.loads((vdir / "hyper_cfg.json").read_text())
            .get("clamp", args.max_clamp)
        )

        backbone = _build_peft_backbone(
            variant, args.base_ckpt, tok, use_grad_ckpt=False
        )
        model, _, _ = _wrap_with_hypernet(
            backbone,
            gdim_ckpt,
            idim,
            hierarchical=args.hierarchical_hypernet,
            device=device,
            clamp_val=clamp_cfg,
        )

        # load the saved weights ­– now guaranteed to match sizes
        model.backbone.load_state_dict(
            torch.load(vdir / "peft_placeholders.safetensors", map_location="cpu"),
            strict=False,
        )
        model.hypernet.load_state_dict(hnet_state, strict=True)
        model.clamp_range = model.hypernet.clamp_range = clamp_cfg

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        val_tf, val_ctx = _eval_split(
            model,
            dl_val,
            device,
            pad_id,
            args.max_len,
            use_amp,
            variant=variant,
            split_name="val",
            progress_pct=0.10,
        )
        summary.append((variant, val_tf, val_ctx))
        logging.info(
            "[%-8s] Val‑CE‑TF %.4f (PPL %.1f) • Val‑CE‑CTX %.4f",
            variant, val_tf, _ppl(val_tf), val_ctx
        )
        
    if summary:
        logging.info("========= FINAL SUMMARY =========")
        for v, ce_tf, ce_ctx in summary:
            logging.info("Variant %-8s | Val-CE-TF %.4f | PPL %.1f | Val-CE-CTX %.4f", v, ce_tf, _ppl(ce_tf), ce_ctx)
        logging.info("=================================")

    logging.info("Run complete • wall-time %.1fs", time.time() - t0_wall)


if __name__ == "__main__":
    if torch.multiprocessing.get_start_method(allow_none=True) != "spawn":
        torch.multiprocessing.set_start_method("spawn", force=True)
    _T0 = time.time()
    main()
    print(f"Script runtime: {time.time() - _T0:.1f}s")