#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_nonhypernet.py
==========================

Single‑GPU training of five nonhypernet-baselines — LoRA, Adapter, Bias‑only, Prefix,
and **Base full fine‑tune** — on the 10000‑user Reddit corpus
(≈3.3M lines).  No hyper‑network: the PEFT placeholders **θ̄** (or the full
model for *base*) are trained directly.

Training Strategy
-----------------
*Teacher forcing* — model sees **[context || reply]**; loss on reply tokens
only (context masked to −100).

Learning‑rate schedule
----------------------
AdamW, **3% warm‑up** → cosine decay to 0.

Implementation Outline
----------------------
1. Parse CLI; build `RedditConversationDataset`.
2. For each variant  
  • freeze backbone (except *base*), insert PEFT wrapper,  
  • enable gradient checkpointing when available,  
  • train `--train_steps`, logging CE / PPL,  
  • save PEFT or full checkpoints, append variant to checklist.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast                    # modern autocast
from torch.cuda.amp import GradScaler             # same as hyper-net
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    utils as hf_utils,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ──────────────────────────────────────────────────────────────
# Silence noisy warnings (matches hyper-net script behavior)
# ──────────────────────────────────────────────────────────────
def _silence_warnings() -> None:
    import warnings

    noisy = [
        r"Deterministic behavior was enabled.*cublas",
        r"stateless\.functional_call.*deprecated",
        r"torch\.cuda\.amp\.GradScaler\(.*\).*deprecated",
        r"torch\.cuda\.amp\.autocast\(.*\).*deprecated",
        r"non-deterministic algorithm.*attention_backward",
        r"Converting a tensor with requires_grad=True to a scalar",
    ]
    libs = r"(?i).*?(torch|transformers).*"
    for pat in noisy:
        warnings.filterwarnings("ignore", message=pat, category=UserWarning,  module=libs)
        warnings.filterwarnings("ignore", message=pat, category=FutureWarning, module=libs)
    warnings.filterwarnings("ignore", category=FutureWarning, module=libs)
    warnings.filterwarnings("ignore", category=UserWarning,  module=libs)


_silence_warnings()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPLY_BUDGET = 128  # tokens always kept free for the reply
REPLY_SEP_ID: int | None = None  # special token ID for <|reply|> in tokenizer

# ---------------------------------------------------------------------------
# Local modules and directories
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / "data_scripts"), str(ROOT / "PEFT_scripts")])
OPT_DIR_DEFAULT = "/sciclone/home/thwalsh/hypernets/results"

from dataset import RedditConversationDataset  # noqa: E402
from adapter import ModelAdapter  # noqa: E402
from bias import BiasTuningModel  # noqa: E402
from lora import LoRAQKV  # noqa: E402
from prefix import ModelPrefixOnly  # noqa: E402

# ---------------------------------------------------------------------------
# PEFT configuration and bookkeeping
# ---------------------------------------------------------------------------

PEFT_CFG: Dict[str, Dict] = {
    "lora": {"lora_rank": 64, "lora_alpha": 128.0},
    "adapter": {"adapter_bottleneck_dim": 128, "adapter_dropout": 0.10},
    "bias": {},
    "prefix": {"prefix_length": 20},
    "base": {},
}

# ──────────────────────────────────────────────────────────────────────────────
#  Expected number of *trainable* parameters added by each PEFT head
# ──────────────────────────────────────────────────────────────────────────────


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
    total["prefix"] = plen * hidden_size  # shared embeddings
    total["bias"] += 2 * hidden_size  # word-embedding & lm-head biases
    return total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ppl(ce: float) -> float:
    return math.exp(ce) if ce < 50 else float("inf")


def _align(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if a.size(1) != b.size(1):
        m = min(a.size(1), b.size(1))
        return a[:, :m, :], b[:, :m]
    return a, b


def _ce_loss(out, labels) -> torch.Tensor:
    l, y = _align(out.logits[..., :-1, :], labels[..., 1:])
    return F.cross_entropy(l.reshape(-1, l.size(-1)), y.reshape(-1), ignore_index=-100)


def _make_concat_inputs(batch: Dict[str, torch.Tensor], pad: int, L: int) -> None:
    ctx = batch["input_ids"]
    tgt = batch["labels"].clone()
    tgt[tgt == -100] = pad

    ctx = ctx[:, : max(1, L - REPLY_BUDGET)]
    room = L - ctx.size(1) - 1
    tgt_trim = tgt[:, : max(room, 0)]

    sep = torch.full((ctx.size(0), 1), REPLY_SEP_ID, dtype=ctx.dtype, device=ctx.device)
    concat = torch.cat([ctx, sep, tgt_trim], dim=1)

    if concat.size(1) < L:
        pad_blk = torch.full((concat.size(0), L - concat.size(1)), pad, dtype=concat.dtype, device=concat.device)
        concat = torch.cat([concat, pad_blk], dim=1)

    attn = (concat != pad).long()
    lbl = torch.full_like(concat, -100)
    start = ctx.size(1) + 1
    lbl[:, start : start + tgt_trim.size(1)] = tgt_trim

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


def _build_scheduler(opt: torch.optim.Optimizer, steps: int, warm_frac: float = 0.05):
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


def _enable_gradient_checkpointing(m):
    fn = getattr(m, "gradient_checkpointing_enable", None)
    if callable(fn):
        fn()
    elif hasattr(m, "model") and callable(getattr(m.model, "gradient_checkpointing_enable", None)):
        m.model.gradient_checkpointing_enable()


def _wrap_model(variant: str, backbone: torch.nn.Module, cfg: dict) -> torch.nn.Module:
    """
    Inject the requested PEFT head, freeze or thaw parameters as needed,
    and return the wrapped backbone.
    """
    # ───────── LoRA ───────────────────────────────────────────────────
    if variant == "lora":
        for lyr in backbone.gpt_neox.layers:
            lyr.attention.query_key_value = LoRAQKV(
                lyr.attention.query_key_value,
                rank=cfg["lora_rank"],
                alpha=cfg["lora_alpha"],
            )

    # ───────── Adapter ────────────────────────────────────────────────
    elif variant == "adapter":
        backbone = ModelAdapter(
            backbone,
            bottleneck_dim=cfg["adapter_bottleneck_dim"],
            adapter_dropout=cfg["adapter_dropout"],
            use_layer_norm=True,
        )

        # mark ONLY the bottleneck weights & biases as trainable
        for name, param in backbone.named_parameters():
            needs_grad = (
                ".down_proj." in name.lower()
                or ".up_proj." in name.lower()
            )
            param.requires_grad_(needs_grad)

    # ───────── Bias-only ──────────────────────────────────────────────
    elif variant == "bias":
        backbone = BiasTuningModel(backbone)
        for name, param in backbone.named_parameters():
            param.requires_grad_(name.endswith(".bias") and param.ndim == 1)

    # ───────── Prefix-tuning ──────────────────────────────────────────
    elif variant == "prefix":
        backbone = ModelPrefixOnly(
            backbone,
            prefix_length=cfg["prefix_length"],
            embed_dim=backbone.config.hidden_size,
        )

    # ───────── Full fine-tune (“base”) ────────────────────────────────
    train_backbone = (variant == "base")
    if train_backbone:
        for p in backbone.parameters():
            p.requires_grad_(True)
        _enable_gradient_checkpointing(backbone)

    return backbone


# ---------------------------------------------------------------------------
# Audit helper
# ---------------------------------------------------------------------------


def _audit_trainable(model: torch.nn.Module, variant: str) -> None:
    hidden_size = model.config.hidden_size
    n_layers = getattr(model.config, "num_hidden_layers", len(model.gpt_neox.layers))
    expected = _compute_peft_param_budget(hidden_size, n_layers).get(variant)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info("[%s] trainable=%d • expected=%s", variant, trainable, expected or "all")

    if expected and trainable != expected:
        logging.warning("[%s] parameter-count mismatch (%d vs %d)", variant, trainable, expected)


# ---------------------------------------------------------------------------
# Evaluation utilities (mirrors hyper-net version)
# ---------------------------------------------------------------------------


def _eval_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    pad_id: int,
    seq_len: int,
    use_amp: bool,
) -> Tuple[float, float]:
    ce_tf = tok_tf = ce_cx = samp_cx = 0.0
    model.eval()

    gpu_idx = 0
    if device.type == "cuda":
        gpu_idx = device.index if device.index is not None else 0
    bf16_ok = device.type == "cuda" and torch.cuda.get_device_capability(gpu_idx)[0] >= 8
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            _make_concat_inputs(batch, pad_id, seq_len)

            with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                out = model(**{k: batch[k] for k in ("input_ids", "attention_mask", "labels")})

            nt = (batch["labels"] != -100).sum().item()
            ce_tf += _ce_loss(out, batch["labels"]).item() * nt
            tok_tf += nt
            ce_cx += _ctx_only_ce(out.logits, batch["input_ids"], batch["labels"]).item() * batch["input_ids"].size(0)
            samp_cx += batch["input_ids"].size(0)

    return ce_tf / max(1, tok_tf), ce_cx / max(1, samp_cx)


# ---------------------------------------------------------------------------
# Main training loop (identical logging style to hyper-net)
# ---------------------------------------------------------------------------


def _train(
    model: torch.nn.Module,
    dl_tr: DataLoader,
    dl_val: DataLoader,
    device: torch.device,
    steps: int,
    log_int: int,
    lr: float,
    wd: float,
    use_amp: bool,
    grad_accum: int,
    variant: str,
    pad_id: int,
    seq_len: int,
) -> Dict[str, float]:
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=wd)
    sched = _build_scheduler(opt, steps)
    scaler = GradScaler(enabled=use_amp)

    ce_sum = tok_sum = 0.0
    step = 0
    t0 = time.time()
    model.train()
    opt.zero_grad(set_to_none=True)

    gpu_idx = 0
    if device.type == "cuda":
        gpu_idx = device.index if device.index is not None else 0
    bf16_ok = device.type == "cuda" and torch.cuda.get_device_capability(gpu_idx)[0] >= 8
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16

    while step < steps:
        progressed = False
        for b_idx, batch in enumerate(dl_tr):
            batch = {k: v.to(device) for k, v in batch.items()}
            _make_concat_inputs(batch, pad_id, seq_len)

            nt = (batch["labels"] != -100).sum().item()
            if nt == 0:
                continue

            with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                out = model(**{k: batch[k] for k in ("input_ids", "attention_mask", "labels")})
                loss = _ce_loss(out, batch["labels"]) / grad_accum

            ce_sum += loss.item() * nt * grad_accum
            tok_sum += nt

            scaler.scale(loss).backward()
            if (b_idx + 1) % grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                progressed = True

                if step % log_int == 0 or step == steps:
                    mce = ce_sum / tok_sum
                    logging.info(
                        "[%-6s] step %5d/%d • CE %.4f • PPL %.1f • lr %.3e",
                        variant,
                        step,
                        steps,
                        mce,
                        _ppl(mce),
                        sched.get_last_lr()[0],
                    )
                if step >= steps:
                    break
        if not progressed:
            logging.error("No usable batches (all targets empty); aborting training loop.")
            break

    train_ce, train_ctx = _eval_split(model, dl_tr, device, pad_id, seq_len, use_amp)
    val_ce, val_ctx = _eval_split(model, dl_val, device, pad_id, seq_len, use_amp)
    return {
        "train_ce": train_ce,
        "train_ctx": train_ctx,
        "val_ce": val_ce,
        "val_ctx": val_ctx,
        "elapsed": time.time() - t0,
    }


def _load_optuna_lr(opt_dir: str, variant: str, default: float) -> float:
    p = Path(opt_dir) / f"opt_hypernet_{variant}.json"
    if p.is_file():
        try:
            return json.loads(p.read_text())[variant]["best_trial"]["lr"]
        except Exception:
            pass
    return default


def main() -> None:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_parquet", required=True)
    ap.add_argument("--val_parquet", required=True)
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--models_output_dir", required=True)
    ap.add_argument("--train_steps", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--weight_decay", type=float, default=1.0e-4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--log_interval", type=int, default=500)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--opt_results_dir", default=OPT_DIR_DEFAULT)
    ap.add_argument(
        "--checklist",
        default="/sciclone/home/thwalsh/hypernets/log_files/train_nonhypernet_checklist.txt",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    hf_utils.logging.set_verbosity_error()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    set_seed(142)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.fp32)

    df_tr = pd.read_parquet(args.train_parquet)
    df_val = pd.read_parquet(args.val_parquet)

    tok = AutoTokenizer.from_pretrained(args.base_ckpt, local_files_only=True)

    if "<|reply|>" not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": ["<|reply|>"]})

    global REPLY_SEP_ID
    REPLY_SEP_ID = tok.convert_tokens_to_ids("<|reply|>")

    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        tok.pad_token_id = tok.convert_tokens_to_ids("[PAD]")

    ds_tr = RedditConversationDataset(df_tr, tok, max_length=args.max_len)
    ds_val = RedditConversationDataset(df_val, tok, max_length=args.max_len)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda"))
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

    ck = Path(args.checklist)
    ck.parent.mkdir(parents=True, exist_ok=True)
    done = set(ck.read_text().splitlines()) if ck.exists() else set()

    out_root = Path(args.models_output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    log_int = args.log_interval if args.log_interval > 0 else max(1, args.train_steps // 10)
    pad_id = tok.pad_token_id

    for variant in ("lora", "adapter", "bias", "prefix", "base"):
        if variant in done:
            logging.info("[%s] already finished — skipping", variant)
            continue

        cfg = PEFT_CFG[variant]
        lr_val = args.lr or _load_optuna_lr(args.opt_results_dir, variant, 4.0e-4)
        logging.info("[%s] lr=%.3e", variant, lr_val)

        cfg_b = AutoConfig.from_pretrained(args.base_ckpt, local_files_only=True)
        cfg_b.use_cache = False
        backbone = AutoModelForCausalLM.from_pretrained(args.base_ckpt, config=cfg_b, local_files_only=True)
        backbone.resize_token_embeddings(len(tok))
        for p in backbone.parameters():
            p.requires_grad = False

        model = _wrap_model(variant, backbone, cfg).to(device)
        if torch.cuda.device_count() > 1:          # keep batch 48×4 feasible
            logging.info("[%s] using DataParallel across %d GPUs",
                        variant, torch.cuda.device_count())
            model = torch.nn.DataParallel(model)
            
        _audit_trainable(model, variant)

        stats = _train(
            model,
            dl_tr,
            dl_val,
            device,
            args.train_steps,
            log_int,
            lr_val,
            args.weight_decay,
            use_amp,
            args.grad_accum,
            variant,
            pad_id,
            args.max_len,
        )

        logging.info(
            "[%-6s] DONE • Train CE %.4f (CTX %.4f) • Val CE %.4f (PPL %.1f) • Val CTX %.4f (PPL %.1f) • Δt %.1fs",
            variant,
            stats["train_ce"],
            stats["train_ctx"],
            stats["val_ce"],
            _ppl(stats["val_ce"]),
            stats["val_ctx"],
            _ppl(stats["val_ctx"]),
            stats["elapsed"],
        )

        vdir = out_root / f"nonhypernet_{variant}_model"
        vdir.mkdir(parents=True, exist_ok=True)

        if variant == "base":
            model.save_pretrained(vdir)
        else:
            placeholders = OrderedDict((k, v.detach().cpu()) for k, v in model.state_dict().items() if v.requires_grad)
            torch.save(placeholders, vdir / "peft_placeholders.safetensors")
            (getattr(model, "config", None) or backbone.config).save_pretrained(vdir)

        with ck.open("a") as fh:
            fh.write(f"{variant}\n")

    logging.info("Training complete • wall-time %.1fs", time.time() - t0)


if __name__ == "__main__":
    _T0 = time.time()
    main()
    print(f"Script runtime: {time.time() - _T0:.1f}s")