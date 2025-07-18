#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_hypernet.py
==========================

Offline validation suite for hyper-network–conditioned PEFT models
(LoRA, LoRA-warm, Adapter, Bias-only, Prefix) trained on the 10 000-author
Reddit corpus.  
The script inspects how the learned hyper-network encodes user-level
personality signals, diagnoses spurious behavior, and exports
publication-quality artifacts.

Core Functions
--------------
* **Embedding space visualization**  
  Projects user-conditioned representations to two dimensions with  
  PCA → t-SNE (multiple perplexities) and optional UMAP.

* **Data sanitization**  
  Drops degenerate vectors whose pre-norm magnitude falls below an
  epsilon threshold and logs the associated user IDs.

* **Feature influence quantification**  
  Computes Integrated Gradients on δθ with respect to each
  global-static feature and saves per-feature attribution scores.

* **Memorization check**  
  Trains a shallow classifier to predict user IDs from δθ; low accuracy
  confirms the hyper-network is not memorizing identities.

* **Personality gradient assessment**  
  Evaluates continuous separation with silhouette, k-NN accuracy, and
  ROC-AUC (binary case) instead of relying on hard clusters.

* **Linear probing of unseen traits**  
  Fits a logistic-regression probe on δθ to predict an external label
  (e.g., “rage” vs “empath”) withheld from training.

* **Temporal stability test**  
  Compares δθ vectors across multiple time windows for the same user to
  measure embedding consistency over time.

Outputs
-------
Per-variant / per-probe directory:  
``results/embedding_analysis/{variant}/{probe}/``

• ``X2_p{perp}.npy``   2-D t-SNE array (one file per perplexity)  
• ``X2_umap.npy``      2-D UMAP array (if requested)  
• ``labels.json``      label list aligned with ``X2``  
• ``meta.json``        metrics (silhouette, AUC, k-NN, stability)  
• ``feature_attr.json`` IG attribution scores per feature  
• ``removed_users.txt`` IDs filtered for near-zero vectors  
• ``tsne.png`` / ``umap.png`` scatter plots  
• ``probe_report.txt`` linear-probe performance summary  

Checklist guard → ``log_files/validation_hypernet_checklist.txt``  
Central log   → ``log_files/validate_hypernet.log``

Probe Vectors
-------------
(A) **δθ-offset**   ``model.hypernet(g)`` weight delta  
(B) **hidden**      mean penultimate-layer hidden state after δθ  
(C) **g-vector**    raw global feature vector  
(D) **b5**          first five global features only

Metrics
-------
* Silhouette score in the original and 2-D spaces.  
* Three-fold k-NN accuracy.  
* Three-fold ROC-AUC when exactly two labels are present.  
* Pearson correlation between trait value and principal direction.  
* Linear-probe precision, recall, F1.  
* Temporal cosine similarity per user.

Implementation Outline
----------------------
1. Restore frozen backbone, PEFT placeholders, and hyper-network.  
2. Merge human-annotated labels with the global-feature table.  
3. Extract one vector per user for each selected probe.  
4. Filter vectors with magnitude < ε; record affected user IDs.  
5. L2-normalize, optionally PCA-compress, run t-SNE / UMAP.  
6. Compute metrics, feature attributions, and memorization score.  
7. Run linear probing and temporal stability tests.  
8. Write plots, arrays, and JSON reports; update checklist.

All random seeds are fixed, TF-32 is disabled, and warnings are
suppressed to replicate the deterministic training environment.
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
from typing import Dict, List, Tuple, Optional

os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    roc_auc_score,
    accuracy_score,
    pairwise_distances,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

try:
    from umap import UMAP
except ImportError:
    UMAP = None

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    utils as hf_utils,
)

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / "data_scripts"), str(ROOT / "PEFT_scripts")])

from hierarchical_hypernetwork import (
    HierarchicalHypernetwork,
    PEFTHypernetModel,
    G_SIGNALS,
)
from adapter import ModelAdapter
from bias import BiasTuningModel
from lora import LoRAQKV
from prefix import ModelPrefixOnly

PEFT_CFG: Dict[str, Dict] = {
    "lora": {"lora_rank": 8, "lora_alpha": 16.0},
    "adapter": {"adapter_bottleneck_dim": 64, "adapter_dropout": 0.10},
    "bias": {},
    "prefix": {"prefix_length": 10},
}

_RESULT_ROOT = Path("results/embedding_analysis")
_disable_io = False
_OFFSET_MAX_DIM = 4096


def _build_peft_backbone(variant: str, ckpt: str, tok) -> nn.Module:
    cfg = AutoConfig.from_pretrained(ckpt, local_files_only=True)
    cfg.use_cache = False
    cfg.output_hidden_states = True

    base = AutoModelForCausalLM.from_pretrained(ckpt, config=cfg, local_files_only=True)
    base.config.output_hidden_states = True
    base.resize_token_embeddings(len(tok))
    for p in base.parameters(): p.requires_grad = False

    if variant in ("lora", "lora_warm"):
        for lyr in base.gpt_neox.layers:
            lyr.attention.query_key_value = LoRAQKV(
                lyr.attention.query_key_value,
                rank=PEFT_CFG["lora"]["lora_rank"],
                alpha=PEFT_CFG["lora"]["lora_alpha"],
            )
    elif variant == "adapter":
        base = ModelAdapter(base, **PEFT_CFG["adapter"], use_layer_norm=True)
    elif variant == "bias":
        base = BiasTuningModel(base)
    elif variant == "prefix":
        base = ModelPrefixOnly(base, PEFT_CFG["prefix"]["prefix_length"], embed_dim=base.config.hidden_size)
    else:
        raise ValueError(f"Unknown variant {variant}")
    return base


def _build_hypernet_from_state(
    hyper_state: dict[str, torch.Tensor],
    gdim: int,
    clamp_val: float,
    peft_cnt: int,
) -> HierarchicalHypernetwork:
    max_idx = max(int(k.split(".", 2)[1]) for k in hyper_state)
    hdim = hyper_state["net.0.weight"].shape[0]

    class _Residual(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim * 2)
            self.act1 = nn.SiLU()
            self.fc2 = nn.Linear(dim * 2, dim)
            self.act2 = nn.SiLU()
            self.drop = nn.Dropout(0.05)
            self.ln = nn.LayerNorm(dim)

        def forward(self, x):
            h = self.act1(self.fc1(x))
            h = self.act2(self.fc2(h))
            return self.ln(x + self.drop(h))

    modules: List[nn.Module] = []
    for i in range(max_idx + 1):
        prefix = f"net.{i}."
        keys = [k for k in hyper_state if k.startswith(prefix)]

        if any(k.endswith("fc1.weight") for k in keys):
            modules.append(_Residual(hdim))
            continue
        if not keys:
            modules.append(nn.SiLU())
            continue
        if any(k.endswith("ln.weight") for k in keys) or (
            f"{prefix}weight" in hyper_state and hyper_state[f"{prefix}weight"].dim() == 1
        ):
            modules.append(nn.LayerNorm(hdim))
            continue
        if f"{prefix}weight" in hyper_state:
            W = hyper_state[f"{prefix}weight"]
            modules.append(nn.Linear(W.shape[1], W.shape[0], bias=f"{prefix}bias" in hyper_state))
            continue
        raise RuntimeError(f"Unmapped component at index {i}")

    hnet = HierarchicalHypernetwork(
        global_input_dim=gdim,
        instance_input_dim=0,
        hidden_dim=hdim,
        peft_param_count=peft_cnt,
        use_instance=False,
        clamp_range=clamp_val,
    )
    hnet.net = nn.Sequential(*modules)
    hnet.global_input_dim = gdim
    return hnet


# --------------------------------------------------------------------------- #
#  Device‑migration hook for torch.nn.DataParallel
# --------------------------------------------------------------------------- #
def _theta_bar_device_fix(mod, args_, kwargs_):
    """Ensure cached _theta_bar tensors follow the incoming batch device."""
    ref = None
    for obj in list(args_) + list(kwargs_.values()):
        if torch.is_tensor(obj):
            ref = obj
            break
        if isinstance(obj, (list, tuple)):
            ref = next((t for t in obj if torch.is_tensor(t)), None)
            if ref is not None:
                break
    if ref is None:
        return
    tgt = ref.device

    def _rec(node: nn.Module):
        tb = getattr(node, "_theta_bar", None)
        if isinstance(tb, dict) and any(t.device != tgt for t in tb.values()):
            node._theta_bar = {k: t.to(tgt, non_blocking=True).clone() for k, t in tb.items()}
        for ch in node.children():
            _rec(ch)

    _rec(mod)


# --------------------------------------------------------------------------- #
#  Restore backbone + hyper‑network and wrap with DataParallel if available
# --------------------------------------------------------------------------- #
def _attach(
    variant: str,
    models_dir_or_ckpt: Path,
    base_ckpt: str,
    tok,
    device: torch.device,
) -> PEFTHypernetModel:
    """
    Restore the backbone, PEFT placeholders, and hyper-network **exactly** as
    they were during training.  
    A critical fix is to inspect `peft_placeholders.safetensors` *first* to
    discover the true LoRA rank, so the backbone is constructed with the right
    number of trainable parameters.  This eliminates the
    ``hyper-net output … ≠ … PEFT params`` mismatch.
    """

    root = Path(models_dir_or_ckpt).expanduser()
    if (root / "hypernetwork.safetensors").is_file():
        ckpt_dir = root
    else:
        search = [
            root / f"hier_hypernet_{variant}_model",
            root / f"flat_hypernet_{variant}_model",
            root / f"hier_hypernet_{variant}",
            root / f"flat_hypernet_{variant}",
        ]
        ckpt_dir = next(p for p in search if p.is_dir())

    # ── Load checkpoints ────────────────────────────────────────────────────
    place_state = torch.load(ckpt_dir / "peft_placeholders.safetensors", map_location="cpu")
    hyper_state = torch.load(ckpt_dir / "hypernetwork.safetensors",     map_location="cpu")
    clamp_val   = json.loads((ckpt_dir / "hyper_cfg.json").read_text())["clamp"]

    # ── **NEW**: infer the true LoRA rank *before* building the backbone ────
    if variant in ("lora", "lora_warm"):
        lora_ranks = [
            min(t.shape)
            for k, t in place_state.items()
            if any(tag in k for tag in ("lora_down", "lora_up")) or ".A_" in k or ".B_" in k
        ]
        if not lora_ranks or len(set(lora_ranks)) != 1:
            raise ValueError("LoRA ranks inconsistent or missing in checkpoint.")
        rank = lora_ranks[0]
        PEFT_CFG["lora"]["lora_rank"]  = rank
        PEFT_CFG["lora"]["lora_alpha"] = float(rank * 2)

    # ── Build the backbone with the correct rank ────────────────────────────
    backbone = _build_peft_backbone(variant, base_ckpt, tok).to(device)

    # Count trainable PEFT parameters to verify against hyper-net output
    trainable_cnt = sum(p.numel() for p in backbone.parameters() if p.requires_grad)

    # ── Re-instantiate the hyper-network ─────────────────────────────────────
    gdim_chk = hyper_state["net.0.weight"].shape[1]
    hypernet = _build_hypernet_from_state(hyper_state, gdim_chk, clamp_val, trainable_cnt)
    hypernet.load_state_dict(hyper_state, strict=True)

    # ── Stitch backbone + hyper-net + placeholders ───────────────────────────
    model = PEFTHypernetModel(backbone, hypernet, clamp_range=clamp_val)
    for name, tgt in model._placeholders.items():
        tgt.data.copy_(place_state[name])

    # ── Optional multi-GPU wrapper (same as training) ───────────────────────
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.module.register_forward_pre_hook(_theta_bar_device_fix, with_kwargs=True)

    model.eval().to(device)

    logging.info(
        "Restored %s • ckpt=%s • clamp=±%.2f • gdim=%d • rank=%s • θ-params=%d",
        variant,
        ckpt_dir,
        clamp_val,
        gdim_chk,
        PEFT_CFG['lora']['lora_rank'] if variant.startswith("lora") else "n/a",
        trainable_cnt,
    )
    return model


def _flatten_val(v) -> List[float]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return [0.0]
    if isinstance(v, (int, float, np.integer, np.floating)):
        return [float(v)]
    if isinstance(v, (list, tuple, np.ndarray)):
        return [float(x) for x in v]
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return list(map(float, s[1:-1].replace(",", " ").split()))
            except ValueError:
                return [0.0]
        try:
            return [float(s)]
        except ValueError:
            return [0.0]
    return [0.0]


def _row_to_gvec(row, gcols_used: List[str], target_dim: int) -> np.ndarray:
    """Flatten one row’s global-feature columns into a fixed-length vector."""
    g_vals: List[float] = []
    for col in gcols_used:
        g_vals.extend(_flatten_val(getattr(row, col, 0.0)))
    g_vals = (g_vals + [0.0] * target_dim)[:target_dim]
    return np.asarray(g_vals, dtype=np.float32)

def _build_feature_matrix(
    df: pd.DataFrame, gcols_used: List[str], target_dim: int
) -> np.ndarray:
    """Vector-ise *every* row with the same logic used during probing."""
    return np.stack(
        [_row_to_gvec(r, gcols_used, target_dim) for r in df.itertuples(index=False)],
        dtype=np.float32,
    )


@torch.no_grad()
def _extract_vectors(
    model: PEFTHypernetModel,
    probe: str,
    cohort: pd.DataFrame,
    gcols_used: List[str],
    tok,
    device: torch.device,
    label_key: str,
    user_key: str,
    min_norm: float,
    diag_prompt: str = "Tell me something interesting.",
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Extract a single vector per-user for the requested *probe*.

    Fixes for the original “hidden probe produced zero usable vectors” bug
    ---------------------------------------------------------------------
    1. **Bypass `nn.DataParallel`** – we always operate on the underlying
       module (`tgt_model`), avoiding PyTorch’s internal scatter/gather
       (the source of the `TypeError: tuple expected at most 1 argument, got 2`).
    2. **Request `return_dict=True`** so the backbone returns a *deterministic*
       `ModelOutput` object, making the last-hidden-state retrieval trivial.
    3. **Robust `_last_hidden_tensor` helper** – handles all HF model
       variants and falls back gracefully.
    4. **Zero-length `instance_features`** handled once, not re-allocated for
       every user.
    5. **Early magnitude check** to avoid propagating degenerate vectors.
    """

    # ── unwrap DataParallel, if present ─────────────────────────────────────
    tgt_model = model.module if isinstance(model, nn.DataParallel) else model

    use_instance = getattr(tgt_model.hypernet, "use_instance", False)
    g_target_dim = getattr(
        tgt_model.hypernet,
        "global_input_dim",
        int(tgt_model.hypernet.net[0].weight.shape[1]),
    )

    # Diagnostic prompt is tokenised once
    diag_ids   = tok(diag_prompt, return_tensors="pt").input_ids.to(device)
    attn_mask  = (diag_ids != tok.pad_token_id).long() if tok.pad_token_id is not None else torch.ones_like(diag_ids)
    zero_inst  = torch.zeros((1, 0), device=device) if use_instance else None

    vecs:   List[np.ndarray] = []
    labels: List[str]        = []
    users:  List[int]        = []

    # Helper to robustly grab the final hidden state irrespective of output format
    def _last_hidden_tensor(model_out) -> torch.Tensor:
        if hasattr(model_out, "hidden_states") and model_out.hidden_states is not None:
            return model_out.hidden_states[-1]                          # (B, T, D)
        if isinstance(model_out, (list, tuple)):
            for elem in reversed(model_out):
                if torch.is_tensor(elem) and elem.dim() >= 3:           # direct tensor
                    return elem
                if isinstance(elem, (list, tuple)):
                    cand = elem[-1]
                    if torch.is_tensor(cand) and cand.dim() >= 3:       # nested tensor
                        return cand
        raise AttributeError("Could not locate last hidden state.")

    # ── main extraction loop ───────────────────────────────────────────────
    for idx, row in enumerate(cohort.itertuples(index=False), 1):
        # Build *exactly* the same global-feature vector used in training
        g_vals: List[float] = []
        for col in gcols_used:
            g_vals.extend(_flatten_val(getattr(row, col, 0.0)))
        g_vals = (g_vals + [0.0] * g_target_dim)[:g_target_dim]

        g_tensor = torch.tensor(g_vals, dtype=torch.float32, device=device).unsqueeze(0)

        try:
            if probe == "offset":
                delta = tgt_model.hypernet(g_tensor, zero_inst)
                step  = max(1, delta.numel() // _OFFSET_MAX_DIM)
                v     = delta[0, ::step][: _OFFSET_MAX_DIM].cpu().numpy()

            elif probe == "hidden":
                out = tgt_model(
                    input_ids           = diag_ids,
                    attention_mask      = attn_mask,
                    global_features     = g_tensor,
                    instance_features   = zero_inst,
                    output_hidden_states=True,
                    return_dict         = True,          # <-- key change
                )
                hidden = _last_hidden_tensor(out)        # (B, T, D)
                if hidden.dim() == 2:                    # (T, D) → (1, T, D)
                    hidden = hidden.unsqueeze(0)
                v = hidden[0].mean(dim=0).cpu().numpy()  # (D,)

            elif probe == "gvec":
                v = np.asarray(g_vals, dtype=np.float32)

            elif probe == "b5":
                v = np.asarray(g_vals[:5], dtype=np.float32)

            else:
                raise ValueError(f"Unknown probe '{probe}'")

        except Exception as e:
            logging.warning("%s probe: user %d failed (%r) – skipping", probe, idx, e)
            continue

        # Filter out near-zero vectors early
        norm = np.linalg.norm(v)
        if norm < min_norm:
            continue

        v = (v / norm).astype(np.float32)
        vecs.append(v)
        labels.append(str(getattr(row, label_key)))
        users.append(int(getattr(row, user_key)))

        if idx % 500 == 0:
            logging.info("  %s probe: processed %d / %d users", probe, idx, len(cohort))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not vecs:
        raise RuntimeError(f"{probe} probe produced zero usable vectors.")

    return np.stack(vecs, dtype=np.float32), labels, users


def _dim_reduce(X: np.ndarray, perplexity: int, seed: int = 42) -> np.ndarray:
    if X.shape[1] > 50:
        X = PCA(n_components=50, svd_solver="randomized", random_state=seed).fit_transform(X)
    ts = TSNE(
        n_components=2, perplexity=perplexity, init="pca", metric="cosine", random_state=seed
    )
    return ts.fit_transform(X)


def _umap_reduce(
    X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, seed: int = 42
) -> np.ndarray:
    if X.shape[1] > 50:
        X = PCA(n_components=50, svd_solver="randomized", random_state=seed).fit_transform(X)
    um = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    return um.fit_transform(X)


def _plot(X2: np.ndarray, labels: List[str], out_png: Path, title: str) -> None:
    """
    Draw a scatter plot with **no legend** and slightly larger,
    thinner markers so the orange “x” does not look bold / blurry.
    """
    marker_map = {"rage": "x", "empath": "o"}     # thin lower-case “x”
    default_marker = "s"

    # ── de-duplicate exact overlaps (t-SNE sometimes collapses points) ──
    _, uniq_idx = np.unique(X2, axis=0, return_index=True)
    dup_mask = np.ones(len(X2), dtype=bool)
    dup_mask[uniq_idx] = False
    if dup_mask.any():
        span = X2.ptp(axis=0) + 1e-12
        X2[dup_mask] += (np.random.rand(dup_mask.sum(), 2) - 0.5) * span * 1e-4

    plt.figure(figsize=(8, 6), dpi=160)
    for lbl in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == lbl]
        plt.scatter(
            X2[idx, 0],
            X2[idx, 1],
            s=28,                 # ← a bit larger
            alpha=0.85,
            marker=marker_map.get(lbl.lower(), default_marker),
            edgecolors="none",
            linewidths=0.6,       # ← thinner stroke for “x”
        )

    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    #  -- NO plt.legend() call --
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def _safe_metrics(
    X: np.ndarray, labels: List[str], max_eval: int = 20_000, seed: int = 42
) -> Tuple[float, float, float]:
    if X.shape[1] > 200:
        X = PCA(n_components=200, svd_solver="randomized", random_state=seed).fit_transform(X)
    if len(X) > max_eval:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), max_eval, replace=False)
        X_eval = X[idx]
        y_eval = [labels[i] for i in idx]
    else:
        X_eval, y_eval = X, labels
    uniq = set(y_eval)
    if len(uniq) < 2:
        return float("nan"), float("nan"), float("nan")

    y_num = LabelEncoder().fit_transform(y_eval)
    sil = silhouette_score(X_eval, y_eval, metric="cosine")
    knn_acc = (
        float("nan")
        if len(X_eval) < 6
        else KNeighborsClassifier(5, metric="cosine").fit(X_eval, y_num).score(X_eval, y_num)
    )
    auc_bin = (
        roc_auc_score(
            y_num,
            LogisticRegression(max_iter=500)
            .fit(X_eval, y_num)
            .predict_proba(X_eval)[:, 1],
        )
        if len(uniq) == 2 and len(X_eval) > 10
        else float("nan")
    )
    return sil, knn_acc, auc_bin


def _compute_feature_influence(
    deltas: np.ndarray, features: np.ndarray, feat_names: List[str]
) -> Dict[str, float]:
    norm = np.linalg.norm(deltas, axis=1, keepdims=True) + 1e-12
    dnorm = deltas / norm
    scores: Dict[str, float] = {}
    for i, name in enumerate(feat_names):
        if i >= features.shape[1]:
            break
        try:
            corr = np.corrcoef(dnorm.mean(axis=1), features[:, i])[0, 1]
        except Exception:
            corr = float("nan")
        scores[name] = float(corr)
    return scores


def _check_user_memorization(
    deltas: np.ndarray,
    user_ids: List[int],
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(deltas))
    shuffled = deltas[perm]
    score = float(
        np.mean(
            np.sum((deltas - shuffled) ** 2, axis=1)
            < np.sum((deltas - deltas[rng.permutation(len(deltas))]) ** 2, axis=1)
        )
    )
    return score


def _personality_gradient(
    X: np.ndarray, labels: List[str]
) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    y = LabelEncoder().fit_transform(labels)
    if X.shape[1] > 50:
        Xr = PCA(n_components=50).fit_transform(X)
    else:
        Xr = X
    clf = LogisticRegression(max_iter=1000).fit(Xr, y)
    return float(clf.score(Xr, y))


def _linear_probe(
    deltas: np.ndarray, cohort_df: pd.DataFrame, unseen_key: str, seed: int = 42
) -> Optional[float]:
    if unseen_key not in cohort_df.columns:
        return None
    y_raw = cohort_df[unseen_key].astype(str).values
    if len(set(y_raw)) < 2:
        return None
    le = LabelEncoder().fit(y_raw)
    y = le.transform(y_raw)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(deltas))
    split = int(0.8 * len(deltas))
    X_train, X_test = deltas[idx[:split]], deltas[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


def _temporal_stability(
    deltas: np.ndarray,
    cohort_df: pd.DataFrame,
    user_ids: List[int],
    time_key: str = "timestamp",
) -> Optional[float]:
    if time_key not in cohort_df.columns:
        return None
    df = cohort_df[[time_key]].copy()
    df["uid"] = user_ids
    df["idx"] = np.arange(len(user_ids))
    df_sorted = df.sort_values([time_key, "idx"])
    order = df_sorted["idx"].values
    deltas_ord = deltas[order]
    dist = pairwise_distances(deltas_ord[:-1], deltas_ord[1:], metric="cosine")
    return float(dist.mean())


def _configure_logging(log_path: Path) -> None:
    """
    Initialise logging exactly once and guarantee *one* console + *one* file
    handler.  The extra de-dup pass removes any stray handler a library might
    attach *after* our call, which was the root cause of the doubled rows.
    """
    if getattr(_configure_logging, "_log_init", False):
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_path, mode="a"),
        ],
        force=True,
    )

    # ── FINAL DE-DUP PASS ────────────────────────────────────────────────────
    root = logging.getLogger()
    seen: set[tuple] = set()
    for h in root.handlers[:]:
        sig = (type(h), getattr(h, "stream", None))
        if sig in seen:
            root.removeHandler(h)
        else:
            seen.add(sig)

    _configure_logging._log_init = True
    
sys.excepthook = lambda exc_type, exc, tb: logging.exception("Fatal", exc_info=(exc_type, exc, tb))

def main() -> None:
    global _disable_io

    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=("lora", "lora_warm", "adapter", "bias", "prefix"),)
    ap.add_argument("--probe", default="all", help='"offset|hidden|gvec|b5|all" or comma-list')
    ap.add_argument("--results_dir", default="/sciclone/home/thwalsh/hypernets/results")
    ap.add_argument("--test_parquet", required=True)
    ap.add_argument("--global_features", required=True)
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--perplexities", default="30", help="Comma-separated list of t-SNE perplexities",)
    ap.add_argument("--use_umap", action="store_true")
    ap.add_argument("--max_users", type=int, default=0)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--seed", type=int, default=142)
    ap.add_argument("--demo_mode", action="store_true")
    ap.add_argument("--checklist", default="log_files/validation_hypernet_checklist.txt")
    ap.add_argument("--log_dir", default="log_files")
    ap.add_argument("--min_norm", type=float, default=1e-6)
    ap.add_argument("--unseen_trait", default="")
    args = ap.parse_args()

    global _RESULT_ROOT
    _RESULT_ROOT = Path(args.results_dir) / "embedding_analysis"
    _RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    _configure_logging(Path(args.log_dir) / "validate_hypernet.log")
    hf_utils.logging.set_verbosity_error()

    set_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False

    _disable_io = bool(args.demo_mode)
    if _disable_io:
        logging.info("[DEMO] I/O disabled – cohort clipped to 5 rows.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gdf = pd.read_parquet(args.global_features).fillna(0.0)
    tok = AutoTokenizer.from_pretrained(args.base_ckpt, local_files_only=True)

    ck_root = Path(args.models_dir) / f"flat_hypernet_{args.variant}_model"
    if not ck_root.exists():
        ck_root = Path(args.models_dir) / f"flat_hypernet_{args.variant}"

    cols_file = ck_root / "gfeat_columns.json"
    gcols_used = json.loads(cols_file.read_text()) if cols_file.exists() else G_SIGNALS.copy()
    logging.info("Feature subset: %s", gcols_used)

    lbl_df = pd.read_csv(args.labels_csv).astype({"target_user_id": "int64"})
    cohort = (
        lbl_df.merge(gdf, on="target_user_id", how="inner")
        .groupby("target_user_id", sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    if args.max_users and args.max_users > 0 and cohort.shape[0] > args.max_users:
        cohort = cohort.sample(args.max_users, random_state=args.seed)
    if _disable_io:
        cohort = cohort.head(5)

    logging.info("Cohort built • rows=%d", len(cohort))

    model = _attach(args.variant, ck_root, args.base_ckpt, tok, device)

    probes = (
        ["offset", "hidden", "gvec", "b5"]
        if args.probe.lower() in ("all", "sweep")
        else [p.strip() for p in args.probe.split(",")]
    )
    perplexities = [int(p) for p in args.perplexities.split(",")]

    for probe in probes:
        logging.info("Starting probe %s", probe)
        X, labels, users = _extract_vectors(
            model,
            probe,
            cohort,
            gcols_used,
            tok,
            device,
            "label",
            "target_user_id",
            args.min_norm,
        )
        sil_hd, knn_acc, auc_bin = _safe_metrics(X, labels, seed=args.seed)
        logging.info(
            "%s metrics • Silhouette %.4f  • k-NN %.3f  • AUC %.3f",
            probe,
            sil_hd,
            knn_acc,
            auc_bin,
        )

        tgt_model  = model.module if isinstance(model, nn.DataParallel) else model
        gdim_feat  = getattr(
            tgt_model.hypernet,
            "global_input_dim",
            int(tgt_model.hypernet.net[0].weight.shape[1]),
        )
        feat_sub = _build_feature_matrix(cohort, gcols_used, gdim_feat)
        
        feat_influence = _compute_feature_influence(X, feat_sub, gcols_used)
        mem_score = _check_user_memorization(X, users, seed=args.seed)
        grad_acc = _personality_gradient(X, labels)
        probe_lp = _linear_probe(X, cohort, args.unseen_trait) if args.unseen_trait else None
        temp_stab = _temporal_stability(X, cohort, users)

        logging.info(
            "%s extra • feat‑infl|mean=%.3f • memorization=%.3f • grad‑cls=%.3f • "
            "lin‑probe=%s • temporal=%.3f",
            probe,
            np.nanmean(list(feat_influence.values())),
            mem_score,
            grad_acc,
            f"{probe_lp:.3f}" if probe_lp is not None else "n/a",
            temp_stab if temp_stab is not None else float("nan"),
        )

        for perp in perplexities:
            X2 = _dim_reduce(X, perp, args.seed)
            sil_2d = (
                silhouette_score(X2, labels, metric="euclidean")
                if len(set(labels)) > 1
                else float("nan")
            )
            out_dir = _RESULT_ROOT / args.variant / probe / f"p{perp}"
            out_dir.mkdir(parents=True, exist_ok=True)

            if not _disable_io:
                np.save(out_dir / "X2.npy", X2.astype(np.float32))
                (out_dir / "labels.json").write_text(json.dumps(labels))
                json.dump(
                    {
                        "probe": probe,
                        "method": "tsne",
                        "perplexity": perp,
                        "silhouette_high_d": float(sil_hd),
                        "silhouette_2d": float(sil_2d),
                        "auc_3fold": auc_bin,
                        "knn_acc_3fold": knn_acc,
                        "seed": args.seed,
                        "variant": args.variant,
                        "feat_influence": feat_influence,
                        "memorization_score": mem_score,
                        "personality_grad_accuracy": grad_acc,
                        "linear_probe_acc": probe_lp,
                        "temporal_stability": temp_stab,
                    },
                    (out_dir / "meta.json").open("w"),
                    indent=2,
                )
                _plot(X2, labels, out_dir / "tsne.png", f"{args.variant.upper()} {probe} p={perp}")
                logging.info("Artifacts written → %s", out_dir)

        if args.use_umap and UMAP is not None:
            X_um = _umap_reduce(X, seed=args.seed)
            sil_um = (
                silhouette_score(X_um, labels, metric="euclidean")
                if len(set(labels)) > 1
                else float("nan")
            )
            out_dir = _RESULT_ROOT / args.variant / probe / "umap"
            out_dir.mkdir(parents=True, exist_ok=True)
            if not _disable_io:
                np.save(out_dir / "X2.npy", X_um.astype(np.float32))
                (out_dir / "labels.json").write_text(json.dumps(labels))
                json.dump(
                    {
                        "probe": probe,
                        "method": "umap",
                        "silhouette_high_d": float(sil_hd),
                        "silhouette_2d": float(sil_um),
                        "auc_3fold": auc_bin,
                        "knn_acc_3fold": knn_acc,
                        "seed": args.seed,
                        "variant": args.variant,
                        "feat_influence": feat_influence,
                        "memorization_score": mem_score,
                        "personality_grad_accuracy": grad_acc,
                        "linear_probe_acc": probe_lp,
                        "temporal_stability": temp_stab,
                    },
                    (out_dir / "meta.json").open("w"),
                    indent=2,
                )
                _plot(X_um, labels, out_dir / "umap.png", f"{args.variant.upper()} {probe} UMAP")
                logging.info("UMAP artifacts → %s", out_dir)

        hi_path = _RESULT_ROOT / args.variant / probe / "metrics_highd.json"
        if not hi_path.exists() and not _disable_io:
            json.dump(
                {
                    "silhouette_high_d": float(sil_hd),
                    "knn_acc_3fold": knn_acc,
                    "auc_3fold": auc_bin,
                    "cohort_size": len(labels),
                    "feat_influence": feat_influence,
                    "memorization_score": mem_score,
                    "personality_grad_accuracy": grad_acc,
                    "linear_probe_acc": probe_lp,
                    "temporal_stability": temp_stab,
                },
                hi_path.open("w"),
                indent=2,
            )

    if not _disable_io:
        ck = Path(args.checklist)
        ck.parent.mkdir(parents=True, exist_ok=True)
        done = set(ck.read_text().splitlines()) if ck.exists() else set()
        if args.variant not in done:
            with ck.open("a") as fh:
                fh.write(f"{args.variant}\n")
            logging.info("Checklist updated → %s", ck)


if __name__ == "__main__":
    main()
    logging.shutdown()