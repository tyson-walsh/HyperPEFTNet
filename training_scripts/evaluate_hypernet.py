#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_hypernet.py
==========================

Offline evaluation for the four PEFT + hypernetwork variants (LoRA, Adapter,
Bias-only, Prefix) trained on the 10,000-author Reddit corpus.  
The script measures *content fitness*, *stylistic fidelity*, *lexical
diversity*, and *conditioning faithfulness* and can compute permutation-based
feature importance for the 30-dimensional global-static vector (also used when
evaluating the hierarchical hypernetwork).

What the script produces
------------------------
* **Per-variant JSON** – `results/eval_{flat|hier}_hypernet_{variant}.json`  
  ‣ a *summary* block with scalar metrics  
  ‣ the **top-10** generated replies (highest BERTScore, BLEU tie-break) with
    their individual metrics  
* **Combined JSON** – `results/eval_{flat|hier}_hypernet_results.json`
  containing only the summary blocks (ready for plotting).  
* **All generations** – `results/eval_generated.parquet` with one row per
  (conversation, variant) so that downstream scripts can run self-feedback or
  persona consistency checks.  
* **Checklist file** – `log_files/eval_hypernet_checklist.txt` updated
  as each variant finishes, preventing accidental re-runs.  
* **Central log** – `log_files/evaluate_hypernet.log`.

Metrics computed
----------------
Token likelihood
    • Cross-Entropy (CE)  
    • Perplexity (PPL)

Surface overlap
    • BLEU-4 (nltk + smoothing)  
    • METEOR (nltk)  
    • ROUGE-L (rouge-score)

Semantic similarity
    • **BERTScore-F1** (rescaled, `bert-score`)  
    • MiniLM-L6 *cosine similarity* (raw cosine of unit-norm embeddings)

Lexical diversity
    • Distinct-1 and Distinct-2 (type-to-token ratio of unigrams / bigrams)

Conditioning faithfulness
    • Walsh-KL:  KL divergence between the gold global-static vector *g* and
      ĝ predicted from the model’s own generation.

Feature attribution
    • (optional) permutation importance *ΔCE* for every column of **g**.

Mathematical context
--------------------
Let **x** be the conversation context, **g** the 30-D author vector,
and **ŷ** the generated reply.

Token likelihood        ℒ = CE(y, f(x, g; θ))  
Surface overlap         BLEU, METEOR, ROUGE-L  
Semantic similarity     BERTScore-F1(ŷ, y), Cosine(ŷ, y)  
Lexical diversity       Distinct-n  
Persona alignment       KL( ĝ(ŷ) ‖ g )

ΔCEⱼ (permutation importance) is measured by replacing column *j* of **g**
with a random permutation and re-computing CE.

Key implementation details
--------------------------
* **Checkpoint restore** – backbone and PEFT placeholders are frozen; the
  hypernetwork weights are loaded and attached exactly as in training
  (11 layers, 48 hidden, rank 32 bottleneck, per-variant clamp constant).  
* **Generation** – sampling (T = 0.8, top-p = 0.95, no-repeat-3) for up to 64
  tokens given the context prompt; reply and context are concatenated only for
  CE evaluation.  
* **NaN safety** – global and instance feature tables are `fillna(0.0)`-sanitized
  and any NaNs produced by quick feature extraction are converted with
  `np.nan_to_num`, preventing δθ → NaN cascades.  
* **GPU / batching** – automatically exploits all visible GPUs via
  `torch.nn.DataParallel`; batch size is configurable (default 4). MiniLM and
  BERTScore run with `show_progress_bar=False` to keep logs clean.  
* **Flat vs hierarchical** – `--flat_hypernet` (default) evaluates with **g**
  only; `--hierarchical_hypernet` additionally feeds the instance vector **i**.  
* **Demo mode** – `--demo_mode` limits each variant to three batches and
  suppresses all file output.  
* **Reproducibility** – deterministic seed, TF-32 disabled for math parity,
  and TensorFlow / NumPy logger verbosity suppressed.
"""

import sys, types, warnings, os, heapq, argparse, json, logging, math, time
from pathlib import Path
from statistics import mean
from typing import Dict, List, Any, Tuple
from itertools import count

from tqdm import tqdm

# -----------------------------------------------------------------------
# tqdm auto‑shim before any heavy imports so multiprocess pickles remain tiny
# -----------------------------------------------------------------------
_stub = types.ModuleType("tqdm.auto")
_stub.tqdm = tqdm
sys.modules["tqdm.auto"] = _stub
warnings.filterwarnings("ignore", message="IProgress not found")
try:
    from tqdm.utils import TqdmWarning
    warnings.filterwarnings("ignore", category=TqdmWarning)
except Exception:
    pass

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import re 
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow as pa
import pyarrow.parquet as pq
import multiprocessing as _mp
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import single_meteor_score as single_meteor
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    utils as hf_utils,
)

from sentence_transformers import SentenceTransformer
from bert_score import BERTScorer

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()
sys.path.extend([str(ROOT / "data_scripts"), str(ROOT / "PEFT_scripts")])

from hypernetwork_dataset import HypernetConversationDataset
from hierarchical_hypernetwork import (
    HierarchicalHypernetwork,
    PEFTHypernetModel,
    G_SIGNALS,
)
from adapter import ModelAdapter
from bias import BiasTuningModel
from lora import LoRAQKV
from prefix import ModelPrefixOnly

# -----------------------------------------------------------------------
# suppress noisy warnings from core libraries
# -----------------------------------------------------------------------
_libs_re = r"(?i).*?(torch|transformers|bert_score).*"

warnings.filterwarnings(
    "ignore",
    message=r"Deterministic behavior was enabled.*cublas",
    category=UserWarning,
    module=_libs_re,
)
warnings.filterwarnings(
    "ignore",
    message=r"stateless\.functional_call.*deprecated",
    category=FutureWarning,
    module=_libs_re,
)
warnings.filterwarnings(
    "ignore",
    message=r"Converting a tensor with requires_grad=True to a scalar",
    category=UserWarning,
    module=_libs_re,
)
warnings.filterwarnings("ignore", message="Empty reference sentence", module="bert_score")

# -----------------------------------------------------------------------
# constants, globals, and defaults
# -----------------------------------------------------------------------
_COUNTER = count()
_HIDDEN_DIM = 64
_N_LAYERS   = 11
_RANK       = 32

SBERT_PATH          = "/sciclone/home/thwalsh/hypernets/sentence_transformers/all-MiniLM-L6-v2"
BERT_MODEL_PATH     = "/sciclone/home/thwalsh/hypernets/bert_models/roberta-large"
DEFAULT_MODELS_DIR  = "/sciclone/home/thwalsh/hypernets/models"
DEFAULT_RESULTS_DIR = "/sciclone/home/thwalsh/hypernets/results"
DEFAULT_LOG_DIR     = "/sciclone/home/thwalsh/hypernets/log_files"

_default_gfeatures = [
    "gstat_personality_traits",
    "gcat_personality_type",
    "gcat_dominant_facet",
    "gstat_user_len_mean",
    "gstat_user_ttr",
    "gstat_user_post_rate",
    "gstat_user_subreddit_entropy",
]

_CLAMP_VALS: Dict[str, float] = {}
def _infer_arch(h_state: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    """
    Robustly infer (hidden_dim, rank, n_layers) from the hyper-network
    checkpoint.

    • hidden_dim  — out_features of the **first** Linear layer (`net.0.weight`)
    • rank        — the *largest* dimension that is                < hidden_dim
                    *and* paired with hidden_dim in any 2-D weight
                    (excludes the input dimensionality g)
    • n_layers    — number of residual blocks  (= max residual index + 1)

    This avoids confusing the input-projection matrix
    (shape = [hidden_dim, gdim]) with the hidden→rank projection
    (shape = [rank, hidden_dim]).
    """
    # ---------------------------------------------------------------
    # 1) hidden_dim  (rows of net.0.weight)
    # ---------------------------------------------------------------
    hidden_dim = next(
        (w.shape[0] for k, w in h_state.items()
         if k.startswith("net.0.weight") and w.dim() == 2),
        _HIDDEN_DIM,
    )

    # ---------------------------------------------------------------
    # 2) rank  (other dim paired with hidden_dim, choose the largest)
    # ---------------------------------------------------------------
    rank_candidates: list[int] = []
    for w in h_state.values():
        if w.dim() != 2:
            continue
        r, c = w.shape
        if   r == hidden_dim and c < hidden_dim:
            rank_candidates.append(c)
        elif c == hidden_dim and r < hidden_dim:
            rank_candidates.append(r)

    if rank_candidates:
        rank = max(rank_candidates)
    else:                       # very old / odd checkpoints
        rank = _RANK

    # ---------------------------------------------------------------
    # 3) n_layers  (count residual blocks)
    # ---------------------------------------------------------------
    resid_idx = [
        int(k.split(".")[1])
        for k in h_state
        if re.search(r"net\.\d+\.fc1\.weight$", k)
    ]
    n_layers = max(resid_idx) + 1 if resid_idx else _N_LAYERS

    return hidden_dim, rank, n_layers

_EVAL_BSZ = 4 * max(1, torch.cuda.device_count())
_TOPK        = 10
_GEN_NEW     = 64
_DEMO_MAX_BT = 3
_PI_REPS = 1 if torch.cuda.device_count() > 1 else 3

# -----------------------------------------------------------------------
# diagnostic prompt (deterministic seed = 142)
# -----------------------------------------------------------------------
PROMPT_WEST_POINT = "What do you think about the USMA West Point football team?"

REPLY_SEP_ID: int | None = None
PEFT_CFG: Dict[str, Dict] = {
    # cold-start LoRA uses rank-64 down/​up matrices and α = 128
    "lora":    {"lora_rank": 64, "lora_alpha": 128.0},
    "adapter": {"adapter_bottleneck_dim": 128, "adapter_dropout": 0.10},
    "bias":    {},
    "prefix":  {"prefix_length": 20},
}

_parquet_path   = None
_parquet_schema = None
_parquet_writer = None
_disable_io     = False
_bertscorer     = None

# -----------------------------------------------------------------------
# helper to load feature list persisted during training
# -----------------------------------------------------------------------
def load_feature_spec(ckpt_dir: Path) -> List[str]:
    f = ckpt_dir / "gfeat_columns.json"
    if f.is_file():
        try:
            return json.loads(f.read_text())
        except Exception:
            pass
    return _default_gfeatures.copy()

# -----------------------------------------------------------------------
# DataParallel utilities (mirrors training script)
# -----------------------------------------------------------------------
def _theta_bar_device_fix(mod: nn.Module, args, kwargs):
    """
    Ensure the cached `_theta_bar` dict (if present) is on the same device
    as incoming tensors when running inside `torch.nn.DataParallel`.
    """
    ref = None
    for arg in list(args) + list(kwargs.values()):
        if torch.is_tensor(arg):
            ref = arg
            break
        if isinstance(arg, (list, tuple)):
            ref = next((t for t in arg if torch.is_tensor(t)), None)
            if ref is not None:
                break
    if ref is None:
        return
    tgt = ref.device

    def _fix(node: nn.Module):
        tb = getattr(node, "_theta_bar", None)
        if isinstance(tb, dict):
            if any(t.device != tgt for t in tb.values()):
                node._theta_bar = {k: t.to(tgt, non_blocking=True).clone() for k, t in tb.items()}
        for ch in node.children():
            _fix(ch)

    _fix(mod)

# -----------------------------------------------------------------------
# dataset utilities for permutation importance
# -----------------------------------------------------------------------
def _dataset_with_permuted_global(
    ds: HypernetConversationDataset,
    col_idx: int,
    rng: np.random.Generator
) -> HypernetConversationDataset:
    for cand in ("_gdf", "gdf", "global_df", "global_features_df"):
        if hasattr(ds, cand):
            gdf_attr = cand
            break
    else:
        raise AttributeError("No global feature DataFrame attribute found on dataset")

    gdf_perm = getattr(ds, gdf_attr).copy(deep=True)
    col_vals = gdf_perm.iloc[:, col_idx].to_numpy()
    rng.shuffle(col_vals)
    gdf_perm.iloc[:, col_idx] = col_vals

    idf_perm = getattr(ds, "_idf", pd.DataFrame()).copy(deep=True) if hasattr(ds, "_idf") else pd.DataFrame()

    g_cols = getattr(ds, "_g_cols", _default_gfeatures)
    i_cols = getattr(ds, "_i_cols", []) if getattr(ds, "hierarchical", False) else []

    new_ds = HypernetConversationDataset(
        ds.raw_df,
        ds.tokenizer,
        gdf_perm,
        idf_perm,
        hierarchical=getattr(ds, "hierarchical", False),
    )
    new_ds.set_selected_features(g_cols, i_cols)
    return new_ds

# -----------------------------------------------------------------------
# parquet helper for generation dump
# -----------------------------------------------------------------------
def _append_to_parquet(rows: List[Dict[str, Any]], variant: str, results_dir: Path):
    if _disable_io or not rows:
        return
    global _parquet_path, _parquet_schema, _parquet_writer
    if _parquet_path is None:
        _parquet_path = (results_dir / "eval_generated.parquet").as_posix()
    for r in rows:
        r["variant"] = variant
    df_batch = pd.DataFrame(rows, columns=["gid", "target_user_id", "variant", "generated_text"])
    if _parquet_schema is None:
        table = pa.Table.from_pandas(df_batch, preserve_index=False)
        _parquet_schema = table.schema
        _parquet_writer = pq.ParquetWriter(_parquet_path, _parquet_schema, compression="zstd")
    else:
        table = pa.Table.from_pandas(df_batch, preserve_index=False, schema=_parquet_schema)
    _parquet_writer.write_table(table)

# -----------------------------------------------------------------------
# backbone + PEFT constructor
# -----------------------------------------------------------------------
def _build_peft_backbone(variant: str, ckpt: str, tok: AutoTokenizer) -> nn.Module:
    """
    Create the frozen GPT‑NeoX backbone and inject the correct PEFT placeholders.
    The shapes must match those saved during training, including the special
    `lora_warm` variant, and *SDPA/FlashAttention‑2 are disabled* to avoid the
    768 × 384 mismatch seen with the integration patch in Transformers ≥ 4.40.
    """
    cfg = AutoConfig.from_pretrained(ckpt, local_files_only=True)
    cfg.use_cache = False                           # keeps forward deterministic
    # ——— turn off faster attention kernels that break GPT‑NeoX + PEFT weights ———
    cfg.attn_implementation = "eager"               # forces eager (Python) path
    cfg.use_flash_attention_2 = False               # extra safety for v4.41+

    base = AutoModelForCausalLM.from_pretrained(ckpt, config=cfg, local_files_only=True)
    base.resize_token_embeddings(len(tok))
    for p in base.parameters():
        p.requires_grad = False

    v = "lora" if variant == "lora_warm" else variant

    if v == "lora":
        rank  = PEFT_CFG["lora"]["lora_rank"]
        alpha = PEFT_CFG["lora"]["lora_alpha"]
        for lyr in base.gpt_neox.layers:
            lora_qkv = LoRAQKV(lyr.attention.query_key_value, rank=rank, alpha=alpha)
            if hasattr(lora_qkv, "base_weight"):
                lora_qkv.base_weight.requires_grad_(False)
            lyr.attention.query_key_value = lora_qkv

    elif v == "adapter":
        base = ModelAdapter(base, **PEFT_CFG["adapter"], use_layer_norm=True)

    elif v == "bias":
        base = BiasTuningModel(base)

    elif v == "prefix":
        base = ModelPrefixOnly(
            base,
            prefix_length=PEFT_CFG["prefix"]["prefix_length"],
            embed_dim=base.config.hidden_size,
        )

    else:
        raise ValueError(f"Unknown PEFT variant: {variant}")

    return base


# -----------------------------------------------------------------------
# hypernetwork constructor ─ identical architecture as during training
# -----------------------------------------------------------------------
def _build_hypernet(
    backbone: nn.Module,
    gdim: int,
    idim: int,
    clamp_val: float,
    hierarchical: bool,
    *,
    hidden_dim: int | None = None,
    rank: int | None = None,
    n_layers: int | None = None,
    peft_cnt_override: int | None = None,
) -> HierarchicalHypernetwork:
    """
    Re-create the hyper-network with the exact dimensions stored in the
    checkpoint.  If the caller passes no overrides we fall back to the
    compile-time defaults (48-dim hidden, rank-32 bottleneck, 11 layers).
    """
    hidden_dim = hidden_dim or _HIDDEN_DIM
    rank       = rank       or _RANK
    n_layers   = n_layers   or _N_LAYERS

    peft_cnt = (
        peft_cnt_override
        if peft_cnt_override is not None
        else sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    )

    class _Residual(nn.Module):
        def __init__(self, dim: int, hidden: int):
            super().__init__()
            self.fc1  = nn.Linear(dim, hidden, bias=True)
            self.act1 = nn.SiLU()
            self.fc2  = nn.Linear(hidden, dim, bias=True)
            self.act2 = nn.SiLU()
            self.drop = nn.Dropout(0.10)
            self.ln   = nn.LayerNorm(dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:       # type: ignore[override]
            h = self.act1(self.fc1(x))
            h = self.act2(self.fc2(h))
            return self.ln(x + self.drop(h))

    in_dim = gdim + (idim if hierarchical else 0)
    blocks: List[nn.Module] = [
        nn.Linear(in_dim, hidden_dim, bias=True),
        nn.SiLU(),
    ]
    for _ in range(max(0, n_layers - 2)):                        # (n_layers-2) residuals
        blocks.append(_Residual(hidden_dim, hidden_dim * 2))
    blocks.extend(
        [
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, rank, bias=False),
            nn.Linear(rank, peft_cnt, bias=True),
        ]
    )

    hyper = HierarchicalHypernetwork(
        global_input_dim   = gdim,
        instance_input_dim = idim,
        hidden_dim         = hidden_dim,
        peft_param_count   = peft_cnt,
        use_instance       = hierarchical,
    )
    hyper.net = nn.Sequential(*blocks)
    return hyper


# -----------------------------------------------------------------------
# Pad‑collate identical to training/eval builds
# -----------------------------------------------------------------------
class _PadCollate:
    def __init__(self, gdim: int, idim: int):
        self.gdim = gdim
        self.idim = idim

    @staticmethod
    def _fix(vec: torch.Tensor, target: int) -> torch.Tensor:
        if vec.shape[0] < target:
            pad = torch.zeros(target - vec.shape[0], dtype=vec.dtype)
            return torch.cat([vec, pad], dim=0)
        if vec.shape[0] > target:
            return vec[:target]
        return vec

    def __call__(self, batch):
        for smp in batch:
            smp["global_features"] = self._fix(smp["global_features"], self.gdim)
            if self.idim:
                smp["instance_features"] = self._fix(smp["instance_features"], self.idim)
        return torch.utils.data._utils.collate.default_collate(batch)

# -----------------------------------------------------------------------
# concat helper for CE evaluation
# -----------------------------------------------------------------------
def _make_concat_inputs(
    batch: Dict[str, torch.Tensor],
    pad_id: int,
    seq_len: int,
    reply_max: int = 128,
) -> None:
    ctx_ids = batch["input_ids"]
    tgt_ids = batch["labels"].clone()
    tgt_ids[tgt_ids == -100] = pad_id

    _, C = ctx_ids.size()
    max_ctx = max(seq_len - reply_max - 1, 8)          # -1 for SEP
    if C > max_ctx:
        ctx_ids = ctx_ids[:, -max_ctx:]
        C = max_ctx

    sep = torch.full(
        (ctx_ids.size(0), 1), REPLY_SEP_ID,
        dtype=ctx_ids.dtype, device=ctx_ids.device,
    )
    concat = torch.cat([ctx_ids, sep, tgt_ids], dim=1)[:, :seq_len]
    attn   = (concat != pad_id).long()

    labels = concat.clone()
    labels[:, : C + 1] = -100                          # mask ctx + SEP
    labels[labels == pad_id] = -100

    batch["input_ids"]      = concat
    batch["attention_mask"] = attn
    batch["labels"]         = labels

# -----------------------------------------------------------------------
# lexical diversity helpers
# -----------------------------------------------------------------------
def _distinct_n(ids: List[List[int]], n: int) -> float:
    if not ids:
        return 0.0
    grams, total = set(), 0
    for seq in ids:
        total += max(0, len(seq) - n + 1)
        for i in range(max(0, len(seq) - n + 1)):
            grams.add(tuple(seq[i : i + n]))
    return len(grams) / max(1, total)

# -----------------------------------------------------------------------
# heap helper for top‑k
# -----------------------------------------------------------------------
def _select_top_k(heap_store: List[Tuple], k: int) -> List[Dict[str, Any]]:
    return [x[1] for x in heapq.nsmallest(k, heap_store, key=lambda z: z[0])]

# -----------------------------------------------------------------------
# loss extraction utilities
# -----------------------------------------------------------------------
def _extract_loss(out, labels: torch.Tensor) -> torch.Tensor:
    def _align(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if a.size(1) != b.size(1):
            m = min(a.size(1), b.size(1))
            return a[:, :m, :], b[:, :m]
        return a, b

    if getattr(out, "loss", None) is not None:
        return out.loss
    if isinstance(out, dict) and "loss" in out:
        return out["loss"]
    if getattr(out, "logits", None) is not None and torch.is_tensor(out.logits):
        l1, y1 = _align(out.logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous())
        return F.cross_entropy(l1.reshape(-1, l1.size(-1)), y1.reshape(-1), ignore_index=-100)
    if isinstance(out, (list, tuple)):
        logits = [o.logits for o in out if getattr(o, "logits", None) is not None]
        if logits:
            lg = torch.cat(logits, dim=0)
            l1, y1 = _align(lg[..., :-1, :].contiguous(), labels[..., 1:].contiguous())
            return F.cross_entropy(l1.reshape(-1, l1.size(-1)), y1.reshape(-1), ignore_index=-100)
        losses = [o.loss for o in out if getattr(o, "loss", None) is not None]
        if losses:
            return torch.stack([torch.as_tensor(x) for x in losses]).mean()
    raise ValueError("cannot extract loss")

# -----------------------------------------------------------------------
# sentence transformer for cosine similarity
# -----------------------------------------------------------------------
try:
    _cos_model = SentenceTransformer(SBERT_PATH, device=f"cuda:{torch.cuda.current_device()}")
except RuntimeError:
    logging.warning("MiniLM encoder OOM – falling back to CPU")
    _cos_model = SentenceTransformer(SBERT_PATH, device="cpu")

def _cos_sim(a: str, b: str) -> float:
    e = _cos_model.encode([a, b], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    return float((e[0] * e[1]).sum())

# -----------------------------------------------------------------------
# KL helper
# -----------------------------------------------------------------------
def _kl_safe(pred_vec: np.ndarray, ref_vec: np.ndarray) -> float:
    m = min(len(pred_vec), len(ref_vec))
    p = np.abs(pred_vec[:m].astype("float32")) + 1e-8
    q = np.abs(ref_vec[:m].astype("float32")) + 1e-8
    p /= p.sum()
    q /= q.sum()
    return float(F.kl_div(torch.tensor(p).log(), torch.tensor(q), reduction="sum"))

# -----------------------------------------------------------------------
# bind hypernet + placeholders and patch `.generate`
# -----------------------------------------------------------------------
def _attach_hypernet(
    backbone: nn.Module,
    hyper_state: dict[str, torch.Tensor],
    place_state: dict[str, torch.Tensor],
    clamp_val: float,
    gdim: int,
    idim: int,
    hierarchical: bool,
):
    """
    Build a PEFT-hypernetwork model whose output length matches the
    *trainable* placeholder parameters present in the backbone.

    Steps
    -----
    1) Count real placeholder parameters.
    2) Infer hidden_dim / rank / n_layers from checkpoint.
    3) Re-create the hyper-network with that output size.
    4) Load checkpoint (row-trim final projection if oversize).
    5) Wrap everything in PEFTHypernetModel, copy the placeholder
       tensors, patch `.generate()`, and return the ready model.
    """

    # 1) COUNT PLACEHOLDER PARAMS ---------------------------------------
    peft_cnt_place = sum(p.numel() for p in backbone.parameters()
                         if p.requires_grad)

    # 2) ARCH PARAMS FROM CKPT ------------------------------------------
    hidden_dim, rank, n_layers = _infer_arch(hyper_state)

    # 3) RE-BUILD HYPER-NET --------------------------------------------
    hyper = _build_hypernet(
        backbone,
        gdim,
        idim,
        clamp_val,
        hierarchical,
        hidden_dim=hidden_dim,
        rank=rank,
        n_layers=n_layers,
        peft_cnt_override=peft_cnt_place,
    )

    # 4) LOAD CKPT (row-trim if necessary) ------------------------------
    trimmed = {}
    for k, v in hyper_state.items():
        if k not in hyper.state_dict():
            continue
        tgt = hyper.state_dict()[k].shape
        if v.shape == tgt:
            trimmed[k] = v
        elif len(v.shape) == 2 and v.shape[1] == tgt[1] and v.shape[0] >= tgt[0]:
            trimmed[k] = v[:tgt[0]]
        elif len(v.shape) == 1 and v.shape[0] >= tgt[0]:
            trimmed[k] = v[:tgt[0]]
    hyper.load_state_dict(trimmed, strict=False)

    # 5) WRAP, COPY PLACEHOLDERS, PATCH .generate() ---------------------
    model = PEFTHypernetModel(backbone, hyper, clamp_range=clamp_val)
    model.backbone = backbone

    for name, tensor in place_state.items():
        p = model._placeholders[name]
        p.data.copy_(tensor)
        if not hasattr(p, "_key"):
            p._key = name  # preserve original mapping


    # ------------------------------------------------------------------ #
    # 5) patch `.generate()` so GenerationMixin works on this module     #
    # ------------------------------------------------------------------ #
    import types

    def _patched_generate(self, *args, **kwargs):
        """
        Make this PEFT‑hypernetwork module compatible with
        `transformers.GenerationMixin.generate()` and disable accelerated
        SDPA/Flash‑Attn paths that mis‑align with the frozen NeoX weights.
        """
        # ------------------------------------------------------------------
        # 1. strip helper arg used by our caller
        # ------------------------------------------------------------------
        kwargs.pop("variant", None)

        # ------------------------------------------------------------------
        # 2. graft GenerationMixin into the class hierarchy (once)
        # ------------------------------------------------------------------
        from transformers.generation.utils import GenerationMixin

        if not isinstance(self, GenerationMixin):
            self.__class__ = type(
                "PEFTHypernetGen",
                (self.__class__, GenerationMixin),
                {},
            )

        # ------------------------------------------------------------------
        # 3. guarantee every attribute that _validate_model_class() checks
        # ------------------------------------------------------------------
        from transformers import GenerationConfig
        import torch

        # a) configuration handles
        if not hasattr(self, "config"):
            self.config = getattr(self.backbone, "config", None)

        if not hasattr(self, "generation_config"):
            self.generation_config = (
                getattr(self.backbone, "generation_config", None)
                or GenerationConfig.from_model_config(self.config)
            )

        # b) main input names
        self.main_input_name = "input_ids"
        if not hasattr(self.__class__, "main_input_name"):
            self.__class__.main_input_name = "input_ids"
        if not hasattr(self, "_main_input_name"):
            self._main_input_name = "input_ids"

        # c) cache‑support flag (>= 4.40)
        if not hasattr(self, "_supports_cache_class"):
            self._supports_cache_class = True
        if not hasattr(self.__class__, "_supports_cache_class"):
            self.__class__._supports_cache_class = True

        # d) device property
        if not hasattr(self.__class__, "device"):
            def _prop_device(inst):
                try:
                    return next(inst.parameters()).device
                except StopIteration:                      # no parameters
                    return torch.device("cpu")
            self.__class__.device = property(_prop_device)

        # e) **can_generate** hook required by _validate_model_class()
        if not hasattr(self.__class__, "can_generate"):
            def _can_generate(inst):
                return True
            self.__class__.can_generate = _can_generate
        # also bind on the instance to be extra‑safe wrt DataParallel replicas
        self.can_generate = types.MethodType(lambda _self: True, self)

        # ------------------------------------------------------------------
        # 4. ensure extra conditioning tensors survive expand() calls
        # ------------------------------------------------------------------
        if not hasattr(self, "prepare_inputs_for_generation"):
            def _prep(inst, input_ids, attention_mask=None, **model_kwargs):
                payload = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
                for key in ("global_features", "instance_features"):
                    if key in model_kwargs and model_kwargs[key] is not None:
                        payload[key] = model_kwargs[key]
                return payload
            self.prepare_inputs_for_generation = types.MethodType(_prep, self)

        # ------------------------------------------------------------------
        # 5. final safety knob – disable key/value caching (prevents SDPA path)
        # ------------------------------------------------------------------
        kwargs.setdefault("use_cache", False)

        # ------------------------------------------------------------------
        # 6. hand off to the genuine HF generator
        # ------------------------------------------------------------------
        return GenerationMixin.generate(self, *args, **kwargs)
            
    # bind the patched method to *this* model instance (and therefore
    # to all DataParallel children that are shallow‑copied from it)
    model.generate = types.MethodType(_patched_generate, model)

    # clean exit from constructor
    model.eval()
    return model


# -----------------------------------------------------------------------
# multiprocessing / dataloader safety
# -----------------------------------------------------------------------
_PERSIST_OK = _mp.get_start_method(allow_none=True) == "spawn"

# -----------------------------------------------------------------------
# CE for conditioning only helper
# -----------------------------------------------------------------------
def _ctx_only_ce(
    out,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    if isinstance(out, (list, tuple)):
        out = out[0]
    if isinstance(out, dict):
        logits = out["logits"]
    else:
        logits = out.logits if hasattr(out, "logits") else out

    B = min(logits.size(0), labels.size(0), input_ids.size(0))
    logits, labels, input_ids = logits[:B], labels[:B], input_ids[:B]

    reply_mask = labels.ne(-100)
    if not reply_mask.any():
        return torch.tensor(0.0, device=logits.device)

    first_idx = reply_mask.float().argmax(dim=1)
    valid     = first_idx.gt(0)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    rows      = torch.arange(B, device=logits.device)[valid]
    last_ctx  = first_idx[valid] - 1
    sel_logits  = logits[rows, last_ctx, :]
    sel_targets = input_ids[rows, first_idx[valid]]

    return F.cross_entropy(sel_logits, sel_targets, ignore_index=-100)

# -----------------------------------------------------------------------
# top‑level eval CE
# -----------------------------------------------------------------------
def _eval_ce(
    model,
    ds,
    dev,
    pad_collate,
    pad_id: int,
    seq_len: int,
) -> tuple[float, float]:
    ce_tf_sum = tok_sum = 0.0
    ce_ctx_sum = samp_sum = 0.0

    loader = DataLoader(
        ds,
        batch_size=_EVAL_BSZ,
        shuffle=False,
        collate_fn=pad_collate,
        num_workers=2,
        pin_memory=True,
        persistent_workers=_PERSIST_OK,
    )
    for bt in loader:
        bt = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in bt.items()}
        _make_concat_inputs(bt, pad_id, seq_len)

        with torch.no_grad():
            inst = bt["instance_features"] if "instance_features" in bt else None
            out  = model(
                input_ids       = bt["input_ids"],
                attention_mask  = bt["attention_mask"],
                labels          = bt["labels"],
                global_features = bt["global_features"],
                instance_features = inst,
            )

        nt = int((bt["labels"] != -100).sum().item())

        # always collapse to a 0-D tensor before Python-casting
        batch_tf  = _extract_loss(out, bt["labels"]).view(-1).mean().item()
        batch_ctx = _ctx_only_ce(out, bt["input_ids"], bt["labels"]).view(-1).mean().item()
        
        ce_tf_sum  += batch_tf  * nt
        tok_sum    += nt
        ce_ctx_sum += batch_ctx * bt["input_ids"].size(0)
        samp_sum   += bt["input_ids"].size(0)

    ce_tf  = ce_tf_sum  / max(1, tok_sum)
    ce_ctx = ce_ctx_sum / max(1, samp_sum)
    return ce_tf, ce_ctx

# -----------------------------------------------------------------------
# group columns by common stem for permutation importance
# -----------------------------------------------------------------------
def _column_groups(ds) -> Dict[str, list[int]]:
    groups: Dict[str, list[int]] = {}
    for idx, col in enumerate(ds._g_cols):
        root = "_".join(col.split("_")[:3])
        groups.setdefault(root, []).append(idx)
    return groups

# -----------------------------------------------------------------------
# RAND‑g diagnostic
# -----------------------------------------------------------------------
def _rand_g_delta_ce(model, batch, dev):
    gold_g = batch["global_features"].clone()
    rand_g = gold_g[torch.randperm(gold_g.size(0))]
    with torch.no_grad():
        out_gold = model(**batch)
        ce_gold  = _extract_loss(out_gold, batch["labels"]).view(-1).mean().item()

        inst = batch["instance_features"] if "instance_features" in batch else None
        out_rand = model(
            input_ids       = batch["input_ids"],
            attention_mask  = batch["attention_mask"],
            labels          = batch["labels"],
            global_features = rand_g.to(dev),
            instance_features = inst,
        )
        ce_rand = _extract_loss(out_rand, batch["labels"]).view(-1).mean().item()
    return ce_rand - ce_gold

# -----------------------------------------------------------------------
# concise sanity log helper
# -----------------------------------------------------------------------
def _log_extra_sanity(variant : str,
                      ce_tf   : float,
                      ce_ctx  : float,
                      kl_med  : float,
                      kl_p90  : float,
                      clamp   : float,
                      d1      : float,
                      d2      : float,
                      *,
                      enable  : bool = False) -> None:
    if not enable:
        return

    ppl_tf  = math.exp(min(ce_tf,  50))
    ppl_ctx = math.exp(min(ce_ctx, 50))

    logging.info(
        "[%-6s] EVAL • CE %.4f • PPL %.1f • CE_CTX %.4f • PPL_CTX %.1f "
        "• clamp ±%.3f • Dist-1 %.4f • Dist-2 %.4f "
        "• KL_med %.4f • KL_p90 %.4f",
        variant, ce_tf, ppl_tf, ce_ctx, ppl_ctx,
        clamp,  d1, d2,
        kl_med, kl_p90,
    )

# -----------------------------------------------------------------------
# permutation importance main loop
# -----------------------------------------------------------------------
def _perm_importance(model,
                     ds,
                     dev,
                     pad_collate,
                     pad_id,
                     seq_len,
                     *,
                     frac: float = 0.10,
                     seed: int = 0) -> Dict[str, float]:
    import random, torch.utils.data as tud
    rng = random.Random(seed)
    subset_idx = rng.sample(range(len(ds)), int(len(ds) * frac))
    sub_ds = tud.Subset(ds, subset_idx)

    base_tf, _ = _eval_ce(model, sub_ds, dev, pad_collate, pad_id, seq_len)

    saved = model.clamp_range
    model.clamp_range = model.hypernet.clamp_range = min(saved * 10.0, 1e3)
    try:
        col_delta = np.zeros(len(ds._g_cols), dtype=np.float32)
        for j in range(len(ds._g_cols)):
            acc = 0.0
            for _ in range(_PI_REPS):
                perm_ds_full = _dataset_with_permuted_global(ds, j, np.random.default_rng())
                perm_sub = tud.Subset(perm_ds_full, subset_idx)
                ce_tf_perm, _ = _eval_ce(model, perm_sub, dev, pad_collate, pad_id, seq_len)
                acc += ce_tf_perm
            col_delta[j] = (acc / _PI_REPS) - base_tf
    finally:
        model.clamp_range = model.hypernet.clamp_range = saved

    return {root: float(col_delta[idxs].sum())
            for root, idxs in _column_groups(ds).items()}

# -----------------------------------------------------------------------
# helper for fast feature extraction used in KL‑SELF
# -----------------------------------------------------------------------
def _safe_feature_extract(texts: List[str], ds, gdim: int) -> np.ndarray:
    try:
        arr = ds.quick_feature_extract(texts)
    except Exception as e:
        logging.warning("quick_feature_extract failed (%s); falling back to zeros", e)
        arr = np.zeros((len(texts), gdim), dtype=np.float32)

    arr = np.nan_to_num(arr, copy=False)
    if arr.shape[1] < gdim:
        arr = np.pad(arr, ((0, 0), (0, gdim - arr.shape[1])))
    elif arr.shape[1] > gdim:
        arr = arr[:, :gdim]
    return arr

# -----------------------------------------------------------------------
# split prompt and reference
# -----------------------------------------------------------------------
def _extract_prompt(batch_input: torch.Tensor, batch_labels: torch.Tensor) -> List[torch.Tensor]:
    prompts = []
    for inp, lab in zip(batch_input, batch_labels):
        pos = (lab != -100).nonzero(as_tuple=True)
        split = int(pos[0][0].item()) if len(pos[0]) else inp.size(0)
        prompts.append(inp[:split])
    return prompts

# -----------------------------------------------------------------------
# safer METEOR (handles empty cases)
# -----------------------------------------------------------------------
def single_meteor_score_safe(ref_tokens: List[str], gen_tokens: List[str]) -> float:
    if not ref_tokens or not gen_tokens:
        return 0.0
    return single_meteor(ref_tokens, gen_tokens)

# -----------------------------------------------------------------------
# =========================  MAIN ENTRYPOINT  ===========================
# -----------------------------------------------------------------------
def main() -> None:
    global _bertscorer
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--test_parquet", required=True)
    p.add_argument("--global_features_parquet", required=True)
    p.add_argument("--instance_features_parquet")
    p.add_argument("--base_ckpt", required=True)
    p.add_argument("--models_dir", default=DEFAULT_MODELS_DIR)
    p.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--model_root")
    p.add_argument("--flat_hypernet", action="store_true")
    p.add_argument("--hierarchical_hypernet", action="store_true")
    p.add_argument("--feature_permutation", action="store_true")
    p.add_argument("--checklist")
    p.add_argument("--variants", default="lora,lora_warm,adapter,bias,prefix", help="Comma-separated list of PEFT variants to evaluate")
    p.add_argument("--bsz", type=int, default=_EVAL_BSZ)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--seed", type=int, default=142)
    p.add_argument("--demo_mode", action="store_true")
    p.add_argument("--sanity_checks", action="store_true", help="Print Δθ magnitude and sample g-vector for quick debugging")
    p.add_argument("--override_clamp", type=float, help="Temporarily override clamp_range for debugging")
    p.add_argument("--log_dir", default=DEFAULT_LOG_DIR)
    p.add_argument("-f", "--f", help=argparse.SUPPRESS)
    args, _ = p.parse_known_args()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_dir) / "evaluate_hypernet.log"
    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", handlers=[fh, ch])

    global _disable_io
    _disable_io = bool(args.demo_mode)
    if _disable_io:
        logging.info("[DEMO] I/O to JSON / Parquet disabled – running quick pass.")

    if args.model_root:
        args.models_dir = args.model_root
    if args.flat_hypernet:
        args.hierarchical_hypernet = False
    hierarchical = args.hierarchical_hypernet
    mode_tag = "hier" if hierarchical else "flat"

    done_variants = set()
    if args.checklist and Path(args.checklist).is_file() and not _disable_io:
        done_variants = {ln.split("_")[0] for ln in Path(args.checklist).read_text().splitlines()}
        
    set_seed(142)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    hf_utils.logging.set_verbosity_error()
    logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device.lower() != "cuda":
        device = torch.device(args.device)

    bsz_bs = 32 * max(1, torch.cuda.device_count())        # 32, 64, 96 …
    _bertscorer = BERTScorer(
        model_type=BERT_MODEL_PATH,
        num_layers=24,
        lang="en",
        rescale_with_baseline=False,
        batch_size=bsz_bs,
        device=device,
    )

    df_test = pd.read_parquet(args.test_parquet)
    gdf = pd.read_parquet(args.global_features_parquet).fillna(0.0)
    idf = pd.read_parquet(args.instance_features_parquet).fillna(0.0) if args.instance_features_parquet else pd.DataFrame()

    tok = AutoTokenizer.from_pretrained(args.base_ckpt, local_files_only=True)
    if not tok.is_fast:
        raise RuntimeError(
            "Fast tokenizer not available; install `tokenizers` ≥0.14 or "
            "convert the checkpoint so that tokenizer.json is present."
        )

    if "<|reply|>" not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": ["<|reply|>"]})

    global REPLY_SEP_ID
    REPLY_SEP_ID = tok.convert_tokens_to_ids("<|reply|>")

    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        tok.pad_token_id = tok.convert_tokens_to_ids("[PAD]")

    # Build dataset aligned with training spec
    ck_probe = Path(args.models_dir) / f"{mode_tag}_hypernet_lora"
    if not ck_probe.exists():
        ck_probe = Path(args.models_dir) / f"{mode_tag}_hypernet_lora_model"
    gcols = load_feature_spec(ck_probe)

    ds = HypernetConversationDataset(
        df_test, tok, gdf, idf, hierarchical=hierarchical
    )
    ds.set_selected_features(
        gcols,
        [c for c in idf.columns if c not in ("gid", "target_user_id")] if hierarchical else [],
    )

    setattr(ds, "raw_df", df_test)
    setattr(ds, "tokenizer", tok)
    setattr(ds, "_gdf", gdf)
    setattr(ds, "_idf", idf)

    gdim_act = int(ds[0]["global_features"].shape[0])
    idim_act = int(ds[0]["instance_features"].shape[0]) if hierarchical else 0
    pad_collate = _PadCollate(gdim_act, idim_act)

    logging.info("Eval rows=%d • gdim=%d • idim=%d", len(df_test), gdim_act, idim_act)

    author_vec: Dict[int, np.ndarray] = {}
    target_total = gdf["target_user_id"].nunique()

    for item in ds:
        uid = int(item["target_user_id"])
        if uid not in author_vec:
            author_vec[uid] = item["global_features"].cpu().numpy()
            if len(author_vec) == target_total:
                break

    logging.info(
        "Author-vector table built by scanning dataset (%d / %d users, %d-D each)",
        len(author_vec), target_total, gdim_act,
    )

    results_dir = Path(args.results_dir)
    if not _disable_io:
        results_dir.mkdir(parents=True, exist_ok=True)
    combined: Dict[str, Dict] = {}

    bleu_sm = SmoothingFunction().method1
    rouge_sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    try:
        variant_list = (
            args.variants.split(",")                              # via --variants
            if getattr(args, "variants", None)                    # fallback to default
            else ["lora", "lora_warm", "adapter", "bias", "prefix"]
        )
        for variant in variant_list:

            if variant in done_variants:
                logging.info("[%s] already evaluated – skip", variant)
                continue

            candidates = [
                f"{mode_tag}_hypernet_{variant}",
                f"{mode_tag}_hypernet_{variant}_model",
                f"demo_runs/{mode_tag}_hypernet_{variant}",
                f"demo_runs/{mode_tag}_hypernet_{variant}_model",
            ]
            ck = next((Path(args.models_dir) / p for p in candidates if (Path(args.models_dir) / p).exists()), None)
            if ck is None:
                logging.warning("[%s] checkpoint dir missing – skip", variant)
                continue

            if not ck.exists():
                logging.warning("[%s] checkpoint dir missing – skip", variant)
                continue

            backbone = _build_peft_backbone(variant, args.base_ckpt, tok).to(device)

            hyper_path = ck / "hypernetwork.safetensors"
            if not hyper_path.is_file():                       # <–– NEW GUARD
                logging.info("[%s] %s missing – skip", variant, hyper_path.name)
                continue

            hyper_state = torch.load(hyper_path, map_location="cpu")
            
            ph_files = [f for f in ("peft_placeholders.safetensors", "peft_placeholders.pt") if (ck / f).is_file()]
            if not ph_files:
                raise FileNotFoundError(f"PEFT placeholder file missing in {ck}")
            place_state = torch.load(ck / ph_files[0], map_location="cpu")

            cfg_file = ck / "hyper_cfg.json"
            if cfg_file.is_file():
                try:
                    with cfg_file.open() as fh:
                        _CLAMP_VALS[variant] = float(json.load(fh).get("clamp", 0.20))
                except Exception:
                    _CLAMP_VALS[variant] = 0.20
            else:
                _CLAMP_VALS[variant] = 0.20

            model_single = _attach_hypernet(
                backbone,
                hyper_state,
                place_state,
                _CLAMP_VALS[variant],
                gdim_act,
                idim_act,
                hierarchical,
            ).to(device)

            # ---------- multi‑GPU wrap (mirrors training) ----------
            if torch.cuda.device_count() > 1:
                logging.info("[%s] using DataParallel across %d GPUs", variant, torch.cuda.device_count())
                model_single = torch.nn.DataParallel(model_single)
                model_single.module.register_forward_pre_hook(_theta_bar_device_fix, with_kwargs=True)

            model = model_single  # rename for brevity

            if args.override_clamp is not None:
                tgt = model.module if hasattr(model, "module") else model
                tgt.clamp_range = tgt.hypernet.clamp_range = args.override_clamp

            ce_tf_sum = ce_ctx_sum = tok_sum = samp_sum = 0.0
            kl_vals   = []
            top_heap: List[Tuple] = []
            all_ids: List[List[int]] = []
            raw_rows: List[Dict[str, Any]] = []

            loader = DataLoader(
                ds,
                batch_size=args.bsz,
                shuffle=False,
                collate_fn=pad_collate,
                num_workers=2,
                pin_memory=True,
                persistent_workers=_PERSIST_OK,
            )

            eos_id = tok.eos_token_id or tok.convert_tokens_to_ids("</s>")
            LOG_EVERY = 100

            for batch_idx, bt in enumerate(loader):
                if batch_idx % LOG_EVERY == 0:
                    logging.info("[%s] batch %d / %d", variant, batch_idx, len(loader))
                    sys.stdout.flush(); sys.stderr.flush()

                if _disable_io and batch_idx >= _DEMO_MAX_BT:
                    break

                bt = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in bt.items()}
                _make_concat_inputs(bt, tok.pad_token_id, args.max_len)

                with torch.no_grad():
                    inst = bt["instance_features"] if "instance_features" in bt else None
                    out  = model(
                        input_ids       = bt["input_ids"],
                        attention_mask  = bt["attention_mask"],
                        labels          = bt["labels"],
                        global_features = bt["global_features"],
                        instance_features = inst,
                    )

                nt = int((bt["labels"] != -100).sum().item())
                batch_tf  = _extract_loss(out, bt["labels"]).view(-1).mean().item()
                batch_ctx = _ctx_only_ce(out, bt["input_ids"], bt["labels"]).view(-1).mean().item()

                ce_tf_sum  += batch_tf  * nt
                tok_sum    += nt
                ce_ctx_sum += batch_ctx * bt["input_ids"].size(0)
                samp_sum   += bt["input_ids"].size(0)

                prompts = _extract_prompt(bt["input_ids"], bt["labels"])
                prompt_lens = [len(p) for p in prompts]

                valid_idx = [i for i, L in enumerate(prompt_lens) if L >= 2]
                if not valid_idx:
                    continue

                max_len = max(prompt_lens[i] for i in valid_idx)
                attn_masks, padded_prompts, orig_indices = [], [], []
                for i in valid_idx:
                    p = prompts[i]
                    attn = torch.ones(len(p), dtype=torch.long, device=device)
                    if len(p) < max_len:
                        pad = torch.full((max_len - len(p),), tok.pad_token_id, dtype=torch.long, device=device)
                        attn_pad = torch.zeros_like(pad)
                        p = torch.cat([p, pad])
                        attn = torch.cat([attn, attn_pad])
                    padded_prompts.append(p)
                    attn_masks.append(attn)
                    orig_indices.append(i)

                prompt_batch = torch.stack(padded_prompts)
                attn_batch = torch.stack(attn_masks)

                gen_func = model.module.generate if hasattr(model, "module") else model.generate
                gen_ids = gen_func(
                    prompt_batch,
                    attention_mask=attn_batch,
                    global_features=bt["global_features"][orig_indices],
                    instance_features=bt["instance_features"][orig_indices] if hierarchical else None,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.1,
                    max_new_tokens=_GEN_NEW,
                    min_new_tokens=1,
                    eos_token_id=eos_id,
                    pad_token_id=tok.pad_token_id,
                    variant=variant,
                )

                gen_txt_list, ctx_txt_list, gen_only_ids_all = [], [], []
                for j, seq in enumerate(gen_ids):
                    i = orig_indices[j]
                    ctx_tokens = padded_prompts[j][: prompt_lens[i]]
                    ctx_txt_list.append(tok.decode(ctx_tokens, skip_special_tokens=True))

                    gen_only = seq[prompt_lens[i]:]
                    keep_mask = gen_only != tok.pad_token_id
                    if tok.eos_token_id is not None:
                        keep_mask &= gen_only != tok.eos_token_id
                    gen_only = gen_only[keep_mask]

                    if gen_only.numel() == 0:
                        fallback_id = tok.unk_token_id or tok.eos_token_id or tok.pad_token_id
                        gen_only = torch.tensor([fallback_id], device=seq.device)

                    gen_only_ids_all.append(gen_only.tolist())
                    gen_txt_list.append(tok.decode(gen_only, skip_special_tokens=True))

                refs_for_valid = [
                    tok.decode(
                        bt["labels"][i].masked_fill(bt["labels"][i] == -100, tok.pad_token_id),
                        skip_special_tokens=True,
                    )
                    for i in orig_indices
                ]

                _, _, bert_f1 = _bertscorer.score(gen_txt_list, refs_for_valid, verbose=False)

                g_hat = _safe_feature_extract(gen_txt_list, ds, gdim_act)
                for idx_local, gen in enumerate(gen_txt_list):
                    i = orig_indices[idx_local]
                    uid = int(bt["target_user_id"][i])
                    kl_val = _kl_safe(g_hat[idx_local], author_vec[uid])
                    kl_vals.append(kl_val)
                    
                    ref_split, gen_split = refs_for_valid[idx_local].split(), gen.split()
                    bleu = sentence_bleu([ref_split], gen_split, smoothing_function=bleu_sm) if ref_split and gen_split else 0.0
                    meteor = single_meteor_score_safe(ref_split, gen_split)
                    rougeL = rouge_sc.score(refs_for_valid[idx_local], gen)["rougeL"].fmeasure
                    cos = _cos_sim(refs_for_valid[idx_local], gen)
                    bs = float(bert_f1[idx_local])

                    rec = dict(
                        gid=int(bt["gid"][i]),
                        context=ctx_txt_list[idx_local],
                        reference=refs_for_valid[idx_local],
                        generated=gen,
                        metrics=dict(
                            variant=variant,
                            BLEU=bleu,
                            METEOR=meteor,
                            ROUGE_L=rougeL,
                            BERTScore=bs,
                            CosSim=cos,
                            Distinct1=0,
                            Distinct2=0,
                        ),
                    )

                    key = (-bs, -bleu, next(_COUNTER))
                    if len(top_heap) < _TOPK:
                        heapq.heappush(top_heap, (key, rec))
                    else:
                        heapq.heappushpop(top_heap, (key, rec))

                    all_ids.append(gen_only_ids_all[idx_local])

                    raw_rows.append({"gid": int(bt["gid"][i]), "target_user_id": uid, "generated_text": gen})

                _append_to_parquet(raw_rows, variant, results_dir)
                raw_rows.clear()

            pad_eos = {tok.pad_token_id, eos_id}
            clean_ids = [[tok_id for tok_id in seq if tok_id not in pad_eos] for seq in all_ids]
            d1 = _distinct_n(clean_ids, 1)
            d2 = _distinct_n(clean_ids, 2)
            top_records = _select_top_k(top_heap, _TOPK)
            for r in top_records:
                r["metrics"]["Distinct1"] = d1
                r["metrics"]["Distinct2"] = d2

            ce_tf  = ce_tf_sum  / max(1, tok_sum)
            ce_ctx = ce_ctx_sum / max(1, samp_sum)

            rand_g_delta = 0.0
            if args.sanity_checks:
                delta_list = []
                for probe_bt in loader:
                    if len(delta_list) >= 3:
                        break
                    probe_bt = {k: (v.to(device) if torch.is_tensor(v) else v)
                                for k, v in probe_bt.items()}
                    _make_concat_inputs(probe_bt, tok.pad_token_id, args.max_len)
                    delta_list.append(_rand_g_delta_ce(model, probe_bt, device))
                rand_g_delta = float(np.mean(delta_list)) if delta_list else 0.0
                print(f"[SANITY][{variant}] RAND-g ΔCE=+{rand_g_delta:.2f}")

            ref_model = model.module if hasattr(model, "module") else model
            total_p = sum(p.numel() for p in ref_model.parameters())
            train_p = sum(p.numel() for p in ref_model.parameters() if p.requires_grad)

            kl_mean    = float(np.mean(kl_vals))    if kl_vals else 0.0
            kl_median  = float(np.median(kl_vals))  if kl_vals else 0.0
            kl_p90     = float(np.percentile(kl_vals, 90)) if kl_vals else 0.0

            _log_extra_sanity(variant, ce_tf, ce_ctx, kl_median, kl_p90,
                              ref_model.clamp_range, d1, d2,
                              enable=args.sanity_checks)

            summary = dict(
                test_ce       = ce_tf,
                test_ppl      = float(math.exp(min(ce_tf, 50))),
                test_ce_ctx   = ce_ctx,
                test_ppl_ctx  = float(math.exp(min(ce_ctx, 50))),
                self_feedback_kl_mean = kl_mean,
                kl_median             = kl_median,
                kl_p90                = kl_p90,
                distinct1_mean = d1,
                distinct2_mean = d2,
                total_params     = total_p,
                trainable_params = train_p,
                bleu_mean     = mean(r["metrics"]["BLEU"]      for _, r in top_heap) if top_heap else 0.0,
                meteor_mean   = mean(r["metrics"]["METEOR"]    for _, r in top_heap) if top_heap else 0.0,
                rougeL_mean   = mean(r["metrics"]["ROUGE_L"]   for _, r in top_heap) if top_heap else 0.0,
                bertscore_mean= mean(r["metrics"]["BERTScore"] for _, r in top_heap) if top_heap else 0.0,
                cosim_mean    = mean(r["metrics"]["CosSim"]    for _, r in top_heap) if top_heap else 0.0,
                rand_g_delta_ce = rand_g_delta,
            )

            # ------------------------------------------------------------------
            # append deterministic prompt‑sample to the summary (seed = 142)
            # ------------------------------------------------------------------
            rng_uid = 142 % len(author_vec)
            tgt_uid = sorted(author_vec.keys())[rng_uid]
            g_tensor = torch.tensor(author_vec[tgt_uid], dtype=torch.float32, device=device).unsqueeze(0)

            inst_tensor = (
                torch.zeros((1, idim_act), dtype=torch.float32, device=device)
                if hierarchical and idim_act > 0 else None
            )

            prompt_ids = tok.encode(PROMPT_WEST_POINT, return_tensors="pt").to(device)
            gen_ids_wp = model_single.generate(
                prompt_ids,
                global_features=g_tensor,
                instance_features=inst_tensor,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                max_new_tokens=64,
                eos_token_id=eos_id,
                pad_token_id=tok.pad_token_id,
                variant=variant,
            )[0][prompt_ids.size(1):]

            wp_response = tok.decode(gen_ids_wp, skip_special_tokens=True)
            summary["west_point_question"] = {
                "target_user_id": int(tgt_uid),
                "prompt": PROMPT_WEST_POINT,
                "response": wp_response,
            }


            if args.feature_permutation and not _disable_io:
                try:
                    summary["feature_importance"] = _perm_importance(
                        model, ds, device, pad_collate, tok.pad_token_id, args.max_len
                    )
                except RuntimeError as e:
                    logging.warning("Feature permutation skipped: %s", e)

            if not _disable_io:
                v_path = results_dir / f"eval_{mode_tag}_hypernet_{variant}.json"
                json.dump({variant: {"summary": summary, "samples": top_records}}, v_path.open("w"), indent=2)
                logging.info("[%s] report → %s", variant, v_path)
                combined[variant] = {"summary": summary}

            if args.checklist and not _disable_io:
                Path(args.checklist).parent.mkdir(parents=True, exist_ok=True)
                with Path(args.checklist).open("a") as fh:
                    fh.write(f"{variant}\n")

            logging.info(
                "[%-6s] EVAL • CE %.4f • PPL %.1f • CE_CTX %.4f • PPL_CTX %.1f "
                "• clamp ±%.3f • Dist-1 %.4f • Dist-2 %.4f "
                "• KL_mean %.4f • KL_med %.4f • KL_p90 %.4f • RANDgΔCE %.2f",
                variant,
                summary["test_ce"],
                summary["test_ppl"],
                summary["test_ce_ctx"],
                summary["test_ppl_ctx"],
                ref_model.clamp_range,
                d1,
                d2,
                summary["self_feedback_kl_mean"],
                summary["kl_median"],
                summary["kl_p90"],
                summary["rand_g_delta_ce"],
            )

            logging.info(
                "[%-6s] EVAL • CE %.4f • PPL %.1f • CE_CTX %.4f • PPL_CTX %.1f "
                "• clamp ±%.3f • Dist-1 %.4f • Dist-2 %.4f "
                "• KL_mean %.4f • KL_med %.4f • KL_p90 %.4f • RANDgΔCE %.2f",
                variant,
                summary["test_ce"],
                summary["test_ppl"],
                summary["test_ce_ctx"],
                summary["test_ppl_ctx"],
                ref_model.clamp_range,
                d1,
                d2,
                summary["self_feedback_kl_mean"],
                summary["kl_median"],
                summary["kl_p90"],
                summary["rand_g_delta_ce"],
            )

    finally:
        if _parquet_writer is not None:
            _parquet_writer.close()
            logging.info("All generations written → %s", _parquet_path)

    if not _disable_io:        
        sub_tag   = "" if "demo_runs" not in str(ck) else "demo_"
        comb_path = results_dir / f"{sub_tag}eval_{mode_tag}_hypernet_results.json"
        
        json.dump(combined, comb_path.open("w"), indent=2)
        logging.info("Combined summary → %s", comb_path)
    else:
        logging.info("[DEMO] run complete – no files written.")

if __name__ == "__main__":
    t0 = time.time()
    main()
    logging.info("Total wall-time %.1fs", time.time() - t0)