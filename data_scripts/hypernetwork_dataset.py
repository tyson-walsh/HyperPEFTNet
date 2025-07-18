#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hypernetwork_dataset_10000.py
=============================

A wrapper that joins tokenized Reddit context–response pairs with flattened
feature vectors, yielding tuples ready for training flat or hierarchical
hyper-networks.

Feature taxonomy
----------------
Signals are grouped by *scope* (GLOBAL vs INSTANCE) and *temporal nature*
(STATIC vs DYNAMIC).  Only a subset of the GLOBAL-STATIC block is loaded here
by default, but you may select different columns at runtime.

GLOBAL-STATIC  <one vector per author>
    1. gstat_personality_traits         5-float centered OCEAN probabilities
    2. gcat_personality_type            16-hot MBTI code
    3. gcat_dominant_facet              5-hot dominant O/C/E/A/N indicator
    4. gstat_user_len_mean              mean post length
    5. gstat_user_ttr                   type-token ratio
    6. gstat_user_post_rate             mean posts per day
    7. gstat_user_subreddit_entropy     subreddit entropy in bits

INSTANCE-STATIC  <one row per reply>
    istat_reply_sentiment               signed reply polarity
    istat_reply_act                     dialogue-act label (0/1/2/3)

INSTANCE-DYNAMIC <one row per reply>
    idyn_sentiment_trend                reply minus thread sentiment
    idyn_user_post_rate                 short-term author cadence
    idyn_semantic_shift                 cosine distance to context

The dataset also returns an integer ``mbti_label`` (0–15, −1 if missing) so an
auxiliary MBTI loss can be enabled during training.  Only one MBTI code and
one dominant OCEAN facet exist per target user; the underlying *\*_full_10000_v2.parquet*
splits guarantee this property.

quick_feature_extract
---------------------
Builds the 30-D global vector on the fly from raw text only.
It uses fast sentiment and personality pipelines and leaves the final four
scalars at zero.

Column layout (N, 30):

    0-4   gstat_personality_traits
    5-20  gcat_personality_type       (one-hot)
    21-25 gcat_dominant_facet         (one-hot)
    26    gstat_user_len_mean         (stub)
    27    gstat_user_ttr              (stub)
    28    gstat_user_post_rate        (0.0)
    29    gstat_user_subreddit_entropy (0.0)

All rows are padded with zeros; no NaNs are produced.

Smoke-test CLI
--------------
Run the file directly to verify dimensionalities and basic statistics:

    $ python hypernetwork_dataset_10000.py --pretok --hierarchical

By default the script loads
*/sciclone/home/thwalsh/hypernets/data/train_full_10000_v2.parquet* together
with the matching global and instance feature tables.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import numbers
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Iterable, List

import warnings
from tqdm.std import TqdmWarning  # type: ignore

warnings.filterwarnings("ignore", category=TqdmWarning)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

# ---------- full author-static feature list (except sent_mean) ----------
G_SIGNALS: List[str] = [
    "gstat_personality_raw",
    "gstat_personality_logits",
    "gstat_personality_traits",
    "gstat_personality_z",
    "gstat_gap_sentiment",
    "gstat_user_sent_var",
    "gstat_user_len_mean",
    "gstat_user_ttr",
    "gstat_user_post_rate",
    "gstat_user_subreddit_entropy",
    "gstat_punct_ratio",
    "gstat_question_ratio",
    "gstat_caps_ratio",
    "gstat_profanity_ratio",
    "gstat_firstperson_ratio",
    "gstat_readability_fk",
    "gstat_weekend_ratio",
    "gstat_link_ratio",
    "gstat_reply_delay_mean",
    "gstat_hour_hist",
]

# ---------- dynamic import of base dataset ------------------------------
def _load_base_dataset() -> type:
    search_paths: List[Path] = []
    try:
        search_paths.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    search_paths.append(Path.cwd())
    search_paths.append(Path("/sciclone/home/thwalsh/hypernets/data_scripts"))

    for directory in search_paths:
        cand = directory / "dataset_10000.py"
        if cand.exists():
            spec = importlib.util.spec_from_file_location("dataset_10000", str(cand))
            mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            assert spec.loader is not None
            spec.loader.exec_module(mod)  # type: ignore[arg-type]
            sys.modules["dataset_10000"] = mod
            return mod.RedditConversationDataset10000  # type: ignore[attr-defined]
    raise ModuleNotFoundError("dataset_10000.py not found")


RedditConversationDataset10000 = _load_base_dataset()

# ---------- helper utilities -------------------------------------------
_ACT2ID = {"statement": 0, "question": 1, "gratitude": 2, "other": 3}


def _is_scalar(x) -> bool:
    return isinstance(x, numbers.Number) or isinstance(x, np.generic)


def _flatten_numeric(v) -> List[float]:
    if _is_scalar(v):
        return [float(v)]
    if isinstance(v, str):
        return [0.0]
    if isinstance(v, Iterable):
        res: List[float] = []
        for elem in v:
            if _is_scalar(elem):
                res.append(float(elem))
            elif isinstance(elem, Iterable):
                res.extend(_flatten_numeric(elem))
        return res or [0.0]
    return [0.0]


# ---------- dataset wrapper --------------------------------------------
class HypernetConversationDataset10000(Dataset):
    _sent_pipe = None  # shared HF sentiment pipeline
    _pers_pipe = None  # shared HF personality pipeline

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        global_features_df: pd.DataFrame,
        instance_features_df: pd.DataFrame,
        *,
        hierarchical: bool = False,
        max_length: int = 512,
        add_special_tokens: bool = True,
        pretokenize: bool = False,
    ):
        super().__init__()
        self.hierarchical = bool(hierarchical)

        # token / text part
        self._text_ds = RedditConversationDataset10000(
            dataframe=df,
            tokenizer=tokenizer,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            pretokenize=False,
        )

        # fast look-up dicts
        self._g_dict = {
            int(r["gid"]): r.drop("gid").to_dict() for _, r in global_features_df.iterrows()
        }
        self._i_dict = {
            (int(r["gid"]), int(r["target_user_id"])): r.drop(
                ["gid", "target_user_id"]
            ).to_dict()
            for _, r in instance_features_df.iterrows()
        }

        # default column selections
        self._g_cols: List[str] = G_SIGNALS.copy()
        self._i_cols: List[str] = (
            [c for c in instance_features_df.columns if c not in ("gid", "target_user_id")]
            if self.hierarchical
            else []
        )

        # cached dims (lazy)
        self._g_dim = None
        self._i_dim = None

        # optional eager pre-tokenize
        self._cache = [self._merge(self._text_ds[i]) for i in range(len(self._text_ds))] if pretokenize else None

    # ------------- quick g-vector extractor -----------------------------
    def quick_feature_extract(self, texts: List[str]) -> np.ndarray:
        import numpy as _np, torch as _torch
        from transformers import pipeline as _pipe

        if HypernetConversationDataset10000._sent_pipe is None:
            HypernetConversationDataset10000._sent_pipe = _pipe(
                "sentiment-analysis",
                model="/sciclone/home/thwalsh/hypernets/models/distilbert-sst2",
                tokenizer="/sciclone/home/thwalsh/hypernets/models/distilbert-sst2",
                device=0 if _torch.cuda.is_available() else -1,
                return_all_scores=True,
            )
        if HypernetConversationDataset10000._pers_pipe is None:
            HypernetConversationDataset10000._pers_pipe = _pipe(
                "text-classification",
                model="/sciclone/home/thwalsh/hypernets/models/Personality_LM",
                tokenizer="/sciclone/home/thwalsh/hypernets/models/Personality_LM",
                device=0 if _torch.cuda.is_available() else -1,
                return_all_scores=True,
            )

        N = len(texts)
        rows = _np.zeros((N, 9), dtype=_np.float32)  # 5 OCEAN + 4 sentiment/placeholder

        sent = HypernetConversationDataset10000._sent_pipe(
            texts, batch_size=32, truncation=True, max_length=256
        )
        pos_prob = _np.array([row[1]["score"] for row in sent], _np.float32)
        rows[:, 5] = pos_prob
        rows[:, 6] = pos_prob - 0.5  # centered sentiment

        pers_logits = HypernetConversationDataset10000._pers_pipe(
            texts, batch_size=32, truncation=True, max_length=64
        )
        logits = _np.array([[p["score"] for p in r] for r in pers_logits], _np.float32)
        probs = _np.exp(logits) / _np.exp(logits).sum(1, keepdims=True)
        centred = probs - _np.median(probs, axis=0, keepdims=True)
        rows[:, 0:5] = centred

        # remaining two placeholders stay at zero
        return rows

    # ------------- public helpers --------------------------------------
    def set_selected_features(
        self,
        global_cols: List[str],
        instance_cols: List[str] | None = None,
        *,
        strict: bool = False,
    ) -> None:
        def _check(requested: List[str], sample_row: dict, scope: str) -> List[str]:
            missing = [c for c in requested if c not in sample_row]
            if missing:
                msg = f"{scope} columns not found and will be ignored -> {missing}"
                if strict:
                    raise KeyError(msg)
                logging.warning(msg)
            return [c for c in requested if c not in missing]

        any_g_row = next(iter(self._g_dict.values()), {})
        any_i_row = next(iter(self._i_dict.values()), {})

        self._g_cols = _check(list(global_cols), any_g_row, "global")

        if self.hierarchical:
            if instance_cols is None:
                raise ValueError("instance_cols must be provided in hierarchical mode")
            self._i_cols = _check(list(instance_cols), any_i_row, "instance")
        else:
            self._i_cols = []

        self._g_dim = None
        self._i_dim = None

    # ------------- convenience read-only properties --------------------
    @property
    def g_dim(self) -> int:
        if self._g_dim is None:
            self._g_dim = int(self[0]["global_features"].numel())
        return self._g_dim

    @property
    def i_dim(self) -> int:
        if self._i_dim is None:
            self._i_dim = int(self[0]["instance_features"].numel())
        return self._i_dim
    
    # ---------- expose the column names in the dataset -----------------
    @property
    def global_columns(self) -> list[str]:
        """
        Current list of author‑level feature columns returned by
        ``set_selected_features``.  Needed by the training harness when
        it trims / re‑orders vectors after loading a checkpoint.
        """
        return list(self._g_cols)

    @property
    def instance_columns(self) -> list[str]:
        """
        Current list of instance‑level feature columns (hierarchical mode).
        Returns an empty list in flat mode.
        """
        return list(self._i_cols)
        
    # ------------- Dataset API ----------------------------------------
    def __len__(self) -> int:
        return len(self._text_ds)

    def __getitem__(self, idx: int) -> dict:
        return self._cache[idx] if self._cache is not None else self._merge(self._text_ds[idx])

    # ------------- internal join --------------------------------------
    def _merge(self, base: dict) -> dict:
        gid = int(base["gid"])
        uid = int(base["target_user_id"])

        g_row = self._g_dict.get(gid, {})
        g_vec: List[float] = []
        for col in self._g_cols:
            g_vec.extend(_flatten_numeric(g_row.get(col, 0.0)))
        g_tensor = torch.tensor(g_vec, dtype=torch.float32)

        if self.hierarchical:
            i_row = self._i_dict.get((gid, uid), {})
            i_vec: List[float] = []
            for col in self._i_cols:
                val = i_row.get(col, 0.0)
                if col == "istat_reply_act":
                    val = _ACT2ID.get(str(val).lower(), _ACT2ID["other"])
                i_vec.extend(_flatten_numeric(val))
            i_tensor = torch.tensor(i_vec, dtype=torch.float32)
        else:
            i_tensor = torch.zeros(0, dtype=torch.float32)

        sample = dict(base)
        sample["global_features"] = g_tensor
        sample["instance_features"] = i_tensor
        return sample


# ---------- smoke-test CLI --------------------------------------------
def _smoke(df, gdf, idf, ds) -> None:
    counts = Counter(map(int, df["target_user_id"]))
    s = ds[0]
    report = f"""
    Hypernet dataset smoke-test
    ---------------------------
    samples               : {len(ds):,}
    unique users          : {df['target_user_id'].nunique():,}
    samples / user        : min={min(counts.values())}, median={median(counts.values())}, max={max(counts.values())}
    global feature rows   : {len(gdf):,}
    instance feature rows : {len(idf):,}
    hierarchical mode     : {ds.hierarchical}
    global vector dim     : {s['global_features'].numel()}
    instance vector dim   : {s['instance_features'].numel()}
    """
    print(textwrap.dedent(report))


_DEFAULT_ROOT = Path("/sciclone/home/thwalsh/hypernets/data")

if __name__ == "__main__":
    t0 = time.time()
    ap = argparse.ArgumentParser(description="Hyper-network dataset smoke-test")
    ap.add_argument(
        "--parquet",
        type=Path,
        default=_DEFAULT_ROOT / "train_full_10000_v2.parquet",
    )
    ap.add_argument(
        "--global_parquet",
        type=Path,
        default=_DEFAULT_ROOT / "global_features_10000.parquet",
    )
    ap.add_argument(
        "--instance_parquet",
        type=Path,
        default=_DEFAULT_ROOT / "instance_features_10000.parquet",
    )
    ap.add_argument(
        "--tokenizer",
        type=Path,
        default="/sciclone/home/thwalsh/hypernets/base_models/pythia_125m_clean",
    )
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--pretok", action="store_true")
    ap.add_argument("--hierarchical", action="store_true")
    ap.add_argument("-f", "--f", help=argparse.SUPPRESS)
    args = ap.parse_args()

    for p in (
        args.parquet,
        args.global_parquet,
        args.instance_parquet,
        args.tokenizer,
    ):
        if not p.exists():
            sys.exit(f"missing path -> {p}")

    df = pd.read_parquet(args.parquet)
    gdf = pd.read_parquet(args.global_parquet)
    idf = pd.read_parquet(args.instance_parquet)

    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        tok.pad_token_id = tok.convert_tokens_to_ids("[PAD]")

    ds = HypernetConversationDataset10000(
        df=df,
        tokenizer=tok,
        global_features_df=gdf,
        instance_features_df=idf,
        hierarchical=args.hierarchical,
        max_length=args.max_len,
        pretokenize=args.pretok,
    )

    ds.set_selected_features(
        G_SIGNALS,
        [
            c
            for c in idf.columns
            if c not in ("gid", "target_user_id")
        ]
        if args.hierarchical
        else [],
    )

    _smoke(df, gdf, idf, ds)
    print(f"completed in {time.time() - t0:.1f} s")