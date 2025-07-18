#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataset_10000.py
================
A modular PyTorch `Dataset` for the cleaned Reddit corpus comprising
10 000 target users.  It builds (context → next-turn) training tuples and
offers a command-line smoke-test to verify dataset integrity.

Mathematical Context
--------------------
For each thread *g* with ordered lines l₁,…,lₙ (lₙ marked “_target”):

    xᵍ = l₁ ⧺ … ⧺ lₙ₋₁      (context)
    yᵍ = lₙ                  (label)

The language model maximises Σ_g log p(yᵍ | xᵍ, θ).

Where it Fits in the Ablation Study
-----------------------------------
This dataset wrapper is imported by every training / evaluation script
(plain PEFT and hyper-network alike) so that the data logic is defined
once and tested thoroughly.

Implementation Outline
----------------------
1.  Group rows by `gid`; extract the final “_target” line and the
    preceding context.
2.  Drop malformed groups (empty context or target).
3.  Tokenise on-the-fly (or cache) and return tensors ready for
    `nn.CrossEntropyLoss` (–100 on `[PAD]` tokens).
4.  `python dataset_10000.py --parquet <file>` runs a smoke-test that
    prints class counts, token-length stats, and basic sanity checks.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

# --------------------------------------------------------------------------- #
#                              Dataset definition
# --------------------------------------------------------------------------- #
class RedditConversationDataset10000(Dataset):
    """
    Returns dictionaries:

        {
          "input_ids":       LongTensor [max_length],
          "attention_mask":  LongTensor [max_length],
          "labels":          LongTensor [max_length],
          "target_user_id":  int,
          "gid":             int,
        }
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        *,
        max_length: int = 512,
        add_special_tokens: bool = True,
        pretokenize: bool = False,
    ):
        required = {"gid", "group_label", "text", "target_user_id"}
        if not required.issubset(dataframe.columns):
            raise ValueError(f"DataFrame must contain {required} – found {dataframe.columns.tolist()}")

        df = dataframe.copy()
        df.sort_values(["gid", "group_label"], inplace=True, ignore_index=True)

        items: list[dict] = []
        for gid, sub in df.groupby("gid", sort=False):
            tgt_rows = sub[sub["group_label"].str.endswith("_target")]
            if tgt_rows.empty:
                continue
            final_row = tgt_rows.iloc[-1]
            tgt_text = str(final_row["text"]).strip()
            if not tgt_text:
                continue

            sub_sorted = sub.sort_values("group_label")
            ctx_lines = sub_sorted.iloc[: sub_sorted.index.get_loc(final_row.name)]["text"].tolist()
            ctx_text = "\n".join(map(str, ctx_lines)).strip()
            if not ctx_text:
                continue

            items.append(
                dict(
                    gid=int(gid),
                    target_user_id=int(final_row["target_user_id"]),
                    context_text=ctx_text,
                    target_text=tgt_text,
                )
            )

        self.items = items
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.cached_data = [self._encode(it) for it in items] if pretokenize else None

    # ------------------------------------------------------------------ API
    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.cached_data[idx] if self.cached_data is not None else self._encode(self.items[idx])

    # ---------------------------------------------------------------- helpers
    def _encode(self, item: dict) -> dict[str, torch.Tensor | int]:
        ctx, tgt = item["context_text"], item["target_text"]

        enc_ctx = self.tokenizer(
            ctx,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )
        enc_tgt = self.tokenizer(
            tgt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )

        labels = enc_tgt["input_ids"].squeeze(0)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        return {
            "input_ids": enc_ctx["input_ids"].squeeze(0),
            "attention_mask": enc_ctx["attention_mask"].squeeze(0),
            "labels": labels,
            "target_user_id": item["target_user_id"],
            "gid": item["gid"],
        }

# --------------------------------------------------------------------------- #
#                               Smoke-test logic
# --------------------------------------------------------------------------- #
def _smoke_test(df: pd.DataFrame, ds: RedditConversationDataset10000, tok: PreTrainedTokenizer):
    n_samples = len(ds)
    n_users = df["target_user_id"].nunique()
    per_user = Counter(map(int, df["target_user_id"]))
    counts = list(per_user.values())

    lens_ctx, lens_tgt = [], []
    step = max(1, n_samples // 20_000)  # sample up to ~20k items
    for i in range(0, n_samples, step):
        item = ds[i]
        lens_ctx.append(int(item["attention_mask"].sum()))
        lens_tgt.append(int((item["labels"] != -100).sum()))

    def _stats(arr):
        return dict(mean=float(sum(arr) / len(arr)), median=int(median(arr)), max=int(max(arr)))

    buckets = defaultdict(int)
    for L in lens_ctx:
        if L <= 64:
            buckets["≤64"] += 1
        elif L <= 128:
            buckets["≤128"] += 1
        elif L <= 256:
            buckets["≤256"] += 1
        else:
            buckets[">256"] += 1

    print(
        textwrap.dedent(
            f"""
            ------------------------ Smoke-test summary ------------------------
            Samples                      : {n_samples:,}
            Users                        : {n_users:,}
            Samples / user               : min={min(counts)}, median={median(counts)}, max={max(counts)}
            Context token lengths        : {_stats(lens_ctx)}
            Target  token lengths        : {_stats(lens_tgt)}
            Context length buckets       : {dict(buckets)}
            Empty text rows (should be 0): {int(df['text'].str.strip().eq('').sum())}
            NaN text rows   (should be 0): {int(df['text'].isna().sum())}
            Tokeniser pad id             : {tok.pad_token_id}
            Sequence length (max_length) : {ds.max_length}
            -------------------------------------------------------------------
            """
        )
    )

# --------------------------------------------------------------------------- #
#                                   CLI
# --------------------------------------------------------------------------- #
DEFAULT_PARQUET = "/sciclone/home/thwalsh/hypernets/data/train_data_10000.parquet"
DEFAULT_TOKENIZER = "/sciclone/home/thwalsh/hypernets/base_models/pythia_125m_clean"

if __name__ == "__main__":
    tic = time.time()

    ap = argparse.ArgumentParser(
        description="Run a quick integrity smoke-test on a 10 000-user Reddit split."
    )
    ap.add_argument(
        "--parquet",
        type=Path,
        default=Path(DEFAULT_PARQUET),
        help=f"Path to a train/val/test parquet file (default: {DEFAULT_PARQUET})",
    )
    ap.add_argument(
        "--tokenizer",
        type=Path,
        default=Path(DEFAULT_TOKENIZER),
        help=f"HuggingFace tokenizer directory (default: {DEFAULT_TOKENIZER})",
    )
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--pretok", action="store_true", help="Pre-tokenise all samples at init")
    # suppress the automatically-added “-f / --f” arg when run via Jupyter
    ap.add_argument("-f", "--f", help=argparse.SUPPRESS)

    args = ap.parse_args()

    if not args.parquet.exists():
        sys.exit(f"Parquet file not found: {args.parquet}")
    if not args.tokenizer.exists():
        sys.exit(f"Tokenizer path not found: {args.tokenizer}")

    print(f"Loading DataFrame → {args.parquet}")
    df_raw = pd.read_parquet(args.parquet)
    print(f"Rows: {len(df_raw):,} | Columns: {df_raw.columns.tolist()}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        tok.pad_token_id = tok.convert_tokens_to_ids("[PAD]")

    ds = RedditConversationDataset10000(
        dataframe=df_raw,
        tokenizer=tok,
        max_length=args.max_len,
        pretokenize=args.pretok,
    )

    _smoke_test(df_raw, ds, tok)

    print(f"Smoke-test finished in {time.time() - tic:.1f} s")