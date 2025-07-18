#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
clean_split_data.py
=========================

Reads the full Reddit corpus for 10 000 target users, performs light text
clean-up, **removes malformed conversation groups**, and emits user-balanced
train/val/test Parquet + CSV files.

Mathematical Context
--------------------
A conversation group *g* with lines l₁…lₙ (lₙ belongs to the target user) is
**invalid** when

    • lₙ.text  == ""                        (empty target)           ,  or
    • Σᵢ^{n-1} |lᵢ.text| == 0               (empty context).

Such groups contain no learnable signal and are pruned prior to splitting.

Where it Fits in the Ablation Study
-----------------------------------
Ensures that (i) language-model training never sees empty inputs,
(ii) every remaining `gid` has matching global/instance feature rows, and
(iii) each user retains a healthy number of threads after filtering.

Implementation Outline
----------------------
1. Ingest the raw dataset (`reddit_threads_2010_2016.parquet`).  
2. Clean each `content` string (ASCII norm, slash-cmd & emoticon markers, etc.).  
3. **Detect + drop** malformed `gid`s (empty target or context).  
4. Perform a per-user 70 / 20 / 10 chain-level split.  
5. Persist `<split>_data.parquet` / `.csv` and
   `coverage_report.csv`.  
6. Log detailed statistics for reproducibility.
"""


from __future__ import annotations
import os, re, time, random, logging
from pathlib import Path

import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq

# ───────────────────────────── paths ────────────────────────────── #
DATA_DIR  = Path("/sciclone/home/thwalsh/hypernets/data")
RAW_FILE  = DATA_DIR / "reddit_threads_2010_2016.parquet"

TRAIN_CSV  = DATA_DIR / "train_data.csv"
VAL_CSV    = DATA_DIR / "val_data.csv"
TEST_CSV   = DATA_DIR / "test_data.csv"

TRAIN_PARQ = DATA_DIR / "train_data.parquet"
VAL_PARQ   = DATA_DIR / "val_data.parquet"
TEST_PARQ  = DATA_DIR / "test_data.parquet"

COVERAGE_CSV = DATA_DIR / "coverage_report.csv"

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.20, 0.10
RANDOM_SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s"
)

# ──────────────────────── text utilities ───────────────────────── #
def replace_deleted_removed(txt: str) -> str:
    if not isinstance(txt, str):
        return "this is a filler comment"
    t = txt.strip().lower()
    if t == "[deleted]":
        return "Sorry, this comment was deleted by the user or moderator"
    if t == "[removed]":
        return "Sorry, this comment was removed by the user or moderator"
    return txt

def interpret_slash_commands_emoticons(text: str) -> str:
    text = re.sub(r"/([a-zA-Z0-9]+)", r"[slashcmd: \1]", text)
    emoticon_map = {
        r":-?\)": "[smiley]",
        r":-?\(": "[sadface]",
        r":-?D":  "[grin]",
    }
    for pat, repl in emoticon_map.items():
        text = re.sub(pat, repl, text)
    return text

def remove_non_ascii(text: str) -> str:
    return "".join(ch for ch in text if ch.isascii())

def _collapse_periods(m):
    s = m.group(0)
    return "..." if len(s) > 3 else s

def deep_clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = "this is a filler comment"
    text = replace_deleted_removed(text)
    text = remove_non_ascii(text)
    text = interpret_slash_commands_emoticons(text)

    text = text.replace(">", "")
    text = re.sub(r"(?i)(https?://\S+|www\.\S+)", "this hyperlink", text)
    text = re.sub(r"\.{2,}", _collapse_periods, text)
    text = re.sub(r"[()\[\]]+", ",", text)
    text = re.sub(r",+", ",", text)
    text = re.sub(r"[*_]+", "", text)
    text = re.sub(r'([!?"\'])(\1+)', r'\1', text)
    text = re.sub(r'\^+', '', text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text or text == ",":
        text = "this is a filler comment"

    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()

    alnum = sum(ch.isalnum() for ch in text)
    if len(text) > 15 and alnum / len(text) < 0.2:
        text = "this is a filler comment"
    return text

def parse_chain_id(label: str) -> str:
    return label.rsplit("_", 1)[0] if isinstance(label, str) else ""

def parse_gid(chain_id: str) -> int:
    try:
        return int(chain_id.replace("group", "", 1))
    except Exception:
        return -1

# ─────────────────────── per-user splitting ────────────────────── #
def split_per_user(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(RANDOM_SEED)
    dfs_train, dfs_val, dfs_test = [], [], []

    for uid, sub_df in df.groupby("target_user_id", sort=False):
        chains = sub_df["chain_id"].unique().tolist()
        rng.shuffle(chains)

        n = len(chains)
        n_train = int(TRAIN_RATIO * n)
        n_val   = int(VAL_RATIO   * n)
        n_test  = n - n_train - n_val

        tr, va, te = set(chains[:n_train]), set(chains[n_train:n_train+n_val]), set(chains[-n_test:])

        dfs_train.append(sub_df[sub_df["chain_id"].isin(tr)])
        dfs_val.append(  sub_df[sub_df["chain_id"].isin(va)])
        dfs_test.append( sub_df[sub_df["chain_id"].isin(te)])

    return (
        pd.concat(dfs_train, ignore_index=True),
        pd.concat(dfs_val,   ignore_index=True),
        pd.concat(dfs_test,  ignore_index=True),
    )

# ───────────────────────── main routine ────────────────────────── #
def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(RAW_FILE)

    logging.info("Reading raw data …")
    df = pd.read_parquet(RAW_FILE)
    logging.info("Loaded %d rows", len(df))

    req_cols = {"target_user_id", "group_label", "author", "content", "created_utc"}
    if not req_cols.issubset(df.columns):
        raise ValueError(f"Dataset missing columns: {req_cols - set(df.columns)}")

    df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce")

    if "created_utc" in df.columns:
        df.sort_values(["target_user_id", "group_label", "created_utc"], inplace=True, ignore_index=True)
    else:
        df.sort_values(["target_user_id", "group_label"], inplace=True, ignore_index=True)

    logging.info("Cleaning text …")
    df["cleaned_content"] = df["content"].apply(deep_clean_text)

    fillers = {
        "this is a filler comment",
        "sorry, this comment was deleted by the user or moderator",
        "sorry, this comment was removed by the user or moderator",
    }
    pre = len(df)
    df = df[~df["cleaned_content"].str.lower().isin(fillers)].copy()
    logging.info("Removed %d placeholder rows", pre - len(df))

    df["chain_id"] = df["group_label"].apply(parse_chain_id)
    df["gid"]      = df["chain_id"].apply(parse_gid)

    logging.info("Detecting malformed threads (empty target or context) …")
    bad_mask = (
        df.groupby("gid", sort=False)
          .apply(lambda g: g.tail(1)["cleaned_content"].iloc[0] == "" or
                           g.iloc[:-1]["cleaned_content"].str.len().sum() == 0)
          .rename("is_bad")
    )
    bad_gids = bad_mask[bad_mask].index
    logging.info("Found %d malformed conversation groups", len(bad_gids))
    df = df[~df["gid"].isin(bad_gids)].reset_index(drop=True)

    logging.info("Splitting per user (70/20/10) …")
    df_train, df_val, df_test = split_per_user(df)

    def _final(dfx: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "group_label":    dfx["group_label"],
            "text":           dfx["cleaned_content"],
            "gid":            dfx["gid"],
            "target_user_id": dfx["target_user_id"],
            "author":         dfx["author"],
            "created_utc":    dfx["created_utc"],
        })

    train_f, val_f, test_f = map(_final, (df_train, df_val, df_test))

    logging.info("Writing splits …")
    train_f.to_csv(TRAIN_CSV, index=False)
    val_f.to_csv(VAL_CSV,     index=False)
    test_f.to_csv(TEST_CSV,   index=False)

    pq.write_table(pa.Table.from_pandas(train_f), TRAIN_PARQ)
    pq.write_table(pa.Table.from_pandas(val_f),   VAL_PARQ)
    pq.write_table(pa.Table.from_pandas(test_f),  TEST_PARQ)

    logging.info("Generating per-user coverage report …")
    cov = (
        pd.concat([train_f, val_f], ignore_index=True)
          .groupby("target_user_id")["gid"]
          .nunique()
          .rename("thread_count")
          .sort_values(ascending=False)
    )
    cov.to_csv(COVERAGE_CSV, header=True)
    logging.info("Coverage CSV → %s  (median = %.0f threads)", COVERAGE_CSV, cov.median())

    def _summary(name, dfx):
        logging.info("%5s  rows=%8d  groups=%6d", name, len(dfx), dfx["gid"].nunique())

    logging.info("===== SPLIT RESULTS =====")
    _summary("Train", train_f)
    _summary("Val",   val_f)
    _summary("Test",  test_f)
    logging.info("Total unique groups    : %d", df["gid"].nunique())
    logging.info("Total unique users     : %d", df["target_user_id"].nunique())
    logging.info("Cleaning + splitting done.")

if __name__ == "__main__":
    _t0 = time.time()
    main()
    logging.info("Finished in %.2f seconds.", time.time() - _t0)