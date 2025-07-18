#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_data.py (updated 2025-06-30)
====================================
This script gathers 300 full conversation threads for each of 10000
target users, selected from a 50000-user candidate pool, using the
2010-to-2016 Reddit corpus.  A conversation thread contains every comment
from the root submission through the user’s final comment in that thread.
Outputs are written as Parquet and CSV.

Mathematical Context
--------------------
Let each user *u* be assigned an integer ID.  We require each finished user to
have 300 complete threads, each thread being the chain:

    submission → …intermediate comments… → final user comment

If a user posts multiple times in a thread we keep only their *last* comment
as the anchor.  Each chain is labeled with a unique integer derived from
``uid * 1_000_000 + thread_index`` so collisions are impossible.

Where it Fits in the Ablation Study
-----------------------------------
1) Produces ≈ 3,000,000 user-anchored threads once 10000 users are complete.  
2) Preserves full conversational context so downstream scripts can rely on
   intact chains.  
3) Supports future train / val / test splits that keep whole threads together.

Implementation Outline
----------------------
1) Load candidate users (50000 max) from *all_user_data_2005_2016.parquet*.  
2) Iterate over the monthly Reddit dumps (2010-01 → 2016-12).  
3) For users still below 300 threads, collect their latest comment in each
   link_id and build the full chain up to the submission root.  
4) Tag every line with a ``group{big}_{idx}`` label, marking the final user
   comment as ``…_target``.  
5) When 10000 users reach 300 threads we stop, combine monthly parts, and
   write reddit_threads_2010_2016.parquet and .csv.
"""

import os
import bz2
import lzma
import zstandard as zstd
import logging
import time
import json
import html
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from collections import defaultdict
from dask.distributed import Client
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

COMMENTS_DIR        = "/sciclone/data10/twford/reddit/reddit/comments"
SUBMISSIONS_DIR     = "/sciclone/data10/twford/reddit/reddit/submissions"
ALL_USERS_PARQUET   = "/sciclone/home/thwalsh/hypernets/data/all_user_data_2005_2016.parquet"
OUTPUT_DIR          = "/sciclone/home/thwalsh/hypernets/data"
OUTPUT_FILE         = os.path.join(OUTPUT_DIR, "reddit_threads_2010_2016.parquet")
USER_PROGRESS_FILE  = os.path.join(OUTPUT_DIR, "user_progress.csv")
LOG_DIR             = "/sciclone/home/thwalsh/hypernets/log_files"

# Sampling / processing parameters
COMMENTS_PER_USER   = 300            # target threads per finished user
USERS_TO_COLLECT    = 10_000         # stop after this many users reach the target
CANDIDATE_POOL      = 50_000         # pool size
NUM_WORKERS         = 8
CHUNK_SIZE          = 10

# Quality / trimming parameters
MIN_CHAIN_LEN       = 2              # submission + anchor
MIN_TARGET_TOKENS   = 3
N_CONTEXT           = 5
TOKEN_CAP           = 384

# Date range to scan
START_YYYYMM        = "2010-01"
END_YYYYMM          = "2016-12"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "extract_data.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info(
    f"extract_data.py launched: pool={CANDIDATE_POOL}, "
    f"goal={USERS_TO_COLLECT} users × {COMMENTS_PER_USER} threads."
)

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

TOKENIZER = AutoTokenizer.from_pretrained(
    "/home/hypernets/bert_models/roberta-large",
    use_fast=True
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def open_compressed_file(fname):
    """Return a text handle for .bz2, .xz, or .zst Reddit dumps."""
    if fname.endswith(".bz2"):
        return bz2.open(fname, "rt", encoding="utf-8")
    if fname.endswith(".xz"):
        return lzma.open(fname, "rt", encoding="utf-8")
    if fname.endswith(".zst"):
        import io
        f = open(fname, "rb")
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        return io.TextIOWrapper(dctx.stream_reader(f), encoding="utf-8")
    raise ValueError(f"Unsupported compression for {fname}")

def parse_datetime(ts):
    """Convert epoch-seconds to pandas Timestamp (NaT on failure)."""
    try:
        return pd.to_datetime(float(ts), unit="s", errors="coerce")
    except Exception:
        return pd.NaT

def is_in_date_range(fname, start_ym, end_ym):
    """True if filename represents a dump in the YYYY-MM span."""
    if not fname.endswith((".bz2", ".xz", ".zst")):
        return False
    if not (fname.startswith(("RC_", "RS_", "RS_v2_"))):
        return False
    try:
        part = fname.split("_")[1].split(".")[0]
        return start_ym <= part <= end_ym
    except Exception:
        return False

def load_submissions_for_month(path):
    """Return {link_id: submission_record} for a monthly submissions dump."""
    subs = {}
    try:
        with open_compressed_file(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    sid = d.get("id", "")
                    lid = f"t3_{sid}"
                    text = (d.get("title", "") + "\n" + d.get("selftext", "")).strip()
                    subs[lid] = {
                        "type": "submission",
                        "created_utc": parse_datetime(d.get("created_utc", 0)),
                        "author": d.get("author", ""),
                        "content": html.unescape(text),
                        "subreddit": d.get("subreddit", ""),
                        "parent_id": "",
                        "comment_id": sid,
                        "link_id": lid,
                    }
                except Exception:
                    continue
    except Exception as e:
        logging.error(f"Failed to load submissions {path}: {e}")
    return subs

def build_chain_bottom_up(comment_id, comments, subs, parent_map):
    """Return the comment-to-submission chain in chronological order."""
    path = []
    cur = comment_id
    while True:
        path.append(cur)
        p = parent_map.get(cur, "")
        if not p:
            break
        if p.startswith("t3_"):
            path.append(p)
            break
        cur = p.split("_", 1)[-1]
    path.reverse()
    chain = []
    for pid in path:
        rec = subs.get(pid) if pid.startswith("t3_") else comments.get(pid)
        if rec:
            chain.append(rec)
    return chain

def load_candidate_users():
    """
    Load up to CANDIDATE_POOL users active between 2010-01-01 and 2016-12-31.
    Returns (user_list, {user: uid}).
    """
    df = pd.read_parquet(ALL_USERS_PARQUET)
    df["earliest_date"] = pd.to_datetime(df["earliest_date"], errors="coerce")
    df["latest_date"]   = pd.to_datetime(df["latest_date"],   errors="coerce")
    mask = (df["earliest_date"] >= "2010-01-01") & (df["latest_date"] <= "2016-12-31")
    users = df.loc[mask, "user"].dropna().unique().tolist()
    if len(users) > CANDIDATE_POOL:
        users = users[:CANDIDATE_POOL]
    logging.info(f"Loaded {len(users)} candidate users (pool size)")
    return users, {u: i + 1 for i, u in enumerate(users)}

def load_user_progress(path):
    """Return {user: threads_collected} from CSV or empty dict."""
    if not os.path.exists(path):
        return {}
    prog = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            u, c = ln.strip().split(",", 1)
            prog[int(u)] = int(c)
    return prog

def save_user_progress(path, data):
    """Overwrite CSV progress file."""
    with open(path, "w", encoding="utf-8") as f:
        for u, c in data.items():
            f.write(f"{u},{c}\n")

# ---------------------------------------------------------------------------
# Progress reconciliation helper
# ---------------------------------------------------------------------------

def reconcile_counts_with_existing_parts(counts: dict[int, int]) -> dict[int, int]:
    """
    Scan existing month-level Parquet files and ensure the per-user thread
    counts match the number of anchor rows already on disk.
    """
    month_parts = [
        os.path.join(OUTPUT_DIR, f)
        for f in os.listdir(OUTPUT_DIR)
        if f.startswith("reddit_threads_") and f.endswith(".parquet")
           and len(f.split("_")) == 3  # YYYY-MM parts only
    ]
    if not month_parts:
        return counts

    anchor_totals = defaultdict(int)
    cols = ["target_user_id", "group_label"]
    for p in month_parts:
        df_p = pq.read_table(p, columns=cols).to_pandas()
        anchors = df_p[df_p["group_label"].str.endswith("_target")]
        anchor_totals.update(anchors["target_user_id"].value_counts().to_dict())

    for uid, already in anchor_totals.items():
        counts[uid] = max(counts.get(uid, 0), already)
    return counts

# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def token_count(text: str) -> int:
    return len(TOKENIZER.encode(text, add_special_tokens=False))

def chain_token_len(chain) -> int:
    return sum(token_count(rec["content"]) for rec in chain[:-1])

# ---------------------------------------------------------------------------
# Chunk processing
# ---------------------------------------------------------------------------

TRIM_STAT = {"trimmed_rows": 0, "rowcap_kept": 0, "token_trim": 0, "short_target": 0}

def process_chunk(comment_paths, sub_path, candidates, uid_map, counts, completed):
    subs = load_submissions_for_month(sub_path)
    rows = []
    local_cnt = dict(counts)
    local_done = set(completed)

    for cf in comment_paths:
        try:
            comments, parent_map = {}, {}
            user_map = defaultdict(lambda: defaultdict(list))
            with open_compressed_file(cf) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    cid = d.get("id", "")
                    if not cid:
                        continue
                    rec = {
                        "type": "comment",
                        "created_utc": parse_datetime(d.get("created_utc", 0)),
                        "author": d.get("author", ""),
                        "content": html.unescape((d.get("body", "") or "").strip()),
                        "subreddit": d.get("subreddit", ""),
                        "parent_id": d.get("parent_id", ""),
                        "comment_id": cid,
                        "link_id": d.get("link_id", ""),
                    }
                    comments[cid] = rec
                    parent_map[cid] = rec["parent_id"]
                    author = rec["author"]
                    if (
                        author in candidates
                        and author not in local_done
                        and local_cnt.get(author, 0) < COMMENTS_PER_USER
                    ):
                        user_map[author][rec["link_id"]].append(cid)
        except Exception as e:
            logging.error(f"Error reading {cf}: {e}")
            continue

        for user, thread_map in user_map.items():
            if user in local_done:
                continue
            if len(local_done) >= USERS_TO_COLLECT:
                break
            for lid, cids in thread_map.items():
                if lid not in subs:
                    continue
                final_cid = max(
                    cids,
                    key=lambda cid: comments[cid]["created_utc"] or pd.Timestamp.min,
                )
                chain = build_chain_bottom_up(final_cid, comments, subs, parent_map)
                if (
                    not chain
                    or len(chain) < MIN_CHAIN_LEN
                    or chain[-1]["author"] != user
                ):
                    continue
                if token_count(chain[-1]["content"]) < MIN_TARGET_TOKENS:
                    TRIM_STAT["short_target"] += 1
                    continue
                if len(chain) > N_CONTEXT + 1:
                    TRIM_STAT["trimmed_rows"] += len(chain) - (N_CONTEXT + 1)
                    chain = chain[-(N_CONTEXT + 1):]
                else:
                    TRIM_STAT["rowcap_kept"] += 1
                while len(chain) > 2 and chain_token_len(chain) > TOKEN_CAP:
                    chain.pop(0)
                    TRIM_STAT["token_trim"] += 1
                if len(chain) < 2:
                    continue
                num = local_cnt.get(user, 0) + 1
                big = uid_map[user] * 1_000_000 + num
                for idx, rec in enumerate(chain):
                    label = (
                        f"group{big}_{idx + 1}"
                        if idx < len(chain) - 1
                        else f"group{big}_target"
                    )
                    rows.append(
                        {
                            "target_user_id": uid_map[user],
                            "group_label": label,
                            "created_utc": rec["created_utc"],
                            "author": rec["author"],
                            "content": rec["content"],
                            "subreddit": rec["subreddit"],
                            "parent_id": rec["parent_id"],
                            "comment_id": rec["comment_id"],
                            "link_id": rec["link_id"],
                        }
                    )
                local_cnt[user] = num
                if num >= COMMENTS_PER_USER:
                    local_done.add(user)
                    logging.info(f"User {user} reached {COMMENTS_PER_USER} threads.")
                    break
            if len(local_done) >= USERS_TO_COLLECT:
                break

    df_chunk = pd.DataFrame(rows)
    if "created_utc" in df_chunk.columns and not pd.api.types.is_datetime64_any_dtype(
        df_chunk["created_utc"]
    ):
        df_chunk["created_utc"] = pd.to_datetime(
            df_chunk["created_utc"], errors="coerce"
        )
    return df_chunk, local_cnt, local_done

# ---------------------------------------------------------------------------
# Final combination
# ---------------------------------------------------------------------------

def combine_and_filter(completed, uid_map):
    parts = [
        os.path.join(OUTPUT_DIR, f)
        for f in os.listdir(OUTPUT_DIR)
        if f.startswith("reddit_threads_") and f.endswith(".parquet") and len(f.split("_")) == 3
    ]
    dfs = []
    rev = {v: k for k, v in uid_map.items()}
    for p in parts:
        df = pd.read_parquet(p)
        df["usr"] = df["target_user_id"].map(rev)
        df = df[df["usr"].isin(completed)]
        if not df.empty:
            dfs.append(df.drop(columns="usr"))
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        pq.write_table(pa.Table.from_pandas(combined), OUTPUT_FILE)
        combined.to_csv(OUTPUT_FILE.replace(".parquet", ".csv"), index=False)
        logging.info(f"Wrote final dataset with {len(combined)} rows")
    else:
        logging.info("No data to combine for final output")

# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    client = Client(n_workers=NUM_WORKERS, threads_per_worker=1)
    logging.info(f"Dask client started with {NUM_WORKERS} workers")

    users, uid_map = load_candidate_users()
    user_set = set(users)

    progress = load_user_progress(USER_PROGRESS_FILE)
    progress = reconcile_counts_with_existing_parts(progress)
    counts   = defaultdict(int, progress)
    done     = {u for u, c in counts.items() if c >= COMMENTS_PER_USER}

    cmts = sorted(
        [f for f in os.listdir(COMMENTS_DIR) if is_in_date_range(f, START_YYYYMM, END_YYYYMM)]
    )
    subs = sorted(
        [f for f in os.listdir(SUBMISSIONS_DIR) if is_in_date_range(f, START_YYYYMM, END_YYYYMM)]
    )
    sub_map = {fn.split("_")[1].split(".")[0]: fn for fn in subs}
    com_map = defaultdict(list)
    for fn in cmts:
        key = fn.split("_")[1].split(".")[0]
        com_map[key].append(fn)

    for month in sorted(set(sub_map) & set(com_map)):
        if len(done) >= USERS_TO_COLLECT:
            break
        out_month = os.path.join(OUTPUT_DIR, f"reddit_threads_{month}.parquet")
        if os.path.exists(out_month):
            logging.info(f"Month {month} already processed; skipping")
            continue

        sub_path = os.path.join(SUBMISSIONS_DIR, sub_map[month])
        comment_files = com_map[month]
        chunk_files = []
        for i in range(0, len(comment_files), CHUNK_SIZE):
            if len(done) >= USERS_TO_COLLECT:
                break
            chunk_paths = [
                os.path.join(COMMENTS_DIR, f) for f in comment_files[i : i + CHUNK_SIZE]
            ]
            df_chunk, counts, done = process_chunk(
                chunk_paths, sub_path, user_set, uid_map, counts, done
            )
            save_user_progress(USER_PROGRESS_FILE, counts)

            if not df_chunk.empty:
                chunk_file = os.path.join(
                    OUTPUT_DIR, f"reddit_threads_{month}_chunk_{i}.parquet"
                )
                pq.write_table(pa.Table.from_pandas(df_chunk), chunk_file)
                chunk_files.append(chunk_file)

        if chunk_files:
            month_dfs = [pd.read_parquet(cf) for cf in chunk_files]
            month_df = pd.concat(month_dfs, ignore_index=True)
            pq.write_table(pa.Table.from_pandas(month_df), out_month)
            for cf in chunk_files:
                os.remove(cf)

        logging.info(f"Trim stats cumulative: {TRIM_STAT}")

    if len(done) < USERS_TO_COLLECT:
        logging.error(
            "Only %d users reached %d threads — extraction failed.",
            len(done), COMMENTS_PER_USER
        )
        client.close()
        raise RuntimeError("Extraction finished with insufficient qualified users")

    combine_and_filter(done, uid_map)
    client.close()
    logging.info(f"Completed in {(time.time() - t0) / 60:.2f} minutes "
                 f"— final trim stats {TRIM_STAT}")
    print(f"Done. Elapsed {(time.time() - t0) / 60:.2f} min")

if __name__ == "__main__":
    tstart = time.time()
    main()
    tend = time.time()
    logging.info(f"Finished in {tend - tstart:.2f} seconds.")
    print(f"Done. Elapsed: {tend - tstart:.2f} sec")