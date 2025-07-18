#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hypernetwork_feature_builder.py
=====================================

Creates the **global-** and **instance-level** feature tables that supply the
conditioning signals for our hypernetwork-conditioned PEFT models.  
For every Reddit *thread* we emit exactly **one** global row and **one**
instance row; these are later joined onto the cleaned train/val/test splits.

A tiny MLP hyper-network consumes the resulting vectors and predicts additive
parameter offsets **δθ** that are fused into a *frozen* PEFT backbone
(Bias-only, Adapter, LoRA, or Prefix) at run-time.

Feature taxonomy
----------------
All features fall along two orthogonal axes:

* **Scope**  GLOBAL (author-level) vs. INSTANCE (reply-level)
* **Temporal nature** STATIC (long-lived, slow to drift) vs. DYNAMIC
  (thread-specific, short-lived)

GLOBAL-STATIC — one vector per author *u*
-----------------------------------------
Hard-baked stylistic and behavioural fingerprints aggregated over the full
2010-2016 posting history.  After concatenation the block has **59 numeric
components** (see dimensional notes in brackets).

1. **Bias-corrected OCEAN probabilities**  
   *Column* `gstat_personality_traits` (ℝ⁵)  
   Per-post logits from **KevSun/Personality_LM** are soft-maxed, the
   corpus-median of each trait is subtracted to remove model bias, and the
   centred vectors are averaged across all posts so that each component
   reflects the *relative* strength (positive = above-median, negative =
   below-median) of **Agreeableness, Openness, Conscientiousness,
   Extraversion, Neuroticism**.

   1a. **Personality raw probs** `gstat_personality_raw` (ℝ⁵) — plain means
       *before* bias correction, useful for audits and ablations.  
   1b. **Personality logits**    `gstat_personality_logits` (ℝ⁵) — averaged,
       unnormalised, retain calibration information.  
   1c. **Trait *z*-scores**      `gstat_personality_z` (ℝ⁵) — traits after
       z-scoring across all 10 000 users; large magnitudes mark *extreme*
       personality profiles that the hyper-net may want to accentuate or damp.

2. **Gap sentiment** `gstat_gap_sentiment` (scalar)  
   Author’s mean post polarity **minus** the global Reddit baseline measured
   on 1 000 random comments; zero therefore denotes “average Reddit tone”.

3. **User sentiment mean** `gstat_user_sent_mean` (scalar)  
   Straight average of the DistilBERT polarity over all posts.

4. **User sentiment variance** `gstat_user_sent_var` (scalar)  
   Captures emotional *volatility* — do they swing from praise to rant or keep
   a steady tone?

5. **Mean post length** `gstat_user_len_mean` (scalar, tokens)  
   Typical generated-token budget, informative for depth vs. terseness.

6. **Type-token ratio** `gstat_user_ttr` (scalar)  
   Lexical diversity; high values suggest richer vocabularies.

7. **Post rate** `gstat_user_post_rate` (scalar, posts / day)  
   Activity level averaged over the six-year window.

8. **Subreddit entropy** `gstat_user_subreddit_entropy` (scalar, bits)  
   Diversity of topical participation; 0 bit = mono-sub, ≈ log₂( #subs ) at
   the other extreme.

9. **Punctuation density** `gstat_punct_ratio` (scalar)  
   Non-alphanumeric char ratio, proxy for expressive punctuation (“!!!”, “??”).

10. **Question ratio** `gstat_question_ratio` (scalar)  
    Share of posts that *end* with “?”, signalling inquisitiveness.

11. **Capital-word ratio** `gstat_caps_ratio` (scalar)  
    Fraction of tokens that are **ALL-CAPS** (len > 1), a shoutiness cue.

12. **Profanity ratio** `gstat_profanity_ratio` (scalar)  
    Hits per token against a 40-word swear lexicon; distinguishes polite,
    neutral and foul-mouthed authors.

13. **First-person ratio** `gstat_firstperson_ratio` (scalar)  
    Frequency of “I / me / my …”, a self-reference indicator.

14. **Readability (Flesch–Kincaid)** `gstat_readability_fk` (scalar)  
    Median grade level; large numbers mark convoluted prose,
    small numbers mark very simple language.

15. **Weekend posting ratio** `gstat_weekend_ratio` (scalar)  
    Proportion of comments published on Sat/Sun — lifestyle / schedule cue.

16. **Hyperlink ratio** `gstat_link_ratio` (scalar)  
    Fraction of posts containing at least one *http(s)://* link; separates
    citation-heavy users from chatty storytellers.

17. **Reply-delay mean** `gstat_reply_delay_mean` (scalar, minutes)  
    Average lag between the last context comment and the author’s reply;
    smaller = reactive, larger = contemplative.

18. **Hour-of-day histogram** `gstat_hour_hist` (ℝ²⁴)  
    Normalised 24-bin vector telling *when* the author is online; circadian
    fingerprint that the hyper-net can exploit for temporal coherence.

GLOBAL-DYNAMIC — one row per *thread g*
---------------------------------------
* **Thread sentiment** `gdyn_thread_sentiment` (scalar) — mean polarity of the
  *context* surrounding the target reply.  
* **Thread embedding** `gdyn_thread_embed` (ℝ³⁸⁴) — MiniLM sentence embedding
  of the concatenated context giving the hyper-net a semantic snapshot.

INSTANCE-STATIC — one row per *reply r*
---------------------------------------
* **Reply sentiment** `istat_reply_sentiment` (scalar) — polarity of the
  gold reply to be predicted.  
* **Dialogue act** `istat_reply_act` (categorical) —  
  0 = statement 1 = question 2 = gratitude.

INSTANCE-DYNAMIC — one row per *reply r*
----------------------------------------
* **Sentiment trend** `idyn_sentiment_trend` (scalar)  
  Reply sentiment − thread sentiment; positive = cheerier than context.  
* **Short-term post rate** `idyn_user_post_rate` (scalar)  
  Author’s posting cadence over the *past 7 days* — bursts vs. lulls.  
* **Semantic shift** `idyn_semantic_shift` (scalar)  
  1 − cosine(reply embedding, context embedding); large values flag replies
  that introduce *new* topical material.

Dimensional summary
-------------------
* **GLOBAL-STATIC** 59 floats  
  (20 personality + 24 hour-hist + 15 scalar cues)  
* **GLOBAL-DYNAMIC** 1 + 384 = 385 floats per thread  
* **INSTANCE** (Static 2 + Dynamic 3) = 5 floats per reply

Mathematical framing
--------------------
Let **g ∈ ℝ⁵⁹** and **i ∈ ℝ⁵**.

* **Flat mode**  `δθ = H_φ(g)`  
* **Hierarchical** `δθ = H_φ(g, i)`

The MLP **H<sub>φ</sub>** clamps δθ to ±0.05 and adds it to the PEFT
parameters θ̄ **on-the-fly**, giving us author-specific (flat) or
author-plus-reply-specific (hierarchical) adaptation without touching the
20 B-parameter backbone.

Pipeline overview
-----------------
1. **Load** the cleaned train/val/test splits (≈ 3.34 M rows).  
2. **Infer** sentiment, personality, MiniLM embeddings and the 15 stylistic
   ratios in GPU batches.  
3. **Accumulate** author-level statistics, reply delays, and 24-hour posting
   histograms.  
4. **Write**  
   * `global_features.parquet` (one row / thread)  
   * `instance_features.parquet` (one row / reply)  
   plus JSON side-cars for column order and z-score normalisation.  
5. **Resume / multi-GPU** friendly via shard filtering and chunk checkpoints.

Reference
---------
Wang R., Sun K. (2024). *Continuous Output Personality Detection Models via
Mixed Strategy Training*, arXiv:2406.16223.
"""

# ————————————————————— IMPORTS (GPU-agnostic) ————————————————————— #
import json, logging, os, sys, time, warnings, argparse, re, string
from math import log2
from pathlib import Path
from statistics import mean, median
from typing import Dict, List
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch.multiprocessing as mp

# ---------- optional readability dependency ------------------------------ #
try:
    import textstat as _textstat                                   # noqa: F401
    def _fk_grade(txt: str) -> float:                              # noqa: E302
        return _textstat.flesch_kincaid_grade(txt) if txt else 0.0
except ImportError:
    def _fk_grade(txt: str) -> float:                              # noqa: E302
        return 0.0
# ———————————————————————————————————————————————————————————————— #

# ————————————————————— CONFIG ————————————————————— #
DATA_DIR   = Path("/sciclone/home/thwalsh/hypernets/data")
TRAIN_PARQ = DATA_DIR / "train_data.parquet"
VAL_PARQ   = DATA_DIR / "val_data.parquet"
TEST_PARQ  = DATA_DIR / "test_data.parquet"

OUT_G_PARQ = DATA_DIR / "global_features.parquet"
OUT_I_PARQ = DATA_DIR / "instance_features.parquet"
OUT_STATS  = DATA_DIR / "feature_norm_stats.json"

SENT_MODEL    = "/sciclone/home/thwalsh/hypernets/models/distilbert-sst2"
SBERT_PATH    = "/sciclone/home/thwalsh/hypernets/sentence_transformers/all-MiniLM-L6-v2"
PERSONA_MODEL = "/sciclone/home/thwalsh/hypernets/models/Personality_LM"

CORPUS_DAYS = (pd.Timestamp("2016-01-01") - pd.Timestamp("2010-01-01")).days + 1

TRAIT_ABBRS  = ["A", "O", "C", "E", "N"]
TRAIT_NAMES: list[str] = []   # filled once torch is available

_PROFANITY = {
    "fuck","shit","damn","bitch","bastard","asshole","dick","crap","piss","darn",
    "bollocks","bugger","bloody","shithead","shitheads","fucker","motherfucker",
    "fucking","cunt","cunts","cock","prick","whore","slut","douche","douchebag",
}
_FIRST_PERSON = {"i", "me", "my", "mine"}

_PUNCT_TABLE = str.maketrans("", "", string.ascii_letters + string.digits + " ")

def _init_trait_names(torch_mod):
    global TRAIT_NAMES
    if TRAIT_NAMES:
        return
    args_bin = Path(PERSONA_MODEL) / "training_args.bin"
    if args_bin.exists():
        try:
            TRAIT_NAMES = [l.lower() for l in torch_mod.load(args_bin)["label_list"]]
        except Exception:
            TRAIT_NAMES = [
                "agreeableness", "openness",
                "conscientiousness", "extraversion", "neuroticism",
            ]
    else:
        TRAIT_NAMES = [
            "agreeableness", "openness",
            "conscientiousness", "extraversion", "neuroticism",
        ]

# ——— LOGGING ——— #
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ————————————————————— PIPELINES ————————————————————— #
def build_sentiment_pipe(gpu_id: int):
    from transformers import pipeline, logging as hf_log
    hf_log.set_verbosity_error()
    return pipeline(
        "sentiment-analysis",
        model=SENT_MODEL,
        tokenizer=SENT_MODEL,
        device=gpu_id if gpu_id >= 0 else -1,
        return_all_scores=True,
        batch_size=64,
    )

def build_personality_pipe(gpu_id: int):
    from transformers import pipeline, logging as hf_log
    hf_log.set_verbosity_error()
    return pipeline(
        "text-classification",
        model=PERSONA_MODEL,
        tokenizer=PERSONA_MODEL,
        device=gpu_id if gpu_id >= 0 else -1,
        return_all_scores=True,
        batch_size=64,
    )

# ————————————————————— UTILITIES ————————————————————— #
def preprocess(txt: str, max_toks: int = 256) -> str:
    toks = (txt or "").split()
    return " ".join(toks[:max_toks])

POLAR_CACHE: Dict[str, float] = {}
POLAR_CACHE_MAX = 500_000

def _polar_score_from_pipe(batch: list[str]) -> list[float]:
    outs = SENT_PIPE(batch, truncation=True, max_length=256, padding=True)
    return [
        next(d["score"] for d in s if d["label"].upper().startswith("POS")) -
        next(d["score"] for d in s if d["label"].upper().startswith("NEG"))
        for s in outs
    ]

def polarity_batch(texts: list[str]) -> list[float]:
    need_lookup = [t for t in texts if t not in POLAR_CACHE]
    for i in range(0, len(need_lookup), 64):
        sub = need_lookup[i : i + 64]
        for txt, sc in zip(sub, _polar_score_from_pipe(sub)):
            POLAR_CACHE[txt] = sc
    while len(POLAR_CACHE) > POLAR_CACHE_MAX:
        POLAR_CACHE.pop(next(iter(POLAR_CACHE)))
    return [POLAR_CACHE[t] for t in texts]

def mean_embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros(384, dtype=np.float32)
    vecs = SENT_EMBED.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return vecs.mean(axis=0).astype(np.float32)

def dialogue_act(text: str) -> int:
    t = (text or "").lower().strip()
    if t.endswith("?"):
        return 1
    if "thank" in t:
        return 2
    return 0

def _cols_sidecar(p: Path) -> Path:
    return p.with_name(p.stem + "_cols.json")

def _replace_stem(p: Path, new_stem: str) -> Path:
    return p.with_name(f"{new_stem}{p.suffix}")

def _punct_ratio(text: str) -> float:
    if not text:
        return 0.0
    punct_chars = text.translate(_PUNCT_TABLE)
    return len(punct_chars) / max(1, len(text))

def _caps_ratio(text: str) -> float:
    toks = text.split()
    if not toks:
        return 0.0
    caps = sum(1 for t in toks if t.isupper() and len(t) > 1)
    return caps / len(toks)

def _profanity_ratio(text: str) -> float:
    toks = re.findall(r"[A-Za-z']+", text.lower())
    if not toks:
        return 0.0
    bad = sum(1 for t in toks if t in _PROFANITY)
    return bad / len(toks)

def _firstperson_ratio(text: str) -> float:
    toks = re.findall(r"[A-Za-z']+", text.lower())
    if not toks:
        return 0.0
    fp = sum(1 for t in toks if t in _FIRST_PERSON)
    return fp / len(toks)

def _link_flag(text: str) -> bool:
    return bool(re.search(r"https?://", text))

# ————————————————————— MAIN ————————————————————— #
def main(gpu_id: int) -> None:
    import torch
    from sentence_transformers import SentenceTransformer, util as sbert_util

    _init_trait_names(torch)

    t0 = time.time()
    device = torch.device(
        f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id >= 0 else "cpu"
    )
    logging.info("Device chosen by rank: %s", device)

    cpu_threads = int(os.getenv("CPU_THREADS", "8"))
    torch.set_num_threads(cpu_threads)
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    logging.info("Using %d CPU threads for Torch / NumPy", cpu_threads)

    if (OUT_G_PARQ.exists() or OUT_I_PARQ.exists()) and not (
        "--resume" in sys.argv or "--force" in sys.argv
    ):
        raise RuntimeError(
            "Output Parquet file(s) already exist. "
            "Run with --resume (to continue) or --force (to overwrite) explicitly."
        )

    # build GPU-specific pipelines
    global SENT_PIPE, PERSONA_PIPE, SENT_EMBED
    SENT_PIPE = build_sentiment_pipe(gpu_id)
    PERSONA_PIPE = build_personality_pipe(gpu_id)
    SENT_EMBED = SentenceTransformer(SBERT_PATH, device=device)

    # ——— load & shard rows ——— #
    df_full = pd.concat(
        map(pd.read_parquet, (TRAIN_PARQ, VAL_PARQ, TEST_PARQ)), ignore_index=True
    )

    SHARDS = int(os.environ.get("WORLD_SIZE", "1"))
    RANK = int(os.environ.get("WORKER_ID", "0"))

    df = df_full[df_full["gid"] % SHARDS == RANK].reset_index(drop=True)
    logging.info(
        "Rank %s retained %d / %d rows after sharding filter",
        RANK,
        len(df),
        len(df_full),
    )

    n_total = df["gid"].nunique()

    id2label = {
        int(k): v.lower() for k, v in PERSONA_PIPE.model.config.id2label.items()
    }
    idx_remap = {
        pi: TRAIT_NAMES.index(lbl) for pi, lbl in id2label.items() if lbl in TRAIT_NAMES
    }
    for i in range(len(id2label)):
        idx_remap.setdefault(i, i)
    logging.info("Trait index remap: %s", idx_remap)

    global_sent = (
        df["text"]
        .sample(min(1000, len(df)))
        .apply(lambda t: polarity_batch([preprocess(t)])[0])
        .mean()
    )

    persona_cache: Dict[int, Dict[str, object]] = {}
    user_stat_cache: Dict[int, Dict[str, float]] = {}
    raw_vecs: list[np.ndarray] = []

    _delay_sum: Dict[int, float] = defaultdict(float)
    _delay_cnt: Dict[int, int] = defaultdict(int)
    _hour_hist: Dict[int, np.ndarray] = defaultdict(
        lambda: np.zeros(24, dtype=np.float32)
    )

    CHUNK_SIZE = 100_000
    G_WRITER = None
    I_WRITER = None
    FIRST_CHUNK_LOG = True

    CHECKPOINT_PATH = (
        DATA_DIR / f"feature_builder_ckpt_{os.environ.get('WORKER_ID','0')}.json"
    )

    def _save_ckpt(last_gid: int) -> None:
        with open(CHECKPOINT_PATH, "w") as fh:
            json.dump({"last_gid": last_gid}, fh)

    def _load_ckpt() -> int:
        if CHECKPOINT_PATH.exists():
            try:
                return int(json.load(open(CHECKPOINT_PATH))["last_gid"])
            except Exception:
                pass
        return -1

    last_done_gid = _load_ckpt() if "--resume" in sys.argv else -1
    latest_gid = last_done_gid

    def _flush_chunks(g_buf: list[dict], i_buf: list[dict]) -> None:
        import pyarrow as pa, pyarrow.parquet as pq

        nonlocal G_WRITER, I_WRITER, FIRST_CHUNK_LOG, latest_gid

        if g_buf:
            tbl = pa.Table.from_pylist(g_buf)
            if G_WRITER is None:
                G_WRITER = pq.ParquetWriter(
                    OUT_G_PARQ, tbl.schema, compression="snappy"
                )
            G_WRITER.write_table(tbl)
            g_buf.clear()

        if i_buf:
            tbl = pa.Table.from_pylist(i_buf)
            if I_WRITER is None:
                I_WRITER = pq.ParquetWriter(
                    OUT_I_PARQ, tbl.schema, compression="snappy"
                )
            I_WRITER.write_table(tbl)
            i_buf.clear()

        _save_ckpt(latest_gid)
        if FIRST_CHUNK_LOG:
            logging.info(
                "[rank %s] writer-append active – chunk size = %d",
                os.environ.get("WORKER_ID", "0"),
                CHUNK_SIZE,
            )
            FIRST_CHUNK_LOG = False
        import gc

        gc.collect()

    def log_progress(processed_threads: int) -> None:
        if not raw_vecs:
            return
        mat = np.vstack(raw_vecs)
        mu_raw = mat.mean(axis=0).round(3).tolist()
        logging.info(
            "Progress %6d / %6d (%.1f %%) | users %5d | μraw=%s",
            processed_threads,
            n_total,
            processed_threads / n_total * 100,
            len(raw_vecs),
            mu_raw,
        )

    g_rows, i_rows = [], []
    for n, (gid, grp) in enumerate(df.groupby("gid", sort=False), 1):
        if gid <= last_done_gid:
            continue

        tgt_mask = grp["group_label"].str.endswith("_target")
        if not tgt_mask.any():
            continue

        reply_row = grp.loc[tgt_mask].iloc[-1]
        reply_txt = preprocess(reply_row["text"])
        if not reply_txt:
            continue

        ctx_texts = [preprocess(t) for t in grp.loc[~tgt_mask, "text"].tolist()]
        uid = int(reply_row["target_user_id"])

        ctx_sent = float(mean(polarity_batch(ctx_texts))) if ctx_texts else 0.0
        ctx_embed = mean_embed(ctx_texts)

        reply_sent = polarity_batch([reply_txt])[0]
        reply_embed = SENT_EMBED.encode(
            reply_txt,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        sentiment_trend = reply_sent - ctx_sent
        semantic_shift = float(1 - sbert_util.cos_sim(ctx_embed, reply_embed).item())

        # ---------- accumulate user-level statistics -------------------
        if uid not in user_stat_cache:
            u_rows = df[df["target_user_id"] == uid]
            toks = u_rows["text"].str.split().apply(len)
            corpus = toks.sum()
            sent_series = u_rows["text"].apply(
                lambda t: polarity_batch([preprocess(t)])[0]
            )
            punct_ratio = u_rows["text"].apply(_punct_ratio).mean()
            question_ratio = (u_rows["text"].str.strip().str.endswith("?")).mean()
            caps_ratio = u_rows["text"].apply(_caps_ratio).mean()
            profanity_ratio = u_rows["text"].apply(_profanity_ratio).mean()
            firstperson_ratio = u_rows["text"].apply(_firstperson_ratio).mean()
            readability_scores = u_rows["text"].apply(_fk_grade)
            readability_fk = float(median(readability_scores.tolist()))
            link_ratio = u_rows["text"].apply(_link_flag).mean()
            weekend_ratio = (u_rows["created_utc"].dt.dayofweek >= 5).mean()

            st = dict(
                sent_mean=float(sent_series.mean()),
                sent_var=float(sent_series.var(ddof=0)),
                len_mean=float(toks.mean()),
                ttr=float(
                    len(set(" ".join(u_rows["text"].tolist()).split())) / max(1, corpus)
                ),
                post_rate=float(len(u_rows) / CORPUS_DAYS),
                punct_ratio=punct_ratio,
                question_ratio=question_ratio,
                caps_ratio=caps_ratio,
                profanity_ratio=profanity_ratio,
                firstperson_ratio=firstperson_ratio,
                readability_fk=readability_fk,
                weekend_ratio=weekend_ratio,
                link_ratio=link_ratio,
            )
            if "subreddit" in u_rows.columns:
                probs = u_rows["subreddit"].value_counts(normalize=True) + 1e-9
                st["sr_entropy"] = float(-sum(p * log2(p) for p in probs))
            else:
                st["sr_entropy"] = 0.0
            user_stat_cache[uid] = st

        # ---------- reply delay & hour histogram -----------------------
        try:
            last_context_time = grp.loc[~tgt_mask, "created_utc"].max()
            reply_time = reply_row["created_utc"]
            if pd.notna(last_context_time) and pd.notna(reply_time):
                delay_min = (reply_time - last_context_time).total_seconds() / 60.0
                _delay_sum[uid] += delay_min
                _delay_cnt[uid] += 1
        except Exception:
            pass

        for ts in grp["created_utc"]:
            if pd.notna(ts):
                _hour_hist[uid][ts.hour] += 1

        # ---------- personality vectors (cached) -----------------------
        if uid not in persona_cache:
            posts = (
                df[df["target_user_id"] == uid]["text"]
                .head(200)
                .apply(preprocess)
                .tolist()
            )
            prob_sum = np.zeros(5, dtype=np.float32)
            logit_sum = np.zeros(5, dtype=np.float32)
            pipe_out = PERSONA_PIPE(
                posts, batch_size=64, truncation=True, padding=True, max_length=64
            )
            for logits_list in pipe_out:
                logits = torch.tensor([d["score"] for d in logits_list])
                probs = torch.softmax(logits, dim=0)
                for pi, p in enumerate(probs):
                    ti = idx_remap[pi]
                    if ti < 5:
                        prob_sum[ti] += p.item()
                        logit_sum[ti] += logits[pi].item()
            mean_prob = (prob_sum / len(posts)).tolist()
            persona_cache[uid] = {
                "prob_raw": mean_prob,
                "logits": (logit_sum / len(posts)).tolist(),
            }
            raw_vecs.append(np.array(mean_prob, dtype=np.float32))

        g_rows.append(
            dict(
                gid=gid,
                target_user_id=uid,
                gstat_personality_raw=persona_cache[uid]["prob_raw"],
                gstat_personality_logits=persona_cache[uid]["logits"],
                gstat_gap_sentiment=(user_stat_cache[uid]["sent_mean"] - global_sent),
                gstat_user_sent_mean=user_stat_cache[uid]["sent_mean"],
                gstat_user_sent_var=user_stat_cache[uid]["sent_var"],
                gstat_user_len_mean=user_stat_cache[uid]["len_mean"],
                gstat_user_ttr=user_stat_cache[uid]["ttr"],
                gstat_user_post_rate=user_stat_cache[uid]["post_rate"],
                gstat_user_subreddit_entropy=user_stat_cache[uid]["sr_entropy"],
                gstat_punct_ratio=user_stat_cache[uid]["punct_ratio"],
                gstat_question_ratio=user_stat_cache[uid]["question_ratio"],
                gstat_caps_ratio=user_stat_cache[uid]["caps_ratio"],
                gstat_profanity_ratio=user_stat_cache[uid]["profanity_ratio"],
                gstat_firstperson_ratio=user_stat_cache[uid]["firstperson_ratio"],
                gstat_readability_fk=user_stat_cache[uid]["readability_fk"],
                gstat_weekend_ratio=user_stat_cache[uid]["weekend_ratio"],
                gstat_link_ratio=user_stat_cache[uid]["link_ratio"],
                gdyn_thread_sentiment=ctx_sent,
                gdyn_thread_embed=ctx_embed.tolist(),
            )
        )

        i_rows.append(
            dict(
                gid=gid,
                target_user_id=uid,
                istat_reply_sentiment=reply_sent,
                istat_reply_act=dialogue_act(reply_txt),
                idyn_sentiment_trend=sentiment_trend,
                idyn_user_post_rate=user_stat_cache[uid]["post_rate"],
                idyn_semantic_shift=semantic_shift,
            )
        )

        latest_gid = gid

        if n % CHUNK_SIZE == 0:
            _flush_chunks(g_rows, i_rows)
        if n % 20_000 == 0:
            log_progress(n)
        if n % 10_000 == 0:
            _save_ckpt(latest_gid)

    _flush_chunks(g_rows, i_rows)
    if G_WRITER:
        G_WRITER.close()
    if I_WRITER:
        I_WRITER.close()
    _save_ckpt(latest_gid)
    logging.info("All chunks written – exiting main loop.")

    # ——— merging & stats ——— #
    logging.info("Loading full Parquet files for statistics …")
    g_df = pd.read_parquet(OUT_G_PARQ)
    i_df = pd.read_parquet(OUT_I_PARQ)

    # ------------------------------------------------------------------ #
    # ❶ guarantee 1‑row‑per‑gid & 1‑row‑per‑(gid, uid) before any further
    #   processing so downstream scripts no longer need to deduplicate
    # ------------------------------------------------------------------ #
    pre_dup = len(g_df)
    g_df = g_df.drop_duplicates("gid", keep="first").reset_index(drop=True)
    logging.info("Dropped %d duplicate global rows", pre_dup - len(g_df))

    pre_i_dup = len(i_df)
    i_df = (
        i_df.drop_duplicates(["gid", "target_user_id"], keep="first")
        .reset_index(drop=True)
    )
    logging.info(
        "Dropped %d duplicate instance rows", pre_i_dup - len(i_df)
    )

    # ---------- hour histogram & reply‑delay ---------------------------
    for uid, hist in _hour_hist.items():
        total = hist.sum()
        _hour_hist[uid] = (
            (hist / total).tolist() if total > 0 else np.zeros(24, np.float32).tolist()
        )
    g_df["gstat_hour_hist"] = g_df["target_user_id"].apply(
        lambda u: _hour_hist[int(u)]
    )
    g_df["gstat_reply_delay_mean"] = g_df["target_user_id"].apply(
        lambda u: _delay_sum[int(u)] / _delay_cnt[int(u)]
        if _delay_cnt[int(u)]
        else 0.0
    )

    # ---------- bias‑correct OCEAN & z‑score ---------------------------
    med = np.median(np.vstack(g_df["gstat_personality_raw"]), axis=0)
    g_df["gstat_personality_traits"] = g_df["gstat_personality_raw"].apply(
        lambda v: (np.array(v) - med).tolist()
    )

    traits_mat = np.vstack(g_df["gstat_personality_traits"])
    g_mean = traits_mat.mean(axis=0)
    g_std = traits_mat.std(axis=0) + 1e-6
    g_df["gstat_personality_z"] = g_df["gstat_personality_traits"].apply(
        lambda v: ((np.array(v) - g_mean) / g_std).tolist()
    )

    list_cols = {
        "gstat_personality_raw",
        "gstat_personality_logits",
        "gstat_personality_traits",
        "gstat_personality_z",
        "gdyn_thread_embed",
        "gstat_hour_hist",
    }
    for c in g_df.columns:
        if c not in list_cols and g_df[c].dtype != object:
            g_df[c] = g_df[c].astype("float32")
    for c in i_df.columns:
        if c not in ("gid", "target_user_id", "istat_reply_act"):
            i_df[c] = i_df[c].astype("float32")

    skip = list_cols | {"gid", "target_user_id"}
    norm_stats = {
        c: {
            "mean": float(g_df[c].mean()),
            "std": float(g_df[c].std(ddof=0) + 1e-6),
        }
        for c in g_df.columns
        if c not in skip
    }
    for c, s in norm_stats.items():
        g_df[c] = (g_df[c] - s["mean"]) / s["std"]

    g_df.to_parquet(OUT_G_PARQ, index=False)
    i_df.to_parquet(OUT_I_PARQ, index=False)

    with open(OUT_STATS, "w") as fh:
        json.dump(norm_stats, fh)
    with open(_cols_sidecar(OUT_G_PARQ), "w") as fh:
        json.dump(list(g_df.columns), fh)
    with open(_cols_sidecar(OUT_I_PARQ), "w") as fh:
        json.dump(list(i_df.columns), fh)

    logging.info(
        "Done | global rows %d | instance rows %d | wall-time %.1fs",
        len(g_df),
        len(i_df),
        time.time() - t0,
    )

# ————————————————————— MULTI-GPU ENTRYPOINT ————————————————————— #
def _worker(rank: int, world_size: int, argv: list[str]) -> None:
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["WORKER_ID"] = str(rank)

    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)

    global OUT_G_PARQ, OUT_I_PARQ, OUT_STATS
    OUT_G_PARQ = _replace_stem(OUT_G_PARQ, f"{OUT_G_PARQ.stem}_w{rank}")
    OUT_I_PARQ = _replace_stem(OUT_I_PARQ, f"{OUT_I_PARQ.stem}_w{rank}")
    OUT_STATS = _replace_stem(OUT_STATS, f"{OUT_STATS.stem}_w{rank}")

    if "--force" not in argv and "--resume" not in argv:
        ckpt = Path(OUT_G_PARQ.parent) / f"feature_builder_ckpt_{rank}.json"
        if ckpt.exists():
            argv.append("--resume")
            logging.info("[rank %d] checkpoint detected – enabling --resume", rank)

    sys.argv = [sys.argv[0]] + argv
    main(rank)

def merge_parquets(world_size: int) -> None:
    import pyarrow.parquet as pq, pyarrow as pa

    def _merge(stem: Path):
        parts = [_replace_stem(stem, f"{stem.stem}_w{r}") for r in range(world_size)]
        tables = [pq.read_table(p) for p in parts]
        pq.write_table(pa.concat_tables(tables), stem, compression="snappy")
        for p in parts:
            p.unlink(missing_ok=True)

    def _merge_sidecar(stem: Path):
        cols = None
        for r in range(world_size):
            p = _cols_sidecar(_replace_stem(stem, f"{stem.stem}_w{r}"))
            if p.exists():
                with open(p) as fh:
                    cols_r = json.load(fh)
                if cols is None:
                    cols = cols_r
                p.unlink(missing_ok=True)
        if cols is not None:
            with open(_cols_sidecar(stem), "w") as fh:
                json.dump(cols, fh)

    _merge(OUT_G_PARQ)
    _merge(OUT_I_PARQ)
    _merge_sidecar(OUT_G_PARQ)
    _merge_sidecar(OUT_I_PARQ)
    stats = {}
    for r in range(world_size):
        p = _replace_stem(OUT_STATS, f"{OUT_STATS.stem}_w{r}")
        with open(p) as fh:
            stats_r = json.load(fh)
        for k, v in stats_r.items():
            stats.setdefault(k, {"mean": 0.0, "std": 0.0})
            stats[k]["mean"] += v["mean"] / world_size
            stats[k]["std"] += v["std"] / world_size
        p.unlink(missing_ok=True)
    with open(OUT_STATS, "w") as fh:
        json.dump(stats, fh)

# ————————————————————— CLI ————————————————————— #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shards", type=int, default=1, help="Number of worker processes / GPUs"
    )
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    if (
        args.shards == 1
        and "--force" not in rest
        and "--resume" not in rest
        and (OUT_G_PARQ.exists() or OUT_I_PARQ.exists())
        and (DATA_DIR / "feature_builder_ckpt_0.json").exists()
    ):
        rest.append("--resume")
        logging.info("Checkpoint detected – automatically enabling --resume")
    if args.shards == 1:
        main(0)
    else:
        mp.spawn(_worker, args=(args.shards, rest), nprocs=args.shards, join=True)
        merge_parquets(args.shards)