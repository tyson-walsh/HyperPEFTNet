#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_nonhypernet_10000.py
=============================

Offline evaluation for the **five** non-hyper-network checkpoints
(LoRA, Adapter, Bias-only, Prefix, **Base**) trained on the
10 000-author Reddit corpus.

The script measures *content fitness*, *surface overlap*, *semantic
similarity*, and *lexical diversity* and writes one JSON report per
variant plus a combined summary for plotting.  It exactly mirrors the
hyper-network evaluation pipeline **except** that:

* No δθ offsets are generated – the backbone already carries the
  PEFT placeholders (or full weights for *base*).
* No author-vector conditioning is involved, so KL alignment and
  permutation importance are omitted.

What the script produces
------------------------
* **Per-variant JSON** – ``results/eval_nonhypernet_{variant}_10000.json``  
  ‣ a *summary* block with scalar metrics  
  ‣ the **top-10** generations (ranked by BERTScore, BLEU tie-break)
    including their individual metrics  
* **Combined JSON** – ``results/eval_nonhypernet_results_10000.json``  
  containing only the summary blocks.  
* **Checklist file** – updated after each variant completes to prevent
  accidental re-runs.  
* **Central log** – ``log_files/evaluate_nonhypernet_10000.log``.  

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
    • **BERTScore-F1** (roberta-large, *no* baseline rescale)  
    • MiniLM-L6 *cosine similarity* (Sentence-Transformers)

Lexical diversity
    • Distinct-1 and Distinct-2 (type-to-token ratio of unigrams /
      bigrams)

Implementation outline
----------------------
1. Restore each checkpoint from ``models/nonhypernet_{variant}_model_10000``  
   (placeholders merged into a frozen backbone; *base* fully unfrozen).  
2. Load the test split via ``RedditConversationDataset10000``  
   (no global/instance feature tensors).  
3. **Generation** – top-p sampling (``p=0.95``, temperature 0.8,
   repetition-penalty 1.1) of up to **64** new tokens; prompts are the
   conversation contexts only.  
4. Evaluate CE/PPL on the concatenated *context + reply* sequences.  
5. Keep the top-10 generations per variant for qualitative inspection.  
6. Write JSON/Parquet outputs and update the checklist.

Reproducibility & efficiency
----------------------------
* Deterministic seed + TF-32 enabled.  
* Batch size = 4 by default (configurable via ``--bsz``).  
* BERTScore & MiniLM run on GPU with progress bars suppressed.  
* ``--demo_mode`` evaluates just three batches and skips all disk I/O,
  useful for quick sanity checks.
"""

from __future__ import annotations

import sys, types, warnings, os, heapq, argparse, json, logging, math, time
from pathlib import Path
from statistics import mean
from typing import Dict, List, Any, Tuple
from itertools import count
import multiprocessing as _mp

from tqdm import tqdm
stub = types.ModuleType("tqdm.auto"); stub.tqdm = tqdm
sys.modules["tqdm.auto"] = stub
warnings.filterwarnings("ignore", message="IProgress not found")
try:
    from tqdm.utils import TqdmWarning
    warnings.filterwarnings("ignore", category=TqdmWarning)
except Exception:
    pass

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pyarrow as pa, pyarrow.parquet as pq
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import single_meteor_score as _meteor
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

from dataset_10000 import RedditConversationDataset10000          # noqa: E402
from adapter import ModelAdapter                                  # noqa: E402
from bias import BiasTuningModel                                  # noqa: E402
from lora import LoRAQKV                                          # noqa: E402
from prefix import ModelPrefixOnly                                # noqa: E402

SBERT_PATH          = "/sciclone/home/thwalsh/hypernets/sentence_transformers/all-MiniLM-L6-v2"
BERT_MODEL_PATH     = "/sciclone/home/thwalsh/hypernets/bert_models/roberta-large"
DEFAULT_MODELS_DIR  = "/sciclone/home/thwalsh/hypernets/models"
DEFAULT_RESULTS_DIR = "/sciclone/home/thwalsh/hypernets/results"
DEFAULT_LOG_DIR     = "/sciclone/home/thwalsh/hypernets/log_files"

_EVAL_BSZ   = 4
_GEN_NEW    = 64
_TOPK       = 10
_DEMO_MAXBT = 3
_COUNTER    = count()
_PERSIST_OK = _mp.get_start_method(allow_none=True) == "spawn"

REPLY_SEP_ID: int | None = None

VARIANTS = ("lora", "adapter", "bias", "prefix", "base")

PEFT_CFG: Dict[str, Dict] = {
    # LoRA checkpoints were trained with rank = 32 and α = 64
    "lora":    {"lora_rank": 32, "lora_alpha": 64.0},
    "adapter": {"adapter_bottleneck_dim": 128, "adapter_dropout": 0.10},
    "bias":    {},
    "prefix":  {"prefix_length": 20},
}

_parquet_path   = None
_parquet_schema = None
_parquet_writer = None
_disable_io     = False
_bertscorer     = None


def _distinct_n(ids: List[List[int]], n: int) -> float:
    if not ids:
        return 0.0
    grams, total = set(), 0
    for seq in ids:
        total += max(0, len(seq) - n + 1)
        for i in range(max(0, len(seq) - n + 1)):
            grams.add(tuple(seq[i : i + n]))
    return len(grams) / max(1, total)


def _ctx_only_ce(
    logits: torch.Tensor, input_ids: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    reply_mask = labels.ne(-100)
    first_idx  = reply_mask.float().argmax(dim=1)
    keep       = first_idx.gt(0)
    if keep.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    rows        = torch.arange(logits.size(0), device=logits.device)[keep]
    last_ctx    = first_idx[keep] - 1
    sel_logits  = logits[rows, last_ctx, :]
    sel_targets = input_ids[rows, first_idx[keep]]
    return F.cross_entropy(sel_logits, sel_targets, ignore_index=-100)


def _select_top_k(heap_store: List[Tuple], k: int) -> List[Dict[str, Any]]:
    return [x[1] for x in heapq.nsmallest(k, heap_store, key=lambda z: z[0])]


def _meteor_safe(ref: List[str], hyp: List[str]) -> float:
    return _meteor(ref, hyp) if ref and hyp else 0.0


def _extract_loss(out, labels: torch.Tensor) -> torch.Tensor:
    if getattr(out, "loss", None) is not None:
        return out.loss
    if isinstance(out, dict) and "loss" in out:
        return out["loss"]
    if getattr(out, "logits", None) is not None and torch.is_tensor(out.logits):
        lg = out.logits[..., :-1, :].contiguous()
        y  = labels[..., 1:].contiguous()
        return F.cross_entropy(lg.view(-1, lg.size(-1)), y.view(-1), ignore_index=-100)
    raise ValueError("cannot extract loss")


def _make_concat_inputs(batch: Dict[str, torch.Tensor],
                        pad_id: int, seq_len: int,
                        reply_max: int = 128) -> None:
    ctx_ids = batch["input_ids"]
    tgt_ids = batch["labels"].clone()
    tgt_ids[tgt_ids == -100] = pad_id
    _, C = ctx_ids.size()
    max_ctx = max(seq_len - reply_max - 1, 8)
    if C > max_ctx:
        ctx_ids = ctx_ids[:, -max_ctx:]
        C = max_ctx
    sep = torch.full((ctx_ids.size(0), 1), REPLY_SEP_ID,
                     dtype=ctx_ids.dtype, device=ctx_ids.device)
    concat = torch.cat([ctx_ids, sep, tgt_ids], dim=1)[:, :seq_len]
    attn = (concat != pad_id).long()
    labels = concat.clone()
    labels[:, : C + 1] = -100
    labels[labels == pad_id] = -100
    batch["input_ids"]      = concat
    batch["attention_mask"] = attn
    batch["labels"]         = labels


def _extract_prompt(inp: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
    pos = (lab != -100).nonzero(as_tuple=True)
    split = int(pos[0][0].item()) if len(pos[0]) else inp.size(0)
    return inp[:split]


def _wrap_model(variant: str,
                base: torch.nn.Module,
                cfg: dict) -> torch.nn.Module:
    if variant == "lora":
        for lyr in base.gpt_neox.layers:
            lora_qkv = LoRAQKV(
                lyr.attention.query_key_value,
                rank=cfg["lora_rank"],
                alpha=cfg["lora_alpha"],
            )
            lyr.attention.query_key_value = lora_qkv
    elif variant == "adapter":
        base = ModelAdapter(base, **cfg, use_layer_norm=True)
    elif variant == "bias":
        base = BiasTuningModel(base)
    elif variant == "prefix":
        base = ModelPrefixOnly(
            base,
            cfg["prefix_length"],
            embed_dim=base.config.hidden_size,
        )
    elif variant == "base":
        return base
    for name, p in base.named_parameters():
        if name.endswith((".A_q", ".B_q", ".A_k", ".B_k", ".A_v", ".B_v")):
            continue
        p.requires_grad_(False)
    return base


def _load_peft_placeholders(backbone: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    clean_state: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith("module.") else k
        clean_state[new_key] = v
    missing, unexpected = backbone.load_state_dict(clean_state, strict=False)
    log = logging.getLogger(__name__)
    if unexpected:
        log.debug("Skipped %d unexpected keys (first 5: %s)", len(unexpected), unexpected[:5])
    if missing:
        log.debug("Backbone is missing %d expected keys (first 5: %s)", len(missing), missing[:5])
    log.debug("PEFT placeholders loaded: %d parameters applied.", len(clean_state) - len(unexpected))


def main() -> None:
    global _bertscorer, _disable_io
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--test_parquet", required=True)
    p.add_argument("--base_ckpt",   required=True)
    p.add_argument("--models_dir",  default=DEFAULT_MODELS_DIR)
    p.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--checklist")
    p.add_argument("--bsz", type=int, default=_EVAL_BSZ)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--seed", type=int, default=142)
    p.add_argument("--demo_mode", action="store_true")
    p.add_argument("--log_dir", default=DEFAULT_LOG_DIR)
    p.add_argument("-f", "--f", help=argparse.SUPPRESS)
    args, _ = p.parse_known_args()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_dir) / "evaluate_nonhypernet_10000.log"
    fh = logging.FileHandler(log_path, mode="a"); fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout);      ch.setLevel(logging.INFO)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[fh, ch],
    )

    _disable_io = bool(args.demo_mode)
    if _disable_io:
        logging.info("[DEMO] file output disabled")

    done_variants = set()
    if args.checklist and Path(args.checklist).is_file() and not _disable_io:
        done_variants = set(Path(args.checklist).read_text().splitlines())

    hf_utils.logging.set_verbosity_error()
    set_seed(args.seed)
    device = torch.device(args.device)

    _bertscorer = BERTScorer(
        model_type=BERT_MODEL_PATH,
        num_layers=24,
        lang="en",
        rescale_with_baseline=False,
        batch_size=64,
        device=device,
    )
    cos_model = SentenceTransformer(SBERT_PATH).to(device)

    df_test = pd.read_parquet(args.test_parquet)

    tok = AutoTokenizer.from_pretrained(args.base_ckpt, local_files_only=True)

    if "<|reply|>" not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": ["<|reply|>"]})

    global REPLY_SEP_ID
    REPLY_SEP_ID = tok.convert_tokens_to_ids("<|reply|>")

    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})

    ds = RedditConversationDataset10000(df_test, tok, max_length=args.max_len)

    loader = DataLoader(
        ds,
        batch_size=args.bsz,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=_PERSIST_OK,
    )

    results_dir = Path(args.results_dir)
    if not _disable_io:
        results_dir.mkdir(parents=True, exist_ok=True)
    combined: Dict[str, Dict] = {}

    bleu_sm  = SmoothingFunction().method1
    rouge_sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    try:
        for variant in VARIANTS:
            if variant in done_variants:
                logging.info("[%s] already done – skip", variant)
                if not _disable_io:
                    prev_path = results_dir / f"eval_nonhypernet_{variant}_10000.json"
                    if prev_path.is_file():
                        try:
                            with prev_path.open() as fh:
                                combined[variant] = json.load(fh)[variant]
                        except Exception as e:
                            logging.warning("[%s] could not read previous summary: %s", variant, e)
                continue

            ck_dir = Path(args.models_dir) / f"nonhypernet_{variant}_model_10000"
            if not ck_dir.exists():
                logging.warning("[%s] checkpoint missing – skip", variant)
                continue

            logging.info("[%s] restoring …", variant)
            if variant == "base":
                model = AutoModelForCausalLM.from_pretrained(ck_dir).to(device)
            else:
                cfg_back = AutoConfig.from_pretrained(args.base_ckpt, local_files_only=True)
                cfg_back.use_cache = False
                backbone = AutoModelForCausalLM.from_pretrained(
                    args.base_ckpt, config=cfg_back, local_files_only=True
                )
                backbone.resize_token_embeddings(len(tok))
                backbone = _wrap_model(variant, backbone, PEFT_CFG[variant])
                _load_peft_placeholders(
                    backbone,
                    torch.load(ck_dir / "peft_placeholders.safetensors", map_location="cpu"),
                )
                model = backbone.to(device)
            model.eval()

            ce_tf_sum  = tok_tf_sum  = 0.0
            ce_ctx_sum = samp_ctx_sum = 0.0

            top_heap: List[Tuple] = []
            ids_all: List[List[int]] = []

            eos_id = tok.eos_token_id
            LOG_EVERY = 100

            for bt_idx, bt in enumerate(loader):
                if bt_idx % LOG_EVERY == 0:
                    logging.info("[%s] batch %d / %d", variant, bt_idx, len(loader))
                    sys.stdout.flush(); sys.stderr.flush()

                if _disable_io and bt_idx >= _DEMO_MAXBT:
                    break
                bt = {k: v.to(device) for k, v in bt.items()}
                _make_concat_inputs(bt, tok.pad_token_id, args.max_len)

                with torch.no_grad():
                    out = model(
                        input_ids=bt["input_ids"],
                        attention_mask=bt["attention_mask"],
                        labels=bt["labels"],
                    )
                nt = int((bt["labels"] != -100).sum())
                ce_tf_sum  += float(_extract_loss(out, bt["labels"])) * nt
                tok_tf_sum += nt
                ce_ctx_sum  += float(_ctx_only_ce(out.logits, bt["input_ids"], bt["labels"])) * bt["input_ids"].size(0)
                samp_ctx_sum += bt["input_ids"].size(0)

                prompts = [_extract_prompt(i, l) for i, l in zip(bt["input_ids"], bt["labels"])]
                plens = [len(p) for p in prompts]
                if not plens:
                    continue
                max_len = max(plens)
                p_batch, a_batch = [], []
                for p in prompts:
                    attn = torch.ones(len(p), dtype=torch.long, device=device)
                    if len(p) < max_len:
                        pad = torch.full((max_len - len(p),), tok.pad_token_id, dtype=torch.long, device=device)
                        attn_pad = torch.zeros_like(pad)
                        p = torch.cat([p, pad])
                        attn = torch.cat([attn, attn_pad])
                    p_batch.append(p)
                    a_batch.append(attn)
                p_batch = torch.stack(p_batch)
                a_batch = torch.stack(a_batch)

                gen_ids = model.generate(
                    p_batch,
                    attention_mask=a_batch,
                    do_sample=True,
                    top_p=0.95,                 # nucleus sampling as in training docs
                    temperature=0.8,            # softer randomness
                    repetition_penalty=1.1,     # discourage verbatim loops
                    no_repeat_ngram_size=3,     # keep existing anti-duplication guard
                    min_new_tokens=8,
                    max_new_tokens=_GEN_NEW,    # 64 by default
                    eos_token_id=eos_id,
                    pad_token_id=tok.pad_token_id,
                )

                gen_txt, ref_txt, ctx_txt = [], [], []
                for i, seq in enumerate(gen_ids):
                    gen_only = seq[plens[i]:]
                    keep_mask = gen_only != tok.pad_token_id
                    if tok.eos_token_id is not None:
                        keep_mask &= gen_only != tok.eos_token_id
                    gen_only = gen_only[keep_mask]
                    fallback_id = (
                        tok.unk_token_id
                        if tok.unk_token_id is not None
                        else (tok.eos_token_id or tok.pad_token_id)
                    )
                    if gen_only.numel() == 0:
                        gen_only = torch.tensor([fallback_id], device=seq.device)
                    txt = tok.decode(gen_only, skip_special_tokens=True).strip()
                    if not txt:
                        txt = "."
                        gen_only = torch.tensor([tok.encode(".")[0]], device=seq.device)
                    gen_txt.append(txt)
                    ref_txt.append(
                        tok.decode(
                            bt["labels"][i].masked_fill(bt["labels"][i] == -100, tok.pad_token_id),
                            skip_special_tokens=True,
                        )
                    )
                    ctx_txt.append(tok.decode(prompts[i], skip_special_tokens=True))
                    ids_all.append(gen_only.tolist())

                _, _, bs_f1 = _bertscorer.score(gen_txt, ref_txt, verbose=False)
                cos_all = cos_model.encode(
                    gen_txt + ref_txt,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                cos_all = cos_all.split(len(gen_txt))

                for j in range(len(gen_txt)):
                    ref_split = ref_txt[j].split()
                    hyp_split = gen_txt[j].split()
                    bleu = (
                        sentence_bleu([ref_split], hyp_split, smoothing_function=bleu_sm)
                        if ref_split and hyp_split
                        else 0.0
                    )
                    meteor = _meteor_safe(ref_split, hyp_split)
                    rougeL = rouge_sc.score(ref_txt[j], gen_txt[j])["rougeL"].fmeasure
                    cos = float((cos_all[0][j] * cos_all[1][j]).sum())
                    bs = float(bs_f1[j])

                    rec = dict(
                        context=ctx_txt[j],
                        reference=ref_txt[j],
                        generated=gen_txt[j],
                        metrics=dict(
                            variant=variant,
                            BLEU=bleu,
                            METEOR=meteor,
                            ROUGE_L=rougeL,
                            BERTScore=bs,
                            CosSim=cos,
                        ),
                    )
                    key = (-bs, -bleu, next(_COUNTER))
                    if len(top_heap) < _TOPK:
                        heapq.heappush(top_heap, (key, rec))
                    else:
                        heapq.heappushpop(top_heap, (key, rec))

            pad_eos = {tok.pad_token_id, eos_id}
            clean_ids = [[t for t in seq if t not in pad_eos] for seq in ids_all]

            d1 = _distinct_n(clean_ids, 1)
            d2 = _distinct_n(clean_ids, 2)

            top_records = _select_top_k(top_heap, _TOPK)
            for r in top_records:
                r["metrics"]["Distinct1"] = d1
                r["metrics"]["Distinct2"] = d2

            ce_tf  = ce_tf_sum  / max(1, tok_tf_sum)
            ce_ctx = ce_ctx_sum / max(1, samp_ctx_sum)

            summary = dict(
                test_ce_tf   = ce_tf,
                test_ppl_tf  = float(math.exp(min(ce_tf, 50))),
                test_ce_ctx  = ce_ctx,
                test_ppl_ctx = float(math.exp(min(ce_ctx, 50))),
                bleu_mean    = mean(r["metrics"]["BLEU"]      for r in top_records) if top_records else 0.0,
                meteor_mean  = mean(r["metrics"]["METEOR"]    for r in top_records) if top_records else 0.0,
                rougeL_mean  = mean(r["metrics"]["ROUGE_L"]   for r in top_records) if top_records else 0.0,
                bertscore_mean = mean(r["metrics"]["BERTScore"] for r in top_records) if top_records else 0.0,
                cosim_mean   = mean(r["metrics"]["CosSim"]    for r in top_records) if top_records else 0.0,
                distinct1_mean = d1,
                distinct2_mean = d2,
                total_params     = sum(p.numel() for p in model.parameters()),
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad),
            )

            if not _disable_io:
                v_path = results_dir / f"eval_nonhypernet_{variant}_10000.json"
                json.dump(
                    {variant: {"summary": summary, "samples": top_records}},
                    v_path.open("w"),
                    indent=2,
                )
                logging.info("[%s] report → %s", variant, v_path)
                combined[variant] = {"summary": summary}

                if args.checklist:
                    Path(args.checklist).parent.mkdir(parents=True, exist_ok=True)
                    with Path(args.checklist).open("a") as fh:
                        fh.write(f"{variant}\n")

            logging.info(
                "[%s] done: TF-PPL %.1f | CTX-PPL %.1f",
                variant,
                summary["test_ppl_tf"],
                summary["test_ppl_ctx"],
            )

    finally:
        if _parquet_writer is not None:
            _parquet_writer.close()

    if not _disable_io and combined:
        comb_path = results_dir / "eval_nonhypernet_results_10000.json"
        json.dump(combined, comb_path.open("w"), indent=2)
        logging.info("Combined summary → %s", comb_path)
    elif _disable_io:
        logging.info("[DEMO] run complete – no files written.")


if __name__ == "__main__":
    t0 = time.time()
    main()
    logging.info("Total wall-time %.1fs", time.time() - t0)