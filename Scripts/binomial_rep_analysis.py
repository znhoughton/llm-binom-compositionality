#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
binomial_rep_analysis.py
------------------------
Analyse representations of binomial expressions across:
  - Layers (all hidden layers, processed sequentially)
  - Checkpoints (log-sampled, up to 20 per model)
  - Model sizes (125m, 350m, 1.3b OPT BabyLM variants)

Input CSV columns used:
  Word1, Word2       — the two words (Word1 is alphabetically first)
  Alpha              — "Word1 and Word2" phrase string
  Nonalpha           — "Word2 and Word1" phrase string
  OverallFreq        — training corpus frequency (used as-is)
  RelFreq            — relative frequency of Alpha ordering
  Attested           — filter: keep only rows where Attested == 1

For each attested binomial, for each layer × checkpoint × model, computes:
  - self_sim_AB:      mean off-diagonal of centred kernel K=AA^T (Alpha ordering)
  - self_sim_BA:      same for Nonalpha ordering
  - self_sim_ratio:   self_sim_AB / self_sim_BA
  - procrustes_dist:  normalised residual of orthogonal Procrustes(A → B)

Output: Scripts/../results/binomial_representations.csv
Plots:  Scripts/../Plots/
"""

import os
import csv
import math
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoModel

# ---------------------------------------------------------------------------
# PATHS  (script lives in Scripts/, data in Data/, outputs in results/+Plots/)
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent          # Scripts/
PROJECT_ROOT = SCRIPT_DIR.parent                        # project root

BINOMS_CSV        = str(PROJECT_ROOT / "Data" / "nonce_and_attested_binoms.csv")
OUT_DIR           = str(PROJECT_ROOT / "results")
OUT_CSV           = str(PROJECT_ROOT / "results" / "binomial_representations.csv")
SENTENCE_POOL_CSV = str(PROJECT_ROOT / "results" / "sentence_pool.csv")
PLOTS_DIR         = str(PROJECT_ROOT / "Plots")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

MAX_SENTENCES_PER_ORDERING = 500
MIN_SENTENCES_SOFT_WARN    = 500    # print warning if below this
MIN_SENTENCES_HARD         = 0      # set > 0 to hard-skip low-count binomials

CLAUDE_MODEL     = "claude-haiku-4-5-20251001"
MAX_WORKERS      = 5
MAX_RETRIES      = 5
RETRY_WAIT       = 10
REQUEST_BUFFER   = 20
MAX_PER_REQUEST  = 80

DEFAULT_BATCH_SIZE  = 512
N_LOG_CHECKPOINTS   = 20
BINOMIAL_CHUNK_SIZE = 594   # process all binomials in one chunk on A100

# Actual padded seq_len is driven by the batch's longest sentence (~40–60 tokens
# for these short phrases), not max_length=512, so memory is far lower than the
# worst-case estimate.  The OOM fallback will halve automatically if a batch of
# unusually long sentences causes an issue.
MODEL_CONFIGS = {
    "znhoughton/opt-babylm-125m-20eps-seed964": {
        "tokens_per_step": 1_638_400,
        "tokenizer":       "znhoughton/opt-babylm-125m-20eps-seed964",
        "size_label":      "125m",
        "batch_size":      4096,
    },
    "znhoughton/opt-babylm-350m-20eps-seed964": {
        "tokens_per_step": 819_200,
        "tokenizer":       "znhoughton/opt-babylm-350m-20eps-seed964",
        "size_label":      "350m",
        "batch_size":      4096,
    },
    "znhoughton/opt-babylm-1.3b-20eps-seed964": {
        "tokens_per_step": 1_024_000,
        "tokenizer":       "znhoughton/opt-babylm-1.3b-20eps-seed964",
        "size_label":      "1.3b",
        "batch_size":      2048,  # start high; OOM fallback will halve if needed
    },
}

# ---------------------------------------------------------------------------
# TOKEN UTILITIES
# ---------------------------------------------------------------------------

def find_phrase_span_in_tokens(
    phrase: str,
    sentence: str,
    offset_mapping: List[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    """
    Find the token span (start, end inclusive) of `phrase` in `sentence`
    using the tokenizer's offset mapping. O(n) scan.
    """
    char_start = sentence.lower().find(phrase.lower())
    if char_start == -1:
        return None
    char_end = char_start + len(phrase)

    tok_start, tok_end = None, None
    for i, (cs, ce) in enumerate(offset_mapping):
        if cs == ce:  # special token
            continue
        if tok_start is None and cs >= char_start:
            tok_start = i
        if tok_start is not None and ce <= char_end:
            tok_end = i
        if cs >= char_end:
            break

    if tok_start is None or tok_end is None:
        return None
    return tok_start, tok_end

# ---------------------------------------------------------------------------
# CHECKPOINT UTILITIES
# ---------------------------------------------------------------------------

def get_model_checkpoints(repo_id: str, tokens_per_step: int) -> List[Dict]:
    api  = HfApi()
    refs = api.list_repo_refs(repo_id)
    checkpoints = []
    for tag in refs.tags:
        if not tag.name.startswith("step-"):
            continue
        try:
            step = int(tag.name.split("-")[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append({
            "checkpoint": tag.name,
            "tag":        tag.name,
            "step":       step,
            "tokens":     step * tokens_per_step,
        })
    checkpoints.sort(key=lambda x: x["step"])
    if checkpoints:
        print(f"  Found {len(checkpoints)} checkpoints "
              f"(steps {checkpoints[0]['step']} → {checkpoints[-1]['step']})")
    else:
        print(f"  ⚠️  No step-* tags found for {repo_id}")
    return checkpoints


def log_sample_checkpoints(checkpoints: List[Dict], n: int = 20) -> List[Dict]:
    total = len(checkpoints)
    if total <= n:
        return checkpoints
    indices = sorted(set(
        min(int(round(math.exp(x))) - 1, total - 1)
        for x in np.linspace(math.log(1), math.log(total), n)
    ))
    sampled = [checkpoints[i] for i in indices]
    print(f"  Log-sampled {len(sampled)}/{total} checkpoints")
    return sampled

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_binomials(csv_path: str) -> pd.DataFrame:
    """
    Load CSV and return only attested binomials (Attested == 1).
    Normalises phrase strings to lowercase for matching.
    """
    df = pd.read_csv(csv_path)
    df = df[df["Attested"] == 1].copy()
    df = df.reset_index(drop=True)

    # Normalise
    df["Word1"]    = df["Word1"].str.strip().str.lower()
    df["Word2"]    = df["Word2"].str.strip().str.lower()
    # Alpha / Nonalpha are the canonical phrase strings from the CSV;
    # we rebuild them from Word1/Word2 to ensure case consistency.
    df["phrase_AB"] = df["Word1"] + " and " + df["Word2"]
    df["phrase_BA"] = df["Word2"] + " and " + df["Word1"]

    print(f"  Loaded {len(df)} attested binomials from {csv_path}")
    return df

# ---------------------------------------------------------------------------
# SENTENCE GENERATION (Claude API)
# ---------------------------------------------------------------------------

def _load_sentence_pool(path: str) -> Dict[str, List[str]]:
    pool: Dict[str, List[str]] = {}
    if not Path(path).exists():
        return pool
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sents = pool.setdefault(row["phrase"], [])
            if row["sentence"] not in sents:
                sents.append(row["sentence"])
    return pool


def _save_sentence_pool(pool: Dict[str, List[str]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["phrase", "sentence"])
        writer.writeheader()
        for phrase, sents in pool.items():
            for sent in sents:
                writer.writerow({"phrase": phrase, "sentence": sent})


def _generate_for_pair(
    phrase_ab: str,
    phrase_ba: str,
    n: int,
    client,
) -> Tuple[List[str], List[str]]:
    n_request = min(n + REQUEST_BUFFER, MAX_PER_REQUEST)
    prompt = (
        f'Write exactly {n_request} natural English sentences that each contain '
        f'the exact phrase "{phrase_ab}", then exactly {n_request} sentences that '
        f'each contain "{phrase_ba}".\n\n'
        f'Separate the two groups with a line containing only "---".\n'
        f'Output one sentence per line. No numbering, bullets, or labels. '
        f'Each sentence should be 10-30 words and use the phrase naturally in '
        f'varied contexts.'
    )
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_WAIT * (attempt + 1)
                tqdm.write(f"    [retry {attempt+1}/{MAX_RETRIES} in {wait}s] {e}")
                time.sleep(wait)
            else:
                raise

    text  = resp.content[0].text.strip()
    parts = text.split("---", 1)

    def parse_block(block: str, phrase: str) -> List[str]:
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        return [l for l in lines if phrase.lower() in l.lower()]

    sents_ab = parse_block(parts[0], phrase_ab)[:n]
    sents_ba = parse_block(parts[1] if len(parts) > 1 else "", phrase_ba)[:n]
    return sents_ab, sents_ba


def collect_sentences(
    binoms_df: pd.DataFrame,
    max_per_ordering: int = MAX_SENTENCES_PER_ORDERING,
) -> Dict[str, List[str]]:
    """
    Generate up to `max_per_ordering` sentences for every (AB, BA) ordering
    using the Claude API. Resume-safe via SENTENCE_POOL_CSV.
    Returns {phrase_string: [sentence, ...]}
    """
    pool = _load_sentence_pool(SENTENCE_POOL_CSV)

    pairs = [(row["phrase_AB"], row["phrase_BA"])
             for _, row in binoms_df.iterrows()]
    remaining = [
        (ab, ba) for ab, ba in pairs
        if len(pool.get(ab, [])) < max_per_ordering
        or len(pool.get(ba, [])) < max_per_ordering
    ]

    print(f"\nGenerating sentences for {len(binoms_df)} binomials via Claude API ...")
    print(f"  Already complete: {len(pairs) - len(remaining)}  |  "
          f"Remaining: {len(remaining)}")

    if remaining:
        from anthropic import Anthropic
        client    = Anthropic()
        pool_lock = threading.Lock()

        def process_pair(phrase_ab, phrase_ba):
            with pool_lock:
                sents_ab = list(pool.get(phrase_ab, []))
                sents_ba = list(pool.get(phrase_ba, []))

            while len(sents_ab) < max_per_ordering or len(sents_ba) < max_per_ordering:
                needed = max(max_per_ordering - len(sents_ab),
                             max_per_ordering - len(sents_ba))
                chunk  = min(needed + REQUEST_BUFFER, MAX_PER_REQUEST)
                try:
                    new_ab, new_ba = _generate_for_pair(
                        phrase_ab, phrase_ba, chunk, client)
                except Exception as e:
                    tqdm.write(f"  [ERROR] {phrase_ab}: {e}")
                    break
                sents_ab = list(dict.fromkeys(sents_ab + new_ab))[:max_per_ordering]
                sents_ba = list(dict.fromkeys(sents_ba + new_ba))[:max_per_ordering]

            with pool_lock:
                pool[phrase_ab] = sents_ab
                pool[phrase_ba] = sents_ba
                _save_sentence_pool(pool, SENTENCE_POOL_CSV)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_pair, ab, ba): (ab, ba)
                       for ab, ba in remaining}
            with tqdm(total=len(remaining), desc="Generating") as pbar:
                for fut in as_completed(futures):
                    fut.result()
                    pbar.update(1)

        print(f"  Saved sentence pool → {SENTENCE_POOL_CSV}")

    below = [(p, len(pool.get(p, [])))
             for ab, ba in pairs
             for p in (ab, ba)
             if len(pool.get(p, [])) < MIN_SENTENCES_SOFT_WARN]
    if below:
        print(f"\n⚠️  Orderings with fewer than {MIN_SENTENCES_SOFT_WARN} sentences:")
        for phrase, n in sorted(below, key=lambda x: x[1]):
            print(f"    {phrase!r:45s}: {n}")
    else:
        print(f"  All orderings reached {MAX_SENTENCES_PER_ORDERING} sentences.")

    return {p: pool.get(p, []) for ab, ba in pairs for p in (ab, ba)}

# ---------------------------------------------------------------------------
# REPRESENTATION EXTRACTION
# ---------------------------------------------------------------------------

@torch.inference_mode()
def extract_representations(
    model,
    tokenizer,
    phrase_sentence_map: Dict[str, List[str]],
    device: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Forward-pass all sentences in shared batches.  For each batch:
      1. Compute token-span masks on CPU (from offset mappings).
      2. Run one vectorised masked-mean on GPU per layer.
      3. Transfer all valid pooled vectors in a single .cpu() call per layer.

    This replaces O(B * n_layers) small GPU→CPU transfers with O(n_layers).

    Returns:
        { phrase: { layer_idx: np.ndarray (n_sentences, hidden_dim) } }
    """
    pairs: List[Tuple[str, str]] = [
        (phrase, sentence)
        for phrase, sentences in phrase_sentence_map.items()
        for sentence in sentences
    ]

    layer_accum: Dict[str, Dict[int, List[np.ndarray]]] = {
        p: {} for p in phrase_sentence_map
    }
    current_bs = batch_size
    i = 0

    with tqdm(total=len(pairs), desc="  Sentences", leave=True) as pbar:
        while i < len(pairs):
            batch_pairs = pairs[i:i + current_bs]
            sentences_in_batch = [s for _, s in batch_pairs]
            try:
                enc = tokenizer(
                    sentences_in_batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                )
                offset_mappings = enc.pop("offset_mapping").tolist()
                seq_len = enc["input_ids"].shape[1]
                enc = enc.to(device)

                outputs       = model(**enc, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # tuple of (B, T, D)

                # Build span mask (B, T) on CPU; track which sentences are valid.
                B = len(batch_pairs)
                span_mask = torch.zeros(B, seq_len, dtype=torch.bool)
                valid_map: List[Tuple[int, str]] = []
                for b_idx, (phrase, sentence) in enumerate(batch_pairs):
                    span = find_phrase_span_in_tokens(
                        phrase, sentence, offset_mappings[b_idx]
                    )
                    if span is not None:
                        span_mask[b_idx, span[0]:span[1] + 1] = True
                        valid_map.append((b_idx, phrase))

                if valid_map:
                    span_mask_dev = span_mask.to(device)
                    valid_bidx = [b for b, _ in valid_map]

                    for layer_idx, layer_h in enumerate(hidden_states):
                        # Vectorised masked mean — stays on GPU in model dtype.
                        dtype = layer_h.dtype
                        span_f = span_mask_dev.to(dtype).unsqueeze(-1)  # (B, T, 1)
                        counts = span_f.squeeze(-1).sum(dim=1, keepdim=True).clamp(min=1.0)
                        pooled = (layer_h * span_f).sum(dim=1) / counts  # (B, D)

                        # ONE transfer per layer: select valid rows, cast to fp32.
                        pooled_np = pooled[valid_bidx].float().cpu().numpy()
                        for vi, (_, phrase) in enumerate(valid_map):
                            layer_accum[phrase].setdefault(layer_idx, []).append(pooled_np[vi])

                    del span_mask_dev

                pbar.update(len(batch_pairs))
                i += current_bs

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    current_bs = max(1, current_bs // 2)
                    warnings.warn(f"OOM — reducing batch size to {current_bs}")
                else:
                    raise

    results = {}
    for phrase, accum in tqdm(layer_accum.items(), desc="  Stacking", leave=True):
        results[phrase] = {layer_idx: np.stack(vecs, axis=0)
                           for layer_idx, vecs in accum.items()}
    return results

# ---------------------------------------------------------------------------
# SCORES
# ---------------------------------------------------------------------------

def _batch_self_similarity(X: torch.Tensor) -> torch.Tensor:
    """
    Batched self-similarity: X (N, n, D) → (N,) scores.

    WHAT IT MEASURES
    ----------------
    Self-similarity asks: across n different sentences that all contain the
    same phrase, does the model produce consistent representations?  High
    self-similarity means the representation barely changes with context;
    low (more negative) means it varies a lot.

    NAIVE APPROACH — O(n²D)
    -----------------------
    Build the (n, n) Gram matrix  K = X Xᵀ,  then centre it:
        Kc = K - col_mean - row_mean + grand_mean   (double-centering)
    and return the mean of the off-diagonal elements of Kc.

    SHORTCUT — O(nD), no Gram matrix
    ---------------------------------
    Two facts about the centred kernel Kc:

    Fact 1 — sum of ALL elements of Kc is zero.
        Double-centering subtracts every row mean and every column mean, so
        the total always cancels to 0.

    Fact 2 — the diagonal of Kc equals the squared distance to the mean:
        Kc[i,i] = ||x_i - μ||²   where μ = (1/n) Σ x_i

        Proof:  Kc[i,i] = K[i,i] - 2·(row mean of row i) + grand mean
                         = xᵢᵀxᵢ - (2/n)·xᵢᵀ(Σxⱼ) + (1/n²)·||Σxⱼ||²
                         = ||xᵢ||² - 2xᵢᵀμ + ||μ||²  =  ||xᵢ - μ||²

    Combining Facts 1 and 2:
        sum(off-diagonal) = -sum(diagonal) = -Σᵢ ||xᵢ - μ||² = -||Xc||_F²

        mean(off-diagonal) = -||Xc||_F² / (n(n-1))

    where Xc = X - μ is the mean-centred data matrix.  This only requires
    computing centred vectors and their norms — no n×n matrix ever formed.
    """
    _, n, _ = X.shape
    mu    = X.mean(dim=1, keepdim=True)          # (N, 1, D)  — per-binomial mean
    X_c   = X - mu                               # (N, n, D)  — mean-centred reps
    return -X_c.pow(2).sum(dim=(1, 2)) / (n * (n - 1))  # (N,)


def compute_scores_batched(
    chunk_reps: Dict[str, Dict[int, np.ndarray]],
    pairs: List[Tuple[str, str]],
    device: str,
) -> Dict[str, List[Dict]]:
    """
    Compute self-similarity and Procrustes scores for all binomial pairs,
    batched across all N binomials simultaneously for each layer.

    BATCHING STRATEGY
    -----------------
    The naive approach loops over each binomial and calls a per-binomial
    scoring function.  For N=594 binomials × 25 layers that's ~15,000
    small GPU operations — each with Python and kernel-launch overhead,
    and none saturating the GPU.

    Instead we process all N binomials at once per layer, stacking their
    representations into (N, n, D) tensors and using batched operations
    (torch.bmm, torch.linalg.eigh, torch.linalg.svd).  This issues
    ~25 GPU calls total (one set per layer) and keeps the GPU fully loaded.

    PROCRUSTES RESIDUAL SHORTCUT
    ----------------------------
    Orthogonal Procrustes finds the rotation R that best aligns A onto B:
        R* = argmin_{RᵀR=I}  ||AR - B||_F

    The solution is R* = UVᵀ  from SVD(AᵀB) = USVᵀ.

    The residual at R* can be derived without ever constructing R*:

        ||AR* - B||_F²
          = ||AR*||_F² - 2·tr(R*ᵀAᵀB) + ||B||_F²
          = ||A||_F²   - 2·tr(R*ᵀAᵀB) + ||B||_F²    (rotation preserves norm)

    At R* = UVᵀ:
        tr(R*ᵀAᵀB) = tr(VUᵀ · USVᵀ) = tr(VSVᵀ) = tr(S) = Σᵢ σᵢ(AᵀB)

    So:  ||AR* - B||_F² = ||A||_F² + ||B||_F² - 2·Σᵢ σᵢ(AᵀB)

    We only need the singular values of AᵀB — no need to form R* or
    compute AR* - B.

    THIN-FACTORISATION SHORTCUT (avoids (N, D, D) matrix)
    ------------------------------------------------------
    A has shape (N, n, D) with n=500 « D=2048.  Computing AᵀB directly
    gives a (N, D, D) tensor (≈10 GB for 1.3B) and requires SVD of
    2048×2048 matrices — O(D³) per matrix.

    Instead, we factor through A's thin SVD:

        A = U_A · diag(S_A) · V_Aᵀ        (U_A: n×n,  V_A: D×n, V_Aᵀ V_A = I)

    Then:
        AᵀB = V_A · diag(S_A) · U_Aᵀ · B  =  V_A · C
                                                       where C = diag(S_A)(U_AᵀB)

    Key lemma: left-multiplying by V_A (which has orthonormal columns)
    does not change singular values.  Proof via the AB~BA eigenvalue
    property:

        σᵢ²(V_A C) = eigenvalues of (V_A C)(V_A C)ᵀ = V_A C Cᵀ V_Aᵀ
        non-zero eigenvalues of V_A (C Cᵀ V_Aᵀ)
            = non-zero eigenvalues of (C Cᵀ V_Aᵀ) V_A
            = non-zero eigenvalues of C Cᵀ (V_Aᵀ V_A)
            = non-zero eigenvalues of C Cᵀ             (since V_Aᵀ V_A = I)
            = σᵢ²(C)

    So  Σᵢ σᵢ(AᵀB) = Σᵢ σᵢ(C)  where C is (N, n, D) — SVD of n×n matrices
    instead of D×D, giving O(n²D) instead of O(D³) per layer (~17× faster
    for 1.3B where D=2048, n=500).

    U_A and S_A are obtained from eigh(AAᵀ) rather than a full SVD:
        AAᵀ = U_A · diag(S_A²) · U_Aᵀ   (symmetric, so eigh is exact)
    so  S_A = sqrt(eigenvalues)  and  U_A = eigenvectors.
    ||A||_F² = tr(AAᵀ) = Σ eigenvalues, so we reuse L_A for norm_A_sq.

    Returns {phrase_AB: [score_row_per_layer, ...]}.
    """
    valid_pairs = [
        (ab, ba) for ab, ba in pairs
        if chunk_reps.get(ab) and chunk_reps.get(ba)
    ]
    all_layers = sorted({
        layer
        for ab, ba in valid_pairs
        for layer in set(chunk_reps[ab]) & set(chunk_reps[ba])
    })

    scores_by_ab: Dict[str, List[Dict]] = {ab: [] for ab, _ in valid_pairs}

    for layer_idx in tqdm(all_layers, desc="  Layers", position=0):
        layer_pairs = [
            (ab, ba) for ab, ba in valid_pairs
            if layer_idx in chunk_reps[ab] and layer_idx in chunk_reps[ba]
        ]
        if not layer_pairs:
            continue

        steps = tqdm(
            ["stack", "self-sim AB", "self-sim BA", "procrustes eigh",
             "procrustes svd", "collect"],
            desc=f"    layer {layer_idx:>2d}",
            leave=False,
            position=1,
        )

        steps.set_description(f"    layer {layer_idx:>2d}  stack")
        A_arrs = [chunk_reps[ab][layer_idx] for ab, _  in layer_pairs]
        B_arrs = [chunk_reps[ba][layer_idx] for _,  ba in layer_pairs]
        n = min(min(len(a) for a in A_arrs), min(len(b) for b in B_arrs))
        if n < 2:
            tqdm.write(f"  ⚠️  layer {layer_idx}: n={n} < 2, skipping layer (need ≥2 sentences for self-similarity)")
            steps.close()
            continue
        A = torch.from_numpy(np.stack([a[:n] for a in A_arrs])).to(device).float()
        B = torch.from_numpy(np.stack([b[:n] for b in B_arrs])).to(device).float()
        steps.update(1)

        steps.set_description(f"    layer {layer_idx:>2d}  self-sim AB")
        ss_ab = _batch_self_similarity(A)
        if device != "cpu": torch.cuda.synchronize()
        steps.update(1)

        steps.set_description(f"    layer {layer_idx:>2d}  self-sim BA")
        ss_ba = _batch_self_similarity(B)
        if device != "cpu": torch.cuda.synchronize()
        steps.update(1)

        # ---- Procrustes ----
        # Goal: ||AR* - B||_F / ||B||_F  where R* is the optimal rotation.
        #
        # Shortcut 1 — residual without forming R*:
        #   ||AR* - B||_F² = ||A||_F² + ||B||_F² - 2·Σ σᵢ(AᵀB)
        #
        # Shortcut 2 — thin factorisation to avoid (N, D, D):
        #   eigh(AAᵀ) → U_A (left singular vectors), L_A (= S_A²)
        #   C = diag(S_A) · U_Aᵀ B   is (N, n, D)  instead of (N, D, D)
        #   Σ σᵢ(AᵀB) = Σ σᵢ(C)  because V_A (orthonormal cols) doesn't
        #   change singular values.
        #   Also reuse L_A: ||A||_F² = tr(AAᵀ) = Σ eigenvalues.
        steps.set_description(f"    layer {layer_idx:>2d}  procrustes eigh")
        AAT       = torch.bmm(A, A.transpose(1, 2))   # (N, n, n) = A Aᵀ
        L_A, U_A  = torch.linalg.eigh(AAT)            # L_A = S_A², U_A = left singular vecs
        norm_A_sq = L_A.clamp(min=0).sum(dim=1)       # ||A||_F² = Σ eigenvalues
        norm_B_sq = B.pow(2).sum(dim=(1, 2))           # ||B||_F²
        if device != "cpu": torch.cuda.synchronize()
        steps.update(1)

        steps.set_description(f"    layer {layer_idx:>2d}  procrustes svd")
        S_A      = L_A.clamp(min=0).sqrt()                              # (N, n) singular values of A
        C        = S_A.unsqueeze(-1) * torch.bmm(U_A.transpose(1, 2), B)  # (N, n, D): diag(S_A) U_Aᵀ B
        S        = torch.linalg.svd(C, full_matrices=False, driver='gesvd').S  # (N, n): σᵢ(AᵀB) via σᵢ(C)
        resid_sq = (norm_A_sq + norm_B_sq - 2.0 * S.sum(dim=1)).clamp(min=0.0)
        proc     = resid_sq.sqrt() / norm_B_sq.sqrt().clamp(min=1e-10)  # normalised residual
        if device != "cpu": torch.cuda.synchronize()
        steps.update(1)

        steps.set_description(f"    layer {layer_idx:>2d}  collect")
        ss_ab_np = ss_ab.cpu().numpy()
        ss_ba_np = ss_ba.cpu().numpy()
        proc_np  = proc.cpu().numpy()
        steps.update(1)
        steps.close()

        for i, (ab, ba) in enumerate(layer_pairs):
            s_ab  = float(ss_ab_np[i])
            s_ba  = float(ss_ba_np[i])
            ratio = s_ab / s_ba if (math.isfinite(s_ba) and s_ba != 0.0) else float("nan")
            scores_by_ab[ab].append({
                "layer":           layer_idx,
                "n_sentences_AB":  len(A_arrs[i]),
                "n_sentences_BA":  len(B_arrs[i]),
                "self_sim_AB":     s_ab,
                "self_sim_BA":     s_ba,
                "self_sim_ratio":  ratio,
                "procrustes_dist": float(proc_np[i]),
            })

    return scores_by_ab

# ---------------------------------------------------------------------------
# OUTPUT HELPERS
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "model", "model_size", "checkpoint", "step", "tokens",
    "word1", "word2", "phrase_AB", "phrase_BA",
    "overall_freq", "rel_freq",
    "layer",
    "n_sentences_AB", "n_sentences_BA",
    "self_sim_AB", "self_sim_BA", "self_sim_ratio",
    "procrustes_dist",
]


def load_completed(out_csv: str) -> set:
    """
    Returns set of (model, checkpoint, phrase_AB) that have a full layer set.
    Infers the expected layer count from the most common count in the CSV;
    any entry with fewer rows is treated as incomplete and will be re-run.
    """
    if not Path(out_csv).exists():
        return set()

    from collections import Counter
    layer_counts: Dict[tuple, int] = {}
    with open(out_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["model"], row["checkpoint"], row["phrase_AB"])
            layer_counts[key] = layer_counts.get(key, 0) + 1

    if not layer_counts:
        return set()

    expected = Counter(layer_counts.values()).most_common(1)[0][0]
    incomplete = [k for k, v in layer_counts.items() if v < expected]
    if incomplete:
        print(f"  ⚠️  {len(incomplete)} incomplete entries (< {expected} layers) "
              f"will be re-run.")
    return {k for k, v in layer_counts.items() if v == expected}


def open_results_file(out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    exists = Path(out_csv).exists()
    f = open(out_csv, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
    if not exists:
        w.writeheader()
    return f, w

# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def main():
    # ── Load attested binomials ───────────────────────────────────────────────
    print(f"Loading binomials ...")
    binoms_df = load_binomials(BINOMS_CSV)

    # ── Collect sentences via Claude API ─────────────────────────────────────
    sentences_by_phrase = collect_sentences(binoms_df)

    # ── Hard-skip check ───────────────────────────────────────────────────────
    if MIN_SENTENCES_HARD > 0:
        skip_mask = binoms_df.apply(
            lambda r: (
                len(sentences_by_phrase.get(r["phrase_AB"], [])) < MIN_SENTENCES_HARD
                or
                len(sentences_by_phrase.get(r["phrase_BA"], [])) < MIN_SENTENCES_HARD
            ), axis=1
        )
        if skip_mask.any():
            print(f"\n⛔ Hard-skipping {skip_mask.sum()} binomials "
                  f"(below {MIN_SENTENCES_HARD} sentences in at least one ordering)")
        binoms_df = binoms_df[~skip_mask].reset_index(drop=True)

    print(f"\n{len(binoms_df)} binomials proceeding to representation analysis.")

    # ── Build phrase→sentence map for extraction ──────────────────────────────
    phrase_sentence_map: Dict[str, List[str]] = {}
    for _, row in binoms_df.iterrows():
        phrase_sentence_map[row["phrase_AB"]] = sentences_by_phrase[row["phrase_AB"]]
        phrase_sentence_map[row["phrase_BA"]] = sentences_by_phrase[row["phrase_BA"]]

    # ── Open results CSV ──────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    completed = load_completed(OUT_CSV)
    if completed:
        print(f"  Resuming — {len(completed)} (model, checkpoint, binomial) combinations already done.")
    out_file, writer = open_results_file(OUT_CSV)

    try:
        for model_name, config in MODEL_CONFIGS.items():
            size_label = config["size_label"]
            print(f"\n{'='*60}")
            print(f"Model: {model_name}  [{size_label}]")
            print("=" * 60)

            checkpoints = get_model_checkpoints(model_name,
                                                config["tokens_per_step"])
            if not checkpoints:
                print("  No checkpoints found, skipping.")
                continue
            checkpoints = log_sample_checkpoints(checkpoints, n=N_LOG_CHECKPOINTS)

            tokenizer = AutoTokenizer.from_pretrained(
                config["tokenizer"], use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            for ckpt in checkpoints:
                print(f"\n  Checkpoint: {ckpt['checkpoint']}  "
                      f"(step={ckpt['step']}, tokens={ckpt['tokens']:,})")

                tmp_cache = tempfile.mkdtemp(prefix="hf_ckpt_")
                try:
                    device = "cuda" if torch.cuda.is_available() else "cpu"

                    load_kw = dict(
                        low_cpu_mem_usage=True,
                        cache_dir=tmp_cache,
                    )
                    if ckpt["tag"]:
                        load_kw["revision"] = ckpt["tag"]

                    model = (AutoModel
                             .from_pretrained(model_name, **load_kw)
                             .to(device)
                             .eval())

                    n_chunks = math.ceil(len(binoms_df) / BINOMIAL_CHUNK_SIZE)
                    for chunk_idx in range(n_chunks):
                        chunk_df = binoms_df.iloc[
                            chunk_idx * BINOMIAL_CHUNK_SIZE:
                            (chunk_idx + 1) * BINOMIAL_CHUNK_SIZE
                        ]
                        chunk_df = chunk_df[
                            ~chunk_df["phrase_AB"].apply(
                                lambda ab: (model_name, ckpt["checkpoint"], ab) in completed
                            )
                        ]
                        if chunk_df.empty:
                            continue

                        chunk_map = {
                            p: phrase_sentence_map[p]
                            for _, row in chunk_df.iterrows()
                            for p in (row["phrase_AB"], row["phrase_BA"])
                        }

                        print(f"  Extracting chunk {chunk_idx+1}/{n_chunks} ...")
                        chunk_reps = extract_representations(
                            model, tokenizer, chunk_map, device,
                            batch_size=config.get("batch_size", DEFAULT_BATCH_SIZE),
                        )

                        print(f"  Computing scores ...")
                        chunk_pairs = [(r["phrase_AB"], r["phrase_BA"])
                                       for _, r in chunk_df.iterrows()]
                        all_scores = compute_scores_batched(
                            chunk_reps, chunk_pairs, device)

                        for _, row in chunk_df.iterrows():
                            ab, ba = row["phrase_AB"], row["phrase_BA"]
                            scores = all_scores.get(ab)
                            if not scores:
                                print(f"    ⚠️  No reps for ({row['Word1']}, "
                                      f"{row['Word2']}), skipping.")
                                continue
                            completed.add((model_name, ckpt["checkpoint"], ab))
                            for score_row in scores:
                                writer.writerow({
                                    "model":        model_name,
                                    "model_size":   size_label,
                                    "checkpoint":   ckpt["checkpoint"],
                                    "step":         ckpt["step"],
                                    "tokens":       ckpt["tokens"],
                                    "word1":        row["Word1"],
                                    "word2":        row["Word2"],
                                    "phrase_AB":    ab,
                                    "phrase_BA":    ba,
                                    "overall_freq": row.get("OverallFreq", ""),
                                    "rel_freq":     row.get("RelFreq", ""),
                                    **score_row,
                                })

                        del chunk_reps
                        out_file.flush()

                    print(f"  ✅ Done.")

                finally:
                    del model
                    torch.cuda.empty_cache()
                    shutil.rmtree(tmp_cache, ignore_errors=True)

    finally:
        out_file.close()

    print(f"\n🏁 Pipeline complete.  Results → {OUT_CSV}")


if __name__ == "__main__":
    main()
