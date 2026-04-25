#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
binomial_rep_analysis.py
------------------------
Analyse representations of binomial expressions across:
  - Layers (all hidden layers, processed sequentially)
  - Checkpoints (log-sampled, up to 10 per model)
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

import argparse
import json
import os
import csv
import math
import subprocess
import shutil
import sys
import tempfile
import warnings
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


DEFAULT_BATCH_SIZE  = 512
N_LOG_CHECKPOINTS   = 10
BINOMIAL_CHUNK_SIZE = 100   # ~20 GB peak CPU RAM per worker for 1.3b (2×100×500×25×2048×4 bytes)

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
        "job_weight":      1.0,   # relative per-checkpoint cost (used for load balancing)
    },
    "znhoughton/opt-babylm-350m-20eps-seed964": {
        "tokens_per_step": 819_200,
        "tokenizer":       "znhoughton/opt-babylm-350m-20eps-seed964",
        "size_label":      "350m",
        "batch_size":      4096,
        "job_weight":      3.5,
    },
    "znhoughton/opt-babylm-1.3b-20eps-seed964": {
        "tokens_per_step": 409_600,
        "tokenizer":       "znhoughton/opt-babylm-1.3b-20eps-seed964",
        "size_label":      "1.3b",
        "batch_size":      512,   # reduced from 2048; OOM fallback will halve further if needed
        "job_weight":      26.0,
    },
}

# External (non-BabyLM) models — used by get_cosine_similarity.py and
# get_compositional_similarity.py for final-checkpoint-only runs.
# These do not have tokens_per_step or job_weight (no checkpoint traversal).
EXTRA_MODEL_CONFIGS = {
    "allenai/OLMo-2-0425-1B": {
        "size_label":        "olmo-1b",
        "tokenizer":         "allenai/OLMo-2-0425-1B",
        "batch_size":        256,
        "chunk_size":        100,
        "trust_remote_code": True,
        "torch_dtype":       "float16",
        "job_weight":        7.0,
    },
    "allenai/OLMo-2-1124-7B": {
        "size_label":        "olmo-7b",
        "tokenizer":         "allenai/OLMo-2-1124-7B",
        "batch_size":        32,
        "chunk_size":        50,
        "trust_remote_code": True,
        "torch_dtype":       "float16",
        "job_weight":        50.0,
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
        # Use ce > char_start (not cs >= char_start) so that BPE tokens that
        # include a preceding space (cs = char_start - 1) are not skipped.
        if tok_start is None and ce > char_start:
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
    if n == 1:
        return [checkpoints[-1]]
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




def _swap_phrase(sentence: str, phrase_ab: str, phrase_ba: str) -> str:
    """
    Replace phrase_ab with phrase_ba in sentence, preserving the capitalisation
    of the first character (handles sentence-initial occurrences).
    """
    import re
    def _repl(match: re.Match) -> str:
        orig = match.group(0)
        result = phrase_ba
        if orig[0].isupper():
            result = result[0].upper() + result[1:]
        return result
    return re.sub(re.escape(phrase_ab), _repl, sentence, flags=re.IGNORECASE)


def collect_sentences(
    binoms_df: pd.DataFrame,
) -> Dict[str, List[str]]:
    """
    Load AB sentences from the pre-built sentence pool and derive BA sentences
    by swapping the phrase in-place.  BA sentences are never stored in the pool
    — they are always re-derived here so the two orderings are guaranteed to
    share identical sentence contexts.
    Returns {phrase_string: [sentence, ...]} for both AB and BA.
    """
    pool = _load_sentence_pool(SENTENCE_POOL_CSV)
    pairs = [(row["phrase_AB"], row["phrase_BA"])
             for _, row in binoms_df.iterrows()]

    result: Dict[str, List[str]] = {}
    for ab, ba in pairs:
        sents_ab = pool.get(ab, [])
        result[ab] = sents_ab
        result[ba] = [_swap_phrase(s, ab, ba) for s in sents_ab]

    below = [(ab, len(result.get(ab, [])))
             for ab, _ in pairs
             if len(result.get(ab, [])) < MIN_SENTENCES_SOFT_WARN]
    if below:
        print(f"\n  WARNING: orderings with fewer than {MIN_SENTENCES_SOFT_WARN} sentences:")
        for phrase, n in sorted(below, key=lambda x: x[1]):
            print(f"    {phrase!r:45s}: {n}")

    return result

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
            # Prepend a space so sentence-initial words always receive the
            # space-prefix BPE form (e.g. "Ġgums"), matching their mid-sentence
            # tokenisation.  Without this, the first word of a phrase at
            # position 0 may split into fewer/more subwords than the same word
            # appearing mid-sentence, causing AB/BA phrase-token count
            # mismatches for sentences where the phrase starts the sentence.
            sentences_in_batch = [" " + s.lower() for _, s in batch_pairs]
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
                for b_idx, ((phrase, _), norm_sentence) in enumerate(
                    zip(batch_pairs, sentences_in_batch)
                ):
                    span = find_phrase_span_in_tokens(
                        phrase, norm_sentence, offset_mappings[b_idx]
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
        A = A - A.mean(dim=1, keepdim=True)            # centre each cloud (remove mean position)
        B = B - B.mean(dim=1, keepdim=True)
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


def load_completed(csv_paths) -> set:
    """
    Returns set of (model, checkpoint, phrase_AB) that have a full layer set.
    Accepts a single path or a list of paths — all are read and merged so that
    results already in the main CSV and results written to a temp CSV during a
    parallel run are both respected.
    Infers the expected layer count per model separately (different model sizes
    have different depths, e.g. 125m=13 layers, 350m/1.3b=25 layers), so the
    mode is computed within each model group, not globally.
    Any entry with fewer rows than its model's mode is treated as incomplete.
    """
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    from collections import Counter, defaultdict
    layer_counts: Dict[tuple, int] = {}
    for path in csv_paths:
        if not Path(path).exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["model"], row["checkpoint"], row["phrase_AB"])
                layer_counts[key] = layer_counts.get(key, 0) + 1

    if not layer_counts:
        return set()

    # Group by model name so each model's expected layer count is computed
    # independently (models of different sizes have different layer counts).
    model_groups: Dict[str, Dict[tuple, int]] = defaultdict(dict)
    for key, count in layer_counts.items():
        model_groups[key[0]][key] = count

    completed: set = set()
    for model_name, mc in model_groups.items():
        expected = Counter(mc.values()).most_common(1)[0][0]
        incomplete = [k for k, v in mc.items() if v < expected]
        if incomplete:
            print(f"  ⚠️  {len(incomplete)} incomplete entries for {model_name} "
                  f"(< {expected} layers) will be re-run.")
        completed.update(k for k, v in mc.items() if v == expected)
    return completed


def open_results_file(out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    exists = Path(out_csv).exists()
    f = open(out_csv, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
    if not exists:
        w.writeheader()
    return f, w

def merge_temp_csv(tmp_csv: str, out_csv: str):
    """Append all rows from tmp_csv into out_csv, then delete tmp_csv."""
    if not Path(tmp_csv).exists():
        return
    print(f"\n  Merging {Path(tmp_csv).name} → {Path(out_csv).name} ...")
    exists = Path(out_csv).exists()
    with open(tmp_csv, newline="", encoding="utf-8") as f_in, \
         open(out_csv, "a", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in reader:
            writer.writerow(row)
    Path(tmp_csv).unlink()
    print(f"  Done — temp file removed.")

# ---------------------------------------------------------------------------
# PER-CHECKPOINT HELPER
# ---------------------------------------------------------------------------

def _process_checkpoint(
    model_name: str,
    config: Dict,
    ckpt: Dict,
    tokenizer,
    binoms_df: "pd.DataFrame",
    phrase_sentence_map: Dict[str, List[str]],
    completed: set,
    device: str,
    writer,
    out_file,
):
    """Load one checkpoint, extract representations, compute scores, write results."""
    size_label = config["size_label"]
    print(f"\n  Checkpoint: {ckpt['checkpoint']}  "
          f"(step={ckpt['step']}, tokens={ckpt['tokens']:,})")

    # Check completion before downloading the model — skip the download entirely
    # if every binomial for this checkpoint is already done.
    remaining_df = binoms_df[
        ~binoms_df["phrase_AB"].apply(
            lambda ab: (model_name, ckpt["checkpoint"], ab) in completed
        )
    ]
    if remaining_df.empty:
        print(f"  ✅ Already complete — skipping download.")
        return

    use_tmp_cache = bool(ckpt["tag"])
    tmp_cache = tempfile.mkdtemp(prefix="hf_ckpt_") if use_tmp_cache else None
    model = None
    try:
        load_kw = dict(low_cpu_mem_usage=True)
        if ckpt["tag"]:
            load_kw["revision"] = ckpt["tag"]
            load_kw["cache_dir"] = tmp_cache
        if config.get("trust_remote_code"):
            load_kw["trust_remote_code"] = True
        if config.get("torch_dtype") == "float16":
            load_kw["torch_dtype"] = torch.float16
        if config.get("device_map"):
            load_kw["device_map"] = config["device_map"]
        model = AutoModel.from_pretrained(model_name, **load_kw)
        if not config.get("device_map"):
            model = model.to(device)
        model = model.eval()

        chunk_sz = config.get("chunk_size", BINOMIAL_CHUNK_SIZE)
        n_chunks = math.ceil(len(remaining_df) / chunk_sz)
        for chunk_idx in range(n_chunks):
            chunk_df = remaining_df.iloc[
                chunk_idx * chunk_sz:(chunk_idx + 1) * chunk_sz
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
            chunk_pairs = [(r["phrase_AB"], r["phrase_BA"]) for _, r in chunk_df.iterrows()]
            all_scores = compute_scores_batched(chunk_reps, chunk_pairs, device)

            for _, row in chunk_df.iterrows():
                ab, ba = row["phrase_AB"], row["phrase_BA"]
                scores = all_scores.get(ab)
                if not scores:
                    print(f"    ⚠️  No reps for ({row['Word1']}, {row['Word2']}), skipping.")
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
        if model is not None:
            del model
        torch.cuda.empty_cache()
        if tmp_cache:
            shutil.rmtree(tmp_cache, ignore_errors=True)


def _load_tokenizer(config: Dict):
    tok = AutoTokenizer.from_pretrained(
        config["tokenizer"], use_fast=True,
        trust_remote_code=config.get("trust_remote_code", False),
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None, choices=[0, 1],
                        help="Worker GPU index (set by coordinator, not by user).")
    parser.add_argument("--jobs-file", default=None,
                        help="JSON file listing this worker's assigned checkpoints.")
    args = parser.parse_args()

    # ── Multi-GPU coordinator ─────────────────────────────────────────────────
    # If 2+ GPUs are available and this is not already a worker, act as
    # coordinator: collect sentences, build the full job list, assign jobs to
    # GPUs via greedy bin-packing, spawn worker subprocesses, then merge.
    if args.gpu is None and torch.cuda.device_count() >= 2:
        print(f"Detected {torch.cuda.device_count()} GPUs — running with dynamic load balancing.")

        # Merge any leftover temp CSVs from a previous interrupted run so that
        # load_completed sees all previously written results and doesn't re-run
        # checkpoints that are already done.
        os.makedirs(OUT_DIR, exist_ok=True)
        for gpu_id in range(torch.cuda.device_count()):
            tmp = str(Path(OUT_CSV).with_name(
                Path(OUT_CSV).stem + f"_gpu{gpu_id}_tmp.csv"
            ))
            if Path(tmp).exists():
                print(f"  Found leftover temp file from previous run — merging {Path(tmp).name} ...")
                merge_temp_csv(tmp, OUT_CSV)

        # Sentence collection must finish before workers start (avoids
        # concurrent writes to SENTENCE_POOL_CSV).
        print(f"\nLoading binomials ...")
        binoms_df = load_binomials(BINOMS_CSV)
        collect_sentences(binoms_df)

        # Build full job list, skipping checkpoints already fully complete.
        # A checkpoint is complete when every binomial has a full layer set.
        existing_completed = load_completed(OUT_CSV)
        n_binomials = len(binoms_df)
        from collections import Counter
        ckpt_done_counts = Counter(
            (model, ckpt_name)
            for model, ckpt_name, _ in existing_completed
        )

        all_jobs = []
        for model_name, config in MODEL_CONFIGS.items():
            ckpts = get_model_checkpoints(model_name, config["tokens_per_step"])
            if not ckpts:
                continue
            ckpts = log_sample_checkpoints(ckpts, n=N_LOG_CHECKPOINTS)
            for ckpt in ckpts:
                if ckpt_done_counts.get((model_name, ckpt["checkpoint"]), 0) >= n_binomials:
                    continue
                all_jobs.append({
                    "model_name": model_name,
                    "ckpt":       ckpt,
                    "weight":     config["job_weight"],
                })

        for model_name, config in EXTRA_MODEL_CONFIGS.items():
            if ckpt_done_counts.get((model_name, "final"), 0) >= n_binomials:
                continue
            all_jobs.append({
                "model_name": model_name,
                "ckpt":       {"checkpoint": "final", "tag": None, "step": 0, "tokens": 0},
                "weight":     config.get("job_weight", 10.0),
            })

        if not all_jobs:
            print("  All checkpoints already complete — nothing to do.")
            return

        # Greedy bin-packing: assign heaviest jobs first to the least-loaded GPU.
        gpu_jobs: List[List] = [[], []]
        loads = [0.0, 0.0]
        for job in sorted(all_jobs, key=lambda j: j["weight"], reverse=True):
            gpu_id = loads.index(min(loads))
            gpu_jobs[gpu_id].append(job)
            loads[gpu_id] += job["weight"]

        print(f"  GPU 0: {len(gpu_jobs[0])} checkpoints  (estimated load {loads[0]:.0f})")
        print(f"  GPU 1: {len(gpu_jobs[1])} checkpoints  (estimated load {loads[1]:.0f})")

        # Write per-GPU job files and spawn workers.
        os.makedirs(OUT_DIR, exist_ok=True)
        job_files = []
        procs = []
        for gpu_id in range(2):
            jf = str(Path(OUT_DIR) / f"_jobs_gpu{gpu_id}.json")
            with open(jf, "w") as f:
                json.dump(gpu_jobs[gpu_id], f)
            job_files.append(jf)

            p = subprocess.Popen(
                [sys.executable, __file__,
                 "--gpu", str(gpu_id), "--jobs-file", jf],
            )
            procs.append(p)

        exit_codes = [p.wait() for p in procs]
        failed = [gpu_id for gpu_id, code in enumerate(exit_codes) if code != 0]
        if failed:
            print(f"\n⚠️  GPU worker(s) {failed} exited with errors — "
                  f"some checkpoints may be incomplete. Check output above.")

        for jf in job_files:
            Path(jf).unlink(missing_ok=True)

        # Merge temp CSVs sequentially here (not in workers) to avoid
        # concurrent writes to OUT_CSV if both workers finish near-simultaneously.
        for gpu_id in range(2):
            tmp = str(Path(OUT_CSV).with_name(
                Path(OUT_CSV).stem + f"_gpu{gpu_id}_tmp.csv"
            ))
            merge_temp_csv(tmp, OUT_CSV)

        print("\n🏁 Both GPU workers finished.")
        return

    # ── Worker / single-GPU path ──────────────────────────────────────────────
    active_csv = (
        str(Path(OUT_CSV).with_name(Path(OUT_CSV).stem + f"_gpu{args.gpu}_tmp.csv"))
        if args.gpu is not None else OUT_CSV
    )

    print(f"Loading binomials ...")
    binoms_df = load_binomials(BINOMS_CSV)
    sentences_by_phrase = collect_sentences(binoms_df)

    if MIN_SENTENCES_HARD > 0:
        skip_mask = binoms_df.apply(
            lambda r: (
                len(sentences_by_phrase.get(r["phrase_AB"], [])) < MIN_SENTENCES_HARD
                or len(sentences_by_phrase.get(r["phrase_BA"], [])) < MIN_SENTENCES_HARD
            ), axis=1,
        )
        if skip_mask.any():
            print(f"\n⛔ Hard-skipping {skip_mask.sum()} binomials "
                  f"(below {MIN_SENTENCES_HARD} sentences in at least one ordering)")
        binoms_df = binoms_df[~skip_mask].reset_index(drop=True)

    print(f"\n{len(binoms_df)} binomials proceeding to representation analysis.")

    phrase_sentence_map: Dict[str, List[str]] = {}
    for _, row in binoms_df.iterrows():
        phrase_sentence_map[row["phrase_AB"]] = sentences_by_phrase[row["phrase_AB"]]
        phrase_sentence_map[row["phrase_BA"]] = sentences_by_phrase[row["phrase_BA"]]

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_sources = [OUT_CSV] if active_csv == OUT_CSV else [OUT_CSV, active_csv]
    completed = load_completed(csv_sources)
    if completed:
        print(f"  Resuming — {len(completed)} (model, checkpoint, binomial) combinations already done.")
    out_file, writer = open_results_file(active_csv)
    if args.gpu is not None and torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        if args.jobs_file:
            # ── Dynamic worker: process the coordinator-assigned job list ─────
            with open(args.jobs_file) as f:
                my_jobs = json.load(f)
            print(f"  GPU {args.gpu} worker: {len(my_jobs)} checkpoints assigned.")

            # Group by model so the tokenizer is loaded once per model.
            # Sort by weight ascending (smallest model first) so lighter jobs
            # finish quickly before the long heavy-model run begins.
            for model_name, model_jobs in groupby(
                sorted(my_jobs, key=lambda j: (j["weight"], j["model_name"])),
                key=lambda j: j["model_name"],
            ):
                config = MODEL_CONFIGS.get(model_name) or EXTRA_MODEL_CONFIGS[model_name]
                print(f"\n{'='*60}")
                print(f"Model: {model_name}  [{config['size_label']}]  (GPU {args.gpu})")
                print("=" * 60)
                tokenizer = _load_tokenizer(config)
                for job in model_jobs:
                    _process_checkpoint(
                        model_name, config, job["ckpt"], tokenizer,
                        binoms_df, phrase_sentence_map, completed,
                        device, writer, out_file,
                    )
        else:
            # ── Single-GPU: process all models sequentially ───────────────────
            for model_name, config in MODEL_CONFIGS.items():
                print(f"\n{'='*60}")
                print(f"Model: {model_name}  [{config['size_label']}]")
                print("=" * 60)
                checkpoints = get_model_checkpoints(model_name, config["tokens_per_step"])
                if not checkpoints:
                    print("  No checkpoints found, skipping.")
                    continue
                checkpoints = log_sample_checkpoints(checkpoints, n=N_LOG_CHECKPOINTS)
                tokenizer = _load_tokenizer(config)
                for ckpt in checkpoints:
                    _process_checkpoint(
                        model_name, config, ckpt, tokenizer,
                        binoms_df, phrase_sentence_map, completed,
                        device, writer, out_file,
                    )

            # Extra models (OLMo etc.) — final checkpoint only
            for model_name, config in EXTRA_MODEL_CONFIGS.items():
                print(f"\n{'='*60}")
                print(f"Model: {model_name}  [{config['size_label']}]  (final checkpoint only)")
                print("=" * 60)
                tokenizer = _load_tokenizer(config)
                _process_checkpoint(
                    model_name, config,
                    {"checkpoint": "final", "tag": None, "step": 0, "tokens": 0},
                    tokenizer, binoms_df, phrase_sentence_map, completed,
                    device, writer, out_file,
                )
    finally:
        out_file.close()

    # In coordinated mode (--jobs-file set) the coordinator handles merging
    # after all workers exit. Only merge here for standalone --gpu runs.
    if active_csv != OUT_CSV and args.jobs_file is None:
        merge_temp_csv(active_csv, OUT_CSV)

    print(f"\n🏁 Pipeline complete.  Results → {OUT_CSV}")


if __name__ == "__main__":
    main()
