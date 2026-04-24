Okay#!/usr/bin/env python3
"""
get_linear_decodability.py
--------------------------
Re-extracts phrase representations from OPT BabyLM checkpoints and computes
linear decodability for each attested binomial: how accurately a logistic
regression classifier (10-fold stratified CV) can decode the ordering (AB vs
BA) from the layer representations.

Joinable to binomial_representations.csv on (model, checkpoint, phrase_AB, layer).

Output: Data/linear_decodability.csv

Usage:
    python Scripts/get_linear_decodability.py --layers last --n-checkpoints 1
"""

import argparse
import csv
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import torch
from transformers import AutoModel

# ---------------------------------------------------------------------------
# Reuse utilities from the main analysis script
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from binomial_rep_analysis import (
    BINOMS_CSV, MODEL_CONFIGS, BINOMIAL_CHUNK_SIZE, DEFAULT_BATCH_SIZE,
    N_LOG_CHECKPOINTS, PROJECT_ROOT,
    load_binomials, collect_sentences, extract_representations,
    get_model_checkpoints, log_sample_checkpoints, _load_tokenizer,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
OUT_CSV = str(Path(PROJECT_ROOT) / "Data" / "linear_decodability.csv")

FIELDNAMES = [
    "model", "model_size", "checkpoint", "step", "tokens",
    "phrase_AB", "layer", "label", "logit",
]

N_FOLDS     = 10    # stratified k-fold splits
N_PCA_COMPS = 100   # PCA components before logistic regression


# ---------------------------------------------------------------------------
# RESUME LOGIC
# ---------------------------------------------------------------------------
def load_completed(path: str) -> set:
    """Returns set of (model, checkpoint, phrase_AB, layer) already written."""
    completed: set = set()
    if not Path(path).exists():
        return completed
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            completed.add((row["model"], row["checkpoint"],
                           row["phrase_AB"], int(row["layer"])))
    return completed


def open_output(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = Path(path).exists()
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
    if not exists:
        w.writeheader()
    return f, w


# ---------------------------------------------------------------------------
# DECODABILITY
# ---------------------------------------------------------------------------
def get_fold_predictions(A: np.ndarray, B: np.ndarray):
    """
    10-fold stratified CV for one (binomial, layer).
    Returns (logits, labels) — one entry per held-out test example across all
    folds — so the caller can write one row per sentence to the output CSV.

    A: (n_AB, D) AB representations  (label 0)
    B: (n_BA, D) BA representations  (label 1)

    PCA is fitted inside each training fold to prevent leakage.
    Returns (None, None) if too few samples.
    """
    n = min(len(A), len(B))
    if n < N_FOLDS * 2:
        return None, None

    X = np.concatenate([A[:n], B[:n]], axis=0)       # (2n, D)
    y = np.array([0] * n + [1] * n, dtype=np.int32)  # 0=AB, 1=BA

    pipe = Pipeline([
        ("pca", PCA(n_components=min(N_PCA_COMPS, n - 1))),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
    ])
    cv     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    logits = cross_val_predict(pipe, X, y, cv=cv, method="decision_function")
    return logits, y


# ---------------------------------------------------------------------------
# PER-CHECKPOINT PROCESSING
# ---------------------------------------------------------------------------
def process_checkpoint(
    model_name, config, ckpt, tokenizer,
    binoms_df, phrase_sentence_map,
    completed, writer, out_file,
    device, layers_filter,
):
    size_label = config["size_label"]
    print(f"\n  Checkpoint: {ckpt['checkpoint']}  "
          f"(step={ckpt['step']}, tokens={ckpt['tokens']:,})")

    tmp_cache = tempfile.mkdtemp(prefix="hf_dec_")
    model = None
    try:
        load_kw = dict(low_cpu_mem_usage=True, cache_dir=tmp_cache)
        if ckpt["tag"]:
            load_kw["revision"] = ckpt["tag"]
        model = AutoModel.from_pretrained(model_name, **load_kw).to(device).eval()

        n_chunks = math.ceil(len(binoms_df) / BINOMIAL_CHUNK_SIZE)
        for chunk_idx in range(n_chunks):
            chunk_df = binoms_df.iloc[
                chunk_idx * BINOMIAL_CHUNK_SIZE:(chunk_idx + 1) * BINOMIAL_CHUNK_SIZE
            ]
            chunk_map = {
                phrase: phrase_sentence_map[phrase]
                for _, row in chunk_df.iterrows()
                for phrase in (row["phrase_AB"], row["phrase_BA"])
            }

            print(f"  Extracting chunk {chunk_idx + 1}/{n_chunks} ...")
            chunk_reps = extract_representations(
                model, tokenizer, chunk_map, device,
                batch_size=config.get("batch_size", DEFAULT_BATCH_SIZE),
            )

            # Determine which layers to score
            all_layers = sorted({
                layer
                for _, row in chunk_df.iterrows()
                for layer in (
                    set(chunk_reps.get(row["phrase_AB"], {})) &
                    set(chunk_reps.get(row["phrase_BA"], {}))
                )
            })
            if layers_filter == "last":
                all_layers = [max(all_layers)] if all_layers else []
            elif layers_filter is not None:
                keep = {int(x) for x in layers_filter.split(",")}
                all_layers = [l for l in all_layers if l in keep]

            for layer_idx in tqdm(all_layers, desc="  Layers"):
                for _, row in chunk_df.iterrows():
                    ab, ba = row["phrase_AB"], row["phrase_BA"]
                    if (model_name, ckpt["checkpoint"], ab, layer_idx) in completed:
                        continue
                    reps_ab = chunk_reps.get(ab, {}).get(layer_idx)
                    reps_ba = chunk_reps.get(ba, {}).get(layer_idx)
                    if reps_ab is None or reps_ba is None:
                        continue

                    logits, labels = get_fold_predictions(reps_ab, reps_ba)
                    if logits is None:
                        continue
                    completed.add((model_name, ckpt["checkpoint"], ab, layer_idx))
                    for logit, label in zip(logits, labels):
                        writer.writerow({
                            "model":      model_name,
                            "model_size": size_label,
                            "checkpoint": ckpt["checkpoint"],
                            "step":       ckpt["step"],
                            "tokens":     ckpt["tokens"],
                            "phrase_AB":  ab,
                            "layer":      layer_idx,
                            "label":      int(label),
                            "logit":      float(logit),
                        })

                out_file.flush()

            del chunk_reps

        print(f"  ✅ Done.")

    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()
        shutil.rmtree(tmp_cache, ignore_errors=True)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def _checkpoints_from_csv(csv_path: str) -> dict:
    """
    Read unique (model, checkpoint, step, tokens) combinations from an existing
    results CSV (e.g. binomial_representations.csv) and return them as a dict:
        { model_name: [{"checkpoint": ..., "tag": ..., "step": ..., "tokens": ...}, ...] }
    This guarantees the decodability run uses exactly the same checkpoints as
    the Procrustes run, regardless of any new checkpoints added to HuggingFace.
    """
    import pandas as pd
    df = pd.read_csv(csv_path, usecols=["model", "checkpoint", "step", "tokens"])
    result = {}
    for model_name, grp in df.drop_duplicates().groupby("model"):
        result[model_name] = [
            {"checkpoint": row["checkpoint"],
             "tag":        row["checkpoint"],
             "step":       int(row["step"]),
             "tokens":     int(row["tokens"])}
            for _, row in grp.iterrows()
        ]
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compute linear decodability of binomial ordering from LLM representations."
    )
    parser.add_argument(
        "--n-checkpoints", type=int, default=N_LOG_CHECKPOINTS,
        help="Number of log-sampled checkpoints (1 = final only, default=%(default)s). "
             "Ignored when --checkpoints-from is set.",
    )
    parser.add_argument(
        "--layers", type=str, default="last",
        help="Layers to score: 'all', 'last' (default), or comma-separated indices.",
    )
    parser.add_argument(
        "--checkpoints-from", type=str, default=None,
        metavar="CSV",
        help="Read checkpoints from an existing results CSV instead of re-sampling "
             "(e.g. results/binomial_representations.csv). Guarantees the same "
             "checkpoint set as the Procrustes run.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    layers_filter = None if args.layers == "all" else args.layers

    print(f"\nLoading binomials and sentence pool ...")
    binoms_df           = load_binomials(BINOMS_CSV)
    phrase_sentence_map = collect_sentences(binoms_df)

    completed = load_completed(OUT_CSV)
    print(f"  {len(completed)} entries already complete.")

    out_file, writer = open_output(OUT_CSV)

    # Build per-model checkpoint list.
    if args.checkpoints_from:
        print(f"\nReading checkpoints from {args.checkpoints_from} ...")
        ckpts_by_model = _checkpoints_from_csv(args.checkpoints_from)
    else:
        ckpts_by_model = None   # will sample below

    try:
        for model_name, config in MODEL_CONFIGS.items():
            print(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")
            tokenizer = _load_tokenizer(config)

            if ckpts_by_model is not None:
                checkpoints = ckpts_by_model.get(model_name, [])
                print(f"  Using {len(checkpoints)} checkpoints from results CSV.")
            else:
                checkpoints = get_model_checkpoints(model_name, config["tokens_per_step"])
                checkpoints = log_sample_checkpoints(checkpoints, args.n_checkpoints)

            for ckpt in checkpoints:
                process_checkpoint(
                    model_name, config, ckpt, tokenizer,
                    binoms_df, phrase_sentence_map,
                    completed, writer, out_file,
                    device, layers_filter,
                )
    finally:
        out_file.close()

    print(f"\nSaved → {OUT_CSV}")
    if Path(OUT_CSV).exists():
        import pandas as pd
        df = pd.read_csv(OUT_CSV)
        n_binoms = df.groupby(["model", "checkpoint", "phrase_AB", "layer"]).ngroups
        log_loss = np.log1p(np.exp(np.where(df["label"] == 1, -df["logit"], df["logit"])))
        print(f"  {len(df):,} rows  |  {n_binoms:,} binomial×checkpoint×layer combinations")
        print(f"  Mean log-loss: {log_loss.mean():.3f}  (chance = {np.log(2):.3f}"
              f", range: {log_loss.min():.3f} – {log_loss.max():.3f})")


if __name__ == "__main__":
    main()
