#!/usr/bin/env python3
"""
get_cosine_similarity.py
------------------------
Extracts phrase representations from OPT BabyLM checkpoints and computes
the mean pairwise cosine similarity between matched AB and BA sentence pairs
for each attested binomial.

For each binomial, sentence i in the AB pool and sentence i in the BA pool
are the same sentence with the phrase order swapped — so cosine similarity
directly measures how much swapping the phrase order changes the model's
representation.

  High cosine_sim → model represents the two orderings similarly
  Low cosine_sim  → model represents them very differently

Output: Data/cosine_similarity.csv

Usage:
    python Scripts/get_cosine_similarity.py --layers last --n-checkpoints 1
    python Scripts/get_cosine_similarity.py \\
        --checkpoints-from Data/binomial_representations.csv --layers last
"""

import argparse
import csv
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).parent))
from binomial_rep_analysis import (
    BINOMS_CSV, MODEL_CONFIGS, DEFAULT_BATCH_SIZE,
    N_LOG_CHECKPOINTS, PROJECT_ROOT,
    load_binomials, collect_sentences, extract_representations,
    get_model_checkpoints, log_sample_checkpoints, _load_tokenizer,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
OUT_CSV = str(Path(PROJECT_ROOT) / "Data" / "cosine_similarity.csv")

FIELDNAMES = [
    "model", "model_size", "checkpoint", "step", "tokens",
    "phrase_AB", "layer", "cosine_sim",
]


# ---------------------------------------------------------------------------
# RESUME LOGIC
# ---------------------------------------------------------------------------
def load_completed(path: str) -> set:
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
# COSINE SIMILARITY
# ---------------------------------------------------------------------------
def paired_cosine_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Per-pair cosine similarity between matched rows of A and B.
    A, B: (n, D)
    Returns: (n,) cosine similarities
    """
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    return (A_norm * B_norm).sum(axis=1)


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

    tmp_cache = tempfile.mkdtemp(prefix="hf_cos_")
    model = None
    try:
        load_kw = dict(low_cpu_mem_usage=True, cache_dir=tmp_cache)
        if ckpt["tag"]:
            load_kw["revision"] = ckpt["tag"]
        model = AutoModel.from_pretrained(model_name, **load_kw).to(device).eval()

        all_map = {
            phrase: phrase_sentence_map[phrase]
            for _, row in binoms_df.iterrows()
            for phrase in (row["phrase_AB"], row["phrase_BA"])
        }

        print(f"  Extracting representations for all binomials ...")
        all_reps = extract_representations(
            model, tokenizer, all_map, device,
            batch_size=config.get("batch_size", DEFAULT_BATCH_SIZE),
        )

        all_layers = sorted({
            layer
            for _, row in binoms_df.iterrows()
            for layer in (
                set(all_reps.get(row["phrase_AB"], {})) &
                set(all_reps.get(row["phrase_BA"], {}))
            )
        })
        if layers_filter == "last":
            all_layers = [max(all_layers)] if all_layers else []
        elif layers_filter is not None:
            keep = {int(x) for x in layers_filter.split(",")}
            all_layers = [l for l in all_layers if l in keep]

        for layer_idx in tqdm(all_layers, desc="  Layers"):
            for _, row in binoms_df.iterrows():
                ab = row["phrase_AB"]
                ba = row["phrase_BA"]

                if (model_name, ckpt["checkpoint"], ab, layer_idx) in completed:
                    continue
                if layer_idx not in all_reps.get(ab, {}):
                    continue
                if layer_idx not in all_reps.get(ba, {}):
                    continue

                A = all_reps[ab][layer_idx]
                B = all_reps[ba][layer_idx]
                n = min(len(A), len(B))
                if n == 0:
                    continue

                sims = paired_cosine_sim(A[:n], B[:n])

                completed.add((model_name, ckpt["checkpoint"], ab, layer_idx))
                for sim in sims:
                    writer.writerow({
                        "model":      model_name,
                        "model_size": size_label,
                        "checkpoint": ckpt["checkpoint"],
                        "step":       ckpt["step"],
                        "tokens":     ckpt["tokens"],
                        "phrase_AB":  ab,
                        "layer":      layer_idx,
                        "cosine_sim": float(sim),
                    })

            out_file.flush()

        print(f"  Done.")

    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()
        shutil.rmtree(tmp_cache, ignore_errors=True)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def _checkpoints_from_csv(csv_path: str) -> dict:
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
        description="Compute paired cosine similarity of AB vs BA representations."
    )
    parser.add_argument(
        "--n-checkpoints", type=int, default=N_LOG_CHECKPOINTS,
        help="Number of log-sampled checkpoints (1 = final only). "
             "Ignored when --checkpoints-from is set.",
    )
    parser.add_argument(
        "--layers", type=str, default="last",
        help="Layers to score: 'all', 'last' (default), or comma-separated indices.",
    )
    parser.add_argument(
        "--checkpoints-from", type=str, default=None, metavar="CSV",
        help="Read checkpoints from an existing results CSV.",
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

    if args.checkpoints_from:
        print(f"\nReading checkpoints from {args.checkpoints_from} ...")
        ckpts_by_model = _checkpoints_from_csv(args.checkpoints_from)
    else:
        ckpts_by_model = None

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
        print(f"  {len(df):,} rows  |  {n_binoms:,} binomial×checkpoint×layer combinations")
        print(f"  Mean cosine similarity: {df['cosine_sim'].mean():.4f}  "
              f"(range: {df['cosine_sim'].min():.4f} – {df['cosine_sim'].max():.4f})")


if __name__ == "__main__":
    main()
