#!/usr/bin/env python3
"""
aggregate_cosine_similarity.py
------------------------------
Collapses the per-sentence cosine_similarity.csv (one row per sentence pair)
down to one row per (model, model_size, checkpoint, step, tokens, phrase_AB, layer)
by taking the mean cosine_sim across sentences.

Run this once to compact an existing large CSV produced by an older version of
get_cosine_similarity.py (which wrote one row per sentence).  After this the
file matches what the updated script produces directly.

Usage:
    python Scripts/aggregate_cosine_similarity.py
"""

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV  = str(PROJECT_ROOT / "Data" / "cosine_similarity.csv")
OUTPUT_CSV = str(PROJECT_ROOT / "Data" / "cosine_similarity.csv")

GROUP_COLS = ["model", "model_size", "checkpoint", "step", "tokens",
              "phrase_AB", "layer"]
CHUNK_SIZE = 2_000_000

def main():
    input_path  = Path(INPUT_CSV)
    output_path = Path(OUTPUT_CSV)

    if not input_path.exists():
        print(f"Not found: {INPUT_CSV}")
        sys.exit(1)

    size_gb = input_path.stat().st_size / 1e9
    print(f"Input:  {INPUT_CSV}  ({size_gb:.1f} GB)")

    # Accumulate running (sum, count) per group key — memory-safe because
    # the number of unique groups is small (binomials × checkpoints × layers).
    agg: dict = {}   # key → [sum, count]

    print(f"Reading in chunks of {CHUNK_SIZE:,} rows ...")
    reader = pd.read_csv(input_path, chunksize=CHUNK_SIZE,
                         dtype={"step": "Int64", "tokens": "Int64",
                                "layer": "Int64"})
    for chunk in tqdm(reader, desc="Chunks"):
        for key, grp in chunk.groupby(GROUP_COLS):
            s = grp["cosine_sim"].sum()
            n = len(grp)
            if key in agg:
                agg[key][0] += s
                agg[key][1] += n
            else:
                agg[key] = [s, n]

    print(f"  {len(agg):,} unique (model × checkpoint × binomial × layer) groups")

    rows = []
    for key, (s, n) in agg.items():
        row = dict(zip(GROUP_COLS, key))
        row["cosine_sim"] = s / n
        rows.append(row)

    out_df = pd.DataFrame(rows, columns=GROUP_COLS + ["cosine_sim"])
    out_df.to_csv(output_path, index=False)

    size_mb = output_path.stat().st_size / 1e6
    print(f"Output: {OUTPUT_CSV}  ({size_mb:.1f} MB)")
    print(f"  {len(out_df):,} rows  |  mean cosine_sim: {out_df['cosine_sim'].mean():.4f}")


if __name__ == "__main__":
    main()
