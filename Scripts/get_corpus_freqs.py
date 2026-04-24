#!/usr/bin/env python3
"""
get_corpus_freqs.py
-------------------
Counts phrase-level frequencies for all attested binomials in the BabyLM
training corpus, producing corpus-derived overall_freq and rel_freq to
replace the web-corpus estimates in the binomials CSV.

For each binomial (word1, word2) we count:
  freq_AB  = occurrences of "word1 and word2" (case-insensitive)
  freq_BA  = occurrences of "word2 and word1" (case-insensitive)
  overall_freq = freq_AB + freq_BA
  rel_freq     = freq_AB / overall_freq   (NaN if both zero)

Output: Data/babylm_corpus_freqs.csv
  columns: phrase_AB, phrase_BA, freq_AB, freq_BA, overall_freq, rel_freq

Usage:
    python Scripts/get_corpus_freqs.py
"""

import csv
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATASET_ID    = "znhoughton/babylm-150m-v3"
DATASET_SPLIT = "train"
BINOMS_CSV    = str(Path(__file__).resolve().parent.parent / "Data" / "nonce_and_attested_binoms.csv")
OUT_CSV       = str(Path(__file__).resolve().parent.parent / "Data" / "babylm_corpus_freqs.csv")

# ---------------------------------------------------------------------------
# LOAD PHRASES
# ---------------------------------------------------------------------------
def load_phrases(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df[df["Attested"] == 1].copy()
    df["word1"] = df["Word1"].str.strip().str.lower()
    df["word2"] = df["Word2"].str.strip().str.lower()
    df["phrase_AB"] = df["word1"] + " and " + df["word2"]
    df["phrase_BA"] = df["word2"] + " and " + df["word1"]
    return df[["phrase_AB", "phrase_BA"]].drop_duplicates()

# ---------------------------------------------------------------------------
# COUNT PHRASES
# ---------------------------------------------------------------------------
def count_phrases(ds, phrases_df):
    """
    Single pass over the corpus, counting all target phrases simultaneously.
    Uses re.findall with a compiled alternation pattern for efficiency.
    """
    # Build lookup: normalised phrase string → counter key
    all_phrases = list(
        set(phrases_df["phrase_AB"].tolist() + phrases_df["phrase_BA"].tolist())
    )
    counts = {p: 0 for p in all_phrases}

    # Compile one regex that matches any of the target phrases (word-boundary
    # anchored, case-insensitive). Escape each phrase in case of special chars.
    pattern = re.compile(
        r"(?<!\w)(" +
        "|".join(re.escape(p) for p in all_phrases) +
        r")(?!\w)",
        re.IGNORECASE,
    )

    print(f"Counting {len(all_phrases)} phrase variants across corpus ...")
    for doc in tqdm(ds, desc="Corpus pass", unit="doc"):
        text = doc["text"]
        if not text:
            continue
        for match in pattern.findall(text.lower()):
            if match in counts:
                counts[match] += 1

    return counts

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print(f"Loading {DATASET_ID} ({DATASET_SPLIT}) ...")
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    print(f"  {len(ds):,} documents loaded.")

    phrases_df = load_phrases(BINOMS_CSV)
    print(f"  {len(phrases_df)} attested binomials to count.")

    counts = count_phrases(ds, phrases_df)

    rows = []
    for _, row in phrases_df.iterrows():
        ab = row["phrase_AB"]
        ba = row["phrase_BA"]
        freq_ab = counts.get(ab, 0)
        freq_ba = counts.get(ba, 0)
        total   = freq_ab + freq_ba
        rel     = freq_ab / total if total > 0 else float("nan")
        rows.append({
            "phrase_AB":    ab,
            "phrase_BA":    ba,
            "freq_AB":      freq_ab,
            "freq_BA":      freq_ba,
            "overall_freq": total,
            "rel_freq":     rel,
        })

    out_df = pd.DataFrame(rows).sort_values("phrase_AB")
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")

    # Summary
    n_zero = (out_df["overall_freq"] == 0).sum()
    print(f"  Binomials with zero corpus occurrences: {n_zero} / {len(out_df)}")
    print(f"  overall_freq range: {out_df['overall_freq'].min()} – {out_df['overall_freq'].max():,}")
    print(f"  Top 10 by frequency:")
    print(out_df.nlargest(10, "overall_freq")[["phrase_AB","overall_freq","rel_freq"]].to_string(index=False))


if __name__ == "__main__":
    main()
