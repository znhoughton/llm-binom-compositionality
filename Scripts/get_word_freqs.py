#!/usr/bin/env python3
"""
get_word_freqs.py
-----------------
Counts unigram frequencies for all words appearing in attested binomials
from the BabyLM training corpus.

Output: Data/babylm_word_freqs.csv
  columns: word, freq

Usage:
    python Scripts/get_word_freqs.py
"""

import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

DATASET_ID = "znhoughton/babylm-150m-v3"
DATASET_SPLIT = "train"
BINOMS_CSV = str(Path(__file__).resolve().parent.parent / "Data" / "nonce_and_attested_binoms.csv")
OUT_CSV    = str(Path(__file__).resolve().parent.parent / "Data" / "babylm_word_freqs.csv")

df = pd.read_csv(BINOMS_CSV)
df = df[df["Attested"] == 1]
words = set(
    df["Word1"].str.strip().str.lower().tolist() +
    df["Word2"].str.strip().str.lower().tolist()
)
counts = {w: 0 for w in words}
print(f"Counting {len(words)} unique words ...")

pattern = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(w) for w in words) + r")(?!\w)",
    re.IGNORECASE,
)

ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
print(f"  {len(ds):,} documents loaded.")

for doc in tqdm(ds, desc="Corpus pass", unit="doc"):
    if not doc["text"]:
        continue
    for match in pattern.findall(doc["text"].lower()):
        if match in counts:
            counts[match] += 1

out_df = pd.DataFrame([{"word": w, "freq": c} for w, c in counts.items()]) \
           .sort_values("word")
out_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved -> {OUT_CSV}")
print(out_df.sort_values("freq", ascending=False).head(10).to_string(index=False))
