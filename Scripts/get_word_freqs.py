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
from multiprocessing import Pool, cpu_count

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

DATASET_ID  = "znhoughton/babylm-150m-v3"
BINOMS_CSV  = str(Path(__file__).resolve().parent.parent / "Data" / "nonce_and_attested_binoms.csv")
OUT_CSV     = str(Path(__file__).resolve().parent.parent / "Data" / "babylm_word_freqs.csv")

# ---------------------------------------------------------------------------
# Build target word set (module-level so multiprocessing workers inherit it)
# ---------------------------------------------------------------------------
_df = pd.read_csv(BINOMS_CSV)
_df = _df[_df["Attested"] == 1]
TARGET_WORDS = frozenset(
    _df["Word1"].str.strip().str.lower().tolist() +
    _df["Word2"].str.strip().str.lower().tolist()
)

_WORD_RE = re.compile(r"[a-zA-Z]+")


def _count_doc(text: str) -> dict:
    if not text:
        return {}
    counts = {}
    for word in _WORD_RE.findall(text.lower()):
        if word in TARGET_WORDS:
            counts[word] = counts.get(word, 0) + 1
    return counts


def main():
    print(f"Counting {len(TARGET_WORDS)} unique words ...")
    ds = load_dataset(DATASET_ID, split="train")
    print(f"  {len(ds):,} documents loaded.")

    counts = {w: 0 for w in TARGET_WORDS}
    n_workers = cpu_count()
    print(f"  Using {n_workers} CPU cores.")

    with Pool(n_workers) as pool:
        for doc_counts in tqdm(
            pool.imap(_count_doc, ds["text"], chunksize=500),
            total=len(ds), desc="Corpus pass", unit="doc",
        ):
            for word, c in doc_counts.items():
                counts[word] += c

    out_df = (
        pd.DataFrame([{"word": w, "freq": c} for w, c in counts.items()])
        .sort_values("word")
    )
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")
    print(out_df.sort_values("freq", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
