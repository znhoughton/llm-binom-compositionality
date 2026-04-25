#!/usr/bin/env python3
"""
get_infinigram_freqs.py
-----------------------
Queries the infini-gram API (https://api.infini-gram.io/) for phrase and
word counts in the Dolma v1.7 corpus — the training data for OLMo models.

For each attested binomial, retrieves:
  freq_AB      = count of "word1 and word2" in Dolma
  freq_BA      = count of "word2 and word1" in Dolma
  overall_freq = freq_AB + freq_BA
  rel_freq     = freq_AB / overall_freq  (NaN if both zero)
  word1_freq   = count of "word1" as a unigram
  word2_freq   = count of "word2" as a unigram

Output: Data/infinigram_freqs.csv

Usage:
    python Scripts/get_infinigram_freqs.py
"""

import csv
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

BINOMS_CSV = str(Path(__file__).resolve().parent.parent / "Data" / "nonce_and_attested_binoms.csv")
OUT_CSV    = str(Path(__file__).resolve().parent.parent / "Data" / "infinigram_freqs.csv")

API_URL    = "https://api.infini-gram.io/"
CORPUS     = "v4_dolma-v1_7_llama"   # Dolma v1.7, LLaMA tokenizer
SLEEP_SEC  = 0.1                      # polite delay between requests
MAX_RETRY  = 3


def query_count(text: str) -> int:
    """Return the count of `text` as an exact n-gram in CORPUS."""
    payload = {"corpus": CORPUS, "query_type": "count", "query": text}
    for attempt in range(MAX_RETRY):
        try:
            resp = requests.post(API_URL, json=payload, timeout=30)
            resp.raise_for_status()
            return int(resp.json().get("count", 0))
        except Exception as e:
            if attempt == MAX_RETRY - 1:
                print(f"  WARNING: failed to query '{text}' after {MAX_RETRY} attempts: {e}")
                return -1
            time.sleep(2 ** attempt)
    return -1


def load_completed(path: str) -> set:
    done = set()
    if not Path(path).exists():
        return done
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add(row["phrase_AB"])
    return done


def main():
    df = pd.read_csv(BINOMS_CSV)
    df = df[df["Attested"] == 1].copy()
    df["word1"]     = df["Word1"].str.strip().str.lower()
    df["word2"]     = df["Word2"].str.strip().str.lower()
    df["phrase_AB"] = df["word1"] + " and " + df["word2"]
    df["phrase_BA"] = df["word2"] + " and " + df["word1"]

    completed = load_completed(OUT_CSV)
    remaining = df[~df["phrase_AB"].isin(completed)]
    print(f"{len(df)} binomials total; {len(completed)} already done; "
          f"{len(remaining)} to query.")

    # Pre-compute unique word queries so each word is only queried once
    unique_words: dict[str, int] = {}

    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    exists = Path(OUT_CSV).exists()
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["phrase_AB", "phrase_BA", "freq_AB", "freq_BA",
                      "overall_freq", "rel_freq", "word1_freq", "word2_freq"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()

        for _, row in tqdm(remaining.iterrows(), total=len(remaining), desc="Binomials"):
            ab, ba = row["phrase_AB"], row["phrase_BA"]
            w1, w2 = row["word1"], row["word2"]

            freq_ab = query_count(ab);  time.sleep(SLEEP_SEC)
            freq_ba = query_count(ba);  time.sleep(SLEEP_SEC)

            if w1 not in unique_words:
                unique_words[w1] = query_count(w1);  time.sleep(SLEEP_SEC)
            if w2 not in unique_words:
                unique_words[w2] = query_count(w2);  time.sleep(SLEEP_SEC)

            total = freq_ab + freq_ba
            rel   = freq_ab / total if total > 0 else float("nan")

            writer.writerow({
                "phrase_AB":    ab,
                "phrase_BA":    ba,
                "freq_AB":      freq_ab,
                "freq_BA":      freq_ba,
                "overall_freq": total,
                "rel_freq":     rel,
                "word1_freq":   unique_words[w1],
                "word2_freq":   unique_words[w2],
            })
            f.flush()

    print(f"\nSaved -> {OUT_CSV}")
    result = pd.read_csv(OUT_CSV)
    n_zero = (result["overall_freq"] == 0).sum()
    print(f"  {len(result)} binomials  |  {n_zero} with zero corpus occurrences")
    print(f"  overall_freq range: {result['overall_freq'].min()} – {result['overall_freq'].max():,}")


if __name__ == "__main__":
    main()
