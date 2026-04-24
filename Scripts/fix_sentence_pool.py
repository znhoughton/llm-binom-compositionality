#!/usr/bin/env python3
"""
fix_sentence_pool.py
--------------------
Replaces all BA sentences in the sentence pool with versions derived by
swapping the phrase in the corresponding AB sentence.  This removes the
context confound introduced by generating AB and BA sentences independently.

Before: AB and BA each have 500 independently-generated sentences.
After:  BA sentences are the AB sentences with the ordering swapped in-place.

Usage:
    python Scripts/fix_sentence_pool.py
"""

import csv
import re
import shutil
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BINOMS_CSV        = PROJECT_ROOT / "Data" / "nonce_and_attested_binoms.csv"
SENTENCE_POOL_CSV = PROJECT_ROOT / "results" / "sentence_pool.csv"


def swap_phrase(sentence: str, phrase_ab: str, phrase_ba: str) -> str:
    """Replace phrase_ab with phrase_ba, preserving sentence-initial capitalisation."""
    def _repl(match: re.Match) -> str:
        orig = match.group(0)
        result = phrase_ba
        if orig[0].isupper():
            result = result[0].upper() + result[1:]
        return result
    return re.sub(re.escape(phrase_ab), _repl, sentence, flags=re.IGNORECASE)


def load_pairs() -> list[tuple[str, str]]:
    import pandas as pd
    df = pd.read_csv(BINOMS_CSV)
    df = df[df["Attested"] == 1].copy()
    df["phrase_AB"] = df["Word1"].str.strip().str.lower() + " and " + df["Word2"].str.strip().str.lower()
    df["phrase_BA"] = df["Word2"].str.strip().str.lower() + " and " + df["Word1"].str.strip().str.lower()
    return list(zip(df["phrase_AB"], df["phrase_BA"]))


def load_pool(path: Path) -> dict[str, list[str]]:
    pool: dict[str, list[str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pool.setdefault(row["phrase"], []).append(row["sentence"])
    return pool


def save_pool(pool: dict[str, list[str]], path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["phrase", "sentence"])
        writer.writeheader()
        for phrase, sentences in pool.items():
            for sentence in sentences:
                writer.writerow({"phrase": phrase, "sentence": sentence})


def main():
    pairs = load_pairs()
    print(f"Loaded {len(pairs)} attested binomial pairs.")

    pool = load_pool(SENTENCE_POOL_CSV)
    print(f"Loaded sentence pool: {sum(len(v) for v in pool.values()):,} sentences "
          f"across {len(pool)} phrases.")

    # Back up original pool
    backup = SENTENCE_POOL_CSV.with_name("sentence_pool_old_confounded.csv")
    if not backup.exists():
        shutil.copy2(SENTENCE_POOL_CSV, backup)
        print(f"Backed up original pool -> {backup.name}")

    n_replaced = 0
    n_missing_ab = 0
    for phrase_ab, phrase_ba in pairs:
        sents_ab = pool.get(phrase_ab, [])
        if not sents_ab:
            print(f"  WARNING: no AB sentences found for '{phrase_ab}' — skipping.")
            n_missing_ab += 1
            continue
        derived_ba = [swap_phrase(s, phrase_ab, phrase_ba) for s in sents_ab]
        pool[phrase_ba] = derived_ba
        n_replaced += 1

    print(f"\nDerived BA sentences for {n_replaced} pairs "
          f"({n_missing_ab} skipped — no AB sentences).")

    save_pool(pool, SENTENCE_POOL_CSV)
    total = sum(len(v) for v in pool.values())
    print(f"Saved updated pool: {total:,} sentences across {len(pool)} phrases.")
    print(f"Done. Original backed up as {backup.name}.")


if __name__ == "__main__":
    main()
