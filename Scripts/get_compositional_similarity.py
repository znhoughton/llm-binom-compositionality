#!/usr/bin/env python3
"""
get_compositional_similarity.py
---------------------------------
For each attested binomial, computes the mean cosine similarity between:
  - In-context rep:  mean over phrase-span tokens extracted from full sentence contexts
  - Isolated rep:    mean over all non-special tokens when the phrase is run alone

  High cosine_sim -> context does not strongly change the phrase representation
  Low cosine_sim  -> phrase representation is heavily modulated by sentence context

This is a measure of compositionality / context-invariance. If the model represents
"men and women" the same way regardless of what surrounds it, it is treating the
phrase as a stable, compositional unit. If the surrounding context strongly shifts
the representation, the phrase is being interpreted more holistically/idiomatically.

Both orderings (AB and BA) are scored separately, so the analysis can test whether
preferred orderings have more stable (or more context-sensitive) representations.

Output: Data/compositional_similarity.csv

Usage:
    python Scripts/get_compositional_similarity.py --layers last --n-checkpoints 1
    python Scripts/get_compositional_similarity.py \\
        --checkpoints-from Data/binomial_representations.csv --layers last
    python Scripts/get_compositional_similarity.py \\
        --model allenai/OLMo-2-1124-7B --trust-remote-code --chunk-size 50 --layers last
"""

import argparse
import csv
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

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
OUT_CSV = str(Path(PROJECT_ROOT) / "Data" / "compositional_similarity.csv")

FIELDNAMES = [
    "model", "model_size", "checkpoint", "step", "tokens",
    "phrase_AB", "ordering", "layer", "mean_cosine_sim",
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
            completed.add((
                row["model"], row["checkpoint"],
                row["phrase_AB"], row["ordering"], int(row["layer"]),
            ))
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
# ISOLATED PHRASE REPRESENTATIONS
# ---------------------------------------------------------------------------
@torch.inference_mode()
def extract_isolated_representations(
    model,
    tokenizer,
    phrases: List[str],
    device: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Embed each phrase in isolation (no surrounding sentence context) and return
    the mean of all non-special tokens' hidden states at each layer.

    Prepends a space so BPE tokenization matches the mid-sentence form used in
    extract_representations (e.g. "Ġmen" rather than "men").

    Returns {phrase: {layer_idx: np.ndarray of shape (D,)}}
    """
    results: Dict[str, Dict[int, np.ndarray]] = {p: {} for p in phrases}

    for start in range(0, len(phrases), batch_size):
        batch_phrases = phrases[start:start + batch_size]
        batch_texts = [" " + p for p in batch_phrases]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offset_mappings = enc.pop("offset_mapping").tolist()
        attn_mask = enc["attention_mask"].tolist()
        enc = enc.to(device)

        outputs = model(**enc, output_hidden_states=True)

        for layer_idx, layer_h in enumerate(outputs.hidden_states):
            layer_np = layer_h.float().cpu().numpy()  # (B, T, D)
            for b_idx, phrase in enumerate(batch_phrases):
                offsets = offset_mappings[b_idx]
                # Keep only real (non-padding, non-special) tokens
                valid = [
                    j for j, (cs, ce) in enumerate(offsets)
                    if attn_mask[b_idx][j] == 1 and cs != ce
                ]
                if not valid:
                    continue
                results[phrase][layer_idx] = layer_np[b_idx, valid, :].mean(axis=0)

    return results


# ---------------------------------------------------------------------------
# COSINE SIMILARITY (many rows vs one vector)
# ---------------------------------------------------------------------------
def contextual_vs_isolated_cosine(
    ctx: np.ndarray,  # (n, D) — one row per sentence
    iso: np.ndarray,  # (D,)   — single isolated rep
) -> np.ndarray:
    """Per-sentence cosine similarity between contextual reps and isolated rep."""
    ctx_norm = ctx / np.linalg.norm(ctx, axis=1, keepdims=True).clip(min=1e-10)
    iso_norm = iso / max(np.linalg.norm(iso), 1e-10)
    return ctx_norm @ iso_norm  # (n,)


# ---------------------------------------------------------------------------
# PER-CHECKPOINT PROCESSING
# ---------------------------------------------------------------------------
def process_checkpoint(
    model_name, config, ckpt, tokenizer,
    binoms_df, phrase_sentence_map,
    completed, writer, out_file,
    device, layers_filter, chunk_size=None,
):
    size_label = config["size_label"]
    print(f"\n  Checkpoint: {ckpt['checkpoint']}  "
          f"(step={ckpt['step']}, tokens={ckpt['tokens']:,})")

    use_tmp_cache = bool(ckpt["tag"])
    tmp_cache = tempfile.mkdtemp(prefix="hf_compsim_") if use_tmp_cache else None
    model = None
    try:
        load_kw = dict(low_cpu_mem_usage=True)
        if ckpt["tag"]:
            load_kw["revision"] = ckpt["tag"]
            load_kw["cache_dir"] = tmp_cache
        if config.get("trust_remote_code"):
            load_kw["trust_remote_code"] = True
        model = AutoModel.from_pretrained(model_name, **load_kw).to(device).eval()

        n_chunks = math.ceil(len(binoms_df) / chunk_size) if chunk_size else 1

        for chunk_idx in range(n_chunks):
            chunk_df = (
                binoms_df.iloc[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
                if chunk_size else binoms_df
            )
            if n_chunks > 1:
                print(f"  Chunk {chunk_idx + 1}/{n_chunks} ...")

            all_phrases = list({
                phrase
                for _, row in chunk_df.iterrows()
                for phrase in (row["phrase_AB"], row["phrase_BA"])
            })

            chunk_map = {p: phrase_sentence_map[p] for p in all_phrases}

            print(f"  Extracting in-context representations ...")
            ctx_reps = extract_representations(
                model, tokenizer, chunk_map, device,
                batch_size=config.get("batch_size", DEFAULT_BATCH_SIZE),
            )

            print(f"  Extracting isolated representations ...")
            iso_reps = extract_isolated_representations(
                model, tokenizer, all_phrases, device,
                batch_size=config.get("batch_size", DEFAULT_BATCH_SIZE),
            )

            all_layers = sorted({
                layer
                for p in all_phrases
                for layer in set(ctx_reps.get(p, {})) & set(iso_reps.get(p, {}))
            })
            if layers_filter == "last":
                all_layers = [max(all_layers)] if all_layers else []
            elif layers_filter is not None:
                keep = {int(x) for x in layers_filter.split(",")}
                all_layers = [l for l in all_layers if l in keep]

            for layer_idx in tqdm(all_layers, desc="  Layers"):
                for _, row in chunk_df.iterrows():
                    ab = row["phrase_AB"]
                    ba = row["phrase_BA"]
                    for phrase, ordering in [(ab, "AB"), (ba, "BA")]:
                        key = (model_name, ckpt["checkpoint"], ab, ordering, layer_idx)
                        if key in completed:
                            continue
                        if layer_idx not in ctx_reps.get(phrase, {}):
                            continue
                        if layer_idx not in iso_reps.get(phrase, {}):
                            continue

                        ctx = ctx_reps[phrase][layer_idx]  # (n, D)
                        iso = iso_reps[phrase][layer_idx]  # (D,)
                        sims = contextual_vs_isolated_cosine(ctx, iso)
                        mean_sim = float(sims.mean())

                        completed.add(key)
                        writer.writerow({
                            "model":          model_name,
                            "model_size":     size_label,
                            "checkpoint":     ckpt["checkpoint"],
                            "step":           ckpt["step"],
                            "tokens":         ckpt["tokens"],
                            "phrase_AB":      ab,
                            "ordering":       ordering,
                            "layer":          layer_idx,
                            "mean_cosine_sim": mean_sim,
                        })

                out_file.flush()

            del ctx_reps, iso_reps

        print(f"  Done.")

    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()
        if tmp_cache:
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
        description="Compute compositional similarity: in-context vs isolated phrase reps."
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
    parser.add_argument(
        "--model", type=str, default=None,
        help="Run on a single HuggingFace model (bypasses MODEL_CONFIGS, uses final weights only).",
    )
    parser.add_argument(
        "--size-label", type=str, default=None,
        help="Size label for --model (e.g. '7b'). Defaults to the model name.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for --model (default {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true",
        help="Pass trust_remote_code=True to from_pretrained (needed for some models e.g. OLMo).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Process binomials in chunks of this size (useful for large models to limit RAM).",
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

    if args.model:
        models_to_run = {
            args.model: {
                "size_label":        args.size_label or args.model,
                "batch_size":        args.batch_size,
                "tokenizer":         args.model,
                "trust_remote_code": args.trust_remote_code,
            }
        }
        ckpts_override = {
            args.model: [{"checkpoint": "final", "tag": None, "step": 0, "tokens": 0}]
        }
    else:
        models_to_run  = MODEL_CONFIGS
        ckpts_override = None

    try:
        for model_name, config in models_to_run.items():
            print(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")
            tokenizer = _load_tokenizer(config)

            if ckpts_override is not None:
                checkpoints = ckpts_override[model_name]
            elif ckpts_by_model is not None:
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
                    chunk_size=args.chunk_size,
                )
    finally:
        out_file.close()

    print(f"\nSaved -> {OUT_CSV}")
    if Path(OUT_CSV).exists():
        import pandas as pd
        df = pd.read_csv(OUT_CSV)
        n_combos = df.groupby(["model", "checkpoint", "phrase_AB", "ordering", "layer"]).ngroups
        print(f"  {len(df):,} rows  |  {n_combos:,} phrase×ordering×checkpoint×layer combinations")
        print(f"  Mean compositional similarity: {df['mean_cosine_sim'].mean():.4f}  "
              f"(range: {df['mean_cosine_sim'].min():.4f} – {df['mean_cosine_sim'].max():.4f})")


if __name__ == "__main__":
    main()
