#!/usr/bin/env bash
# Full run: all hidden layers, 10 log-sampled checkpoints.
#
# BabyLM models (125m, 350m, 1.3b): 10 log-sampled checkpoints, all layers.
# OLMo models (1B, 7B): final checkpoint only (no intermediate checkpoints
# configured); all layers.
#
# Usage (activate your env first):
#   conda activate PRenv
#   bash Scripts/run_full.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "============================================================"
echo "FULL RUN: all layers, 10 log-sampled checkpoints"
echo "============================================================"

echo -e "\n--- Procrustes + self-similarity (BabyLM; all layers, 10 checkpoints) ---"
python Scripts/binomial_rep_analysis.py

echo -e "\n--- Cosine similarity (BabyLM + OLMo; all layers, 10 checkpoints) ---"
python Scripts/get_cosine_similarity.py \
    --layers all \
    --n-checkpoints 10

echo -e "\n--- Compositional similarity (BabyLM + OLMo; all layers, 10 checkpoints) ---"
python Scripts/get_compositional_similarity.py \
    --layers all \
    --n-checkpoints 10

echo -e "\n============================================================"
echo "Full run complete."
echo "============================================================"
