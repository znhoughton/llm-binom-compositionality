#!/usr/bin/env bash
# Pilot run: last layer, final checkpoint only.
# Use this to verify outputs before committing to the full run.
#
# Usage (activate your env first):
#   conda activate PRenv
#   bash Scripts/run_pilot.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "============================================================"
echo "PILOT: last layer, final checkpoint"
echo "============================================================"

echo -e "\n--- Cosine similarity (AB vs BA, in-context) ---"
python Scripts/get_cosine_similarity.py \
    --layers last \
    --n-checkpoints 1

echo -e "\n--- Compositional similarity (holistic vs word-average) ---"
python Scripts/get_compositional_similarity.py \
    --layers last \
    --n-checkpoints 1

echo -e "\n============================================================"
echo "Pilot run complete."
echo "============================================================"
