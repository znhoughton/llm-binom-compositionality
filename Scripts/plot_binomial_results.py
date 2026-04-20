#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_binomial_results.py
------------------------
Generates plots for the three analysis dimensions:
  1. Across layers      — how scores change by layer, for a given checkpoint
  2. Across training    — how scores change over checkpoints, for a given layer
  3. Across model sizes — comparing 125m / 350m / 1.3b

All plots are saved to Scripts/../Plots/

Usage:
    python Scripts/plot_binomial_results.py

Expects:
    Scripts/../results/binomial_representations.csv
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RESULTS_CSV = str(PROJECT_ROOT / "results" / "binomial_representations.csv")
PLOTS_DIR   = str(PROJECT_ROOT / "Plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# PLOT STYLE
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.dpi":       150,
})

MODEL_COLOURS = {
    "125m": "#4C72B0",
    "350m": "#DD8452",
    "1.3b": "#55A868",
}

SCORE_LABELS = {
    "procrustes_dist": "Procrustes Distance (asymmetry)",
    "self_sim_AB":     "Self-Similarity AB (α-ordering)",
    "self_sim_BA":     "Self-Similarity BA (non-α ordering)",
    "self_sim_ratio":  "Self-Similarity Ratio (AB / BA)",
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure numeric
    for col in ["layer", "step", "tokens", "overall_freq", "rel_freq",
                "procrustes_dist", "self_sim_AB", "self_sim_BA",
                "self_sim_ratio", "n_sentences_AB", "n_sentences_BA"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def save(fig, name: str):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def freq_quartile_label(q: int) -> str:
    return ["Q1 (lowest freq)", "Q2", "Q3", "Q4 (highest freq)"][q]


def add_freq_quartiles(df: pd.DataFrame) -> pd.DataFrame:
    """Add a freq_quartile column (0–3) based on overall_freq."""
    df = df.copy()
    df["freq_quartile"] = pd.qcut(
        df["overall_freq"], q=4, labels=False, duplicates="drop"
    )
    return df

# ---------------------------------------------------------------------------
# DIMENSION 1: ACROSS LAYERS
# Plots score vs layer, separately per model size.
# Uses the last (most-trained) checkpoint for each model.
# Lines are coloured by log-frequency quartile.
# ---------------------------------------------------------------------------

def plot_across_layers(df: pd.DataFrame, score: str = "procrustes_dist"):
    print(f"\nPlotting across layers ({score}) ...")

    # Use last available checkpoint per model
    last_ckpts = (df.groupby("model_size")["step"]
                    .max().reset_index()
                    .rename(columns={"step": "max_step"}))
    df_last = df.merge(last_ckpts, on="model_size")
    df_last = df_last[df_last["step"] == df_last["max_step"]]

    df_last = add_freq_quartiles(df_last)
    quartile_colours = plt.cm.viridis(np.linspace(0.15, 0.85, 4))

    model_sizes = ["125m", "350m", "1.3b"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, size in zip(axes, model_sizes):
        sub = df_last[df_last["model_size"] == size]
        if sub.empty:
            ax.set_title(f"{size} (no data)")
            continue

        for q in range(4):
            q_sub = sub[sub["freq_quartile"] == q]
            if q_sub.empty:
                continue
            # Mean score per layer across binomials in this quartile
            layer_means = (q_sub.groupby("layer")[score]
                               .mean().reset_index())
            ax.plot(
                layer_means["layer"], layer_means[score],
                color=quartile_colours[q],
                label=freq_quartile_label(q),
                linewidth=2,
            )

        ax.set_title(f"{size}", fontweight="bold")
        ax.set_xlabel("Layer")
        if ax is axes[0]:
            ax.set_ylabel(SCORE_LABELS.get(score, score))

    axes[-1].legend(title="Frequency quartile",
                    bbox_to_anchor=(1.02, 1), loc="upper left",
                    fontsize=9)
    fig.suptitle(f"{SCORE_LABELS.get(score, score)} across layers\n"
                 f"(final checkpoint, mean per quartile)",
                 y=1.02)
    save(fig, f"across_layers_{score}.png")


# ---------------------------------------------------------------------------
# DIMENSION 2: ACROSS TRAINING
# Score vs training tokens, for a fixed layer (default: last layer).
# One panel per model size, lines coloured by frequency quartile.
# ---------------------------------------------------------------------------

def plot_across_training(df: pd.DataFrame,
                         score: str = "procrustes_dist",
                         layer: Optional[int] = None):
    """
    layer=None → use the final hidden layer (max layer index) per model.
    """
    print(f"\nPlotting across training ({score}) ...")

    if layer is None:
        # Use last hidden layer (excluding embedding layer 0)
        max_layer = int(df["layer"].max())
        df_layer  = df[df["layer"] == max_layer]
        layer_label = f"layer {max_layer} (final)"
    else:
        df_layer    = df[df["layer"] == layer]
        layer_label = f"layer {layer}"

    df_layer = add_freq_quartiles(df_layer)
    quartile_colours = plt.cm.viridis(np.linspace(0.15, 0.85, 4))

    model_sizes = ["125m", "350m", "1.3b"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, size in zip(axes, model_sizes):
        sub = df_layer[df_layer["model_size"] == size]
        if sub.empty:
            ax.set_title(f"{size} (no data)")
            continue

        for q in range(4):
            q_sub = sub[sub["freq_quartile"] == q]
            if q_sub.empty:
                continue
            token_means = (q_sub.groupby("tokens")[score]
                               .mean().reset_index()
                               .sort_values("tokens"))
            ax.plot(
                token_means["tokens"] / 1e6,
                token_means[score],
                color=quartile_colours[q],
                label=freq_quartile_label(q),
                linewidth=2,
            )

        ax.set_title(f"{size}", fontweight="bold")
        ax.set_xlabel("Training tokens (M)")
        if ax is axes[0]:
            ax.set_ylabel(SCORE_LABELS.get(score, score))

    axes[-1].legend(title="Frequency quartile",
                    bbox_to_anchor=(1.02, 1), loc="upper left",
                    fontsize=9)
    fig.suptitle(
        f"{SCORE_LABELS.get(score, score)} across training\n"
        f"({layer_label}, mean per quartile)",
        y=1.02,
    )
    save(fig, f"across_training_{score}.png")


# ---------------------------------------------------------------------------
# DIMENSION 3: ACROSS MODEL SIZES
# Score vs log-frequency, one panel per model size, coloured by layer.
# Uses the last checkpoint only.
# ---------------------------------------------------------------------------

def plot_across_models(df: pd.DataFrame, score: str = "procrustes_dist"):
    print(f"\nPlotting across models ({score}) ...")

    last_ckpts = (df.groupby("model_size")["step"]
                    .max().reset_index()
                    .rename(columns={"step": "max_step"}))
    df_last = df.merge(last_ckpts, on="model_size")
    df_last = df_last[df_last["step"] == df_last["max_step"]]
    df_last = df_last[df_last["overall_freq"] > 0].copy()
    df_last["log_freq"] = np.log10(df_last["overall_freq"])

    # Mean score per binomial × model × layer (average over sentences)
    # Then scatter: x=log_freq, y=score, coloured by layer
    model_sizes = ["125m", "350m", "1.3b"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    # Use a few representative layers: first, middle, last
    all_layers  = sorted(df_last["layer"].unique())
    n_layers    = len(all_layers)
    rep_layers  = [all_layers[0],
                   all_layers[n_layers // 2],
                   all_layers[-1]]
    layer_cmap  = plt.cm.plasma(np.linspace(0.1, 0.9, len(rep_layers)))

    for ax, size in zip(axes, model_sizes):
        sub = df_last[df_last["model_size"] == size]
        if sub.empty:
            ax.set_title(f"{size} (no data)")
            continue

        for layer_col, lyr in zip(layer_cmap, rep_layers):
            l_sub = sub[sub["layer"] == lyr]
            if l_sub.empty:
                continue
            binom_means = (l_sub.groupby(["word1", "word2", "log_freq"])
                               [score].mean().reset_index())
            ax.scatter(
                binom_means["log_freq"], binom_means[score],
                color=layer_col, alpha=0.55, s=18,
                label=f"layer {lyr}",
            )
            # Regression line
            x = binom_means["log_freq"].values
            y = binom_means[score].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() > 2:
                m, b = np.polyfit(x[mask], y[mask], 1)
                xr   = np.linspace(x[mask].min(), x[mask].max(), 50)
                ax.plot(xr, m * xr + b, color=layer_col, linewidth=1.5)

        ax.set_title(f"{size}", fontweight="bold")
        ax.set_xlabel("log₁₀(overall frequency)")
        if ax is axes[0]:
            ax.set_ylabel(SCORE_LABELS.get(score, score))

    axes[-1].legend(title="Layer",
                    bbox_to_anchor=(1.02, 1), loc="upper left",
                    fontsize=9)
    fig.suptitle(
        f"{SCORE_LABELS.get(score, score)} vs log-frequency across models\n"
        f"(final checkpoint, representative layers)",
        y=1.02,
    )
    save(fig, f"across_models_{score}.png")


# ---------------------------------------------------------------------------
# BONUS: Correlation heatmap  —  per layer, per model, r(score, log_freq)
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame, score: str = "procrustes_dist"):
    print(f"\nPlotting correlation heatmap ({score}) ...")

    last_ckpts = (df.groupby("model_size")["step"]
                    .max().reset_index()
                    .rename(columns={"step": "max_step"}))
    df_last = df.merge(last_ckpts, on="model_size")
    df_last = df_last[df_last["step"] == df_last["max_step"]]
    df_last = df_last[df_last["overall_freq"] > 0].copy()
    df_last["log_freq"] = np.log10(df_last["overall_freq"])

    model_sizes = ["125m", "350m", "1.3b"]
    layers      = sorted(df_last["layer"].unique())

    corr_matrix = np.full((len(layers), len(model_sizes)), np.nan)

    for j, size in enumerate(model_sizes):
        for i, lyr in enumerate(layers):
            sub = df_last[(df_last["model_size"] == size) &
                          (df_last["layer"] == lyr)]
            binom_means = (sub.groupby(["word1", "word2", "log_freq"])
                             [score].mean().reset_index())
            x = binom_means["log_freq"].values
            y = binom_means[score].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() > 5:
                r, _ = spearmanr(x[mask], y[mask])
                corr_matrix[i, j] = r

    fig, ax = plt.subplots(figsize=(5, max(4, len(layers) * 0.25)))
    im = ax.imshow(corr_matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Spearman r")
    ax.set_xticks(range(len(model_sizes)))
    ax.set_xticklabels(model_sizes)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=7)
    ax.set_xlabel("Model size")
    ax.set_ylabel("Layer")
    ax.set_title(f"Spearman r({SCORE_LABELS.get(score, score)},\n"
                 f"log freq) per layer & model")
    save(fig, f"correlation_heatmap_{score}.png")


# ---------------------------------------------------------------------------
# BONUS: Training dynamics for individual high/low freq binomials
# ---------------------------------------------------------------------------

def plot_training_dynamics_examples(df: pd.DataFrame,
                                    score: str = "procrustes_dist",
                                    model_size: str = "125m",
                                    n_examples: int = 5):
    """
    For one model size, plot the training trajectory of individual binomials,
    highlighting high-frequency vs low-frequency examples.
    Uses the last layer.
    """
    print(f"\nPlotting training dynamics examples ({score}, {model_size}) ...")

    max_layer = int(df["layer"].max())
    sub = df[(df["model_size"] == model_size) &
             (df["layer"] == max_layer) &
             (df["overall_freq"] > 0)].copy()

    if sub.empty:
        print(f"  No data for {model_size}, skipping.")
        return

    # Pick top-n and bottom-n by frequency
    binom_freqs = (sub.groupby(["word1", "word2"])["overall_freq"]
                     .first().reset_index()
                     .sort_values("overall_freq"))
    low_freq  = binom_freqs.head(n_examples)
    high_freq = binom_freqs.tail(n_examples)

    fig, ax = plt.subplots(figsize=(9, 5))

    for _, row in high_freq.iterrows():
        traj = (sub[(sub["word1"] == row["word1"]) &
                    (sub["word2"] == row["word2"])]
                .groupby("tokens")[score].mean()
                .reset_index().sort_values("tokens"))
        ax.plot(traj["tokens"] / 1e6, traj[score],
                color="#D95F02", alpha=0.7, linewidth=1.5,
                label=f"{row['word1']} and {row['word2']}"
                      f" (f={int(row['overall_freq']):,})")

    for _, row in low_freq.iterrows():
        traj = (sub[(sub["word1"] == row["word1"]) &
                    (sub["word2"] == row["word2"])]
                .groupby("tokens")[score].mean()
                .reset_index().sort_values("tokens"))
        ax.plot(traj["tokens"] / 1e6, traj[score],
                color="#1B9E77", alpha=0.7, linewidth=1.5,
                label=f"{row['word1']} and {row['word2']}"
                      f" (f={int(row['overall_freq']):,})")

    ax.set_xlabel("Training tokens (M)")
    ax.set_ylabel(SCORE_LABELS.get(score, score))
    ax.set_title(f"{score} over training — {model_size}\n"
                 f"Orange = high freq, Green = low freq")
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    save(fig, f"training_dynamics_{score}_{model_size}.png")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

# Allow Optional import at module level for the type hint above
from typing import Optional


def main():
    print(f"Loading results from {RESULTS_CSV} ...")
    df = load_results(RESULTS_CSV)
    print(f"  {len(df):,} rows loaded.")

    if df.empty:
        print("No data found — run binomial_rep_analysis.py first.")
        return

    for score in ["procrustes_dist", "self_sim_AB", "self_sim_BA"]:
        plot_across_layers(df, score=score)
        plot_across_training(df, score=score)
        plot_across_models(df, score=score)
        plot_correlation_heatmap(df, score=score)

    for size in ["125m", "350m", "1.3b"]:
        plot_training_dynamics_examples(
            df, score="procrustes_dist", model_size=size
        )

    print(f"\n🏁 All plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
