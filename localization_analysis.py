"""
Comparative Localization Analysis for GPT-2 vs Gemma-3n

This script compares the localization of sparse subnetworks between:
- GPT-2 Medium (standard transformer)
- Gemma-3n (MatFormer with nested representations)

Key metrics:
- Gini coefficient: Higher = more concentrated updates
- Entropy: Lower = more localized
- Top-k concentration: Higher = updates in fewer layers
- Block-level heatmaps: Visual comparison of update distribution

Hypothesis: Gemma-3n's MatFormer architecture will show MORE localized
sparse subnetworks due to its nested FFN structure.
"""

import os
import json
import argparse
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import torch

from configs import get_model_config, get_name_filter_for_model, analyze_matformer_nesting
from utils import (
    compute_gini_coefficient,
    compute_entropy_concentration,
    compute_topk_concentration,
    extract_block_index,
)


@dataclass
class ModelResults:
    """Container for analysis results of a single model."""
    model_name: str
    steps: List[int]
    coverages: List[float]
    sparsities: List[float]
    gini_scores: List[float]
    entropy_scores: List[float]
    topk_scores: List[float]
    block_norms: List[Dict[int, float]]
    nesting_analysis: Optional[Dict] = None


def load_state(step: int, ckpt_dir: str) -> Dict[str, torch.Tensor]:
    """Load checkpoint state dict for a given step."""
    path = os.path.join(ckpt_dir, f"step_{step:03d}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def collect_tracked_names(
    state_dict: Dict[str, torch.Tensor],
    model_name: str
) -> List[str]:
    """Collect parameter names to track using model-specific filter."""
    name_filter = get_name_filter_for_model(model_name)
    names = []
    for name, tensor in state_dict.items():
        if not name.endswith("weight"):
            continue
        if not any(tok in name for tok in name_filter):
            continue
        if tensor.ndim != 2:
            continue
        names.append(name)
    return sorted(names)


def compute_layer_fro_norms(
    base_sd: Dict[str, torch.Tensor],
    cur_sd: Dict[str, torch.Tensor],
    tracked_names: List[str],
) -> Dict[str, float]:
    """Compute Frobenius norm of weight delta for each layer."""
    layer_norms = {}
    for name in tracked_names:
        W0 = base_sd[name].to(torch.float32)
        Wt = cur_sd[name].to(torch.float32)
        delta = Wt - W0
        layer_norms[name] = torch.linalg.norm(delta).item()
        del W0, Wt, delta
    return layer_norms


def compute_block_norms(
    base_sd: Dict[str, torch.Tensor],
    cur_sd: Dict[str, torch.Tensor],
    tracked_names: List[str],
) -> Dict[int, float]:
    """Compute total Frobenius norm per transformer block."""
    block_norms = defaultdict(float)
    for name in tracked_names:
        block_idx = extract_block_index(name)
        W0 = base_sd[name].to(torch.float32)
        Wt = cur_sd[name].to(torch.float32)
        delta = Wt - W0
        block_norms[block_idx] += torch.linalg.norm(delta).item()
        del W0, Wt, delta
    return dict(block_norms)


def analyze_model(
    model_name: str,
    ckpt_dir: str,
    steps: List[int],
    tau: float = 1e-6,
) -> ModelResults:
    """
    Run full localization analysis for a single model.

    Returns ModelResults containing all metrics over training.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name}")
    print(f"Checkpoint dir: {ckpt_dir}")
    print(f"{'='*60}")

    # Load base checkpoint
    base_sd = load_state(steps[0], ckpt_dir)
    tracked_names = collect_tracked_names(base_sd, model_name)

    print(f"  Tracking {len(tracked_names)} parameter tensors")

    coverages = []
    sparsities = []
    gini_scores = []
    entropy_scores = []
    topk_scores = []
    block_norms_over_time = []

    # Get final subnetwork for coverage calculation
    final_sd = load_state(steps[-1], ckpt_dir)
    final_layer_norms = compute_layer_fro_norms(base_sd, final_sd, tracked_names)
    final_active_set = set(
        name for name, norm in final_layer_norms.items() if norm > tau
    )
    del final_sd

    N_total = sum(base_sd[name].numel() for name in tracked_names)

    for step in steps:
        try:
            cur_sd = load_state(step, ckpt_dir)

            # Sparsity (fraction of layers with significant updates)
            layer_norms = compute_layer_fro_norms(base_sd, cur_sd, tracked_names)
            active_layers = set(name for name, norm in layer_norms.items() if norm > tau)
            sparsity = len(active_layers) / len(tracked_names) if tracked_names else 0

            # Coverage
            coverage = len(active_layers & final_active_set) / len(final_active_set) if final_active_set else 0

            # Localization metrics
            gini = compute_gini_coefficient(layer_norms)
            entropy = compute_entropy_concentration(layer_norms)
            topk = compute_topk_concentration(layer_norms, k=3)

            # Block-level norms
            block_norms = compute_block_norms(base_sd, cur_sd, tracked_names)

            coverages.append(coverage)
            sparsities.append(sparsity)
            gini_scores.append(gini)
            entropy_scores.append(entropy)
            topk_scores.append(topk)
            block_norms_over_time.append(block_norms)

            print(f"  Step {step:3d}: gini={gini:.3f}, entropy={entropy:.3f}, topk={topk:.3f}")

            del cur_sd

        except FileNotFoundError as e:
            print(f"  Step {step:3d}: MISSING ({e})")
            coverages.append(float('nan'))
            sparsities.append(float('nan'))
            gini_scores.append(float('nan'))
            entropy_scores.append(float('nan'))
            topk_scores.append(float('nan'))
            block_norms_over_time.append({})

    return ModelResults(
        model_name=model_name,
        steps=steps,
        coverages=coverages,
        sparsities=sparsities,
        gini_scores=gini_scores,
        entropy_scores=entropy_scores,
        topk_scores=topk_scores,
        block_norms=block_norms_over_time,
    )


def plot_comparative_metrics(
    results: Dict[str, ModelResults],
    output_dir: str = "plots",
):
    """
    Generate comparative plots for multiple models.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Color scheme for models
    colors = {
        "gpt2-medium": "#1f77b4",
        "gemma-3n-E2B": "#ff7f0e",
        "gemma-3n-E4B": "#2ca02c",
    }

    def get_color(model_name):
        for key, color in colors.items():
            if key in model_name.lower():
                return color
        return "#7f7f7f"  # gray default

    # 1. Gini coefficient comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for model_name, res in results.items():
        color = get_color(model_name)
        short_name = model_name.split("/")[-1]

        # Gini
        axes[0].plot(res.steps, res.gini_scores, marker="o", markersize=4,
                     linestyle="-", color=color, label=short_name)

        # Entropy
        axes[1].plot(res.steps, res.entropy_scores, marker="o", markersize=4,
                     linestyle="-", color=color, label=short_name)

        # Top-k
        axes[2].plot(res.steps, res.topk_scores, marker="o", markersize=4,
                     linestyle="-", color=color, label=short_name)

    axes[0].set_xlabel("PPO Step")
    axes[0].set_ylabel("Gini Coefficient")
    axes[0].set_title("Update Concentration (Gini)\nHigher = More Localized")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.7)
    axes[0].set_ylim([0, 1])

    axes[1].set_xlabel("PPO Step")
    axes[1].set_ylabel("Normalized Entropy")
    axes[1].set_title("Update Distribution (Entropy)\nLower = More Localized")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.7)
    axes[1].set_ylim([0, 1])

    axes[2].set_xlabel("PPO Step")
    axes[2].set_ylabel("Top-3 Concentration")
    axes[2].set_title("Top-3 Layer Concentration\nHigher = More Localized")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.7)
    axes[2].set_ylim([0, 1])

    plt.tight_layout()
    path = os.path.join(output_dir, "comparative_localization_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {path}")
    plt.close()

    # 2. Side-by-side heatmaps
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 8))
    if len(results) == 1:
        axes = [axes]

    for ax, (model_name, res) in zip(axes, results.items()):
        short_name = model_name.split("/")[-1]

        # Get all block indices
        all_blocks = set()
        for bn in res.block_norms:
            all_blocks.update(bn.keys())
        all_blocks = sorted([b for b in all_blocks if b >= 0])

        if not all_blocks:
            ax.text(0.5, 0.5, "No block data", ha="center", va="center")
            ax.set_title(f"{short_name}\n(No data)")
            continue

        # Build heatmap
        heatmap = np.zeros((len(all_blocks), len(res.steps)))
        for t_idx, bn in enumerate(res.block_norms):
            for b_idx, block in enumerate(all_blocks):
                heatmap[b_idx, t_idx] = bn.get(block, 0.0)

        # Normalize per model for better visualization
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        im = ax.imshow(heatmap, aspect="auto", cmap="hot", interpolation="nearest")
        ax.set_xlabel("PPO Step")
        ax.set_ylabel("Transformer Block")
        ax.set_title(f"{short_name}\nNormalized Update Distribution")

        # Axis labels
        step_ticks = range(0, len(res.steps), max(1, len(res.steps) // 10))
        ax.set_xticks(list(step_ticks))
        ax.set_xticklabels([str(res.steps[i]) for i in step_ticks], rotation=45)
        ax.set_yticks(range(len(all_blocks)))
        ax.set_yticklabels([str(b) for b in all_blocks])

        plt.colorbar(im, ax=ax, label="Normalized ||ΔW||_F")

    plt.tight_layout()
    path = os.path.join(output_dir, "comparative_heatmaps.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {path}")
    plt.close()

    # 3. Statistical comparison table
    print("\n" + "=" * 60)
    print("STATISTICAL COMPARISON")
    print("=" * 60)

    summary = {}
    for model_name, res in results.items():
        short_name = model_name.split("/")[-1]

        # Compute statistics over training (excluding step 0)
        gini_vals = [g for g in res.gini_scores[1:] if not np.isnan(g)]
        entropy_vals = [e for e in res.entropy_scores[1:] if not np.isnan(e)]
        topk_vals = [t for t in res.topk_scores[1:] if not np.isnan(t)]

        summary[short_name] = {
            "mean_gini": np.mean(gini_vals) if gini_vals else 0,
            "std_gini": np.std(gini_vals) if gini_vals else 0,
            "mean_entropy": np.mean(entropy_vals) if entropy_vals else 0,
            "std_entropy": np.std(entropy_vals) if entropy_vals else 0,
            "mean_topk": np.mean(topk_vals) if topk_vals else 0,
            "std_topk": np.std(topk_vals) if topk_vals else 0,
            "final_gini": res.gini_scores[-1] if not np.isnan(res.gini_scores[-1]) else 0,
            "final_entropy": res.entropy_scores[-1] if not np.isnan(res.entropy_scores[-1]) else 0,
            "final_topk": res.topk_scores[-1] if not np.isnan(res.topk_scores[-1]) else 0,
        }

        print(f"\n{short_name}:")
        print(f"  Gini:    mean={summary[short_name]['mean_gini']:.3f} ± {summary[short_name]['std_gini']:.3f}, "
              f"final={summary[short_name]['final_gini']:.3f}")
        print(f"  Entropy: mean={summary[short_name]['mean_entropy']:.3f} ± {summary[short_name]['std_entropy']:.3f}, "
              f"final={summary[short_name]['final_entropy']:.3f}")
        print(f"  Top-3:   mean={summary[short_name]['mean_topk']:.3f} ± {summary[short_name]['std_topk']:.3f}, "
              f"final={summary[short_name]['final_topk']:.3f}")

    # Save summary as JSON
    summary_path = os.path.join(output_dir, "localization_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved summary: {summary_path}")

    return summary


def plot_layer_histogram(
    results: Dict[str, ModelResults],
    output_dir: str = "plots",
):
    """
    Plot histogram of updates across layers for final checkpoint.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, (model_name, res) in zip(axes, results.items()):
        short_name = model_name.split("/")[-1]

        # Get final block norms
        final_norms = res.block_norms[-1] if res.block_norms else {}
        blocks = sorted([b for b in final_norms.keys() if b >= 0])

        if not blocks:
            ax.text(0.5, 0.5, "No block data", ha="center", va="center")
            ax.set_title(f"{short_name}\n(No data)")
            continue

        norms = [final_norms.get(b, 0) for b in blocks]

        ax.bar(blocks, norms, color="steelblue", alpha=0.8)
        ax.set_xlabel("Transformer Block")
        ax.set_ylabel("||ΔW||_F")
        ax.set_title(f"{short_name}\nFinal Update Distribution")
        ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    path = os.path.join(output_dir, "layer_histogram_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Comparative Localization Analysis for GPT-2 vs Gemma-3n"
    )
    parser.add_argument(
        "--gpt2_ckpt_dir",
        type=str,
        default="checkpoints_gpt2",
        help="Directory containing GPT-2 checkpoints"
    )
    parser.add_argument(
        "--gemma_ckpt_dir",
        type=str,
        default="checkpoints_gemma",
        help="Directory containing Gemma-3n checkpoints"
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=600,
        help="Maximum step to analyze"
    )
    parser.add_argument(
        "--step_interval",
        type=int,
        default=20,
        help="Interval between checkpoints"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots/comparative",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1e-6,
        help="Tolerance threshold for weight updates"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["gpt2-medium"],
        help="Models to analyze (space-separated)"
    )
    parser.add_argument(
        "--ckpt_dirs",
        type=str,
        nargs="+",
        default=["checkpoints"],
        help="Checkpoint directories (must match --models order)"
    )

    args = parser.parse_args()

    steps = list(range(0, args.max_step + 1, args.step_interval))

    # Validate inputs
    if len(args.models) != len(args.ckpt_dirs):
        print("Error: --models and --ckpt_dirs must have same number of arguments")
        return

    # Analyze each model
    results = {}
    for model_name, ckpt_dir in zip(args.models, args.ckpt_dirs):
        if os.path.exists(ckpt_dir):
            res = analyze_model(model_name, ckpt_dir, steps, args.tau)
            results[model_name] = res
        else:
            print(f"Warning: Checkpoint directory not found: {ckpt_dir}")

    if not results:
        print("No models analyzed. Check checkpoint directories.")
        return

    # Generate comparative plots
    summary = plot_comparative_metrics(results, args.output_dir)
    plot_layer_histogram(results, args.output_dir)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
