# analyze_sparse_subnetwork.py
"""
Sparse Subnetwork Analysis for PPO-trained models.

Supports both GPT-2 and Gemma-3n architectures with:
- Coverage analysis (how early does the final subnetwork emerge)
- Density analysis (what fraction of weights are updated)
- Module-level concentration (which transformer blocks are most updated)
- Localization metrics (Gini, entropy, top-k concentration)
"""

import os
import gc
import re
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict
import torch
import numpy as np
import matplotlib.pyplot as plt

from configs import get_model_config, get_name_filter_for_model
from utils import (
    compute_gini_coefficient,
    compute_entropy_concentration,
    compute_topk_concentration,
    extract_block_index,
)


# Default configuration - can be overridden via command line
MODEL_NAME = "gpt2-medium"
CKPT_DIR = "checkpoints"

# which steps you saved; adjust if different
# Checkpoints are saved every 20 steps, up to step 600
STEPS = list(range(0, 601, 20))   # 0, 20, 40, ..., 600
TAUS = [1e-7, 1e-6, 1e-5]          # Tolerances to analyze


def load_state(step: int, ckpt_dir: str = CKPT_DIR) -> Dict[str, torch.Tensor]:
    """Load checkpoint state dict for a given step."""
    path = os.path.join(ckpt_dir, f"step_{step:03d}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    sd = torch.load(path, map_location="cpu")
    return sd


def collect_tracked_names(
    state_dict: Dict[str, torch.Tensor],
    model_name: str = MODEL_NAME
) -> List[str]:
    """
    Collect parameter names to track, using model-specific name filter.
    Only tracks 2D weight matrices from attention/MLP layers.
    """
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
    return sorted(names)  # sort for consistency

def compute_active_indices(
    base_sd: Dict[str, torch.Tensor],
    cur_sd: Dict[str, torch.Tensor],
    tracked_names: List[str],
    tau: float,
) -> set:
    """
    Compute indices of weights with |ΔW| > tau WITHOUT building the full concatenated vector.
    Returns a set of global indices (accounting for offsets from each layer).
    This is memory-efficient: processes one layer at a time and only stores indices.
    """
    active_indices = set()
    offset = 0

    for name in tracked_names:
        W0 = base_sd[name].to(torch.float32)
        Wt = cur_sd[name].to(torch.float32)
        dW_abs = (Wt - W0).abs().view(-1)  # Flatten to 1D

        # Find active indices in this layer
        mask = (dW_abs > tau)
        local_indices = torch.nonzero(mask, as_tuple=False).view(-1)

        # Convert to global indices by adding offset
        global_indices = (local_indices + offset).tolist()
        active_indices.update(global_indices)

        # Update offset for next layer
        offset += dW_abs.numel()

        # Free memory immediately
        del W0, Wt, dW_abs, mask, local_indices

    return active_indices


def compute_module_concentration(
    base_sd: Dict[str, torch.Tensor],
    cur_sd: Dict[str, torch.Tensor],
    tracked_names: List[str],
    tau: float,
) -> Tuple[Dict[int, int], Dict[int, float]]:
    """
    Group updates by transformer block (layer index).

    Returns:
        block_update_counts: Dict mapping block index to count of updated weights
        block_fro_norms: Dict mapping block index to Frobenius norm of weight delta
    """
    block_update_counts = defaultdict(int)
    block_fro_norms = defaultdict(float)

    for name in tracked_names:
        # Extract block index from parameter name
        block_idx = extract_block_index(name)

        W0 = base_sd[name].to(torch.float32)
        Wt = cur_sd[name].to(torch.float32)
        delta = Wt - W0

        # Count active weights
        num_active = (delta.abs() > tau).sum().item()
        block_update_counts[block_idx] += num_active

        # Compute Frobenius norm
        fro_norm = torch.linalg.norm(delta).item()
        block_fro_norms[block_idx] += fro_norm

        del W0, Wt, delta

    return dict(block_update_counts), dict(block_fro_norms)


def compute_layer_fro_norms(
    base_sd: Dict[str, torch.Tensor],
    cur_sd: Dict[str, torch.Tensor],
    tracked_names: List[str],
) -> Dict[str, float]:
    """
    Compute Frobenius norm of weight delta for each layer.
    Used for localization metrics.
    """
    layer_norms = {}

    for name in tracked_names:
        W0 = base_sd[name].to(torch.float32)
        Wt = cur_sd[name].to(torch.float32)
        delta = Wt - W0
        layer_norms[name] = torch.linalg.norm(delta).item()
        del W0, Wt, delta

    return layer_norms

def main(
    model_name: str = MODEL_NAME,
    ckpt_dir: str = CKPT_DIR,
    steps: List[int] = None,
    taus: List[float] = None,
):
    """
    Main analysis function with configurable parameters.

    Args:
        model_name: Name of the model (for name filter selection)
        ckpt_dir: Directory containing checkpoints
        steps: List of checkpoint steps to analyze
        taus: List of tolerance thresholds
    """
    if steps is None:
        steps = STEPS
    if taus is None:
        taus = TAUS

    print("=" * 60)
    print("Sparse Subnetwork Analysis for Multiple Tolerances")
    print(f"Model: {model_name}")
    print("=" * 60)

    # 1. load base checkpoint (once)
    print(f"\nLoading base checkpoint from {ckpt_dir}...")
    base_sd = load_state(steps[0], ckpt_dir)
    print(f"✓ Loaded base step {steps[0]}")

    # 2. decide which parameters to track
    tracked_names = collect_tracked_names(base_sd, model_name)
    N_total = sum(base_sd[name].numel() for name in tracked_names)
    print(f"✓ Tracking {len(tracked_names)} parameter tensors ({N_total:,} total weights)")
    if len(tracked_names) == 0:
        print("✗ No parameters tracked, exiting.")
        return

    all_results = {}
    localization_over_time = {}

    for tau in taus:
        print(f"\n{'─'*20} Analyzing for tau = {tau:.1e} {'─'*20}")

        # 3. Define final sparse subnetwork for this tau
        final_sd = load_state(steps[-1], ckpt_dir)
        set_final = compute_active_indices(base_sd, final_sd, tracked_names, tau)
        N_final = len(set_final)
        sparsity_final = N_final / N_total if N_total > 0 else 0.0

        print(f"  Final subnetwork size: {N_final:,} weights ({100 * sparsity_final:.4f}%)")

        # Free final state dict before looping
        del final_sd
        gc.collect()

        # 4. Compute coverage, sparsity, and localization for each checkpoint
        coverages = []
        sparsities = []
        gini_scores = []
        entropy_scores = []
        topk_scores = []
        block_norms_over_time = []  # For heatmap

        for step in steps:
            try:
                cur_sd = load_state(step, ckpt_dir)
                set_t = compute_active_indices(base_sd, cur_sd, tracked_names, tau)
                N_t = len(set_t)
                sparsity_t = N_t / N_total if N_total > 0 else 0.0
                inter = len(set_final & set_t)
                coverage_t = inter / N_final if N_final > 0 else 0.0

                sparsities.append(sparsity_t)
                coverages.append(coverage_t)

                # Compute localization metrics
                layer_norms = compute_layer_fro_norms(base_sd, cur_sd, tracked_names)
                gini = compute_gini_coefficient(layer_norms)
                entropy = compute_entropy_concentration(layer_norms)
                topk = compute_topk_concentration(layer_norms, k=3)

                gini_scores.append(gini)
                entropy_scores.append(entropy)
                topk_scores.append(topk)

                # Module concentration for heatmap
                block_counts, block_norms = compute_module_concentration(
                    base_sd, cur_sd, tracked_names, tau
                )
                block_norms_over_time.append(block_norms)

                print(f"    Step {step:3d}: sparsity={sparsity_t:.6f}, coverage={coverage_t:.6f}, "
                      f"gini={gini:.3f}, entropy={entropy:.3f}")

                del cur_sd, set_t
                gc.collect()
            except FileNotFoundError as e:
                print(f"    ⚠ Step {step:3d}: {e}")
                sparsities.append(float('nan'))
                coverages.append(float('nan'))
                gini_scores.append(float('nan'))
                entropy_scores.append(float('nan'))
                topk_scores.append(float('nan'))
                block_norms_over_time.append({})

        all_results[tau] = {
            "coverages": coverages,
            "sparsities": sparsities,
            "gini": gini_scores,
            "entropy": entropy_scores,
            "topk": topk_scores,
            "block_norms": block_norms_over_time,
        }

    # 5. Plot combined curves
    print(f"\n{'='*20} Generating Combined Plots {'='*20}")
    os.makedirs("plots", exist_ok=True)
    steps_np = np.array(steps, dtype=float)
    model_short = model_name.split("/")[-1]

    # Combined coverage curve
    plt.figure(figsize=(10, 6))
    for tau, results in all_results.items():
        plt.plot(steps_np, results["coverages"], marker="o", linestyle="-", markersize=4, label=f"τ = {tau:.0e}")
    plt.xlabel("PPO Step", fontsize=12)
    plt.ylabel("Coverage of Final Subnetwork", fontsize=12)
    plt.title(f"Subnetwork Coverage vs. Training Step ({model_short})", fontsize=14)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.ylim([0, 1.05])
    coverage_path = f"plots/coverage_combined_{model_short}.png"
    plt.savefig(coverage_path, dpi=150)
    print(f"✓ Saved combined coverage plot: {coverage_path}")
    plt.close()

    # Combined density curve
    plt.figure(figsize=(10, 6))
    for tau, results in all_results.items():
        densities_pct = [x * 100 for x in results["sparsities"]]
        plt.plot(steps_np, densities_pct, marker="o", linestyle="-", markersize=4, label=f"τ = {tau:.0e}")
    plt.xlabel("PPO Step", fontsize=12)
    plt.ylabel("Global Density (% of Updated Weights)", fontsize=12)
    plt.title(f"Global Density vs. Training Step ({model_short})", fontsize=14)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    density_path = f"plots/density_combined_{model_short}.png"
    plt.savefig(density_path, dpi=150)
    print(f"✓ Saved combined density plot: {density_path}")
    plt.close()

    # Localization metrics over time (using middle tau)
    tau_mid = taus[len(taus) // 2]
    results_mid = all_results[tau_mid]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Gini coefficient
    axes[0].plot(steps_np, results_mid["gini"], marker="o", linestyle="-", color="blue")
    axes[0].set_xlabel("PPO Step")
    axes[0].set_ylabel("Gini Coefficient")
    axes[0].set_title(f"Update Concentration (Gini)\n{model_short}, τ={tau_mid:.0e}")
    axes[0].grid(True, linestyle="--", alpha=0.7)
    axes[0].set_ylim([0, 1])

    # Entropy
    axes[1].plot(steps_np, results_mid["entropy"], marker="o", linestyle="-", color="green")
    axes[1].set_xlabel("PPO Step")
    axes[1].set_ylabel("Normalized Entropy")
    axes[1].set_title(f"Update Distribution (Entropy)\n{model_short}, τ={tau_mid:.0e}")
    axes[1].grid(True, linestyle="--", alpha=0.7)
    axes[1].set_ylim([0, 1])

    # Top-k concentration
    axes[2].plot(steps_np, results_mid["topk"], marker="o", linestyle="-", color="red")
    axes[2].set_xlabel("PPO Step")
    axes[2].set_ylabel("Top-3 Layer Concentration")
    axes[2].set_title(f"Top-3 Layer Concentration\n{model_short}, τ={tau_mid:.0e}")
    axes[2].grid(True, linestyle="--", alpha=0.7)
    axes[2].set_ylim([0, 1])

    plt.tight_layout()
    localization_path = f"plots/localization_metrics_{model_short}.png"
    plt.savefig(localization_path, dpi=150)
    print(f"✓ Saved localization metrics plot: {localization_path}")
    plt.close()

    # Block-level heatmap
    block_norms_list = results_mid["block_norms"]
    if block_norms_list and any(block_norms_list):
        # Get all unique block indices
        all_blocks = set()
        for bn in block_norms_list:
            all_blocks.update(bn.keys())
        all_blocks = sorted([b for b in all_blocks if b >= 0])  # Exclude -1

        if all_blocks:
            # Build heatmap matrix
            heatmap_data = np.zeros((len(all_blocks), len(steps)))
            for t_idx, bn in enumerate(block_norms_list):
                for b_idx, block in enumerate(all_blocks):
                    heatmap_data[b_idx, t_idx] = bn.get(block, 0.0)

            plt.figure(figsize=(14, 8))
            plt.imshow(heatmap_data, aspect="auto", cmap="hot", interpolation="nearest")
            plt.colorbar(label="Frobenius Norm of ΔW")
            plt.xlabel("PPO Step")
            plt.ylabel("Transformer Block")
            plt.title(f"Weight Update Distribution Across Blocks ({model_short})")
            plt.xticks(range(len(steps)), [str(s) for s in steps], rotation=45)
            plt.yticks(range(len(all_blocks)), [str(b) for b in all_blocks])
            plt.tight_layout()
            heatmap_path = f"plots/block_heatmap_{model_short}.png"
            plt.savefig(heatmap_path, dpi=150)
            print(f"✓ Saved block heatmap: {heatmap_path}")
            plt.close()

    print("\nAnalysis complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sparse Subnetwork Analysis")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help="Model name (e.g., 'gpt2-medium', 'google/gemma-3n-E2B-it')")
    parser.add_argument("--ckpt_dir", type=str, default=CKPT_DIR,
                        help="Directory containing checkpoints")
    parser.add_argument("--max_step", type=int, default=600,
                        help="Maximum step to analyze")
    parser.add_argument("--step_interval", type=int, default=20,
                        help="Interval between checkpoints")

    args = parser.parse_args()

    steps = list(range(0, args.max_step + 1, args.step_interval))
    main(model_name=args.model, ckpt_dir=args.ckpt_dir, steps=steps)

