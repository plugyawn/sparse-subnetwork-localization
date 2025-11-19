import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration (shared between plotting functions) ---

# Theoretical mean ranks for different shapes based on layer index
# (stable_rank, effective_rank)
THEORETICAL_RANKS = {
    0: (455, 989),    # For mlp.c_fc.weight (1024x4096)
    1: (412, 977),    # For attn.c_attn.weight (1024x3072)
    2: (256, 824)     # For attn.c_proj.weight (1024x1024)
}

# Define layer mapping (layer number -> (layer name, filename component))
LAYER_INFO = {
    0: ("mlp.c_fc.weight", "mlpfc"),
    1: ("attn.c_attn.weight", "attnweight"),
    2: ("attn.c_proj.weight", "proj")
}

CSV_DIR = Path(__file__).parent / "csvwandb"

# --- Plotting Functions ---

def plot_all_layer_ranks():
    """Generates a 3-subplot figure showing all ranks for all 3 layers."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    rank_types = ["classical", "effective", "stable"]

    for layer_idx in range(3):
        ax = axes[layer_idx]
        layer_name, filename_component = LAYER_INFO[layer_idx]
        
        for rank_type in rank_types:
            filename = f"{rank_type}_attn_{layer_idx}_{filename_component}.csv"
            filepath = CSV_DIR / filename
            
            if filepath.exists():
                df = pd.read_csv(filepath)
                rank_col = [col for col in df.columns if rank_type in col and "__MIN" not in col and "__MAX" not in col]
                if rank_col:
                    ax.plot(df["Step"], df[rank_col[0]], label=rank_type.capitalize(), marker='o', markersize=4)

        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Rank", fontsize=12)
        ax.set_title(f"{layer_name}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

        stable_rank, effective_rank = THEORETICAL_RANKS[layer_idx]
        ax.axhline(y=stable_rank, color='green', linestyle='--', label=f'Expected Stable Rank ({stable_rank})')
        ax.axhline(y=effective_rank, color='orange', linestyle='--', label=f'Expected Effective Rank ({effective_rank})')
        ax.legend(fontsize=10, framealpha=0.2)

    plt.tight_layout()
    plt.savefig(CSV_DIR.parent / "layer_ranks.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved to {CSV_DIR.parent / 'layer_ranks.png'}")
    plt.show()


def plot_mlp_effective_rank():
    """Generates a single plot for the Effective Rank of the MLP FC layer."""
    layer_idx = 0  # MLP FC layer
    rank_type = "effective"
    
    layer_name, filename_component = LAYER_INFO[layer_idx]
    filename = f"{rank_type}_attn_{layer_idx}_{filename_component}.csv"
    filepath = CSV_DIR / filename

    if not filepath.exists():
        print(f"Error: Could not find required file {filepath}")
        return

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot data
    df = pd.read_csv(filepath)
    rank_col = [col for col in df.columns if rank_type in col and "__MIN" not in col and "__MAX" not in col][0]
    ax.plot(df["Step"], df[rank_col], label="Effective Rank", marker='o', markersize=5, color='orange')

    # Plot horizontal line for expected rank
    _, effective_rank = THEORETICAL_RANKS[layer_idx]
    ax.axhline(y=effective_rank, color='red', linestyle='--', label=f'Expected Effective Rank ({effective_rank})')

    # Customize plot
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Effective Rank", fontsize=12)
    ax.set_title(f"Effective Rank of {layer_name}", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=900, top=1050) # Scale Y-axis to see detail
    ax.legend(fontsize=10, framealpha=0.5)

    # Save and show
    plt.tight_layout()
    save_path = CSV_DIR.parent / "mlp_effective_rank.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()


def plot_attn_proj_effective_rank():
    """Generates a single plot for the Effective Rank of the Attention Projection layer."""
    layer_idx = 2  # Attention Projection layer
    rank_type = "effective"
    
    layer_name, filename_component = LAYER_INFO[layer_idx]
    filename = f"{rank_type}_attn_{layer_idx}_{filename_component}.csv"
    filepath = CSV_DIR / filename

    if not filepath.exists():
        print(f"Error: Could not find required file {filepath}")
        return

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot data
    df = pd.read_csv(filepath)
    rank_col = [col for col in df.columns if rank_type in col and "__MIN" not in col and "__MAX" not in col][0]
    ax.plot(df["Step"], df[rank_col], label="Effective Rank", marker='o', markersize=5, color='orange')

    # Plot horizontal line for expected rank
    _, effective_rank = THEORETICAL_RANKS[layer_idx]
    ax.axhline(y=effective_rank, color='red', linestyle='--', label=f'Expected Effective Rank ({effective_rank})')

    # Customize plot
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Effective Rank", fontsize=12)
    ax.set_title(f"Effective Rank of {layer_name}", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=700) # Scale Y-axis to see detail from the start
    ax.legend(fontsize=10, framealpha=0.5)

    # Save and show
    plt.tight_layout()
    save_path = CSV_DIR.parent / "attn_proj_effective_rank.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()


def plot_stable_ranks():
    """Generates side-by-side plots for Stable Rank of MLP FC and Attention Projection layers."""
    rank_type = "stable"
    layer_indices = [0, 2]  # mlp.c_fc.weight and attn.c_proj.weight
    
    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, layer_idx in enumerate(layer_indices):
        ax = axes[idx]
        layer_name, filename_component = LAYER_INFO[layer_idx]
        filename = f"{rank_type}_attn_{layer_idx}_{filename_component}.csv"
        filepath = CSV_DIR / filename
        
        if not filepath.exists():
            print(f"Error: Could not find required file {filepath}")
            continue
        
        # Plot data
        df = pd.read_csv(filepath)
        rank_col = [col for col in df.columns if rank_type in col and "__MIN" not in col and "__MAX" not in col]
        if rank_col:
            ax.plot(df["Step"], df[rank_col[0]], label="Stable Rank", marker='o', markersize=5, color='green')
        
        # Plot horizontal line for expected stable rank
        stable_rank, _ = THEORETICAL_RANKS[layer_idx]
        ax.axhline(y=stable_rank, color='green', linestyle='--', label=f'Expected Stable Rank ({stable_rank})')
        
        # Customize plot
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Stable Rank", fontsize=12)
        ax.set_title(f"Stable Rank of {layer_name}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.legend(fontsize=10, framealpha=0.5)
    
    # Save and show
    plt.tight_layout()
    save_path = CSV_DIR.parent / "stable_ranks.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    plot_stable_ranks()
    # plot_attn_proj_effective_rank()
    # plot_mlp_effective_rank()
    # plot_all_layer_ranks()

