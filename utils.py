import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple, Dict, List
import re
from collections import defaultdict


# ============================================================================
# LOCALIZATION METRICS - For measuring how concentrated weight updates are
# ============================================================================

def compute_gini_coefficient(layer_updates: Dict[str, float]) -> float:
    """
    Gini coefficient: 0 = uniform distribution, 1 = concentrated in one layer.

    Measures inequality in the distribution of weight updates across layers.
    Higher values indicate more localized/concentrated updates.
    """
    values = np.array(list(layer_updates.values()))
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(values)
    gini = (2 * np.sum((np.arange(1, n + 1) * values)) / (n * np.sum(values))) - (n + 1) / n
    return float(gini)


def compute_entropy_concentration(layer_updates: Dict[str, float]) -> float:
    """
    Entropy: high = distributed, low = concentrated.

    Measures the entropy of the distribution of weight updates across layers.
    Lower entropy indicates more localized updates.

    Returns: Normalized entropy in [0, 1] where 0 is maximally concentrated
             and 1 is uniform distribution.
    """
    values = np.array(list(layer_updates.values()))
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    probs = values / (values.sum() + 1e-10)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    # Normalize by max entropy (uniform distribution)
    max_entropy = np.log(len(values))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def compute_topk_concentration(layer_updates: Dict[str, float], k: int = 3) -> float:
    """
    Fraction of total updates in top-k layers.

    Higher values indicate more concentrated updates in a few layers.
    """
    values = np.array(list(layer_updates.values()))
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    total = values.sum()
    topk = np.sort(values)[-k:].sum()
    return float(topk / (total + 1e-10))


def compute_layer_update_distribution(stats: Dict[str, Dict]) -> Dict[str, float]:
    """
    Extract per-layer Frobenius norm of weight deltas from rank stats.

    Args:
        stats: Dictionary from get_layer_rank_stats()

    Returns:
        Dictionary mapping layer names to their Frobenius norm of updates
    """
    return {name: s["fro_norm"] for name, s in stats.items()}


def extract_block_index(param_name: str) -> int:
    """
    Extract the transformer block/layer index from a parameter name.

    Handles different naming conventions:
    - GPT-2: "transformer.h.5.attn.c_attn.weight" -> 5
    - Gemma-3n: "model.layers.5.mlp.gate_proj.weight" -> 5

    Returns -1 if no block index found.
    """
    # Try various patterns for block indexing
    patterns = [
        r'layers?[._](\d+)',      # model.layers.5 or model.layer.5
        r'\.h\.(\d+)\.',          # transformer.h.5
        r'block[._]?(\d+)',       # block5 or block.5
        r'encoder[._](\d+)',      # encoder.5
        r'decoder[._](\d+)',      # decoder.5
    ]

    for pattern in patterns:
        match = re.search(pattern, param_name)
        if match:
            return int(match.group(1))
    return -1


def extract_module_type(param_name: str) -> str:
    """
    Extract the module type from a parameter name.

    Returns one of: 'attention', 'mlp', 'embedding', 'lm_head', 'other'
    """
    name_lower = param_name.lower()

    if any(x in name_lower for x in ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'c_attn']):
        return 'attention'
    elif any(x in name_lower for x in ['mlp', 'fc', 'gate_proj', 'up_proj', 'down_proj', 'c_fc', 'c_proj']):
        return 'mlp'
    elif 'embed' in name_lower:
        return 'embedding'
    elif 'lm_head' in name_lower:
        return 'lm_head'
    else:
        return 'other'


def compute_block_level_updates(
    stats: Dict[str, Dict],
    group_by: str = 'block'
) -> Dict[int, float]:
    """
    Group weight updates by transformer block index.

    Args:
        stats: Dictionary from get_layer_rank_stats()
        group_by: 'block' for block index, 'module' for module type

    Returns:
        Dictionary mapping block index to total Frobenius norm of updates
    """
    block_updates = defaultdict(float)

    for name, s in stats.items():
        if group_by == 'block':
            key = extract_block_index(name)
        else:
            key = extract_module_type(name)
        block_updates[key] += s["fro_norm"]

    return dict(block_updates)


def compute_localization_summary(stats: Dict[str, Dict], k: int = 3) -> Dict[str, float]:
    """
    Compute all localization metrics from layer rank stats.

    Returns a dictionary with:
    - gini: Gini coefficient of update distribution
    - entropy: Normalized entropy (lower = more concentrated)
    - topk_concentration: Fraction of updates in top-k layers
    - num_active_layers: Number of layers with non-zero updates
    """
    layer_updates = compute_layer_update_distribution(stats)

    return {
        "gini": compute_gini_coefficient(layer_updates),
        "entropy": compute_entropy_concentration(layer_updates),
        "topk_concentration": compute_topk_concentration(layer_updates, k=k),
        "num_active_layers": sum(1 for v in layer_updates.values() if v > 1e-10),
        "total_layers": len(layer_updates),
    }


def compute_per_block_stats(stats: Dict[str, Dict]) -> Dict[int, Dict[str, float]]:
    """
    Compute aggregated statistics per transformer block.

    Returns dictionary mapping block index to:
    - total_fro_norm: Sum of Frobenius norms
    - mean_percent_updated: Average percent of weights updated
    - num_params: Number of parameter tensors in this block
    """
    block_stats = defaultdict(lambda: {
        'total_fro_norm': 0.0,
        'percent_updated_sum': 0.0,
        'num_params': 0
    })

    for name, s in stats.items():
        block_idx = extract_block_index(name)
        block_stats[block_idx]['total_fro_norm'] += s["fro_norm"]
        block_stats[block_idx]['percent_updated_sum'] += s["percent_updated"]
        block_stats[block_idx]['num_params'] += 1

    # Compute means
    result = {}
    for block_idx, data in block_stats.items():
        result[block_idx] = {
            'total_fro_norm': data['total_fro_norm'],
            'mean_percent_updated': data['percent_updated_sum'] / data['num_params'] if data['num_params'] > 0 else 0,
            'num_params': data['num_params']
        }

    return dict(result)

def get_rank(matrix : torch.Tensor) -> int:
    return torch.linalg.matrix_rank(matrix).item()

    

def nuclear_rank(matrix: torch.Tensor) -> float:
    """ This is just something I made up tbh, but it is supposed to be a 'normalized' nuclear norm. """
    return torch.sum(torch.linalg.svdvals(matrix))/torch.linalg.norm(matrix, ord=2)



def get_effective_rank(matrix: torch.Tensor, stability_eps: float = 1e-10) -> float:
    """ The effective rank of a matrix A is obtained by first calculating the singular 
    values sigma_1, .. , sigma_n and making the distribution p_i = sigma_i / sum_j sigma_j.
     The effective rank is then exp(entropy(p)).
     
    See this paper https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf for more details. 
    """
    matrix = matrix.to(torch.float32)
    singular_values, _ = torch.sort(torch.linalg.svdvals(matrix), descending=True)
    dist = singular_values / torch.sum(singular_values)
    mask = dist > stability_eps # if s_i < eps then s_i*log(s_i) is set to 0
    entropy = -torch.sum(dist[mask] * torch.log(dist[mask]))
    return torch.exp(entropy).item()




def get_stable_rank(matrix: torch.Tensor, stability_eps: float = 1e-10) -> float:
    """ The stable rank of a matrix A is obtained by calculating (sum s_i^2)/ (max_i s_i^2).
    Here s_i are the singular values of A. Note that sum s_i^2 = ||A||_F^2 so we can just 
    compute the sum of the squared entries of the matrix."""
    mat32 = matrix.to(torch.float32)
    fro_sq = torch.sum(mat32 * mat32)
    spec = torch.linalg.norm(mat32, ord=2)
    return float(fro_sq / (spec * spec + stability_eps))




@torch.no_grad()
def get_layer_rank_stats(model: nn.Module, 
                        init_state: dict[str, torch.Tensor], 
                        name_filter: tuple[str, ...] = ("attn", "mlp", "c_attn", "c_proj", "c_fc", "lm_head"),
                        tolerance: float = 1e-5) -> dict[str, dict]:
    stats = {} # of type Dict[str, Dict[str, float]] where the outer key is the layer name and the inner key is the stat name.
    for name, param in model.named_parameters():
        if not name.endswith("weight"):
            continue
        if not any(substring in name for substring in name_filter):
            continue
        if param.dim() != 2: # skipping layer norm and bias parameters
            continue
        if name not in init_state:
            continue
        W_0 = init_state[name]
        W = param.detach().cpu()

        dW = W - W_0
        classical_rank = torch.linalg.matrix_rank(dW.to(torch.float32)).item()
        eff_rank = get_effective_rank(dW) 
        stable_rank = get_stable_rank(dW)
        fro_norm = float(torch.linalg.norm(dW.to(torch.float32)))

        flat_dW = dW.view(-1).to(torch.float32)
        abs_flat_dW = torch.abs(flat_dW)
        mean_abs = torch.mean(abs_flat_dW)
        std_abs = torch.std(abs_flat_dW)
        thresh = mean_abs + 2 * std_abs
        frac_large = torch.mean((abs_flat_dW > thresh).float()).item()
        percent_updated = torch.mean((abs_flat_dW > tolerance).float()).item()

        stats[name] = {"shape": dW.shape, "classical_rank": classical_rank, "eff_rank": eff_rank, "stable_rank": stable_rank, 
        "fro_norm": fro_norm, "frac_large": frac_large, "percent_updated": percent_updated}


    return stats


def experiment_1(n=1024):
    """
    Runs an experiment to compare rank measures for random n x n matrices.
    :param n: The dimension of the square matrices to generate.
    """
    # Automatically select GPU if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running experiment for {n}x{n} matrices on {device.type} ---")
    
    samples = 100
    
    # Move matrix creation and computation to the selected device
    matrix_list = [torch.randn(n, n, device=device) for _ in range(samples)]
    
    effective_rank_list = [get_effective_rank(matrix) for matrix in matrix_list]
    classic_rank_list = [get_rank(matrix) for matrix in matrix_list]
    stable_rank_list = [get_stable_rank(matrix) for matrix in matrix_list]

    # Calculate and print the mean of each rank type
    mean_effective = sum(effective_rank_list) / len(effective_rank_list)
    mean_stable = sum(stable_rank_list) / len(stable_rank_list)
    mean_classic = sum(classic_rank_list) / len(classic_rank_list)

    # --- Calculate and print the standard deviation ---
    # Convert lists to tensors, ensuring they have a float dtype for .std()
    effective_tensor = torch.tensor(effective_rank_list, dtype=torch.float32)
    stable_tensor = torch.tensor(stable_rank_list, dtype=torch.float32)
    classic_tensor = torch.tensor(classic_rank_list, dtype=torch.float32)

    std_effective = effective_tensor.std().item()
    std_stable = stable_tensor.std().item()
    std_classic = classic_tensor.std().item()

    print(f"\nMean Effective Rank: {mean_effective:.2f} (Std Dev: {std_effective:.4f})")
    print(f"Mean Stable Rank   : {mean_stable:.2f} (Std Dev: {std_stable:.4f})")
    print(f"Mean Classic Rank  : {mean_classic:.2f} (Std Dev: {std_classic:.4f})\n")
    
    # Plot Histograms (as you have it now)
    plt.hist(effective_rank_list, bins=10, label="Effective Rank", alpha=0.7)
    plt.hist(stable_rank_list, bins=10, label="Stable Rank", alpha=0.7)
    
    # --- Manually create a thick bar for the classical rank ---
    bar_start = n - 1
    bar_end = n
    bar_width = bar_end - bar_start
    bar_center = bar_start + (bar_width / 2)
    
    # The height is the number of samples, which is 100
    count_of_classical_rank = len(classic_rank_list) 
    
    plt.bar(bar_center, count_of_classical_rank, width=bar_width, 
            color='green', alpha=0.7, label=f"Classic Rank = {classic_rank_list[0]}")
    
    # --- Add the means as custom ticks on the X-axis ---
    # Get the current ticks
    current_ticks = plt.xticks()[0]
    
    # Combine current ticks with the calculated means, then sort
    new_ticks = sorted(list(set(current_ticks.tolist() + [mean_effective, mean_stable, mean_classic])))
    
    # Set the new ticks on the x-axis
    plt.xticks(ticks=new_ticks, rotation=45) # Rotate for better readability if they overlap
    
    plt.ylim(top=max(plt.ylim()[1], count_of_classical_rank * 1.05))

    # We can remove the axvline calls and just have a clean legend
    plt.legend()
    plt.title(f"Rank Distributions for {n}x{n} Full-Rank Random Matrices")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.show()      

if __name__ == "__main__":
    experiment_1()