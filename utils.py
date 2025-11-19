import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple, Dict

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