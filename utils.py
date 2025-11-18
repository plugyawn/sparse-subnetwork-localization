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

        stats[name] = {"shape": dW.shape, "eff_rank": eff_rank, "stable_rank": stable_rank, 
        "fro_norm": fro_norm, "frac_large": frac_large, "percent_updated": percent_updated}


    return stats


def experiment_1():
    samples = 50000
    matrix_list = [torch.randn(20, 20) for _ in range(samples)]
    effective_rank_list = [get_effective_rank(matrix) for matrix in matrix_list]
    classic_rank_list = [get_rank(matrix) for matrix in matrix_list]
    stable_rank_list = [get_stable_rank(matrix) for matrix in matrix_list]
    nuclear_rank_list = [nuclear_rank(matrix) for matrix in matrix_list]
    plt.hist(effective_rank_list, bins=20, label="Effective Rank", alpha=0.7)
    plt.hist(stable_rank_list, bins=20, label="Stable Rank", alpha=0.7)
    plt.hist(classic_rank_list, bins=20, label="Classic Rank", alpha=0.7)
    plt.hist(nuclear_rank_list, bins=20, label="Nuclear Rank", alpha=0.7)
    plt.legend()
    plt.title("Effective Rank vs Stable Rank vs Classic Rank vs Nuclear Rank")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.show()      

if __name__ == "__main__":
    experiment_1()