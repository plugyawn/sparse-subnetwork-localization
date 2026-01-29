"""
Gemma-3n specific configuration for sparse subnetwork analysis.

Gemma-3n uses MatFormer architecture with nested representations and Mix-n-Match layers.
This module provides configuration for:
- Model loading and processor setup
- Layer name filters for Gemma architecture
- Nesting level extraction for MatFormer analysis
"""

from dataclasses import dataclass
from typing import Tuple, Dict, List
import re


# Available Gemma-3n models
GEMMA3N_MODELS = {
    "E2B": "google/gemma-3n-E2B-it",  # ~2B effective params
    "E4B": "google/gemma-3n-E4B-it",  # ~4B effective params
}


# Layer name filters for Gemma-3/Gemma-3n architecture
# These are the parameter names that contain trainable weights we want to track
GEMMA_NAME_FILTER = (
    "self_attn",
    "mlp",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
)

# GPT-2 layer name filter (for comparison)
GPT2_NAME_FILTER = (
    "attn",
    "mlp",
    "c_attn",
    "c_proj",
    "c_fc",
    "lm_head",
)


@dataclass
class ModelConfig:
    """Configuration for a specific model type."""
    name: str
    hf_id: str
    name_filter: Tuple[str, ...]
    use_processor: bool  # True for multimodal models like Gemma-3n
    is_matformer: bool
    num_layers: int
    hidden_dim: int


# Pre-defined model configurations
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "gpt2-medium": ModelConfig(
        name="gpt2-medium",
        hf_id="gpt2-medium",
        name_filter=GPT2_NAME_FILTER,
        use_processor=False,
        is_matformer=False,
        num_layers=24,
        hidden_dim=1024,
    ),
    "gemma-3n-E2B": ModelConfig(
        name="gemma-3n-E2B",
        hf_id="google/gemma-3n-E2B-it",
        name_filter=GEMMA_NAME_FILTER,
        use_processor=True,
        is_matformer=True,
        num_layers=26,  # Approximate - verify from model config
        hidden_dim=2048,
    ),
    "gemma-3-4b": ModelConfig(
        name="gemma-3-4b",
        hf_id="google/gemma-3-4b-it",
        name_filter=GEMMA_NAME_FILTER,
        use_processor=False,
        is_matformer=False,
        num_layers=0,  # Populate from model config at runtime if needed
        hidden_dim=0,
    ),
    "gemma-3n-E4B": ModelConfig(
        name="gemma-3n-E4B",
        hf_id="google/gemma-3n-E4B-it",
        name_filter=GEMMA_NAME_FILTER,
        use_processor=True,
        is_matformer=True,
        num_layers=30,  # Approximate - verify from model config
        hidden_dim=2304,
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """
    Get configuration for a model by name or HuggingFace ID.

    Args:
        model_name: Either a short name (e.g., "gpt2-medium") or HF ID

    Returns:
        ModelConfig for the specified model
    """
    # Check direct match
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]

    # Check HF ID match
    for config in MODEL_CONFIGS.values():
        if config.hf_id == model_name:
            return config

    name_lower = model_name.lower()
    if "gemma-3n" in name_lower:
        return ModelConfig(
            name=model_name,
            hf_id=model_name,
            name_filter=GEMMA_NAME_FILTER,
            use_processor=True,
            is_matformer=True,
            num_layers=0,
            hidden_dim=0,
        )
    if "gemma-3" in name_lower:
        return ModelConfig(
            name=model_name,
            hf_id=model_name,
            name_filter=GEMMA_NAME_FILTER,
            use_processor=False,
            is_matformer=False,
            num_layers=0,
            hidden_dim=0,
        )

    # Default to GPT-2 style config for unknown models
    return ModelConfig(
        name=model_name,
        hf_id=model_name,
        name_filter=GPT2_NAME_FILTER,
        use_processor=False,
        is_matformer=False,
        num_layers=24,
        hidden_dim=1024,
    )


def get_name_filter_for_model(model_name: str) -> Tuple[str, ...]:
    """Get the appropriate layer name filter for a model."""
    config = get_model_config(model_name)
    return config.name_filter


class MatFormerNestingAnalyzer:
    """
    Analyzer for MatFormer's nested representation structure.

    MatFormer has nested FFN blocks where smaller submodels are contained within
    larger ones. This class helps identify which "nesting level" or "shell"
    a weight belongs to.
    """

    def __init__(self, hidden_dim: int, num_nesting_levels: int = 4):
        """
        Initialize the nesting analyzer.

        Args:
            hidden_dim: The full hidden dimension of the model
            num_nesting_levels: Number of nesting levels (typically 4 for MatFormer)
        """
        self.hidden_dim = hidden_dim
        self.num_nesting_levels = num_nesting_levels

        # Compute dimension boundaries for each nesting level
        # In MatFormer, each level contains approximately hidden_dim / num_levels dimensions
        self.level_boundaries = []
        step = hidden_dim // num_nesting_levels
        for i in range(num_nesting_levels):
            start = i * step
            end = (i + 1) * step if i < num_nesting_levels - 1 else hidden_dim
            self.level_boundaries.append((start, end))

    def get_nesting_level(self, dim_index: int) -> int:
        """
        Get the nesting level for a given dimension index.

        Returns:
            0 = innermost (smallest submodel)
            num_nesting_levels - 1 = outermost (full model)
        """
        for level, (start, end) in enumerate(self.level_boundaries):
            if start <= dim_index < end:
                return level
        return self.num_nesting_levels - 1

    def compute_level_update_distribution(
        self,
        weight_delta: "torch.Tensor",  # noqa: F821
        tau: float = 1e-6
    ) -> Dict[int, int]:
        """
        Compute how many weight updates fall into each nesting level.

        For MLP layers, analyzes the distribution of updates across the
        hidden dimension to see if updates cluster in inner or outer shells.

        Args:
            weight_delta: The change in weights (shape: [out_dim, in_dim] or similar)
            tau: Threshold for considering a weight "updated"

        Returns:
            Dictionary mapping nesting level to count of updated weights
        """
        import torch

        level_counts = {i: 0 for i in range(self.num_nesting_levels)}

        if weight_delta.dim() != 2:
            return level_counts

        # Analyze updates along the hidden dimension
        abs_delta = weight_delta.abs()

        # For each output dimension, check if it's updated
        for dim_idx in range(min(weight_delta.shape[0], self.hidden_dim)):
            row_max = abs_delta[dim_idx].max().item()
            if row_max > tau:
                level = self.get_nesting_level(dim_idx)
                level_counts[level] += 1

        return level_counts


def analyze_matformer_nesting(
    base_state: Dict[str, "torch.Tensor"],  # noqa: F821
    current_state: Dict[str, "torch.Tensor"],  # noqa: F821
    hidden_dim: int,
    tau: float = 1e-6
) -> Dict[str, Dict[int, int]]:
    """
    Analyze weight updates across MatFormer nesting levels for all MLP layers.

    Args:
        base_state: Initial model state dict
        current_state: Current model state dict
        hidden_dim: Model's hidden dimension
        tau: Threshold for considering a weight updated

    Returns:
        Dictionary mapping layer name to nesting level distribution
    """
    import torch

    analyzer = MatFormerNestingAnalyzer(hidden_dim)
    results = {}

    for name in base_state.keys():
        if 'mlp' not in name.lower() and 'fc' not in name.lower():
            continue
        if not name.endswith('weight'):
            continue
        if name not in current_state:
            continue

        W0 = base_state[name]
        Wt = current_state[name]

        if W0.dim() != 2:
            continue

        dW = (Wt.to(torch.float32) - W0.to(torch.float32))
        level_dist = analyzer.compute_level_update_distribution(dW, tau)
        results[name] = level_dist

    return results
