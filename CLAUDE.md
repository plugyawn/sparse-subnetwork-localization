# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research mono-repository investigating sparse subnetworks in LLM fine-tuning and reverse-engineering Google's Gemma-3n edge-optimized model.

**Current Hypothesis**: Gemma-3n's MatFormer architecture (with nested representations and Mix-n-Match layers) will exhibit MORE localized sparse subnetworks during RL fine-tuning compared to standard transformers, due to its nested FFN structure causing RL to preferentially update weights in specific nested "shells".

Three sub-projects:
- **reverse-engineering-gemma-3n**: Extracts and analyzes TFLite model to identify architectural innovations (MatFormer, LAuReL, AltUp, per-layer embeddings)
- **SparseLLM-RL**: PPO-based RL fine-tuning tracking sparse weight updates (supports GPT-2 and Gemma 3/3n)
- **sparsity_in_rl**: Comparative sparsity analysis between SFT and RL fine-tuned models

## Commands

### SparseLLM-RL

```bash
# Setup (CRITICAL: TRL must be version 0.8.6)
conda create -n sparsellm-rl python=3.10
conda activate sparsellm-rl
pip install --no-upgrade -r SparseLLM-RL/requirements.txt

# Train PPO model (saves checkpoints every 20 steps to checkpoints/)
# Default: GPT-2 medium. Edit policy_name in main_ppo.py for Gemma models.
python SparseLLM-RL/main_ppo.py

# Analyze sparse subnetworks from checkpoints
python SparseLLM-RL/analyze_sparse_subnetwork.py --model gpt2-medium --ckpt_dir checkpoints

# Comparative analysis between models (GPT-2 vs Gemma-3n)
python SparseLLM-RL/localization_analysis.py \
  --models gpt2-medium google/gemma-3n-E4B-it \
  --ckpt_dirs checkpoints_gpt2 checkpoints_gemma3n \
  --output_dir plots/comparative

# Generate rank visualizations
cd SparseLLM-RL/plots && python layer_ranks.py
```

### sparsity_in_rl

```bash
python sparsity_in_rl/src/check_sparsity.py \
  --sft_model <MODEL_NAME> \
  --rl_model <MODEL_NAME> \
  --tolerances 1e-5
```

### reverse-engineering-gemma-3n

```bash
python reverse-engineering-gemma-3n/parse.py      # Parse TFLite operations
python reverse-engineering-gemma-3n/dumptensor.py # Extract tensors
```

## Architecture

### SparseLLM-RL Training Pipeline

- **Policy**: GPT-2-medium or Gemma 3/3n (fine-tuned via PPO)
- **Reward Model**: lvwerra/distilbert-imdb (frozen sentiment classifier)
- **Reference**: Same as policy (frozen, for KL penalty)
- **Data**: IMDB reviews as prompts

Training loop: sample prompts → generate responses → score with reward model → normalize rewards → PPO step → save checkpoint → log rank + localization statistics

Key config in `main_ppo.py`:
```python
@dataclass
class ExpConfig:
    policy_name: str = "gpt2-medium"  # or "google/gemma-3n-E4B-it"
    batch_size: int = 32
    num_updates: int = 600
    save_ckpt_every: int = 20
    tau: float = 1e-6  # Weight update threshold
    log_localization: bool = True  # Track Gini, entropy, top-k concentration
```

### Supported Models

| Model | Architecture | HuggingFace ID | Notes |
|-------|-------------|----------------|-------|
| GPT-2 Medium | Standard Transformer | `gpt2-medium` | Baseline, proven TRL compatibility |
| Gemma 3 4B | Standard Transformer | `google/gemma-3-4b-it` | Requires TRL patch |
| Gemma 3n E2B | MatFormer (nested) | `google/gemma-3n-E2B-it` | ~2B effective params, requires TRL patch |
| Gemma 3n E4B | MatFormer (nested) | `google/gemma-3n-E4B-it` | ~4B effective params, requires TRL patch |

### Sparse Subnetwork Tracking

Tracks which weights change during training:
- **Classical rank**: Linear independent components
- **Effective rank**: Entropy-based (captures singular value distribution)
- **Stable rank**: ||A||_F² / ||A||_2² (numerically robust)
- **Coverage**: Proportion of final active weights present at earlier steps

### Localization Metrics

Measures how concentrated weight updates are across layers:
- **Gini coefficient**: 0 = uniform, 1 = concentrated in one layer (higher = more localized)
- **Entropy**: Normalized entropy of update distribution (lower = more localized)
- **Top-k concentration**: Fraction of total updates in top-k layers (higher = more localized)
- **Block-level heatmaps**: Visual distribution of updates across transformer blocks

### Gemma-3n Architecture (Reconstructed)

4 TFLite subgraphs identified:
- `TF_LITE_EMBEDDER`: Token→embedding (vocab=262K, dim=2048)
- `TF_LITE_PER_LAYER_EMBEDDER`: Per-layer embeddings (256 dim × 30 layers)
- `TF_LITE_PREFILL_DECODE`: Main transformer (3895 tensors, 3009 ops)
- `TF_LITE_VISION_ENCODER`: MobileNetV4-based vision processor

Key innovation: Per-layer embeddings allow loading token-specific parameters on-demand from flash storage (~1GB model accessed 4KB at a time).

## Critical Constraints

1. **DO NOT UPGRADE TRL**: SparseLLM-RL requires TRL 0.8.6 (breaking API changes in 0.9+)
2. **GPU Required**: Default config uses `device="cuda"` (modify ExpConfig for CPU)
3. **W&B Logging**: Enabled by default; set `use_wandb=False` in ExpConfig to disable
4. **Large Files**: `tflite_opcode_dump.txt` is 91K lines; use grep/head selectively

## TRL + Gemma 3/3n Compatibility

Gemma 3/3n models have a known TRL compatibility issue ([Issue #3828](https://github.com/huggingface/trl/issues/3828)):
- `AutoModelForCausalLMWithValueHead` fails with `UnboundLocalError: hidden_size`
- Gemma 3 stores config in `text_config.hidden_size` instead of `hidden_size`

**Solution**: The patch in `SparseLLM-RL/configs/trl_gemma_patch.py` is automatically applied when importing `main_ppo.py`. For standalone scripts:

```python
from configs.trl_gemma_patch import patch_trl_for_gemma
patch_trl_for_gemma()  # Must be called BEFORE importing TRL models

from trl import AutoModelForCausalLMWithValueHead
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "google/gemma-3n-E4B-it",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
```

## Key Files

```
SparseLLM-RL/
├── main_ppo.py                    # Main training script
├── utils.py                       # Rank metrics + localization metrics
├── analyze_sparse_subnetwork.py   # Post-training analysis
├── localization_analysis.py       # Comparative analysis (GPT-2 vs Gemma)
└── configs/
    ├── __init__.py
    ├── gemma3n_config.py          # Model configs, name filters, MatFormer analyzer
    └── trl_gemma_patch.py         # TRL compatibility patch for Gemma
```
