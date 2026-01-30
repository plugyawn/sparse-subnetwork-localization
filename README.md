# Sparse Subnetwork Localization in LLM Fine-Tuning

Investigating whether **MatFormer architectures** (like Gemma-3n) exhibit more localized sparse subnetworks during RL fine-tuning compared to standard transformers (GPT-2).

## Hypothesis

Gemma-3n's MatFormer architecture—with nested FFN blocks where smaller submodels are explicitly contained within larger ones—will show **more concentrated weight updates** during PPO fine-tuning than GPT-2. This structural constraint may cause RL to preferentially update weights in specific nested "shells" rather than distributing updates uniformly.

## Key Metrics

| Metric | Interpretation |
|--------|----------------|
| **Gini Coefficient** | 0 = uniform updates, 1 = concentrated in one layer |
| **Normalized Entropy** | Lower = more localized updates |
| **Top-k Concentration** | Fraction of total updates in top-k layers |
| **Block Heatmaps** | Visual distribution of updates across transformer blocks |

## Supported Models

| Model | Params | HuggingFace ID |
|-------|--------|----------------|
| GPT-2 Medium | 355M | `gpt2-medium` |
| Gemma-3n E2B | ~2B effective | `google/gemma-3n-E2B-it` |
| Gemma-3n E4B | ~4B effective | `google/gemma-3n-E4B-it` |

## Installation

```bash
# Create environment (CRITICAL: TRL must be version 0.8.6)
conda create -n sparse-localization python=3.10
conda activate sparse-localization
pip install -r requirements.txt
```

## Usage

### 0. GRPO (CPU-friendly smoke test)

Minimal GRPO runner that works without TRL value heads and can run a 1-step CPU smoke test:

```bash
# CPU smoke test (tiny Gemma random weights)
python main_grpo.py --smoke
```

For real runs, pass a Gemma-3/Gemma-3n model and a reward model:

```bash
python main_grpo.py \
  --policy google/gemma-3-4b-it \
  --reward_model lvwerra/distilbert-imdb \
  --dataset imdb \
  --batch_size 2 \
  --group_size 2 \
  --max_steps 5 \
  --eval_every 1
```

Short-run preset (good for quick GPU sanity checks, with checkpointing). On Gemma-3 / Gemma-3n this enables Adafactor + gradient checkpointing by default:

```bash
python main_grpo.py \
  --policy google/gemma-3n-E2B-it \
  --reward_model lvwerra/distilbert-imdb \
  --dataset imdb \
  --device cuda \
  --dtype bfloat16 \
  --short
```

You can reduce update variance with larger groups or advantage clipping:

```bash
python main_grpo.py --policy google/gemma-3n-E2B-it --group_size 4 --adv_clip 5.0 --short
```

If you OOM with full Adam, try Adafactor:

```bash
python main_grpo.py \
  --policy google/gemma-3n-E4B-it \
  --reward_model lvwerra/distilbert-imdb \
  --dataset imdb \
  --device cuda \
  --dtype bfloat16 \
  --optimizer adafactor \
  --gradient_checkpointing \
  --short
```

If you're still OOM, try offloading the ref/reward model and shortening sequences:

```bash
python main_grpo.py \
  --policy google/gemma-3n-E2B-it \
  --reward_model lvwerra/distilbert-imdb \
  --reward_device cpu \
  --ref_device cpu \
  --max_prompt_tokens 96 \
  --max_new_tokens 32 \
  --device cuda \
  --dtype bfloat16 \
  --short
```

Easy GRPO sanity check (dummy prompts + heuristic reward, no reward model):

```bash
python main_grpo.py --easy --policy google/gemma-3n-E2B-it --device cuda --dtype bfloat16
```

Disable the progress bar if you want clean logs:

```bash
python main_grpo.py --no_tqdm ...
```

### 1. Train with PPO (generates checkpoints)

```bash
# GPT-2 Medium (default)
python main_ppo.py

# For Gemma-3n, edit main_ppo.py:
#   policy_name: str = "google/gemma-3n-E2B-it"
python main_ppo.py
```

Training saves checkpoints every 20 steps to `checkpoints/` and logs localization metrics to W&B.

### 2. Analyze Sparse Subnetworks

```bash
# Single model analysis
python analyze_sparse_subnetwork.py --model gpt2-medium --ckpt_dir checkpoints

# With custom parameters
python analyze_sparse_subnetwork.py \
  --model google/gemma-3n-E2B-it \
  --ckpt_dir checkpoints_gemma \
  --max_step 600 \
  --step_interval 20
```

Generates:
- Coverage curves (how early does the final subnetwork emerge)
- Density curves (fraction of weights updated over time)
- Localization metrics (Gini, entropy, top-k)
- Block-level heatmaps

### 3. Comparative Analysis (GPT-2 vs Gemma-3n)

```bash
python localization_analysis.py \
  --models gpt2-medium google/gemma-3n-E2B-it \
  --ckpt_dirs checkpoints_gpt2 checkpoints_gemma \
  --output_dir plots/comparative
```

Generates side-by-side comparisons and statistical summaries.

## Project Structure

```
.
├── main_ppo.py                 # PPO training with localization logging
├── utils.py                    # Rank metrics + localization functions
├── analyze_sparse_subnetwork.py # Post-hoc subnetwork analysis
├── localization_analysis.py    # Comparative analysis script
├── configs/
│   ├── __init__.py
│   └── gemma3n_config.py       # Gemma-3n/MatFormer specific config
├── plots/                      # Generated visualizations
└── checkpoints/                # Model checkpoints (gitignored)
```

## Training Pipeline

```
IMDB Prompts → Generate Responses → Reward Model Score → PPO Update → Save Checkpoint
                                                              ↓
                                                    Log Localization Metrics
                                                    (Gini, Entropy, Top-k)
```

- **Policy**: GPT-2 Medium or Gemma-3n (fine-tuned via PPO)
- **Reward Model**: `lvwerra/distilbert-imdb` (frozen sentiment classifier)
- **Reference**: Same as policy (frozen, for KL penalty)
- **Task**: Generate positive sentiment continuations

## Key Configuration

```python
@dataclass
class ExpConfig:
    policy_name: str = "gpt2-medium"  # or "google/gemma-3n-E2B-it"
    batch_size: int = 32
    num_updates: int = 600
    save_ckpt_every: int = 20
    tau: float = 1e-6                 # Weight update threshold
    log_localization: bool = True     # Enable localization metrics
    topk_layers: int = 3              # For top-k concentration
```

## Expected Results

If the hypothesis is correct:
- Gemma-3n will have **higher Gini coefficient** (more concentrated updates)
- Gemma-3n will have **lower entropy** (less distributed)
- Gemma-3n's **top-3 layer percentage** will be higher
- Updates may cluster in "outer shell" layers (later FFN dimensions)

## Background

This project extends the findings from ["Reinforcement Learning Finetunes Small Subnetworks in Large Language Models"](https://arxiv.org/abs/2505.11711) by investigating whether architectural choices (specifically MatFormer's nested structure) influence the localization of sparse subnetworks during RL fine-tuning.

## References

- [MatFormer: Nested Transformer for Elastic Inference](https://arxiv.org/abs/2310.07707)
- [Gemma-3n Technical Report](https://ai.google.dev/gemma)
- [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)
- [RL Finetunes Small Subnetworks](https://arxiv.org/abs/2505.11711)

## License

MIT
