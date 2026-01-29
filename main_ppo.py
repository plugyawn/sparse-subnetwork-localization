import os
import random
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from datasets import load_dataset
import wandb
from utils import *
from typing import List, Dict, Optional, Tuple
import numpy
import scipy
import trl
import transformers
from configs import get_model_config, get_name_filter_for_model, MODEL_CONFIGS

# Apply TRL patch for Gemma 3/3n compatibility BEFORE importing TRL models
# This fixes: UnboundLocalError: hidden_size referenced before assignment
from configs.trl_gemma_patch import patch_trl_for_gemma
patch_trl_for_gemma()

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("trl", trl.__version__)
print("transformers", transformers.__version__)

@dataclass
class ExpConfig:
    # Model configuration
    policy_name: str = "gpt2-medium"   # base LM - can be "gpt2-medium", "google/gemma-3n-E2B-it", etc.
    reward_model_name: str = "lvwerra/distilbert-imdb"

    # Generation settings
    max_prompt_tokens: int = 128
    max_new_tokens: int = 64
    prompt_prefix_words: int = 8           # use only the first N words of each IMDB review are fed as the prompt
    target_sentiment: str = "Positive review"     # The beginning of the prompt is this btw, then we have teh IMDB part

    # PPO hyperparameters
    batch_size: int = 32              # PPO batch per update
    mini_batch_size: int = 8         #  32 / 8 = 4
    ppo_epochs: int = 4
    learning_rate: float = 5e-6       # reduced for stability
    target_kl: float = 3
    cliprange: float = 0.2            # PPO clipping range
    num_updates: int = 600                            # PPO updates

    # Logging and checkpointing
    rank_log_every: int = 20                          # log rank stats every N steps
    tau: float = 1e-6                                  # tolerance for "updated" weights
    save_ckpt_every: int = 20                         # save model every N steps
    ckpt_dir: str = "checkpoints"

    # Hardware settings
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    use_wandb: bool = True             # toggle W&B logging

    # Localization analysis settings
    log_localization: bool = True      # log localization metrics (gini, entropy, etc.)
    topk_layers: int = 3               # for top-k concentration metric

    # Model-specific settings (auto-detected from policy_name)
    _name_filter: Optional[Tuple[str, ...]] = field(default=None, repr=False)
    _is_gemma: bool = field(default=False, repr=False)

    def __post_init__(self):
        """Auto-detect model-specific settings."""
        model_config = get_model_config(self.policy_name)
        self._name_filter = model_config.name_filter
        self._is_gemma = "gemma" in self.policy_name.lower()

        # Adjust settings for Gemma-3n (larger model needs smaller batch)
        if self._is_gemma:
            print(f"[config] Detected Gemma model: {self.policy_name}")
            print(f"[config] Using name filter: {self._name_filter}")
            # Reduce batch size for larger models if not explicitly set
            # (user can override by setting batch_size before __post_init__)


def get_experiment_name(cfg: ExpConfig) -> str:
    """Generate experiment name for wandb based on config."""
    model_short = cfg.policy_name.split("/")[-1]
    return f"ppo-{model_short}-sentiment"


cfg = ExpConfig()


def load_imdb_dataset(max_samples: int = 10000) -> List[str]:
    ds = load_dataset("imdb", split="train")
    texts = ds[:max_samples]["text"]
    random.shuffle(texts)

    return texts

def build_policy_and_ref_model() -> tuple[AutoTokenizer, nn.Module, nn.Module]:
    """
    Build policy and reference models with appropriate loading for model type.

    Handles both GPT-2 style models and Gemma-3n with proper tokenizer/processor setup.
    """
    dtype = cfg.dtype
    model_config = get_model_config(cfg.policy_name)

    # Load tokenizer (Gemma-3n uses processor but we extract tokenizer for text-only tasks)
    if model_config.use_processor:
        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(cfg.policy_name, trust_remote_code=True)
            tok = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
        except Exception as e:
            print(f"[warning] Could not load processor, falling back to tokenizer: {e}")
            tok = AutoTokenizer.from_pretrained(cfg.policy_name, trust_remote_code=True, padding_side="right")
    else:
        tok = AutoTokenizer.from_pretrained(cfg.policy_name, padding_side="right")

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Common model loading kwargs
    model_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "use_safetensors": True,
    }

    # Add trust_remote_code for Gemma models
    if cfg._is_gemma:
        model_kwargs["trust_remote_code"] = True
        print(f"[model] Loading Gemma-3n model with trust_remote_code=True")

    # Load policy model
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg.policy_name,
        **model_kwargs,
    ).to(cfg.device)
    policy.config.use_cache = False
    policy.gradient_checkpointing_disable()

    # Load reference model
    ref = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg.policy_name,
        **model_kwargs,
    ).to(cfg.device)
    ref.config.use_cache = False
    ref.requires_grad_(False)
    ref.eval()

    # Print model info
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[model] {cfg.policy_name}: {total_params:,} total params, {trainable_params:,} trainable")

    return tok, policy, ref

def reward_model() -> tuple[AutoTokenizer, nn.Module]:
    rm_tok = AutoTokenizer.from_pretrained(cfg.reward_model_name)
    rm = AutoModelForSequenceClassification.from_pretrained(
        cfg.reward_model_name, 
        torch_dtype=torch.float32, 
        use_safetensors=True
    ).to(cfg.device)
    rm.eval()
    return rm_tok, rm

@torch.no_grad()
def compute_rewards(rm_tok, rm, text: List[str]) -> torch.Tensor:
    """ Reward = logit(pos) - logit(neg) """
    enc = rm_tok(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(cfg.device)
    logits = rm(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    reward = logits[:, 1] - logits[:, 0]
    return reward.detach().to(torch.float32)


def maybe_save_checkpoint(step: int, policy: nn.Module):
    if step % cfg.save_ckpt_every != 0:
        return
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    path = os.path.join(cfg.ckpt_dir, f"step_{step:03d}.pt")
    torch.save(policy.state_dict(), path)
    print(f"[ckpt] saved {path}")

def main():
    torch.manual_seed(0)
    random.seed(0)
    print(f"okayyyy now Using device: {cfg.device}")

    if cfg.use_wandb:
        os.environ.pop("WANDB_DISABLED", None)
        experiment_name = get_experiment_name(cfg)
        wandb.init(project="sparse-subnetwork-localization", name=experiment_name, config=vars(cfg))
    else:
        os.environ["WANDB_DISABLED"] = "true"

    # models
    tok, policy, ref = build_policy_and_ref_model()
    rm_tok, rm = reward_model()

    # PPO config (TRL 0.8.x API)
    # Note: In TRL 0.8.6, adaptive KL control is enabled by default when target_kl is set
    ppo_config = PPOConfig(
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        mini_batch_size=cfg.mini_batch_size,
        ppo_epochs=cfg.ppo_epochs,
        target_kl=cfg.target_kl,  # This enables adaptive KL penalty control
        cliprange=cfg.cliprange,
        remove_unused_columns=False,
    )

    print('set ppo config')

    trainer = PPOTrainer(
        ppo_config,   # config (positional)
        policy,       # model
        ref,          # reference model
        tok,          # tokenizer
    )

    print('set trainer')


    # initial state for ΔW
    init_state: Dict[str, torch.Tensor] = {
        name: p.detach().cpu().clone()
        for name, p in policy.named_parameters()
        if p.requires_grad
    }

    print('set init state')

    # save ckpt at step 0 (for ΔW baseline, and later sparse-subnetwork analysis)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    torch.save(trainer.model.state_dict(), os.path.join(cfg.ckpt_dir, "step_000.pt"))

    # data
    prompts = load_imdb_dataset()
    print(f"Loaded {len(prompts)} IMDB prompts")

    def sample_batch(bs: int) -> List[str]:
        return random.sample(prompts, bs)

    print('starting ppo loop...')
    # PPO loop
    for step in range(1, cfg.num_updates + 1):
        # ---- sample prompts ----
        batch_texts = sample_batch(cfg.batch_size)
        batch_prefixes = [" ".join(t.split()[:cfg.prompt_prefix_words]) for t in batch_texts]
        
        # Format: "Sentiment: positive\nReview: <truncated_review>"
        batch_prompts_with_sentiment = [
            f"Sentiment: {cfg.target_sentiment}. {prefix}" 
            for prefix in batch_prefixes
        ]

        enc = tok(
            batch_prompts_with_sentiment,  
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_prompt_tokens,
        ).to(trainer.accelerator.device)

        input_ids = enc["input_ids"]  
        attn = enc["attention_mask"]  
        lengths = attn.sum(dim=1).tolist()  

        query_tensors = [input_ids[i, :lengths[i]].clone() for i in range(len(lengths))]

        print(f"[step {step}] sampling responses...") if step % 5 == 0 else None
        gen_tensors = trainer.generate(
            query_tensors,
            max_new_tokens=cfg.max_new_tokens,
            min_new_tokens=12,  
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1,
            pad_token_id=tok.eos_token_id,
        )  
        print(f"[step {step}] generated {len(gen_tensors) if isinstance(gen_tensors, list) else gen_tensors.size(0)} sequences.") if step % 5 == 0 else None

        # Extract response portion from each generated sequence
        # With right padding, prompt is at the start, so slice at each item's true length
        response_list = []
        valid_indices = []  # track which samples have valid (non-empty) responses
        for idx, (gen_seq, prompt_len) in enumerate(zip(gen_tensors, lengths)):
            response = gen_seq[prompt_len:]  # slice off the prompt, keep only response
            # Filter out empty responses - don't create synthetic tokens
            if len(response) > 0:
                response_list.append(response)
                valid_indices.append(idx)

        if len(response_list) == 0:
            print(f"[step {step}] all responses were empty; skipping step.")
            continue
        
        avg_len = float(torch.tensor([r.numel() for r in response_list]).float().mean())
        print(f"[step {step}] avg_response_len={avg_len:.1f}") if step % 5 == 0 else None
        
        # Filter query_tensors and batch data to match valid responses
        query_tensors_filtered = [query_tensors[i] for i in valid_indices]
        lengths_filtered = [lengths[i] for i in valid_indices]
        batch_texts_filtered = [batch_texts[i] for i in valid_indices]
        batch_prefixes_filtered = [batch_prefixes[i] for i in valid_indices]

        # Pad responses and masks consistently for PPO step
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        max_len = max(r.size(0) for r in response_list)
        padded_resps, padded_masks = [], []
        for r in response_list:
            pad_len = max_len - r.size(0)
            if pad_len > 0:
                r = torch.cat([r, torch.full((pad_len,), pad_id, device=r.device, dtype=r.dtype)])
            mask = (r != pad_id).long()
            padded_resps.append(r)
            padded_masks.append(mask)

        response_tensors = torch.stack(padded_resps)  

        generated_texts = tok.batch_decode(response_tensors, skip_special_tokens=True)
        
        
        # Score only the generated text for a cleaner signal on the model's output quality
        rewards_vec = compute_rewards(rm_tok, rm, generated_texts)  # [B] - only for generated text
        
        # Whitening: center and scale rewards to prevent reward bias and control advantage magnitude
        rewards_mean = rewards_vec.mean()
        rewards_std = rewards_vec.std(unbiased=False) + 1e-6  
        rewards_normalized = (rewards_vec - rewards_mean) / rewards_std
        
        reward_scale = 1.0 # i was gonna rmeove this but whatevs
        rewards_scaled = rewards_normalized * reward_scale
        
        reward_tensors = [r.unsqueeze(0) for r in rewards_scaled]  
        
        raw_max = rewards_vec.max().item()
        print(f"[step {step}] computed rewards, raw_mean={rewards_vec.mean().item():.4f}, raw_std={rewards_vec.std().item():.4f}, raw_max={raw_max:.4f}, scaled_mean={rewards_scaled.mean().item():.4f}, scaled_std={rewards_scaled.std().item():.4f}")
        
        if step % 5 == 0 or step <= 3:  
            sample_idx = 0
            if len(generated_texts) > sample_idx:
                # Show IMDB prompt and GPT-generated text separately for clarity
                imdb_prompt = batch_prefixes_filtered[sample_idx]
                gpt_generated = generated_texts[sample_idx]
                
                print(f"[step {step}] === Sample Output ===")
                print(f"[step {step}] IMDB PROMPT (first {cfg.prompt_prefix_words} words): {imdb_prompt}")
                print(f"[step {step}] GPT GENERATED TEXT (scored for reward): {gpt_generated[:400]}{'...' if len(gpt_generated) > 400 else ''}")
                print(f"[step {step}] Reward: raw={rewards_vec[sample_idx].item():.4f}, scaled={rewards_scaled[sample_idx].item():.4f}")
                print(f"[step {step}] ====================")

        stats = trainer.step(query_tensors_filtered, padded_resps, reward_tensors, padded_masks)
        
        # Debugging: print KL and ratio stats
        def safe_get_float(key, default=float('nan')):
            val = stats.get(key, default)
            if isinstance(val, torch.Tensor):
                return val.item() if val.numel() == 1 else float(val.mean().item())
            elif hasattr(val, '__len__') and not isinstance(val, str):  # array-like (numpy, list, etc.)
                try:
                    if isinstance(val, numpy.ndarray):
                        return float(val.item() if val.size == 1 else val.mean())
                except:
                    pass
                try:
                    # Try to get mean if it's an array
                    return float(sum(val) / len(val)) if len(val) > 0 else default
                except:
                    return default
            else:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return default
        
        kl_val = safe_get_float("objective/kl")
        kl_dist_val = safe_get_float("objective/kl_dist")
        print(f"[step {step}] ppo step done. KL={kl_val:.4f}, KL_dist={kl_dist_val:.4f}, stats keys: {list(stats.keys())[:5]} ...")
        
        # Log key metrics to wandb every step for detailed monitoring
        if cfg.use_wandb and wandb.run is not None:
            wandb.log(
                {
                    "reward/raw_mean": rewards_vec.mean().item(),
                    "objective/kl": kl_val,
                    "ppo/policy/entropy": safe_get_float("ppo/policy/entropy"),
                },
                step=step,
            )
        
        response_lens = [r.numel() for r in response_list]
        if response_lens:
            min_len, max_len = min(response_lens), max(response_lens)
            print(f"[step {step}] response lengths: min={min_len}, max={max_len}, avg={avg_len:.1f}") if step % 5 == 0 else None

        # log PPO stats every 10 steps
        if step % 10 == 0:
            decoded_responses = tok.batch_decode(response_tensors, skip_special_tokens=True)
            log_batch = {"query": batch_texts_filtered, "response": decoded_responses}
            trainer.log_stats(
                stats=stats,
                batch=log_batch,
                rewards=rewards_vec,
            )

        # ---- rank + sparsity + localization logging ----
        if step % cfg.rank_log_every == 0:
            # get_layer_rank_stats with model-specific name filter
            layer_stats = get_layer_rank_stats(
                trainer.model,
                init_state,
                name_filter=cfg._name_filter,
                tolerance=cfg.tau
            )

            # Compute localization metrics
            localization = compute_localization_summary(layer_stats, k=cfg.topk_layers)
            block_stats = compute_per_block_stats(layer_stats)

            # log a few representative layers (avoid spamming W&B)
            if cfg.use_wandb and wandb.run is not None:
                for name, s in list(layer_stats.items())[:10]:
                    # Log all rank types together for each layer to plot them on the same graph
                    wandb.log(
                        {
                            f"Ranks/{name}": {
                                "classical": s.get("classical_rank", float('nan')),
                                "effective": s["eff_rank"],
                                "stable": s["stable_rank"],
                            },
                            # Log other metrics separately as before
                            f"norm/fro/{name}": s["fro_norm"],
                            f"sparsity/percent_updated/{name}": s["percent_updated"],
                        },
                        step=step,
                    )

                # Log localization metrics
                if cfg.log_localization:
                    wandb.log(
                        {
                            "localization/gini": localization["gini"],
                            "localization/entropy": localization["entropy"],
                            "localization/topk_concentration": localization["topk_concentration"],
                            "localization/num_active_layers": localization["num_active_layers"],
                        },
                        step=step,
                    )

                    # Log per-block Frobenius norms for heatmap visualization
                    for block_idx, stats in block_stats.items():
                        if block_idx >= 0:  # Skip non-layer params
                            wandb.log(
                                {
                                    f"block_updates/block_{block_idx:02d}": stats["total_fro_norm"],
                                },
                                step=step,
                            )

            # diagnostic print
            some_name = sorted(layer_stats.keys())[0]
            print(
                f"[step {step}] {some_name}: "
                f"eff={layer_stats[some_name]['eff_rank']:.1f}, "
                f"stable={layer_stats[some_name]['stable_rank']:.1f}, "
                f"fro={layer_stats[some_name]['fro_norm']:.2e}, "
                f"percent_updated={layer_stats[some_name]['percent_updated']:.3f}, "
                f"frac_large={layer_stats[some_name]['frac_large']:.3f}"
            )
            print(
                f"[step {step}] LOCALIZATION: "
                f"gini={localization['gini']:.3f}, "
                f"entropy={localization['entropy']:.3f}, "
                f"top{cfg.topk_layers}_conc={localization['topk_concentration']:.3f}"
            )

        # ---- save checkpoints for sparse-subnetwork analysis ----
        maybe_save_checkpoint(step, trainer.model)

    # final save
    trainer.save_pretrained(f"ppo-{cfg.policy_name}-sentiment-final")
    print("Training finished, model saved.")



if __name__ == "__main__":
    main()
