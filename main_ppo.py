import os
import random
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
import torch
import torch.nn as nn
from dataclasses import dataclass
from datasets import load_dataset
os.environ["WANDB_DISABLED"] = "true"
import wandb
from utils import *
from typing import List, Dict
import numpy
import scipy
import trl
import transformers

print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("trl", trl.__version__)
print("transformers", transformers.__version__)

@dataclass
class ExpConfig:
    policy_name: str = "gpt2"   # base LM
    reward_model_name: str = "lvwerra/distilbert-imdb"
    max_prompt_tokens: int = 128
    max_new_tokens: int = 64

    batch_size: int = 16              # PPO batch per update
    mini_batch_size: int = 4
    ppo_epochs: int = 4
    learning_rate: float = 1e-5
    target_kl: float = 0.1
    num_updates: int = 200                            # PPO updates

    rank_log_every: int = 20                          # log rank stats every N steps
    tau: float = 1e-5                                  # tolerance for "updated" weights

    save_ckpt_every: int = 10                         # save model every N steps
    ckpt_dir: str = "checkpoints"

    device: str = "cuda"

    dtype: torch.dtype = torch.float32  # use fp32 for stability during generation/updates


cfg = ExpConfig()


def load_imdb_dataset(max_samples: int = 10000) -> List[str]:
    ds = load_dataset("imdb", split="train")
    texts = ds[:max_samples]["text"]
    random.shuffle(texts)

    return texts

def build_policy_and_ref_model() -> tuple[AutoTokenizer, nn.Module, nn.Module]:
    dtype = cfg.dtype
    tok = AutoTokenizer.from_pretrained(cfg.policy_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg.policy_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(cfg.device)
    policy.config.use_cache = False
    policy.gradient_checkpointing_disable()

    ref = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg.policy_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(cfg.device)
    ref.config.use_cache = False
    ref.requires_grad_(False)

    return tok, policy, ref

def reward_model() -> tuple[AutoTokenizer, nn.Module]:
    rm_tok = AutoTokenizer.from_pretrained(cfg.reward_model_name)
    rm = AutoModelForSequenceClassification.from_pretrained(
        cfg.reward_model_name, 
        torch_dtype=cfg.dtype, 
        use_safetensors=True
    ).to(cfg.device)
    rm.eval()
    return rm_tok, rm

@torch.no_grad()
def compute_rewards(rm_tok, rm, text: List[str]) -> torch.Tensor:
    """ The reward is just Logit_positive - Logit_negative where Logit_positive is the logit of the positive class and Logit_negative is the logit of the negative class. After softmax these would become probabilities of the positive and negative classes. The logits are the output of the sentiment classifier. """
    enc = rm_tok(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(cfg.device)
    logits = rm(**enc).logits
    reward = logits[:, 1] - logits[:, 0]
    return reward.detach().to(torch.float16)


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
    print(f"now Using device: {cfg.device}")

    # models
    tok, policy, ref = build_policy_and_ref_model()
    rm_tok, rm = reward_model()

    # PPO config (TRL 0.8.x API)
    ppo_config = PPOConfig(
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        mini_batch_size=cfg.mini_batch_size,
        ppo_epochs=cfg.ppo_epochs,
        target_kl=cfg.target_kl,
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

        enc = tok(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_prompt_tokens,
        ).to(trainer.accelerator.device)

        batch_query_tensor = enc["input_ids"]  # [B, L]
        query_tensors = [q for q in batch_query_tensor]  # list[Tensor[L]]

        # ---- generate with current policy ----
        print(f"[step {step}] sampling responses...")
        gen_tensors = trainer.generate(
            query_tensors,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0,
            pad_token_id=tok.eos_token_id,
        )  # [B, L+L_new]
        print(f"[step {step}] generated {len(gen_tensors) if isinstance(gen_tensors, list) else gen_tensors.size(0)} sequences.")

        if isinstance(gen_tensors, list):
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            gen_tensors = torch.nn.utils.rnn.pad_sequence(
                gen_tensors, batch_first=True, padding_value=pad_id
            )

        response_tensors = gen_tensors[:, batch_query_tensor.shape[1]:]  # just new tokens
        response_list = [resp for resp in response_tensors]  # TRL 0.8.x expects list
        response_masks = [
            (resp != (tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)).long()
            for resp in response_list
        ]

        # full text for reward
        full_texts = tok.batch_decode(gen_tensors, skip_special_tokens=True)

        # ---- compute rewards ----
        rewards_vec = compute_rewards(rm_tok, rm, full_texts)  # [B]
        reward_tensors = [r.unsqueeze(0) for r in rewards_vec]         # list[Tensor[1]]
        print(f"[step {step}] computed rewards, mean={rewards_vec.mean().item():.4f}")

        # ---- PPO update ----
        stats = trainer.step(query_tensors, response_list, reward_tensors, response_masks)
        print(f"[step {step}] ppo step done. stats keys: {list(stats.keys())[:5]} ...")

        # log PPO stats every 10 steps
        if step % 10 == 0:
            decoded_responses = tok.batch_decode(response_tensors, skip_special_tokens=True)
            log_batch = {"query": batch_texts, "response": decoded_responses}
            trainer.log_stats(
                stats=stats,
                batch=log_batch,
                rewards=rewards_vec,
            )

        # ---- rank + sparsity logging ----
        if step % cfg.rank_log_every == 0:
            # get_layer_rank_stats should accept (model, init_state, tau=...)
            layer_stats = get_layer_rank_stats(trainer.model, init_state, tolerance=cfg.tau)

            # log a few representative layers (avoid spamming W&B)
            for name, s in list(layer_stats.items())[:10]:
                wandb.log(
                    {
                        f"rank/eff/{name}": s["eff_rank"],
                        f"rank/stable/{name}": s["stable_rank"],
                        f"norm/fro/{name}": s["fro_norm"],
                        f"sparsity/frac_large/{name}": s["frac_large"],
                        f"sparsity/percent_updated/{name}": s["percent_updated"],
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

        # ---- save checkpoints for sparse-subnetwork analysis ----
        maybe_save_checkpoint(step, trainer.model)

    # final save
    trainer.save_pretrained(f"ppo-{cfg.policy_name}-sentiment-final")
    print("Training finished, model saved.")



if __name__ == "__main__":
    main()
