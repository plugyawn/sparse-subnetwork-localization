import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
)

from configs import get_model_config, analyze_matformer_nesting
from tqdm import tqdm


@dataclass
class GRPOConfig:
    # Model / data
    policy_name: str = "google/gemma-3-4b-it"
    ref_name: Optional[str] = None
    reward_model_name: Optional[str] = None
    dataset: str = "imdb"  # or "dummy"
    max_samples: int = 2000
    prompt_prefix_words: int = 16
    prompt_prefix: str = "Sentiment: Positive review."

    # Generation
    max_prompt_tokens: int = 128
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95

    # GRPO
    batch_size: int = 2
    group_size: int = 2
    learning_rate: float = 1e-5
    kl_coef: float = 0.1
    max_steps: int = 1
    eval_every: int = 1
    save_dir: str = "checkpoints_grpo"
    save_every: int = 5
    optimizer: str = "adamw"  # adamw | adafactor | adamw8bit
    foreach: bool = False
    gradient_checkpointing: bool = False
    ref_device: Optional[str] = None
    # MatFormer shell logging (Gemma-3n)
    log_matformer: bool = True
    matformer_tau: float = 1e-6
    matformer_log_every: int = 1
    freeze_vision: bool = False
    reward_device: Optional[str] = None
    tqdm: bool = True
    tqdm_update_every: int = 1

    # Runtime
    seed: int = 0
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    use_wandb: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def maybe_patch_config(config) -> None:
    """Patch Gemma-3/3n configs to expose top-level fields expected by tooling."""
    if hasattr(config, "text_config"):
        if not hasattr(config, "hidden_size") and hasattr(config.text_config, "hidden_size"):
            config.hidden_size = config.text_config.hidden_size
        if not hasattr(config, "vocab_size") and hasattr(config.text_config, "vocab_size"):
            config.vocab_size = config.text_config.vocab_size


def load_tokenizer(model_name: str, use_processor: bool, trust_remote_code: bool) -> AutoTokenizer:
    if use_processor:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    else:
        tok = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="right",
            trust_remote_code=trust_remote_code,
        )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_policy_and_ref(cfg: GRPOConfig) -> Tuple[AutoTokenizer, torch.nn.Module, torch.nn.Module]:
    model_cfg = get_model_config(cfg.policy_name)
    use_processor = model_cfg.use_processor
    trust_remote_code = "gemma" in cfg.policy_name.lower()

    tok = load_tokenizer(cfg.policy_name, use_processor, trust_remote_code)

    model_kwargs = {
        "dtype": cfg.dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
    }
    policy = AutoModelForCausalLM.from_pretrained(cfg.policy_name, **model_kwargs).to(cfg.device)
    maybe_patch_config(policy.config)
    policy.config.use_cache = False
    policy.train()

    ref_name = cfg.ref_name or cfg.policy_name
    ref_device = cfg.ref_device or cfg.device
    ref = AutoModelForCausalLM.from_pretrained(ref_name, **model_kwargs).to(ref_device)
    maybe_patch_config(ref.config)
    ref.config.use_cache = False
    ref.requires_grad_(False)
    ref.eval()

    return tok, policy, ref


def build_matformer_base_state(policy: torch.nn.Module) -> dict[str, torch.Tensor]:
    base_state: dict[str, torch.Tensor] = {}
    for name, param in policy.named_parameters():
        if not param.requires_grad:
            continue
        if not name.endswith("weight"):
            continue
        lower = name.lower()
        if "mlp" not in lower and "fc" not in lower:
            continue
        if param.dim() != 2:
            continue
        base_state[name] = param.detach().cpu().clone()
    return base_state


def compute_matformer_shell_summary(
    base_state: dict[str, torch.Tensor],
    policy: torch.nn.Module,
    hidden_dim: int,
    tau: float,
) -> dict[int, int]:
    current_state: dict[str, torch.Tensor] = {}
    for name, param in policy.named_parameters():
        if name in base_state:
            current_state[name] = param.detach().cpu().clone()
    level_dist_by_layer = analyze_matformer_nesting(base_state, current_state, hidden_dim, tau)
    total_counts: dict[int, int] = {}
    for _, level_counts in level_dist_by_layer.items():
        for level, count in level_counts.items():
            total_counts[level] = total_counts.get(level, 0) + count
    return total_counts


def load_reward_model(cfg: GRPOConfig) -> Tuple[Optional[AutoTokenizer], Optional[torch.nn.Module]]:
    if not cfg.reward_model_name:
        return None, None
    rm_tok = AutoTokenizer.from_pretrained(cfg.reward_model_name)
    rm_device = cfg.reward_device or cfg.device
    rm = AutoModelForSequenceClassification.from_pretrained(
        cfg.reward_model_name,
        torch_dtype=torch.float32,
        use_safetensors=True,
    ).to(rm_device)
    rm.eval()
    return rm_tok, rm


def load_prompts(cfg: GRPOConfig) -> List[str]:
    if cfg.dataset == "dummy":
        return [
            "Write a short positive movie review about friendship.",
            "Explain why patience matters when learning a new skill.",
            "Summarize a story about a lost dog finding its way home.",
            "Describe a pleasant surprise at work.",
        ]

    ds = load_dataset(cfg.dataset, split="train")
    texts = ds[: cfg.max_samples]["text"]
    random.shuffle(texts)
    prompts = [" ".join(t.split()[: cfg.prompt_prefix_words]) for t in texts]
    return prompts


def expand_group(prompts: List[str], group_size: int) -> Tuple[List[str], torch.Tensor]:
    expanded = []
    group_ids = []
    for idx, prompt in enumerate(prompts):
        for _ in range(group_size):
            expanded.append(prompt)
            group_ids.append(idx)
    return expanded, torch.tensor(group_ids, dtype=torch.long)


def build_prompts(cfg: GRPOConfig, batch_prompts: List[str]) -> List[str]:
    prefix = cfg.prompt_prefix.strip()
    if prefix:
        return [f"{prefix} {p}" for p in batch_prompts]
    return batch_prompts


@torch.no_grad()
def compute_rewards_from_rm(
    rm_tok,
    rm,
    texts: List[str],
    device: str,
) -> torch.Tensor:
    enc = rm_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    logits = rm(**enc).logits
    reward = logits[:, 1] - logits[:, 0]
    return reward.to(torch.float32)


def compute_rewards_heuristic(texts: List[str]) -> torch.Tensor:
    # Simple length-based reward for smoke tests
    return torch.tensor([len(t.split()) for t in texts], dtype=torch.float32)


def compute_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    model_device = next(model.parameters()).device
    input_ids = input_ids.to(model_device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model_device)
    prompt_lens = prompt_lens.to(model_device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    positions = torch.arange(labels.size(1), device=labels.device).unsqueeze(0)
    response_mask = positions >= (prompt_lens - 1).unsqueeze(1)
    if pad_token_id is not None:
        response_mask = response_mask & (labels != pad_token_id)
    token_logprobs = token_logprobs * response_mask
    return token_logprobs.sum(dim=1)


def group_normalize(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    group_ids = group_ids.to(rewards.device)
    num_groups = int(group_ids.max().item()) + 1 if group_ids.numel() > 0 else 0
    means = torch.zeros(num_groups, device=rewards.device)
    stds = torch.zeros(num_groups, device=rewards.device)
    for i in range(num_groups):
        mask = group_ids == i
        group_vals = rewards[mask]
        means[i] = group_vals.mean()
        stds[i] = group_vals.std(unbiased=False) + eps
    return (rewards - means[group_ids]) / stds[group_ids]


@torch.no_grad()
def run_eval(
    model: torch.nn.Module,
    tok: AutoTokenizer,
    prompts: List[str],
    cfg: GRPOConfig,
    rm_tok=None,
    rm=None,
) -> float:
    model.eval()
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=cfg.max_prompt_tokens).to(cfg.device)
    gen = model.generate(
        **enc,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=True,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        temperature=cfg.temperature,
        pad_token_id=tok.eos_token_id,
    )
    decoded = tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    if rm is not None:
        rewards = compute_rewards_from_rm(rm_tok, rm, decoded, cfg.device)
    else:
        rewards = compute_rewards_heuristic(decoded)
    model.train()
    return float(rewards.mean().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal GRPO runner (CPU-friendly smoke test).")
    parser.add_argument("--policy", dest="policy_name", type=str, default=GRPOConfig.policy_name)
    parser.add_argument("--ref", dest="ref_name", type=str, default=None)
    parser.add_argument("--reward_model", type=str, default=None)
    parser.add_argument("--reward_device", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="imdb")
    parser.add_argument("--max_samples", type=int, default=GRPOConfig.max_samples)
    parser.add_argument("--max_prompt_tokens", type=int, default=GRPOConfig.max_prompt_tokens)
    parser.add_argument("--max_new_tokens", type=int, default=GRPOConfig.max_new_tokens)
    parser.add_argument("--batch_size", type=int, default=GRPOConfig.batch_size)
    parser.add_argument("--group_size", type=int, default=GRPOConfig.group_size)
    parser.add_argument("--max_steps", type=int, default=GRPOConfig.max_steps)
    parser.add_argument("--eval_every", type=int, default=GRPOConfig.eval_every)
    parser.add_argument("--save_dir", type=str, default=GRPOConfig.save_dir)
    parser.add_argument("--save_every", type=int, default=GRPOConfig.save_every)
    parser.add_argument("--optimizer", type=str, default=GRPOConfig.optimizer, choices=["adamw", "adafactor", "adamw8bit"])
    parser.add_argument("--foreach", action="store_true", help="Enable torch foreach optimizer kernels.")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--ref_device", type=str, default=None)
    parser.add_argument("--freeze_vision", action="store_true")
    parser.add_argument("--tqdm", dest="tqdm", action="store_true", help="Enable tqdm progress bar.")
    parser.add_argument("--no_tqdm", dest="tqdm", action="store_false", help="Disable tqdm progress bar.")
    parser.add_argument("--tqdm_update_every", type=int, default=GRPOConfig.tqdm_update_every)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--short", action="store_true", help="Short-run config for quick GPU tests.")
    parser.add_argument("--smoke", action="store_true", help="Use tiny config for CPU smoke test.")
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    cfg = GRPOConfig(
        policy_name=args.policy_name,
        ref_name=args.ref_name,
        reward_model_name=args.reward_model,
        reward_device=args.reward_device,
        tqdm=args.tqdm,
        tqdm_update_every=args.tqdm_update_every,
        dataset=args.dataset,
        max_samples=args.max_samples,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        group_size=args.group_size,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        save_dir=args.save_dir,
        save_every=args.save_every,
        optimizer=args.optimizer,
        foreach=args.foreach,
        gradient_checkpointing=args.gradient_checkpointing,
        ref_device=args.ref_device,
        freeze_vision=args.freeze_vision,
        device=args.device,
        dtype=dtype_map[args.dtype],
    )

    if args.short:
        cfg.max_steps = max(cfg.max_steps, 20)
        cfg.eval_every = 5
        cfg.save_every = 5
        cfg.batch_size = max(cfg.batch_size, 1)
        cfg.group_size = max(cfg.group_size, 2)
        cfg.max_prompt_tokens = min(cfg.max_prompt_tokens, 96)
        cfg.max_new_tokens = min(cfg.max_new_tokens, 32)
        cfg.matformer_log_every = 5
        if "gemma-3" in cfg.policy_name.lower():
            cfg.optimizer = "adafactor"
            cfg.foreach = False
            cfg.gradient_checkpointing = True
            cfg.freeze_vision = True

    if args.smoke:
        cfg.policy_name = "tiny-random/gemma-3n"
        cfg.ref_name = None
        cfg.reward_model_name = None
        cfg.dataset = "dummy"
        cfg.batch_size = 1
        cfg.group_size = 2
        cfg.max_steps = 1
        cfg.eval_every = 1
        cfg.max_new_tokens = 16
        cfg.max_prompt_tokens = 64
        cfg.device = "cpu"
        cfg.dtype = torch.float32
        cfg.log_matformer = True
        cfg.save_every = 0
        cfg.optimizer = "adamw"
        cfg.foreach = False
        cfg.freeze_vision = False
        cfg.tqdm = False

    set_seed(cfg.seed)

    tok, policy, ref = load_policy_and_ref(cfg)
    rm_tok, rm = load_reward_model(cfg)

    prompts = load_prompts(cfg)
    if cfg.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
        if hasattr(policy, "enable_input_require_grads"):
            policy.enable_input_require_grads()

    if cfg.freeze_vision:
        for name, param in policy.named_parameters():
            if "vision" in name:
                param.requires_grad = False

    if cfg.optimizer == "adafactor":
        from transformers.optimization import Adafactor
        optimizer = Adafactor(
            policy.parameters(),
            lr=cfg.learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    elif cfg.optimizer == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except Exception as exc:
            raise RuntimeError("bitsandbytes is required for adamw8bit") from exc
        optimizer = bnb.optim.AdamW8bit(policy.parameters(), lr=cfg.learning_rate)
    else:
        optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.learning_rate, foreach=cfg.foreach)

    eval_prompts = build_prompts(cfg, prompts[: min(4, len(prompts))])

    model_cfg = get_model_config(cfg.policy_name)
    hidden_dim = model_cfg.hidden_dim
    if hidden_dim <= 0:
        if hasattr(policy.config, "hidden_size"):
            hidden_dim = int(policy.config.hidden_size)
        elif hasattr(policy.config, "text_config") and hasattr(policy.config.text_config, "hidden_size"):
            hidden_dim = int(policy.config.text_config.hidden_size)

    base_state = {}
    if cfg.log_matformer and model_cfg.is_matformer and hidden_dim > 0:
        base_state = build_matformer_base_state(policy)

    if cfg.save_every and cfg.save_every > 0:
        os.makedirs(cfg.save_dir, exist_ok=True)
        torch.save(policy.state_dict(), os.path.join(cfg.save_dir, "step_000.pt"))

    step_iter = range(1, cfg.max_steps + 1)
    if cfg.tqdm:
        step_iter = tqdm(step_iter, desc="grpo", dynamic_ncols=True)

    def log(msg: str) -> None:
        if cfg.tqdm and hasattr(step_iter, "write"):
            step_iter.write(msg)
        else:
            print(msg)

    for step in step_iter:
        step_start = time.time()
        batch_prompts = random.sample(prompts, cfg.batch_size)
        batch_prompts = build_prompts(cfg, batch_prompts)

        expanded, group_ids = expand_group(batch_prompts, cfg.group_size)
        enc = tok(
            expanded,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_prompt_tokens,
        ).to(cfg.device)

        prompt_lens = enc["attention_mask"].sum(dim=1)
        gen = policy.generate(
            **enc,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            temperature=cfg.temperature,
            pad_token_id=tok.eos_token_id,
        )

        gen_attention = (gen != tok.pad_token_id).long()
        decoded = tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)

        if rm is not None:
            rewards = compute_rewards_from_rm(rm_tok, rm, decoded, cfg.reward_device or cfg.device)
        else:
            rewards = compute_rewards_heuristic(decoded).to(cfg.device)

        logp = compute_logprobs(policy, gen, gen_attention, prompt_lens, tok.pad_token_id)
        with torch.no_grad():
            logp_ref = compute_logprobs(ref, gen, gen_attention, prompt_lens, tok.pad_token_id)

        # Move rewards/logp_ref to policy device for math
        rewards = rewards.to(logp.device)
        logp_ref = logp_ref.to(logp.device)

        adv = group_normalize(rewards, group_ids.to(rewards.device))
        kl = (logp - logp_ref)
        loss = (-(adv.detach() * logp) + cfg.kl_coef * kl).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        step_time = time.time() - step_start
        if cfg.tqdm:
            if step % max(1, cfg.tqdm_update_every) == 0:
                step_iter.set_postfix(
                    loss=f"{loss.item():.4f}",
                    reward=f"{rewards.mean().item():.3f}",
                    kl=f"{kl.mean().item():.4f}",
                    sec=f"{step_time:.2f}",
                )
        else:
            log(
                f"[step {step}] loss={loss.item():.4f} "
                f"reward_mean={rewards.mean().item():.3f} "
                f"reward_std={rewards.std(unbiased=False).item():.3f} "
                f"kl_mean={kl.mean().item():.4f}"
            )

        if (
            cfg.log_matformer
            and model_cfg.is_matformer
            and hidden_dim > 0
            and step % cfg.matformer_log_every == 0
        ):
            level_counts = compute_matformer_shell_summary(
                base_state,
                policy,
                hidden_dim,
                cfg.matformer_tau,
            )
            total = sum(level_counts.values()) + 1e-8
            level_fracs = {k: v / total for k, v in sorted(level_counts.items())}
            log(f"[step {step}] MatFormer shells counts={level_counts} fracs={level_fracs}")

        if cfg.eval_every and step % cfg.eval_every == 0:
            eval_reward = run_eval(policy, tok, eval_prompts, cfg, rm_tok, rm)
            log(f"[step {step}] eval_reward_mean={eval_reward:.3f}")

        if cfg.save_every and cfg.save_every > 0 and step % cfg.save_every == 0:
            os.makedirs(cfg.save_dir, exist_ok=True)
            ckpt_path = os.path.join(cfg.save_dir, f"step_{step:03d}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            log(f"[step {step}] saved checkpoint {ckpt_path}")

    log("GRPO smoke run complete.")


if __name__ == "__main__":
    main()
