"""
Patch TRL to work with Gemma 3/3n models.

Known Issues (as of Jan 2026):
1. Gemma 3/3n + TRL: `AutoModelForCausalLMWithValueHead` fails with
   `UnboundLocalError: hidden_size referenced before assignment`
   because Gemma 3 stores config in `text_config.hidden_size` not at top level.
   (https://github.com/huggingface/trl/issues/3828)

2. Gemma 2 + TRL: Numerical instability (`inf`, `nan`) with batch_size > 1
   (https://github.com/huggingface/trl/issues/1941)

This module provides patches to fix issue #1 for Gemma 3/3n models.
"""

import warnings


_PATCH_APPLIED = False


def patch_trl_for_gemma():
    """
    Apply TRL compatibility patch for Gemma 3/3n models.

    This must be called BEFORE loading any Gemma model with
    `AutoModelForCausalLMWithValueHead`.

    The patch handles Gemma 3's nested config structure where
    hidden_size is stored in config.text_config.hidden_size
    instead of config.hidden_size.

    Example:
        from configs.trl_gemma_patch import patch_trl_for_gemma
        patch_trl_for_gemma()

        # Now safe to load Gemma models
        from trl import AutoModelForCausalLMWithValueHead
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            "google/gemma-3n-E4B-it",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    """
    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        return

    try:
        from trl.models import modeling_value_head
    except ImportError:
        warnings.warn(
            "[trl_gemma_patch] Could not import trl.models.modeling_value_head. "
            "TRL may not be installed or has incompatible version."
        )
        return

    # Check if ValueHead exists
    if not hasattr(modeling_value_head, 'ValueHead'):
        warnings.warn(
            "[trl_gemma_patch] trl.models.modeling_value_head.ValueHead not found. "
            "TRL version may be incompatible."
        )
        return

    original_init = modeling_value_head.ValueHead.__init__

    def patched_init(self, config, **kwargs):
        """
        Patched __init__ for ValueHead that handles Gemma 3/3n nested config.

        Gemma 3/3n models store their hidden_size in config.text_config.hidden_size
        instead of config.hidden_size. This patch propagates that value up to
        config.hidden_size before calling the original __init__.
        """
        # Handle Gemma 3/3n nested config structure
        if hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
            if not hasattr(config, 'hidden_size') or config.hidden_size is None:
                config.hidden_size = config.text_config.hidden_size

        # Handle other nested configs (e.g., LLaVA, vision-language models)
        if not hasattr(config, 'hidden_size') or config.hidden_size is None:
            # Try common nested config patterns
            nested_config_attrs = ['text_config', 'llm_config', 'language_config']
            for attr in nested_config_attrs:
                if hasattr(config, attr):
                    nested = getattr(config, attr)
                    if hasattr(nested, 'hidden_size') and nested.hidden_size is not None:
                        config.hidden_size = nested.hidden_size
                        break

        return original_init(self, config, **kwargs)

    modeling_value_head.ValueHead.__init__ = patched_init
    _PATCH_APPLIED = True
    print("[trl_gemma_patch] Applied TRL Gemma compatibility patch")


def is_patch_applied() -> bool:
    """Check if the TRL Gemma patch has been applied."""
    return _PATCH_APPLIED


def reset_patch():
    """
    Reset the patch state (for testing purposes).
    Note: This does not actually remove the monkey-patch from TRL.
    """
    global _PATCH_APPLIED
    _PATCH_APPLIED = False
