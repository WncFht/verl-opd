# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Per-instance dense Qwen output adapter; preserve root AND head FSDP hooks."""

from types import MethodType


def install_full_vocab_adapter(model):
    if hasattr(model, "_full_vocab_original_forward"):
        return
    if model.config.model_type not in ("qwen2", "qwen3"):
        raise ValueError("full reverse KL supports dense Qwen2/Qwen3 models")
    if model.lm_head.bias is not None:
        raise ValueError("full reverse KL requires a bias-free LM head")
    model._full_vocab_original_forward = model.forward
    model.lm_head._full_vocab_original_forward = model.lm_head.forward
    model.forward = MethodType(full_vocab_forward, model)
    model.lm_head.forward = MethodType(full_vocab_head_forward, model.lm_head)


def full_vocab_head_forward(self, hidden, *, full_vocab_callback=None):
    if full_vocab_callback is None:
        return self._full_vocab_original_forward(hidden)
    if self.bias is not None:
        raise ValueError("full reverse KL currently requires a bias-free LM head")
    return full_vocab_callback(hidden, self.weight)


def full_vocab_forward(self, *args, full_vocab_callback=None, **kwargs):
    if full_vocab_callback is None:
        return self._full_vocab_original_forward(*args, **kwargs)
    kwargs.update(return_dict=True, output_hidden_states=False, use_cache=False)
    from torch.profiler import record_function as full_kl_range

    with full_kl_range("backbone"):
        output = self.model(*args, **kwargs)
    return full_vocab_callback(output.last_hidden_state, self.lm_head)
