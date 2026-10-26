# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Single-teacher hidden layout and frozen output head (outside student state)."""

import json
from pathlib import Path

from safetensors import safe_open


class FrozenTeacherHead:
    """Load a dense Qwen safetensors output weight; never register on the student."""

    def __init__(self, path):
        from verl.utils.fs import copy_to_local

        root = Path(copy_to_local(path))
        config = json.loads((root / "config.json").read_text())
        if config.get("model_type") not in ("qwen2", "qwen3"):
            raise ValueError("full reverse KL requires a dense Qwen2/Qwen3 teacher head")
        key = "model.embed_tokens.weight" if config.get("tie_word_embeddings", False) else "lm_head.weight"
        index = root / "model.safetensors.index.json"
        files = (
            [root / json.loads(index.read_text())["weight_map"][key]]
            if index.exists()
            else [root / "model.safetensors"]
        )
        with safe_open(files[0], framework="pt", device="cpu") as stream:
            self.cpu_weight = stream.get_tensor(key).detach().contiguous()
        if self.cpu_weight.shape != (config["vocab_size"], config["hidden_size"]):
            raise ValueError("teacher head does not match model configuration")
        self.weight = None

    def get(self, device, dtype):
        if self.weight is None or self.weight.device != device or self.weight.dtype != dtype:
            self.weight = self.cpu_weight.to(device=device, dtype=dtype)
        return self.weight

    def release(self):
        self.weight = None
