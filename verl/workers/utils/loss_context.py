# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Per-call normalization and the narrow compatibility boundary for policy losses."""

from copy import copy
from dataclasses import asdict, dataclass

from omegaconf import DictConfig, open_dict, read_write

from verl.utils import tensordict_utils as tu


@dataclass(frozen=True)
class LossContext:
    dp_size: int = 1
    batch_num_tokens: int | None = None
    global_batch_size: int | None = None
    loss_scale_factor: float | None = None

    @classmethod
    def from_batch(cls, data, config):
        """Keep explicit global denominators; never infer them from a filtered mask."""
        return cls(
            dp_size=tu.get(data, "dp_size"),
            batch_num_tokens=tu.get(data, "batch_num_tokens", default=None),
            global_batch_size=tu.get(data, "global_batch_size", default=None),
            loss_scale_factor=getattr(config, "loss_scale_factor", None),
        )

    def aggregate(self, values, mask, mode):
        from verl.trainer.ppo.core_algos import agg_loss

        return agg_loss(values, mask, mode, **asdict(self))

    def normalize_token_sum(self, total):
        if self.batch_num_tokens is None or self.batch_num_tokens <= 0:
            raise ValueError("full KL requires positive globally valid token count")
        return total * self.dp_size / self.batch_num_tokens

    def bind_legacy_config(self, config):
        """Give upstream policy functions their ABI without mutating shared config.

        Only the small runtime dictionary is replaced. Model/data objects and
        unrelated nested configuration are not deep-copied.
        """
        bound = copy(config)
        if isinstance(bound, DictConfig):
            with read_write(bound), open_dict(bound):
                bound.global_batch_info = asdict(self)
        else:
            object.__setattr__(bound, "global_batch_info", asdict(self))
        return bound
