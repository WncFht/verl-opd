# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Static objective metadata, importable without tensor kernels or worker configs."""

from dataclasses import dataclass, field

from verl.base_config import BaseConfig


@dataclass
class DistillationLossSettings(BaseConfig):
    names: str | list[str] = field(default_factory=list)
    use_topk: bool = False
    use_estimator: bool = False
    payload_kind: str | None = None

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if self.payload_kind is None:
            object.__setattr__(self, "payload_kind", "topk" if self.use_topk else "sampled")
        if self.payload_kind not in ("sampled", "topk", "hidden"):
            raise ValueError(f"Unknown teacher payload: {self.payload_kind}")
        if sum([self.use_topk, self.use_estimator, self.payload_kind == "hidden"]) != 1:
            raise ValueError("Expected exactly one sampled, top-k, or hidden payload")


_BUILTINS = (
    DistillationLossSettings(names="full_reverse_kl", payload_kind="hidden"),
    DistillationLossSettings(names="forward_kl_topk", use_topk=True),
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True),
)
DISTILLATION_SETTINGS_REGISTRY = {name: spec for spec in _BUILTINS for name in spec.names}


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY)}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]
