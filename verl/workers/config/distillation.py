# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from omegaconf import OmegaConf

from verl.base_config import BaseConfig
from verl.utils.config import omega_conf_to_dataclass

from .engine import FSDPEngineConfig
from .rollout import RolloutConfig

__all__ = ["DistillationLossConfig", "DistillationTeacherModelConfig", "DistillationConfig"]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class FullVocabConfig(BaseConfig):
    """Bounded token blocks and hidden transport for exact reverse KL."""

    token_chunk_size: int = 512
    async_write: bool = False
    prefetch: bool = False

    def __post_init__(self):
        if type(self.token_chunk_size) is not int or self.token_chunk_size <= 0:
            raise ValueError("full_vocab.token_chunk_size must be a positive integer")


@dataclass
class DistillationLossConfig(BaseConfig):
    """Configuration for distillation loss settings.

    loss_mode (str):
        Distillation loss function to use.
    topk (int, optional):
        Number of top tokens to consider for top-k distillation losses.
    use_task_rewards (bool):
        Whether to include task rewards alongside distillation loss.
    distillation_loss_coef (float):
        Coefficient for distillation loss when combined with task rewards.
    loss_max_clamp (float, optional):
        Maximum value to clamp distillation loss. If None, no clamping is applied.
    log_prob_min_clamp (float, optional):
        Minimum value to clamp log probabilities for stability, e.g., log q - log p where p or q are
        very close to zero. If None, no clamping is applied.
    use_policy_gradient (bool):
        Whether to incorporate distillation loss as a reward, as done
        by https://thinkingmachines.ai/blog/on-policy-distillation/. Recommended to use loss_mode=k1.
        Otherwise, distillation loss is directly backpropagated as a supervised loss,
        as in https://arxiv.org/abs/2306.13649. Recommended to use loss_mode=k3 or forward_kl_topk.
    policy_loss_mode (str):
        Name of the policy loss to use when use_policy_gradient is true.
    clip_ratio (float):
        PPO clipping ratio for policy loss.
    clip_ratio_low (float):
        Lower bound for PPO clipping ratio.
    clip_ratio_high (float):
        Upper bound for PPO clipping ratio.
    loss_settings (DistillationLossSettings, optional):
        Runtime-populated settings based on loss_mode. Not set by user.
    """

    loss_mode: str = "k3"
    full_vocab: FullVocabConfig = field(default_factory=FullVocabConfig)
    topk: Optional[int] = 128
    use_task_rewards: bool = True
    distillation_loss_coef: float = 1.0
    loss_max_clamp: Optional[float] = 10.0
    log_prob_min_clamp: Optional[float] = -10.0

    # Chunked top-K log-probs (opt-in, avoids [B, T, V] log_softmax buffer
    # at long context). Only consumed by ``loss_mode='forward_kl_topk'``.
    # Default ``False`` to preserve short-context performance (chunked path
    # has ~6x time overhead at N=14K, V=152K). Set ``True`` when hitting OOM
    # at long context (>=64K tokens, V=152K) where the baseline path OOMs.
    use_chunked_topk: bool = False
    # Tokens per chunk along (B*T) when ``use_chunked_topk=True``. Larger
    # chunks reduce kernel-launch overhead but increase per-chunk memory;
    # smaller chunks reduce per-chunk memory but increase kernel-launch
    # overhead (saved-tensor total stays constant either way).
    chunked_topk_chunk_size: int = 4096

    use_policy_gradient: bool = True
    policy_loss_mode: str = "vanilla"
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    # Store distillation loss settings for computing the specified loss_mode
    # Not set by user, populated at runtime
    loss_settings: Optional[dict] = field(default=None, metadata={"omegaconf_ignore": True})

    def __post_init__(self):
        self._mutable_fields.add("loss_settings")
        from verl.workers.config.distillation_spec import DistillationLossSettings, get_distillation_loss_settings

        self.loss_settings: DistillationLossSettings = get_distillation_loss_settings(self.loss_mode)

        if isinstance(self.full_vocab, dict):
            object.__setattr__(self, "full_vocab", FullVocabConfig(**self.full_vocab))
        if self.loss_mode == "full_reverse_kl":
            if self.use_policy_gradient or self.use_task_rewards or self.policy_loss_mode != "vanilla":
                raise ValueError("full_reverse_kl requires direct loss without task rewards or sampled PG")
            if self.loss_max_clamp is not None or self.log_prob_min_clamp is not None:
                raise ValueError("full_reverse_kl requires unclamped loss and log probabilities")

        if self.policy_loss_mode != "vanilla":
            raise NotImplementedError(
                f"Only vanilla policy loss is currently supported when use_policy_gradient is True, "
                f"but got {self.policy_loss_mode}."
            )

        if self.use_policy_gradient and self.loss_mode == "forward_kl_topk":
            print(
                "WARNING: forward_kl_topk is most effective as a supervised distillation loss "
                "(use_policy_gradient=False). With policy gradient, the update uses only the sampled"
                " token's logprob ∇logπ(a), so the top-k distributional signal (how non-sampled logits "
                "should move) is largely unused."
            )

        if not self.use_policy_gradient and self.loss_mode == "k1":
            raise ValueError(
                "Directly backpropagating k1 loss is incorrect since gradient of k1 loss"
                " wrt model weights does not depend on teacher log probabilities."
            )


@dataclass
class DistillationTeacherModelConfig(BaseConfig):
    """Configuration for on-policy distillation teacher.

    key (str, optional):
        Identifier to route examples to the teacher model in multi-teacher setting.
    model_path (str, optional):
        Model path for the teacher model. Can be a local path or a Hugging Face model
    inference (RolloutConfig):
        Rollout configuration for the teacher model inference during distillation.
    num_replicas (int):
        Number of inference replicas of this teacher to launch. Each replica occupies
        `per_replica_world_size` GPUs (= inference.data_parallel_size *
        inference.tensor_model_parallel_size * inference.pipeline_model_parallel_size),
        so the teacher's total GPU footprint is
        `num_replicas * per_replica_world_size`.
    """

    _mutable_fields = BaseConfig._mutable_fields | {"num_replicas", "key"}

    key: Optional[str] = None
    model_path: Optional[str] = None
    inference: RolloutConfig = field(default_factory=RolloutConfig)
    num_replicas: Optional[int] = 0

    @property
    def per_replica_world_size(self) -> int:
        return (
            self.inference.tensor_model_parallel_size
            * self.inference.data_parallel_size
            * self.inference.pipeline_model_parallel_size
        )

    @property
    def world_size(self) -> int:
        return self.num_replicas * self.per_replica_world_size

    def check_configured(self):
        if self.model_path is None:
            raise ValueError("model_path must be specified for distillation teacher model config.")
        if self.key is None:
            raise ValueError("key must be specified for distillation teacher model config.")
        if self.num_replicas is None:
            raise ValueError("num_replicas must be specified for distillation teacher model config.")

    def validate_and_prepare_for_distillation(self, use_topk: bool, topk: Optional[int]) -> None:
        # Prompt + Response from student are fed into teacher as context
        max_model_len = self.inference.max_model_len
        student_prompt_length = self.inference.prompt_length
        student_response_length = self.inference.response_length
        required_context_len = student_prompt_length + student_response_length + 1
        if max_model_len is not None and required_context_len > max_model_len:
            raise ValueError(
                "Distillation teacher inference requires room for the student prompt, the full student "
                f"response, and one generated token, but got {student_prompt_length=}, "
                f"{student_response_length=}, {required_context_len=}, {max_model_len=}."
            )
        self.inference.prompt_length = self.inference.prompt_length + self.inference.response_length
        self.inference.response_length = 1
        self._validate_topk_logprobs(use_topk=use_topk, topk=topk)

    def _validate_topk_logprobs(self, use_topk: bool, topk: Optional[int]) -> None:
        if not use_topk:
            return
        if topk is None:
            raise ValueError("topk must be specified when use_topk is True.")

        engine_name = self.inference.name
        engine_kwargs = self.inference.engine_kwargs
        match engine_name:
            case "vllm":
                vllm_engine_kwargs = dict(engine_kwargs.get("vllm", {}))
                max_logprobs = vllm_engine_kwargs.get("max_logprobs")
                if max_logprobs is None:
                    vllm_engine_kwargs["max_logprobs"] = topk
                    max_logprobs = topk
                if max_logprobs < topk:
                    raise ValueError(
                        f"VLLM max_logprobs ({max_logprobs}) must be >= distillation_loss topk "
                        f"({topk}) to enable distillation loss computation."
                    )
                engine_kwargs["vllm"] = vllm_engine_kwargs
            case "sglang":
                # SGLang's top_logprobs_num is a per-request parameter, so there is no
                # engine-boot cap to align (unlike vLLM's max_logprobs). The async
                # server translates sampling_params["prompt_logprobs"] into
                # return_logprob + logprob_start_len=0 + top_logprobs_num at call time.
                pass
            case _:
                raise NotImplementedError(
                    f"DistillationTeacherModelConfig does not support inference engine {engine_name}"
                )


@dataclass
class DistillationConfig(BaseConfig):
    """Configuration for on-policy distillation.

    enabled (bool):
        Whether on-policy distillation is enabled.
    n_gpus_per_node (int):
        Number of GPUs per node in the teacher resource pool.
    nnodes (int):
        Number of nodes in the teacher resource pool.
    teacher_models (dict[str, TeacherModelConfig]):
        Configurations for teacher models used for multi-teacher distillation.
    teacher_key (str):
        Key to route examples to the appropriate teacher model in multi-teacher setups. Should correspond to a field in
        the data proto, e.g., data_source.
    distillation_loss (DistillationLossConfig):
    Configuration for distillation loss settings.

    NOTE: The `teacher_model` entry is in the `teacher_models` dict by default.
    Since it is popped when other teacher entries are added, using `teacher_model` as
    one of several keys silently drops it. For example, the following CLI overrides result
    in ONLY `teacher_model2` being used:

    ```bash
    distillation.teacher_models.teacher_model.key=openai/gsm8k
    distillation.teacher_models.teacher_model.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    Instead, give the first teacher a different name:

    ```bash
    +distillation.teacher_models.teacher_model1.key=openai/gsm8k
    +distillation.teacher_models.teacher_model1.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    """

    _mutable_fields = BaseConfig._mutable_fields | {"teacher_models", "n_gpus_per_node", "nnodes"}

    enabled: bool = False
    execution_mode: str = "standalone"
    shared_teacher: FSDPEngineConfig = field(
        default_factory=lambda: FSDPEngineConfig(
            strategy="fsdp2",
            forward_only=True,
            model_dtype="bfloat16",
            param_offload=True,
            forward_only_cpu_offload=False,
            infer_max_token_len_per_gpu=32768,
            use_torch_compile=False,
        )
    )
    n_gpus_per_node: int = 0
    nnodes: int = 0
    teacher_models: dict[str, DistillationTeacherModelConfig] = field(default_factory=dict)
    teacher_key: str = "data_source"
    distillation_loss: DistillationLossConfig = field(default_factory=DistillationLossConfig)

    def __post_init__(self):
        if self.execution_mode not in ("standalone", "shared_sync"):
            raise ValueError(f"Unknown distillation.execution_mode: {self.execution_mode}")
        if not self.enabled:
            return

        if self.execution_mode == "shared_sync":
            if len(self.teacher_models) != 1 or "teacher_model" not in self.teacher_models:
                raise ValueError("shared_sync requires exactly one teacher_models.teacher_model")
            if self.n_gpus_per_node != 0 or self.nnodes != 0:
                raise ValueError("shared_sync requires distillation.n_gpus_per_node=0 and nnodes=0")
            loss = self.distillation_loss
            if loss.loss_mode != "full_reverse_kl" and (
                loss.loss_mode != "k1" or not loss.use_policy_gradient or loss.use_task_rewards
            ):
                raise ValueError("shared_sync currently requires k1 + policy gradient without task rewards")
            engine = self.shared_teacher
            required = {
                "strategy": "fsdp2",
                "forward_only": True,
                "param_offload": True,
                "optimizer_offload": False,
                "offload_policy": False,
                "forward_only_cpu_offload": False,
                "reshard_after_forward": True,
                "ulysses_sequence_parallel_size": 1,
                "use_dynamic_bsz": True,
                "use_fused_kernels": False,
                "use_remove_padding": True,
            }
            for name, value in required.items():
                if getattr(engine, name) != value:
                    raise ValueError(f"shared_sync requires shared_teacher.{name}={value}")
            if engine.qat.enable or not engine.infer_max_token_len_per_gpu or engine.infer_max_token_len_per_gpu < 1:
                raise ValueError("shared_sync requires a positive teacher token budget and no QAT")
        self.teacher_models = self._resolve_teacher_models()
        if self.distillation_loss.loss_mode == "full_reverse_kl" and self.execution_mode != "shared_sync":
            raise ValueError("full_reverse_kl currently requires shared_sync")
        if self.execution_mode == "shared_sync":
            return
        teacher_world_size_sum = 0
        for teacher_model in self.teacher_models.values():
            teacher_model.validate_and_prepare_for_distillation(
                use_topk=self.distillation_loss.loss_settings.use_topk,
                topk=self.distillation_loss.topk,
            )
            teacher_world_size_sum += teacher_model.world_size
        total_pool_size = self.n_gpus_per_node * self.nnodes
        if teacher_world_size_sum != total_pool_size:
            raise ValueError(
                f"Sum of teacher (num_replicas * per_replica_world_size) ({teacher_world_size_sum}) must match "
                f"the distillation resource pool size "
                f"({self.n_gpus_per_node=} * {self.nnodes=} = {total_pool_size})."
            )

    def _resolve_teacher_models(self) -> dict[str, DistillationTeacherModelConfig]:
        assert "teacher_model" in self.teacher_models
        if len(self.teacher_models) == 1:
            # Single teacher occupies the entire teacher resource pool.
            teacher_model = self.teacher_models["teacher_model"]
            if self.execution_mode == "shared_sync":
                teacher_model.num_replicas = 0  # No teacher inference servers.
            else:
                inference = teacher_model.inference
                per_replica = (
                    inference.tensor_model_parallel_size
                    * inference.data_parallel_size
                    * inference.pipeline_model_parallel_size
                )
                pool_size = self.n_gpus_per_node * self.nnodes
                if pool_size % per_replica != 0:
                    raise ValueError(
                        f"Single teacher's per_replica_world_size ({per_replica}) must divide the distillation "
                        f"resource pool size ({self.n_gpus_per_node=} * {self.nnodes=} = {pool_size})."
                    )
                teacher_model.num_replicas = pool_size // per_replica
            teacher_model.key = "default"
        else:
            # Multiple teachers: remove default single teacher config
            self.teacher_models.pop("teacher_model")

        # Teacher models dict is keyed by teacher_key instead of YAML entry name
        teacher_models = {}
        for teacher_config in self.teacher_models.values():
            teacher_config = omega_conf_to_dataclass(teacher_config, dataclass_type=DistillationTeacherModelConfig)
            teacher_config.check_configured()
            if teacher_config.key in teacher_models:
                raise ValueError(f"Duplicate teacher key {teacher_config.key} found in teacher models.")
            teacher_models[teacher_config.key] = teacher_config
        return teacher_models

    def validate_shared_sync(self, config, *, use_reference_policy=False, use_critic=False):
        """Reject unsupported shared lifecycles before creating Ray workers."""
        if not self.enabled or self.execution_mode != "shared_sync":
            return
        required = {
            "trainer.use_v1": True,
            "trainer.nnodes": 1,
            "trainer.v1.trainer_mode": "sync",
            "trainer.v1.sync.parameter_sync_step": 1,
            "actor_rollout_ref.hybrid_engine": True,
            "actor_rollout_ref.actor.strategy": "fsdp2",
            "actor_rollout_ref.actor.fsdp_config.param_offload": True,
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
            "actor_rollout_ref.actor.fsdp_config.offload_policy": False,
            "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size": 1,
            "actor_rollout_ref.actor.ulysses_sequence_parallel_size": 1,
            "actor_rollout_ref.model.use_remove_padding": True,
            "actor_rollout_ref.model.use_fused_kernels": False,
            "actor_rollout_ref.rollout.name": "vllm",
            "actor_rollout_ref.rollout.free_cache_engine": True,
            "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
            "actor_rollout_ref.rollout.data_parallel_size": 1,
            "actor_rollout_ref.rollout.pipeline_model_parallel_size": 1,
            "actor_rollout_ref.rollout.agent.default_agent_loop": "single_turn_agent",
        }
        for path, expected in required.items():
            if OmegaConf.select(config, path, default=expected) != expected:
                raise ValueError(f"shared_sync requires {path}={expected}")
        disabled = (
            "reward.reward_model.enable",
            "reward.reward_model.enable_resource_pool",
            "algorithm.filter_groups.enable",
            "skip.rollout_tq.enable",
            "skip.rollout.enable",
            "actor_rollout_ref.model.lora.rank",
            "actor_rollout_ref.model.lora_rank",
            "actor_rollout_ref.model.lora_adapter_path",
            "actor_rollout_ref.model.mtp.enable",
            "actor_rollout_ref.actor.fsdp_config.qat.enable",
            "actor_rollout_ref.rollout.multi_turn.enable",
            "actor_rollout_ref.rollout.topk_log_probs",
            "actor_rollout_ref.rollout.agent.agent_loop_manager_class",
            "actor_rollout_ref.rollout.agent.agent_loop_config_path",
            "trainer.v1.sampler.custom_sampler.path",
            "trainer.v1.sampler.custom_sampler.name",
        )
        for path in disabled:
            if OmegaConf.select(config, path, default=None):
                raise ValueError(f"shared_sync does not support {path}")
        if use_reference_policy or use_critic:
            raise ValueError("shared_sync does not support reference or critic workers")
        world_size = config.trainer.n_gpus_per_node
        if world_size < 1 or self.shared_teacher.fsdp_size not in (-1, world_size):
            raise ValueError("shared teacher fsdp_size must match the actor world size (or be -1)")
        max_length = config.data.max_prompt_length + config.data.max_response_length
        if self.shared_teacher.infer_max_token_len_per_gpu < max_length:
            raise ValueError("shared teacher token budget must hold a complete training sequence")
