# SPDX-License-Identifier: Apache-2.0
import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict
from transfer_queue import KVBatchMeta

from verl.trainer.distillation.losses import distillation_loss
from verl.trainer.distillation.shared_teacher import SharedTeacherScorer
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import DistillationConfig, DistillationLossConfig

ROOT = Path(__file__).resolve().parents[3]


def nested(*rows):
    return torch.nested.nested_tensor(list(rows), layout=torch.jagged)


def test_sampled_opd_keeps_gradient_and_private_normalization_context():
    lp = torch.tensor([-0.8, -1.2, 0.0], requires_grad=True)
    batch = TensorDict(
        {
            "prompts": nested(torch.tensor([1])),
            "responses": nested(torch.tensor([2, 3])),
            "loss_mask": nested(torch.ones(2)),
            "response_mask": nested(torch.ones(2)),
            "old_log_probs": nested(lp[:2].detach()),
            "teacher_logprobs": nested(torch.tensor([[-0.5], [-1.3], [0.0]])),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(batch, dp_size=1, batch_num_tokens=2, global_batch_size=1)
    actor = SimpleNamespace(loss_agg_mode="token-mean", loss_scale_factor=None, global_batch_info={"sentinel": 9})
    objective = DistillationLossConfig(
        loss_mode="k1", use_policy_gradient=True, use_task_rewards=False, loss_max_clamp=None, log_prob_min_clamp=None
    )
    original = dict(objective.global_batch_info)
    a, _ = distillation_loss(actor, SimpleNamespace(distillation_loss=objective), {"log_probs": lp}, batch)
    tu.assign_non_tensor(batch, batch_num_tokens=4)
    b, _ = distillation_loss(actor, SimpleNamespace(distillation_loss=objective), {"log_probs": lp}, batch)
    torch.testing.assert_close(a, 2 * b)
    torch.testing.assert_close(torch.autograd.grad(a, lp)[0], torch.tensor([-0.15, 0.05, 0.0]))
    assert actor.global_batch_info == {"sentinel": 9} and objective.global_batch_info == original


@pytest.mark.parametrize("payload_kind", ["sampled", "hidden"])
def test_shared_teacher_scores_and_offloads_before_publishing(payload_kind):
    calls = []
    worker = SimpleNamespace(
        to=lambda device, **kw: calls.append(device),
        infer_batch=lambda batch: (calls.append("infer") or SimpleNamespace(futures=["result"])),
    )

    def wait(futures):
        assert futures == ["result"]
        calls.append("wait")

    def get(**kw):
        assert calls[-1] == "cpu"
        return TensorDict(
            {"input_ids": nested(torch.tensor([1, 2, 3])), "log_probs": nested(torch.tensor([-1.0, -2.0, 99.0]))},
            batch_size=[1],
        )

    def put(**kw):
        calls.append("publish")
        output = kw["fields"]
        assert output["teacher_ids"].unbind()[0].squeeze(-1).tolist() == [2, 3, 0]
        assert output["teacher_logprobs"].unbind()[0].squeeze(-1).tolist() == [-1.0, -2.0, 0.0]

    scorer = SharedTeacherScorer(worker, payload_kind=payload_kind, get=get, put=put, wait=wait)
    batch = KVBatchMeta(keys=["k"], tags=["test"], partition_id="train", fields=["input_ids"])
    assert scorer.score(batch, {}) is batch
    assert calls == ["device", "infer", "wait", "cpu"] + (["publish"] if payload_kind == "sampled" else [])


@pytest.mark.parametrize("stage", ["forward", "wait", "offload"])
def test_shared_teacher_failure_prevents_publication_and_attempts_cleanup(stage):
    calls = []

    def to(device, **kwargs):
        calls.append(device)
        if stage == "offload" and device == "cpu":
            raise RuntimeError("offload failed")

    def infer(batch):
        if stage == "forward":
            raise RuntimeError("forward failed")
        return SimpleNamespace(futures=[])

    def wait(futures):
        if stage == "wait":
            raise RuntimeError("wait failed")

    worker = SimpleNamespace(to=to, infer_batch=infer)
    scorer = SharedTeacherScorer(
        worker,
        payload_kind="sampled",
        get=lambda **kw: pytest.fail("read after failure"),
        put=lambda **kw: pytest.fail("published after failure"),
        wait=wait,
    )
    batch = KVBatchMeta(keys=["k"], tags=["test"], partition_id="train", fields=[])
    with pytest.raises(RuntimeError, match=stage):
        scorer.score(batch, {})
    assert calls == ["device", "cpu"]


@pytest.mark.parametrize("mode", ["full_reverse_kl", "k1"])
def test_e20_launcher_resolves_the_real_training_protocol(mode):
    env = os.environ | {"DRY_RUN": "1", "LOSS_MODE": mode, "GPUS": "4", "PYTHON_BIN": "/usr/bin/python3"}
    command = shlex.split(
        subprocess.check_output(["bash", str(ROOT / "examples/opd_full_vocab/run_e20.sh")], env=env, text=True)
    )
    overrides = command[command.index("verl.trainer.main_ppo") + 1 :]
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=overrides)
    OmegaConf.resolve(cfg)
    distillation = omega_conf_to_dataclass(cfg.distillation, DistillationConfig)
    distillation.validate_shared_sync(cfg, use_reference_policy=False, use_critic=False)
    assert cfg.trainer.total_training_steps == 256
    assert cfg.actor_rollout_ref.actor.ppo_epochs == 16
    assert cfg.data.train_batch_size == cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 64
    assert cfg.trainer.test_freq == 32 and cfg.trainer.save_freq == 64 and not cfg.trainer.val_before_train
    assert cfg.data.max_prompt_length == 4096 and cfg.data.max_response_length == 7168
    assert cfg.actor_rollout_ref.rollout.val_kwargs.max_tokens == 31744
    assert cfg.actor_rollout_ref.rollout.val_kwargs.n == 16
    assert cfg.actor_rollout_ref.actor.optim.lr == 1e-6 and cfg.data.seed == 42
    assert distillation.distillation_loss.loss_settings.payload_kind == (
        "hidden" if mode == "full_reverse_kl" else "sampled"
    )
    assert cfg.actor_rollout_ref.rollout.calculate_log_probs == (mode == "k1")
    assert not cfg.algorithm.filter_groups.enable
    assert not cfg.trainer.v1.sampler.sync_refill_failed_groups


@pytest.mark.parametrize("full", [False, True])
def test_trainer_shared_stage_order_and_full_vocab_skips_sampled_preparation(full):
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    calls = []
    batch = SimpleNamespace(extra_info={})

    def event(name):
        def call(*args, **kwargs):
            calls.append(name)
            return batch

        return call

    trainer = SimpleNamespace(
        replay_buffer=SimpleNamespace(sample=lambda **kwargs: (batch, {})),
        global_steps=1,
        config=SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(temperature=1.0)),
            trainer=SimpleNamespace(critic_warmup=0),
        ),
        reward_loop_manager=SimpleNamespace(reward_loop_worker_handles=[]),
        on_sample_begin=lambda: calls.append("sample_begin"),
        on_sample_end=lambda: calls.append("sleep_rollout"),
        _balance_batch=event("balance"),
        _uses_shared_teacher=lambda: True,
        _uses_full_vocab=lambda: full,
        _compute_shared_teacher_log_probs=event("teacher"),
        _compute_old_log_prob=event("anchor"),
        _compute_advantage=event("advantage"),
        _update_actor=event("actor"),
        use_reference_policy=False,
        use_critic=False,
    )
    assert PPOTrainer._step_once(trainer, {}, {}, 4) is batch
    assert calls == ["sample_begin", "sleep_rollout", "balance", "teacher"] + (
        [] if full else ["anchor", "advantage"]
    ) + ["actor"]
