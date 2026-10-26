# SPDX-License-Identifier: Apache-2.0
"""Independent exact objectives, global scaling and production head adapters."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM

from verl.models.transformers.full_vocab import install_full_vocab_adapter
from verl.trainer.distillation.full_vocab import full_reverse_kl
from verl.trainer.distillation.losses import distillation_loss
from verl.utils import tensordict_utils as tu
from verl.utils.metric import Metric
from verl.workers.config.distillation import DistillationLossConfig
from verl.workers.utils.loss_context import LossContext


def reference(h, w, t, u):
    lp = (h @ w.T).log_softmax(-1)
    lt = (t.detach() @ u.detach().T).log_softmax(-1)[:, : len(w)]
    return (lp.exp() * (lp - lt)).sum()


@pytest.mark.parametrize("tokens,extra", [(0, 0), (1, 3), (7, 3)])
@pytest.mark.parametrize("chunk", [1, 3, 16])
def test_exact_value_and_both_student_gradients(tokens, extra, chunk):
    torch.manual_seed(17)
    h, w, t, u = [
        torch.randn(*shape, dtype=torch.float64, requires_grad=True)
        for shape in [(tokens, 4), (11, 4), (tokens, 6), (11 + extra, 6)]
    ]
    expected = reference(h, w, t, u)
    wanted = torch.autograd.grad(expected, (h, w))
    actual = full_reverse_kl(h, w, t, u, token_chunk=chunk)
    got = torch.autograd.grad(actual, (h, w, t, u), allow_unused=True)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(got[:2], wanted, strict=True):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-11)
    assert got[2:] == (None, None)


def full_config():
    return DistillationLossConfig(
        loss_mode="full_reverse_kl",
        use_policy_gradient=False,
        use_task_rewards=False,
        loss_max_clamp=None,
        log_prob_min_clamp=None,
    )


def test_global_normalization_and_metric_with_uneven_rank_microbatches():
    torch.manual_seed(23)
    h, w = [torch.randn(*shape, dtype=torch.float64, requires_grad=True) for shape in [(7, 4), (11, 4)]]
    t, u = torch.randn(7, 6, dtype=torch.float64), torch.randn(14, 6, dtype=torch.float64)
    actor = SimpleNamespace(loss_agg_mode="token-mean", loss_scale_factor=None)
    dcfg = SimpleNamespace(distillation_loss=full_config())
    locals_, metrics = [], []
    for partitions in [[slice(0, 1), slice(1, 3)], [slice(3, 7), slice(7, 7)]]:
        rank_metrics = Metric("sum")
        for index in partitions:
            raw = full_reverse_kl(h[index], w, t[index], u, token_chunk=2)
            batch = TensorDict({}, batch_size=[])
            tu.assign_non_tensor(batch, dp_size=2, batch_num_tokens=7)
            loss, observed = distillation_loss(actor, dcfg, {"full_kl_sum": raw, "full_kl_count": len(h[index])}, batch)
            locals_.append(loss)
            rank_metrics.extend(observed["distillation/full_kl"])
        metrics.append(rank_metrics)
    global_loss = sum(locals_) / 2
    expected = reference(h, w, t, u) / 7
    torch.testing.assert_close(global_loss, expected)
    assert Metric.aggregate_dp(metrics) == pytest.approx(expected.item())
    for a, b in zip(torch.autograd.grad(global_loss, (h, w)), torch.autograd.grad(expected, (h, w)), strict=True):
        torch.testing.assert_close(a, b)
    with pytest.raises(ValueError, match="positive"):
        LossContext(2, 0).normalize_token_sum(h.sum())


@pytest.mark.parametrize("kind", ["namespace", "omega"])
def test_normalization_context_does_not_mutate_shared_configuration(kind):
    config = SimpleNamespace(global_batch_info={"sentinel": 7})
    if kind == "omega":
        config = OmegaConf.create({"global_batch_info": {"sentinel": 7}})
        OmegaConf.set_readonly(config, True)
    a = LossContext(2, 6, 2).bind_legacy_config(config)
    b = LossContext(4, 12, 4).bind_legacy_config(config)
    assert dict(config.global_batch_info) == {"sentinel": 7}
    assert a.global_batch_info["batch_num_tokens"] == 6
    assert b.global_batch_info["batch_num_tokens"] == 12


@pytest.mark.parametrize("kind", ["qwen2", "qwen3"])
@pytest.mark.parametrize("tied", [False, True])
def test_head_adapter_keeps_normal_forward_hooks_and_backbone_gradients(kind, tied):
    torch.manual_seed(29)
    config_type, model_type = (Qwen2Config, Qwen2ForCausalLM) if kind == "qwen2" else (Qwen3Config, Qwen3ForCausalLM)
    cfg = config_type(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=tied,
    )
    model = model_type(cfg).double()
    ids = torch.tensor([[3, 4, 5, 6]])
    plain = model(ids).logits
    install_full_vocab_adapter(model)
    torch.testing.assert_close(model(ids).logits, plain)
    calls = []
    hook = model.lm_head.register_forward_pre_hook(lambda *args: calls.append(True))
    teacher_logits = torch.randn(4, 32, dtype=torch.float64)

    def objective(hidden, head):
        return head(
            hidden.squeeze(0),
            full_vocab_callback=lambda h, w: full_reverse_kl(
                h, w, teacher_logits, torch.eye(32, dtype=torch.float64), token_chunk=2
            ),
        )

    loss = model(ids, full_vocab_callback=objective)
    lp = plain.squeeze(0).log_softmax(-1)
    expected = (lp.exp() * (lp - teacher_logits.log_softmax(-1))).sum()
    parameters = tuple(model.parameters())
    for a, b in zip(torch.autograd.grad(loss, parameters), torch.autograd.grad(expected, parameters), strict=True):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)
    assert len(calls) == 1
    hook.remove()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 acceptance")
def test_real_cuda_bf16_head_hidden_vjp():
    torch.manual_seed(31)
    h, w = [
        (torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.2).requires_grad_()
        for shape in [(5, 16), (8193, 16)]
    ]
    t, u = [torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.2 for shape in [(5, 12), (8196, 12)]]
    actual = full_reverse_kl(h, w, t, u, token_chunk=2)
    # FP64 starts from the same BF16 hidden/head inputs, including teacher extra rows.
    expected = reference(h.double(), w.double(), t.double(), u.double())
    torch.testing.assert_close(actual.double(), expected, atol=2e-5, rtol=3e-4)
    for a, b in zip(torch.autograd.grad(actual, (h, w)), torch.autograd.grad(expected, (h, w)), strict=True):
        relative = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)
        assert relative < 0.04


@pytest.mark.parametrize("tied", [False, True])
def test_production_engine_hidden_capture_and_head_loss_on_real_cpu_models(tmp_path, tied):
    from verl.trainer.distillation.hidden import FrozenTeacherHead
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

    torch.manual_seed(37)
    student = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            tie_word_embeddings=tied,
        )
    ).double()
    teacher = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=35,
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            tie_word_embeddings=tied,
        )
    ).double()
    teacher.save_pretrained(tmp_path / "teacher")
    ids = torch.tensor([[3, 4, 5, 6, 7]])
    lp = student(ids).logits[0].log_softmax(-1)[[1, 3]]
    lt = teacher(ids).logits[0].detach().log_softmax(-1)[[1, 3], :32]
    expected = (lp.exp() * (lp - lt)).sum()
    for model in (student, teacher):
        install_full_vocab_adapter(model)

    def engine(model):
        result = object.__new__(FSDPEngineWithLMHead)
        result.module = model
        result.model_config = SimpleNamespace(enable_full_vocab=True)
        return result

    source, actor = engine(teacher), engine(student)
    batch = TensorDict(
        {
            "input_ids": torch.nested.nested_tensor([ids[0]], layout=torch.jagged),
            "loss_mask": torch.nested.nested_tensor([torch.tensor([1, 0, 1])], layout=torch.jagged),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(batch, temperature=1.0)
    with torch.no_grad():
        labels = source._full_vocab_output(batch, {"input_ids": ids}, teacher=True)
    assert labels["teacher_positions"].unbind()[0].tolist() == [1, 3]
    batch.update(labels)
    actor.full_vocab_config = SimpleNamespace(token_chunk_size=1)
    actor.full_vocab_teacher_head = FrozenTeacherHead(str(tmp_path / "teacher"))
    output = actor._full_vocab_output(batch, {"input_ids": ids}, teacher=False)
    assert output["full_kl_count"] == 2
    torch.testing.assert_close(output["full_kl_sum"], expected)
    parameters = tuple(student.parameters())
    for actual, wanted in zip(
        torch.autograd.grad(output["full_kl_sum"], parameters), torch.autograd.grad(expected, parameters), strict=True
    ):
        torch.testing.assert_close(actual, wanted, atol=1e-10, rtol=1e-9)
