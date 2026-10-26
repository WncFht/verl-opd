# SPDX-License-Identifier: Apache-2.0
"""Two-GPU production worker acceptance: teacher, Full-KL reuse and exact resume.

Run with torchrun --standalone --nproc-per-node=2 on allocated idle GPUs.
"""

import argparse
import copy
import inspect
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from tensordict import TensorDict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from verl.trainer.distillation.losses import distillation_loss
from verl.utils import tensordict_utils as tu
from verl.utils.distributed import initialize_global_process_group
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig, TrainingWorkerConfig
from verl.workers.config.distillation import DistillationLossConfig
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker


def nested(rows):
    return torch.nested.nested_tensor(rows, layout=torch.jagged)


def worker(path, world, teacher):
    config = TrainingWorkerConfig(
        model_type="language_model",
        model_config=HFModelConfig(
            path=str(path), enable_gradient_checkpointing=False, enable_full_vocab=True, use_fused_kernels=False
        ),
        engine_config=FSDPEngineConfig(
            strategy="fsdp2",
            forward_only=teacher,
            forward_only_cpu_offload=False,
            model_dtype="bfloat16" if teacher else "float32",
            fsdp_size=world,
            param_offload=True,
            optimizer_offload=not teacher,
            use_torch_compile=False,
            infer_max_token_len_per_gpu=8,
            max_token_len_per_gpu=8,
            use_no_sync_for_gradient_accumulation=False,
        ),
        optimizer_config=None if teacher else FSDPOptimizerConfig(lr=1e-3, total_training_steps=6),
    )
    result = TrainingWorker(config)
    result.reset()
    return result


def snapshot(actor):
    def local(value):
        if isinstance(value, torch.Tensor):
            if hasattr(value, "to_local"):
                value = value.to_local()
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: local(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(local(v) for v in value)
        return copy.deepcopy(value)

    return local(
        {
            "model": actor.engine.module.state_dict(),
            "optimizer": actor.engine.optimizer.state_dict(),
            "scheduler": actor.engine.lr_scheduler.state_dict(),
        }
    )


def equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            equal(x, y)
    else:
        assert a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tied", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    _, rank, world = initialize_global_process_group(timeout_second=180)
    assert world == 2
    paths = [tempfile.mkdtemp(prefix="opd-full-fsdp-") if rank == 0 else None]
    dist.broadcast_object_list(paths)
    root = Path(paths[0])
    if rank == 0:
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(
                WordLevel({"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, **{f"t{i}": i for i in range(3, 32)}}, unk_token="[UNK]")
            ),
            pad_token="[PAD]",
            eos_token="[EOS]",
            unk_token="[UNK]",
        )
        for name, vocab, hidden in [("student", 32, 32), ("teacher", 35, 48)]:
            torch.manual_seed(41 if name == "student" else 43)
            model = Qwen2ForCausalLM(
                Qwen2Config(
                    vocab_size=vocab,
                    hidden_size=hidden,
                    intermediate_size=64,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    tie_word_embeddings=args.tied,
                    attention_dropout=0.0,
                    eos_token_id=1,
                    pad_token_id=0,
                )
            )
            model.save_pretrained(root / name)
            tokenizer.save_pretrained(root / name)
    dist.barrier()
    actor = worker(root / "student", world, False)
    teacher = worker(root / "teacher", world, True)
    loss_config = DistillationLossConfig(
        loss_mode="full_reverse_kl",
        use_policy_gradient=False,
        use_task_rewards=False,
        full_vocab={"token_chunk_size": 1},
        loss_max_clamp=None,
        log_prob_min_clamp=None,
    )
    actor.engine.configure_full_vocab(loss_config.full_vocab, str(root / "teacher"))

    # The worker loss ABI includes dp_group; direct callback keeps the same loss.
    def callback(model_output, data, dp_group):
        return distillation_loss(
            SimpleNamespace(loss_agg_mode="token-mean", loss_scale_factor=None),
            SimpleNamespace(distillation_loss=loss_config),
            model_output,
            data,
        )

    actor.set_loss_fn(callback)
    base = TensorDict(
        {
            "input_ids": nested([torch.tensor([3, 4, 5, 6, 7, 8]), torch.tensor([9, 10, 11, 12])]),
            "position_ids": nested([torch.arange(6), torch.arange(4)]),
            "prompts": nested([torch.tensor([3, 4, 5]), torch.tensor([9, 10])]),
            "responses": nested([torch.tensor([6, 7, 8]), torch.tensor([11, 12])]),
            "loss_mask": nested(
                [torch.tensor([1, 0, 1]), torch.ones(2, dtype=torch.long)]
                if rank == 0
                else [torch.zeros(3, dtype=torch.long), torch.zeros(2, dtype=torch.long)]
            ),
        },
        batch_size=[2],
    )
    base["response_mask"] = base["loss_mask"].clone()
    before = snapshot(actor)
    records = []

    def outer():
        batch = base.clone()
        tu.assign_non_tensor(
            batch, temperature=1.0, compute_loss=False, full_vocab_teacher=True, calculate_entropy=False
        )
        labels = teacher.infer_batch(batch)
        assert all(p.device.type == "cpu" for p in teacher.engine.module.parameters())
        batch.update(labels.select("teacher_hidden", "teacher_positions"))
        tu.assign_non_tensor(
            batch,
            full_vocab_teacher=False,
            full_vocab_train=True,
            epochs=2,
            mini_batch_size=world * 2,
            global_batch_size=world * 2,
            seed=42,
            dataloader_kwargs={"shuffle": True},
            calculate_entropy=False,
        )
        result = inspect.unwrap(ActorRolloutRefWorker.update_actor)(SimpleNamespace(actor=actor), batch)
        assert actor.engine.full_vocab_teacher_head.weight is None
        assert all(p.device.type == "cpu" for p in actor.engine.module.parameters())
        values = tu.get(result, "metrics")["loss"]
        assert len(values) == 2 and all(torch.isfinite(torch.tensor(v)) for v in values)
        records.append([float(v) for v in values])

    outer()
    actor.save_checkpoint(str(root / "checkpoint"), None, global_step=1, max_ckpt_to_keep=1)
    outer()
    expected = snapshot(actor)
    assert any(not torch.equal(before["model"][k], expected["model"][k]) for k in before["model"])
    actor.load_checkpoint(str(root / "checkpoint"), None)
    outer()
    equal(snapshot(actor), expected)
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "world_size": world,
                    "tied": args.tied,
                    "teacher_extra_rows": 3,
                    "empty_rank": True,
                    "mask_hole": True,
                    "updates_per_outer": 2,
                    "loss": records,
                    "resume_model_optimizer_scheduler_exact": True,
                },
                indent=2,
            )
            + "\n"
        )
    dist.barrier()
    if rank == 0:
        shutil.rmtree(root)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
