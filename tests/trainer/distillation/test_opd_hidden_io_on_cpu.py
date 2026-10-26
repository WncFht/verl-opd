# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from tensordict import NonTensorStack, TensorDict

from verl.trainer.distillation.hidden_io import HiddenBatchIO
from verl.utils import tensordict_utils as tu
from verl.utils import transferqueue_utils as tqu


def batch(keys):
    data = TensorDict({"sample_keys": NonTensorStack(*keys)}, batch_size=[len(keys)])
    tu.assign_non_tensor(data, sample_partition="test")
    return data


def payload(value):
    return TensorDict(
        {
            "teacher_hidden": torch.nested.nested_tensor([torch.full((1, 3), value)], layout=torch.jagged),
            "teacher_positions": torch.nested.nested_tensor([torch.tensor([1])], layout=torch.jagged),
        },
        batch_size=[1],
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_writer_reader_preserve_key_order(asynchronous):
    store = {}

    def put(*, keys, partition_id, fields):
        assert partition_id == "test"
        store[keys[0]] = fields.clone()

    def get(*, keys, partition_id, select_fields):
        return store[keys[0]].select(*select_fields)

    data = batch(["a", "b"])
    tu.assign_non_tensor(data, full_vocab_teacher=True, full_vocab_async_write=asynchronous)
    batches = [batch(["a"]), batch(["b"])]
    with HiddenBatchIO(data, batches, get=get, put=put) as writer:
        for i, mb in enumerate(writer):
            out = {"model_output": dict(payload(i).items())}
            writer.publish(out, mb)
            assert "teacher_hidden" not in out["model_output"]
    tu.assign_non_tensor(data, full_vocab_teacher=False, full_vocab_train=True, full_vocab_prefetch=asynchronous)
    with HiddenBatchIO(data, list(reversed(batches)), get=get, put=put) as reader:
        result = [
            mb["teacher_hidden"].values().item()
            if mb["teacher_hidden"].values().numel() == 1
            else mb["teacher_hidden"].values()[0, 0].item()
            for mb in reader
        ]
    assert result == [1, 0]
    assert all("teacher_hidden" not in mb for mb in batches)


def test_async_failures_are_not_lost():
    data = batch(["a"])
    tu.assign_non_tensor(data, full_vocab_teacher=True, full_vocab_async_write=True)

    def fail(**kwargs):
        raise RuntimeError("write failed")

    with pytest.raises(RuntimeError, match="write failed"):
        with HiddenBatchIO(data, [data], get=lambda **kw: None, put=fail) as writer:
            writer.publish({"model_output": dict(payload(0).items())}, data)
    assert not any(t.name.startswith("full-kl-hidden") for t in threading.enumerate())


def test_prefetch_reads_next_batch_while_current_is_consumed():
    started = threading.Event()
    release = threading.Event()
    data = batch(["a", "b"])
    tu.assign_non_tensor(data, full_vocab_train=True, full_vocab_prefetch=True)

    def get(*, keys, **kwargs):
        if keys == ["b"]:
            started.set()
            if not release.wait(timeout=5):
                raise RuntimeError("prefetch consumer did not release read")
        return payload(1)

    try:
        with HiddenBatchIO(data, [batch(["a"]), batch(["b"])], get=get, put=lambda **kw: None) as reader:
            it = iter(reader)
            next(it)
            assert started.wait(timeout=5)
            release.set()
            next(it)
            with pytest.raises(StopIteration):
                next(it)
    finally:
        release.set()


def test_prefetch_read_error_propagates():
    data = batch(["a"])
    tu.assign_non_tensor(data, full_vocab_train=True, full_vocab_prefetch=True)

    def fail(**kwargs):
        raise RuntimeError("read failed")

    with pytest.raises(RuntimeError, match="read failed"):
        with HiddenBatchIO(data, [data], get=fail, put=lambda **kw: None) as reader:
            list(reader)
    assert not any(t.name.startswith("full-kl-hidden") for t in threading.enumerate())


@pytest.mark.parametrize("role", ["full_vocab_teacher", "full_vocab_train", "unrelated_consumer"])
def test_index_metadata_retains_sample_keys(monkeypatch, role):
    meta = SimpleNamespace(
        size=2,
        partition_ids=["train", "train"],
        global_indexes=[8, 2],
        extra_info={role: True, "retain_sample_identity": True},
    )
    client = SimpleNamespace(
        async_kv_retrieve_keys=AsyncMock(return_value=["b", "a"]),
        async_get_data=AsyncMock(return_value=TensorDict({"input_ids": torch.ones(2, 2)}, batch_size=[2])),
    )
    monkeypatch.setattr(tqu.tq, "get_client", lambda: client)
    result = asyncio.run(tqu._async_meta_to_realdata(meta))
    assert list(result["sample_keys"]) == ["b", "a"]
    assert tu.get_non_tensor_data(result, "sample_partition", default=None) == "train"
    client.async_kv_retrieve_keys.assert_awaited_once_with(global_indexes=[8, 2], partition_id="train")
