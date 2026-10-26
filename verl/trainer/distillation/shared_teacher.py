# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared-sync scoring protocol; standalone teacher managers remain independent."""

import logging

import torch
from tensordict import TensorDict
from transfer_queue import KVBatchMeta

from verl.utils.debug import marked_timer

logger = logging.getLogger(__name__)


class SharedTeacherScorer:
    def __init__(self, worker, *, payload_kind, get, put, wait, async_write=False):
        if payload_kind not in ("sampled", "hidden"):
            raise ValueError("shared-sync scoring supports sampled or hidden payloads")
        self.worker = worker
        self.payload_kind = payload_kind
        self.async_write = async_write
        self.get, self.put, self.wait = get, put, wait

    def score(self, batch: KVBatchMeta, timing_raw: dict) -> KVBatchMeta:
        """Score a completed rollout batch, release teacher, then publish aligned labels."""
        teacher_batch = KVBatchMeta(
            keys=list(batch.keys),
            tags=list(batch.tags),
            partition_id=batch.partition_id,
            fields=["input_ids", "position_ids", "loss_mask"],
            extra_info={
                "temperature": 1.0,
                "compute_loss": False,
                "calculate_entropy": False,
                "distillation_use_topk": False,
                "disable_auto_offload": True,
                "full_vocab_teacher": self.payload_kind == "hidden",
                "full_vocab_async_write": self.async_write,
                "retain_sample_identity": self.payload_kind == "hidden",
            },
        )
        try:
            with marked_timer("teacher_transfer", timing_raw):
                self.worker.to("device", model=True, optimizer=False, grad=False)
            with marked_timer("teacher_forward", timing_raw):
                output = self.worker.infer_batch(teacher_batch)
                # KVBatchMeta is not supported by DataProtoFuture.get().
                self.wait(output.futures)
        except BaseException:
            # Preserve the primary error. Distributed process failures are fatal;
            # this is best-effort cleanup, not a recovery/barrier protocol.
            try:
                self.worker.to("cpu", model=True, optimizer=False, grad=False)
            except Exception:
                logger.exception("Shared teacher cleanup failed after scoring failure")
            raise
        else:
            with marked_timer("teacher_transfer", timing_raw):
                self.worker.to("cpu", model=True, optimizer=False, grad=False)

        if self.payload_kind == "hidden":
            # Workers publish the compact hidden payload directly through the TQ bridge.
            return batch

        data = self.get(keys=batch.keys, partition_id=batch.partition_id, select_fields=["input_ids", "log_probs"])
        teacher_ids, teacher_logprobs = [], []
        if len(data) != len(batch):
            raise ValueError("Incomplete shared teacher output batch")
        for tokens, log_probs in zip(data["input_ids"].unbind(), data["log_probs"].unbind(), strict=True):
            if log_probs.shape != tokens.shape or tokens.numel() == 0:
                raise ValueError("Shared teacher output must match each full token sequence")
            probs = log_probs.float().clone()
            if not torch.isfinite(probs[:-1]).all():
                raise ValueError("Shared teacher returned non-finite token log probabilities")
            # FSDP already scores x[i+1] at position i. Only the final dummy is replaced.
            probs[-1] = 0
            ids = torch.cat((tokens[1:].to(torch.int32), tokens.new_zeros(1, dtype=torch.int32)))
            teacher_ids.append(ids.unsqueeze(-1))
            teacher_logprobs.append(probs.unsqueeze(-1))
        fields = TensorDict(
            {
                "teacher_ids": torch.nested.nested_tensor(teacher_ids, layout=torch.jagged),
                "teacher_logprobs": torch.nested.nested_tensor(teacher_logprobs, layout=torch.jagged),
            },
            batch_size=[len(batch)],
        )
        self.put(keys=batch.keys, partition_id=batch.partition_id, fields=fields)
        return batch
