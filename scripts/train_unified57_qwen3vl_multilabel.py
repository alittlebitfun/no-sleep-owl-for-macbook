#!/usr/bin/env python3
"""Train the fixed Bosideng Unified57 PN/PU classifier on 8 GPUs.

The training contract deliberately has two separate state-loading paths:

* ``--init-from-aggregate18`` copies only the 288 LoRA tensors and the 18
  overlapping classifier rows from a format-v2 JD18 checkpoint.  Optimizer,
  RNG, step, and sampler state start fresh.
* ``--resume`` accepts only a format-v3 Unified57 checkpoint and restores the
  optimizer, per-rank RNG state, global-batch cursor, and sampling statistics.

Each per-GPU microbatch is a single forward containing six uniform records and
two dictionary-balanced records by default.  PU positives may come from either
stream; PU unlabeled ranking references can come only from the uniform stream.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import itertools
import json
import math
import os
import random
import signal
import statistics
import tempfile
import time
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.distributed as dist
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

try:
    from scripts.build_unified57_masked_dataset import (
        DEFAULT_SCHEMA_PATH,
        load_schema,
    )
except ModuleNotFoundError:  # Support direct execution from scripts/.
    from build_unified57_masked_dataset import (  # type: ignore[no-redef]
        DEFAULT_SCHEMA_PATH,
        load_schema,
    )


FORMAT_VERSION = 3
EXPECTED_LORA_TENSORS = 288
DEFAULT_IMAGE_MAX_PIXELS = 336 * 336
DEFAULT_MICRO_BATCH_SIZE = 8
DEFAULT_UNIFORM_PER_RANK = 6
DEFAULT_BALANCED_PER_RANK = 2
DEFAULT_PU_MARGIN = 1.0
DEFAULT_PU_LOSS_WEIGHT = 0.2
DEFAULT_MAX_STEPS = 20
LOSS_CONTRACT_VERSION = "unified57_pn_sample_mean_bce_pairwise_pu_v1"
MASK_CONTRACT_VERSION = "unified57_known_pu_positive_disjoint_v1"
SAMPLER_CONTRACT_VERSION = "unified57_two_stream_binding_weighted_v1"
PU_OUTPUT_SEMANTICS = "uncalibrated_confidence"

AGGREGATE18_CONFIG_SHA256 = (
    "651d8163065f78a46239453a0d5776d47087f61036c11321a40e465a0d0fe29b"
)
AGGREGATE18_CHECKPOINT_SHA256 = (
    "e7cc9f8464498ce883d39ef1d7d7eaf4cbd51546931ae7c4b114131055ba46f9"
)
AGGREGATE18_TAG_ORDER = (
    "连帽",
    "毛领",
    "立领",
    "翻领",
    "无领",
    "H型",
    "O型",
    "X型",
    "A型",
    "长款",
    "中款",
    "短款",
    "压胶充绒",
    "压胶袋盖",
    "压胶门襟",
    "平行绗线",
    "菱形绗线",
    "葫芦型绗线",
)
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")

VISION_PROMPT = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "识别图中服装的可见结构、工艺和面辅料属性。<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_unified57_schema(path: Path | str = DEFAULT_SCHEMA_PATH) -> dict:
    """Load the shared schema and reassert the trainer-specific mode split."""
    schema = load_schema(path)
    modes = Counter(schema["label_training_modes"].values())
    if modes != {"pn": 36, "pu": 20, "unsupported": 1}:
        raise ValueError("Unified57 training requires 36 PN, 20 PU, and 1 unsupported")
    if schema["unsupported_labels"] != ["假两件"]:
        raise ValueError("假两件 must be the sole unsupported Unified57 label")
    return schema


def _unwrap_model(model: nn.Module) -> nn.Module:
    while hasattr(model, "module"):
        model = model.module  # type: ignore[assignment]
    return model


def _load_torch_payload(
    checkpoint: Path | str | Mapping[str, object],
    *,
    expected_sha256: str | None = None,
) -> tuple[dict, str | None]:
    if isinstance(checkpoint, Mapping):
        if expected_sha256 is not None:
            raise ValueError("cannot verify SHA256 for an in-memory checkpoint mapping")
        return dict(checkpoint), None
    path = Path(checkpoint)
    checkpoint_sha256 = _sha256_file(path)
    if expected_sha256 is not None and checkpoint_sha256 != expected_sha256:
        raise ValueError(
            "Aggregate18 checkpoint SHA256 mismatch: "
            f"expected={expected_sha256}, actual={checkpoint_sha256}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a dictionary payload")
    return payload, checkpoint_sha256


def validate_aggregate18_config(path: Path | str) -> dict:
    """Fail closed unless the audited Aggregate18 delivery contract is exact."""
    config_path = Path(path)
    actual_hash = _sha256_file(config_path)
    if actual_hash != AGGREGATE18_CONFIG_SHA256:
        raise ValueError(
            "Aggregate18 config SHA256 mismatch: "
            f"expected={AGGREGATE18_CONFIG_SHA256}, actual={actual_hash}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_fields = {
        "hidden_size": 4096,
    }
    for field, expected in expected_fields.items():
        if config.get(field) != expected:
            raise ValueError(f"Aggregate18 config {field} must be {expected!r}")
    lora = config.get("lora") or {}
    if (
        lora.get("rank") != 16
        or lora.get("alpha") != 32
        or tuple(lora.get("target_modules") or ()) != LORA_TARGET_MODULES
    ):
        raise ValueError("Aggregate18 LoRA rank/alpha/target_modules drifted")
    classifier = config.get("classifier") or {}
    if classifier.get("num_labels") != 18:
        raise ValueError("Aggregate18 config must describe an 18-row classifier")
    if tuple(config.get("tag_order") or ()) != AGGREGATE18_TAG_ORDER:
        raise ValueError("Aggregate18 tag_order differs from the audited explicit order")
    return {"sha256": actual_hash, "config": config}


def transfer_aggregate18_v2_checkpoint(
    model: nn.Module,
    checkpoint: Path | str | Mapping[str, object],
    *,
    target_tag_order: Sequence[str],
    source_tag_order: Sequence[str] = AGGREGATE18_TAG_ORDER,
    expected_lora_tensors: int = EXPECTED_LORA_TENSORS,
    expected_checkpoint_sha256: str | None = None,
) -> dict:
    """Copy audited v2 weights without importing any training state.

    All LoRA keys and shapes must match exactly.  Classifier rows are copied by
    explicit tag name, leaving the normal initialization of the other 39 rows
    byte-for-byte unchanged.
    """
    if not isinstance(checkpoint, Mapping) and expected_checkpoint_sha256 is None:
        raise ValueError("an expected Aggregate18 checkpoint SHA256 is required")
    payload, checkpoint_sha256 = _load_torch_payload(
        checkpoint,
        expected_sha256=expected_checkpoint_sha256,
    )
    if payload.get("format_version") != 2:
        raise ValueError("Aggregate18 initialization requires checkpoint format_version=2")
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("Aggregate18 v2 checkpoint is missing its model mapping")
    state = dict(state)
    declared = payload.get("trainable_names")
    if declared is not None and set(declared) != set(state):
        raise ValueError("Aggregate18 trainable_names do not match the model payload")

    source_tags = tuple(source_tag_order)
    target_tags = tuple(target_tag_order)
    if source_tags != AGGREGATE18_TAG_ORDER:
        raise ValueError("source_tag_order differs from the audited Aggregate18 order")
    if len(target_tags) != 57 or len(set(target_tags)) != 57:
        raise ValueError("target_tag_order must contain 57 unique tags")
    if any(tag not in target_tags for tag in source_tags):
        raise ValueError("all Aggregate18 tags must exist in the Unified57 schema")

    user_model = _unwrap_model(model)
    parameters = dict(user_model.named_parameters())
    source_lora = {name: value for name, value in state.items() if "lora_" in name}
    target_lora = {
        name: parameter for name, parameter in parameters.items() if "lora_" in name
    }
    if len(source_lora) != expected_lora_tensors:
        raise ValueError(
            f"Aggregate18 checkpoint must contain exactly {expected_lora_tensors} LoRA "
            f"tensors; found {len(source_lora)}"
        )
    if len(target_lora) != expected_lora_tensors:
        raise ValueError(
            f"Unified57 model must expose exactly {expected_lora_tensors} LoRA tensors; "
            f"found {len(target_lora)}"
        )
    if set(source_lora) != set(target_lora):
        missing = sorted(set(target_lora) - set(source_lora))
        unexpected = sorted(set(source_lora) - set(target_lora))
        raise ValueError(
            "Aggregate18 LoRA exact-key mismatch: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    allowed_source_keys = set(source_lora) | {"classifier.weight", "classifier.bias"}
    if set(state) != allowed_source_keys:
        raise ValueError(
            "Aggregate18 model payload must contain only the 288 LoRA tensors and "
            f"classifier weight/bias; unexpected={sorted(set(state) - allowed_source_keys)}"
        )
    for name in sorted(source_lora):
        source_tensor = source_lora[name]
        if not torch.is_tensor(source_tensor):
            raise ValueError(f"LoRA payload {name} is not a tensor")
        if source_tensor.shape != target_lora[name].shape:
            raise ValueError(
                f"LoRA shape mismatch for {name}: source={tuple(source_tensor.shape)}, "
                f"target={tuple(target_lora[name].shape)}"
            )

    classifier_weight = parameters.get("classifier.weight")
    classifier_bias = parameters.get("classifier.bias")
    source_weight = state.get("classifier.weight")
    source_bias = state.get("classifier.bias")
    if classifier_weight is None or classifier_bias is None:
        raise ValueError("Unified57 model must expose classifier.weight and classifier.bias")
    if not torch.is_tensor(source_weight) or not torch.is_tensor(source_bias):
        raise ValueError("Aggregate18 checkpoint is missing classifier weight/bias tensors")
    expected_weight_shape = (18, classifier_weight.shape[1])
    if tuple(source_weight.shape) != expected_weight_shape:
        raise ValueError(
            "Aggregate18 classifier.weight shape mismatch: "
            f"expected={expected_weight_shape}, actual={tuple(source_weight.shape)}"
        )
    if tuple(source_bias.shape) != (18,):
        raise ValueError("Aggregate18 classifier.bias shape mismatch; expected (18,)")
    if tuple(classifier_weight.shape) != (57, source_weight.shape[1]):
        raise ValueError("Unified57 classifier.weight must have 57 rows and matching width")
    if tuple(classifier_bias.shape) != (57,):
        raise ValueError("Unified57 classifier.bias must have 57 rows")

    mapping = [
        {
            "tag": tag,
            "source_index": source_index,
            "target_index": target_tags.index(tag),
        }
        for source_index, tag in enumerate(source_tags)
    ]
    mapped_targets = {entry["target_index"] for entry in mapping}
    if len(mapping) != 18 or len(mapped_targets) != 18:
        raise ValueError("Aggregate18 classifier overlap must contain exactly 18 rows")
    untouched = sorted(set(range(57)) - mapped_targets)
    untouched_weight = classifier_weight.detach()[untouched].clone()
    untouched_bias = classifier_bias.detach()[untouched].clone()

    with torch.no_grad():
        for name, source_tensor in source_lora.items():
            target_lora[name].copy_(source_tensor)
        for entry in mapping:
            source_index = int(entry["source_index"])
            target_index = int(entry["target_index"])
            classifier_weight[target_index].copy_(source_weight[source_index])
            classifier_bias[target_index].copy_(source_bias[source_index])

    if untouched:
        torch.testing.assert_close(classifier_weight.detach()[untouched], untouched_weight)
        torch.testing.assert_close(classifier_bias.detach()[untouched], untouched_bias)
    for entry in mapping:
        source_index = int(entry["source_index"])
        target_index = int(entry["target_index"])
        torch.testing.assert_close(
            classifier_weight.detach()[target_index], source_weight[source_index]
        )
        torch.testing.assert_close(
            classifier_bias.detach()[target_index], source_bias[source_index]
        )

    return {
        "source_format_version": 2,
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_global_step": payload.get("global_step"),
        "source_optimizer_present": "optimizer" in payload,
        "source_rng_present": "rng_state" in payload or "cuda_rng_state" in payload,
        "copied_lora_tensors": len(source_lora),
        "classifier_row_mapping": mapping,
        "untouched_initialized_rows": untouched,
        "optimizer_loaded": False,
        "fresh_training_state": {
            "global_step": 0,
            "global_batch_cursor": 0,
            "optimizer": "fresh",
            "rng": "fresh",
        },
    }


def build_pairwise_pu_masks(
    *,
    known_mask: torch.Tensor,
    pu_positive_mask: torch.Tensor,
    uniform_stream_mask: torch.Tensor,
    pu_indices: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select P from both streams and U exclusively from uniform slots."""
    if known_mask.shape != pu_positive_mask.shape or known_mask.ndim != 2:
        raise ValueError("known_mask and pu_positive_mask must share a 2D shape")
    if uniform_stream_mask.ndim != 1 or uniform_stream_mask.shape[0] != known_mask.shape[0]:
        raise ValueError("uniform_stream_mask must contain one value per sample")
    known = known_mask.to(dtype=torch.bool)
    positive_all = pu_positive_mask.to(device=known.device, dtype=torch.bool)
    if torch.any(known & positive_all):
        raise ValueError("known_mask and pu_positive_mask must remain disjoint")
    indexes = torch.as_tensor(list(pu_indices), device=known.device, dtype=torch.long)
    positives = positive_all.index_select(1, indexes)
    known_pu = known.index_select(1, indexes)
    uniform = uniform_stream_mask.to(device=known.device, dtype=torch.bool).unsqueeze(1)
    unlabeled = uniform & ~known_pu & ~positives
    return positives, unlabeled


def pairwise_positive_unlabeled_ranking_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    unlabeled_mask: torch.Tensor,
    *,
    margin: float = DEFAULT_PU_MARGIN,
) -> tuple[torch.Tensor, dict[str, list[int]]]:
    """FP32 hinge ranking over positive-vs-unlabeled pairs, averaged by tag."""
    if logits.shape != positive_mask.shape or logits.shape != unlabeled_mask.shape:
        raise ValueError("PU logits, positive_mask, and unlabeled_mask must share a shape")
    if logits.ndim != 2:
        raise ValueError("PU ranking expects [batch, pu_labels] logits")
    if margin <= 0:
        raise ValueError("PU pairwise margin must be positive")
    positive = positive_mask.to(device=logits.device, dtype=torch.bool)
    unlabeled = unlabeled_mask.to(device=logits.device, dtype=torch.bool)
    if torch.any(positive & unlabeled):
        raise ValueError("a PU cell cannot be both positive and unlabeled")
    scores = logits.float()
    losses: list[torch.Tensor] = []
    positive_counts: list[int] = []
    unlabeled_counts: list[int] = []
    pair_counts: list[int] = []
    for label_index in range(scores.shape[1]):
        positive_scores = scores[:, label_index][positive[:, label_index]]
        unlabeled_scores = scores[:, label_index][unlabeled[:, label_index]]
        p_count = int(positive_scores.numel())
        u_count = int(unlabeled_scores.numel())
        positive_counts.append(p_count)
        unlabeled_counts.append(u_count)
        pair_counts.append(p_count * u_count)
        if p_count and u_count:
            pair_margin = (
                float(margin)
                - positive_scores.unsqueeze(1)
                + unlabeled_scores.unsqueeze(0)
            )
            losses.append(F.relu(pair_margin).mean())
    loss = torch.stack(losses).mean() if losses else scores.sum() * 0.0
    return loss, {
        "positive_counts": positive_counts,
        "unlabeled_counts": unlabeled_counts,
        "pair_counts": pair_counts,
    }


@dataclass(frozen=True)
class SampleReference:
    record_index: int
    stream: str
    balanced_label_index: int | None = None
    padding: bool = False
    fallback: bool = False


class TwoStreamSchedule:
    """Deterministic global schedule, later sharded rank-major without drift."""

    def __init__(
        self,
        records: Sequence[Mapping[str, object]],
        *,
        tag_order: Sequence[str],
        training_modes: Mapping[str, str],
        world_size: int = 8,
        uniform_per_rank: int = DEFAULT_UNIFORM_PER_RANK,
        balanced_per_rank: int = DEFAULT_BALANCED_PER_RANK,
        seed: int = 20260717,
    ) -> None:
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if uniform_per_rank <= 0 or balanced_per_rank < 0:
            raise ValueError("two-stream quotas must be non-negative with uniform > 0")
        if not records:
            raise ValueError("two-stream schedule requires at least one training record")
        record_ids = [record.get("record_id") for record in records]
        if any(not isinstance(record_id, str) or not record_id for record_id in record_ids):
            raise ValueError("every scheduled record must have a non-empty record_id")
        duplicate_ids = sorted(
            record_id for record_id, count in Counter(record_ids).items() if count > 1
        )
        if duplicate_ids:
            raise ValueError(f"two-stream schedule has duplicate record_id values: {duplicate_ids[:5]}")
        tag_order = tuple(tag_order)
        if len(tag_order) != 57:
            raise ValueError("two-stream schedule requires the fixed 57-label order")
        self.records = records
        self.tag_order = tag_order
        self.world_size = world_size
        self.uniform_per_rank = uniform_per_rank
        self.balanced_per_rank = balanced_per_rank
        self.micro_batch_size = uniform_per_rank + balanced_per_rank
        if len(records) < world_size * self.micro_batch_size:
            raise ValueError(
                "training records must fill one deduplicated global two-stream batch"
            )
        self.seed = seed
        self.balanced_candidate_weights: dict[int, dict[int, float]] = {}
        for label_index, tag in enumerate(tag_order):
            if training_modes[tag] == "unsupported":
                continue
            candidates: dict[int, float] = {}
            for record_index, record in enumerate(records):
                raw_weight = record.get("dictionary_binding_count", 0)
                if isinstance(raw_weight, bool) or not isinstance(raw_weight, int):
                    raise ValueError("dictionary_binding_count must be a non-negative integer")
                if raw_weight <= 0:
                    continue
                labels = record.get("labels") or []
                known = record.get("known_mask") or []
                pu_positive = record.get("pu_positive_mask") or []
                is_positive = (
                    label_index < len(labels)
                    and labels[label_index] == 1.0
                    and (
                        (training_modes[tag] == "pn" and bool(known[label_index]))
                        or (
                            training_modes[tag] == "pu"
                            and bool(pu_positive[label_index])
                        )
                    )
                )
                if is_positive:
                    candidates[record_index] = float(raw_weight)
            if candidates:
                self.balanced_candidate_weights[label_index] = candidates

        uniform_global = world_size * uniform_per_rank
        balanced_global = world_size * balanced_per_rank
        permutation = list(range(len(records)))
        uniform_rng = random.Random(seed)
        uniform_rng.shuffle(permutation)
        step_count = math.ceil(len(permutation) / uniform_global)
        balanced_rng = random.Random(seed ^ 0xB057D57)
        balanced_labels = sorted(self.balanced_candidate_weights)
        all_indexes = list(range(len(records)))
        self.global_batches: list[list[SampleReference]] = []
        padding_count = 0
        balanced_fallback_count = 0
        balanced_cursor = 0

        for step in range(step_count):
            start = step * uniform_global
            real_uniform = permutation[start : start + uniform_global]
            uniform_references = [
                SampleReference(index, "uniform") for index in real_uniform
            ]
            used = set(real_uniform)
            if len(uniform_references) < uniform_global:
                for index in itertools.cycle(permutation):
                    if len(uniform_references) >= uniform_global:
                        break
                    if index in used and len(records) >= uniform_global:
                        continue
                    uniform_references.append(
                        SampleReference(index, "uniform", padding=True)
                    )
                    used.add(index)
                    padding_count += 1

            balanced_references: list[SampleReference] = []
            for _ in range(balanced_global):
                label_index = (
                    balanced_labels[balanced_cursor % len(balanced_labels)]
                    if balanced_labels
                    else None
                )
                balanced_cursor += 1
                selected: int | None = None
                if label_index is not None:
                    candidates = self.balanced_candidate_weights[label_index]
                    indexes = list(candidates)
                    weights = [candidates[index] for index in indexes]
                    for _attempt in range(max(8, len(indexes) * 2)):
                        proposal = balanced_rng.choices(indexes, weights=weights, k=1)[0]
                        if proposal not in used:
                            selected = proposal
                            break
                    if selected is None:
                        available = [index for index in indexes if index not in used]
                        if available:
                            selected = max(available, key=lambda index: candidates[index])
                fallback = selected is None
                if fallback:
                    available = [index for index in all_indexes if index not in used]
                    if not available:
                        available = all_indexes
                    selected = available[balanced_rng.randrange(len(available))]
                    balanced_fallback_count += 1
                used.add(selected)
                balanced_references.append(
                    SampleReference(
                        selected,
                        "balanced",
                        balanced_label_index=None if fallback else label_index,
                        fallback=fallback,
                    )
                )

            rank_major: list[SampleReference] = []
            for rank in range(world_size):
                rank_major.extend(
                    uniform_references[
                        rank * uniform_per_rank : (rank + 1) * uniform_per_rank
                    ]
                )
                rank_major.extend(
                    balanced_references[
                        rank * balanced_per_rank : (rank + 1) * balanced_per_rank
                    ]
                )
            self.global_batches.append(rank_major)

        self.config = {
            "contract_version": SAMPLER_CONTRACT_VERSION,
            "seed": seed,
            "world_size": world_size,
            "uniform_per_rank": uniform_per_rank,
            "balanced_per_rank": balanced_per_rank,
            "micro_batch_size_per_gpu": self.micro_batch_size,
            "global_batch_size": world_size * self.micro_batch_size,
            "uniform_records": len(records),
            "global_batch_count": len(self.global_batches),
            "uniform_padding_count": padding_count,
            "balanced_fallback_count": balanced_fallback_count,
            "weight_field": "dictionary_binding_count",
        }
        serialized = [
            [asdict(reference) for reference in batch] for batch in self.global_batches
        ]
        self.schedule_sha256 = hashlib.sha256(_canonical_json(serialized)).hexdigest()

    def rank_batches(
        self, rank: int, *, start_global_batch: int = 0
    ) -> "RankBatchSampler":
        return RankBatchSampler(self, rank, start_global_batch=start_global_batch)


class RankBatchSampler:
    def __init__(
        self,
        schedule: TwoStreamSchedule,
        rank: int,
        *,
        start_global_batch: int = 0,
    ) -> None:
        if not 0 <= rank < schedule.world_size:
            raise ValueError("rank is outside the schedule world size")
        if not 0 <= start_global_batch <= len(schedule.global_batches):
            raise ValueError("global batch cursor is outside the schedule")
        self.schedule = schedule
        self.rank = rank
        self.start_global_batch = start_global_batch

    def __iter__(self):
        micro = self.schedule.micro_batch_size
        start = self.rank * micro
        for batch in self.schedule.global_batches[self.start_global_batch :]:
            yield batch[start : start + micro]

    def __len__(self) -> int:
        return len(self.schedule.global_batches) - self.start_global_batch


def _optimizer_group_name_map(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> list[list[str]]:
    names_by_id = {id(parameter): name for name, parameter in _unwrap_model(model).named_parameters()}
    result: list[list[str]] = []
    for group in optimizer.param_groups:
        names: list[str] = []
        for parameter in group["params"]:
            name = names_by_id.get(id(parameter))
            if name is None:
                raise ValueError("optimizer contains a parameter absent from model.named_parameters")
            names.append(name)
        result.append(names)
    return result


def _trainable_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in _unwrap_model(model).named_parameters()
        if parameter.requires_grad
    }


def _capture_local_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def build_v3_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    tag_order: Sequence[str],
    schema_sha256: str,
    manifest_sha256: str,
    sampler_config: Mapping[str, object],
    sampler_schedule_sha256: str,
    cursor: Mapping[str, int],
    sampling_statistics: Mapping[str, object],
    world_size: int,
    rank_rng_states: Sequence[Mapping[str, object]],
    elapsed_seconds: float = 0.0,
    pos_weight_audit: Mapping[str, object] | None = None,
    initialization_audit: Mapping[str, object] | None = None,
    loss_contract: Mapping[str, object] | None = None,
    run_contract: Mapping[str, object] | None = None,
) -> dict:
    """Construct a trainable-only, fully resumable Unified57 format-v3 state."""
    tags = list(tag_order)
    if len(tags) != 57 or len(set(tags)) != 57:
        raise ValueError("v3 checkpoint tag_order must contain 57 unique tags")
    if len(rank_rng_states) != world_size:
        raise ValueError("v3 checkpoint needs one RNG payload per rank")
    trainable = _trainable_model_state(model)
    resolved_loss_contract = dict(
        loss_contract
        or {
            "version": LOSS_CONTRACT_VERSION,
            "pn": "per-sample masked BCE over known cells",
            "pu": "FP32 positive-vs-unlabeled pairwise hinge ranking",
            "pu_margin": DEFAULT_PU_MARGIN,
            "pu_loss_weight": DEFAULT_PU_LOSS_WEIGHT,
            "pu_unlabeled_stream": "uniform_only",
        }
    )
    return {
        "format_version": FORMAT_VERSION,
        "tag_order": tags,
        "schema_sha256": schema_sha256,
        "manifest_sha256": manifest_sha256,
        "mask_contract_version": MASK_CONTRACT_VERSION,
        "loss_contract": resolved_loss_contract,
        "run_contract": copy.deepcopy(dict(run_contract or {})),
        "pu_output_semantics": PU_OUTPUT_SEMANTICS,
        "model": trainable,
        "trainable_names": sorted(trainable),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "optimizer_group_name_map": _optimizer_group_name_map(model, optimizer),
        "cursor": dict(cursor),
        "sampler_config": dict(sampler_config),
        "sampler_schedule_sha256": sampler_schedule_sha256,
        "sampling_statistics": copy.deepcopy(dict(sampling_statistics)),
        "pos_weight_audit": copy.deepcopy(dict(pos_weight_audit or {})),
        "initialization_audit": copy.deepcopy(dict(initialization_audit or {})),
        "world_size": world_size,
        "rank_rng_states": list(rank_rng_states),
        "elapsed_seconds": float(elapsed_seconds),
    }


def _check_equal_metadata(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"{name} mismatch: checkpoint={actual!r}, expected={expected!r}")


def restore_v3_checkpoint(
    checkpoint: Path | str | Mapping[str, object],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_tag_order: Sequence[str],
    expected_schema_sha256: str,
    expected_manifest_sha256: str,
    expected_sampler_config: Mapping[str, object],
    expected_sampler_schedule_sha256: str,
    expected_world_size: int,
    rank: int,
    expected_loss_contract: Mapping[str, object] | None = None,
    expected_pos_weight_audit: Mapping[str, object] | None = None,
    expected_run_contract: Mapping[str, object] | None = None,
) -> dict:
    """Validate every resume contract before mutating model or optimizer state."""
    payload, _ = _load_torch_payload(checkpoint)
    _check_equal_metadata("format_version", payload.get("format_version"), FORMAT_VERSION)
    _check_equal_metadata("tag_order", payload.get("tag_order"), list(expected_tag_order))
    _check_equal_metadata(
        "schema_sha256", payload.get("schema_sha256"), expected_schema_sha256
    )
    _check_equal_metadata(
        "manifest_sha256", payload.get("manifest_sha256"), expected_manifest_sha256
    )
    _check_equal_metadata(
        "sampler_config", payload.get("sampler_config"), dict(expected_sampler_config)
    )
    _check_equal_metadata(
        "sampler_schedule_sha256",
        payload.get("sampler_schedule_sha256"),
        expected_sampler_schedule_sha256,
    )
    _check_equal_metadata("world_size", payload.get("world_size"), expected_world_size)
    _check_equal_metadata(
        "mask_contract_version",
        payload.get("mask_contract_version"),
        MASK_CONTRACT_VERSION,
    )
    if expected_loss_contract is not None:
        _check_equal_metadata(
            "loss_contract", payload.get("loss_contract"), dict(expected_loss_contract)
        )
    if expected_pos_weight_audit is not None:
        _check_equal_metadata(
            "pos_weight_audit",
            payload.get("pos_weight_audit"),
            dict(expected_pos_weight_audit),
        )
    if expected_run_contract is not None:
        _check_equal_metadata(
            "run_contract", payload.get("run_contract"), dict(expected_run_contract)
        )
    expected_optimizer_map = _optimizer_group_name_map(model, optimizer)
    _check_equal_metadata(
        "optimizer_group_name_map",
        payload.get("optimizer_group_name_map"),
        expected_optimizer_map,
    )

    user_model = _unwrap_model(model)
    current = {
        name: parameter
        for name, parameter in user_model.named_parameters()
        if parameter.requires_grad
    }
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("v3 checkpoint model payload is missing")
    declared = set(payload.get("trainable_names") or [])
    if declared != set(state) or set(state) != set(current):
        raise ValueError("v3 checkpoint trainable parameter names differ from the model")
    for name, parameter in current.items():
        source = state[name]
        if not torch.is_tensor(source) or source.shape != parameter.shape:
            raise ValueError(f"v3 model tensor shape mismatch for {name}")
    rng_states = payload.get("rank_rng_states")
    if not isinstance(rng_states, Sequence) or len(rng_states) != expected_world_size:
        raise ValueError("v3 checkpoint rank RNG state count differs from world size")
    if not 0 <= rank < len(rng_states):
        raise ValueError("rank has no RNG state in the v3 checkpoint")

    with torch.no_grad():
        for name, parameter in current.items():
            parameter.copy_(state[name])
    optimizer.load_state_dict(payload["optimizer"])
    rank_rng = rng_states[rank]
    if "cpu" in rank_rng:
        torch.set_rng_state(rank_rng["cpu"])
    if torch.cuda.is_available() and rank_rng.get("cuda"):
        torch.cuda.set_rng_state_all(rank_rng["cuda"])
    if "python" in rank_rng:
        random.setstate(rank_rng["python"])
    return payload


def load_v3_model_state_for_inference(
    checkpoint: Path | str | Mapping[str, object],
    *,
    model: nn.Module,
    expected_tag_order: Sequence[str],
    expected_schema_sha256: str,
) -> dict:
    """Load only trainable Unified57 tensors for evaluator/exporter reuse."""
    payload, _ = _load_torch_payload(checkpoint)
    _check_equal_metadata("format_version", payload.get("format_version"), FORMAT_VERSION)
    _check_equal_metadata("tag_order", payload.get("tag_order"), list(expected_tag_order))
    _check_equal_metadata(
        "schema_sha256", payload.get("schema_sha256"), expected_schema_sha256
    )
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("v3 checkpoint model payload is missing")
    user_model = _unwrap_model(model)
    trainable = {
        name: parameter
        for name, parameter in user_model.named_parameters()
        if parameter.requires_grad
    }
    if set(payload.get("trainable_names") or []) != set(state):
        raise ValueError("v3 checkpoint trainable_names differ from its model payload")
    if set(state) != set(trainable):
        raise ValueError("v3 checkpoint trainable names differ from inference model")
    for name, parameter in trainable.items():
        source = state[name]
        if not torch.is_tensor(source) or source.shape != parameter.shape:
            raise ValueError(f"v3 model tensor shape mismatch for {name}")
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(state[name])
    return payload


def _validate_binary_vector(
    value: object, *, name: str, width: int, floating: bool
) -> list[float] | list[int]:
    if not isinstance(value, list) or len(value) != width:
        raise ValueError(f"{name} must contain exactly {width} values")
    result: list[float] | list[int] = []
    for item in value:
        if item not in (0, 1, 0.0, 1.0, False, True):
            raise ValueError(f"{name} must contain binary values")
        result.append(float(item) if floating else int(item))
    return result


def validate_unified57_record(record: Mapping[str, object], schema: Mapping[str, object]) -> dict:
    """Validate a masked training row without ever materializing unknown negatives."""
    record_id = str(record.get("record_id") or "<missing>")
    if record.get("schema_sha256") != schema["schema_sha256"]:
        raise ValueError(f"{record_id}: schema_sha256 mismatch")
    labels = _validate_binary_vector(
        record.get("labels"), name=f"{record_id}.labels", width=57, floating=True
    )
    known = _validate_binary_vector(
        record.get("known_mask"),
        name=f"{record_id}.known_mask",
        width=57,
        floating=False,
    )
    pu_positive = _validate_binary_vector(
        record.get("pu_positive_mask"),
        name=f"{record_id}.pu_positive_mask",
        width=57,
        floating=False,
    )
    sources = record.get("sources")
    if not isinstance(sources, list) or not sources or not all(
        isinstance(source, str) and source for source in sources
    ):
        raise ValueError(f"{record_id}: sources must be a non-empty string list")
    for field in ("binding_count", "dictionary_binding_count", "jd_binding_count"):
        value = record.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{record_id}: {field} must be a non-negative integer")
    if record["binding_count"] != (
        record["dictionary_binding_count"] + record["jd_binding_count"]
    ):
        raise ValueError(f"{record_id}: binding counts do not add up")
    modes = schema["label_training_modes"]
    for index, tag in enumerate(schema["labels"]):
        if known[index] and pu_positive[index]:
            raise ValueError(f"{record_id}: known and PU masks overlap at {tag}")
        if known[index] and modes[tag] != "pn":
            raise ValueError(f"{record_id}: {tag} mode={modes[tag]} cannot enter PN BCE")
        if pu_positive[index]:
            if modes[tag] != "pu":
                raise ValueError(f"{record_id}: {tag} mode={modes[tag]} cannot enter PU loss")
            if labels[index] != 1.0:
                raise ValueError(f"{record_id}: PU positive {tag} must have labels=1")
        if not known[index] and not pu_positive[index] and labels[index] != 0.0:
            raise ValueError(f"{record_id}: unknown {tag} must retain the neutral 0 encoding")
        if modes[tag] == "unsupported" and (known[index] or pu_positive[index]):
            raise ValueError(f"{record_id}: unsupported {tag} cannot participate in loss")
    normalized = dict(record)
    normalized["labels"] = labels
    normalized["known_mask"] = known
    normalized["pu_positive_mask"] = pu_positive
    normalized["sources"] = list(sources)
    return normalized


class Unified57ManifestDataset(Dataset):
    def __init__(self, manifest: Path | str, schema: Mapping[str, object]) -> None:
        self.manifest = Path(manifest)
        self.records: list[dict] = []
        seen_record_ids: set[str] = set()
        with self.manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{self.manifest}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(raw, dict):
                    raise ValueError(f"{self.manifest}:{line_number}: row must be an object")
                split = raw.get("split")
                if split not in (None, "train"):
                    raise ValueError(
                        f"{self.manifest}:{line_number}: training manifest contains split={split!r}"
                    )
                normalized = validate_unified57_record(raw, schema)
                record_id = str(normalized["record_id"])
                if record_id in seen_record_ids:
                    raise ValueError(
                        f"{self.manifest}:{line_number}: duplicate record_id {record_id!r}"
                    )
                seen_record_ids.add(record_id)
                self.records.append(normalized)
        if not self.records:
            raise ValueError("Unified57 training manifest is empty")
        self.source_names = sorted(
            {source for record in self.records for source in record["sources"]}
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, reference: int | SampleReference) -> dict:
        if isinstance(reference, SampleReference):
            record = dict(self.records[reference.record_index])
            record["_stream"] = reference.stream
            record["_balanced_label_index"] = reference.balanced_label_index
            record["_padding"] = reference.padding
            record["_fallback"] = reference.fallback
            return record
        record = dict(self.records[int(reference)])
        record["_stream"] = "uniform"
        record["_balanced_label_index"] = None
        record["_padding"] = False
        record["_fallback"] = False
        return record


def _decode_resized_rgb(source: Image.Image, image_max_pixels: int) -> Image.Image:
    original_size = (source.width, source.height)
    if image_max_pixels <= 0 or source.width * source.height <= image_max_pixels:
        target_size = original_size
    else:
        scale = math.sqrt(image_max_pixels / float(source.width * source.height))
        target_size = (
            max(1, int(source.width * scale)),
            max(1, int(source.height * scale)),
        )
    if target_size != original_size and source.format in {"JPEG", "MPO"}:
        try:
            source.draft("RGB", target_size)
        except (AttributeError, OSError):
            pass
    image = source.convert("RGB")
    if image.size != target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)
    return image


def _open_training_image(
    record: Mapping[str, object], manifest_parent: Path, image_max_pixels: int
) -> Image.Image:
    image_value = record.get("image_path") or record.get("local_image_path")
    if image_value:
        path = Path(str(image_value))
        if not path.is_absolute():
            path = manifest_parent / path
        with Image.open(path) as source:
            return _decode_resized_rgb(source, image_max_pixels)
    image_url = record.get("image_url")
    if not image_url:
        raise ValueError(f"record {record.get('record_id')} has no image path or URL")
    request = urllib.request.Request(
        str(image_url), headers={"User-Agent": "unified57-trainer/1.0"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read()
    with Image.open(io.BytesIO(payload)) as source:
        return _decode_resized_rgb(source, image_max_pixels)


class Unified57BatchCollator:
    def __init__(
        self,
        processor: object,
        manifest_parent: Path,
        image_max_pixels: int,
        source_names: Sequence[str],
    ) -> None:
        self.processor = processor
        self.manifest_parent = manifest_parent
        self.image_max_pixels = image_max_pixels
        self.source_names = list(source_names)

    def __call__(self, records: list[dict]) -> dict:
        images = [
            _open_training_image(record, self.manifest_parent, self.image_max_pixels)
            for record in records
        ]
        batch = self.processor(  # type: ignore[operator]
            images=images,
            text=[VISION_PROMPT] * len(records),
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(
            [record["labels"] for record in records], dtype=torch.float32
        )
        batch["known_mask"] = torch.tensor(
            [record["known_mask"] for record in records], dtype=torch.bool
        )
        batch["pu_positive_mask"] = torch.tensor(
            [record["pu_positive_mask"] for record in records], dtype=torch.bool
        )
        batch["uniform_stream_mask"] = torch.tensor(
            [record["_stream"] == "uniform" for record in records], dtype=torch.bool
        )
        batch["source_membership"] = torch.tensor(
            [
                [source in record["sources"] for source in self.source_names]
                for record in records
            ],
            dtype=torch.int64,
        )
        batch["dictionary_binding_count"] = torch.tensor(
            [record["dictionary_binding_count"] for record in records],
            dtype=torch.int64,
        )
        batch["balanced_label_index"] = torch.tensor(
            [
                -1
                if record["_balanced_label_index"] is None
                else record["_balanced_label_index"]
                for record in records
            ],
            dtype=torch.int64,
        )
        batch["padding_mask"] = torch.tensor(
            [record["_padding"] for record in records], dtype=torch.bool
        )
        batch["fallback_mask"] = torch.tensor(
            [record["_fallback"] for record in records], dtype=torch.bool
        )
        batch["record_ids"] = [record["record_id"] for record in records]
        return dict(batch)


def compute_pn_pos_weight(
    records: Sequence[Mapping[str, object]],
    schema: Mapping[str, object],
    *,
    cap: float = 20.0,
    schedule: TwoStreamSchedule | None = None,
) -> tuple[torch.Tensor, dict]:
    """Count known cells over the deterministic exposure plan only."""
    if cap < 1.0:
        raise ValueError("pos_weight cap must be at least 1.0")
    positive = torch.zeros(57, dtype=torch.float64)
    negative = torch.zeros(57, dtype=torch.float64)
    if schedule is None:
        exposed_records = list(records)
        count_scope = "unique train records, known_mask cells only"
        schedule_sha256 = None
    else:
        if schedule.records is not records and list(schedule.records) != list(records):
            raise ValueError("pos_weight schedule records differ from the training records")
        exposed_records = [
            records[reference.record_index]
            for batch in schedule.global_batches
            for reference in batch
        ]
        count_scope = "deterministic two-stream schedule exposures"
        schedule_sha256 = schedule.schedule_sha256
    for record in exposed_records:
        labels = torch.tensor(record["labels"], dtype=torch.float64)
        known = torch.tensor(record["known_mask"], dtype=torch.bool)
        positive += known & (labels >= 0.5)
        negative += known & (labels < 0.5)
    weights = torch.ones(57, dtype=torch.float64)
    no_positive: list[str] = []
    no_negative: list[str] = []
    raw = torch.ones(57, dtype=torch.float64)
    for index, tag in enumerate(schema["labels"]):
        if schema["label_training_modes"][tag] != "pn":
            continue
        if positive[index] == 0:
            no_positive.append(tag)
        if negative[index] == 0:
            no_negative.append(tag)
        raw[index] = negative[index] / positive[index].clamp_min(1.0)
        weights[index] = raw[index].clamp(1.0, cap)
    audit = {
        "count_scope": count_scope,
        "record_exposures": len(exposed_records),
        "sampler_schedule_sha256": schedule_sha256,
        "cap": float(cap),
        "positive_counts": positive.to(dtype=torch.int64).tolist(),
        "negative_counts": negative.to(dtype=torch.int64).tolist(),
        "raw_weights": raw.tolist(),
        "final_weights": weights.tolist(),
        "no_positive": no_positive,
        "no_negative": no_negative,
        "tag_order": list(schema["labels"]),
        "schema_sha256": schema["schema_sha256"],
    }
    return weights.float(), audit


def masked_pn_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    known_mask: torch.Tensor,
    *,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    """Mean known-cell BCE per sample, then mean over PN-active samples."""
    if logits.shape != labels.shape or logits.shape != known_mask.shape:
        raise ValueError("PN logits, labels, and known_mask must share a shape")
    if pos_weight.ndim != 1 or pos_weight.shape[0] != logits.shape[1]:
        raise ValueError("pos_weight must contain one value per label")
    known = known_mask.to(device=logits.device, dtype=torch.bool)
    scores = logits.float()
    targets = labels.to(device=logits.device, dtype=torch.float32)
    cell_loss = F.binary_cross_entropy_with_logits(
        scores,
        targets,
        pos_weight=pos_weight.to(device=logits.device, dtype=torch.float32),
        reduction="none",
    )
    counts = known.sum(dim=1)
    active = counts > 0
    if not torch.any(active):
        return scores.sum() * 0.0
    sample_loss = (cell_loss * known).sum(dim=1) / counts.clamp_min(1)
    return sample_loss[active].mean()


def initial_sampling_statistics(tag_order: Sequence[str], source_names: Sequence[str]) -> dict:
    return {
        "source_exposures": {source: 0 for source in source_names},
        "stream_exposures": {"uniform": 0, "balanced": 0},
        "uniform_padding_exposures": 0,
        "balanced_fallback_exposures": 0,
        "dictionary_binding_count_exposure": 0,
        "pn_participation_by_label": {tag: 0 for tag in tag_order},
        "pu_positive_participation_by_label": {tag: 0 for tag in tag_order},
        "pu_unlabeled_participation_by_label": {tag: 0 for tag in tag_order},
        "pu_pair_participation_by_label": {tag: 0 for tag in tag_order},
        "balanced_selection_by_label": {tag: 0 for tag in tag_order},
        "optimizer_steps": 0,
        "global_sample_exposures": 0,
        "total_supervised_bits": 0,
    }


def _all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def update_sampling_statistics(
    statistics_payload: dict,
    *,
    batch: Mapping[str, object],
    pu_indices: Sequence[int],
    pu_stats: Mapping[str, Sequence[int]],
    tag_order: Sequence[str],
    source_names: Sequence[str],
) -> None:
    device = batch["known_mask"].device  # type: ignore[union-attr]
    known_counts = _all_reduce_sum(
        batch["known_mask"].sum(dim=0, dtype=torch.int64)  # type: ignore[union-attr]
    )
    pu_positive_counts = _all_reduce_sum(
        batch["pu_positive_mask"].sum(dim=0, dtype=torch.int64)  # type: ignore[union-attr]
    )
    source_counts = _all_reduce_sum(
        batch["source_membership"].sum(dim=0, dtype=torch.int64)  # type: ignore[union-attr]
    )
    uniform_count = _all_reduce_sum(
        batch["uniform_stream_mask"].sum(dtype=torch.int64).reshape(1)  # type: ignore[union-attr]
    )
    local_batch_count = torch.tensor(
        [batch["known_mask"].shape[0]], device=device, dtype=torch.int64  # type: ignore[union-attr]
    )
    global_batch_count = _all_reduce_sum(local_batch_count)
    padding_count = _all_reduce_sum(
        batch["padding_mask"].sum(dtype=torch.int64).reshape(1)  # type: ignore[union-attr]
    )
    fallback_count = _all_reduce_sum(
        batch["fallback_mask"].sum(dtype=torch.int64).reshape(1)  # type: ignore[union-attr]
    )
    binding_exposure = _all_reduce_sum(
        batch["dictionary_binding_count"].sum(dtype=torch.int64).reshape(1)  # type: ignore[union-attr]
    )
    balance_counts = torch.zeros(57, device=device, dtype=torch.int64)
    balanced_indexes = batch["balanced_label_index"]  # type: ignore[assignment]
    valid_balanced = balanced_indexes[balanced_indexes >= 0]
    if valid_balanced.numel():
        balance_counts.scatter_add_(
            0,
            valid_balanced,
            torch.ones_like(valid_balanced, dtype=torch.int64),
        )
    _all_reduce_sum(balance_counts)
    pu_p = _all_reduce_sum(
        torch.tensor(pu_stats["positive_counts"], device=device, dtype=torch.int64)
    )
    pu_u = _all_reduce_sum(
        torch.tensor(pu_stats["unlabeled_counts"], device=device, dtype=torch.int64)
    )
    pu_pairs = _all_reduce_sum(
        torch.tensor(pu_stats["pair_counts"], device=device, dtype=torch.int64)
    )

    for index, tag in enumerate(tag_order):
        statistics_payload["pn_participation_by_label"][tag] += int(known_counts[index])
        statistics_payload["pu_positive_participation_by_label"][tag] += int(
            pu_positive_counts[index]
        )
        statistics_payload["balanced_selection_by_label"][tag] += int(balance_counts[index])
    for position, label_index in enumerate(pu_indices):
        tag = tag_order[label_index]
        statistics_payload["pu_unlabeled_participation_by_label"][tag] += int(
            pu_u[position]
        )
        statistics_payload["pu_pair_participation_by_label"][tag] += int(
            pu_pairs[position]
        )
        # build_pairwise_pu_masks is authoritative; this value intentionally
        # mirrors it instead of treating every unknown cell as a PU reference.
        statistics_payload["pu_positive_participation_by_label"][tag] += (
            int(pu_p[position]) - int(pu_positive_counts[label_index])
        )
    for index, source in enumerate(source_names):
        statistics_payload["source_exposures"][source] += int(source_counts[index])
    uniform_value = int(uniform_count[0])
    global_value = int(global_batch_count[0])
    statistics_payload["stream_exposures"]["uniform"] += uniform_value
    statistics_payload["stream_exposures"]["balanced"] += global_value - uniform_value
    statistics_payload["uniform_padding_exposures"] += int(padding_count[0])
    statistics_payload["balanced_fallback_exposures"] += int(fallback_count[0])
    statistics_payload["dictionary_binding_count_exposure"] += int(binding_exposure[0])
    statistics_payload["global_sample_exposures"] += global_value
    statistics_payload["total_supervised_bits"] += int(known_counts.sum()) + int(pu_p.sum())


def dataset_contract_audit(
    records: Sequence[Mapping[str, object]], schema: Mapping[str, object]
) -> dict:
    known = [0] * 57
    pu_positive = [0] * 57
    sources = Counter()
    for record in records:
        for index, value in enumerate(record["known_mask"]):
            known[index] += int(value)
        for index, value in enumerate(record["pu_positive_mask"]):
            pu_positive[index] += int(value)
        sources.update(record["sources"])
    return {
        "records": len(records),
        "source_records": dict(sorted(sources.items())),
        "known_supervised_bits": sum(known),
        "pu_positive_bits": sum(pu_positive),
        "total_supervised_bits": sum(known) + sum(pu_positive),
        "known_by_label": dict(zip(schema["labels"], known)),
        "pu_positive_by_label": dict(zip(schema["labels"], pu_positive)),
        "schema_sha256": schema["schema_sha256"],
    }


def _atomic_json(payload: Mapping[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _gather_rank_rng_states(world_size: int) -> list[dict]:
    local = _capture_local_rng_state()
    if not dist.is_initialized():
        return [local]
    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    if any(state is None for state in gathered):
        raise RuntimeError("failed to gather RNG state from every distributed rank")
    return [state for state in gathered if state is not None]


def save_v3_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    rank: int,
    world_size: int,
    tag_order: Sequence[str],
    schema_sha256: str,
    manifest_sha256: str,
    sampler_config: Mapping[str, object],
    sampler_schedule_sha256: str,
    cursor: Mapping[str, int],
    sampling_statistics: Mapping[str, object],
    elapsed_seconds: float,
    pos_weight_audit: Mapping[str, object],
    initialization_audit: Mapping[str, object],
    loss_contract: Mapping[str, object],
    run_contract: Mapping[str, object],
) -> None:
    rng_states = _gather_rank_rng_states(world_size)
    if rank == 0:
        payload = build_v3_checkpoint(
            model=model,
            optimizer=optimizer,
            tag_order=tag_order,
            schema_sha256=schema_sha256,
            manifest_sha256=manifest_sha256,
            sampler_config=sampler_config,
            sampler_schedule_sha256=sampler_schedule_sha256,
            cursor=cursor,
            sampling_statistics=sampling_statistics,
            world_size=world_size,
            rank_rng_states=rng_states,
            elapsed_seconds=elapsed_seconds,
            pos_weight_audit=pos_weight_audit,
            initialization_audit=initialization_audit,
            loss_contract=loss_contract,
            run_contract=run_contract,
        )
        checkpoint_dir = output_dir / "checkpoints"
        latest = checkpoint_dir / "latest.pt"
        numbered = checkpoint_dir / f"step_{int(cursor['global_step']):08d}.pt"
        _atomic_torch_save(payload, latest)
        _atomic_torch_save(payload, numbered)
        versions = sorted(checkpoint_dir.glob("step_*.pt"))
        for obsolete in versions[:-3]:
            obsolete.unlink(missing_ok=True)
    if dist.is_initialized():
        dist.barrier()


def _setup_distributed(expected_world_size: int) -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("Unified57 training requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    if expected_world_size > 0 and world_size != expected_world_size:
        raise RuntimeError(
            f"expected {expected_world_size} workers, got {world_size}; "
            "use --expected-world-size 0 only for an intentional debug run"
        )
    return local_rank, rank, world_size, torch.device("cuda", local_rank)


def _load_qwen3vl_classifier(
    model_path: str,
    *,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    head_dropout: float,
) -> nn.Module:
    try:
        from scripts.jd_multilabel_training_core import load_qwen3vl_classifier
    except ModuleNotFoundError:
        try:
            from jd_multilabel_training_core import load_qwen3vl_classifier  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "scripts/jd_multilabel_training_core.py is required on the training host"
            ) from exc
    return load_qwen3vl_classifier(
        model_path,
        num_labels=57,
        use_lora=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        head_dropout=head_dropout,
        gradient_checkpointing=True,
    )


def _seed_model_initialization(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seed_fresh_training(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def _move_tensor_batch(batch: Mapping[str, object], device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _model_inputs(batch: Mapping[str, object]) -> dict:
    excluded = {
        "labels",
        "known_mask",
        "pu_positive_mask",
        "uniform_stream_mask",
        "source_membership",
        "dictionary_binding_count",
        "balanced_label_index",
        "padding_mask",
        "fallback_mask",
        "record_ids",
    }
    return {key: value for key, value in batch.items() if key not in excluded}


def _finite_gradient_audit(model: nn.Module, *, require_nonzero_lora: bool) -> dict:
    total = 0
    with_gradient = 0
    finite = 0
    lora_total = 0
    lora_with_gradient = 0
    lora_nonzero = 0
    for name, parameter in _unwrap_model(model).named_parameters():
        if not parameter.requires_grad:
            continue
        total += 1
        is_lora = "lora_" in name
        lora_total += int(is_lora)
        if parameter.grad is None:
            continue
        with_gradient += 1
        if is_lora:
            lora_with_gradient += 1
        if torch.isfinite(parameter.grad).all():
            finite += 1
        else:
            raise FloatingPointError(f"non-finite gradient in {name}")
        if is_lora and torch.count_nonzero(parameter.grad).item() > 0:
            lora_nonzero += 1
    if with_gradient != total:
        raise RuntimeError(f"only {with_gradient}/{total} trainable tensors received gradients")
    if lora_total != EXPECTED_LORA_TENSORS:
        raise RuntimeError(
            f"expected {EXPECTED_LORA_TENSORS} trainable LoRA tensors, found {lora_total}"
        )
    if lora_with_gradient != lora_total:
        raise RuntimeError(
            f"only {lora_with_gradient}/{lora_total} LoRA tensors received gradients"
        )
    if require_nonzero_lora and lora_nonzero == 0:
        raise RuntimeError("all LoRA gradients are zero on the first optimizer step")
    return {
        "trainable_tensors": total,
        "with_gradient": with_gradient,
        "finite": finite,
        "lora_tensors": lora_total,
        "lora_nonzero": lora_nonzero,
    }


def _distributed_stop_flag(device: torch.device, local_stop: bool) -> bool:
    value = torch.tensor([int(local_stop)], device=device, dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value.item())


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="8-GPU Unified57 Qwen3-VL LoRA training with PN/PU two-stream loss"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    state_group = parser.add_mutually_exclusive_group(required=True)
    state_group.add_argument("--init-from-aggregate18", type=Path)
    state_group.add_argument("--resume", type=Path)
    state_group.add_argument("--fresh", action="store_true")
    parser.add_argument("--aggregate18-config", type=Path)
    parser.add_argument(
        "--aggregate18-checkpoint-sha256",
        default=AGGREGATE18_CHECKPOINT_SHA256,
    )
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=DEFAULT_MICRO_BATCH_SIZE)
    parser.add_argument("--uniform-per-rank", type=int, default=DEFAULT_UNIFORM_PER_RANK)
    parser.add_argument("--balanced-per-rank", type=int, default=DEFAULT_BALANCED_PER_RANK)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--image-max-pixels", type=int, default=DEFAULT_IMAGE_MAX_PIXELS)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--wall-clock-hours", type=float, default=5.0)
    parser.add_argument("--reserve-minutes", type=float, default=10.0)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--pu-margin", type=float, default=DEFAULT_PU_MARGIN)
    parser.add_argument("--pu-loss-weight", type=float, default=DEFAULT_PU_LOSS_WEIGHT)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    args = parser.parse_args(argv)
    if args.micro_batch_size != args.uniform_per_rank + args.balanced_per_rank:
        parser.error("--micro-batch-size must equal uniform_per_rank + balanced_per_rank")
    if args.gradient_accumulation != 1:
        parser.error("Unified57 v1 requires --gradient-accumulation 1")
    if args.max_steps < 0:
        parser.error("--max-steps must be non-negative")
    if args.init_from_aggregate18 and args.aggregate18_config is None:
        parser.error("--aggregate18-config is required with --init-from-aggregate18")
    if args.aggregate18_config is not None and not args.init_from_aggregate18:
        parser.error("--aggregate18-config is valid only for Aggregate18 initialization")
    if args.aggregate18_checkpoint_sha256 != AGGREGATE18_CHECKPOINT_SHA256:
        parser.error("only the audited Aggregate18 latest.pt SHA256 is allowed")
    return args


parse_args = _parse_args


def _loss_contract(args: argparse.Namespace, pu_tags: Sequence[str]) -> dict:
    return {
        "version": LOSS_CONTRACT_VERSION,
        "pn": "per-sample masked BCE over known_mask cells",
        "pn_pos_weight_scope": "deterministic two-stream schedule known-cell exposures",
        "pu": "FP32 positive-vs-unlabeled pairwise hinge ranking",
        "pu_tags": list(pu_tags),
        "pu_margin": float(args.pu_margin),
        "pu_loss_weight": float(args.pu_loss_weight),
        "pu_unlabeled_stream": "uniform_only",
        "pu_pair_scope": "rank_local_microbatch",
        "pn_reduction_scope": "rank_local_active_sample_mean_then_ddp_mean",
        "unsupported_tags": ["假两件"],
        "unknown_is_negative": False,
        "pu_output_semantics": PU_OUTPUT_SEMANTICS,
    }


def _run_contract(args: argparse.Namespace) -> dict:
    model_path = Path(args.model)
    config_path = model_path / "config.json"
    return {
        "base_model": str(args.model),
        "base_model_config_sha256": (
            _sha256_file(config_path) if config_path.is_file() else None
        ),
        "image_max_pixels": int(args.image_max_pixels),
        "micro_batch_size_per_gpu": int(args.micro_batch_size),
        "gradient_accumulation": int(args.gradient_accumulation),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "head_dropout": float(args.head_dropout),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "dtype": "bfloat16",
        "vision_prompt_sha256": hashlib.sha256(VISION_PROMPT.encode("utf-8")).hexdigest(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    local_rank, rank, world_size, device = _setup_distributed(args.expected_world_size)
    del local_rank
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schema = load_unified57_schema(args.schema)
    dataset = Unified57ManifestDataset(args.manifest, schema)
    manifest_sha256 = _sha256_file(args.manifest)
    pu_indices = [
        index
        for index, tag in enumerate(schema["labels"])
        if schema["label_training_modes"][tag] == "pu"
    ]
    if len(pu_indices) != 20:
        raise RuntimeError("schema must expose exactly 20 pure-PU labels")
    pu_tags = [schema["labels"][index] for index in pu_indices]
    schedule = TwoStreamSchedule(
        dataset.records,
        tag_order=schema["labels"],
        training_modes=schema["label_training_modes"],
        world_size=world_size,
        uniform_per_rank=args.uniform_per_rank,
        balanced_per_rank=args.balanced_per_rank,
        seed=args.seed,
    )
    pos_weight, pos_weight_audit = compute_pn_pos_weight(
        dataset.records, schema, cap=args.max_pos_weight, schedule=schedule
    )
    data_audit = dataset_contract_audit(dataset.records, schema)
    loss_contract = _loss_contract(args, pu_tags)
    run_contract = _run_contract(args)
    _seed_model_initialization(args.seed)
    model = _load_qwen3vl_classifier(
        args.model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        head_dropout=args.head_dropout,
    )

    initialization_audit: dict[str, object]
    if args.init_from_aggregate18:
        config_audit = validate_aggregate18_config(args.aggregate18_config)
        initialization_audit = transfer_aggregate18_v2_checkpoint(
            model,
            args.init_from_aggregate18,
            target_tag_order=schema["labels"],
            expected_checkpoint_sha256=args.aggregate18_checkpoint_sha256,
        )
        initialization_audit["source_config_sha256"] = config_audit["sha256"]
        initialization_audit["mode"] = "aggregate18_v2_weight_transfer"
    elif args.resume:
        initialization_audit = {"mode": "unified57_v3_resume"}
    else:
        initialization_audit = {
            "mode": "fresh_unified57",
            "optimizer_loaded": False,
            "fresh_training_state": {
                "global_step": 0,
                "global_batch_cursor": 0,
                "optimizer": "fresh",
                "rng": "fresh",
            },
        }

    model.to(device)
    trainable_named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable_named],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    statistics_payload = initial_sampling_statistics(schema["labels"], dataset.source_names)
    cursor = {"global_step": 0, "global_batch_cursor": 0}
    prior_elapsed = 0.0
    if args.resume:
        resumed = restore_v3_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            expected_tag_order=schema["labels"],
            expected_schema_sha256=schema["schema_sha256"],
            expected_manifest_sha256=manifest_sha256,
            expected_sampler_config=schedule.config,
            expected_sampler_schedule_sha256=schedule.schedule_sha256,
            expected_world_size=world_size,
            rank=rank,
            expected_loss_contract=loss_contract,
            expected_pos_weight_audit=pos_weight_audit,
            expected_run_contract=run_contract,
        )
        cursor = dict(resumed["cursor"])
        statistics_payload = copy.deepcopy(resumed["sampling_statistics"])
        initialization_audit = copy.deepcopy(resumed.get("initialization_audit") or {})
        prior_elapsed = float(resumed.get("elapsed_seconds", 0.0))
    else:
        _seed_fresh_training(args.seed, rank)

    if cursor["global_batch_cursor"] > len(schedule.global_batches):
        raise ValueError("resume global_batch_cursor exceeds the deterministic schedule")
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
        )
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    collator = Unified57BatchCollator(
        processor,
        args.manifest.parent,
        args.image_max_pixels,
        dataset.source_names,
    )
    loader_options: dict[str, object] = {
        "dataset": dataset,
        "batch_sampler": schedule.rank_batches(
            rank, start_global_batch=cursor["global_batch_cursor"]
        ),
        "collate_fn": collator,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_options["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(**loader_options)
    pos_weight = pos_weight.to(device)
    started_at = time.monotonic()
    deadline_seconds = args.wall_clock_hours * 3600.0 - args.reserve_minutes * 60.0
    if deadline_seconds <= 0:
        raise ValueError("wall-clock budget must exceed the reserved shutdown window")
    stop_signal = {"requested": False}

    def request_stop(_signum, _frame):
        stop_signal["requested"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    step_times: list[float] = []
    last_loss = {"total": 0.0, "pn": 0.0, "pu": 0.0}
    gradient_audit: dict[str, object] = {}
    pending_microbatches = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    if rank == 0:
        _atomic_json(data_audit, args.output_dir / "dataset_contract_audit.json")
        _atomic_json(pos_weight_audit, args.output_dir / "pos_weight_audit.json")
        _atomic_json(initialization_audit, args.output_dir / "transfer_audit.json")
        _atomic_json(
            {
                "schema_sha256": schema["schema_sha256"],
                "manifest_sha256": manifest_sha256,
                "sampler_config": schedule.config,
                "sampler_schedule_sha256": schedule.schedule_sha256,
                "loss_contract": loss_contract,
                "run_contract": run_contract,
                "defaults": {
                    "image_max_pixels": args.image_max_pixels,
                    "micro_batch_size_per_gpu": args.micro_batch_size,
                    "gradient_accumulation": args.gradient_accumulation,
                    "global_effective_batch": (
                        world_size * args.micro_batch_size * args.gradient_accumulation
                    ),
                },
            },
            args.output_dir / "run_contract.json",
        )

    stop_reason: str | None = None
    target_reached = bool(args.max_steps and cursor["global_step"] >= args.max_steps)
    if target_reached:
        stop_reason = "max_steps_already_reached"
    loader_iterator = iter(loader)
    last_saved_cursor: tuple[int, int] | None = None
    final_checkpoint_seconds = 0.0
    while not target_reached:
        step_started = time.monotonic()  # Includes DataLoader/processor wait.
        elapsed = prior_elapsed + (time.monotonic() - started_at)
        local_deadline = elapsed >= deadline_seconds
        if _distributed_stop_flag(device, stop_signal["requested"] or local_deadline):
            stop_reason = "signal_or_deadline"
            break
        try:
            raw_batch = next(loader_iterator)
        except StopIteration:
            stop_reason = "schedule_exhausted"
            break
        elapsed = prior_elapsed + (time.monotonic() - started_at)
        local_deadline = elapsed >= deadline_seconds
        if _distributed_stop_flag(device, stop_signal["requested"] or local_deadline):
            stop_reason = "signal_or_deadline"
            break
        global_batch_index = cursor["global_batch_cursor"]
        batch = _move_tensor_batch(raw_batch, device)
        accumulation_boundary = (pending_microbatches + 1) >= args.gradient_accumulation
        sync_context = contextlib.nullcontext()
        if (
            isinstance(model, DistributedDataParallel)
            and not accumulation_boundary
        ):
            sync_context = model.no_sync()
        with sync_context:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**_model_inputs(batch))["logits"]
            pn_loss = masked_pn_bce_loss(
                logits,
                batch["labels"],
                batch["known_mask"],
                pos_weight=pos_weight,
            )
            positive_mask, unlabeled_mask = build_pairwise_pu_masks(
                known_mask=batch["known_mask"],
                pu_positive_mask=batch["pu_positive_mask"],
                uniform_stream_mask=batch["uniform_stream_mask"],
                pu_indices=pu_indices,
            )
            pu_loss, pu_stats = pairwise_positive_unlabeled_ranking_loss(
                logits.index_select(
                    1, torch.tensor(pu_indices, device=device, dtype=torch.long)
                ),
                positive_mask,
                unlabeled_mask,
                margin=args.pu_margin,
            )
            loss = pn_loss + float(args.pu_loss_weight) * pu_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at global batch {global_batch_index}: {loss.item()}"
                )
            (loss / args.gradient_accumulation).backward()
        pending_microbatches += 1
        cursor["global_batch_cursor"] = global_batch_index + 1
        update_sampling_statistics(
            statistics_payload,
            batch=batch,
            pu_indices=pu_indices,
            pu_stats=pu_stats,
            tag_order=schema["labels"],
            source_names=dataset.source_names,
        )
        last_loss = {
            "total": float(loss.detach().item()),
            "pn": float(pn_loss.detach().item()),
            "pu": float(pu_loss.detach().item()),
        }
        if not accumulation_boundary:
            continue
        gradient_audit = _finite_gradient_audit(
            model, require_nonzero_lora=cursor["global_step"] == 0
        )
        torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in trainable_named], max_norm=1.0
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        pending_microbatches = 0
        cursor["global_step"] += 1
        statistics_payload["optimizer_steps"] = cursor["global_step"]
        elapsed = prior_elapsed + (time.monotonic() - started_at)
        if cursor["global_step"] % args.save_every == 0:
            save_v3_checkpoint(
                model=model,
                optimizer=optimizer,
                output_dir=args.output_dir,
                rank=rank,
                world_size=world_size,
                tag_order=schema["labels"],
                schema_sha256=schema["schema_sha256"],
                manifest_sha256=manifest_sha256,
                sampler_config=schedule.config,
                sampler_schedule_sha256=schedule.schedule_sha256,
                cursor=cursor,
                sampling_statistics=statistics_payload,
                elapsed_seconds=elapsed,
                pos_weight_audit=pos_weight_audit,
                initialization_audit=initialization_audit,
                loss_contract=loss_contract,
                run_contract=run_contract,
            )
            last_saved_cursor = (
                cursor["global_step"],
                cursor["global_batch_cursor"],
            )
        step_seconds = time.monotonic() - step_started
        progress = {
            "format_version": FORMAT_VERSION,
            "global_step": cursor["global_step"],
            "global_batch_cursor": cursor["global_batch_cursor"],
            "global_batch_count": len(schedule.global_batches),
            "loss": last_loss,
            "loss_finite": True,
            "gradients": gradient_audit,
            "step_seconds": step_seconds,
            "elapsed_seconds": prior_elapsed + (time.monotonic() - started_at),
            "schema_sha256": schema["schema_sha256"],
            "sampler_schedule_sha256": schedule.schedule_sha256,
        }
        if rank == 0:
            _atomic_json(progress, args.output_dir / "progress.json")
            if cursor["global_step"] % args.log_every == 0:
                print(json.dumps(progress, ensure_ascii=False), flush=True)
        # Gate timing includes decode/processor wait, optimizer, periodic
        # checkpoint, progress persistence, and log output.
        step_times.append(time.monotonic() - step_started)
        if args.max_steps and cursor["global_step"] >= args.max_steps:
            target_reached = True
            stop_reason = "max_steps_reached"
        elif cursor["global_batch_cursor"] >= len(schedule.global_batches):
            target_reached = True
            stop_reason = "schedule_exhausted"

    elapsed = prior_elapsed + (time.monotonic() - started_at)
    current_cursor = (cursor["global_step"], cursor["global_batch_cursor"])
    if current_cursor != last_saved_cursor:
        checkpoint_started = time.monotonic()
        save_v3_checkpoint(
            model=model,
            optimizer=optimizer,
            output_dir=args.output_dir,
            rank=rank,
            world_size=world_size,
            tag_order=schema["labels"],
            schema_sha256=schema["schema_sha256"],
            manifest_sha256=manifest_sha256,
            sampler_config=schedule.config,
            sampler_schedule_sha256=schedule.schedule_sha256,
            cursor=cursor,
            sampling_statistics=statistics_payload,
            elapsed_seconds=elapsed,
            pos_weight_audit=pos_weight_audit,
            initialization_audit=initialization_audit,
            loss_contract=loss_contract,
            run_contract=run_contract,
        )
        final_checkpoint_seconds = time.monotonic() - checkpoint_started
    elapsed = prior_elapsed + (time.monotonic() - started_at)
    stable_times = step_times[2:] if len(step_times) > 2 else step_times
    mean_step = statistics.fmean(stable_times) if stable_times else 0.0
    global_batch_size = world_size * args.micro_batch_size
    throughput = global_batch_size / mean_step if mean_step else 0.0
    projected_seconds = (
        mean_step * len(schedule.global_batches) + final_checkpoint_seconds
        if mean_step
        else 0.0
    )
    smoke_target_complete = bool(
        args.max_steps and cursor["global_step"] >= args.max_steps
    )
    smoke_measurement_complete = bool(
        smoke_target_complete and len(step_times) >= 20
    )
    formal_target_complete = bool(
        not args.max_steps
        and cursor["global_batch_cursor"] >= len(schedule.global_batches)
    )
    run_complete = smoke_target_complete if args.max_steps else formal_target_complete
    smoke_report = {
        "status": "complete" if run_complete else "partial",
        "stop_reason": stop_reason or "unknown",
        "target_reached": run_complete,
        "smoke_measurement_complete": smoke_measurement_complete,
        "global_step": cursor["global_step"],
        "global_batch_cursor": cursor["global_batch_cursor"],
        "elapsed_seconds": elapsed,
        "steps_measured": len(step_times),
        "warmup_steps_excluded": min(2, len(step_times)),
        "mean_step_seconds": mean_step,
        "p95_step_seconds": _percentile(stable_times, 0.95),
        "global_samples_per_second": throughput,
        "projected_full_schedule_seconds": projected_seconds,
        "final_checkpoint_seconds": final_checkpoint_seconds,
        "formal_projection_within_4h50": bool(
            smoke_measurement_complete
            and projected_seconds
            and projected_seconds <= 17400
        ),
        "last_loss": last_loss,
        "loss_finite": all(math.isfinite(value) for value in last_loss.values()),
        "gradient_audit": gradient_audit,
        "sampling_statistics": statistics_payload,
        "schema_sha256": schema["schema_sha256"],
        "manifest_sha256": manifest_sha256,
        "sampler_schedule_sha256": schedule.schedule_sha256,
        "next_action": (
            "set formal max_steps from the measured full-schedule projection"
            if smoke_measurement_complete
            else "collect a complete 20-step smoke before choosing formal parameters"
        ),
    }
    if rank == 0:
        _atomic_json(smoke_report, args.output_dir / "smoke_report.json")
        print(json.dumps(smoke_report, ensure_ascii=False), flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    # A graceful partial run has already persisted checkpoint/report state.
    # Return success from every rank so torchrun does not wrap it as a child
    # failure; the launcher maps report.status=partial to its status sidecar.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
