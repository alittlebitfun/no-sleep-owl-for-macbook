from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # The local lightweight test command has no torch.
    torch = None
    nn = None

if torch is not None:
    from scripts.train_unified57_qwen3vl_multilabel import (
        AGGREGATE18_TAG_ORDER,
        TwoStreamSchedule,
        build_pairwise_pu_masks,
        build_v3_checkpoint,
        compute_pn_pos_weight,
        load_unified57_schema,
        pairwise_positive_unlabeled_ranking_loss,
        restore_v3_checkpoint,
        transfer_aggregate18_v2_checkpoint,
    )


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "configs" / "bosideng_unified57_schema.json"
TRAINER_PATH = ROOT / "scripts" / "train_unified57_qwen3vl_multilabel.py"
LAUNCHER_PATH = ROOT / "scripts" / "launch_unified57_node1.sh"
requires_torch = pytest.mark.skipif(torch is None, reason="PyTorch is installed on node1")


if torch is not None:
    class _TinyTransferModel(nn.Module):
        def __init__(self, *, lora_shape: tuple[int, ...] = (2, 2)) -> None:
            super().__init__()
            self.lora_adapter = nn.Parameter(torch.zeros(lora_shape))
            self.classifier = nn.Linear(3, 57)


    class _FullLoraContractModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_tensors = nn.ParameterDict(
                {
                    f"lora_{index:03d}": nn.Parameter(torch.zeros(1))
                    for index in range(288)
                }
            )
            self.classifier = nn.Linear(1, 57)


def _source_state(model, *, lora_value: float = 7.0) -> dict:
    target = dict(model.named_parameters())
    lora = {
        name: torch.full_like(parameter, lora_value)
        for name, parameter in target.items()
        if "lora_" in name
    }
    weight = torch.arange(18 * model.classifier.in_features, dtype=torch.float32).reshape(
        18, model.classifier.in_features
    )
    bias = torch.arange(18, dtype=torch.float32) + 100.0
    payload = {**lora, "classifier.weight": weight, "classifier.bias": bias}
    return {
        "format_version": 2,
        "model": payload,
        "trainable_names": sorted(payload),
        "optimizer": {"source_optimizer_must_not_load": True},
        "rng_state": torch.arange(4, dtype=torch.uint8),
        "global_step": 751,
    }


def _optimizer_snapshot(optimizer) -> dict:
    return copy.deepcopy(optimizer.state_dict())


def _records_for_sampler(schema: dict) -> list[dict]:
    labels = schema["labels"]
    pu_index = labels.index("前门襟")
    pn_index = labels.index("连帽")

    def row(record_id: str, dictionary_binding_count: int, binding_count: int) -> dict:
        target = [0.0] * 57
        known = [0] * 57
        pu = [0] * 57
        target[pu_index] = 1.0
        pu[pu_index] = 1
        target[pn_index] = 1.0
        known[pn_index] = 1
        return {
            "record_id": record_id,
            "labels": target,
            "known_mask": known,
            "pu_positive_mask": pu,
            "sources": ["dictionary_v4"],
            "binding_count": binding_count,
            "dictionary_binding_count": dictionary_binding_count,
            "jd_binding_count": 0,
            "schema_sha256": schema["schema_sha256"],
        }

    records = [row("heavy", 5, 1), row("light", 1, 999)]
    for index in range(62):
        records.append(
            {
                **row(f"uniform-{index}", 0, 1000 + index),
                "sources": ["jd_complete23"],
                "pu_positive_mask": [0] * 57,
                "labels": [0.0] * 57,
                "known_mask": [0] * 57,
            }
        )
    return records


def test_static_contract_fixes_336_square_and_safe_smoke_defaults() -> None:
    trainer = TRAINER_PATH.read_text(encoding="utf-8")
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "DEFAULT_IMAGE_MAX_PIXELS = 336 * 336" in trainer
    assert "DEFAULT_MICRO_BATCH_SIZE = 8" in trainer
    assert "DEFAULT_UNIFORM_PER_RANK = 6" in trainer
    assert "DEFAULT_BALANCED_PER_RANK = 2" in trainer
    assert 'MAX_STEPS="${MAX_STEPS:-20}"' in launcher
    assert 'IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-112896}"' in launcher
    assert "--nproc-per-node=8" in launcher
    assert trainer.index("step_started = time.monotonic()") < trainer.index(
        "raw_batch = next(loader_iterator)"
    )
    assert "while not target_reached:" in trainer
    assert '"status": "complete" if run_complete else "partial"' in trainer
    assert "return 0 if run_complete else 3" in trainer
    assert "smoke_report.previous.json" in launcher
    assert "training_exit=3" in launcher
    assert 'write_status "partial" 3' in launcher


def test_schema_digest_and_training_mode_counts_are_self_consistent() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    stored = schema.pop("schema_sha256")
    canonical = json.dumps(
        schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    modes = list(schema["label_training_modes"].values())

    assert hashlib.sha256(canonical).hexdigest() == stored
    assert modes.count("pn") == 36
    assert modes.count("pu") == 20
    assert modes.count("unsupported") == 1
    assert schema["unsupported_labels"] == ["假两件"]


@requires_torch
def test_transfer_maps_classifier_rows_by_explicit_tag_name_and_keeps_new_rows() -> None:
    schema = load_unified57_schema(SCHEMA_PATH)
    model = _TinyTransferModel()
    before_weight = model.classifier.weight.detach().clone()
    before_bias = model.classifier.bias.detach().clone()
    checkpoint = _source_state(model)

    audit = transfer_aggregate18_v2_checkpoint(
        model,
        checkpoint,
        target_tag_order=schema["labels"],
        source_tag_order=AGGREGATE18_TAG_ORDER,
        expected_lora_tensors=1,
    )

    source_weight = checkpoint["model"]["classifier.weight"]
    source_bias = checkpoint["model"]["classifier.bias"]
    mapped = {entry["target_index"] for entry in audit["classifier_row_mapping"]}
    assert len(mapped) == 18
    for source_index, tag in enumerate(AGGREGATE18_TAG_ORDER):
        target_index = schema["labels"].index(tag)
        torch.testing.assert_close(model.classifier.weight[target_index], source_weight[source_index])
        torch.testing.assert_close(model.classifier.bias[target_index], source_bias[source_index])
    for target_index in set(range(57)) - mapped:
        torch.testing.assert_close(model.classifier.weight[target_index], before_weight[target_index])
        torch.testing.assert_close(model.classifier.bias[target_index], before_bias[target_index])
    assert audit["copied_lora_tensors"] == 1
    assert audit["optimizer_loaded"] is False


@requires_torch
def test_transfer_requires_all_288_lora_keys_with_exact_shapes() -> None:
    schema = load_unified57_schema(SCHEMA_PATH)
    model = _FullLoraContractModel()
    checkpoint = _source_state(model)
    missing_key = sorted(name for name in checkpoint["model"] if "lora_" in name)[0]
    del checkpoint["model"][missing_key]
    checkpoint["trainable_names"] = sorted(checkpoint["model"])

    with pytest.raises(ValueError, match="288 LoRA"):
        transfer_aggregate18_v2_checkpoint(
            model,
            checkpoint,
            target_tag_order=schema["labels"],
        )

    checkpoint = _source_state(model)
    bad_key = sorted(name for name in checkpoint["model"] if "lora_" in name)[0]
    checkpoint["model"][bad_key] = torch.zeros(2)
    with pytest.raises(ValueError, match="shape mismatch"):
        transfer_aggregate18_v2_checkpoint(
            model,
            checkpoint,
            target_tag_order=schema["labels"],
        )


@requires_torch
def test_aggregate18_init_never_loads_optimizer_rng_or_cursor() -> None:
    schema = load_unified57_schema(SCHEMA_PATH)
    model = _TinyTransferModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer_before = _optimizer_snapshot(optimizer)
    rng_before = torch.get_rng_state().clone()

    audit = transfer_aggregate18_v2_checkpoint(
        model,
        _source_state(model),
        target_tag_order=schema["labels"],
        expected_lora_tensors=1,
    )

    assert optimizer.state_dict() == optimizer_before
    torch.testing.assert_close(torch.get_rng_state(), rng_before)
    assert audit["fresh_training_state"] == {
        "global_step": 0,
        "global_batch_cursor": 0,
        "optimizer": "fresh",
        "rng": "fresh",
    }


@requires_torch
def test_schema_hash_mismatch_is_rejected_before_resume_mutates_model(tmp_path: Path) -> None:
    schema = load_unified57_schema(SCHEMA_PATH)
    model = _TinyTransferModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint = build_v3_checkpoint(
        model=model,
        optimizer=optimizer,
        tag_order=schema["labels"],
        schema_sha256="0" * 64,
        manifest_sha256="1" * 64,
        sampler_config={"world_size": 1},
        sampler_schedule_sha256="2" * 64,
        cursor={"global_step": 3, "global_batch_cursor": 4},
        sampling_statistics={"source_exposures": {"dictionary_v4": 2}},
        world_size=1,
        rank_rng_states=[{"cpu": torch.get_rng_state(), "cuda": []}],
    )
    checkpoint_path = tmp_path / "bad-schema.pt"
    torch.save(checkpoint, checkpoint_path)
    before = model.classifier.weight.detach().clone()

    with pytest.raises(ValueError, match="schema_sha256 mismatch"):
        restore_v3_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            expected_tag_order=schema["labels"],
            expected_schema_sha256=schema["schema_sha256"],
            expected_manifest_sha256="1" * 64,
            expected_sampler_config={"world_size": 1},
            expected_sampler_schedule_sha256="2" * 64,
            expected_world_size=1,
            rank=0,
        )
    torch.testing.assert_close(model.classifier.weight, before)


@requires_torch
def test_pairwise_loss_uses_unlabeled_only_as_a_relative_reference() -> None:
    logits = torch.tensor([[2.0], [0.5], [8.0]], dtype=torch.bfloat16, requires_grad=True)
    positive = torch.tensor([[1], [0], [0]], dtype=torch.bool)
    known = torch.zeros_like(positive)
    uniform = torch.tensor([False, True, False])
    positive_mask, unlabeled_mask = build_pairwise_pu_masks(
        known_mask=known,
        pu_positive_mask=positive,
        uniform_stream_mask=uniform,
        pu_indices=[0],
    )

    loss, stats = pairwise_positive_unlabeled_ranking_loss(
        logits,
        positive_mask,
        unlabeled_mask,
        margin=1.0,
    )
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(0.0)
    assert stats["positive_counts"] == [1]
    assert stats["unlabeled_counts"] == [1]
    assert stats["pair_counts"] == [1]
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[2].item() == 0.0  # balanced-stream unknown was not a target

    no_positive_loss, _ = pairwise_positive_unlabeled_ranking_loss(
        logits.detach().float(),
        torch.zeros_like(positive_mask),
        unlabeled_mask,
        margin=1.0,
    )
    assert no_positive_loss.item() == 0.0


@requires_torch
def test_only_uniform_slots_can_be_pu_unlabeled_while_both_streams_can_be_positive() -> None:
    known = torch.zeros((4, 2), dtype=torch.bool)
    pu_positive = torch.tensor([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=torch.bool)
    uniform = torch.tensor([True, False, True, False])

    positives, unlabeled = build_pairwise_pu_masks(
        known_mask=known,
        pu_positive_mask=pu_positive,
        uniform_stream_mask=uniform,
        pu_indices=[0, 1],
    )

    assert positives.tolist() == [[True, False], [False, True], [False, False], [False, False]]
    assert unlabeled.tolist() == [[False, True], [False, False], [True, True], [False, False]]


@requires_torch
def test_two_stream_schedule_uses_dictionary_binding_count_as_record_weight() -> None:
    schema = load_unified57_schema(SCHEMA_PATH)
    records = _records_for_sampler(schema)

    schedule = TwoStreamSchedule(
        records,
        tag_order=schema["labels"],
        training_modes=schema["label_training_modes"],
        world_size=8,
        uniform_per_rank=6,
        balanced_per_rank=2,
        seed=17,
    )

    pu_index = schema["labels"].index("前门襟")
    assert schedule.balanced_candidate_weights[pu_index] == {0: 5.0, 1: 1.0}
    assert schedule.config["micro_batch_size_per_gpu"] == 8
    assert schedule.config["global_batch_size"] == 64
    assert schedule.config["weight_field"] == "dictionary_binding_count"
    uniform_indexes = [
        reference.record_index
        for batch in schedule.global_batches
        for reference in batch
        if reference.stream == "uniform" and not reference.padding
    ]
    assert sorted(uniform_indexes) == list(range(len(records)))


@requires_torch
def test_v3_checkpoint_contains_contract_metadata_and_restores_optimizer_and_cursor(
    tmp_path: Path,
) -> None:
    schema = load_unified57_schema(SCHEMA_PATH)
    model = _TinyTransferModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model.classifier(torch.ones(2, 3)).sum() + model.lora_adapter.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    saved_weight = model.classifier.weight.detach().clone()
    saved_optimizer = _optimizer_snapshot(optimizer)
    sampler_config = {
        "world_size": 1,
        "uniform_per_rank": 6,
        "balanced_per_rank": 2,
        "weight_field": "dictionary_binding_count",
    }
    stats = {
        "source_exposures": {"jd_complete23": 6, "dictionary_v4": 2},
        "pn_participation_by_label": {"连帽": 4},
        "pu_positive_participation_by_label": {"前门襟": 2},
        "pu_unlabeled_participation_by_label": {"前门襟": 6},
    }
    checkpoint = build_v3_checkpoint(
        model=model,
        optimizer=optimizer,
        tag_order=schema["labels"],
        schema_sha256=schema["schema_sha256"],
        manifest_sha256="a" * 64,
        sampler_config=sampler_config,
        sampler_schedule_sha256="b" * 64,
        cursor={"global_step": 9, "global_batch_cursor": 11},
        sampling_statistics=stats,
        world_size=1,
        rank_rng_states=[{"cpu": torch.get_rng_state(), "cuda": []}],
    )
    assert checkpoint["format_version"] == 3
    assert checkpoint["tag_order"] == schema["labels"]
    assert checkpoint["schema_sha256"] == schema["schema_sha256"]
    assert checkpoint["sampling_statistics"] == stats
    assert checkpoint["pu_output_semantics"] == "uncalibrated_confidence"
    path = tmp_path / "checkpoint-v3.pt"
    torch.save(checkpoint, path)

    with torch.no_grad():
        model.classifier.weight.zero_()
    optimizer.state.clear()
    restored = restore_v3_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        expected_tag_order=schema["labels"],
        expected_schema_sha256=schema["schema_sha256"],
        expected_manifest_sha256="a" * 64,
        expected_sampler_config=sampler_config,
        expected_sampler_schedule_sha256="b" * 64,
        expected_world_size=1,
        rank=0,
    )

    torch.testing.assert_close(model.classifier.weight, saved_weight)
    assert restored["cursor"] == {"global_step": 9, "global_batch_cursor": 11}
    assert restored["sampling_statistics"] == stats
    assert optimizer.state_dict()["param_groups"] == saved_optimizer["param_groups"]
    assert optimizer.state_dict()["state"]


@requires_torch
def test_schema_loader_rejects_tampered_hash(tmp_path: Path) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    schema["label_order_note"] = "tampered"
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="schema SHA256 mismatch"):
        load_unified57_schema(path)


@requires_torch
def test_checkpoint_path_must_match_explicit_aggregate18_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema = load_unified57_schema(SCHEMA_PATH)
    checkpoint_path = tmp_path / "aggregate18.pt"
    torch.save(_source_state(_TinyTransferModel()), checkpoint_path)
    actual_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    transfer_aggregate18_v2_checkpoint(
        _TinyTransferModel(),
        checkpoint_path,
        target_tag_order=schema["labels"],
        expected_lora_tensors=1,
        expected_checkpoint_sha256=actual_hash,
    )
    def forbidden_torch_load(*_args, **_kwargs):
        raise AssertionError("hash mismatch must fail before torch.load")

    monkeypatch.setattr(torch, "load", forbidden_torch_load)
    with pytest.raises(ValueError, match="checkpoint SHA256 mismatch"):
        transfer_aggregate18_v2_checkpoint(
            _TinyTransferModel(),
            checkpoint_path,
            target_tag_order=schema["labels"],
            expected_lora_tensors=1,
            expected_checkpoint_sha256="0" * 64,
        )


@requires_torch
def test_schedule_rejects_duplicate_record_ids() -> None:
    schema = load_unified57_schema(SCHEMA_PATH)
    records = _records_for_sampler(schema)
    records[1]["record_id"] = records[0]["record_id"]

    with pytest.raises(ValueError, match="record_id"):
        TwoStreamSchedule(
            records,
            tag_order=schema["labels"],
            training_modes=schema["label_training_modes"],
            world_size=8,
            uniform_per_rank=6,
            balanced_per_rank=2,
            seed=17,
        )


@requires_torch
def test_pos_weight_audit_enumerates_deterministic_schedule_exposures() -> None:
    schema = load_unified57_schema(SCHEMA_PATH)
    records = _records_for_sampler(schema)
    schedule = TwoStreamSchedule(
        records,
        tag_order=schema["labels"],
        training_modes=schema["label_training_modes"],
        world_size=8,
        uniform_per_rank=6,
        balanced_per_rank=2,
        seed=17,
    )

    _, audit = compute_pn_pos_weight(records, schema, cap=20.0, schedule=schedule)

    assert audit["sampler_schedule_sha256"] == schedule.schedule_sha256
    assert audit["count_scope"] == "deterministic two-stream schedule exposures"
    assert audit["record_exposures"] == sum(len(batch) for batch in schedule.global_batches)


@requires_torch
def test_v3_resume_rejects_run_contract_drift_before_model_mutation(tmp_path: Path) -> None:
    schema = load_unified57_schema(SCHEMA_PATH)
    model = _TinyTransferModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint = build_v3_checkpoint(
        model=model,
        optimizer=optimizer,
        tag_order=schema["labels"],
        schema_sha256=schema["schema_sha256"],
        manifest_sha256="a" * 64,
        sampler_config={"world_size": 1},
        sampler_schedule_sha256="b" * 64,
        cursor={"global_step": 1, "global_batch_cursor": 1},
        sampling_statistics={},
        world_size=1,
        rank_rng_states=[{"cpu": torch.get_rng_state(), "cuda": []}],
        run_contract={"image_max_pixels": 112896, "lora_alpha": 32},
    )
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    before = model.classifier.weight.detach().clone()

    with pytest.raises(ValueError, match="run_contract mismatch"):
        restore_v3_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            expected_tag_order=schema["labels"],
            expected_schema_sha256=schema["schema_sha256"],
            expected_manifest_sha256="a" * 64,
            expected_sampler_config={"world_size": 1},
            expected_sampler_schedule_sha256="b" * 64,
            expected_world_size=1,
            rank=0,
            expected_run_contract={"image_max_pixels": 196000, "lora_alpha": 32},
        )
    torch.testing.assert_close(model.classifier.weight, before)
