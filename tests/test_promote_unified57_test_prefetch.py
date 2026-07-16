from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from scripts import promote_unified57_test_prefetch as promoter
from scripts.promote_unified57_test_prefetch import (
    PromotionConfig,
    PromotionError,
    main,
    publish_prefetch,
    validate_prefetch,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_json_bytes(row) for row in rows))


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _base_provenance(model: Path) -> dict:
    files = []
    for relative, role in (
        ("config.json", "config"),
        ("model.safetensors", "weights"),
        ("preprocessor_config.json", "processor"),
        ("tokenizer.json", "tokenizer"),
    ):
        path = model / relative
        files.append(
            {
                "path": relative,
                "role": role,
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    payload = {"version": "unified57_base_artifacts_v1", "files": files}
    return {**payload, "manifest_sha256": _canonical_sha256(payload)}


def _fixture(tmp_path: Path, *, records: int = 9) -> PromotionConfig:
    project = tmp_path / "project"
    source = tmp_path / "run" / "test_prefetch"
    evaluation = tmp_path / "run" / "evaluation"
    data = tmp_path / "dataset"
    model = tmp_path / "model"
    cache = tmp_path / "cache"
    checkpoint = tmp_path / "run" / "checkpoints" / "latest.pt"
    evaluator = project / "scripts" / "evaluate_unified57_multilabel.py"
    trainer = project / "scripts" / "train_unified57_qwen3vl_multilabel.py"
    cache_builder = project / "scripts" / "build_unified57_eval_image_cache.py"
    classifier_core = project / "scripts" / "jd_multilabel_training_core.py"
    evaluation_core = project / "scripts" / "unified57_evaluation_core.py"
    schema = project / "configs" / "bosideng_unified57_schema.json"
    trainable = project / "configs" / "expected_trainable_manifest.json"
    base = project / "configs" / "base_artifact_manifest.json"
    validation_manifest = data / "val.jsonl"
    test_manifest = data / "test.jsonl"

    evaluator.parent.mkdir(parents=True)
    evaluator.write_text("# frozen evaluator\n", encoding="utf-8")
    trainer.write_text('VISION_PROMPT = "识别服装字典词"\n', encoding="utf-8")
    cache_builder.write_text(
        'def training_vision_prompt():\n    return "识别服装字典词"\n',
        encoding="utf-8",
    )
    classifier_core.write_text("# frozen classifier core\n", encoding="utf-8")
    evaluation_core.write_text("# frozen evaluation core\n", encoding="utf-8")
    model.mkdir(parents=True)
    _write_json(model / "config.json", {"model_type": "qwen3_vl"})
    (model / "model.safetensors").write_bytes(b"weights")
    _write_json(model / "preprocessor_config.json", {"size": 336})
    _write_json(model / "tokenizer.json", {"version": "1.0"})
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"frozen-checkpoint")

    labels = [f"标签{i:02d}" for i in range(57)]
    internal_schema_sha = "1" * 64
    _write_json(
        schema,
        {
            "schema_version": "test-v1",
            "schema_sha256": internal_schema_sha,
            "labels": labels,
            "label_training_modes": {label: "pn" for label in labels},
        },
    )
    _write_json(
        trainable,
        {
            "version": "unified57_expected_trainable_v1",
            "schema_sha256": internal_schema_sha,
            "base_model_config_sha256": _sha256(model / "config.json"),
        },
    )
    base_payload = _base_provenance(model)
    base_contract_sha = base_payload["manifest_sha256"]
    _write_json(base, base_payload)
    _write_jsonl(validation_manifest, [{"record_id": "validation-0"}])

    manifest_rows = []
    for index in range(records):
        manifest_rows.append(
            {
                "record_id": f"record-{index:03d}",
                "schema_sha256": internal_schema_sha,
                "labels": [0.0] * 57,
                "known_mask": [0] * 57,
                "pu_positive_mask": [0] * 57,
                "sources": ["jd_complete23"],
            }
        )
    _write_jsonl(test_manifest, manifest_rows)
    preflight = tmp_path / "run" / "posttrain" / "preflight_contract.json"
    _write_json(
        preflight,
        {
            "dataset_sha256": {
                "val": _sha256(validation_manifest),
                "test": _sha256(test_manifest),
            },
            "leakage_passed": True,
            "leakage_counts": {
                "cross_split_components": 0,
                "cross_split_exact_phash": 0,
                "cross_split_sha256": 0,
            },
            "base_artifact_manifest_sha256": base_contract_sha,
            "base_artifact_manifest_file_sha256": _sha256(base),
            "base_model_config_sha256": _sha256(model / "config.json"),
            "expected_trainable_manifest_sha256": _sha256(trainable),
        },
    )

    cache.mkdir(parents=True)
    (cache / "cache_manifest.jsonl").write_bytes(
        b'{"record_id":"cache-record"}\n'
    )
    cache_manifest_sha = _sha256(cache / "cache_manifest.jsonl")
    decoder_sha = "4" * 64
    _write_json(
        cache / "complete.json",
        {
            "version": "bosideng_unified57_eval_rgb_cache_v1",
            "status": "complete",
            "record_count": records + 1,
            "split_counts": {"validation": 1, "test": records},
            "validation_manifest_sha256": _sha256(validation_manifest),
            "test_manifest_sha256": _sha256(test_manifest),
            "image_max_pixels": 112896,
            "decoder_contract_sha256": decoder_sha,
            "cache_manifest_sha256": cache_manifest_sha,
        },
    )
    complete_marker_sha = _sha256(cache / "complete.json")
    _write_json(
        cache / "runtime_validated.json",
        {
            "version": "bosideng_unified57_eval_rgb_cache_v1",
            "status": "validated",
            "record_count": records + 1,
            "cache_manifest_sha256": cache_manifest_sha,
            "complete_marker_sha256": complete_marker_sha,
            "image_max_pixels": 112896,
            "decoder_contract_sha256": decoder_sha,
        },
    )

    prediction_rows: list[dict] = []
    shard_dir = source / "prediction_shards"
    for rank in range(8):
        rows = []
        for manifest_row in manifest_rows[rank::8]:
            row = dict(manifest_row)
            row.update(
                {
                    "width": 100,
                    "height": 200,
                    "aspect_ratio": 0.5,
                    "scores": [0.25] * 57,
                    "checkpoint_sha256": _sha256(checkpoint),
                }
            )
            rows.append(row)
            prediction_rows.append(row)
        shard = shard_dir / f"test.rank{rank:02d}-of-08.jsonl"
        _write_jsonl(shard, rows)
        metadata = {
            "split": "test",
            "rank": rank,
            "world_size": 8,
            "checkpoint_sha256": _sha256(checkpoint),
            "manifest_sha256": _sha256(test_manifest),
            "schema_sha256": internal_schema_sha,
            "image_cache_manifest_sha256": cache_manifest_sha,
            "image_cache_complete_marker_sha256": complete_marker_sha,
            "image_cache_decoder_contract_sha256": decoder_sha,
        }
        _write_json(
            shard.with_name(shard.name + ".progress.json"),
            {
                "version": 1,
                **metadata,
                "metadata": metadata,
                "durable_records": len(rows),
                "durable_offset": shard.stat().st_size,
                "next_local_index": len(rows),
                "last_record_id": rows[-1]["record_id"] if rows else None,
                "complete": True,
            },
        )

    by_id = {row["record_id"]: row for row in prediction_rows}
    ordered_predictions = [by_id[row["record_id"]] for row in manifest_rows]
    _write_jsonl(source / "test_predictions_float32.jsonl", ordered_predictions)
    _write_json(
        source / "prediction_only_report.json",
        {
            "report_version": "bosideng-unified57-ddp-evaluation-v1",
            "mode": "prediction_only",
            "split": "test",
            "state": "complete",
            "expected_records": records,
            "predicted_records": records,
            "checkpoint_sha256": _sha256(checkpoint),
            "manifest_sha256": _sha256(test_manifest),
            "schema_sha256": internal_schema_sha,
            "world_size": 8,
            "batch_size": 8,
            "image_cache": {
                "enabled": True,
                "cache_manifest_sha256": cache_manifest_sha,
                "complete_marker_sha256": complete_marker_sha,
                "decoder_contract_sha256": decoder_sha,
            },
        },
    )

    argv = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        str(evaluator),
        "--model",
        str(model),
        "--model-config-sha256",
        _sha256(model / "config.json"),
        "--schema",
        str(schema),
        "--schema-file-sha256",
        _sha256(schema),
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-sha256",
        _sha256(checkpoint),
        "--validation-manifest",
        str(validation_manifest),
        "--validation-manifest-sha256",
        _sha256(validation_manifest),
        "--test-manifest",
        str(test_manifest),
        "--test-manifest-sha256",
        _sha256(test_manifest),
        "--expected-trainable-manifest-sha256",
        _sha256(trainable),
        "--base-artifact-manifest-sha256",
        base_contract_sha,
        "--output-dir",
        str(source),
        "--wall-clock-seconds",
        "2400",
        "--expected-world-size",
        "8",
        "--batch-size",
        "8",
        "--num-workers",
        "4",
        "--image-max-pixels",
        "112896",
        "--image-cache-root",
        str(cache),
        "--lora-rank",
        "16",
        "--lora-alpha",
        "32",
        "--lora-dropout",
        "0.05",
        "--head-dropout",
        "0.1",
        "--prediction-only-split",
        "test",
    ]
    runtime_files = {
        "base_artifact_manifest": {"path": str(base), "sha256": _sha256(base)},
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "evaluator": {"path": str(evaluator), "sha256": _sha256(evaluator)},
        "expected_trainable_manifest": {
            "path": str(trainable),
            "sha256": _sha256(trainable),
        },
        "model_config": {
            "path": str(model / "config.json"),
            "sha256": _sha256(model / "config.json"),
        },
        "schema": {"path": str(schema), "sha256": _sha256(schema)},
        "test_manifest": {
            "path": str(test_manifest),
            "sha256": _sha256(test_manifest),
        },
        "eval_image_cache_builder": {
            "path": str(cache_builder),
            "sha256": _sha256(cache_builder),
        },
        "classifier_core": {
            "path": str(classifier_core),
            "sha256": _sha256(classifier_core),
        },
        "unified57_trainer": {
            "path": str(trainer),
            "sha256": _sha256(trainer),
        },
        "evaluation_core": {
            "path": str(evaluation_core),
            "sha256": _sha256(evaluation_core),
        },
    }
    _write_json(
        source / "invocation_snapshot.json",
        {
            "version": 1,
            "unit": "prefetch.service",
            "main_pid": 1234,
            "captured_at_unix": 12345.0,
            "argv": argv,
            "runtime_files": runtime_files,
        },
    )
    (evaluation / "prediction_shards").mkdir(parents=True)

    return PromotionConfig(
        source_dir=source,
        evaluation_dir=evaluation,
        evaluator_path=evaluator,
        python_executable=Path(sys.executable),
        model_dir=model,
        schema_path=schema,
        checkpoint_path=checkpoint,
        validation_manifest_path=validation_manifest,
        test_manifest_path=test_manifest,
        expected_trainable_manifest_path=trainable,
        base_artifact_manifest_path=base,
        trainer_path=trainer,
        cache_builder_path=cache_builder,
        classifier_core_path=classifier_core,
        evaluation_core_path=evaluation_core,
        preflight_contract_path=preflight,
        image_cache_root=cache,
        prefetch_service_unit="prefetch.service",
        formal_service_unit="posttrain.service",
        expected_records=records,
    )


def test_validate_accepts_complete_contract_and_builds_inference_contract(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)

    result = validate_prefetch(config)

    assert result["status"] == "pass"
    assert result["records"] == 9
    assert result["rank_counts"] == [2, 1, 1, 1, 1, 1, 1, 1]
    assert len(result["inventory"]) == 16
    assert len(result["merged_predictions_sha256"]) == 64
    assert len(result["ordered_record_ids_sha256"]) == 64
    assert len(result["inference_contract_sha256"]) == 64
    contract = result["inference_contract"]
    assert contract["model"]["path"] == str(config.model_dir)
    assert contract["schema"]["internal_sha256"] == "1" * 64
    assert contract["base_artifact"]["contract_sha256"] == json.loads(
        config.base_artifact_manifest_path.read_text()
    )["manifest_sha256"]
    assert contract["lora"] == {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "head_dropout": 0.1,
    }
    assert contract["batch_size"] == 8
    assert contract["world_size"] == 8
    assert contract["image_max_pixels"] == 112896
    assert contract["vision_prompt_sha256"] == hashlib.sha256(
        "识别服装字典词".encode("utf-8")
    ).hexdigest()
    assert contract["runtime_dependencies"] == {
        "eval_image_cache_builder": {
            "path": str(config.cache_builder_path),
            "sha256": _sha256(config.cache_builder_path),
        },
        "classifier_core": {
            "path": str(config.classifier_core_path),
            "sha256": _sha256(config.classifier_core_path),
        },
        "unified57_trainer": {
            "path": str(config.trainer_path),
            "sha256": _sha256(config.trainer_path),
        },
        "evaluation_core": {
            "path": str(config.evaluation_core_path),
            "sha256": _sha256(config.evaluation_core_path),
        },
    }


def _fake_systemctl(tmp_path: Path, values: dict[str, str]) -> Path:
    (tmp_path / "proc").mkdir(exist_ok=True)
    executable = tmp_path / "systemctl"
    output = "".join(f"{key}={value}\n" for key, value in values.items())
    executable.write_text(
        "#!/bin/sh\ncat <<'EOF'\n" + output + "EOF\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def test_publish_rejects_an_active_service_with_running_processes(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    cgroup_root = tmp_path / "cgroup"
    proc_root = tmp_path / "proc"
    (cgroup_root / "posttrain.service").mkdir(parents=True)
    (cgroup_root / "posttrain.service" / "cgroup.procs").write_text("4321\n")
    (proc_root / "4321").mkdir(parents=True)
    (proc_root / "4321" / "status").write_text("Name:\ttest\nState:\tR (running)\n")
    systemctl = _fake_systemctl(
        tmp_path,
        {
            "LoadState": "loaded",
            "ActiveState": "active",
            "SubState": "running",
            "ControlGroup": "/posttrain.service",
            "MainPID": "4321",
        },
    )

    with pytest.raises(PromotionError, match="not stopped or fully SIGSTOP"):
        publish_prefetch(
            config,
            systemctl_path=systemctl,
            cgroup_root=cgroup_root,
            proc_root=proc_root,
        )

    assert list((config.evaluation_dir / "prediction_shards").iterdir()) == []
    assert not (config.evaluation_dir / "test_prefetch_import_contract.json").exists()


def test_publish_stages_backs_up_and_atomically_installs_complete_shards(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    target = config.evaluation_dir / "prediction_shards"
    old_shard = target / "test.rank00-of-08.jsonl"
    old_sidecar = target / "test.rank00-of-08.jsonl.progress.json"
    old_shard.write_bytes(b"partial\n")
    old_sidecar.write_bytes(b'{"complete":false}\n')
    old_receipt = config.evaluation_dir / "test_prefetch_import_contract.json"
    old_receipt.write_bytes(b'{"old":true}\n')
    systemctl = _fake_systemctl(
        tmp_path,
        {
            "LoadState": "loaded",
            "ActiveState": "inactive",
            "SubState": "dead",
            "ControlGroup": "",
            "MainPID": "0",
        },
    )

    result = publish_prefetch(
        config,
        systemctl_path=systemctl,
        cgroup_root=tmp_path / "cgroup",
        proc_root=tmp_path / "proc",
    )

    source = config.source_dir / "prediction_shards"
    target_files = sorted(path.name for path in target.iterdir())
    assert target_files == sorted(path.name for path in source.iterdir())
    for name in target_files:
        assert _sha256(target / name) == _sha256(source / name)
    receipt = json.loads(old_receipt.read_text())
    assert receipt["version"] == "bosideng-unified57-test-prefetch-import-v1"
    assert receipt["status"] == "complete"
    assert receipt["quiescence"]["mode"] == "stopped"
    assert receipt["inference_contract_sha256"] == result["inference_contract_sha256"]
    assert receipt["inference_contract"]["merged_predictions_sha256"] == result[
        "merged_predictions_sha256"
    ]
    assert receipt["inference_contract"]["prediction_shard_inventory"] == result[
        "inventory"
    ]
    backup = Path(receipt["backup_dir"])
    assert (backup / old_shard.name).read_bytes() == b"partial\n"
    assert (backup / old_sidecar.name).read_bytes() == b'{"complete":false}\n'
    assert (backup / "test_prefetch_import_contract.json").read_bytes() == b'{"old":true}\n'
    assert receipt["previous_target_inventory"] == {
        old_shard.name: {
            "bytes": len(b"partial\n"),
            "sha256": hashlib.sha256(b"partial\n").hexdigest(),
        },
        old_sidecar.name: {
            "bytes": len(b'{"complete":false}\n'),
            "sha256": hashlib.sha256(b'{"complete":false}\n').hexdigest(),
        },
        "test_prefetch_import_contract.json": {
            "bytes": len(b'{"old":true}\n'),
            "sha256": hashlib.sha256(b'{"old":true}\n').hexdigest(),
        },
    }
    assert not any(path.name.startswith(".test_prefetch_stage") for path in config.evaluation_dir.iterdir())


def test_publish_rejects_sigstopped_service_with_an_open_test_shard_fd(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    target = config.evaluation_dir / "prediction_shards"
    partial = target / "test.rank00-of-08.jsonl"
    partial.write_bytes(b"partial\n")
    cgroup_root = tmp_path / "cgroup"
    proc_root = tmp_path / "proc"
    (cgroup_root / "posttrain.service").mkdir(parents=True)
    (cgroup_root / "posttrain.service" / "cgroup.procs").write_text("4321\n")
    (proc_root / "4321" / "fd").mkdir(parents=True)
    (proc_root / "4321" / "status").write_text("State:\tT (stopped)\n")
    (proc_root / "4321" / "fd" / "7").symlink_to(partial)
    systemctl = _fake_systemctl(
        tmp_path,
        {
            "LoadState": "loaded",
            "ActiveState": "active",
            "SubState": "running",
            "ControlGroup": "/posttrain.service",
            "MainPID": "4321",
        },
    )

    with pytest.raises(PromotionError, match="open formal test shard"):
        publish_prefetch(
            config,
            systemctl_path=systemctl,
            cgroup_root=cgroup_root,
            proc_root=proc_root,
        )

    assert partial.read_bytes() == b"partial\n"
    assert not (config.evaluation_dir / "test_prefetch_import_contract.json").exists()


def test_publish_rolls_back_without_touching_live_files_when_staging_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture(tmp_path)
    target = config.evaluation_dir / "prediction_shards"
    partial = target / "test.rank00-of-08.jsonl"
    partial.write_bytes(b"original-partial\n")
    old_receipt = config.evaluation_dir / "test_prefetch_import_contract.json"
    old_receipt.write_bytes(b'{"old":true}\n')
    systemctl = _fake_systemctl(
        tmp_path,
        {
            "LoadState": "loaded",
            "ActiveState": "inactive",
            "SubState": "dead",
            "ControlGroup": "",
            "MainPID": "0",
        },
    )
    original_copy2 = promoter.shutil.copy2
    calls = 0

    def fail_second_copy(source: Path, destination: Path) -> Path:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected staging failure")
        return original_copy2(source, destination)

    monkeypatch.setattr(promoter.shutil, "copy2", fail_second_copy)

    with pytest.raises(OSError, match="injected staging failure"):
        publish_prefetch(
            config,
            systemctl_path=systemctl,
            cgroup_root=tmp_path / "cgroup",
            proc_root=tmp_path / "proc",
        )

    assert partial.read_bytes() == b"original-partial\n"
    assert old_receipt.read_bytes() == b'{"old":true}\n'
    assert sorted(path.name for path in target.iterdir()) == [partial.name]
    assert not any(path.name.startswith(".test_prefetch_stage") for path in config.evaluation_dir.iterdir())


def _cli_args(config: PromotionConfig) -> list[str]:
    return [
        "--source-dir",
        str(config.source_dir),
        "--evaluation-dir",
        str(config.evaluation_dir),
        "--evaluator",
        str(config.evaluator_path),
        "--python-executable",
        str(config.python_executable),
        "--model",
        str(config.model_dir),
        "--schema",
        str(config.schema_path),
        "--checkpoint",
        str(config.checkpoint_path),
        "--validation-manifest",
        str(config.validation_manifest_path),
        "--test-manifest",
        str(config.test_manifest_path),
        "--expected-trainable-manifest",
        str(config.expected_trainable_manifest_path),
        "--base-artifact-manifest",
        str(config.base_artifact_manifest_path),
        "--trainer",
        str(config.trainer_path),
        "--cache-builder",
        str(config.cache_builder_path),
        "--classifier-core",
        str(config.classifier_core_path),
        "--evaluation-core",
        str(config.evaluation_core_path),
        "--preflight-contract",
        str(config.preflight_contract_path),
        "--image-cache-root",
        str(config.image_cache_root),
        "--prefetch-service-unit",
        config.prefetch_service_unit,
        "--formal-service-unit",
        config.formal_service_unit,
        "--expected-records",
        str(config.expected_records),
    ]


def test_validate_cli_emits_machine_readable_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _fixture(tmp_path)

    exit_code = main(["validate", *_cli_args(config)])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "pass"
    assert payload["records"] == config.expected_records
    assert payload["inference_contract_sha256"] == hashlib.sha256(
        json.dumps(
            payload["inference_contract"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_validate_rejects_a_cache_manifest_changed_after_completion(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    (config.image_cache_root / "cache_manifest.jsonl").write_bytes(b"tampered\n")

    with pytest.raises(PromotionError, match="cache manifest file SHA256"):
        validate_prefetch(config)


def test_validate_rejects_semantically_wrong_trainable_contract_even_when_snapshot_matches(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    _write_json(
        config.expected_trainable_manifest_path,
        {
            "version": "unified57_expected_trainable_v1",
            "schema_sha256": "9" * 64,
            "base_model_config_sha256": _sha256(config.model_dir / "config.json"),
        },
    )
    new_sha = _sha256(config.expected_trainable_manifest_path)
    snapshot_path = config.source_dir / "invocation_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    flag_index = snapshot["argv"].index("--expected-trainable-manifest-sha256")
    snapshot["argv"][flag_index + 1] = new_sha
    snapshot["runtime_files"]["expected_trainable_manifest"]["sha256"] = new_sha
    _write_json(snapshot_path, snapshot)

    with pytest.raises(PromotionError, match="trainable schema contract"):
        validate_prefetch(config)


def test_publish_rejects_inactive_service_with_residual_cgroup_process(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    cgroup_root = tmp_path / "cgroup"
    proc_root = tmp_path / "proc"
    (cgroup_root / "posttrain.service").mkdir(parents=True)
    (cgroup_root / "posttrain.service" / "cgroup.procs").write_text("4321\n")
    (proc_root / "4321").mkdir(parents=True)
    (proc_root / "4321" / "status").write_text("State:\tR (running)\n")
    systemctl = _fake_systemctl(
        tmp_path,
        {
            "LoadState": "loaded",
            "ActiveState": "inactive",
            "SubState": "dead",
            "ControlGroup": "/posttrain.service",
            "MainPID": "0",
        },
    )

    with pytest.raises(PromotionError, match="residual processes"):
        publish_prefetch(
            config,
            systemctl_path=systemctl,
            cgroup_root=cgroup_root,
            proc_root=proc_root,
        )


def test_publish_fails_closed_when_sigstopped_process_fds_cannot_be_inspected(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    cgroup_root = tmp_path / "cgroup"
    proc_root = tmp_path / "proc"
    (cgroup_root / "posttrain.service").mkdir(parents=True)
    (cgroup_root / "posttrain.service" / "cgroup.procs").write_text("4321\n")
    (proc_root / "4321").mkdir(parents=True)
    (proc_root / "4321" / "status").write_text("State:\tT (stopped)\n")
    systemctl = _fake_systemctl(
        tmp_path,
        {
            "LoadState": "loaded",
            "ActiveState": "active",
            "SubState": "running",
            "ControlGroup": "/posttrain.service",
            "MainPID": "4321",
        },
    )

    with pytest.raises(PromotionError, match="file descriptors"):
        publish_prefetch(
            config,
            systemctl_path=systemctl,
            cgroup_root=cgroup_root,
            proc_root=proc_root,
        )


def test_publish_rechecks_quiescence_after_staging_before_replacing_live_files(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    target = config.evaluation_dir / "prediction_shards"
    partial = target / "test.rank00-of-08.jsonl"
    partial.write_bytes(b"original\n")
    cgroup_root = tmp_path / "cgroup"
    proc_root = tmp_path / "proc"
    (cgroup_root / "posttrain.service").mkdir(parents=True)
    (cgroup_root / "posttrain.service" / "cgroup.procs").write_text("4321\n")
    (proc_root / "4321" / "fd").mkdir(parents=True)
    status_path = proc_root / "4321" / "status"
    status_path.write_text("State:\tT (stopped)\n")
    count_path = tmp_path / "systemctl.count"
    systemctl = tmp_path / "systemctl"
    systemctl.write_text(
        "#!/bin/sh\n"
        f"COUNT={count_path!s}\n"
        "N=0\n"
        "test -f \"$COUNT\" && N=$(cat \"$COUNT\")\n"
        "N=$((N+1))\n"
        "echo \"$N\" > \"$COUNT\"\n"
        f"test \"$N\" -lt 2 || printf 'State:\\tR (running)\\n' > {status_path!s}\n"
        "cat <<'EOF'\n"
        "LoadState=loaded\nActiveState=active\nSubState=running\n"
        "ControlGroup=/posttrain.service\nMainPID=4321\nEOF\n",
        encoding="utf-8",
    )
    systemctl.chmod(0o755)

    with pytest.raises(PromotionError, match="not stopped or fully SIGSTOP"):
        publish_prefetch(
            config,
            systemctl_path=systemctl,
            cgroup_root=cgroup_root,
            proc_root=proc_root,
        )

    assert int(count_path.read_text()) >= 2
    assert partial.read_bytes() == b"original\n"
    assert not (config.evaluation_dir / "test_prefetch_import_contract.json").exists()


def test_manifest_contract_rejects_a_non_neutral_unknown_label(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    schema = json.loads(config.schema_path.read_text())
    row = json.loads(config.test_manifest_path.read_text().splitlines()[0])
    row["labels"][0] = 1.0
    row["known_mask"][0] = 0
    row["pu_positive_mask"][0] = 0

    with pytest.raises(PromotionError, match="unknown label is non-neutral"):
        promoter.validate_manifest_rows([row], schema)


@pytest.mark.parametrize(
    "flag",
    [
        "--model",
        "--model-config-sha256",
        "--schema-file-sha256",
        "--checkpoint-sha256",
        "--expected-trainable-manifest-sha256",
        "--base-artifact-manifest-sha256",
        "--test-manifest-sha256",
        "--expected-world-size",
        "--batch-size",
        "--image-max-pixels",
        "--image-cache-root",
        "--lora-rank",
        "--lora-alpha",
        "--lora-dropout",
        "--head-dropout",
    ],
)
def test_validate_rejects_any_tampered_frozen_invocation_option(
    tmp_path: Path, flag: str
) -> None:
    config = _fixture(tmp_path)
    snapshot_path = config.source_dir / "invocation_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    index = snapshot["argv"].index(flag)
    snapshot["argv"][index + 1] = "tampered"
    _write_json(snapshot_path, snapshot)

    with pytest.raises(PromotionError, match="invocation argv"):
        validate_prefetch(config)


def test_validate_calls_cache_builder_and_rejects_prompt_drift(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    config.cache_builder_path.write_text(
        'def training_vision_prompt():\n    return "不同提示词"\n',
        encoding="utf-8",
    )
    snapshot_path = config.source_dir / "invocation_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    snapshot["runtime_files"]["eval_image_cache_builder"]["sha256"] = _sha256(
        config.cache_builder_path
    )
    _write_json(snapshot_path, snapshot)

    with pytest.raises(PromotionError, match="differs from trainer VISION_PROMPT"):
        validate_prefetch(config)


def test_publish_cli_writes_the_import_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _fixture(tmp_path)
    systemctl = _fake_systemctl(
        tmp_path,
        {
            "LoadState": "loaded",
            "ActiveState": "inactive",
            "SubState": "dead",
            "ControlGroup": "",
            "MainPID": "0",
        },
    )

    exit_code = main(
        [
            "publish",
            *_cli_args(config),
            "--systemctl-path",
            str(systemctl),
            "--cgroup-root",
            str(tmp_path / "cgroup"),
            "--proc-root",
            str(tmp_path / "proc"),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    receipt_path = Path(payload["receipt_path"])
    assert receipt_path == config.evaluation_dir / "test_prefetch_import_contract.json"
    assert json.loads(receipt_path.read_text())["inference_contract_sha256"] == payload[
        "inference_contract_sha256"
    ]


def test_publish_restores_partial_files_if_replacement_fails_mid_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture(tmp_path)
    target = config.evaluation_dir / "prediction_shards"
    old_shard = target / "test.rank00-of-08.jsonl"
    old_sidecar = target / "test.rank00-of-08.jsonl.progress.json"
    old_shard.write_bytes(b"old-shard\n")
    old_sidecar.write_bytes(b"old-sidecar\n")
    receipt = config.evaluation_dir / "test_prefetch_import_contract.json"
    receipt.write_bytes(b"old-receipt\n")
    systemctl = _fake_systemctl(
        tmp_path,
        {
            "LoadState": "loaded",
            "ActiveState": "inactive",
            "SubState": "dead",
            "ControlGroup": "",
            "MainPID": "0",
        },
    )
    original_replace = promoter.os.replace
    installs = 0

    def fail_second_install(source: Path | str, destination: Path | str) -> None:
        nonlocal installs
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            source_path.parent.name.startswith(".test_prefetch_stage.")
            and destination_path.parent == target
        ):
            installs += 1
            if installs == 2:
                raise OSError("injected install failure")
        original_replace(source, destination)

    monkeypatch.setattr(promoter.os, "replace", fail_second_install)

    with pytest.raises(OSError, match="injected install failure"):
        publish_prefetch(
            config,
            systemctl_path=systemctl,
            cgroup_root=tmp_path / "cgroup",
            proc_root=tmp_path / "proc",
        )

    assert sorted(path.name for path in target.iterdir()) == sorted(
        [old_shard.name, old_sidecar.name]
    )
    assert old_shard.read_bytes() == b"old-shard\n"
    assert old_sidecar.read_bytes() == b"old-sidecar\n"
    assert receipt.read_bytes() == b"old-receipt\n"
