from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pytest

from scripts import package_unified57_delivery as delivery
from scripts.supervise_unified57_posttrain import (
    DeadlineExceeded,
    PostTrainSupervisor,
    SupervisorConfig,
    atomic_write_json,
    build_final_report,
    build_preflight_contract,
    build_reproduction_result,
    evaluation_is_complete,
    evaluation_exit_is_resumable,
    evaluation_slice_is_resumable,
    main,
    reproduction_result_is_current,
    run_owned_process,
    validate_candidate_resume,
    validate_sealed_resume,
    verify_sha256s,
    validate_training_completion,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _fake_base_model(root: Path) -> Path:
    root.mkdir()
    for name, payload in {
        "config.json": b'{"model_type":"qwen3_vl"}\n',
        "model.safetensors.index.json": b'{"weight_map":{"x":"model-00001-of-00001.safetensors"}}\n',
        "model-00001-of-00001.safetensors": b"weights",
        "preprocessor_config.json": b'{"size":336}\n',
        "tokenizer_config.json": b'{"model_max_length":32768}\n',
    }.items():
        (root / name).write_bytes(payload)
    return root


def _dataset(root: Path) -> tuple[dict[str, str], Path]:
    root.mkdir()
    hashes = {}
    for split in ("train", "val", "test"):
        path = root / f"{split}.jsonl"
        path.write_text(json.dumps({"record_id": split}) + "\n")
        hashes[split] = _sha256(path)
    leakage = root / "leakage_check.json"
    _write_json(
        leakage,
        {
            "cross_split_components": [],
            "cross_split_exact_phash": [],
            "cross_split_sha256": [],
            "pHash_hamming_threshold": 2,
            "passed": True,
        },
    )
    hashes["leakage"] = _sha256(leakage)
    return hashes, leakage


def _write_valid_trainable(path: Path) -> None:
    tensors = {
        f"backbone.layer.{index:03d}.lora_A.default.weight": {
            "shape": [16, 32],
            "dtype": "torch.float32",
        }
        for index in range(144)
    }
    tensors.update(
        {
            f"backbone.layer.{index:03d}.lora_B.default.weight": {
                "shape": [32, 16],
                "dtype": "torch.float32",
            }
            for index in range(144)
        }
    )
    tensors["classifier.weight"] = {"shape": [57, 4096], "dtype": "torch.float32"}
    tensors["classifier.bias"] = {"shape": [57], "dtype": "torch.float32"}
    _write_json(
        path,
        {
            "version": "unified57_expected_trainable_v1",
            "contract_kind": "aggregate18_v2_plus_unified57_classifier",
            "tensors": tensors,
        },
    )


def test_preflight_uses_canonical_base_manifest_hash_and_file_sha_for_trainable(
    tmp_path: Path,
) -> None:
    base = _fake_base_model(tmp_path / "base")
    dataset_hashes, leakage = _dataset(tmp_path / "dataset")
    expected_trainable = tmp_path / "expected_trainable_manifest.json"
    _write_valid_trainable(expected_trainable)
    base_manifest = tmp_path / "base_artifact_manifest.json"
    base_payload = delivery.build_base_artifact_provenance(base)
    _write_json(base_manifest, base_payload)

    contract = build_preflight_contract(
        dataset_root=tmp_path / "dataset",
        base_model=base,
        base_manifest_path=base_manifest,
        expected_trainable_manifest=expected_trainable,
        expected_dataset_sha256=dataset_hashes,
    )

    assert contract["base_artifact_manifest_sha256"] == base_payload["manifest_sha256"]
    assert contract["base_artifact_manifest_sha256"] != _sha256(base_manifest)
    assert contract["expected_trainable_manifest_sha256"] == _sha256(expected_trainable)
    assert contract["leakage_check_sha256"] == _sha256(leakage)


def test_preflight_rejects_nonzero_visual_leakage(tmp_path: Path) -> None:
    base = _fake_base_model(tmp_path / "base")
    dataset_hashes, leakage = _dataset(tmp_path / "dataset")
    payload = json.loads(leakage.read_text())
    payload["passed"] = False
    payload["cross_split_sha256"] = ["collision"]
    _write_json(leakage, payload)
    dataset_hashes["leakage"] = _sha256(leakage)
    trainable = tmp_path / "trainable.json"
    _write_json(trainable, {"version": "v1"})
    base_manifest = tmp_path / "base_manifest.json"
    _write_json(base_manifest, delivery.build_base_artifact_provenance(base))

    with pytest.raises(ValueError, match="leakage"):
        build_preflight_contract(
            dataset_root=tmp_path / "dataset",
            base_model=base,
            base_manifest_path=base_manifest,
            expected_trainable_manifest=trainable,
            expected_dataset_sha256=dataset_hashes,
        )


def test_preflight_rejects_unusable_trainable_contract_before_evaluation(
    tmp_path: Path,
) -> None:
    base = _fake_base_model(tmp_path / "base")
    dataset_hashes, _ = _dataset(tmp_path / "dataset")
    trainable = tmp_path / "trainable.json"
    _write_json(
        trainable,
        {
            "version": "unified57_expected_trainable_v1",
            "contract_kind": "synthetic_test",
            "tensors": {},
        },
    )
    base_manifest = tmp_path / "base_manifest.json"
    _write_json(base_manifest, delivery.build_base_artifact_provenance(base))
    with pytest.raises(ValueError, match="trainable"):
        build_preflight_contract(
            dataset_root=tmp_path / "dataset",
            base_model=base,
            base_manifest_path=base_manifest,
            expected_trainable_manifest=trainable,
            expected_dataset_sha256=dataset_hashes,
        )


def test_training_gate_requires_exit_zero_full_schedule_and_checkpoint(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    _write_json(run / "exit_status.json", {"state": "exited", "exit_code": 0})
    _write_json(
        run / "smoke_report.json",
        {
            "status": "complete",
            "global_step": 910,
            "target_reached": True,
            "loss_finite": True,
            "gradient_audit": {
                "trainable_tensors": 290,
                "with_gradient": 290,
                "finite": 290,
                "lora_tensors": 288,
                "lora_nonzero": 288,
            },
            "sampling_statistics": {
                "optimizer_steps": 910,
                "global_sample_exposures": 58240,
                "stream_exposures": {"uniform": 43680, "balanced": 14560},
                "balanced_selection_by_label": {
                    **{f"tag-{index}": 260 for index in range(56)},
                    "假两件": 0,
                },
            },
        },
    )
    _write_json(
        run / "progress.json",
        {"global_step": 910, "global_batch_cursor": 910, "global_batch_count": 910},
    )

    result = validate_training_completion(run, expected_steps=910)
    assert result["checkpoint_sha256"] == _sha256(checkpoint)
    assert result["global_step"] == 910

    _write_json(run / "exit_status.json", {"state": "failed", "exit_code": 4})
    with pytest.raises(ValueError, match="training exit"):
        validate_training_completion(run, expected_steps=910)

    _write_json(run / "exit_status.json", {"state": "exited", "exit_code": 0})
    summary = json.loads((run / "smoke_report.json").read_text())
    summary["loss_finite"] = False
    _write_json(run / "smoke_report.json", summary)
    with pytest.raises(ValueError, match="loss"):
        validate_training_completion(run, expected_steps=910)
    summary["loss_finite"] = True
    summary["gradient_audit"]["finite"] = 289
    _write_json(run / "smoke_report.json", summary)
    with pytest.raises(ValueError, match="gradient"):
        validate_training_completion(run, expected_steps=910)
    summary["gradient_audit"]["finite"] = 290
    summary["sampling_statistics"]["global_sample_exposures"] = 58239
    _write_json(run / "smoke_report.json", summary)
    with pytest.raises(ValueError, match="sampling"):
        validate_training_completion(run, expected_steps=910)
    summary["sampling_statistics"]["global_sample_exposures"] = 58240
    summary["sampling_statistics"]["balanced_selection_by_label"]["tag-0"] = 259
    _write_json(run / "smoke_report.json", summary)
    with pytest.raises(ValueError, match="balanced"):
        validate_training_completion(run, expected_steps=910)


def test_evaluation_ready_requires_complete_test_and_accepted_verdict(
    tmp_path: Path,
) -> None:
    report = tmp_path / "evaluation_report.json"
    _write_json(report, {"status": "partial", "test": {"complete": False}})
    assert evaluation_is_complete(report) is False
    _write_json(
        report,
        {
            "status": "success",
            "test": {"complete": True},
            "provenance": {"checkpoint_sha256": "a" * 64},
        },
    )
    assert evaluation_is_complete(report) is True
    assert (
        evaluation_is_complete(
            report, expected_provenance={"checkpoint_sha256": "a" * 64}
        )
        is True
    )
    assert (
        evaluation_is_complete(
            report, expected_provenance={"checkpoint_sha256": "b" * 64}
        )
        is False
    )
    _write_json(report, {"status": "fail", "test": {"complete": True}})
    assert evaluation_is_complete(report) is False


def test_torchrun_exit_one_is_resumable_only_with_fresh_partial_sidecar(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluation"
    output.mkdir()
    assert evaluation_exit_is_resumable(1, output) is False
    _write_json(output / "status.json", {"state": "partial", "test_complete": False})
    assert evaluation_slice_is_resumable(output) is True
    assert evaluation_exit_is_resumable(1, output) is True


def test_owned_process_timeout_kills_only_started_process_group(tmp_path: Path) -> None:
    child_pid = tmp_path / "child.pid"
    command = [
        sys.executable,
        "-c",
        (
            "import pathlib,time,os; "
            f"pathlib.Path({str(child_pid)!r}).write_text(str(os.getpid())); "
            "time.sleep(60)"
        ),
    ]
    with pytest.raises(DeadlineExceeded):
        run_owned_process(command, timeout_seconds=0.2, log_path=tmp_path / "owned.log")
    pid = int(child_pid.read_text())
    for _ in range(40):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.025)
    else:
        pytest.fail("owned timed-out child was not terminated")


def test_reproduction_result_binds_candidate_commands_outputs_and_environment(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    for name in ("lora_and_classifier.safetensors", "model_config.json", "infer.py"):
        (candidate / name).write_bytes(name.encode())
    float_output = tmp_path / "float.jsonl"
    selected_output = tmp_path / "selected.jsonl"
    float_output.write_text("{}\n")
    selected_output.write_text("{}\n")
    commands = [
        [sys.executable, str(candidate / "infer.py"), "--mode", "verification_float32"],
        [sys.executable, str(candidate / "infer.py"), "--mode", "selected_only"],
    ]
    environment = {
        key: "value"
        for key in (
            "gpu",
            "cuda",
            "pytorch",
            "transformers",
            "peft",
            "safetensors",
            "pillow",
        )
    }

    result = build_reproduction_result(
        candidate_dir=candidate,
        float_output=float_output,
        selected_output=selected_output,
        commands=commands,
        environment=environment,
    )

    assert result["candidate_weights_sha256"] == _sha256(
        candidate / "lora_and_classifier.safetensors"
    )
    assert result["reproduced_float32_sha256"] == _sha256(float_output)
    assert all(str(candidate / "infer.py") in command for command in result["commands"])


def test_resume_accepts_only_a_fully_verified_sealed_package(tmp_path: Path) -> None:
    delivery_dir = tmp_path / "delivery"
    delivery_dir.mkdir()
    artifact = delivery_dir / "artifact.bin"
    artifact.write_bytes(b"sealed")
    (delivery_dir / "SHA256SUMS").write_text(
        f"{_sha256(artifact)}  artifact.bin\n", encoding="utf-8"
    )
    result = verify_sha256s(delivery_dir)
    assert result["verified_files"] == 1
    artifact.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        verify_sha256s(delivery_dir)


def test_candidate_resume_is_bound_to_current_checkpoint_metrics_and_manifests(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    metrics = tmp_path / "evaluation_report.json"
    test_manifest = tmp_path / "test.jsonl"
    metrics.write_text("{}\n")
    test_manifest.write_text("{}\n")
    weights = candidate / "lora_and_classifier.safetensors"
    weights.write_bytes(b"weights")
    verification_dir = candidate / "verification"
    verification_dir.mkdir()
    references = {}
    for name in (
        "verification_32_manifest.jsonl",
        "reference_32_float32.jsonl",
        "reference_32_selected_only.jsonl",
    ):
        path = verification_dir / name
        path.write_text("{}\n")
        references[name] = _sha256(path)
    expected = {
        "checkpoint_sha256": "1" * 64,
        "metrics_sha256": _sha256(metrics),
        "test_manifest_sha256": _sha256(test_manifest),
        "trainable_manifest_sha256": "2" * 64,
        "base_artifact_manifest_sha256": "3" * 64,
    }
    _write_json(
        candidate / "VERIFICATION.json",
        {
            "status": "pending_reproduction",
            "provenance": {**expected, "weights_sha256": _sha256(weights)},
            "references": references,
        },
    )
    validate_candidate_resume(candidate, expected=expected)
    payload = json.loads((candidate / "VERIFICATION.json").read_text())
    payload["provenance"]["checkpoint_sha256"] = "9" * 64
    _write_json(candidate / "VERIFICATION.json", payload)
    with pytest.raises(ValueError, match="candidate.*checkpoint"):
        validate_candidate_resume(candidate, expected=expected)


def test_reproduction_resume_rejects_outputs_from_another_candidate(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    for name in ("lora_and_classifier.safetensors", "model_config.json", "infer.py"):
        (candidate / name).write_bytes(name.encode())
    float_output = tmp_path / "float.jsonl"
    selected_output = tmp_path / "selected.jsonl"
    float_output.write_text("{}\n")
    selected_output.write_text("{}\n")
    result = tmp_path / "reproduction.json"
    _write_json(
        result,
        {
            "candidate_weights_sha256": "f" * 64,
            "candidate_model_config_sha256": _sha256(candidate / "model_config.json"),
            "candidate_infer_sha256": _sha256(candidate / "infer.py"),
            "reproduced_float32_path": str(float_output),
            "reproduced_float32_sha256": _sha256(float_output),
            "reproduced_selected_only_path": str(selected_output),
            "reproduced_selected_only_sha256": _sha256(selected_output),
        },
    )
    assert reproduction_result_is_current(result, candidate_dir=candidate) is False
    payload = json.loads(result.read_text())
    payload["candidate_weights_sha256"] = _sha256(
        candidate / "lora_and_classifier.safetensors"
    )
    _write_json(result, payload)
    assert reproduction_result_is_current(result, candidate_dir=candidate) is True


def test_sealed_resume_is_bound_to_current_candidate_and_reproduction_result(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    weights = candidate / "lora_and_classifier.safetensors"
    weights.write_bytes(b"weights")
    reproduction = tmp_path / "reproduction.json"
    _write_json(reproduction, {"records": 32})
    delivery_dir = tmp_path / "delivery"
    delivery_dir.mkdir()
    _write_json(
        delivery_dir / "VERIFICATION.json",
        {
            "status": "partial",
            "customer_ready": False,
            "provenance": {"weights_sha256": _sha256(weights)},
            "result_sha256": _sha256(reproduction),
        },
    )
    validate_sealed_resume(
        delivery_dir,
        candidate_dir=candidate,
        reproduction_result=reproduction,
    )
    _write_json(reproduction, {"records": 31})
    with pytest.raises(ValueError, match="sealed.*reproduction"):
        validate_sealed_resume(
            delivery_dir,
            candidate_dir=candidate,
            reproduction_result=reproduction,
        )


def test_final_report_binds_data_contract_and_delivery_verdict(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    atomic_write_json(state, {"stage": "seal", "attempt": 3})
    assert json.loads(state.read_text()) == {"stage": "seal", "attempt": 3}
    preflight = {
        "dataset_sha256": {"train": "1" * 64, "val": "2" * 64, "test": "3" * 64},
        "leakage_check_sha256": "4" * 64,
        "leakage_passed": True,
        "expected_trainable_manifest_sha256": "5" * 64,
        "base_artifact_manifest_sha256": "6" * 64,
    }
    report = build_final_report(
        preflight=preflight,
        training={"global_step": 910, "checkpoint_sha256": "7" * 64},
        evaluation={"status": "success", "test": {"complete": True}},
        delivery={"status": "partial", "customer_ready": False},
        started_at_unix=10.0,
        completed_at_unix=20.0,
        deadline_unix=30.0,
    )
    assert report["status"] == "partial"
    assert report["data_contract"]["leakage_passed"] is True
    assert report["data_contract"]["dataset_sha256"]["test"] == "3" * 64
    assert report["customer_ready"] is False
    late = build_final_report(
        preflight=preflight,
        training={"global_step": 910, "checkpoint_sha256": "7" * 64},
        evaluation={"status": "success", "test": {"complete": True}},
        delivery={"status": "success", "customer_ready": True},
        started_at_unix=10.0,
        completed_at_unix=31.0,
        deadline_unix=30.0,
    )
    assert late["status"] == "fail"
    assert late["customer_ready"] is False


def test_supervisor_preflights_while_training_is_still_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SupervisorConfig(
        project_root=tmp_path / "project",
        dataset_root=tmp_path / "dataset",
        run_dir=tmp_path / "run",
        delivery_dir=tmp_path / "delivery",
        base_model=tmp_path / "base",
        python=Path(sys.executable),
        deadline_unix=time.time() + 60,
        poll_seconds=0.01,
        evaluation_slice_seconds=10,
        packaging_reserve_seconds=1,
        reproduction_batch_size=1,
    )
    supervisor = PostTrainSupervisor(config)
    order: list[str] = []
    preflight = {
        "dataset_sha256": {"train": "1" * 64, "val": "2" * 64, "test": "3" * 64},
        "leakage_check_sha256": "4" * 64,
        "leakage_passed": True,
        "expected_trainable_manifest_sha256": "5" * 64,
        "base_artifact_manifest_sha256": "6" * 64,
    }
    training = {"global_step": 910, "checkpoint_sha256": "7" * 64}
    evaluation = {"status": "success", "test": {"complete": True}}
    delivery_result = {"status": "partial", "customer_ready": False}
    monkeypatch.setattr(
        supervisor, "preflight", lambda: order.append("preflight") or preflight
    )
    monkeypatch.setattr(
        supervisor, "wait_for_training", lambda: order.append("training") or training
    )
    monkeypatch.setattr(
        supervisor,
        "evaluate",
        lambda **_kwargs: order.append("evaluation") or evaluation,
    )
    monkeypatch.setattr(
        supervisor, "build_candidate", lambda **_kwargs: order.append("candidate")
    )
    monkeypatch.setattr(
        supervisor,
        "reproduce",
        lambda: order.append("reproduction") or tmp_path / "reproduction.json",
    )
    monkeypatch.setattr(
        supervisor,
        "seal",
        lambda _path: order.append("seal") or delivery_result,
    )
    supervisor.run()
    assert order == [
        "preflight",
        "training",
        "evaluation",
        "candidate",
        "reproduction",
        "seal",
    ]


def test_past_deadline_is_recorded_as_failure_instead_of_escaping_main(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    code = main(
        [
            "--project-root",
            str(tmp_path / "project"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--run-dir",
            str(run),
            "--delivery-dir",
            str(tmp_path / "delivery"),
            "--base-model",
            str(tmp_path / "base"),
            "--python",
            sys.executable,
            "--deadline",
            "2000-01-01T00:00:00+00:00",
        ]
    )
    assert code == 1
    failure = json.loads((run / "posttrain" / "final_report.json").read_text())
    assert failure["status"] == "fail"
    assert failure["error_type"] == "DeadlineExceeded"
