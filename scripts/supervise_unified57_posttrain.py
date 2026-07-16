#!/usr/bin/env python3
"""Durable post-training evaluator and delivery sealer for Unified57.

The supervisor observes the formal training run without controlling it.  After
training has exited successfully, it verifies the frozen data/base/trainable
contracts, resumes the eight-GPU evaluator, builds a pending delivery
candidate, runs the candidate itself in both verification modes, and seals the
package.  Only subprocess groups created by this process are ever signalled.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shlex
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts.package_unified57_delivery import build_base_artifact_provenance
except ModuleNotFoundError:  # direct invocation from scripts/
    from package_unified57_delivery import build_base_artifact_provenance  # type: ignore


PROJECT_DEFAULT = Path("/maas_data/artifacts/bosideng-model-lab/unified57_20260717")
DATASET_DEFAULT = Path("/maas_data/datasets/bosideng/bosideng_unified57_v1_20260717_r4")
RUN_DEFAULT = PROJECT_DEFAULT / "runs" / "unified57_formal_e1_b8_20260717_0136"
DELIVERY_DEFAULT = PROJECT_DEFAULT / "deliveries" / "bosideng_unified57_20260717"
BASE_MODEL_DEFAULT = Path("/maas_data/tagvlm/Qwen3-VL-8B-Instruct")
PYTHON_DEFAULT = Path("/maas_data/tagvlm/venv4train/bin/python")
DEADLINE_DEFAULT = "2026-07-17T08:07:00+08:00"
EXPECTED_STEPS = 910
EXPECTED_DATASET_SHA256 = {
    "train": "c83b746d952aff2492508b4c76d57d64082b35245ee5c09f64df76cb11ef0478",
    "val": "1cbb2aca6c98c32cb1e7666185a5fa5d5a780836cd8c0bcfca712d19d5a42891",
    "test": "cd81ba30c1266afebbef333a69fb4e8ddcb7b12f0ab94c6edfbc34db0039aef8",
    "leakage": "5c4c2c8e6ce082bbf21b9a447887024e95ad2a245e706cd977f6d74533c5ab68",
}


class DeadlineExceeded(RuntimeError):
    """The fixed end-to-end deadline was reached."""


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path | str, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read {name}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def atomic_write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


def verify_sha256s(directory: Path | str) -> dict[str, Any]:
    root = Path(directory)
    checksum_path = root / "SHA256SUMS"
    if not checksum_path.is_file():
        raise ValueError("sealed package SHA256SUMS is missing")
    declared: dict[str, str] = {}
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line:
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"invalid checksum line {line_number}") from exc
        if (
            len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or not relative
            or relative in declared
        ):
            raise ValueError(f"invalid checksum entry at line {line_number}")
        declared[relative] = digest
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != checksum_path
    }
    if set(declared) != actual_files:
        raise ValueError("sealed package checksum inventory differs from package files")
    for relative, expected in declared.items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f"sealed package checksum mismatch: {relative}")
    return {
        "verified_files": len(declared),
        "sha256s_file_sha256": sha256_file(checksum_path),
    }


def _terminate_owned_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=20)


def run_owned_process(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    log_path: Path | str,
    cwd: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    """Run one owned process group and never signal by name, GPU, or user."""

    if timeout_seconds <= 0:
        raise DeadlineExceeded("no time remains for subprocess")
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"\n[{datetime.now().astimezone().isoformat()}] {shlex.join(command)}\n"
        )
        log.flush()
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            env=dict(env) if env is not None else None,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        try:
            return process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _terminate_owned_group(process)
            raise DeadlineExceeded(
                f"subprocess exceeded {timeout_seconds:.1f}s: {shlex.join(command)}"
            ) from exc
        except BaseException:
            _terminate_owned_group(process)
            raise


def _verify_file(path: Path, expected: str, name: str) -> str:
    if not path.is_file():
        raise ValueError(f"missing {name}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{name} SHA256 mismatch: expected {expected}, got {actual}")
    return actual


def build_preflight_contract(
    *,
    dataset_root: Path,
    base_model: Path,
    base_manifest_path: Path,
    expected_trainable_manifest: Path,
    expected_dataset_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Read and verify every contract before starting expensive evaluation."""

    dataset_hashes = {
        split: _verify_file(
            dataset_root / f"{split}.jsonl",
            expected_dataset_sha256[split],
            f"{split} manifest",
        )
        for split in ("train", "val", "test")
    }
    leakage_path = dataset_root / "leakage_check.json"
    leakage_sha = _verify_file(
        leakage_path, expected_dataset_sha256["leakage"], "leakage audit"
    )
    leakage = load_json(leakage_path, name="leakage audit")
    leakage_lists = (
        "cross_split_components",
        "cross_split_exact_phash",
        "cross_split_sha256",
    )
    if leakage.get("passed") is not True or any(
        leakage.get(key) for key in leakage_lists
    ):
        raise ValueError(
            "visual leakage audit did not pass with zero cross-split collisions"
        )

    frozen_base = load_json(base_manifest_path, name="base artifact manifest")
    actual_base = build_base_artifact_provenance(base_model)
    if frozen_base != actual_base:
        raise ValueError("base artifact manifest differs from current base-model bytes")
    canonical_base_sha = actual_base.get("manifest_sha256")
    if not isinstance(canonical_base_sha, str) or len(canonical_base_sha) != 64:
        raise ValueError("base artifact canonical manifest SHA256 is invalid")
    base_config_sha = next(
        (
            item.get("sha256")
            for item in actual_base.get("files", [])
            if item.get("path") == "config.json"
        ),
        None,
    )
    if not isinstance(base_config_sha, str) or len(base_config_sha) != 64:
        raise ValueError("base model config SHA256 is missing from artifact manifest")

    trainable = load_json(
        expected_trainable_manifest, name="expected trainable manifest"
    )
    if trainable.get("version") != "unified57_expected_trainable_v1":
        raise ValueError("expected trainable manifest version mismatch")
    if trainable.get("contract_kind") in {None, "", "synthetic_test"}:
        raise ValueError(
            "expected trainable manifest contract_kind is not production-safe"
        )
    tensors = trainable.get("tensors")
    if not isinstance(tensors, Mapping) or len(tensors) != 290:
        raise ValueError("expected trainable manifest must contain 290 tensors")
    if sum("lora_" in str(name) for name in tensors) != 288:
        raise ValueError("expected trainable manifest must contain 288 LoRA tensors")
    if not {"classifier.weight", "classifier.bias"}.issubset(tensors):
        raise ValueError("expected trainable manifest lacks the Unified57 classifier")
    trainable_sha = sha256_file(expected_trainable_manifest)
    return {
        "dataset_sha256": dataset_hashes,
        "leakage_check_sha256": leakage_sha,
        "leakage_passed": True,
        "leakage_counts": {key: len(leakage[key]) for key in leakage_lists},
        "base_artifact_manifest_sha256": canonical_base_sha,
        "base_artifact_manifest_file_sha256": sha256_file(base_manifest_path),
        "base_model_config_sha256": base_config_sha,
        "expected_trainable_manifest_sha256": trainable_sha,
    }


def validate_training_completion(
    run_dir: Path | str, *, expected_steps: int = EXPECTED_STEPS
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    exit_status = load_json(run_dir / "exit_status.json", name="training exit status")
    if exit_status.get("state") != "exited" or exit_status.get("exit_code") != 0:
        raise ValueError(f"training exit is not successful: {exit_status}")
    summary = load_json(run_dir / "smoke_report.json", name="training summary")
    progress = load_json(run_dir / "progress.json", name="training progress")
    if (
        summary.get("status") != "complete"
        or summary.get("target_reached") is not True
        or int(summary.get("global_step", -1)) != expected_steps
    ):
        raise ValueError("training summary does not prove full-schedule completion")
    if any(
        int(progress.get(field, -1)) != expected_steps
        for field in ("global_step", "global_batch_cursor", "global_batch_count")
    ):
        raise ValueError("training progress does not equal the full schedule")
    if summary.get("loss_finite") is not True:
        raise ValueError("training loss was not finite through completion")
    gradient = summary.get("gradient_audit")
    expected_gradient = {
        "trainable_tensors": 290,
        "with_gradient": 290,
        "finite": 290,
        "lora_tensors": 288,
        "lora_nonzero": 288,
    }
    if not isinstance(gradient, Mapping) or any(
        gradient.get(key) != value for key, value in expected_gradient.items()
    ):
        raise ValueError(
            "training gradient audit is not 290/290 finite and 288/288 LoRA"
        )
    sampling = summary.get("sampling_statistics")
    if not isinstance(sampling, Mapping):
        raise ValueError("training sampling statistics are missing")
    stream = sampling.get("stream_exposures")
    if (
        sampling.get("optimizer_steps") != expected_steps
        or sampling.get("global_sample_exposures") != 58240
        or not isinstance(stream, Mapping)
        or stream.get("uniform") != 43680
        or stream.get("balanced") != 14560
    ):
        raise ValueError(
            "training sampling totals differ from the frozen 910-step schedule"
        )
    balanced = sampling.get("balanced_selection_by_label")
    if (
        not isinstance(balanced, Mapping)
        or len(balanced) != 57
        or balanced.get("假两件") != 0
        or any(value != 260 for tag, value in balanced.items() if tag != "假两件")
    ):
        raise ValueError("training balanced label exposure is not 56x260 plus 假两件=0")
    checkpoint = run_dir / "checkpoints" / "latest.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise ValueError("final training checkpoint is missing or empty")
    return {
        "global_step": expected_steps,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "summary_sha256": sha256_file(run_dir / "smoke_report.json"),
        "progress_sha256": sha256_file(run_dir / "progress.json"),
        "exit_status": exit_status,
        "sampling_statistics": summary.get("sampling_statistics"),
        "loss_finite": summary.get("loss_finite"),
        "gradient_audit": summary.get("gradient_audit"),
    }


def evaluation_is_complete(
    report_path: Path | str,
    *,
    expected_provenance: Mapping[str, str] | None = None,
) -> bool:
    path = Path(report_path)
    if not path.is_file():
        return False
    try:
        report = load_json(path, name="evaluation report")
    except ValueError:
        return False
    test = report.get("test")
    structurally_complete = (
        report.get("status") in {"success", "partial"}
        and isinstance(test, Mapping)
        and test.get("complete") is True
    )
    if not structurally_complete:
        return False
    if expected_provenance is None:
        return True
    provenance = report.get("provenance")
    return isinstance(provenance, Mapping) and all(
        provenance.get(key) == value for key, value in expected_provenance.items()
    )


def evaluation_slice_is_resumable(output_dir: Path | str) -> bool:
    status_path = Path(output_dir) / "status.json"
    if not status_path.is_file():
        return False
    try:
        status = load_json(status_path, name="evaluation status")
    except ValueError:
        return False
    return status.get("state") == "partial" and status.get("test_complete") is False


def evaluation_exit_is_resumable(exit_code: int, output_dir: Path | str) -> bool:
    return exit_code == 75 or evaluation_slice_is_resumable(output_dir)


def _jsonl_has_exact_records(path: Path, count: int) -> bool:
    if not path.is_file():
        return False
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return len(rows) == count and all(isinstance(row, dict) for row in rows)


def build_reproduction_result(
    *,
    candidate_dir: Path,
    float_output: Path,
    selected_output: Path,
    commands: Sequence[Sequence[str]],
    environment: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "candidate_weights_sha256": sha256_file(
            candidate_dir / "lora_and_classifier.safetensors"
        ),
        "candidate_model_config_sha256": sha256_file(
            candidate_dir / "model_config.json"
        ),
        "candidate_infer_sha256": sha256_file(candidate_dir / "infer.py"),
        "reproduced_float32_path": str(float_output),
        "reproduced_float32_sha256": sha256_file(float_output),
        "reproduced_selected_only_path": str(selected_output),
        "reproduced_selected_only_sha256": sha256_file(selected_output),
        "commands": [shlex.join(command) for command in commands],
        "environment": dict(environment),
    }


def validate_candidate_resume(
    candidate_dir: Path | str, *, expected: Mapping[str, str]
) -> dict[str, Any]:
    candidate_dir = Path(candidate_dir)
    pending = load_json(
        candidate_dir / "VERIFICATION.json", name="candidate verification"
    )
    if pending.get("status") != "pending_reproduction":
        raise ValueError("existing candidate has an invalid resume status")
    provenance = pending.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("candidate provenance is missing")
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"candidate {key} differs from the current run")
    weights = candidate_dir / "lora_and_classifier.safetensors"
    if provenance.get("weights_sha256") != sha256_file(weights):
        raise ValueError("candidate weights SHA256 differs from its provenance")
    references = pending.get("references")
    if not isinstance(references, Mapping):
        raise ValueError("candidate verification references are missing")
    for name in (
        "verification_32_manifest.jsonl",
        "reference_32_float32.jsonl",
        "reference_32_selected_only.jsonl",
    ):
        if references.get(name) != sha256_file(candidate_dir / "verification" / name):
            raise ValueError(f"candidate verification reference drifted: {name}")
    return pending


def reproduction_result_is_current(
    result_path: Path | str, *, candidate_dir: Path | str
) -> bool:
    result_path = Path(result_path)
    candidate_dir = Path(candidate_dir)
    if not result_path.is_file():
        return False
    try:
        result = load_json(result_path, name="reproduction result")
        expected = {
            "candidate_weights_sha256": sha256_file(
                candidate_dir / "lora_and_classifier.safetensors"
            ),
            "candidate_model_config_sha256": sha256_file(
                candidate_dir / "model_config.json"
            ),
            "candidate_infer_sha256": sha256_file(candidate_dir / "infer.py"),
        }
        if any(result.get(key) != value for key, value in expected.items()):
            return False
        for path_key, sha_key in (
            ("reproduced_float32_path", "reproduced_float32_sha256"),
            ("reproduced_selected_only_path", "reproduced_selected_only_sha256"),
        ):
            output = Path(str(result.get(path_key) or ""))
            if not output.is_file() or sha256_file(output) != result.get(sha_key):
                return False
    except (OSError, ValueError):
        return False
    return True


def validate_sealed_resume(
    delivery_dir: Path | str,
    *,
    candidate_dir: Path | str,
    reproduction_result: Path | str,
) -> dict[str, Any]:
    delivery_dir = Path(delivery_dir)
    candidate_dir = Path(candidate_dir)
    reproduction_result = Path(reproduction_result)
    sealed = load_json(
        delivery_dir / "VERIFICATION.json", name="sealed delivery verification"
    )
    if sealed.get("status") not in {"success", "partial"}:
        raise ValueError("sealed delivery status is invalid")
    provenance = sealed.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get(
        "weights_sha256"
    ) != sha256_file(candidate_dir / "lora_and_classifier.safetensors"):
        raise ValueError(
            "sealed delivery candidate weights differ from current candidate"
        )
    if sealed.get("result_sha256") != sha256_file(reproduction_result):
        raise ValueError(
            "sealed delivery reproduction result differs from current evidence"
        )
    return sealed


def build_final_report(
    *,
    preflight: Mapping[str, Any],
    training: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    delivery: Mapping[str, Any],
    started_at_unix: float,
    completed_at_unix: float,
    deadline_unix: float,
) -> dict[str, Any]:
    delivery_status = delivery.get("status")
    within_deadline = completed_at_unix <= deadline_unix
    status = (
        delivery_status
        if delivery_status in {"success", "partial"} and within_deadline
        else "fail"
    )
    return {
        "report_version": "bosideng_unified57_posttrain_v1",
        "status": status,
        "customer_ready": bool(delivery.get("customer_ready")) and status == "success",
        "started_at_unix": started_at_unix,
        "completed_at_unix": completed_at_unix,
        "elapsed_seconds": completed_at_unix - started_at_unix,
        "deadline_unix": deadline_unix,
        "completed_before_deadline": within_deadline,
        "data_contract": {
            "dataset_sha256": preflight["dataset_sha256"],
            "leakage_check_sha256": preflight["leakage_check_sha256"],
            "leakage_passed": preflight["leakage_passed"],
            "leakage_counts": preflight.get("leakage_counts"),
            "expected_trainable_manifest_sha256": preflight[
                "expected_trainable_manifest_sha256"
            ],
            "base_artifact_manifest_sha256": preflight["base_artifact_manifest_sha256"],
        },
        "training": dict(training),
        "evaluation": dict(evaluation),
        "delivery": dict(delivery),
    }


def _parse_deadline(value: str) -> float:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid deadline ISO timestamp: {value}") from exc
    if parsed.tzinfo is None:
        raise ValueError("deadline must include an explicit UTC offset")
    return parsed.timestamp()


def _runtime_environment() -> dict[str, str]:
    import torch

    def version(distribution: str) -> str:
        return importlib.metadata.version(distribution)

    return {
        "gpu": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "unavailable",
        "cuda": str(torch.version.cuda or "unavailable"),
        "pytorch": torch.__version__,
        "transformers": version("transformers"),
        "peft": version("peft"),
        "safetensors": version("safetensors"),
        "pillow": version("Pillow"),
    }


@dataclass(frozen=True)
class SupervisorConfig:
    project_root: Path
    dataset_root: Path
    run_dir: Path
    delivery_dir: Path
    base_model: Path
    python: Path
    deadline_unix: float
    poll_seconds: float
    evaluation_slice_seconds: float
    packaging_reserve_seconds: float
    reproduction_batch_size: int

    @property
    def work_dir(self) -> Path:
        return self.run_dir / "posttrain"

    @property
    def evaluation_dir(self) -> Path:
        return self.run_dir / "evaluation"

    @property
    def candidate_dir(self) -> Path:
        return self.run_dir / "delivery_candidate"


class PostTrainSupervisor:
    def __init__(self, config: SupervisorConfig) -> None:
        self.config = config
        self.started_at = time.time()
        self.stage = "initialize"
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        (self.config.work_dir / "logs").mkdir(exist_ok=True)

    def remaining(self) -> float:
        remaining = self.config.deadline_unix - time.time()
        if remaining <= 0:
            raise DeadlineExceeded("Unified57 hard deadline reached")
        return remaining

    def mark(self, stage: str, **evidence: Any) -> None:
        self.stage = stage
        old: dict[str, Any] = {}
        state_path = self.config.work_dir / "state.json"
        if state_path.is_file():
            try:
                old = load_json(state_path, name="supervisor state")
            except ValueError:
                old = {}
        attempts = dict(old.get("attempts") or {})
        attempts[stage] = int(attempts.get(stage, 0)) + 1
        atomic_write_json(
            state_path,
            {
                "state_version": 1,
                "stage": stage,
                "attempts": attempts,
                "updated_at_unix": time.time(),
                "deadline_unix": self.config.deadline_unix,
                **evidence,
            },
        )

    def wait_for_training(self) -> dict[str, Any]:
        self.mark("waiting_for_training")
        status_path = self.config.run_dir / "exit_status.json"
        while True:
            self.remaining()
            if status_path.is_file():
                status = load_json(status_path, name="training exit status")
                state = status.get("state")
                if state in {"exited", "failed", "partial"}:
                    return validate_training_completion(
                        self.config.run_dir, expected_steps=EXPECTED_STEPS
                    )
            time.sleep(min(self.config.poll_seconds, self.remaining()))

    def preflight(self) -> dict[str, Any]:
        self.mark("preflight")
        result = build_preflight_contract(
            dataset_root=self.config.dataset_root,
            base_model=self.config.base_model,
            base_manifest_path=self.config.project_root
            / "configs"
            / "base_artifact_manifest.json",
            expected_trainable_manifest=self.config.project_root
            / "configs"
            / "expected_trainable_manifest.json",
            expected_dataset_sha256=EXPECTED_DATASET_SHA256,
        )
        self.remaining()
        atomic_write_json(self.config.work_dir / "preflight_contract.json", result)
        return result

    def evaluate(
        self, *, preflight: Mapping[str, Any], training: Mapping[str, Any]
    ) -> dict[str, Any]:
        report_path = self.config.evaluation_dir / "evaluation_report.json"
        schema_path = (
            self.config.project_root / "configs" / "bosideng_unified57_schema.json"
        )
        schema = load_json(schema_path, name="Unified57 schema")
        expected_provenance = {
            "checkpoint_sha256": str(training["checkpoint_sha256"]),
            "validation_manifest_sha256": str(preflight["dataset_sha256"]["val"]),
            "test_manifest_sha256": str(preflight["dataset_sha256"]["test"]),
            "schema_sha256": str(schema["schema_sha256"]),
            "schema_file_sha256": sha256_file(schema_path),
            "model_config_sha256": str(preflight["base_model_config_sha256"]),
            "trainable_manifest_sha256": str(
                preflight["expected_trainable_manifest_sha256"]
            ),
            "base_artifact_manifest_sha256": str(
                preflight["base_artifact_manifest_sha256"]
            ),
        }
        attempt = 0
        while not evaluation_is_complete(
            report_path, expected_provenance=expected_provenance
        ):
            if report_path.is_file():
                report = load_json(report_path, name="evaluation report")
                test_complete = (report.get("test") or {}).get("complete") is True
                if (
                    report.get("status") in {"success", "partial"}
                    and test_complete
                    and not evaluation_is_complete(
                        report_path, expected_provenance=expected_provenance
                    )
                ):
                    raise ValueError(
                        "completed evaluation provenance differs from the current run"
                    )
                if report.get("status") == "fail" and test_complete:
                    raise ValueError(
                        "completed evaluation did not meet the partial gate"
                    )
            remaining = self.remaining()
            available = remaining - self.config.packaging_reserve_seconds
            if available <= 60:
                raise DeadlineExceeded(
                    "insufficient reserved time to finish evaluation"
                )
            wall_clock = min(self.config.evaluation_slice_seconds, available - 30)
            attempt += 1
            self.mark(
                "evaluation", evaluation_attempt=attempt, wall_clock_seconds=wall_clock
            )
            env = dict(os.environ)
            env.update(
                {
                    "PROJECT_ROOT": str(self.config.project_root),
                    "DATASET_ROOT": str(self.config.dataset_root),
                    "RUN_DIR": str(self.config.run_dir),
                    "MODEL": str(self.config.base_model),
                    "PYTHON": str(self.config.python),
                    "OUTPUT_DIR": str(self.config.evaluation_dir),
                    "CHECKPOINT": str(training["checkpoint"]),
                    "CHECKPOINT_SHA256": str(training["checkpoint_sha256"]),
                    "EXPECTED_TRAINABLE_MANIFEST_SHA256": str(
                        preflight["expected_trainable_manifest_sha256"]
                    ),
                    "BASE_ARTIFACT_MANIFEST_SHA256": str(
                        preflight["base_artifact_manifest_sha256"]
                    ),
                    "MODEL_CONFIG_SHA256": str(preflight["base_model_config_sha256"]),
                    "WALL_CLOCK_SECONDS": str(max(1, int(wall_clock))),
                }
            )
            # A newly written sidecar proves this attempt reached the evaluator's
            # clean partial exit.  Removing the prior sidecar prevents an OOM,
            # import error, or NCCL crash from masquerading as resumable work.
            (self.config.evaluation_dir / "status.json").unlink(missing_ok=True)
            code = run_owned_process(
                [
                    "/usr/bin/env",
                    "bash",
                    str(
                        self.config.project_root
                        / "scripts"
                        / "launch_unified57_eval_node1.sh"
                    ),
                ],
                timeout_seconds=min(
                    self.remaining() - self.config.packaging_reserve_seconds,
                    wall_clock + 600,
                ),
                log_path=self.config.work_dir / "logs" / "evaluation.log",
                cwd=self.config.project_root,
                env=env,
            )
            if evaluation_is_complete(
                report_path, expected_provenance=expected_provenance
            ):
                break
            if not evaluation_exit_is_resumable(code, self.config.evaluation_dir):
                raise RuntimeError(
                    f"evaluation launcher exited {code} without a complete report"
                )
        return load_json(report_path, name="evaluation report")

    def build_candidate(
        self, *, training: Mapping[str, Any], preflight: Mapping[str, Any]
    ) -> None:
        candidate_verification = self.config.candidate_dir / "VERIFICATION.json"
        schema_path = (
            self.config.project_root / "configs" / "bosideng_unified57_schema.json"
        )
        thresholds_path = self.config.evaluation_dir / "thresholds.json"
        metrics_path = self.config.evaluation_dir / "evaluation_report.json"
        final_prompt_path = self.config.project_root / "configs" / "final_prompt.txt"
        expected_candidate = {
            "checkpoint_sha256": str(training["checkpoint_sha256"]),
            "schema_file_sha256": sha256_file(schema_path),
            "thresholds_sha256": sha256_file(thresholds_path),
            "final_prompt_sha256": sha256_file(final_prompt_path),
            "metrics_sha256": sha256_file(metrics_path),
            "test_manifest_sha256": str(preflight["dataset_sha256"]["test"]),
            "trainable_manifest_sha256": str(
                preflight["expected_trainable_manifest_sha256"]
            ),
            "base_artifact_manifest_sha256": str(
                preflight["base_artifact_manifest_sha256"]
            ),
        }
        if candidate_verification.is_file():
            validate_candidate_resume(
                self.config.candidate_dir,
                expected=expected_candidate,
            )
            return
        if self.config.candidate_dir.exists():
            raise ValueError(
                "candidate directory exists without a resumable verification file"
            )
        self.mark("build_candidate")
        command = [
            str(self.config.python),
            str(self.config.project_root / "scripts" / "package_unified57_delivery.py"),
            "build-candidate",
            "--checkpoint",
            str(training["checkpoint"]),
            "--schema",
            str(schema_path),
            "--thresholds",
            str(thresholds_path),
            "--metrics",
            str(metrics_path),
            "--test-manifest",
            str(self.config.dataset_root / "test.jsonl"),
            "--predictions",
            str(self.config.evaluation_dir / "test_predictions_float32.jsonl"),
            "--verification-dir",
            str(self.config.evaluation_dir / "verification"),
            "--final-prompt",
            str(final_prompt_path),
            "--expected-trainable-manifest",
            str(
                self.config.project_root
                / "configs"
                / "expected_trainable_manifest.json"
            ),
            "--base-model",
            str(self.config.base_model),
            "--candidate-dir",
            str(self.config.candidate_dir),
        ]
        code = run_owned_process(
            command,
            timeout_seconds=self.remaining(),
            log_path=self.config.work_dir / "logs" / "package.log",
            cwd=self.config.project_root,
        )
        if code != 0 or not candidate_verification.is_file():
            raise RuntimeError(f"candidate build failed with exit code {code}")
        validate_candidate_resume(
            self.config.candidate_dir, expected=expected_candidate
        )

    def _run_reproduction_mode(self, mode: str, destination: Path) -> list[str]:
        command = [
            str(self.config.python),
            str(self.config.candidate_dir / "infer.py"),
            "--base-model",
            str(self.config.base_model),
            "--verification-manifest",
            str(
                self.config.candidate_dir
                / "verification"
                / "verification_32_manifest.jsonl"
            ),
            "--mode",
            mode,
            "--batch-size",
            str(self.config.reproduction_batch_size),
            "--device",
            "cuda:0",
            "--output",
            str(destination.with_suffix(destination.suffix + ".tmp")),
        ]
        if _jsonl_has_exact_records(destination, 32):
            return command
        temporary = Path(command[-1])
        temporary.unlink(missing_ok=True)
        self.mark(f"reproduce_{mode}")
        code = run_owned_process(
            command,
            timeout_seconds=self.remaining(),
            log_path=self.config.work_dir / "logs" / "reproduction.log",
            cwd=self.config.candidate_dir,
        )
        if code != 0 or not _jsonl_has_exact_records(temporary, 32):
            raise RuntimeError(f"candidate reproduction mode {mode} failed with {code}")
        os.replace(temporary, destination)
        return command

    def reproduce(self) -> Path:
        float_output = self.config.work_dir / "reproduced_32_float32.jsonl"
        selected_output = self.config.work_dir / "reproduced_32_selected_only.jsonl"
        result_path = self.config.work_dir / "reproduction_result.json"
        if not reproduction_result_is_current(
            result_path, candidate_dir=self.config.candidate_dir
        ):
            for path in (float_output, selected_output, result_path):
                path.unlink(missing_ok=True)
        commands = [
            self._run_reproduction_mode("verification_float32", float_output),
            self._run_reproduction_mode("selected_only", selected_output),
        ]
        result = build_reproduction_result(
            candidate_dir=self.config.candidate_dir,
            float_output=float_output,
            selected_output=selected_output,
            commands=commands,
            environment=_runtime_environment(),
        )
        atomic_write_json(result_path, result)
        return result_path

    def seal(self, reproduction_result: Path) -> dict[str, Any]:
        verification_path = self.config.delivery_dir / "VERIFICATION.json"
        if verification_path.is_file():
            verification = validate_sealed_resume(
                self.config.delivery_dir,
                candidate_dir=self.config.candidate_dir,
                reproduction_result=reproduction_result,
            )
            if not (self.config.delivery_dir / "SHA256SUMS").is_file():
                raise ValueError(
                    "existing delivery is not a valid sealed resume artifact"
                )
            verify_sha256s(self.config.delivery_dir)
            return verification
        if self.config.delivery_dir.exists():
            raise ValueError(
                "delivery directory exists without a resumable sealed package"
            )
        self.mark("seal")
        command = [
            str(self.config.python),
            str(self.config.project_root / "scripts" / "package_unified57_delivery.py"),
            "seal",
            "--candidate-dir",
            str(self.config.candidate_dir),
            "--reproduction-result",
            str(reproduction_result),
            "--output-dir",
            str(self.config.delivery_dir),
        ]
        code = run_owned_process(
            command,
            timeout_seconds=self.remaining(),
            log_path=self.config.work_dir / "logs" / "package.log",
            cwd=self.config.project_root,
        )
        if code != 0 or not verification_path.is_file():
            raise RuntimeError(f"delivery seal failed with exit code {code}")
        verify_sha256s(self.config.delivery_dir)
        return validate_sealed_resume(
            self.config.delivery_dir,
            candidate_dir=self.config.candidate_dir,
            reproduction_result=reproduction_result,
        )

    def run(self) -> dict[str, Any]:
        training: dict[str, Any] = {}
        preflight: dict[str, Any] = {}
        evaluation: dict[str, Any] = {}
        try:
            self.remaining()
            preflight = self.preflight()
            training = self.wait_for_training()
            evaluation = self.evaluate(preflight=preflight, training=training)
            self.build_candidate(training=training, preflight=preflight)
            reproduction_result = self.reproduce()
            delivery = self.seal(reproduction_result)
            self.remaining()
            completed = time.time()
            report = build_final_report(
                preflight=preflight,
                training=training,
                evaluation=evaluation,
                delivery=delivery,
                started_at_unix=self.started_at,
                completed_at_unix=completed,
                deadline_unix=self.config.deadline_unix,
            )
            atomic_write_json(self.config.work_dir / "final_report.json", report)
            self.mark(
                "complete",
                status=report["status"],
                customer_ready=report["customer_ready"],
            )
            return report
        except BaseException as exc:
            failure = {
                "report_version": "bosideng_unified57_posttrain_v1",
                "status": "fail",
                "customer_ready": False,
                "stage": self.stage,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "started_at_unix": self.started_at,
                "failed_at_unix": time.time(),
                "deadline_unix": self.config.deadline_unix,
                "data_contract": preflight or None,
                "training": training or None,
                "evaluation": evaluation or None,
            }
            atomic_write_json(self.config.work_dir / "final_report.json", failure)
            self.mark("failed", failed_stage=self.stage, error=str(exc))
            raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--run-dir", type=Path, default=RUN_DEFAULT)
    parser.add_argument("--delivery-dir", type=Path, default=DELIVERY_DEFAULT)
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL_DEFAULT)
    parser.add_argument("--python", type=Path, default=PYTHON_DEFAULT)
    parser.add_argument("--deadline", default=DEADLINE_DEFAULT)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--evaluation-slice-seconds", type=float, default=2700.0)
    parser.add_argument("--packaging-reserve-seconds", type=float, default=600.0)
    parser.add_argument("--reproduction-batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    if (
        args.poll_seconds <= 0
        or args.evaluation_slice_seconds <= 0
        or args.packaging_reserve_seconds < 0
        or args.reproduction_batch_size <= 0
    ):
        parser.error("poll/evaluation/batch must be positive and reserve non-negative")
    args.deadline_unix = _parse_deadline(args.deadline)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = SupervisorConfig(
        project_root=args.project_root,
        dataset_root=args.dataset_root,
        run_dir=args.run_dir,
        delivery_dir=args.delivery_dir,
        base_model=args.base_model,
        python=args.python,
        deadline_unix=args.deadline_unix,
        poll_seconds=args.poll_seconds,
        evaluation_slice_seconds=args.evaluation_slice_seconds,
        packaging_reserve_seconds=args.packaging_reserve_seconds,
        reproduction_batch_size=args.reproduction_batch_size,
    )
    supervisor = PostTrainSupervisor(config)

    def deadline_handler(_signum: int, _frame: Any) -> None:
        raise DeadlineExceeded("Unified57 hard deadline reached")

    remaining = config.deadline_unix - time.time()
    previous = signal.signal(signal.SIGALRM, deadline_handler)
    if remaining > 0:
        signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        report = supervisor.run()
    except BaseException as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, ensure_ascii=False))
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"success", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
