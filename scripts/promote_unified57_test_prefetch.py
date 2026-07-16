#!/usr/bin/env python3
"""Validate and transactionally publish Unified57 test prefetch shards."""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


CONTRACT_VERSION = "bosideng-unified57-prefetch-inference-contract-v1"
RECEIPT_VERSION = "bosideng-unified57-test-prefetch-import-v1"
SHA256_LENGTH = 64


class PromotionError(RuntimeError):
    """Raised when validation or publication must fail closed."""


@dataclass(frozen=True)
class PromotionConfig:
    source_dir: Path
    evaluation_dir: Path
    evaluator_path: Path
    python_executable: Path
    model_dir: Path
    schema_path: Path
    checkpoint_path: Path
    validation_manifest_path: Path
    test_manifest_path: Path
    expected_trainable_manifest_path: Path
    base_artifact_manifest_path: Path
    trainer_path: Path
    cache_builder_path: Path
    classifier_core_path: Path
    evaluation_core_path: Path
    preflight_contract_path: Path
    image_cache_root: Path
    prefetch_service_unit: str
    formal_service_unit: str
    expected_records: int = 5441
    world_size: int = 8
    batch_size: int = 8
    num_workers: int = 4
    image_max_pixels: int = 112896
    wall_clock_seconds: int = 2400
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    head_dropout: float = 0.1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PromotionError(message)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _jsonl_row_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, name: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise PromotionError(f"{name} is missing: {path}") from error
    _require(stat.S_ISREG(mode) and not path.is_symlink(), f"{name} is not a regular file: {path}")
    return path


def _directory(path: Path, name: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise PromotionError(f"{name} is missing: {path}") from error
    _require(stat.S_ISDIR(mode) and not path.is_symlink(), f"{name} is not a directory: {path}")
    return path


def _load_json(path: Path, name: str) -> dict[str, Any]:
    _regular_file(path, name)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PromotionError(f"{name} is invalid JSON: {path}") from error
    _require(isinstance(value, dict), f"{name} must be a JSON object")
    return value


def _load_jsonl(path: Path, name: str) -> list[dict[str, Any]]:
    _regular_file(path, name)
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                _require(bool(line.strip()), f"{name} contains a blank line at {line_number}")
                value = json.loads(line)
                _require(isinstance(value, dict), f"{name}:{line_number} must be an object")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PromotionError(f"{name} is invalid JSONL: {path}") from error
    return rows


def _vision_prompt(trainer_path: Path) -> str:
    _regular_file(trainer_path, "trainer script")
    try:
        tree = ast.parse(trainer_path.read_text(encoding="utf-8"), filename=str(trainer_path))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise PromotionError("trainer script cannot be parsed") from error
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "VISION_PROMPT" for target in targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError) as error:
            raise PromotionError("training VISION_PROMPT is not a string literal") from error
        _require(isinstance(value, str) and bool(value), "training VISION_PROMPT is empty")
        return value
    raise PromotionError("training VISION_PROMPT was not found")


def _builder_vision_prompt(config: PromotionConfig) -> str:
    try:
        executable_target = config.python_executable.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise PromotionError("frozen Python executable cannot be resolved") from error
    _require(
        executable_target.is_file() and os.access(config.python_executable, os.X_OK),
        f"frozen Python executable is invalid: {config.python_executable}",
    )
    _regular_file(config.cache_builder_path, "evaluation cache builder")
    program = r'''
import importlib.util
import json
import sys

path = sys.argv[1]
spec = importlib.util.spec_from_file_location("_unified57_frozen_cache_builder", path)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load cache builder")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
value = module.training_vision_prompt()
if not isinstance(value, str) or not value:
    raise RuntimeError("cache builder returned an invalid vision prompt")
print(json.dumps(value, ensure_ascii=False))
'''
    try:
        completed = subprocess.run(
            [
                str(config.python_executable),
                "-I",
                "-c",
                program,
                str(config.cache_builder_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise PromotionError("cannot execute cache builder training_vision_prompt") from error
    _require(
        completed.returncode == 0,
        "cache builder training_vision_prompt execution failed",
    )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise PromotionError("cache builder training_vision_prompt output is invalid") from error
    _require(
        isinstance(value, str) and bool(value),
        "cache builder training_vision_prompt output is invalid",
    )
    return value


def _base_contract_sha256(path: Path) -> str:
    payload = _load_json(path, "base artifact manifest")
    value = payload.get("manifest_sha256")
    _require(
        isinstance(value, str)
        and len(value) == SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value),
        "base artifact manifest_sha256 is invalid",
    )
    return value


def _build_base_artifact_provenance(root: Path) -> dict[str, Any]:
    """Rebuild the supervisor's complete external-base byte contract."""

    _directory(root, "base model")
    config = _regular_file(root / "config.json", "base model config")
    candidates: set[Path] = {config}
    indexes = sorted(root.glob("*.safetensors.index.json"))
    if indexes:
        candidates.update(indexes)
        for index_path in indexes:
            index = _load_json(index_path, "base model weight index")
            weight_map = index.get("weight_map")
            _require(
                isinstance(weight_map, Mapping) and bool(weight_map),
                "base model weight index needs a non-empty weight_map",
            )
            for relative in set(weight_map.values()):
                _require(isinstance(relative, str), "base model shard path is invalid")
                candidates.add(_regular_file(root / relative, f"base model shard {relative}"))
    else:
        shards = sorted(root.glob("*.safetensors"))
        _require(bool(shards), "base model weight index or safetensors shard is required")
        candidates.update(_regular_file(path, "base model shard") for path in shards)
    processor_files = sorted(
        path
        for pattern in ("*processor*.json", "preprocessor_config.json")
        for path in root.glob(pattern)
        if path.is_file() and not path.is_symlink()
    )
    tokenizer_files = sorted(
        path
        for pattern in ("tokenizer*", "vocab*", "merges.txt")
        for path in root.glob(pattern)
        if path.is_file() and not path.is_symlink()
    )
    _require(
        bool(processor_files) and bool(tokenizer_files),
        "base model processor and tokenizer artifacts are required",
    )
    candidates.update(processor_files)
    candidates.update(tokenizer_files)
    files: list[dict[str, Any]] = []
    for path in sorted(candidates, key=lambda item: item.relative_to(root).as_posix()):
        _regular_file(path, "base model artifact")
        relative = path.relative_to(root).as_posix()
        role = (
            "config"
            if relative == "config.json"
            else "weights"
            if "safetensors" in relative
            else "processor"
            if "processor" in relative
            else "tokenizer"
        )
        files.append(
            {
                "path": relative,
                "role": role,
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    payload = {"version": "unified57_base_artifacts_v1", "files": files}
    return {**payload, "manifest_sha256": _sha256_bytes(_canonical_json_bytes(payload))}


def _expected_argv(
    config: PromotionConfig,
    *,
    model_config_sha256: str,
    schema_file_sha256: str,
    checkpoint_sha256: str,
    validation_manifest_sha256: str,
    test_manifest_sha256: str,
    trainable_manifest_sha256: str,
    base_contract_sha256: str,
) -> list[str]:
    return [
        str(config.python_executable),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={config.world_size}",
        str(config.evaluator_path),
        "--model",
        str(config.model_dir),
        "--model-config-sha256",
        model_config_sha256,
        "--schema",
        str(config.schema_path),
        "--schema-file-sha256",
        schema_file_sha256,
        "--checkpoint",
        str(config.checkpoint_path),
        "--checkpoint-sha256",
        checkpoint_sha256,
        "--validation-manifest",
        str(config.validation_manifest_path),
        "--validation-manifest-sha256",
        validation_manifest_sha256,
        "--test-manifest",
        str(config.test_manifest_path),
        "--test-manifest-sha256",
        test_manifest_sha256,
        "--expected-trainable-manifest-sha256",
        trainable_manifest_sha256,
        "--base-artifact-manifest-sha256",
        base_contract_sha256,
        "--output-dir",
        str(config.source_dir),
        "--wall-clock-seconds",
        str(config.wall_clock_seconds),
        "--expected-world-size",
        str(config.world_size),
        "--batch-size",
        str(config.batch_size),
        "--num-workers",
        str(config.num_workers),
        "--image-max-pixels",
        str(config.image_max_pixels),
        "--image-cache-root",
        str(config.image_cache_root),
        "--lora-rank",
        str(config.lora_rank),
        "--lora-alpha",
        str(config.lora_alpha),
        "--lora-dropout",
        str(config.lora_dropout),
        "--head-dropout",
        str(config.head_dropout),
        "--prediction-only-split",
        "test",
    ]


def validate_manifest_rows(
    rows: Sequence[Mapping[str, Any]], schema: Mapping[str, Any]
) -> None:
    """Validate the masked PN/PU supervision contract used by the evaluator."""

    labels = schema.get("labels")
    modes = schema.get("label_training_modes")
    _require(isinstance(labels, list) and len(labels) == 57, "schema must contain 57 labels")
    _require(
        isinstance(modes, dict)
        and list(modes) == labels
        and all(mode in {"pn", "pu", "unsupported"} for mode in modes.values()),
        "schema label training modes are invalid",
    )
    for line_number, row in enumerate(rows, 1):
        for field in ("labels", "known_mask", "pu_positive_mask"):
            values = row.get(field)
            _require(
                isinstance(values, list) and len(values) == 57,
                f"manifest line {line_number} {field} must contain 57 values",
            )
            _require(
                all(value in (0, 1, 0.0, 1.0, False, True) for value in values),
                f"manifest line {line_number} {field} must be binary",
            )
        sources = row.get("sources")
        _require(
            isinstance(sources, list)
            and bool(sources)
            and all(isinstance(source, str) and source for source in sources),
            f"manifest line {line_number} sources are invalid",
        )
        for index, tag in enumerate(labels):
            label = float(row["labels"][index])
            known = int(row["known_mask"][index])
            pu_positive = int(row["pu_positive_mask"][index])
            mode = modes[tag]
            _require(
                not (known and pu_positive),
                f"manifest line {line_number} masks overlap at {tag}",
            )
            _require(
                not known or mode == "pn",
                f"manifest line {line_number} known mask is invalid at {tag}",
            )
            _require(
                not pu_positive or (mode == "pu" and label == 1.0),
                f"manifest line {line_number} PU positive mask is invalid at {tag}",
            )
            _require(
                known or pu_positive or label == 0.0,
                f"manifest line {line_number} unknown label is non-neutral at {tag}",
            )


def validate_prefetch(config: PromotionConfig) -> dict[str, Any]:
    """Validate source evidence without writing to either source or destination."""

    _directory(config.source_dir, "prefetch source")
    _directory(config.evaluation_dir, "evaluation directory")
    _directory(config.model_dir, "base model")
    _directory(config.image_cache_root, "image cache")
    for path, name in (
        (config.evaluator_path, "evaluator"),
        (config.schema_path, "schema"),
        (config.checkpoint_path, "checkpoint"),
        (config.validation_manifest_path, "validation manifest"),
        (config.test_manifest_path, "test manifest"),
        (config.expected_trainable_manifest_path, "expected trainable manifest"),
        (config.base_artifact_manifest_path, "base artifact manifest"),
        (config.model_dir / "config.json", "model config"),
        (config.cache_builder_path, "evaluation cache builder"),
        (config.classifier_core_path, "classifier core"),
        (config.evaluation_core_path, "evaluation core"),
        (config.trainer_path, "Unified57 trainer"),
        (config.preflight_contract_path, "preflight contract"),
    ):
        _regular_file(path, name)

    _require(
        config.cache_builder_path.parent.resolve()
        == config.trainer_path.parent.resolve()
        == config.classifier_core_path.parent.resolve()
        == config.evaluation_core_path.parent.resolve()
        == config.evaluator_path.parent.resolve(),
        "evaluator runtime dependencies must share the frozen scripts directory",
    )

    evaluator_sha = _sha256_file(config.evaluator_path)
    model_config_sha = _sha256_file(config.model_dir / "config.json")
    schema_file_sha = _sha256_file(config.schema_path)
    checkpoint_sha = _sha256_file(config.checkpoint_path)
    validation_manifest_sha = _sha256_file(config.validation_manifest_path)
    test_manifest_sha = _sha256_file(config.test_manifest_path)
    trainable_sha = _sha256_file(config.expected_trainable_manifest_path)
    base_file_sha = _sha256_file(config.base_artifact_manifest_path)
    base_contract_sha = _base_contract_sha256(config.base_artifact_manifest_path)
    cache_builder_sha = _sha256_file(config.cache_builder_path)
    classifier_core_sha = _sha256_file(config.classifier_core_path)
    evaluation_core_sha = _sha256_file(config.evaluation_core_path)
    trainer_sha = _sha256_file(config.trainer_path)

    frozen_base = _load_json(config.base_artifact_manifest_path, "base artifact manifest")
    actual_base = _build_base_artifact_provenance(config.model_dir)
    _require(
        frozen_base == actual_base,
        "base artifact manifest differs from current base-model bytes",
    )
    base_contract_sha = str(actual_base["manifest_sha256"])
    base_config_sha = next(
        (
            item.get("sha256")
            for item in actual_base["files"]
            if item.get("path") == "config.json"
        ),
        None,
    )
    _require(base_config_sha == model_config_sha, "base model config provenance mismatch")

    schema = _load_json(config.schema_path, "schema")
    labels = schema.get("labels")
    schema_internal_sha = schema.get("schema_sha256")
    _require(isinstance(labels, list) and len(labels) == 57, "schema must contain 57 labels")
    _require(
        isinstance(schema_internal_sha, str) and len(schema_internal_sha) == SHA256_LENGTH,
        "schema internal SHA256 is invalid",
    )
    trainable_contract = _load_json(
        config.expected_trainable_manifest_path, "expected trainable manifest"
    )
    _require(
        trainable_contract.get("version") == "unified57_expected_trainable_v1",
        "trainable manifest version mismatch",
    )
    _require(
        trainable_contract.get("schema_sha256") == schema_internal_sha,
        "trainable schema contract mismatch",
    )
    _require(
        trainable_contract.get("base_model_config_sha256") == model_config_sha,
        "trainable model config contract mismatch",
    )

    preflight = _load_json(config.preflight_contract_path, "preflight contract")
    preflight_sha = _sha256_file(config.preflight_contract_path)
    expected_preflight = {
        "base_artifact_manifest_sha256": base_contract_sha,
        "base_artifact_manifest_file_sha256": base_file_sha,
        "base_model_config_sha256": model_config_sha,
        "expected_trainable_manifest_sha256": trainable_sha,
    }
    for key, value in expected_preflight.items():
        _require(preflight.get(key) == value, f"preflight contract mismatch: {key}")
    _require(preflight.get("leakage_passed") is True, "preflight leakage check did not pass")
    leakage_counts = preflight.get("leakage_counts")
    _require(
        isinstance(leakage_counts, dict)
        and leakage_counts
        == {
            "cross_split_components": 0,
            "cross_split_exact_phash": 0,
            "cross_split_sha256": 0,
        },
        "preflight leakage counts are non-zero or incomplete",
    )
    preflight_dataset_sha = preflight.get("dataset_sha256")
    _require(
        isinstance(preflight_dataset_sha, dict)
        and preflight_dataset_sha.get("val") == validation_manifest_sha
        and preflight_dataset_sha.get("test") == test_manifest_sha,
        "preflight dataset manifest contract mismatch",
    )

    test_rows = _load_jsonl(config.test_manifest_path, "test manifest")
    validation_rows = _load_jsonl(config.validation_manifest_path, "validation manifest")
    _require(len(test_rows) == config.expected_records, "test manifest record count mismatch")
    expected_ids = [row.get("record_id") for row in test_rows]
    _require(
        all(isinstance(record_id, str) and record_id for record_id in expected_ids),
        "test manifest contains an invalid record_id",
    )
    _require(len(set(expected_ids)) == len(expected_ids), "test manifest contains duplicate record_ids")
    _require(
        all(row.get("schema_sha256") == schema_internal_sha for row in test_rows),
        "test manifest schema contract mismatch",
    )
    validate_manifest_rows(test_rows, schema)

    cache_marker_path = config.image_cache_root / "complete.json"
    cache_marker = _load_json(cache_marker_path, "image cache completion marker")
    cache_complete_sha = _sha256_file(cache_marker_path)
    _require(cache_marker.get("status") == "complete", "image cache is incomplete")
    _require(cache_marker.get("record_count") == len(test_rows) + len(validation_rows), "cache record count mismatch")
    _require(cache_marker.get("split_counts") == {"validation": len(validation_rows), "test": len(test_rows)}, "cache split counts mismatch")
    _require(cache_marker.get("validation_manifest_sha256") == validation_manifest_sha, "cache validation manifest mismatch")
    _require(cache_marker.get("test_manifest_sha256") == test_manifest_sha, "cache test manifest mismatch")
    _require(cache_marker.get("image_max_pixels") == config.image_max_pixels, "cache image_max_pixels mismatch")
    cache_manifest_sha = cache_marker.get("cache_manifest_sha256")
    decoder_contract_sha = cache_marker.get("decoder_contract_sha256")
    for value, name in (
        (cache_manifest_sha, "cache manifest SHA256"),
        (decoder_contract_sha, "decoder contract SHA256"),
    ):
        _require(isinstance(value, str) and len(value) == SHA256_LENGTH, f"{name} is invalid")
    cache_manifest_path = _regular_file(
        config.image_cache_root / "cache_manifest.jsonl", "image cache manifest"
    )
    _require(
        _sha256_file(cache_manifest_path) == cache_manifest_sha,
        "cache manifest file SHA256 mismatch",
    )
    runtime_validation_path = config.image_cache_root / "runtime_validated.json"
    runtime_validation = _load_json(
        runtime_validation_path, "image cache runtime validation"
    )
    runtime_expected = {
        "version": cache_marker.get("version"),
        "status": "validated",
        "record_count": len(test_rows) + len(validation_rows),
        "cache_manifest_sha256": cache_manifest_sha,
        "complete_marker_sha256": cache_complete_sha,
        "image_max_pixels": config.image_max_pixels,
        "decoder_contract_sha256": decoder_contract_sha,
    }
    for key, value in runtime_expected.items():
        _require(
            runtime_validation.get(key) == value,
            f"cache runtime validation mismatch: {key}",
        )
    runtime_validation_sha = _sha256_file(runtime_validation_path)

    expected_argv = _expected_argv(
        config,
        model_config_sha256=model_config_sha,
        schema_file_sha256=schema_file_sha,
        checkpoint_sha256=checkpoint_sha,
        validation_manifest_sha256=validation_manifest_sha,
        test_manifest_sha256=test_manifest_sha,
        trainable_manifest_sha256=trainable_sha,
        base_contract_sha256=base_contract_sha,
    )
    snapshot = _load_json(config.source_dir / "invocation_snapshot.json", "invocation snapshot")
    _require(snapshot.get("version") == 1, "invocation snapshot version mismatch")
    _require(snapshot.get("unit") == config.prefetch_service_unit, "invocation service unit mismatch")
    _require(
        isinstance(snapshot.get("main_pid"), int)
        and not isinstance(snapshot.get("main_pid"), bool)
        and snapshot["main_pid"] > 0,
        "invocation main_pid is invalid",
    )
    _require(
        isinstance(snapshot.get("captured_at_unix"), (int, float))
        and math.isfinite(float(snapshot["captured_at_unix"]))
        and float(snapshot["captured_at_unix"]) > 0,
        "invocation capture time is invalid",
    )
    _require(snapshot.get("argv") == expected_argv, "invocation argv differs from the frozen contract")

    expected_runtime_files = {
        "base_artifact_manifest": {"path": str(config.base_artifact_manifest_path), "sha256": base_file_sha},
        "checkpoint": {"path": str(config.checkpoint_path), "sha256": checkpoint_sha},
        "evaluator": {"path": str(config.evaluator_path), "sha256": evaluator_sha},
        "expected_trainable_manifest": {"path": str(config.expected_trainable_manifest_path), "sha256": trainable_sha},
        "model_config": {"path": str(config.model_dir / "config.json"), "sha256": model_config_sha},
        "schema": {"path": str(config.schema_path), "sha256": schema_file_sha},
        "test_manifest": {"path": str(config.test_manifest_path), "sha256": test_manifest_sha},
        "eval_image_cache_builder": {
            "path": str(config.cache_builder_path),
            "sha256": cache_builder_sha,
        },
        "classifier_core": {
            "path": str(config.classifier_core_path),
            "sha256": classifier_core_sha,
        },
        "unified57_trainer": {
            "path": str(config.trainer_path),
            "sha256": trainer_sha,
        },
        "evaluation_core": {
            "path": str(config.evaluation_core_path),
            "sha256": evaluation_core_sha,
        },
    }
    _require(snapshot.get("runtime_files") == expected_runtime_files, "invocation runtime file inventory mismatch")

    metadata_common = {
        "split": "test",
        "world_size": config.world_size,
        "checkpoint_sha256": checkpoint_sha,
        "manifest_sha256": test_manifest_sha,
        "schema_sha256": schema_internal_sha,
        "image_cache_manifest_sha256": cache_manifest_sha,
        "image_cache_complete_marker_sha256": cache_complete_sha,
        "image_cache_decoder_contract_sha256": decoder_contract_sha,
    }
    shard_dir = _directory(config.source_dir / "prediction_shards", "prediction shard directory")
    expected_names: set[str] = set()
    inventory: dict[str, dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    rank_counts: list[int] = []
    for rank in range(config.world_size):
        shard_name = f"test.rank{rank:02d}-of-{config.world_size:02d}.jsonl"
        sidecar_name = shard_name + ".progress.json"
        expected_names.update((shard_name, sidecar_name))
        shard_path = _regular_file(shard_dir / shard_name, f"rank {rank} shard")
        sidecar_path = _regular_file(shard_dir / sidecar_name, f"rank {rank} sidecar")
        expected_rank_rows = test_rows[rank :: config.world_size]
        expected_rank_ids = [str(row["record_id"]) for row in expected_rank_rows]
        rows = _load_jsonl(shard_path, f"rank {rank} shard")
        rank_counts.append(len(rows))
        _require([row.get("record_id") for row in rows] == expected_rank_ids, f"rank {rank} stride record order mismatch")

        sidecar = _load_json(sidecar_path, f"rank {rank} sidecar")
        expected_metadata = {**metadata_common, "rank": rank}
        _require(sidecar.get("version") == 1, f"rank {rank} sidecar version mismatch")
        _require(sidecar.get("metadata") == expected_metadata, f"rank {rank} metadata mismatch")
        for key, value in expected_metadata.items():
            _require(sidecar.get(key) == value, f"rank {rank} flattened metadata mismatch: {key}")
        _require(sidecar.get("complete") is True, f"rank {rank} is incomplete")
        _require(sidecar.get("durable_records") == len(rows), f"rank {rank} durable count mismatch")
        _require(sidecar.get("next_local_index") == len(rows), f"rank {rank} cursor mismatch")
        _require(sidecar.get("durable_offset") == shard_path.stat().st_size, f"rank {rank} durable offset mismatch")
        expected_last_id = expected_rank_ids[-1] if expected_rank_ids else None
        _require(sidecar.get("last_record_id") == expected_last_id, f"rank {rank} last_record_id mismatch")

        for source_row, row in zip(expected_rank_rows, rows):
            record_id = str(source_row["record_id"])
            _require(record_id not in by_id, f"duplicate prediction record_id: {record_id}")
            for key, value in source_row.items():
                _require(row.get(key) == value, f"{record_id}: manifest field mismatch: {key}")
            _require(row.get("checkpoint_sha256") == checkpoint_sha, f"{record_id}: checkpoint mismatch")
            scores = row.get("scores")
            _require(
                isinstance(scores, list)
                and len(scores) == 57
                and all(
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    and 0.0 <= float(value) <= 1.0
                    for value in scores
                ),
                f"{record_id}: scores are invalid",
            )
            width, height, ratio = row.get("width"), row.get("height"), row.get("aspect_ratio")
            _require(isinstance(width, int) and not isinstance(width, bool) and width > 0, f"{record_id}: width is invalid")
            _require(isinstance(height, int) and not isinstance(height, bool) and height > 0, f"{record_id}: height is invalid")
            _require(
                isinstance(ratio, (int, float))
                and not isinstance(ratio, bool)
                and math.isfinite(float(ratio))
                and math.isclose(float(ratio), width / height, rel_tol=0.0, abs_tol=1e-12),
                f"{record_id}: aspect_ratio is invalid",
            )
            by_id[record_id] = row

        for path in (shard_path, sidecar_path):
            inventory[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}

    actual_names = {
        path.name
        for path in shard_dir.iterdir()
        if path.name.startswith("test.rank")
    }
    _require(actual_names == expected_names, "prediction shard file set mismatch")
    _require(set(by_id) == set(expected_ids), "global prediction record_id set mismatch")
    ordered_predictions = [by_id[str(record_id)] for record_id in expected_ids]
    merged_bytes = b"".join(_jsonl_row_bytes(row) for row in ordered_predictions)
    merged_path = _regular_file(config.source_dir / "test_predictions_float32.jsonl", "merged predictions")
    _require(merged_path.read_bytes() == merged_bytes, "merged prediction bytes/order mismatch")
    merged_predictions_sha = _sha256_bytes(merged_bytes)
    ordered_ids_sha = _sha256_bytes(("\n".join(map(str, expected_ids)) + "\n").encode("utf-8"))

    report = _load_json(config.source_dir / "prediction_only_report.json", "prediction-only report")
    expected_report_values = {
        "mode": "prediction_only",
        "split": "test",
        "state": "complete",
        "expected_records": config.expected_records,
        "predicted_records": config.expected_records,
        "checkpoint_sha256": checkpoint_sha,
        "manifest_sha256": test_manifest_sha,
        "schema_sha256": schema_internal_sha,
        "world_size": config.world_size,
        "batch_size": config.batch_size,
    }
    for key, value in expected_report_values.items():
        _require(report.get(key) == value, f"prediction-only report mismatch: {key}")
    cache_report = report.get("image_cache")
    _require(isinstance(cache_report, dict) and cache_report.get("enabled") is True, "prediction report cache is disabled")
    for key, value in (
        ("cache_manifest_sha256", cache_manifest_sha),
        ("complete_marker_sha256", cache_complete_sha),
        ("decoder_contract_sha256", decoder_contract_sha),
    ):
        _require(cache_report.get(key) == value, f"prediction report cache mismatch: {key}")

    prompt = _vision_prompt(config.trainer_path)
    builder_prompt = _builder_vision_prompt(config)
    _require(
        builder_prompt == prompt,
        "cache builder training_vision_prompt differs from trainer VISION_PROMPT",
    )
    runtime_dependencies = {
        "eval_image_cache_builder": {
            "path": str(config.cache_builder_path),
            "sha256": cache_builder_sha,
        },
        "classifier_core": {
            "path": str(config.classifier_core_path),
            "sha256": classifier_core_sha,
        },
        "unified57_trainer": {
            "path": str(config.trainer_path),
            "sha256": trainer_sha,
        },
        "evaluation_core": {
            "path": str(config.evaluation_core_path),
            "sha256": evaluation_core_sha,
        },
    }
    inference_contract = {
        "version": CONTRACT_VERSION,
        "invocation_argv": expected_argv,
        "invocation_argv_sha256": _sha256_bytes(_canonical_json_bytes(expected_argv)),
        "evaluator": {"path": str(config.evaluator_path), "sha256": evaluator_sha},
        "model": {
            "path": str(config.model_dir),
            "config_path": str(config.model_dir / "config.json"),
            "config_sha256": model_config_sha,
        },
        "schema": {
            "path": str(config.schema_path),
            "file_sha256": schema_file_sha,
            "internal_sha256": schema_internal_sha,
            "labels": 57,
        },
        "checkpoint": {"path": str(config.checkpoint_path), "sha256": checkpoint_sha},
        "expected_trainable_manifest": {
            "path": str(config.expected_trainable_manifest_path),
            "sha256": trainable_sha,
        },
        "base_artifact": {
            "path": str(config.base_artifact_manifest_path),
            "file_sha256": base_file_sha,
            "contract_sha256": base_contract_sha,
            "files": actual_base["files"],
        },
        "preflight_contract": {
            "path": str(config.preflight_contract_path),
            "sha256": preflight_sha,
        },
        "runtime_dependencies": runtime_dependencies,
        "validation_manifest": {"path": str(config.validation_manifest_path), "sha256": validation_manifest_sha},
        "test_manifest": {
            "path": str(config.test_manifest_path),
            "sha256": test_manifest_sha,
            "records": config.expected_records,
            "ordered_record_ids_sha256": ordered_ids_sha,
        },
        "lora": {
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "head_dropout": config.head_dropout,
        },
        "vision_prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
        "image_max_pixels": config.image_max_pixels,
        "image_cache": {
            "root": str(config.image_cache_root),
            "manifest_path": str(cache_manifest_path),
            "manifest_sha256": cache_manifest_sha,
            "complete_marker_sha256": cache_complete_sha,
            "decoder_contract_sha256": decoder_contract_sha,
            "runtime_validation_sha256": runtime_validation_sha,
        },
        "world_size": config.world_size,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "merged_predictions_sha256": merged_predictions_sha,
        "prediction_shard_inventory": inventory,
    }
    inference_contract_sha = _sha256_bytes(_canonical_json_bytes(inference_contract))
    return {
        "status": "pass",
        "records": config.expected_records,
        "rank_counts": rank_counts,
        "ordered_record_ids_sha256": ordered_ids_sha,
        "merged_predictions_sha256": merged_predictions_sha,
        "inventory": inventory,
        "inference_contract": inference_contract,
        "inference_contract_sha256": inference_contract_sha,
    }


def _systemd_properties(systemctl_path: Path, unit: str) -> dict[str, str]:
    try:
        completed = subprocess.run(
            [
                str(systemctl_path),
                "show",
                unit,
                "--property=LoadState,ActiveState,SubState,ControlGroup,MainPID",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise PromotionError(f"cannot inspect formal service {unit}") from error
    _require(completed.returncode == 0, f"cannot inspect formal service {unit}")
    properties: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        properties[key] = value
    required = {"LoadState", "ActiveState", "SubState", "ControlGroup", "MainPID"}
    _require(required <= set(properties), "systemctl service properties are incomplete")
    return properties


def _process_state(status_path: Path) -> str:
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("State:"):
                fields = line.split()
                _require(len(fields) >= 2, f"invalid process status: {status_path}")
                return fields[1]
    except OSError as error:
        raise PromotionError(f"cannot read process status: {status_path}") from error
    raise PromotionError(f"process state missing: {status_path}")


def _scan_global_conflicts(
    config: PromotionConfig, proc_root: Path
) -> dict[str, Any]:
    """Scan all processes, since the evaluator may run outside systemd."""

    _directory(proc_root, "process filesystem")
    test_prefix = str((config.evaluation_dir / "prediction_shards").resolve()) + os.sep + "test.rank"
    evaluation_output = str(config.evaluation_dir.resolve())
    conflicting_commands: list[dict[str, Any]] = []
    open_test_fds: list[dict[str, Any]] = []
    scanned = 0
    try:
        process_dirs = [path for path in proc_root.iterdir() if path.name.isdigit()]
    except OSError as error:
        raise PromotionError("cannot enumerate process filesystem") from error
    for process_dir in process_dirs:
        pid = int(process_dir.name)
        cmdline_path = process_dir / "cmdline"
        try:
            raw_cmdline = cmdline_path.read_bytes()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise PromotionError(f"cannot inspect command line for PID {pid}") from error
        scanned += 1
        argv = [part.decode("utf-8", "surrogateescape") for part in raw_cmdline.split(b"\0") if part]
        evaluator_running = any(Path(arg).name == "evaluate_unified57_multilabel.py" for arg in argv)
        output_matches = any(
            (arg == "--output-dir" and index + 1 < len(argv) and str(Path(argv[index + 1]).resolve()) == evaluation_output)
            or (arg.startswith("--output-dir=") and str(Path(arg.split("=", 1)[1]).resolve()) == evaluation_output)
            for index, arg in enumerate(argv)
        )
        if evaluator_running and output_matches:
            conflicting_commands.append({"pid": pid, "argv": argv})

        fd_dir = process_dir / "fd"
        try:
            descriptors = list(fd_dir.iterdir())
        except FileNotFoundError:
            # The process may have exited between cmdline and fd enumeration.
            continue
        except OSError as error:
            raise PromotionError(f"cannot inspect file descriptors for PID {pid}") from error
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except FileNotFoundError as error:
                # /proc is inherently racy: an FD or its owning process can
                # disappear after enumeration.  Treat only a demonstrably
                # vanished entry as benign; every persistent inspection error
                # remains fail-closed.
                if not process_dir.exists() or not os.path.lexists(descriptor):
                    continue
                raise PromotionError(
                    f"cannot resolve file descriptor {descriptor.name} for PID {pid}"
                ) from error
            except OSError as error:
                raise PromotionError(
                    f"cannot resolve file descriptor {descriptor.name} for PID {pid}"
                ) from error
            if target.removesuffix(" (deleted)").startswith(test_prefix):
                open_test_fds.append({"pid": pid, "fd": descriptor.name, "target": target})
    _require(not conflicting_commands, "conflicting Unified57 evaluator targets formal evaluation output")
    _require(not open_test_fds, "a process has an open formal test shard file descriptor")
    return {
        "scanned_processes": scanned,
        "conflicting_commands": conflicting_commands,
        "open_test_fds": open_test_fds,
    }


def _inspect_quiescence(
    config: PromotionConfig,
    *,
    systemctl_path: Path,
    cgroup_root: Path,
    proc_root: Path,
) -> dict[str, Any]:
    properties = _systemd_properties(systemctl_path, config.formal_service_unit)
    active = properties["ActiveState"]
    load = properties["LoadState"]
    cgroup_value = properties["ControlGroup"]
    if load == "not-found":
        return {
            "mode": "stopped",
            "service": properties,
            "pids": [],
            "global_scan": _scan_global_conflicts(config, proc_root),
        }
    if active in {"inactive", "failed"}:
        residual_pids: list[int] = []
        if cgroup_value:
            _require(
                cgroup_value.startswith("/")
                and ".." not in Path(cgroup_value).parts,
                "formal service cgroup is invalid",
            )
            cgroup_path = cgroup_root.joinpath(cgroup_value.lstrip("/"))
            process_file = cgroup_path / "cgroup.procs"
            if process_file.exists():
                try:
                    residual_pids = [
                        int(value) for value in process_file.read_text().split()
                    ]
                except (OSError, ValueError) as error:
                    raise PromotionError(
                        "cannot inspect stopped formal service cgroup"
                    ) from error
        _require(
            not residual_pids,
            "inactive formal service still has residual processes",
        )
        return {
            "mode": "stopped",
            "service": properties,
            "pids": residual_pids,
            "global_scan": _scan_global_conflicts(config, proc_root),
        }

    _require(active == "active", "formal service is not stopped or fully SIGSTOP")
    _require(cgroup_value.startswith("/") and ".." not in Path(cgroup_value).parts, "formal service cgroup is invalid")
    cgroup_path = cgroup_root.joinpath(cgroup_value.lstrip("/"))
    try:
        pids = [int(value) for value in (cgroup_path / "cgroup.procs").read_text().split()]
    except (OSError, ValueError) as error:
        raise PromotionError("cannot inspect formal service cgroup") from error
    _require(bool(pids), "formal service is not stopped or fully SIGSTOP")
    states = {pid: _process_state(proc_root / str(pid) / "status") for pid in pids}
    _require(all(state in {"T", "t"} for state in states.values()), "formal service is not stopped or fully SIGSTOP")
    test_prefix = str(
        (config.evaluation_dir / "prediction_shards").resolve()
    ) + os.sep + "test.rank"
    open_test_fds: list[dict[str, Any]] = []
    for pid in pids:
        fd_dir = proc_root / str(pid) / "fd"
        try:
            descriptors = list(fd_dir.iterdir())
        except OSError as error:
            raise PromotionError(f"cannot inspect file descriptors for PID {pid}") from error
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except OSError as error:
                raise PromotionError(
                    f"cannot resolve file descriptor {descriptor.name} for PID {pid}"
                ) from error
            normalized = target.removesuffix(" (deleted)")
            if normalized.startswith(test_prefix):
                open_test_fds.append(
                    {"pid": pid, "fd": descriptor.name, "target": target}
                )
    _require(not open_test_fds, "formal service has an open formal test shard file descriptor")
    return {
        "mode": "sigstop",
        "service": properties,
        "pids": pids,
        "states": states,
        "open_test_fds": open_test_fds,
        "global_scan": _scan_global_conflicts(config, proc_root),
    }


def _publish_prefetch_locked(
    config: PromotionConfig,
    *,
    systemctl_path: Path = Path("/usr/bin/systemctl"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    """Publish validated test shards only after the formal service is quiescent."""

    validation = validate_prefetch(config)
    quiescence = _inspect_quiescence(
        config,
        systemctl_path=systemctl_path,
        cgroup_root=cgroup_root,
        proc_root=proc_root,
    )
    target_dir = _directory(
        config.evaluation_dir / "prediction_shards", "formal prediction shard directory"
    )
    source_dir = _directory(
        config.source_dir / "prediction_shards", "prefetch prediction shard directory"
    )
    _require(
        source_dir.resolve() != target_dir.resolve(),
        "prefetch and formal prediction shard directories overlap",
    )

    stage_dir = Path(
        tempfile.mkdtemp(prefix=".test_prefetch_stage.", dir=config.evaluation_dir)
    )
    backup_root = config.evaluation_dir / "test_prefetch_import_backups"
    backup_root.mkdir(mode=0o755, parents=True, exist_ok=True)
    backup_dir = backup_root / (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "." + uuid.uuid4().hex
    )
    backup_dir.mkdir(mode=0o755, exist_ok=False)
    receipt_path = config.evaluation_dir / "test_prefetch_import_contract.json"
    receipt_temp: Path | None = None
    installed: list[str] = []
    original_names: list[str] = []
    previous_receipt = False
    new_receipt_installed = False
    previous_target_inventory: dict[str, dict[str, Any]] = {}

    def assert_quiescent() -> dict[str, Any]:
        return _inspect_quiescence(
            config,
            systemctl_path=systemctl_path,
            cgroup_root=cgroup_root,
            proc_root=proc_root,
        )

    def fsync_file(path: Path) -> None:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())

    def fsync_dir(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    try:
        devices = {
            config.evaluation_dir.stat().st_dev,
            target_dir.stat().st_dev,
            stage_dir.stat().st_dev,
            backup_dir.stat().st_dev,
        }
        _require(len(devices) == 1, "stage, backup, and target must share one filesystem")

        for name, expected in sorted(validation["inventory"].items()):
            source_path = _regular_file(source_dir / name, f"prefetch artifact {name}")
            staged_path = stage_dir / name
            shutil.copy2(source_path, staged_path)
            fsync_file(staged_path)
            _require(staged_path.stat().st_size == expected["bytes"], f"staged size mismatch: {name}")
            _require(_sha256_file(staged_path) == expected["sha256"], f"staged SHA256 mismatch: {name}")
        fsync_dir(stage_dir)

        quiescence = assert_quiescent()

        for path in sorted(target_dir.iterdir()):
            if not path.name.startswith("test.rank"):
                continue
            _require(
                stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(),
                f"formal test artifact is not a regular file: {path}",
            )
            original_names.append(path.name)
            quiescence = assert_quiescent()
            os.replace(path, backup_dir / path.name)
        if receipt_path.exists() or receipt_path.is_symlink():
            _require(
                stat.S_ISREG(receipt_path.lstat().st_mode) and not receipt_path.is_symlink(),
                "existing import receipt is not a regular file",
            )
            quiescence = assert_quiescent()
            os.replace(receipt_path, backup_dir / receipt_path.name)
            previous_receipt = True
        fsync_dir(target_dir)
        fsync_dir(backup_dir)
        previous_target_inventory = {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in sorted(backup_dir.iterdir())
            if path.is_file() and not path.is_symlink()
        }

        for name in sorted(validation["inventory"]):
            quiescence = assert_quiescent()
            os.replace(stage_dir / name, target_dir / name)
            installed.append(name)
        fsync_dir(target_dir)

        for name, expected in validation["inventory"].items():
            installed_path = _regular_file(target_dir / name, f"installed artifact {name}")
            _require(installed_path.stat().st_size == expected["bytes"], f"installed size mismatch: {name}")
            _require(_sha256_file(installed_path) == expected["sha256"], f"installed SHA256 mismatch: {name}")

        receipt = {
            "version": RECEIPT_VERSION,
            "status": "complete",
            "published_at_unix": time.time(),
            "source_dir": str(config.source_dir),
            "evaluation_dir": str(config.evaluation_dir),
            "backup_dir": str(backup_dir),
            "formal_service_unit": config.formal_service_unit,
            "quiescence": quiescence,
            "invocation_snapshot_sha256": _sha256_file(
                config.source_dir / "invocation_snapshot.json"
            ),
            "ordered_record_ids_sha256": validation[
                "ordered_record_ids_sha256"
            ],
            "merged_predictions_sha256": validation["merged_predictions_sha256"],
            "inventory": validation["inventory"],
            "previous_target_inventory": previous_target_inventory,
            "inference_contract": validation["inference_contract"],
            "inference_contract_sha256": validation[
                "inference_contract_sha256"
            ],
        }
        descriptor, temporary_name = tempfile.mkstemp(
            dir=config.evaluation_dir,
            prefix=".test_prefetch_import_contract.",
            suffix=".tmp",
        )
        receipt_temp = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        quiescence = assert_quiescent()
        os.replace(receipt_temp, receipt_path)
        receipt_temp = None
        new_receipt_installed = True
        fsync_dir(config.evaluation_dir)
        quiescence = assert_quiescent()

    except BaseException:
        if receipt_temp is not None:
            receipt_temp.unlink(missing_ok=True)
        if new_receipt_installed and receipt_path.exists():
            receipt_path.unlink()
        for name in installed:
            (target_dir / name).unlink(missing_ok=True)
        for name in original_names:
            saved = backup_dir / name
            if saved.exists():
                os.replace(saved, target_dir / name)
        saved_receipt = backup_dir / receipt_path.name
        if previous_receipt and saved_receipt.exists():
            os.replace(saved_receipt, receipt_path)
        fsync_dir(target_dir)
        fsync_dir(config.evaluation_dir)
        raise
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)

    return {
        **validation,
        "receipt_path": str(receipt_path),
        "backup_dir": str(backup_dir),
        "quiescence": quiescence,
    }


def publish_prefetch(
    config: PromotionConfig,
    *,
    systemctl_path: Path = Path("/usr/bin/systemctl"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    """Serialize promotion and hold the lock through the post-receipt scan."""

    lock_path = config.evaluation_dir / ".test_prefetch_import.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise PromotionError(f"cannot open exclusive promotion lock: {lock_path}") from error
    try:
        _require(stat.S_ISREG(os.fstat(descriptor).st_mode), "promotion lock is not a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise PromotionError("exclusive promotion lock is already held") from error
        return _publish_prefetch_locked(
            config,
            systemctl_path=systemctl_path,
            cgroup_root=cgroup_root,
            proc_root=proc_root,
        )
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--source-dir", type=Path, required=True)
        subparser.add_argument("--evaluation-dir", type=Path, required=True)
        subparser.add_argument("--evaluator", dest="evaluator_path", type=Path, required=True)
        subparser.add_argument("--python-executable", type=Path, required=True)
        subparser.add_argument("--model", dest="model_dir", type=Path, required=True)
        subparser.add_argument("--schema", dest="schema_path", type=Path, required=True)
        subparser.add_argument("--checkpoint", dest="checkpoint_path", type=Path, required=True)
        subparser.add_argument(
            "--validation-manifest",
            dest="validation_manifest_path",
            type=Path,
            required=True,
        )
        subparser.add_argument(
            "--test-manifest", dest="test_manifest_path", type=Path, required=True
        )
        subparser.add_argument(
            "--expected-trainable-manifest",
            dest="expected_trainable_manifest_path",
            type=Path,
            required=True,
        )
        subparser.add_argument(
            "--base-artifact-manifest",
            dest="base_artifact_manifest_path",
            type=Path,
            required=True,
        )
        subparser.add_argument("--trainer", dest="trainer_path", type=Path, required=True)
        subparser.add_argument(
            "--cache-builder", dest="cache_builder_path", type=Path, required=True
        )
        subparser.add_argument(
            "--classifier-core", dest="classifier_core_path", type=Path, required=True
        )
        subparser.add_argument(
            "--evaluation-core", dest="evaluation_core_path", type=Path, required=True
        )
        subparser.add_argument(
            "--preflight-contract",
            dest="preflight_contract_path",
            type=Path,
            required=True,
        )
        subparser.add_argument("--image-cache-root", type=Path, required=True)
        subparser.add_argument("--prefetch-service-unit", required=True)
        subparser.add_argument("--formal-service-unit", required=True)
        subparser.add_argument("--expected-records", type=int, default=5441)
        subparser.add_argument("--world-size", type=int, default=8)
        subparser.add_argument("--batch-size", type=int, default=8)
        subparser.add_argument("--num-workers", type=int, default=4)
        subparser.add_argument("--image-max-pixels", type=int, default=112896)
        subparser.add_argument("--wall-clock-seconds", type=int, default=2400)
        subparser.add_argument("--lora-rank", type=int, default=16)
        subparser.add_argument("--lora-alpha", type=int, default=32)
        subparser.add_argument("--lora-dropout", type=float, default=0.05)
        subparser.add_argument("--head-dropout", type=float, default=0.1)

    validate_parser = subparsers.add_parser(
        "validate", help="perform a read-only validation"
    )
    add_common(validate_parser)
    publish_parser = subparsers.add_parser(
        "publish", help="publish after a fail-closed service quiescence check"
    )
    add_common(publish_parser)
    publish_parser.add_argument(
        "--systemctl-path", type=Path, default=Path("/usr/bin/systemctl")
    )
    publish_parser.add_argument(
        "--cgroup-root", type=Path, default=Path("/sys/fs/cgroup")
    )
    publish_parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    args = parser.parse_args(argv)
    config_fields = {
        field: getattr(args, field)
        for field in PromotionConfig.__dataclass_fields__
    }
    config = PromotionConfig(**config_fields)
    try:
        if args.command == "validate":
            result = validate_prefetch(config)
        else:
            result = publish_prefetch(
                config,
                systemctl_path=args.systemctl_path,
                cgroup_root=args.cgroup_root,
                proc_root=args.proc_root,
            )
    except PromotionError as error:
        print(
            json.dumps(
                {"status": "fail", "error": str(error)},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
