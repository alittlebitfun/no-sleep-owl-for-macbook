#!/usr/bin/env python3
"""Replay Unified57 customer formats from frozen float32 reproduction scores.

This verifier deliberately uses the delivered ``infer.py --scores-json`` path.
It performs no model construction and never imports GPU libraries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_RECORDS = 32
EXPECTED_LABELS = 57
EXPECTED_SCORE_VALUES = EXPECTED_RECORDS * EXPECTED_LABELS
UNSUPPORTED_TAG = "假两件"
CATEGORY_ORDER = ("局部结构", "廓形", "工艺", "面辅料")
ENVIRONMENT_FIELDS = (
    "gpu",
    "cuda",
    "pytorch",
    "transformers",
    "peft",
    "safetensors",
    "pillow",
)
PROVENANCE_FIELDS = (
    "checkpoint_sha256",
    "schema_sha256",
    "schema_file_sha256",
    "thresholds_sha256",
    "final_prompt_sha256",
    "weights_sha256",
    "metrics_sha256",
    "test_manifest_sha256",
    "trainable_manifest_sha256",
    "base_artifact_manifest_sha256",
)
TWO_DECIMAL_RE = re.compile(r"(?:0\.\d{2}|1\.00)\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class FinalModeVerificationError(ValueError):
    """Raised when package or reproduction evidence violates the contract."""


def sha256_file(path: Path | str) -> str:
    path = Path(path)
    if not path.is_file():
        raise FinalModeVerificationError(f"required file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalModeVerificationError(message)


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalModeVerificationError(f"invalid {name}: {path}: {error}") from error
    _require(isinstance(value, dict), f"{name} must be a JSON object")
    return value


def _load_jsonl(path: Path, *, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                _require(
                    isinstance(value, dict),
                    f"{name} row {line_number} must be a JSON object",
                )
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalModeVerificationError(f"invalid {name}: {path}: {error}") from error
    return rows


def _resolve_evidence_path(raw: object, posttrain_dir: Path, *, name: str) -> Path:
    _require(isinstance(raw, str) and raw, f"{name} path is missing")
    path = Path(raw)
    return (path if path.is_absolute() else posttrain_dir / path).resolve()


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _snapshot_inventory(root: Path, *, name: str) -> dict[str, Any]:
    _require(root.is_dir(), f"{name} directory is missing: {root}")
    entries = list(root.rglob("*"))
    _require(
        not any(path.is_symlink() for path in entries),
        f"{name} inventory must not contain symlinks",
    )
    files: dict[str, str] = {}
    typed_entries: dict[str, dict[str, str]] = {}
    for path in sorted(entries):
        relative = path.relative_to(root).as_posix()
        if path.is_file():
            digest = sha256_file(path)
            files[relative] = digest
            typed_entries[relative] = {"type": "file", "sha256": digest}
        elif path.is_dir():
            typed_entries[relative] = {"type": "directory"}
        else:
            raise FinalModeVerificationError(
                f"{name} inventory contains an unsupported entry: {relative}"
            )
    return {
        "files": files,
        "file_count": len(files),
        "entries": typed_entries,
        "entry_count": len(typed_entries),
        "manifest_sha256": _canonical_json_sha256(typed_entries),
    }


def _resolve_python_executable(raw: str) -> tuple[str, str]:
    located = shutil.which(raw) if os.sep not in raw else raw
    _require(bool(located), f"python executable is not found: {raw}")
    path = Path(str(located)).resolve()
    _require(path.is_file(), f"python executable is not a regular file: {path}")
    return str(path), sha256_file(path)


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _reject_output_overlap(output_dir: Path, protected: Sequence[Path]) -> None:
    for path in protected:
        resolved = path.resolve()
        _require(
            not _paths_overlap(output_dir, resolved),
            f"output path overlaps protected input: {resolved}",
        )


def _validate_sha_binding(
    payload: Mapping[str, Any], key: str, path: Path
) -> str:
    actual = sha256_file(path)
    _require(
        payload.get(key) == actual,
        f"{key} mismatch for {path}: expected {payload.get(key)!r}, actual {actual}",
    )
    return actual


def _validate_scores(value: object, *, record_id: str) -> list[float]:
    _require(
        isinstance(value, list) and len(value) == EXPECTED_LABELS,
        f"{record_id}: scores must contain exactly 57 values",
    )
    scores: list[float] = []
    for index, item in enumerate(value):
        _require(
            not isinstance(item, bool) and isinstance(item, (int, float)),
            f"{record_id}: score {index} must be numeric",
        )
        score = float(item)
        _require(
            math.isfinite(score) and 0.0 <= score <= 1.0,
            f"{record_id}: score {index} must be finite within [0, 1]",
        )
        scores.append(score)
    return scores


def _validate_evidence(row: Mapping[str, Any], *, name: str) -> tuple[str, str, str]:
    record_id = row.get("record_id")
    image_path = row.get("image_path")
    image_sha256 = row.get("image_sha256")
    _require(isinstance(record_id, str) and record_id, f"{name} record_id is invalid")
    _require(isinstance(image_path, str) and image_path, f"{record_id}: image_path is invalid")
    _require(
        isinstance(image_sha256, str) and SHA256_RE.fullmatch(image_sha256) is not None,
        f"{record_id}: image_sha256 is invalid",
    )
    return record_id, image_path, image_sha256


def _validate_selected_only(
    value: object, labels: Sequence[str], *, record_id: str
) -> dict[str, list[str]]:
    _require(
        isinstance(value, Mapping) and tuple(value) == CATEGORY_ORDER,
        f"{record_id}: selected-only categories are invalid",
    )
    known = set(labels)
    selected: dict[str, list[str]] = {}
    seen: set[str] = set()
    for category in CATEGORY_ORDER:
        items = value[category]
        _require(isinstance(items, list), f"{record_id}: selected-only values must be arrays")
        _require(
            all(isinstance(item, str) and item in known for item in items),
            f"{record_id}: selected-only contains an unknown label",
        )
        _require(
            not seen.intersection(items) and len(items) == len(set(items)),
            f"{record_id}: selected-only contains duplicate labels",
        )
        seen.update(items)
        selected[category] = list(items)
    return selected


def _run_infer_format(
    *,
    python_executable: str,
    python_executable_sha256: str,
    infer_path: Path,
    scores: Sequence[float],
    mode: str,
    scratch_dir: Path,
    row_index: int,
    record_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(
        sha256_file(python_executable) == python_executable_sha256,
        "python executable changed before format invocation",
    )
    input_path = scratch_dir / f"{row_index:02d}.scores.json"
    output_path = scratch_dir / f"{row_index:02d}.{mode}.json"
    input_payload = {"scores": list(scores)}
    input_bytes = (
        json.dumps(input_payload, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    input_path.write_bytes(input_bytes)
    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    roundtrip = json.loads(input_path.read_text(encoding="utf-8"))
    _require(
        roundtrip.get("scores") == list(scores),
        f"row {row_index}: float32 JSON roundtrip differs",
    )
    command = [
        python_executable,
        str(infer_path),
        "--scores-json",
        str(input_path),
        "--mode",
        mode,
        "--output",
        str(output_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise FinalModeVerificationError(
            f"candidate infer.py {mode} execution failed: {error}"
        ) from error
    _require(
        completed.returncode == 0,
        (
            f"candidate infer.py {mode} exited {completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        ),
    )
    _require(
        sha256_file(python_executable) == python_executable_sha256,
        "python executable changed during format invocation",
    )
    output_sha256 = sha256_file(output_path)
    ledger = {
        "record_id": record_id,
        "row_index": row_index,
        "mode": mode,
        "argv": list(command),
        "return_code": completed.returncode,
        "python_executable_sha256": python_executable_sha256,
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
    }
    return _load_json(output_path, name=f"{mode} output"), ledger


def _validate_confidence(
    value: object,
    *,
    record_id: str,
    selected_only: Mapping[str, list[str]],
    labels: Sequence[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    _require(
        isinstance(value, Mapping) and tuple(value) == CATEGORY_ORDER,
        f"{record_id}: selected-with-confidence categories are invalid",
    )
    known = set(labels)
    stripped: dict[str, list[str]] = {}
    values: dict[str, str] = {}
    normalized: dict[str, Any] = {}
    for category in CATEGORY_ORDER:
        items = value[category]
        _require(
            isinstance(items, list),
            f"{record_id}: selected-with-confidence values must be arrays",
        )
        names: list[str] = []
        normalized_items: list[dict[str, str]] = []
        for item in items:
            _require(
                isinstance(item, Mapping) and tuple(item) == ("name", "confidence"),
                f"{record_id}: confidence item must contain name and confidence",
            )
            name = item["name"]
            confidence = item["confidence"]
            _require(
                isinstance(name, str) and name in known,
                f"{record_id}: confidence contains an unknown label",
            )
            _require(
                isinstance(confidence, str)
                and TWO_DECIMAL_RE.fullmatch(confidence) is not None,
                f"{record_id}: confidence must be a two-decimal string",
            )
            _require(name not in values, f"{record_id}: confidence contains duplicate labels")
            names.append(name)
            values[name] = confidence
            normalized_items.append({"name": name, "confidence": confidence})
        stripped[category] = names
        normalized[category] = normalized_items
    _require(
        stripped == dict(selected_only),
        f"{record_id}: selected-with-confidence names differ from selected-only",
    )
    return normalized, values


def _validate_all_scores(
    value: object,
    *,
    record_id: str,
    labels: Sequence[str],
    scores: Sequence[float],
) -> dict[str, Any]:
    _require(
        isinstance(value, Mapping) and tuple(value) == ("scores",),
        f"{record_id}: all_scores wrapper is invalid",
    )
    score_map = value["scores"]
    _require(isinstance(score_map, Mapping), f"{record_id}: all_scores must be an object")
    _require(
        list(score_map) == list(labels),
        f"{record_id}: all_scores keys differ from the 57-label schema order",
    )
    _require(
        len(score_map) == EXPECTED_LABELS,
        f"{record_id}: all_scores must contain exactly 57 values",
    )
    normalized: dict[str, str] = {}
    for tag, score in zip(labels, scores):
        value_at_tag = score_map[tag]
        _require(
            isinstance(value_at_tag, str)
            and TWO_DECIMAL_RE.fullmatch(value_at_tag) is not None,
            f"{record_id}: all_scores values must be two-decimal strings",
        )
        expected = "0.00" if tag == UNSUPPORTED_TAG else f"{score:.2f}"
        if tag == UNSUPPORTED_TAG:
            _require(
                value_at_tag == "0.00",
                f"{record_id}: 假两件 must be fixed to 0.00",
            )
        _require(
            value_at_tag == expected,
            f"{record_id}: all_scores differs from float32 input at {tag}",
        )
        normalized[tag] = value_at_tag
    return {"scores": normalized}


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_publish_tree(root: Path) -> None:
    expected = {
        "final_mode_verification.json",
        "reproduced_32_all_scores.jsonl",
        "reproduced_32_selected_with_confidence.jsonl",
    }
    entries = list(root.iterdir())
    _require(
        {path.name for path in entries} == expected and len(entries) == len(expected),
        "publish tree must contain exactly the three agreed files",
    )
    _require(
        all(path.is_file() and not path.is_symlink() for path in entries),
        "publish tree entries must be regular non-symlink files",
    )


def _publish_directory(staging: Path, output_dir: Path) -> None:
    _require(
        not os.path.lexists(output_dir),
        f"output directory already exists: {output_dir}",
    )
    _validate_publish_tree(staging)
    _fsync_directory(staging)
    os.replace(staging, output_dir)
    _fsync_directory(output_dir.parent)
    _validate_publish_tree(output_dir)


def _validate_sealed_inventory(root: Path) -> dict[str, Any]:
    _require(root.is_dir(), f"sealed delivery directory is missing: {root}")
    checksum_path = root / "SHA256SUMS"
    _require(checksum_path.is_file(), "sealed delivery SHA256SUMS is missing")
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise FinalModeVerificationError(
            f"sealed delivery SHA256SUMS is unreadable: {error}"
        ) from error
    _require(lines, "sealed delivery SHA256SUMS is empty")
    declared: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        parts = line.split("  ", 1)
        _require(
            len(parts) == 2,
            f"sealed SHA256SUMS line {line_number} is malformed",
        )
        digest, relative = parts
        path_parts = relative.split("/")
        _require(
            SHA256_RE.fullmatch(digest) is not None,
            f"sealed SHA256SUMS line {line_number} has an invalid digest",
        )
        _require(
            relative
            and not relative.startswith("/")
            and "\\" not in relative
            and all(part not in {"", ".", ".."} for part in path_parts),
            f"sealed SHA256SUMS line {line_number} has an unsafe path",
        )
        _require(
            relative not in declared,
            f"sealed SHA256SUMS contains a duplicate path: {relative}",
        )
        declared[relative] = digest

    tree_entries = list(root.rglob("*"))
    _require(
        not any(path.is_symlink() for path in tree_entries),
        "sealed delivery inventory must not contain symlinks",
    )
    _require(
        all(path.is_file() or path.is_dir() for path in tree_entries),
        "sealed delivery inventory contains an unsupported special entry",
    )
    for directory in (path for path in tree_entries if path.is_dir()):
        _require(
            any(descendant.is_file() for descendant in directory.rglob("*")),
            (
                "sealed SHA256SUMS inventory cannot authenticate empty directory: "
                f"{directory.relative_to(root).as_posix()}"
            ),
        )
    actual = sorted(
        path.relative_to(root).as_posix()
        for path in tree_entries
        if path.is_file() and path != checksum_path
    )
    _require(
        list(declared) == actual,
        "sealed SHA256SUMS inventory differs from the exact file tree",
    )
    for relative, expected in declared.items():
        actual_digest = sha256_file(root / relative)
        _require(
            actual_digest == expected,
            f"sealed checksum mismatch: {relative}",
        )
    return {
        "verified_files": len(declared),
        "sha256s_sha256": sha256_file(checksum_path),
    }


def _validate_candidate_verification(
    candidate_dir: Path, *, candidate_weights_sha256: str
) -> dict[str, Any]:
    verification = _load_json(
        candidate_dir / "VERIFICATION.json", name="candidate verification"
    )
    _require(
        verification.get("status") == "pending_reproduction",
        "candidate status must be pending_reproduction",
    )
    _require(
        verification.get("evaluation_status") in {"success", "partial"},
        "candidate evaluation_status must be success or partial",
    )
    provenance = verification.get("provenance")
    _require(isinstance(provenance, Mapping), "candidate provenance is missing")
    for field in PROVENANCE_FIELDS:
        value = provenance.get(field)
        _require(
            isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
            f"candidate provenance {field} is invalid",
        )
    _require(
        provenance.get("weights_sha256") == candidate_weights_sha256,
        "candidate provenance weights_sha256 differs from candidate weights",
    )
    return verification


def _validate_candidate_references(
    candidate_dir: Path, candidate_verification: Mapping[str, Any]
) -> dict[str, Any]:
    verification_dir = candidate_dir / "verification"
    paths = {
        "verification_32_manifest.jsonl": (
            verification_dir / "verification_32_manifest.jsonl"
        ).resolve(),
        "reference_32_float32.jsonl": (
            verification_dir / "reference_32_float32.jsonl"
        ).resolve(),
        "reference_32_selected_only.jsonl": (
            verification_dir / "reference_32_selected_only.jsonl"
        ).resolve(),
    }
    references = candidate_verification.get("references")
    _require(isinstance(references, Mapping), "candidate references are missing")
    hashes: dict[str, str] = {}
    for name, path in paths.items():
        digest = sha256_file(path)
        _require(
            references.get(name) == digest,
            f"candidate reference {name} SHA256 mismatch",
        )
        hashes[name] = digest
    manifest = _load_jsonl(paths["verification_32_manifest.jsonl"], name="candidate verification manifest")
    reference_float = _load_jsonl(paths["reference_32_float32.jsonl"], name="candidate float reference")
    reference_selected = _load_jsonl(
        paths["reference_32_selected_only.jsonl"], name="candidate selected reference"
    )
    _require(
        all(
            len(rows) == EXPECTED_RECORDS
            for rows in (manifest, reference_float, reference_selected)
        ),
        "candidate verification references must each contain exactly 32 records",
    )
    ids = [row.get("record_id") for row in manifest]
    _require(
        all(isinstance(record_id, str) and record_id for record_id in ids)
        and len(set(ids)) == EXPECTED_RECORDS,
        "candidate verification manifest must contain 32 unique record_ids",
    )
    for name, rows in (
        ("candidate float reference", reference_float),
        ("candidate selected reference", reference_selected),
    ):
        _require(
            [row.get("record_id") for row in rows] == ids,
            f"{name} record order differs from verification manifest",
        )
    return {
        "paths": paths,
        "hashes": hashes,
        "manifest": manifest,
        "reference_float": reference_float,
        "reference_selected": reference_selected,
        "record_ids": ids,
    }


def _validate_reproduction_commands(
    reproduction: Mapping[str, Any], *, infer_path: Path, manifest_path: Path
) -> list[str]:
    commands = reproduction.get("commands")
    _require(
        isinstance(commands, list)
        and len(commands) >= 2
        and all(isinstance(command, str) and command for command in commands),
        "reproduction commands are missing",
    )
    expected_infer = str(infer_path.resolve())
    expected_manifest = str(manifest_path.resolve())

    def semantically_matches(command: str, mode: str) -> bool:
        try:
            argv = shlex.split(command)
        except ValueError:
            return False
        if len(argv) < 6 or not Path(argv[0]).name.lower().startswith("python"):
            return False
        try:
            interpreter = Path(argv[0]).resolve()
        except OSError:
            return False
        if not interpreter.is_file() or str(Path(argv[1]).resolve()) != expected_infer:
            return False

        def exact_option(name: str, expected: str) -> bool:
            positions = [index for index, token in enumerate(argv) if token == name]
            return (
                len(positions) == 1
                and positions[0] + 1 < len(argv)
                and argv[positions[0] + 1] == expected
            )

        return exact_option("--verification-manifest", expected_manifest) and exact_option(
            "--mode", mode
        )

    _require(
        any(semantically_matches(command, "verification_float32") for command in commands)
        and any(semantically_matches(command, "selected_only") for command in commands),
        "reproduction commands must cover the candidate manifest, float32, and selected-only modes",
    )
    return list(commands)


def _anchor_reproduction_rows(
    *,
    float_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    references: Mapping[str, Any],
    labels: Sequence[str],
) -> None:
    ids = [row.get("record_id") for row in float_rows]
    _require(
        all(isinstance(record_id, str) and record_id for record_id in ids)
        and len(set(ids)) == EXPECTED_RECORDS,
        "reproduction must contain 32 unique ordered record_ids",
    )
    _require(
        ids == references["record_ids"]
        and [row.get("record_id") for row in selected_rows] == ids,
        "reproduction record_id order differs from candidate verification references",
    )
    for manifest, expected_float, expected_selected, actual_float, actual_selected in zip(
        references["manifest"],
        references["reference_float"],
        references["reference_selected"],
        float_rows,
        selected_rows,
    ):
        record_id = str(actual_float["record_id"])
        for field in ("record_id", "image_path", "image_sha256"):
            _require(
                actual_float.get(field) == manifest.get(field),
                f"{record_id}: reproduction differs from candidate manifest at {field}",
            )
            if field in expected_float:
                _require(
                    actual_float.get(field) == expected_float.get(field),
                    f"{record_id}: candidate float reference differs at {field}",
                )
        expected_scores = _validate_scores(
            expected_float.get("scores"), record_id=record_id
        )
        actual_scores = _validate_scores(actual_float.get("scores"), record_id=record_id)
        _require(
            actual_scores == expected_scores,
            f"{record_id}: candidate float reference scores differ from reproduction",
        )
        expected_output = _validate_selected_only(
            expected_selected.get("output"), labels, record_id=record_id
        )
        actual_output = _validate_selected_only(
            actual_selected.get("output"), labels, record_id=record_id
        )
        _require(
            actual_output == expected_output,
            f"{record_id}: candidate selected-only reference differs from reproduction",
        )


def _validate_sealed_delivery(
    sealed_dir: Path,
    *,
    candidate_dir: Path,
    candidate_verification: Mapping[str, Any],
    reproduction: Mapping[str, Any],
    reproduction_result_sha256: str,
    candidate_infer_sha256: str,
    candidate_model_config_sha256: str,
    candidate_weights_sha256: str,
    source_float32_sha256: str,
    source_selected_only_sha256: str,
) -> dict[str, Any]:
    inventory = _validate_sealed_inventory(sealed_dir)
    sealed = _load_json(sealed_dir / "VERIFICATION.json", name="sealed verification")

    for filename, expected in (
        ("infer.py", candidate_infer_sha256),
        ("model_config.json", candidate_model_config_sha256),
        ("lora_and_classifier.safetensors", candidate_weights_sha256),
    ):
        _require(
            sha256_file(sealed_dir / filename) == expected,
            f"sealed {filename} differs from the executed candidate",
        )

    candidate_entries = dict(
        _snapshot_inventory(candidate_dir, name="candidate package")["entries"]
    )
    sealed_entries = dict(
        _snapshot_inventory(sealed_dir, name="sealed package")["entries"]
    )
    candidate_entries.pop("VERIFICATION.json", None)
    sealed_entries.pop("VERIFICATION.json", None)
    sealed_entries.pop("SHA256SUMS", None)
    _require(
        sealed_entries == candidate_entries,
        "sealed package content differs from the executed candidate",
    )

    candidate_provenance = candidate_verification.get("provenance")
    sealed_provenance = sealed.get("provenance")
    _require(
        isinstance(sealed_provenance, Mapping)
        and dict(sealed_provenance) == dict(candidate_provenance),
        "sealed provenance differs from candidate provenance",
    )
    _require(
        sealed_provenance.get("weights_sha256") == candidate_weights_sha256,
        "sealed provenance weights_sha256 differs from candidate weights",
    )
    for field in ("evaluation_status", "references"):
        _require(
            sealed.get(field) == candidate_verification.get(field),
            f"sealed candidate metadata differs at {field}",
        )
    _require(
        sealed.get("result_sha256") == reproduction_result_sha256,
        "sealed delivery reproduction result SHA256 mismatch",
    )

    bound_values = {
        "records": EXPECTED_RECORDS,
        "score_values": EXPECTED_SCORE_VALUES,
        "commands": reproduction.get("commands"),
        "environment": reproduction.get("environment"),
        "result_sha256": reproduction_result_sha256,
        "reproduced_float32_sha256": source_float32_sha256,
        "reproduced_selected_only_sha256": source_selected_only_sha256,
    }
    reproduction_32 = sealed.get("reproduction_32")
    _require(
        isinstance(reproduction_32, Mapping),
        "sealed reproduction_32 binding is missing",
    )
    for field, expected in bound_values.items():
        _require(
            sealed.get(field) == expected,
            f"sealed reproduction binding mismatch at {field}",
        )
        _require(
            reproduction_32.get(field) == expected,
            f"sealed reproduction_32 binding mismatch at {field}",
        )

    derived_booleans = {
        "probabilities_exact": True,
        "selected_outputs_exact": True,
        "image_sha256s_exact": True,
    }
    for field, expected in derived_booleans.items():
        _require(
            sealed.get(field) is expected and reproduction_32.get(field) is expected,
            f"sealed derived reproduction truth mismatch at {field}",
        )
    _require(
        type(sealed.get("selected_mismatch_records")) is int
        and sealed.get("selected_mismatch_records") == 0
        and type(reproduction_32.get("selected_mismatch_records")) is int
        and reproduction_32.get("selected_mismatch_records") == 0,
        "sealed derived reproduction truth mismatch at selected_mismatch_records",
    )
    _require(
        type(sealed.get("max_abs_score_delta")) is float
        and sealed.get("max_abs_score_delta") == 0.0
        and type(reproduction_32.get("max_abs_score_delta")) is float
        and reproduction_32.get("max_abs_score_delta") == 0.0,
        "sealed derived reproduction truth mismatch at max_abs_score_delta",
    )

    expected_status = (
        "success"
        if candidate_verification.get("evaluation_status") == "success"
        else "partial"
    )
    _require(
        sealed.get("status") == expected_status,
        f"sealed status must be {expected_status}",
    )
    _require(
        type(sealed.get("customer_ready")) is bool
        and sealed.get("customer_ready") is (expected_status == "success"),
        "sealed customer_ready is inconsistent with status",
    )
    _require(
        type(sealed.get("internal_use_only")) is bool
        and sealed.get("internal_use_only") is (expected_status != "success"),
        "sealed internal_use_only is inconsistent with status",
    )
    expected_reproduction = {
        **bound_values,
        **derived_booleans,
        "selected_mismatch_records": 0,
        "max_abs_score_delta": 0.0,
    }
    expected_sealed = {
        **candidate_verification,
        **expected_reproduction,
        "status": expected_status,
        "customer_ready": expected_status == "success",
        "internal_use_only": expected_status != "success",
        "reproduction_32": expected_reproduction,
    }
    _require(
        sealed == expected_sealed,
        "sealed verification differs from exact candidate plus reproduction metadata",
    )
    return {
        "status": expected_status,
        "customer_ready": expected_status == "success",
        "verification_sha256": sha256_file(sealed_dir / "VERIFICATION.json"),
        **inventory,
    }


def verify_final_modes(
    candidate_dir: Path | str,
    posttrain_dir: Path | str,
    output_dir: Path | str,
    *,
    sealed_delivery_dir: Path | str | None = None,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Execute and verify both format-only customer modes for 32 replay rows."""

    candidate_dir = Path(candidate_dir).resolve()
    posttrain_dir = Path(posttrain_dir).resolve()
    output_dir = Path(output_dir).resolve()
    sealed_dir = (
        Path(sealed_delivery_dir).resolve()
        if sealed_delivery_dir is not None
        else None
    )
    if sealed_dir is not None:
        _require(
            not _paths_overlap(candidate_dir, sealed_dir),
            "candidate and sealed delivery directories must not overlap",
        )
    python_path, python_sha256 = _resolve_python_executable(
        python_executable or sys.executable
    )
    reproduction_path = posttrain_dir / "reproduction_result.json"
    reproduction = _load_json(reproduction_path, name="reproduction result")
    reproduction_result_sha256 = sha256_file(reproduction_path)
    infer_path = candidate_dir / "infer.py"
    model_config_path = candidate_dir / "model_config.json"
    weights_path = candidate_dir / "lora_and_classifier.safetensors"
    candidate_infer_sha256 = _validate_sha_binding(
        reproduction, "candidate_infer_sha256", infer_path
    )
    candidate_model_config_sha256 = _validate_sha_binding(
        reproduction, "candidate_model_config_sha256", model_config_path
    )
    candidate_weights_sha256 = _validate_sha_binding(
        reproduction, "candidate_weights_sha256", weights_path
    )
    candidate_verification = _validate_candidate_verification(
        candidate_dir, candidate_weights_sha256=candidate_weights_sha256
    )
    candidate_references = _validate_candidate_references(
        candidate_dir, candidate_verification
    )
    original_reproduction_commands = _validate_reproduction_commands(
        reproduction,
        infer_path=infer_path,
        manifest_path=candidate_references["paths"][
            "verification_32_manifest.jsonl"
        ],
    )

    config = _load_json(model_config_path, name="model config")
    labels = config.get("tag_order")
    _require(
        isinstance(labels, list) and len(labels) == EXPECTED_LABELS,
        "model tag_order must contain exactly 57 unique labels",
    )
    _require(
        all(isinstance(tag, str) and tag for tag in labels),
        "model tag_order must contain exactly 57 unique string labels",
    )
    _require(
        len(set(labels)) == EXPECTED_LABELS,
        "model tag_order must contain exactly 57 unique labels",
    )
    _require(UNSUPPORTED_TAG in labels, "model tag_order is missing 假两件")

    environment_source = reproduction.get("environment")
    environment: dict[str, str] = {}
    for field in ENVIRONMENT_FIELDS:
        value = environment_source.get(field) if isinstance(environment_source, Mapping) else None
        _require(
            isinstance(value, str) and value,
            f"reproduction environment must record {field}",
        )
        environment[field] = value

    float_path = _resolve_evidence_path(
        reproduction.get("reproduced_float32_path"),
        posttrain_dir,
        name="reproduced float32",
    )
    selected_path = _resolve_evidence_path(
        reproduction.get("reproduced_selected_only_path"),
        posttrain_dir,
        name="reproduced selected-only",
    )
    source_float32_sha256 = _validate_sha_binding(
        reproduction, "reproduced_float32_sha256", float_path
    )
    source_selected_only_sha256 = _validate_sha_binding(
        reproduction, "reproduced_selected_only_sha256", selected_path
    )
    _reject_output_overlap(
        output_dir,
        [
            candidate_dir,
            reproduction_path,
            float_path,
            selected_path,
            *candidate_references["paths"].values(),
            *([sealed_dir] if sealed_dir is not None else []),
        ],
    )
    _require(
        not os.path.lexists(output_dir),
        f"output directory already exists: {output_dir}",
    )
    sealed_state = (
        _validate_sealed_delivery(
            sealed_dir,
            candidate_dir=candidate_dir,
            candidate_verification=candidate_verification,
            reproduction=reproduction,
            reproduction_result_sha256=reproduction_result_sha256,
            candidate_infer_sha256=candidate_infer_sha256,
            candidate_model_config_sha256=candidate_model_config_sha256,
            candidate_weights_sha256=candidate_weights_sha256,
            source_float32_sha256=source_float32_sha256,
            source_selected_only_sha256=source_selected_only_sha256,
        )
        if sealed_dir is not None
        else None
    )
    float_rows = _load_jsonl(float_path, name="reproduced float32")
    selected_rows = _load_jsonl(selected_path, name="reproduced selected-only")
    _require(
        len(float_rows) == EXPECTED_RECORDS,
        "reproduced float32 must contain exactly 32 records",
    )
    _require(
        len(selected_rows) == EXPECTED_RECORDS,
        "reproduced selected-only must contain exactly 32 records",
    )
    _anchor_reproduction_rows(
        float_rows=float_rows,
        selected_rows=selected_rows,
        references=candidate_references,
        labels=labels,
    )

    validated: list[tuple[dict[str, Any], list[float], dict[str, list[str]]]] = []
    for float_row, selected_row in zip(float_rows, selected_rows):
        record_id, image_path, image_sha256 = _validate_evidence(
            float_row, name="reproduced float32"
        )
        selected_evidence = _validate_evidence(
            selected_row, name="reproduced selected-only"
        )
        _require(
            selected_evidence == (record_id, image_path, image_sha256),
            f"{record_id}: selected-only image evidence differs from float32",
        )
        scores = _validate_scores(float_row.get("scores"), record_id=record_id)
        selected = _validate_selected_only(
            selected_row.get("output"), labels, record_id=record_id
        )
        validated.append(
            (
                {
                    "record_id": record_id,
                    "image_path": image_path,
                    "image_sha256": image_sha256,
                },
                scores,
                selected,
            )
        )

    protected_source_paths = {
        "reproduction_result.json": reproduction_path,
        "reproduced_32_float32.jsonl": float_path,
        "reproduced_32_selected_only.jsonl": selected_path,
        **candidate_references["paths"],
    }
    source_input_sha256_before = {
        name: sha256_file(path) for name, path in protected_source_paths.items()
    }
    _require(
        source_input_sha256_before
        == {
            "reproduction_result.json": reproduction_result_sha256,
            "reproduced_32_float32.jsonl": source_float32_sha256,
            "reproduced_32_selected_only.jsonl": source_selected_only_sha256,
            **candidate_references["hashes"],
        },
        "source inputs changed before format verification",
    )

    candidate_inventory_before = _snapshot_inventory(
        candidate_dir, name="candidate package"
    )
    sealed_inventory_before = (
        _snapshot_inventory(sealed_dir, name="sealed package")
        if sealed_dir is not None
        else None
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    scratch = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.scratch-", dir=output_dir.parent)
    )
    commands: list[str] = []
    execution_ledger: list[dict[str, Any]] = []
    score_payloads: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    all_score_rows: list[dict[str, Any]] = []
    try:
        for row_index, (evidence, scores, selected) in enumerate(validated):
            payload = {"scores": list(scores)}
            payload_bytes = (
                json.dumps(payload, ensure_ascii=False) + "\n"
            ).encode("utf-8")
            score_payloads.append(
                {
                    "record_id": evidence["record_id"],
                    "sha256": hashlib.sha256(payload_bytes).hexdigest(),
                    "payload": payload,
                }
            )
            confidence_output, confidence_ledger = _run_infer_format(
                python_executable=python_path,
                python_executable_sha256=python_sha256,
                infer_path=infer_path,
                scores=scores,
                mode="selected_with_confidence",
                scratch_dir=scratch,
                row_index=row_index,
                record_id=evidence["record_id"],
            )
            all_scores_output, all_scores_ledger = _run_infer_format(
                python_executable=python_path,
                python_executable_sha256=python_sha256,
                infer_path=infer_path,
                scores=scores,
                mode="all_scores",
                scratch_dir=scratch,
                row_index=row_index,
                record_id=evidence["record_id"],
            )
            execution_ledger.extend((confidence_ledger, all_scores_ledger))
            commands.extend(
                (
                    shlex.join(confidence_ledger["argv"]),
                    shlex.join(all_scores_ledger["argv"]),
                )
            )
            confidence, confidence_values = _validate_confidence(
                confidence_output,
                record_id=evidence["record_id"],
                selected_only=selected,
                labels=labels,
            )
            all_scores = _validate_all_scores(
                all_scores_output,
                record_id=evidence["record_id"],
                labels=labels,
                scores=scores,
            )
            for tag, confidence_value in confidence_values.items():
                _require(
                    confidence_value == all_scores["scores"][tag],
                    (
                        f"{evidence['record_id']}: confidence for {tag} differs "
                        "from all_scores"
                    ),
                )
            confidence_rows.append({**evidence, "output": confidence})
            all_score_rows.append({**evidence, "output": all_scores})

        _require(
            sha256_file(python_path) == python_sha256,
            "python executable changed during format verification",
        )
        source_input_sha256_after = {
            name: sha256_file(path) for name, path in protected_source_paths.items()
        }
        _require(
            source_input_sha256_after == source_input_sha256_before,
            "source inputs changed during format verification",
        )
        candidate_inventory_after = _snapshot_inventory(
            candidate_dir, name="candidate package"
        )
        _require(
            candidate_inventory_after == candidate_inventory_before,
            "candidate package changed during format verification",
        )
        sealed_inventory_after = (
            _snapshot_inventory(sealed_dir, name="sealed package")
            if sealed_dir is not None
            else None
        )
        _require(
            sealed_inventory_after == sealed_inventory_before,
            "sealed package changed during format verification",
        )
        shutil.rmtree(scratch)
        confidence_path = staging / "reproduced_32_selected_with_confidence.jsonl"
        all_scores_path = staging / "reproduced_32_all_scores.jsonl"
        _write_jsonl(confidence_path, confidence_rows)
        _write_jsonl(all_scores_path, all_score_rows)
        confidence_sha256 = sha256_file(confidence_path)
        all_scores_sha256 = sha256_file(all_scores_path)
        summary: dict[str, Any] = {
            "status": "success",
            "records": EXPECTED_RECORDS,
            "score_values": EXPECTED_SCORE_VALUES,
            "source_float32_sha256": source_float32_sha256,
            "source_selected_only_sha256": source_selected_only_sha256,
            "float32_json_roundtrip_exact": True,
            "selected_only_reformatted_exact": True,
            "selected_with_confidence_names_exact": True,
            "confidence_two_decimal": True,
            "all_scores_exactly_57": True,
            "all_scores_schema_order_exact": True,
            "all_scores_two_decimal": True,
            "unsupported_假两件_fixed_0.00": True,
            "selected_with_confidence_sha256": confidence_sha256,
            "all_scores_sha256": all_scores_sha256,
            "candidate_infer_sha256": candidate_infer_sha256,
            "candidate_model_config_sha256": candidate_model_config_sha256,
            "candidate_weights_sha256": candidate_weights_sha256,
            "reproduction_result_sha256": reproduction_result_sha256,
            "commands": commands,
            "original_reproduction_commands": original_reproduction_commands,
            "environment": environment,
            "python_executable": {
                "path": python_path,
                "sha256": python_sha256,
            },
            "score_payloads": score_payloads,
            "execution_ledger": execution_ledger,
            "candidate_inventory": candidate_inventory_before,
            "sealed_inventory": sealed_inventory_before,
            "candidate_package_unchanged": True,
            "source_inputs_unchanged": True,
            "sealed_package_unchanged": (
                True if sealed_inventory_before is not None else None
            ),
            "input_sha256": {
                **source_input_sha256_before,
                "infer.py": candidate_infer_sha256,
                "model_config.json": candidate_model_config_sha256,
                "lora_and_classifier.safetensors": candidate_weights_sha256,
                "candidate_VERIFICATION.json": sha256_file(
                    candidate_dir / "VERIFICATION.json"
                ),
                **(
                    {
                        "sealed_SHA256SUMS": sealed_state["sha256s_sha256"],
                        "sealed_VERIFICATION.json": sealed_state[
                            "verification_sha256"
                        ],
                    }
                    if sealed_state is not None
                    else {}
                ),
            },
            "output_sha256": {
                "reproduced_32_selected_with_confidence.jsonl": confidence_sha256,
                "reproduced_32_all_scores.jsonl": all_scores_sha256,
            },
            "sealed_delivery_verification_status": (
                sealed_state["status"] if sealed_state is not None else None
            ),
            "sealed_delivery_customer_ready": (
                sealed_state["customer_ready"] if sealed_state is not None else False
            ),
            "sealed_inventory_verified_files": (
                sealed_state["verified_files"] if sealed_state is not None else 0
            ),
        }
        _write_json(staging / "final_mode_verification.json", summary)
        _publish_directory(staging, output_dir)
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(scratch, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Unified57 selected-with-confidence and complete all-score "
            "formats from the frozen 32-row reproduction bundle."
        )
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        required=True,
        help="Unsealed delivery_candidate directory whose infer.py will execute.",
    )
    parser.add_argument(
        "--sealed-delivery-dir",
        type=Path,
        help="Optional independently sealed delivery directory with SHA256SUMS.",
    )
    parser.add_argument("--posttrain-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python used to invoke the delivered infer.py (default: current Python).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = verify_final_modes(
            args.candidate_dir,
            args.posttrain_dir,
            args.output_dir,
            sealed_delivery_dir=args.sealed_delivery_dir,
            python_executable=args.python_executable,
        )
    except (FinalModeVerificationError, OSError) as error:
        print(f"final-mode verification failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
