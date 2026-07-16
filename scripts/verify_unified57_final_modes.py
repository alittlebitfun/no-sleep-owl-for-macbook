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
import uuid
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
    return path if path.is_absolute() else posttrain_dir / path


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
    infer_path: Path,
    scores: Sequence[float],
    mode: str,
    scratch_dir: Path,
    row_index: int,
) -> tuple[dict[str, Any], str]:
    input_path = scratch_dir / f"{row_index:02d}.scores.json"
    output_path = scratch_dir / f"{row_index:02d}.{mode}.json"
    input_path.write_text(
        json.dumps({"scores": list(scores)}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
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
    return _load_json(output_path, name=f"{mode} output"), shlex.join(command)


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


def _publish_directory(staging: Path, output_dir: Path) -> None:
    if not output_dir.exists():
        os.replace(staging, output_dir)
        return
    backup = output_dir.parent / f".{output_dir.name}.backup-{uuid.uuid4().hex}"
    os.replace(output_dir, backup)
    try:
        os.replace(staging, output_dir)
    except BaseException:
        os.replace(backup, output_dir)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def verify_final_modes(
    delivery_dir: Path | str,
    posttrain_dir: Path | str,
    output_dir: Path | str,
    *,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Execute and verify both format-only customer modes for 32 replay rows."""

    delivery_dir = Path(delivery_dir).resolve()
    posttrain_dir = Path(posttrain_dir).resolve()
    output_dir = Path(output_dir).resolve()
    python_executable = python_executable or sys.executable
    reproduction_path = posttrain_dir / "reproduction_result.json"
    reproduction = _load_json(reproduction_path, name="reproduction result")
    reproduction_result_sha256 = sha256_file(reproduction_path)
    delivery_verification_path = delivery_dir / "VERIFICATION.json"
    delivery_verification = (
        _load_json(delivery_verification_path, name="delivery verification")
        if delivery_verification_path.is_file()
        else {}
    )
    if delivery_verification.get("status") in {"success", "partial"} or bool(
        delivery_verification.get("customer_ready")
    ):
        _require(
            delivery_verification.get("result_sha256")
            == reproduction_result_sha256,
            "sealed delivery reproduction result SHA256 mismatch",
        )

    infer_path = delivery_dir / "infer.py"
    model_config_path = delivery_dir / "model_config.json"
    weights_path = delivery_dir / "lora_and_classifier.safetensors"
    candidate_infer_sha256 = _validate_sha_binding(
        reproduction, "candidate_infer_sha256", infer_path
    )
    candidate_model_config_sha256 = _validate_sha_binding(
        reproduction, "candidate_model_config_sha256", model_config_path
    )
    candidate_weights_sha256 = _validate_sha_binding(
        reproduction, "candidate_weights_sha256", weights_path
    )

    config = _load_json(model_config_path, name="model config")
    labels = config.get("tag_order")
    _require(
        isinstance(labels, list)
        and len(labels) == EXPECTED_LABELS
        and len(set(labels)) == EXPECTED_LABELS
        and all(isinstance(tag, str) and tag for tag in labels),
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

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    scratch = staging / ".scratch"
    scratch.mkdir()
    commands: list[str] = []
    confidence_rows: list[dict[str, Any]] = []
    all_score_rows: list[dict[str, Any]] = []
    try:
        for row_index, (evidence, scores, selected) in enumerate(validated):
            confidence_output, confidence_command = _run_infer_format(
                python_executable=python_executable,
                infer_path=infer_path,
                scores=scores,
                mode="selected_with_confidence",
                scratch_dir=scratch,
                row_index=row_index,
            )
            all_scores_output, all_scores_command = _run_infer_format(
                python_executable=python_executable,
                infer_path=infer_path,
                scores=scores,
                mode="all_scores",
                scratch_dir=scratch,
                row_index=row_index,
            )
            commands.extend((confidence_command, all_scores_command))
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
            "environment": environment,
            "input_sha256": {
                "reproduction_result.json": reproduction_result_sha256,
                "reproduced_32_float32.jsonl": source_float32_sha256,
                "reproduced_32_selected_only.jsonl": source_selected_only_sha256,
                "infer.py": candidate_infer_sha256,
                "model_config.json": candidate_model_config_sha256,
                "lora_and_classifier.safetensors": candidate_weights_sha256,
            },
            "output_sha256": {
                "reproduced_32_selected_with_confidence.jsonl": confidence_sha256,
                "reproduced_32_all_scores.jsonl": all_scores_sha256,
            },
            "sealed_delivery_verification_status": delivery_verification.get("status"),
            "sealed_delivery_customer_ready": bool(
                delivery_verification.get("customer_ready")
            ),
        }
        _write_json(staging / "final_mode_verification.json", summary)
        _publish_directory(staging, output_dir)
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Unified57 selected-with-confidence and complete all-score "
            "formats from the frozen 32-row reproduction bundle."
        )
    )
    parser.add_argument(
        "--delivery-dir",
        type=Path,
        required=True,
        help="Delivery candidate or sealed delivery directory containing infer.py.",
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
            args.delivery_dir,
            args.posttrain_dir,
            args.output_dir,
            python_executable=args.python_executable,
        )
    except (FinalModeVerificationError, OSError) as error:
        print(f"final-mode verification failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
