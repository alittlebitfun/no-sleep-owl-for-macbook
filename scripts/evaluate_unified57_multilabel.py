#!/usr/bin/env python3
"""Resumable eight-GPU evaluation for the Bosideng Unified57 classifier.

Validation calibrates and freezes thresholds.  The independent test split is
then evaluated exactly once under that frozen checkpoint/schema/data contract.
Every rank owns a deterministic stride shard, so an interrupted run resumes at
its last fsynced local offset without repeating completed image inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import tempfile
import time
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from scripts.unified57_evaluation_core import (
        BufferedPredictionShard,
        calibrate_thresholds,
        evaluate_views,
        final_format_predictions,
        raw_predictions,
        render_all_scores,
        render_selected_only,
        render_selected_with_confidence,
        select_verification_records,
        validate_all_scores,
        validate_schema,
        validate_selected_only,
        validate_selected_with_confidence,
    )
except ModuleNotFoundError:  # direct invocation from scripts/
    from unified57_evaluation_core import (  # type: ignore
        BufferedPredictionShard,
        calibrate_thresholds,
        evaluate_views,
        final_format_predictions,
        raw_predictions,
        render_all_scores,
        render_selected_only,
        render_selected_with_confidence,
        select_verification_records,
        validate_all_scores,
        validate_schema,
        validate_selected_only,
        validate_selected_with_confidence,
    )

try:
    from scripts.build_unified57_eval_image_cache import (
        load_validated_cache_overlay,
        training_vision_prompt,
    )
except ModuleNotFoundError:  # direct invocation from scripts/
    from build_unified57_eval_image_cache import (  # type: ignore
        load_validated_cache_overlay,
        training_vision_prompt,
    )


REPORT_VERSION = "bosideng-unified57-ddp-evaluation-v1"
DEFAULT_IMAGE_MAX_PIXELS = 336 * 336
SHA256_LENGTH = 64


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, name: str) -> str:
    normalized = value.lower()
    if len(normalized) != SHA256_LENGTH or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{name} must be a lowercase 64-character SHA256")
    return normalized


def verify_file_sha256(path: str | Path, expected_sha256: str, name: str) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = _require_sha256(expected_sha256, name)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{name} SHA256 mismatch: expected {expected}, got {actual}")
    return actual


def _atomic_json(payload: Mapping[str, Any], destination: str | Path) -> None:
    destination = Path(destination)
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
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(rows: Iterable[Mapping[str, Any]], destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_schema(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    verify_file_sha256(path, expected_sha256, "schema file")
    schema = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_schema(schema)
    return schema


def load_manifest(
    path: str | Path,
    schema: Mapping[str, Any],
    *,
    expected_sha256: str,
) -> list[dict[str, Any]]:
    verify_file_sha256(path, expected_sha256, "manifest")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"manifest line {line_number} must be an object")
            normalized = dict(raw)
            if normalized.get("schema_sha256") != schema["schema_sha256"]:
                raise ValueError(f"manifest line {line_number} schema_sha256 mismatch")
            for field in ("labels", "known_mask", "pu_positive_mask"):
                values = normalized.get(field)
                if not isinstance(values, list) or len(values) != 57:
                    raise ValueError(f"manifest line {line_number} {field} must have 57 values")
                if any(value not in (0, 1, 0.0, 1.0, False, True) for value in values):
                    raise ValueError(f"manifest line {line_number} {field} must be binary")
            normalized["labels"] = [float(value) for value in normalized["labels"]]
            normalized["known_mask"] = [int(value) for value in normalized["known_mask"]]
            normalized["pu_positive_mask"] = [int(value) for value in normalized["pu_positive_mask"]]
            sources = normalized.get("sources")
            if not isinstance(sources, list) or not sources or not all(
                isinstance(source, str) and source for source in sources
            ):
                raise ValueError(f"manifest line {line_number} sources invalid")
            for index, tag in enumerate(schema["labels"]):
                known = normalized["known_mask"][index]
                pu = normalized["pu_positive_mask"][index]
                label = normalized["labels"][index]
                mode = schema["label_training_modes"][tag]
                if known and pu:
                    raise ValueError(f"manifest line {line_number} masks overlap at {tag}")
                if known and mode != "pn":
                    raise ValueError(f"manifest line {line_number} known mask invalid at {tag}")
                if pu and (mode != "pu" or label != 1.0):
                    raise ValueError(f"manifest line {line_number} PU positive invalid at {tag}")
                if not known and not pu and label != 0.0:
                    raise ValueError(f"manifest line {line_number} unknown label is non-neutral at {tag}")
            record_id = normalized.get("record_id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"manifest line {line_number} requires record_id")
            if record_id in seen:
                raise ValueError(f"manifest contains duplicate record_id {record_id}")
            if normalized.get("schema_version", schema["schema_version"]) != schema["schema_version"]:
                raise ValueError(f"manifest line {line_number} schema_version mismatch")
            seen.add(record_id)
            rows.append(normalized)
    if not rows:
        raise ValueError("manifest is empty")
    return rows


def partition_records(records: Sequence[dict], rank: int, world_size: int) -> list[dict]:
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError("invalid rank/world_size")
    return list(records[rank::world_size])


def merge_prediction_shards(
    expected_records: Sequence[Mapping[str, Any]],
    shard_paths: Sequence[str | Path],
    destination: str | Path,
    *,
    require_complete: bool = True,
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for shard_path in map(Path, shard_paths):
        if not shard_path.is_file():
            continue
        with shard_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                record_id = row.get("record_id")
                if not isinstance(record_id, str) or not record_id:
                    raise RuntimeError(f"{shard_path}:{line_number} lacks record_id")
                if record_id in by_id:
                    raise RuntimeError(f"duplicate distributed prediction {record_id}")
                by_id[record_id] = row
    expected_ids = [str(record["record_id"]) for record in expected_records]
    unexpected = set(by_id).difference(expected_ids)
    if unexpected:
        raise RuntimeError(f"prediction shards contain unexpected ids: {sorted(unexpected)[:3]}")
    missing = [record_id for record_id in expected_ids if record_id not in by_id]
    if require_complete and missing:
        raise RuntimeError(
            f"distributed prediction incomplete: missing {len(missing)}/{len(expected_ids)} records"
        )
    ordered = [by_id[record_id] for record_id in expected_ids if record_id in by_id]
    _atomic_jsonl(ordered, destination)
    return ordered


def freeze_thresholds(
    path: str | Path,
    validation_rows: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    validation_manifest_sha256: str,
) -> dict[str, Any]:
    path = Path(path)
    checkpoint_sha256 = _require_sha256(checkpoint_sha256, "checkpoint")
    validation_manifest_sha256 = _require_sha256(
        validation_manifest_sha256, "validation manifest"
    )
    if path.exists():
        return load_frozen_thresholds(
            path,
            schema,
            checkpoint_sha256=checkpoint_sha256,
            validation_manifest_sha256=validation_manifest_sha256,
        )
    calibrated = calibrate_thresholds(validation_rows, schema)
    calibrated["checkpoint_sha256"] = checkpoint_sha256
    calibrated["validation_manifest_sha256"] = validation_manifest_sha256
    calibrated["calibration_records"] = len(validation_rows)
    calibrated["frozen_at_unix"] = time.time()
    _atomic_json(calibrated, path)
    return calibrated


def load_frozen_thresholds(
    path: str | Path,
    schema: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    validation_manifest_sha256: str,
    expected_threshold_sha256: str | None = None,
) -> dict[str, Any]:
    path = Path(path)
    if expected_threshold_sha256 is not None:
        verify_file_sha256(path, expected_threshold_sha256, "threshold file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("threshold checkpoint contract mismatch")
    if payload.get("validation_manifest_sha256") != validation_manifest_sha256:
        raise ValueError("threshold validation manifest contract mismatch")
    if payload.get("schema_sha256") != schema["schema_sha256"]:
        raise ValueError("threshold schema contract mismatch")
    if list(payload.get("labels", {})) != schema["labels"]:
        raise ValueError("threshold label order mismatch")
    return payload


def _classify(metrics: Mapping[str, Any], json_validity_rate: float) -> dict[str, Any]:
    values = _performance_values(metrics, "final_format", json_validity_rate)
    gates = {
        "known_micro_f1": values["known_micro_f1"] >= 0.88,
        "jd23_micro_f1": values["jd23_micro_f1"] >= 0.88,
        "macro_f1": values["macro_f1"] >= 0.75,
        "dictionary_positive_macro_recall": values["dictionary_positive_macro_recall"] >= 0.85,
        "trusted_negative_specificity": values["trusted_negative_specificity"] >= 0.90,
        "json_validity_rate": values["json_validity_rate"] == 1.0,
    }
    if all(gates.values()):
        verdict = "success"
    elif values["known_micro_f1"] >= 0.82 and gates["json_validity_rate"]:
        verdict = "partial"
    else:
        verdict = "fail"
    return {"verdict": verdict, "values": values, "success_gates": gates}


def _performance_values(
    metrics: Mapping[str, Any], view_name: str, json_validity_rate: float
) -> dict[str, float]:
    view = metrics[view_name]
    overall = view["overall_36pn"]
    return {
        "known_micro_f1": float(overall["micro"]["f1"]),
        "jd23_micro_f1": float(view["jd23_clean"]["micro"]["f1"]),
        "macro_f1": float(overall["macro"]["f1_both_class_labels"]),
        "dictionary_positive_macro_recall": float(
            metrics["dictionary_all_positive"][view_name]["macro_positive_recall"]
        ),
        "trusted_negative_specificity": float(overall["trusted_negatives"]["specificity"]),
        "json_validity_rate": float(json_validity_rate),
    }


def evaluate_dictionary_positive_recall(
    rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    final_format: bool = True,
) -> dict[str, Any]:
    """Macro recall over every dictionary-supported PN and PU tag.

    PN positives are the known positive cells. PU positives come exclusively
    from ``pu_positive_mask``. Unlabeled cells never enter this metric.
    """

    prediction_function = final_format_predictions if final_format else raw_predictions
    predictions = [prediction_function(row["scores"], thresholds, schema) for row in rows]
    dictionary_pairs = [
        (row, prediction)
        for row, prediction in zip(rows, predictions)
        if "dictionary_v4" in row.get("sources", [])
    ]
    per_label: dict[str, Any] = {}
    recalls: list[float] = []
    total_positive = total_hit = 0
    for index, tag in enumerate(schema["labels"]):
        mode = schema["label_training_modes"][tag]
        if mode == "unsupported":
            continue
        if mode == "pn":
            positive_flags = [
                bool(row["known_mask"][index]) and float(row["labels"][index]) == 1.0
                for row, _ in dictionary_pairs
            ]
            supervision = "known_positive"
        else:
            positive_flags = [
                bool(row["pu_positive_mask"][index]) for row, _ in dictionary_pairs
            ]
            supervision = "pu_positive_mask"
        support = sum(positive_flags)
        if not support:
            continue
        hits = sum(
            int(prediction[index])
            for positive, (_, prediction) in zip(positive_flags, dictionary_pairs)
            if positive
        )
        recall = hits / support
        recalls.append(recall)
        total_positive += support
        total_hit += hits
        per_label[tag] = {
            "mode": mode,
            "supervision": supervision,
            "positive_support": support,
            "selected_positive": hits,
            "recall": recall,
        }
    return {
        "dictionary_records": len(dictionary_pairs),
        "labels_with_positive_support": len(recalls),
        "macro_positive_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "micro_positive_recall": total_hit / total_positive if total_positive else 0.0,
        "positive_support": total_positive,
        "selected_positive": total_hit,
        "per_label": per_label,
    }


def write_delivery_outputs(
    rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    schema: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_views(rows, thresholds, schema)
    metrics["dictionary_all_positive"] = {
        "raw_thresholded": evaluate_dictionary_positive_recall(
            rows, thresholds, schema, final_format=False
        ),
        "final_format": evaluate_dictionary_positive_recall(
            rows, thresholds, schema, final_format=True
        ),
    }
    threshold_items = thresholds.get("labels", {})
    pu_fallback = [
        tag
        for tag in schema["labels"]
        if schema["label_training_modes"][tag] == "pu"
        and threshold_items.get(tag, {}).get("status") == "fallback_insufficient_support"
    ]
    metrics["threshold_calibration"] = {
        "pu_labels": 20,
        "pu_fallback_count": len(pu_fallback),
        "pu_fallback_labels": pu_fallback,
        "pu_metric_semantics": "positive recall, positive-vs-unlabeled concordance and coverage only",
    }
    all_scores: list[dict[str, Any]] = []
    selected_only: list[dict[str, Any]] = []
    selected_confidence: list[dict[str, Any]] = []
    valid = 0
    for row in rows:
        score_payload = render_all_scores(row, schema)
        selected = render_selected_only(row["scores"], thresholds, schema)
        selected_with_confidence = render_selected_with_confidence(
            row["scores"], thresholds, schema
        )
        validate_all_scores(score_payload, schema)
        validate_selected_only(selected, schema)
        validate_selected_with_confidence(selected_with_confidence, schema)
        valid += 1
        all_scores.append(score_payload)
        selected_only.append({"record_id": row["record_id"], "output": selected})
        selected_confidence.append(
            {"record_id": row["record_id"], "output": selected_with_confidence}
        )
    _atomic_jsonl(all_scores, output_dir / "test_all_scores.jsonl")
    _atomic_jsonl(selected_only, output_dir / "test_selected_only.jsonl")
    _atomic_jsonl(selected_confidence, output_dir / "test_selected_with_confidence.jsonl")

    manifest_indices = {str(row["record_id"]): index for index, row in enumerate(rows)}
    verification_rows = select_verification_records(rows, count=32, seed=20260717)
    verification_dir = output_dir / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)
    def source_name(row: Mapping[str, Any]) -> str:
        return str(row.get("source") or "+".join(row.get("sources", [])))

    def selection_bucket(row: Mapping[str, Any]) -> str:
        sources = set(row.get("sources", []))
        source_bucket = (
            "mixed" if len(sources) > 1 else "dictionary" if "dictionary_v4" in sources else "jd"
        )
        ratio = row.get("aspect_ratio")
        if ratio is None and row.get("width") and row.get("height"):
            ratio = float(row["width"]) / float(row["height"])
        aspect = "unknown" if ratio is None else "portrait" if float(ratio) < 0.8 else "landscape" if float(ratio) > 1.25 else "square"
        return f"{source_bucket}:{aspect}"

    verification_manifest = [
        {
            "record_id": row["record_id"],
            "test_manifest_index": manifest_indices[str(row["record_id"])],
            "image_path": row.get("image_path"),
            "image_sha256": row.get("image_sha256"),
            "source": source_name(row),
            "sources": list(row.get("sources", [])),
            "selection_bucket": selection_bucket(row),
        }
        for row in verification_rows
    ]
    float_fields = (
        "record_id", "image_path", "image_sha256", "source", "sources", "width", "height",
        "aspect_ratio", "labels", "known_mask", "pu_positive_mask", "schema_version",
        "schema_sha256", "checkpoint_sha256",
    )
    reference_float = []
    for row in verification_rows:
        item = {field: row.get(field) for field in float_fields}
        item["source"] = source_name(row)
        item["sources"] = list(row.get("sources", []))
        item["scores"] = [float(value) for value in row["scores"]]
        reference_float.append(item)
    reference_selected = [
        {
            "record_id": row["record_id"],
            "output": render_selected_only(row["scores"], thresholds, schema),
        }
        for row in verification_rows
    ]
    _atomic_jsonl(verification_manifest, verification_dir / "verification_32_manifest.jsonl")
    _atomic_jsonl(reference_float, verification_dir / "reference_32_float32.jsonl")
    _atomic_jsonl(reference_selected, verification_dir / "reference_32_selected_only.jsonl")

    # The core selector can satisfy several mandatory strata before filling;
    # cap explicitly because the delivery contract is exactly six records.
    representatives = select_verification_records(rows, count=6, seed=20260718)[:6]
    representative_dir = output_dir / "representative6"
    image_dir = representative_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    representative_manifest: list[dict[str, Any]] = []
    for index, row in enumerate(representatives, 1):
        binary = final_format_predictions(row["scores"], thresholds, schema)
        pn_positive = [
            tag for label_index, tag in enumerate(schema["labels"])
            if schema["label_training_modes"][tag] == "pn"
            and bool(row["known_mask"][label_index]) and float(row["labels"][label_index]) == 1.0
        ]
        pn_false_negative = [
            tag for label_index, tag in enumerate(schema["labels"])
            if schema["label_training_modes"][tag] == "pn"
            and bool(row["known_mask"][label_index]) and float(row["labels"][label_index]) == 1.0
            and not binary[label_index]
        ]
        pn_false_positive = [
            tag for label_index, tag in enumerate(schema["labels"])
            if schema["label_training_modes"][tag] == "pn"
            and bool(row["known_mask"][label_index]) and float(row["labels"][label_index]) == 0.0
            and binary[label_index]
        ]
        pu_positive = [
            tag for label_index, tag in enumerate(schema["labels"])
            if bool(row["pu_positive_mask"][label_index])
        ]
        pu_hits = [tag for tag in pu_positive if binary[schema["labels"].index(tag)]]
        entry = {
            "record_id": row["record_id"],
            "image_path": row.get("image_path"),
            "image_sha256": row.get("image_sha256"),
            "sources": row.get("sources"),
            "selected": render_selected_only(row["scores"], thresholds, schema),
            "all_scores": render_all_scores(row, schema)["scores"],
            "truth_summary": {
                "pn_known_positive": pn_positive,
                "pn_known_negative_count": sum(
                    bool(known) and float(label) == 0.0
                    for known, label in zip(row["known_mask"], row["labels"])
                ),
                "pu_positive": pu_positive,
            },
            "pn_errors": {
                "false_negative": pn_false_negative,
                "false_positive": pn_false_positive,
            },
            "pu_positive_hits": {
                "hit": pu_hits,
                "missed": [tag for tag in pu_positive if tag not in pu_hits],
                "note": "PU unlabeled cells are excluded from accuracy/precision/F1",
            },
        }
        source_path = Path(str(row.get("image_path", "")))
        if source_path.is_file():
            suffix = source_path.suffix.lower() or ".jpg"
            stable_name = str(row.get("image_sha256") or row["record_id"])
            stable_name = "".join(char for char in stable_name if char.isalnum())[:24] or "image"
            copy_path = image_dir / f"{index:02d}_{stable_name}{suffix}"
            shutil.copy2(source_path, copy_path)
            entry["copied_image"] = str(copy_path)
        representative_manifest.append(entry)
    _atomic_jsonl(representative_manifest, representative_dir / "manifest.jsonl")

    output_quality = {
        "records": len(rows),
        "valid_records": valid,
        "json_validity_rate": valid / len(rows) if rows else 0.0,
        "two_decimal_all_scores": True,
        "unsupported_user_score": "0.00",
    }
    classification = _classify(metrics, output_quality["json_validity_rate"])
    raw_performance = _performance_values(
        metrics, "raw_thresholded", output_quality["json_validity_rate"]
    )
    final_performance = _performance_values(
        metrics, "final_format", output_quality["json_validity_rate"]
    )
    performance = {
        "raw_thresholded": raw_performance,
        "final_format": final_performance,
        "final_minus_raw": {
            key: final_performance[key] - raw_performance[key] for key in raw_performance
        },
        "verdict_basis": "final_format",
    }
    report = {
        "metrics": metrics,
        "output_quality": output_quality,
        "classification": classification,
        "performance": performance,
        "verification": {"records": 32, "score_values": 32 * 57},
        "representative_6": representative_manifest,
        "representative_records": 6,
    }
    _atomic_json(report, output_dir / "metrics.json")
    return report


class EvaluationCollator:
    def __init__(
        self,
        processor: Any,
        manifest_parent: Path,
        image_max_pixels: int,
        *,
        image_cache: Mapping[str, Mapping[str, Any]] | None = None,
        split: str | None = None,
    ) -> None:
        self.processor = processor
        self.manifest_parent = manifest_parent
        self.image_max_pixels = image_max_pixels
        self.image_cache = image_cache
        self.split = split

    def __call__(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        from PIL import Image

        metadata_rows: list[dict[str, Any]] = []
        images = []
        for record in records:
            enriched = dict(record)
            record_id = str(record.get("record_id") or "")
            cached = (
                self.image_cache.get(record_id)
                if self.image_cache is not None
                else None
            )
            if self.image_cache is not None and cached is None:
                raise ValueError(f"cache entry missing for {record_id}")
            if cached is not None:
                if self.split is not None and cached.get("split") != self.split:
                    raise ValueError(f"cache split mismatch for {record_id}")
                width = int(cached["original_width"])
                height = int(cached["original_height"])
                with Image.open(Path(str(cached["cache_path"]))) as source:
                    expected_cached_size = (
                        int(cached["cached_width"]),
                        int(cached["cached_height"]),
                    )
                    if source.mode != "RGB" or source.size != expected_cached_size:
                        raise ValueError(f"cache pixel contract mismatch for {record_id}")
                    image = source.convert("RGB")
            else:
                try:
                    from scripts.train_unified57_qwen3vl_multilabel import (
                        _open_training_image,
                    )
                except ModuleNotFoundError:
                    from train_unified57_qwen3vl_multilabel import (  # type: ignore
                        _open_training_image,
                    )
                image_value = record.get("image_path") or record.get("local_image_path")
                if image_value:
                    image_path = Path(str(image_value))
                    if not image_path.is_absolute():
                        image_path = self.manifest_parent / image_path
                    with Image.open(image_path) as source:
                        width, height = source.size
                image = _open_training_image(
                    record, self.manifest_parent, self.image_max_pixels
                )
                if not image_value:
                    width, height = image.size
            enriched["width"] = int(width)
            enriched["height"] = int(height)
            enriched["aspect_ratio"] = (
                float(width) / float(height) if height else None
            )
            metadata_rows.append(enriched)
            images.append(image)
        batch = self.processor(
            images=images,
            text=[training_vision_prompt()] * len(records),
            padding=True,
            return_tensors="pt",
        )
        batch["metadata"] = metadata_rows
        return batch


def setup_distributed(expected_world_size: int):
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("Unified57 distributed evaluation requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if expected_world_size > 0 and world_size != expected_world_size:
        raise RuntimeError(f"expected {expected_world_size} ranks, got {world_size}")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", init_method="env://")
    return local_rank, rank, world_size, torch.device("cuda", local_rank)


def _shard_path(output_dir: Path, split: str, rank: int, world_size: int) -> Path:
    return output_dir / "prediction_shards" / f"{split}.rank{rank:02d}-of-{world_size:02d}.jsonl"


def predict_split(
    *,
    split: str,
    records: Sequence[dict[str, Any]],
    manifest_path: Path,
    manifest_sha256: str,
    model: Any,
    processor: Any,
    device: Any,
    rank: int,
    world_size: int,
    output_dir: Path,
    checkpoint_sha256: str,
    schema: Mapping[str, Any],
    batch_size: int,
    num_workers: int,
    image_max_pixels: int,
    deadline_monotonic: float,
    image_cache: Mapping[str, Mapping[str, Any]] | None = None,
    image_cache_provenance: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]] | None, bool]:
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader

    local_records = partition_records(records, rank, world_size)
    metadata = {
        "split": split,
        "rank": rank,
        "world_size": world_size,
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": manifest_sha256,
        "schema_sha256": schema["schema_sha256"],
        "image_cache_manifest_sha256": (
            image_cache_provenance.get("cache_manifest_sha256")
            if image_cache_provenance
            else None
        ),
        "image_cache_complete_marker_sha256": (
            image_cache_provenance.get("complete_marker_sha256")
            if image_cache_provenance
            else None
        ),
        "image_cache_decoder_contract_sha256": (
            image_cache_provenance.get("decoder_contract_sha256")
            if image_cache_provenance
            else None
        ),
    }
    shard = BufferedPredictionShard(
        _shard_path(output_dir, split, rank, world_size),
        metadata,
        sync_every_records=1000,
        sync_every_seconds=30.0,
    )
    start_index = shard.next_local_index
    if start_index > len(local_records):
        shard.close(complete=False)
        raise RuntimeError("prediction shard cursor exceeds local stride")
    pending = local_records[start_index:]
    loader = DataLoader(
        pending,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=EvaluationCollator(
            processor,
            manifest_path.parent,
            image_max_pixels,
            image_cache=image_cache,
            split=split,
        ),
    )
    complete = True
    model.eval()
    processed = start_index
    with torch.inference_mode():
        for batch in loader:
            if time.monotonic() >= deadline_monotonic:
                complete = False
                break
            metadata_rows = batch.pop("metadata")
            tensor_batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**tensor_batch)
                logits = output["logits"] if isinstance(output, Mapping) else output.logits
            scores = torch.sigmoid(logits.float()).cpu().tolist()
            output_rows = []
            for source, score_vector in zip(metadata_rows, scores):
                if len(score_vector) != 57 or any(not math.isfinite(float(v)) for v in score_vector):
                    raise RuntimeError("model emitted invalid Unified57 scores")
                row = dict(source)
                row["scores"] = [float(value) for value in score_vector]
                row["checkpoint_sha256"] = checkpoint_sha256
                output_rows.append(row)
            processed += len(output_rows)
            shard.append_batch(output_rows, processed)
    complete = complete and processed == len(local_records)
    shard.close(complete=complete)
    completion = torch.tensor(int(complete), dtype=torch.int32, device=device)
    dist.all_reduce(completion, op=dist.ReduceOp.MIN)
    globally_complete = bool(completion.item())
    dist.barrier()
    merged = None
    if rank == 0:
        paths = [_shard_path(output_dir, split, r, world_size) for r in range(world_size)]
        merged = merge_prediction_shards(
            records,
            paths,
            output_dir / f"{split}_predictions_float32.jsonl",
            require_complete=globally_complete,
        )
    dist.barrier()
    return merged, globally_complete


def _freeze_test_contract(path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key, value in contract.items():
            if existing.get(key) != value:
                raise ValueError(f"test run contract mismatch at {key}")
        return existing
    payload = {**contract, "state": "running", "started_at_unix": time.time()}
    _atomic_json(payload, path)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-config-sha256", required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--schema-file-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest-sha256", required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest-sha256", required=True)
    parser.add_argument("--expected-trainable-manifest-sha256", required=True)
    parser.add_argument("--base-artifact-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wall-clock-seconds", type=float, required=True)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-max-pixels", type=int, default=DEFAULT_IMAGE_MAX_PIXELS)
    parser.add_argument("--image-cache-root", type=Path)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    args = parser.parse_args(argv)
    for field in (
        "model_config_sha256", "schema_file_sha256", "checkpoint_sha256",
        "validation_manifest_sha256", "test_manifest_sha256",
        "expected_trainable_manifest_sha256", "base_artifact_manifest_sha256",
    ):
        try:
            _require_sha256(getattr(args, field), field)
        except ValueError as error:
            parser.error(str(error))
    if args.wall_clock_seconds <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        parser.error("wall clock/batch must be positive and workers non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started_wall = time.time()
    started_mono = time.monotonic()
    deadline = started_mono + args.wall_clock_seconds

    # Every file/cache contract is checked before NCCL, model loading, or CUDA
    # allocation. A requested cache therefore fails closed without stranding
    # distributed workers or partially loading the model.
    verify_file_sha256(
        Path(args.model) / "config.json",
        args.model_config_sha256,
        "model config",
    )
    schema = load_schema(args.schema, args.schema_file_sha256)
    checkpoint_sha = verify_file_sha256(
        args.checkpoint, args.checkpoint_sha256, "checkpoint"
    )
    validation_rows = load_manifest(
        args.validation_manifest,
        schema,
        expected_sha256=args.validation_manifest_sha256,
    )
    test_rows = load_manifest(
        args.test_manifest,
        schema,
        expected_sha256=args.test_manifest_sha256,
    )
    image_cache: dict[str, dict[str, Any]] | None = None
    image_cache_provenance: dict[str, Any] | None = None
    if args.image_cache_root is not None:
        image_cache, image_cache_provenance = load_validated_cache_overlay(
            cache_root=args.image_cache_root,
            validation_rows=validation_rows,
            test_rows=test_rows,
            validation_manifest_sha256=args.validation_manifest_sha256,
            test_manifest_sha256=args.test_manifest_sha256,
            image_max_pixels=args.image_max_pixels,
        )
        image_cache_provenance = {
            **image_cache_provenance,
            "enabled": True,
            "cache_root": str(args.image_cache_root),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    local_rank, rank, world_size, device = setup_distributed(args.expected_world_size)

    import torch
    import torch.distributed as dist
    from transformers import AutoProcessor

    try:
        try:
            from scripts.jd_multilabel_training_core import load_qwen3vl_classifier
            from scripts.train_unified57_qwen3vl_multilabel import (
                load_v3_model_state_for_inference,
            )
        except ModuleNotFoundError:
            from jd_multilabel_training_core import load_qwen3vl_classifier  # type: ignore
            from train_unified57_qwen3vl_multilabel import (  # type: ignore
                load_v3_model_state_for_inference,
            )

        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        model = load_qwen3vl_classifier(
            args.model,
            num_labels=57,
            use_lora=True,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            head_dropout=args.head_dropout,
            gradient_checkpointing=False,
        )
        checkpoint_metadata = load_v3_model_state_for_inference(
            args.checkpoint,
            model=model,
            expected_tag_order=schema["labels"],
            expected_schema_sha256=schema["schema_sha256"],
        )
        model.to(device).eval()

        validation_predictions, validation_complete = predict_split(
            split="validation",
            records=validation_rows,
            manifest_path=args.validation_manifest,
            manifest_sha256=args.validation_manifest_sha256,
            model=model,
            processor=processor,
            device=device,
            rank=rank,
            world_size=world_size,
            output_dir=args.output_dir,
            checkpoint_sha256=checkpoint_sha,
            schema=schema,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_max_pixels=args.image_max_pixels,
            deadline_monotonic=deadline,
            image_cache=image_cache,
            image_cache_provenance=image_cache_provenance,
        )
        if not validation_complete:
            if rank == 0:
                _atomic_json(
                    {
                        "report_version": REPORT_VERSION,
                        "state": "partial",
                        "reason": "validation_inference_incomplete_deadline",
                        "predicted_records": len(validation_predictions or []),
                        "expected_records": len(validation_rows),
                    },
                    args.output_dir / "status.json",
                )
            return 75

        threshold_objects: list[Any] = [None, None]
        report: dict[str, Any] | None = None
        status: str | None = None
        if rank == 0:
            threshold_path = args.output_dir / "thresholds.json"
            threshold_objects[0] = freeze_thresholds(
                threshold_path,
                validation_predictions or [],
                schema,
                checkpoint_sha256=checkpoint_sha,
                validation_manifest_sha256=args.validation_manifest_sha256,
            )
            threshold_objects[1] = sha256_file(threshold_path)
        dist.broadcast_object_list(threshold_objects, src=0)
        thresholds, threshold_sha = threshold_objects
        load_frozen_thresholds(
            args.output_dir / "thresholds.json",
            schema,
            checkpoint_sha256=checkpoint_sha,
            validation_manifest_sha256=args.validation_manifest_sha256,
            expected_threshold_sha256=threshold_sha,
        )

        contract = {
            "checkpoint_sha256": checkpoint_sha,
            "thresholds_sha256": threshold_sha,
            "validation_manifest_sha256": args.validation_manifest_sha256,
            "test_manifest_sha256": args.test_manifest_sha256,
            "schema_sha256": schema["schema_sha256"],
            "trainable_manifest_sha256": args.expected_trainable_manifest_sha256,
            "base_artifact_manifest_sha256": args.base_artifact_manifest_sha256,
            "image_cache_manifest_sha256": (
                image_cache_provenance.get("cache_manifest_sha256")
                if image_cache_provenance
                else None
            ),
            "image_cache_complete_marker_sha256": (
                image_cache_provenance.get("complete_marker_sha256")
                if image_cache_provenance
                else None
            ),
            "image_cache_decoder_contract_sha256": (
                image_cache_provenance.get("decoder_contract_sha256")
                if image_cache_provenance
                else None
            ),
        }
        if rank == 0:
            _freeze_test_contract(args.output_dir / "test_run_contract.json", contract)
        dist.barrier()
        test_predictions, test_complete = predict_split(
            split="test",
            records=test_rows,
            manifest_path=args.test_manifest,
            manifest_sha256=args.test_manifest_sha256,
            model=model,
            processor=processor,
            device=device,
            rank=rank,
            world_size=world_size,
            output_dir=args.output_dir,
            checkpoint_sha256=checkpoint_sha,
            schema=schema,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_max_pixels=args.image_max_pixels,
            deadline_monotonic=deadline,
            image_cache=image_cache,
            image_cache_provenance=image_cache_provenance,
        )
        verify_file_sha256(
            args.output_dir / "thresholds.json", threshold_sha, "threshold file after test"
        )
        if rank == 0:
            if test_complete:
                delivery = write_delivery_outputs(
                    test_predictions or [], thresholds, schema, args.output_dir
                )
                validation_metrics = evaluate_views(
                    validation_predictions or [], thresholds, schema
                )
                validation_metrics["dictionary_all_positive"] = {
                    "raw_thresholded": evaluate_dictionary_positive_recall(
                        validation_predictions or [], thresholds, schema, final_format=False
                    ),
                    "final_format": evaluate_dictionary_positive_recall(
                        validation_predictions or [], thresholds, schema, final_format=True
                    ),
                }
                test_contract = json.loads(
                    (args.output_dir / "test_run_contract.json").read_text(encoding="utf-8")
                )
                test_contract.update({"state": "complete", "completed_at_unix": time.time()})
                _atomic_json(test_contract, args.output_dir / "test_run_contract.json")
                status = delivery["classification"]["verdict"]
            else:
                delivery = None
                validation_metrics = None
                status = "partial"
            environment = {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "pytorch": torch.__version__,
                "transformers": package_version("transformers"),
                "peft": package_version("peft"),
                "safetensors": package_version("safetensors"),
                "pillow": package_version("Pillow"),
                "cuda": torch.version.cuda,
                "world_size": world_size,
                "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            }
            provenance = {
                **contract,
                "schema_file_sha256": args.schema_file_sha256,
                "model_config_sha256": args.model_config_sha256,
                "model_path": args.model,
                "checkpoint_path": str(args.checkpoint),
                "validation_manifest": str(args.validation_manifest),
                "test_manifest": str(args.test_manifest),
                "validation_predictions_sha256": sha256_file(
                    args.output_dir / "validation_predictions_float32.jsonl"
                ),
                "predictions_sha256": sha256_file(
                    args.output_dir / "test_predictions_float32.jsonl"
                ),
                "thresholds_sha256": threshold_sha,
                "image_cache": image_cache_provenance or {"enabled": False},
            }
            report = {
                "report_version": REPORT_VERSION,
                "status": status,
                "provenance": provenance,
                "checkpoint": {
                    "global_step": int(checkpoint_metadata.get("cursor", {}).get("global_step", 0)),
                    "format_version": checkpoint_metadata.get("format_version"),
                },
                "validation": {"expected": len(validation_rows), "predicted": len(validation_predictions or []), "complete": validation_complete},
                "test": {"expected": len(test_rows), "predicted": len(test_predictions or []), "complete": test_complete},
                "timing": {
                    "started_at_unix": started_wall,
                    "elapsed_seconds": time.monotonic() - started_mono,
                    "wall_clock_seconds": args.wall_clock_seconds,
                    "deadline_reached": time.monotonic() >= deadline,
                },
                "environment": environment,
                "image_cache": image_cache_provenance or {"enabled": False},
                "thresholds": thresholds,
                "raw_thresholded": (
                    delivery["metrics"]["raw_thresholded"] if delivery else None
                ),
                "final_format": (
                    delivery["metrics"]["final_format"] if delivery else None
                ),
                "format_constraint_loss": (
                    delivery["metrics"]["format_constraint_loss"] if delivery else None
                ),
                "threshold_calibration": (
                    delivery["metrics"]["threshold_calibration"] if delivery else None
                ),
                "dictionary_all_positive": (
                    delivery["metrics"]["dictionary_all_positive"] if delivery else None
                ),
                "performance": delivery["performance"] if delivery else None,
                "classification": delivery["classification"] if delivery else None,
                "output_quality": delivery["output_quality"] if delivery else None,
                "validation_metrics": validation_metrics,
                "reproduction_32": {
                    "status": "pending_reproduction" if test_complete else "blocked_incomplete_test",
                    "records": 32 if test_complete else 0,
                    "score_values": 32 * 57 if test_complete else 0,
                    "reference_float32": "verification/reference_32_float32.jsonl",
                    "reference_selected_only": "verification/reference_32_selected_only.jsonl",
                    "note": "Exact replay is measured by the sealed delivery package; evaluator only freezes references.",
                },
                "representative_6": delivery["representative_6"] if delivery else [],
                "representative_6_paths": {
                    "manifest": "representative6/manifest.jsonl",
                    "images": "representative6/images/",
                },
                "process_cleanup": {
                    "complete": False,
                    "distributed_barrier_reached": False,
                    "unrelated_processes_touched": 0,
                },
            }
        # This is the final distributed barrier. The candidate report is
        # deliberately unpublished until every rank has finished inference and
        # rank-local shard close/fsync work.
        dist.barrier()
        dist.destroy_process_group()
        del model
        torch.cuda.empty_cache()
        if rank == 0:
            assert report is not None and status is not None
            report["process_cleanup"] = {
                "complete": True,
                "distributed_barrier_reached": True,
                "distributed_group_destroyed": True,
                "cuda_cache_released": True,
                "unrelated_processes_touched": 0,
            }
            report["timing"]["elapsed_seconds"] = time.monotonic() - started_mono
            _atomic_json(report, args.output_dir / "evaluation_report.json")
            _atomic_json(
                {"state": status, "test_complete": test_complete},
                args.output_dir / "status.json",
            )
            print(json.dumps(report, ensure_ascii=False), flush=True)
        return 0 if test_complete else 75
    finally:
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
