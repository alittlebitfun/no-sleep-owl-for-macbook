#!/usr/bin/env python3
"""Independently audit a frozen Bosideng Unified57 delivery.

This program is deliberately read-only with respect to every input tree. It
uses only the Python standard library and writes atomic reports below an
explicit output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_MODE_COUNTS = {"pn": 36, "pu": 20, "unsupported": 1}
EXPECTED_CATEGORIES = ("局部结构", "廓形", "工艺", "面辅料")
UNSUPPORTED_TAG = "假两件"
JD23_TAGS = (
    "长款",
    "中款",
    "短款",
    "H型",
    "O型",
    "X型",
    "A型",
    "宽松",
    "连帽",
    "毛领",
    "立领",
    "翻领",
    "无领",
    "压胶充绒",
    "压胶袋盖",
    "压胶门襟",
    "平行绗线",
    "菱形绗线",
    "葫芦型绗线",
    "反光条",
    "按扣",
    "腰带",
    "插肩袖",
)
SUCCESS_THRESHOLDS = {
    "known_micro_f1": 0.88,
    "jd23_micro_f1": 0.88,
    "macro_f1": 0.75,
    "dictionary_positive_macro_recall": 0.85,
    "trusted_negative_specificity": 0.90,
    "json_validity_rate": 1.0,
}
TWO_DECIMAL_RE = re.compile(r"(?:0\.\d{2}|1\.00)\Z")
EVIDENCE_FIELDS = (
    "image_path",
    "image_sha256",
    "source",
    "sources",
    "schema_version",
    "schema_sha256",
    "labels",
    "known_mask",
    "pu_positive_mask",
)
AUDIT_OUTPUT_FILES = (
    "acceptance_audit.json",
    "per_label_metrics.csv",
    "FINAL_REPORT.md",
    "output_modes_audit.json",
    "representative_selection.jsonl",
)


class AuditContractError(ValueError):
    """A frozen input or evidence contract is invalid."""


@dataclass(frozen=True)
class AuditPaths:
    schema: Path
    dataset_root: Path
    evaluation_dir: Path
    posttrain_dir: Path
    output_dir: Path
    delivery_dir: Path | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditContractError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _load_json(path: Path | str, *, name: str) -> Any:
    source = Path(path)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditContractError(f"unable to read {name}: {source}") from exc


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AuditContractError(f"unable to hash file: {path}") from exc
    return digest.hexdigest()


def _load_jsonl(path: Path | str, *, name: str) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    try:
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AuditContractError(
                        f"{name} line {line_number} must contain an object"
                    )
                rows.append(value)
    except AuditContractError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditContractError(f"unable to read {name}: {source}") from exc
    return rows


def load_schema(path: Path | str) -> dict[str, Any]:
    payload = _load_json(path, name="Unified57 schema")
    _require(isinstance(payload, dict), "schema must be a JSON object")
    labels = payload.get("labels")
    _require(
        isinstance(labels, list) and len(labels) == 57 and len(set(labels)) == 57,
        "schema must contain exactly 57 unique labels",
    )
    _require(payload.get("num_labels") == 57, "schema num_labels must be 57")
    modes = payload.get("label_training_modes")
    _require(
        isinstance(modes, dict) and list(modes) == labels,
        "label_training_modes must exactly follow label order",
    )
    counts = Counter(modes.values())
    _require(
        {key: counts.get(key, 0) for key in EXPECTED_MODE_COUNTS}
        == EXPECTED_MODE_COUNTS
        and set(counts) == set(EXPECTED_MODE_COUNTS),
        "schema must have exactly 36 PN / 20 PU / 1 unsupported",
    )
    _require(
        payload.get("unsupported_labels") == [UNSUPPORTED_TAG]
        and modes.get(UNSUPPORTED_TAG) == "unsupported",
        "假两件 must be the sole unsupported label",
    )
    semantic = payload.get("semantic_categories")
    _require(
        isinstance(semantic, dict) and tuple(semantic) == EXPECTED_CATEGORIES,
        "schema must contain four ordered semantic categories",
    )
    flattened: list[str] = []
    for subcategories in semantic.values():
        _require(isinstance(subcategories, dict), "semantic subcategories must be objects")
        for tags in subcategories.values():
            _require(isinstance(tags, list) and tags, "subcategory tags must be non-empty")
            flattened.extend(tags)
    _require(Counter(flattened) == Counter(labels), "semantic schema must cover 57 labels once")
    return payload


def _validate_binary_vector(value: Any, *, name: str, record_id: str) -> list[Any]:
    _require(
        isinstance(value, list) and len(value) == 57,
        f"{record_id}: {name} must contain exactly 57 values",
    )
    _require(
        all(item in (0, 1, 0.0, 1.0, False, True) for item in value),
        f"{record_id}: {name} must be binary",
    )
    return value


def _validate_scores(value: Any, *, record_id: str) -> list[float]:
    _require(
        isinstance(value, list) and len(value) == 57,
        f"{record_id}: scores must contain exactly 57 values",
    )
    scores: list[float] = []
    for item in value:
        _require(
            not isinstance(item, bool) and isinstance(item, (int, float)),
            f"{record_id}: scores must be numeric",
        )
        score = float(item)
        _require(
            math.isfinite(score) and 0.0 <= score <= 1.0,
            f"{record_id}: scores must be finite and within [0,1]",
        )
        scores.append(score)
    return scores


def _validate_manifest_row(row: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    record_id = row.get("record_id")
    _require(isinstance(record_id, str) and record_id, "manifest record_id is invalid")
    _require(
        row.get("schema_sha256") == schema.get("schema_sha256"),
        f"{record_id}: schema_sha256 mismatch",
    )
    _require(
        row.get("schema_version", schema.get("schema_version"))
        == schema.get("schema_version"),
        f"{record_id}: schema_version mismatch",
    )
    _require(
        _is_sha256(row.get("image_sha256")),
        f"{record_id}: image_sha256 must be a lowercase SHA256",
    )
    sources = row.get("sources")
    _require(
        isinstance(sources, list)
        and bool(sources)
        and all(isinstance(item, str) and item for item in sources),
        f"{record_id}: sources must be a non-empty string array",
    )
    labels = _validate_binary_vector(row.get("labels"), name="labels", record_id=record_id)
    known = _validate_binary_vector(
        row.get("known_mask"), name="known_mask", record_id=record_id
    )
    pu = _validate_binary_vector(
        row.get("pu_positive_mask"), name="pu_positive_mask", record_id=record_id
    )
    for index, tag in enumerate(schema["labels"]):
        mode = schema["label_training_modes"][tag]
        is_known = bool(known[index])
        is_pu = bool(pu[index])
        label = float(labels[index])
        _require(not (is_known and is_pu), f"{record_id}: masks overlap at {tag}")
        if is_known:
            _require(mode == "pn", f"{record_id}: known mask is invalid at {tag}")
        elif is_pu:
            _require(
                mode == "pu" and label == 1.0,
                f"{record_id}: PU positive mask is invalid at {tag}",
            )
        else:
            _require(label == 0.0, f"{record_id}: unknown cell is non-neutral at {tag}")


def load_split_evidence(
    manifest_path: Path | str,
    prediction_path: Path | str,
    schema: Mapping[str, Any],
    expected_count: int,
    *,
    expected_checkpoint_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _require(expected_count > 0, "expected split count must be positive")
    manifest = _load_jsonl(manifest_path, name="frozen manifest")
    predictions = _load_jsonl(prediction_path, name="float32 predictions")
    _require(
        len(manifest) == expected_count,
        f"unexpected manifest count: expected {expected_count}, got {len(manifest)}",
    )
    _require(
        len(predictions) == expected_count,
        f"unexpected prediction count: expected {expected_count}, got {len(predictions)}",
    )
    manifest_ids = [row.get("record_id") for row in manifest]
    prediction_ids = [row.get("record_id") for row in predictions]
    _require(
        all(isinstance(record_id, str) and record_id for record_id in manifest_ids)
        and len(set(manifest_ids)) == expected_count,
        "manifest record_id values must be unique",
    )
    _require(
        prediction_ids == manifest_ids,
        "predictions differ from frozen manifest order",
    )
    for truth, scored in zip(manifest, predictions):
        _validate_manifest_row(truth, schema)
        record_id = str(truth["record_id"])
        for field in EVIDENCE_FIELDS:
            _require(
                scored.get(field) == truth.get(field),
                f"{record_id}: prediction evidence differs at {field}",
            )
        checkpoint = scored.get("checkpoint_sha256")
        _require(
            _is_sha256(checkpoint),
            f"{record_id}: prediction checkpoint_sha256 is invalid",
        )
        if expected_checkpoint_sha256 is not None:
            _require(
                checkpoint == expected_checkpoint_sha256,
                f"{record_id}: prediction checkpoint differs from frozen thresholds",
            )
        _validate_scores(scored.get("scores"), record_id=record_id)
    return manifest, predictions


def validate_leakage(path: Path | str) -> dict[str, Any]:
    payload = _load_json(path, name="leakage audit")
    _require(isinstance(payload, dict), "leakage audit must be an object")
    collision_fields = (
        "cross_split_components",
        "cross_split_exact_phash",
        "cross_split_sha256",
    )
    _require(
        payload.get("passed") is True
        and all(isinstance(payload.get(field), list) and not payload[field] for field in collision_fields),
        "visual leakage audit did not pass with zero collisions",
    )
    return payload


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _count_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "known_cells": tp + fp + fn + tn,
        "positive_support": tp + fn,
        "negative_support": tn + fp,
        "known_positive": tp + fn,
        "known_negative": tn + fp,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "specificity": _ratio(tn, tn + fp),
        "fpr": _ratio(fp, fp + tn),
        "accuracy": _ratio(tp + tn, tp + fp + fn + tn),
    }


def _thresholds_map(
    thresholds: Mapping[str, Any], schema: Mapping[str, Any]
) -> dict[str, float | None]:
    entries = thresholds.get("labels")
    _require(
        isinstance(entries, Mapping) and list(entries) == schema["labels"],
        "threshold labels must exactly follow schema order",
    )
    output: dict[str, float | None] = {}
    for tag in schema["labels"]:
        entry = entries[tag]
        _require(isinstance(entry, Mapping), f"threshold entry for {tag} must be an object")
        _require(
            entry.get("mode") in (None, schema["label_training_modes"][tag]),
            f"threshold mode mismatch for {tag}",
        )
        value = entry.get("threshold")
        if tag == UNSUPPORTED_TAG:
            _require(value is None, "unsupported threshold must be null")
            output[tag] = None
            continue
        _require(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= 1.0,
            f"threshold for {tag} must be finite and within [0,1]",
        )
        output[tag] = float(value)
    return output


def _prediction_vectors(
    rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> tuple[list[list[int]], list[list[int]]]:
    threshold_map = _thresholds_map(thresholds, schema)
    positions = {tag: index for index, tag in enumerate(schema["labels"])}
    raw_rows: list[list[int]] = []
    final_rows: list[list[int]] = []
    for row in rows:
        record_id = str(row.get("record_id") or "")
        scores = _validate_scores(row.get("scores"), record_id=record_id)
        raw = [0] * 57
        for index, tag in enumerate(schema["labels"]):
            threshold = threshold_map[tag]
            if threshold is not None and scores[index] >= threshold:
                raw[index] = 1
        final = [0] * 57
        for subcategories in schema["semantic_categories"].values():
            for tags in subcategories.values():
                eligible = [positions[tag] for tag in tags if raw[positions[tag]]]
                if eligible:
                    winner = max(eligible, key=lambda item: (scores[item], -item))
                    final[winner] = 1
        final[positions[UNSUPPORTED_TAG]] = 0
        raw_rows.append(raw)
        final_rows.append(final)
    return raw_rows, final_rows


def _pn_report(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[int]],
    indices: Sequence[int],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    per_label: dict[str, dict[str, Any]] = {}
    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    exact_hits = 0
    exact_rows = 0
    for label_index in indices:
        tp = fp = fn = tn = 0
        for row, predicted in zip(rows, predictions):
            if not bool(row["known_mask"][label_index]):
                continue
            truth = int(float(row["labels"][label_index]) == 1.0)
            guess = int(predicted[label_index])
            tp += truth == 1 and guess == 1
            fp += truth == 0 and guess == 1
            fn += truth == 1 and guess == 0
            tn += truth == 0 and guess == 0
        metrics = _count_metrics(tp, fp, fn, tn)
        per_label[schema["labels"][label_index]] = metrics
        for key in totals:
            totals[key] += int(metrics[key])
    for row, predicted in zip(rows, predictions):
        observed = [index for index in indices if bool(row["known_mask"][index])]
        if observed:
            exact_rows += 1
            exact_hits += all(
                int(float(row["labels"][index]) == 1.0) == int(predicted[index])
                for index in observed
            )
    eligible = [
        item
        for item in per_label.values()
        if item["positive_support"] > 0 and item["negative_support"] > 0
    ]
    evaluated = [item for item in per_label.values() if item["known_cells"] > 0]
    positive_labels = [item for item in evaluated if item["positive_support"] > 0]
    macro = {
        key: _ratio(sum(item[key] for item in evaluated), len(evaluated))
        for key in ("precision", "recall", "f1", "specificity", "fpr", "accuracy")
    }
    macro.update(
        {
            "labels_evaluated": len(evaluated),
            "labels_with_both_classes": len(eligible),
            "f1_both_class_labels": _ratio(
                sum(item["f1"] for item in eligible), len(eligible)
            ),
            "positive_labels_evaluated": len(positive_labels),
            "positive_labels_macro_recall": _ratio(
                sum(item["recall"] for item in positive_labels),
                len(positive_labels),
            ),
        }
    )
    micro = _count_metrics(**totals)
    return {
        "record_count": len(rows),
        "micro": micro,
        "macro": macro,
        "macro_f1": macro["f1_both_class_labels"],
        "labels_with_both_classes": len(eligible),
        "exact_match": _ratio(exact_hits, exact_rows),
        "exact_match_rows": exact_rows,
        "per_label": per_label,
        "trusted_negatives": {
            "tn": micro["tn"],
            "fp": micro["fp"],
            "specificity": micro["specificity"],
            "fpr": micro["fpr"],
        },
    }


def _dictionary_report(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[int]],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    pairs = [
        (row, prediction)
        for row, prediction in zip(rows, predictions)
        if "dictionary_v4" in row.get("sources", [])
    ]
    per_label: dict[str, dict[str, Any]] = {}
    recalls: list[float] = []
    total_support = 0
    total_hits = 0
    for index, tag in enumerate(schema["labels"]):
        mode = schema["label_training_modes"][tag]
        if mode == "unsupported":
            continue
        if mode == "pn":
            flags = [
                bool(row["known_mask"][index])
                and float(row["labels"][index]) == 1.0
                for row, _ in pairs
            ]
            supervision = "known_positive"
        else:
            flags = [bool(row["pu_positive_mask"][index]) for row, _ in pairs]
            supervision = "pu_positive_mask"
        support = sum(flags)
        if not support:
            continue
        hits = sum(
            int(prediction[index])
            for positive, (_, prediction) in zip(flags, pairs)
            if positive
        )
        recall = hits / support
        recalls.append(recall)
        total_support += support
        total_hits += hits
        per_label[tag] = {
            "mode": mode,
            "supervision": supervision,
            "positive_support": support,
            "selected_positive": hits,
            "recall": recall,
        }
    return {
        "dictionary_records": len(pairs),
        "labels_with_positive_support": len(recalls),
        "macro_positive_recall": _ratio(sum(recalls), len(recalls)),
        "micro_positive_recall": _ratio(total_hits, total_support),
        "positive_support": total_support,
        "selected_positive": total_hits,
        "per_label": per_label,
    }


def _pu_report(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[int]],
    schema: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    per_label: dict[str, dict[str, Any]] = {}
    for index, tag in enumerate(schema["labels"]):
        if schema["label_training_modes"][tag] != "pu":
            continue
        positive = [
            int(prediction[index])
            for row, prediction in zip(rows, predictions)
            if bool(row["pu_positive_mask"][index])
        ]
        unlabeled = [
            int(prediction[index])
            for row, prediction in zip(rows, predictions)
            if not bool(row["known_mask"][index])
            and not bool(row["pu_positive_mask"][index])
        ]
        per_label[tag] = {
            "positive_support": len(positive),
            "selected_positive": sum(positive),
            "positive_recall": _ratio(sum(positive), len(positive)),
            "unlabeled_support": len(unlabeled),
            "selected_unlabeled": sum(unlabeled),
            "unlabeled_coverage": _ratio(sum(unlabeled), len(unlabeled)),
        }
    return per_label


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _pu_view_report(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[int]],
    thresholds: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    per_label: dict[str, Any] = {}
    total_positive = total_unlabeled = selected_positive = selected_unlabeled = 0
    concordances: list[float] = []
    entries = thresholds["labels"]
    for index, tag in enumerate(schema["labels"]):
        if schema["label_training_modes"][tag] != "pu":
            continue
        positive_scores = [
            float(row["scores"][index])
            for row in rows
            if bool(row["pu_positive_mask"][index])
        ]
        unlabeled_scores = [
            float(row["scores"][index])
            for row in rows
            if not bool(row["known_mask"][index])
            and not bool(row["pu_positive_mask"][index])
        ]
        positive_flags = [
            int(prediction[index])
            for row, prediction in zip(rows, predictions)
            if bool(row["pu_positive_mask"][index])
        ]
        unlabeled_flags = [
            int(prediction[index])
            for row, prediction in zip(rows, predictions)
            if not bool(row["known_mask"][index])
            and not bool(row["pu_positive_mask"][index])
        ]
        sorted_unlabeled = sorted(unlabeled_scores)
        concordant = 0.0
        for score in positive_scores:
            lower = 0
            while lower < len(sorted_unlabeled) and sorted_unlabeled[lower] < score:
                lower += 1
            upper = lower
            while upper < len(sorted_unlabeled) and sorted_unlabeled[upper] == score:
                upper += 1
            concordant += lower + 0.5 * (upper - lower)
        positive_selected = sum(positive_flags)
        unlabeled_selected = sum(unlabeled_flags)
        concordance = _ratio(
            concordant, len(positive_scores) * len(unlabeled_scores)
        )
        item = {
            "positive_count": len(positive_scores),
            "unlabeled_count": len(unlabeled_scores),
            "positive_recall": _ratio(positive_selected, len(positive_scores)),
            "positive_vs_unlabeled_concordance": concordance,
            "positive_score_quantiles": {
                "p10": _quantile(positive_scores, 0.10),
                "p50": _quantile(positive_scores, 0.50),
                "p90": _quantile(positive_scores, 0.90),
            },
            "unlabeled_score_quantiles": {
                "p10": _quantile(unlabeled_scores, 0.10),
                "p50": _quantile(unlabeled_scores, 0.50),
                "p90": _quantile(unlabeled_scores, 0.90),
            },
            "all_coverage": _ratio(
                positive_selected + unlabeled_selected,
                len(positive_scores) + len(unlabeled_scores),
            ),
            "unlabeled_coverage": _ratio(
                unlabeled_selected, len(unlabeled_scores)
            ),
            "selected_positive_count": positive_selected,
            "selected_unlabeled_count": unlabeled_selected,
            "threshold": float(entries[tag]["threshold"]),
            "calibration_status": entries[tag].get("status"),
        }
        per_label[tag] = item
        total_positive += len(positive_scores)
        total_unlabeled += len(unlabeled_scores)
        selected_positive += positive_selected
        selected_unlabeled += unlabeled_selected
        if positive_scores and unlabeled_scores:
            concordances.append(concordance)
    return {
        "per_label": per_label,
        "summary": {
            "positive_count": total_positive,
            "unlabeled_count": total_unlabeled,
            "support_weighted_positive_recall": _ratio(
                selected_positive, total_positive
            ),
            "macro_positive_vs_unlabeled_concordance": _ratio(
                sum(concordances), len(concordances)
            ),
            "micro_all_coverage": _ratio(
                selected_positive + selected_unlabeled,
                total_positive + total_unlabeled,
            ),
            "micro_unlabeled_coverage": _ratio(
                selected_unlabeled, total_unlabeled
            ),
        },
    }


def _slice_view(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[int]],
    thresholds: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    positions = {tag: index for index, tag in enumerate(schema["labels"])}
    pn_indices = [
        index
        for index, tag in enumerate(schema["labels"])
        if schema["label_training_modes"][tag] == "pn"
    ]

    def subset(sources: list[str]) -> tuple[list[Mapping[str, Any]], list[Sequence[int]]]:
        pairs = [
            (row, prediction)
            for row, prediction in zip(rows, predictions)
            if row.get("sources") == sources
        ]
        return [row for row, _ in pairs], [prediction for _, prediction in pairs]

    jd_rows, jd_predictions = subset(["jd_complete23"])
    dictionary_rows, dictionary_predictions = subset(["dictionary_v4"])
    mixed_rows = [
        row
        for row in rows
        if set(row.get("sources", [])) == {"jd_complete23", "dictionary_v4"}
    ]
    overall = _pn_report(rows, predictions, pn_indices, schema)
    mixed_known = sum(
        bool(row["known_mask"][index])
        for row in mixed_rows
        for index in pn_indices
    )
    mixed_positive = sum(
        bool(row["known_mask"][index]) and float(row["labels"][index]) == 1.0
        for row in mixed_rows
        for index in pn_indices
    )
    return {
        "overall_36pn": overall,
        "jd23_clean": _pn_report(
            jd_rows,
            jd_predictions,
            [positions[tag] for tag in JD23_TAGS],
            schema,
        ),
        "dictionary_pn_clean": _pn_report(
            dictionary_rows, dictionary_predictions, pn_indices, schema
        ),
        "mixed_exact_audit": {
            "record_count": len(mixed_rows),
            "known_cells": mixed_known,
            "known_positive": mixed_positive,
            "known_negative": mixed_known - mixed_positive,
            "fraction_of_overall_known_cells": _ratio(
                mixed_known, overall["micro"]["known_cells"]
            ),
        },
        "pu": _pu_view_report(rows, predictions, thresholds, schema),
    }


def _format_constraint_loss(
    rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Sequence[int]],
    final_rows: Sequence[Sequence[int]],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    positions = {tag: index for index, tag in enumerate(schema["labels"])}
    raw_multi: dict[str, int] = {}
    observed_multi: dict[str, int] = {}
    suppressed_by_subcategory: dict[str, int] = {}
    suppressed_by_tag: Counter[str] = Counter()
    hit_rate: dict[str, dict[str, Any]] = {}
    observed_total = retained_total = forced_false_negatives = 0
    for subcategories in schema["semantic_categories"].values():
        for subcategory, tags in subcategories.items():
            indices = [positions[tag] for tag in tags if tag != UNSUPPORTED_TAG]
            raw_multi[subcategory] = 0
            observed_multi[subcategory] = 0
            suppressed_by_subcategory[subcategory] = 0
            hit = observed_records = 0
            for row, raw, final in zip(rows, raw_rows, final_rows):
                if sum(raw[index] for index in indices) > 1:
                    raw_multi[subcategory] += 1
                for index in indices:
                    if raw[index] and not final[index]:
                        suppressed_by_subcategory[subcategory] += 1
                        suppressed_by_tag[schema["labels"][index]] += 1
                observed = [
                    index
                    for index in indices
                    if (
                        bool(row["known_mask"][index])
                        and float(row["labels"][index]) == 1.0
                    )
                    or bool(row["pu_positive_mask"][index])
                ]
                if observed:
                    observed_records += 1
                    hit += any(final[index] for index in observed)
                    observed_total += len(observed)
                    retained_total += 1
                    forced_false_negatives += max(0, len(observed) - 1)
                    if len(observed) > 1:
                        observed_multi[subcategory] += 1
            hit_rate[subcategory] = {
                "hit": hit,
                "observed_records": observed_records,
                "rate": _ratio(hit, observed_records),
            }
    return {
        "raw_multi_selected_records": raw_multi,
        "formatter_suppressed_predictions": {
            "total": sum(suppressed_by_subcategory.values()),
            "by_subcategory": suppressed_by_subcategory,
            "by_tag": {
                tag: suppressed_by_tag.get(tag, 0)
                for tag in schema["labels"]
                if tag != UNSUPPORTED_TAG
            },
        },
        "observed_multi_positive_records": observed_multi,
        "observed_positive_count": observed_total,
        "oracle_final_recall_ceiling": _ratio(retained_total, observed_total),
        "contract_forced_false_negatives": forced_false_negatives,
        "subcategory_hit_rate": hit_rate,
    }


def recompute_metrics(
    rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    json_validity_rate: float = 1.0,
) -> dict[str, Any]:
    for row in rows:
        _validate_manifest_row(row, schema)
    raw, final = _prediction_vectors(rows, thresholds, schema)
    raw_view = _slice_view(rows, raw, thresholds, schema)
    final_view = _slice_view(rows, final, thresholds, schema)
    raw_dictionary = _dictionary_report(rows, raw, schema)
    final_dictionary = _dictionary_report(rows, final, schema)

    def values(view: Mapping[str, Any], dictionary: Mapping[str, Any]) -> dict[str, float]:
        overall = view["overall_36pn"]
        return {
            "known_micro_f1": float(overall["micro"]["f1"]),
            "jd23_micro_f1": float(view["jd23_clean"]["micro"]["f1"]),
            "macro_f1": float(overall["macro"]["f1_both_class_labels"]),
            "dictionary_positive_macro_recall": float(
                dictionary["macro_positive_recall"]
            ),
            "trusted_negative_specificity": float(
                overall["trusted_negatives"]["specificity"]
            ),
            "json_validity_rate": float(json_validity_rate),
        }

    raw_values = values(raw_view, raw_dictionary)
    final_values = values(final_view, final_dictionary)
    return {
        "values": final_values,
        "raw_values": raw_values,
        "performance": {
            "raw_thresholded": raw_values,
            "final_format": final_values,
            "final_minus_raw": {
                key: final_values[key] - raw_values[key] for key in final_values
            },
            "verdict_basis": "final_format",
        },
        "raw_thresholded": raw_view,
        "final_format": final_view,
        "format_constraint_loss": _format_constraint_loss(rows, raw, final, schema),
        "dictionary_all_positive_views": {
            "raw_thresholded": raw_dictionary,
            "final_format": final_dictionary,
        },
        "overall_36pn": final_view["overall_36pn"],
        "jd23_clean": final_view["jd23_clean"],
        "dictionary_all_positive": final_dictionary,
        "pn_per_label": final_view["overall_36pn"]["per_label"],
        "pu_per_label": _pu_report(rows, final, schema),
        "dictionary_per_label": final_dictionary["per_label"],
        "raw_predictions": raw,
        "final_predictions": final,
    }


def classify_performance(values: Mapping[str, float]) -> dict[str, Any]:
    missing = set(SUCCESS_THRESHOLDS).difference(values)
    _require(not missing, f"performance values are missing: {sorted(missing)}")
    gates = {
        key: (
            float(values[key]) == 1.0
            if key == "json_validity_rate"
            else float(values[key]) >= threshold
        )
        for key, threshold in SUCCESS_THRESHOLDS.items()
    }
    if all(gates.values()):
        verdict = "success"
    elif float(values["known_micro_f1"]) >= 0.82 and gates["json_validity_rate"]:
        verdict = "partial"
    else:
        verdict = "fail"
    return {"verdict": verdict, "success_gates": gates, "values": dict(values)}


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _markdown_report(audit: Mapping[str, Any]) -> str:
    values = audit.get("values", {})
    rows = [
        "# Unified57 最终交付审计",
        "",
        f"- 结论：`{audit.get('verdict', 'integrity_error')}`",
        f"- 完整性：`{'通过' if audit.get('integrity', {}).get('passed') else '失败'}`",
        f"- Validation 数量：{audit.get('counts', {}).get('validation', 'unknown')}",
        f"- Test 数量：{audit.get('counts', {}).get('test', 'unknown')}",
        "",
        "## 六项正式指标",
        "",
    ]
    labels = {
        "known_micro_f1": "Known PN micro F1",
        "jd23_micro_f1": "JD23 clean micro F1",
        "macro_f1": "PN macro F1",
        "dictionary_positive_macro_recall": "字典显式正例 macro recall",
        "trusted_negative_specificity": "可信负例 specificity",
        "json_validity_rate": "JSON 合法率",
    }
    for key, label in labels.items():
        value = values.get(key)
        rows.append(f"- {label}：{float(value):.4f}" if value is not None else f"- {label}：未计算")
    warnings = audit.get("warnings") or []
    if warnings:
        rows.extend(["", "## 风险与说明", ""])
        rows.extend(f"- {warning}" for warning in warnings)
    errors = audit.get("errors") or []
    if errors:
        rows.extend(["", "## 完整性错误", ""])
        rows.extend(f"- {error}" for error in errors)
    rows.append("")
    return "\n".join(rows)


def write_audit_reports(
    output_dir: Path | str,
    audit: Mapping[str, Any],
    per_label_rows: Sequence[Mapping[str, Any]],
) -> None:
    output = Path(output_dir)
    _atomic_text(
        output / "acceptance_audit.json",
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    preferred_fields = [
        "tag",
        "mode",
        "positive_support",
        "negative_support",
        "tp",
        "fp",
        "fn",
        "tn",
        "precision",
        "recall",
        "f1",
        "specificity",
        "unlabeled_support",
        "unlabeled_coverage",
        "dictionary_positive_support",
        "dictionary_positive_recall",
        "warning",
    ]
    extra_fields = sorted(
        {key for row in per_label_rows for key in row}.difference(preferred_fields)
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=preferred_fields + extra_fields)
    writer.writeheader()
    for row in per_label_rows:
        writer.writerow(dict(row))
    _atomic_text(output / "per_label_metrics.csv", buffer.getvalue())
    _atomic_text(output / "FINAL_REPORT.md", _markdown_report(audit))


def _render_selected(
    scores: Sequence[float],
    final: Sequence[int],
    schema: Mapping[str, Any],
) -> dict[str, list[str]]:
    positions = {tag: index for index, tag in enumerate(schema["labels"])}
    return {
        category: sorted(
            (
                tag
                for tags in subcategories.values()
                for tag in tags
                if tag != UNSUPPORTED_TAG and final[positions[tag]]
            ),
            key=positions.__getitem__,
        )
        for category, subcategories in schema["semantic_categories"].items()
    }


def _validate_selected_payload(
    payload: Any,
    expected: Mapping[str, list[str]],
    schema: Mapping[str, Any],
    *,
    record_id: str,
) -> None:
    _require(
        isinstance(payload, Mapping) and tuple(payload) == EXPECTED_CATEGORIES,
        f"{record_id}: selected output must contain four ordered categories",
    )
    positions = {tag: index for index, tag in enumerate(schema["labels"])}
    seen: set[str] = set()
    for category, tags in payload.items():
        _require(
            isinstance(tags, list) and all(isinstance(tag, str) for tag in tags),
            f"{record_id}: selected category must be a string array",
        )
        _require(
            tags == sorted(tags, key=lambda tag: positions.get(tag, 10**9)),
            f"{record_id}: selected tags differ from schema order",
        )
        allowed_category = {
            tag
            for values in schema["semantic_categories"][category].values()
            for tag in values
        }
        _require(set(tags) <= allowed_category, f"{record_id}: selected tag is in wrong category")
        _require(not (set(tags) & seen), f"{record_id}: selected tag is duplicated")
        seen.update(tags)
        _require(UNSUPPORTED_TAG not in tags, f"{record_id}: unsupported tag was selected")
        for subcategory, allowed in schema["semantic_categories"][category].items():
            _require(
                sum(tag in allowed for tag in tags) <= 1,
                f"{record_id}: subcategory {subcategory} has multiple tags",
            )
    _require(dict(payload) == dict(expected), f"{record_id}: selected output differs from scores")


def audit_output_modes(
    evaluation_dir: Path | str,
    rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    root = Path(evaluation_dir)
    files = {
        "selected_only": root / "test_selected_only.jsonl",
        "selected_with_confidence": root / "test_selected_with_confidence.jsonl",
        "all_scores": root / "test_all_scores.jsonl",
    }
    available = [mode for mode, path in files.items() if path.is_file()]
    if require_complete:
        missing = [mode for mode in files if mode not in available]
        _require(not missing, f"required output modes are missing: {missing}")
    _, final_rows = _prediction_vectors(rows, thresholds, schema)
    positions = {tag: index for index, tag in enumerate(schema["labels"])}
    expected_ids = [row["record_id"] for row in rows]
    expected_selected = [
        _render_selected(row["scores"], final, schema)
        for row, final in zip(rows, final_rows)
    ]
    records_by_mode: dict[str, int] = {}
    parsed_by_mode: dict[str, list[dict[str, Any]]] = {}
    for mode in available:
        parsed = _load_jsonl(files[mode], name=f"{mode} output")
        _require(
            [item.get("record_id") for item in parsed] == expected_ids,
            f"{mode} output differs from prediction order",
        )
        parsed_by_mode[mode] = parsed
        records_by_mode[mode] = len(parsed)
    if "selected_only" in parsed_by_mode:
        for record_id, item, expected in zip(
            expected_ids, parsed_by_mode["selected_only"], expected_selected
        ):
            _validate_selected_payload(item.get("output"), expected, schema, record_id=record_id)
    if "selected_with_confidence" in parsed_by_mode:
        for record_id, row, item, expected in zip(
            expected_ids,
            rows,
            parsed_by_mode["selected_with_confidence"],
            expected_selected,
        ):
            payload = item.get("output")
            _require(
                isinstance(payload, Mapping) and tuple(payload) == EXPECTED_CATEGORIES,
                f"{record_id}: selected-with-confidence categories are invalid",
            )
            names: dict[str, list[str]] = {}
            for category, values in payload.items():
                _require(isinstance(values, list), f"{record_id}: confidence values must be arrays")
                names[category] = []
                for value in values:
                    _require(
                        isinstance(value, Mapping) and tuple(value) == ("name", "confidence"),
                        f"{record_id}: confidence item must contain name and confidence",
                    )
                    tag = value["name"]
                    confidence = value["confidence"]
                    _require(
                        isinstance(tag, str)
                        and isinstance(confidence, str)
                        and TWO_DECIMAL_RE.fullmatch(confidence) is not None,
                        f"{record_id}: confidence must be a two-decimal string",
                    )
                    _require(
                        tag in positions,
                        f"{record_id}: unknown confidence tag: {tag}",
                    )
                    _require(
                        confidence == f"{float(row['scores'][positions[tag]]):.2f}",
                        f"{record_id}: confidence differs from float32 score",
                    )
                    names[category].append(tag)
            _validate_selected_payload(names, expected, schema, record_id=record_id)
    if "all_scores" in parsed_by_mode:
        for record_id, row, item in zip(expected_ids, rows, parsed_by_mode["all_scores"]):
            payload = item.get("scores")
            _require(
                isinstance(payload, Mapping) and list(payload) == schema["labels"],
                f"{record_id}: all-scores must contain 57 keys in schema order",
            )
            for index, tag in enumerate(schema["labels"]):
                value = payload[tag]
                _require(
                    isinstance(value, str) and TWO_DECIMAL_RE.fullmatch(value) is not None,
                    f"{record_id}: all-score for {tag} must be a two-decimal string",
                )
                expected_value = "0.00" if tag == UNSUPPORTED_TAG else f"{float(row['scores'][index]):.2f}"
                _require(value == expected_value, f"{record_id}: all-score differs at {tag}")
    cross_consistent = all(mode in parsed_by_mode for mode in files)
    if cross_consistent:
        selected_rows = parsed_by_mode["selected_only"]
        confidence_rows = parsed_by_mode["selected_with_confidence"]
        for selected, confidence in zip(selected_rows, confidence_rows):
            names = {
                category: [item["name"] for item in confidence["output"][category]]
                for category in EXPECTED_CATEGORIES
            }
            _require(
                names == selected["output"],
                f"{selected['record_id']}: selected modes disagree",
            )
    total_records = sum(records_by_mode.values())
    expected_records = len(rows) * len(available)
    return {
        "available_modes": available,
        "complete": len(available) == len(files),
        "records_by_mode": records_by_mode,
        "valid_records": total_records,
        "expected_records": expected_records,
        "json_validity_rate": _ratio(total_records, expected_records),
        "cross_mode_consistent": cross_consistent,
    }


def _source_bucket(row: Mapping[str, Any]) -> str:
    sources = set(row.get("sources", []))
    if sources == {"jd_complete23", "dictionary_v4"}:
        return "mixed"
    if "dictionary_v4" in sources:
        return "dictionary"
    return "jd"


def _representative_metadata(
    row: Mapping[str, Any],
    final: Sequence[int],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    pn_false_negative: list[str] = []
    pn_false_positive: list[str] = []
    pu_positive: list[str] = []
    pu_hit: list[str] = []
    for index, tag in enumerate(schema["labels"]):
        mode = schema["label_training_modes"][tag]
        if mode == "pn" and bool(row["known_mask"][index]):
            truth = int(float(row["labels"][index]) == 1.0)
            if truth and not final[index]:
                pn_false_negative.append(tag)
            elif not truth and final[index]:
                pn_false_positive.append(tag)
        if bool(row["pu_positive_mask"][index]):
            pu_positive.append(tag)
            if final[index]:
                pu_hit.append(tag)
    pu_missed = [tag for tag in pu_positive if tag not in pu_hit]
    outcome = (
        "success"
        if not pn_false_negative and not pn_false_positive and not pu_missed
        else "error"
    )
    return {
        "record_id": row["record_id"],
        "outcome": outcome,
        "source_bucket": _source_bucket(row),
        "sources": list(row.get("sources", [])),
        "image_path": row.get("image_path"),
        "image_sha256": row.get("image_sha256"),
        "selected": _render_selected(row["scores"], final, schema),
        "truth_summary": {
            "pn_known_positive": [
                tag
                for index, tag in enumerate(schema["labels"])
                if bool(row["known_mask"][index]) and float(row["labels"][index]) == 1.0
            ],
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
        "pu_positive_hits": {"hit": pu_hit, "missed": pu_missed},
    }


def _representative_key(row: Mapping[str, Any]) -> tuple[str, str]:
    record_id = str(row["record_id"])
    return hashlib.sha256(f"20260717:{record_id}".encode()).hexdigest(), record_id


def _choose_with_source_coverage(
    rows: Sequence[Mapping[str, Any]], count: int
) -> list[Mapping[str, Any]]:
    ordered = sorted(rows, key=_representative_key)
    selected: list[Mapping[str, Any]] = []
    selected_ids: set[str] = set()
    for bucket in ("jd", "dictionary", "mixed"):
        if len(selected) >= count:
            break
        candidate = next(
            (row for row in ordered if row["source_bucket"] == bucket), None
        )
        if candidate is not None:
            selected.append(candidate)
            selected_ids.add(str(candidate["record_id"]))
    for row in ordered:
        if len(selected) >= count:
            break
        if str(row["record_id"]) not in selected_ids:
            selected.append(row)
            selected_ids.add(str(row["record_id"]))
    return selected


def select_representatives(
    rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    count: int = 6,
) -> list[dict[str, Any]]:
    _require(count > 0, "representative count must be positive")
    _, final_rows = _prediction_vectors(rows, thresholds, schema)
    metadata = [
        _representative_metadata(row, final, schema)
        for row, final in zip(rows, final_rows)
    ]
    success_target = count // 2
    error_target = count - success_target
    successes = [row for row in metadata if row["outcome"] == "success"]
    errors = [row for row in metadata if row["outcome"] == "error"]
    selected = list(_choose_with_source_coverage(successes, success_target))
    selected.extend(_choose_with_source_coverage(errors, error_target))
    return selected


def verify_sealed_inventory(delivery_dir: Path | str) -> dict[str, Any]:
    root = Path(delivery_dir)
    checksum_path = root / "SHA256SUMS"
    _require(checksum_path.is_file(), "sealed SHA256SUMS is missing")
    declared: dict[str, str] = {}
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise AuditContractError("unable to read sealed SHA256SUMS") from exc
    for line_number, line in enumerate(lines, 1):
        if not line:
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise AuditContractError(
                f"invalid sealed checksum line {line_number}"
            ) from exc
        relative_path = Path(relative)
        _require(
            len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest)
            and bool(relative)
            and not relative_path.is_absolute()
            and ".." not in relative_path.parts
            and relative not in declared,
            f"invalid sealed checksum entry at line {line_number}",
        )
        declared[relative] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != checksum_path
    }
    _require(
        set(declared) == actual,
        "sealed inventory differs from declared files",
    )
    for relative, expected in declared.items():
        actual_digest = sha256_file(root / relative)
        _require(
            actual_digest == expected,
            f"sealed checksum mismatch: {relative}",
        )
    return {
        "complete": True,
        "verified_files": len(declared),
        "sha256s_file_sha256": sha256_file(checksum_path),
    }


def audit_posttrain(posttrain_dir: Path | str) -> dict[str, Any]:
    root = Path(posttrain_dir)
    preflight = _load_json(root / "preflight_contract.json", name="posttrain preflight")
    final_report = _load_json(root / "final_report.json", name="posttrain final report")
    _require(isinstance(preflight, Mapping), "posttrain preflight must be an object")
    _require(
        preflight.get("leakage_passed") is True,
        "posttrain preflight did not preserve leakage pass",
    )
    leakage_counts = preflight.get("leakage_counts")
    _require(
        isinstance(leakage_counts, Mapping)
        and all(int(value) == 0 for value in leakage_counts.values()),
        "posttrain preflight contains leakage collisions",
    )
    _require(isinstance(final_report, Mapping), "posttrain final report must be an object")
    status = final_report.get("status")
    _require(status in {"success", "partial", "fail"}, "posttrain final status is invalid")
    if status == "success":
        _require(
            final_report.get("customer_ready") is True,
            "successful posttrain report must be customer ready",
        )
    _require(
        final_report.get("completed_before_deadline") is not False,
        "posttrain report missed the fixed deadline",
    )
    reproduction_path = root / "reproduction_result.json"
    if status == "fail" and not reproduction_path.is_file():
        return {
            "complete": True,
            "status": status,
            "customer_ready": False,
            "completed_before_deadline": final_report.get(
                "completed_before_deadline", True
            ),
            "reproduction_outputs_verified": 0,
            "preflight_sha256": sha256_file(root / "preflight_contract.json"),
            "final_report_sha256": sha256_file(root / "final_report.json"),
            "reproduction_result_sha256": None,
        }
    reproduction = _load_json(
        reproduction_path, name="posttrain reproduction result"
    )
    _require(
        isinstance(reproduction, Mapping),
        "posttrain reproduction result must be an object",
    )
    verified = 0
    for path_key, sha_key in (
        ("reproduced_float32_path", "reproduced_float32_sha256"),
        ("reproduced_selected_only_path", "reproduced_selected_only_sha256"),
    ):
        output_path = Path(str(reproduction.get(path_key) or ""))
        expected = reproduction.get(sha_key)
        _require(output_path.is_file(), f"posttrain reproduction output is missing: {path_key}")
        _require(
            isinstance(expected, str) and sha256_file(output_path) == expected,
            f"posttrain reproduction hash mismatch: {path_key}",
        )
        verified += 1
    return {
        "complete": True,
        "status": status,
        "customer_ready": bool(final_report.get("customer_ready")),
        "completed_before_deadline": final_report.get("completed_before_deadline", True),
        "reproduction_outputs_verified": verified,
        "preflight_sha256": sha256_file(root / "preflight_contract.json"),
        "final_report_sha256": sha256_file(root / "final_report.json"),
        "reproduction_result_sha256": sha256_file(reproduction_path),
    }


def audit_reproduction_bundle(
    posttrain_dir: Path | str,
    candidate_dir: Path | str,
    *,
    evaluation_dir: Path | str,
    test_rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    schema: Mapping[str, Any],
    expected_records: int = 32,
) -> dict[str, Any]:
    """Independently verify candidate references and exact replay outputs."""

    posttrain = Path(posttrain_dir)
    candidate = Path(candidate_dir)
    _require(expected_records > 0, "reproduction expected_records must be positive")
    result_path = posttrain / "reproduction_result.json"
    result = _load_json(result_path, name="posttrain reproduction result")
    _require(isinstance(result, Mapping), "posttrain reproduction result must be an object")

    candidate_files = {
        "candidate_weights_sha256": candidate / "lora_and_classifier.safetensors",
        "candidate_model_config_sha256": candidate / "model_config.json",
        "candidate_infer_sha256": candidate / "infer.py",
    }
    for key, path in candidate_files.items():
        _require(path.is_file(), f"candidate file is missing: {path.name}")
        _require(
            result.get(key) == sha256_file(path),
            f"candidate hash mismatch: {key}",
        )

    commands = result.get("commands")
    infer_path = str(candidate_files["candidate_infer_sha256"])
    _require(
        isinstance(commands, list)
        and all(isinstance(command, str) and infer_path in command for command in commands)
        and any("verification_float32" in command for command in commands)
        and any("selected_only" in command for command in commands),
        "reproduction commands must bind candidate infer.py and both required modes",
    )
    environment = result.get("environment")
    environment_fields = (
        "gpu",
        "cuda",
        "pytorch",
        "transformers",
        "peft",
        "safetensors",
        "pillow",
    )
    _require(
        isinstance(environment, Mapping)
        and all(isinstance(environment.get(key), str) and environment.get(key) for key in environment_fields),
        "reproduction environment is incomplete",
    )

    verification_dir = candidate / "verification"
    reference_paths = {
        "verification_32_manifest.jsonl": verification_dir
        / "verification_32_manifest.jsonl",
        "reference_32_float32.jsonl": verification_dir / "reference_32_float32.jsonl",
        "reference_32_selected_only.jsonl": verification_dir
        / "reference_32_selected_only.jsonl",
    }
    candidate_verification = _load_json(
        candidate / "VERIFICATION.json", name="candidate verification"
    )
    _require(
        isinstance(candidate_verification, Mapping)
        and candidate_verification.get("status") == "pending_reproduction",
        "candidate verification status is invalid",
    )
    provenance = candidate_verification.get("provenance")
    _require(
        isinstance(provenance, Mapping)
        and provenance.get("weights_sha256")
        == result["candidate_weights_sha256"],
        "candidate verification weights provenance mismatch",
    )
    references = candidate_verification.get("references")
    _require(isinstance(references, Mapping), "candidate reference hashes are missing")
    for name, path in reference_paths.items():
        _require(
            references.get(name) == sha256_file(path),
            f"candidate verification reference drifted: {name}",
        )
    formal_verification = Path(evaluation_dir) / "verification"
    for name, candidate_path in reference_paths.items():
        formal_path = formal_verification / name
        _require(
            formal_path.is_file()
            and sha256_file(candidate_path) == sha256_file(formal_path),
            f"candidate reference differs from formal evaluation verification: {name}",
        )

    output_paths: dict[str, Path] = {}
    for path_key, sha_key in (
        ("reproduced_float32_path", "reproduced_float32_sha256"),
        ("reproduced_selected_only_path", "reproduced_selected_only_sha256"),
    ):
        output_path = Path(str(result.get(path_key) or ""))
        _require(output_path.is_file(), f"posttrain reproduction output is missing: {path_key}")
        _require(
            result.get(sha_key) == sha256_file(output_path),
            f"posttrain reproduction hash mismatch: {path_key}",
        )
        output_paths[path_key] = output_path

    manifest = _load_jsonl(reference_paths["verification_32_manifest.jsonl"], name="candidate verification manifest")
    reference_float = _load_jsonl(reference_paths["reference_32_float32.jsonl"], name="candidate float32 reference")
    reference_selected = _load_jsonl(reference_paths["reference_32_selected_only.jsonl"], name="candidate selected reference")
    reproduced_float = _load_jsonl(output_paths["reproduced_float32_path"], name="reproduced float32")
    reproduced_selected = _load_jsonl(output_paths["reproduced_selected_only_path"], name="reproduced selected-only")
    collections = (
        manifest,
        reference_float,
        reference_selected,
        reproduced_float,
        reproduced_selected,
    )
    _require(
        all(len(rows) == expected_records for rows in collections),
        f"reproduction must contain exactly {expected_records} records",
    )
    ids = [row.get("record_id") for row in manifest]
    _require(
        all(isinstance(record_id, str) and record_id for record_id in ids)
        and len(set(ids)) == expected_records,
        "reproduction manifest record identifiers are invalid",
    )
    for rows in collections[1:]:
        _require(
            [row.get("record_id") for row in rows] == ids,
            "reproduction record order differs from verification manifest",
        )

    test_by_id = {str(row.get("record_id")): row for row in test_rows}
    test_index = {
        str(row.get("record_id")): index for index, row in enumerate(test_rows)
    }
    _require(
        len(test_by_id) == len(test_rows),
        "formal test predictions contain duplicate record identifiers",
    )
    for manifest_row, reference_row, selected_row in zip(
        manifest, reference_float, reference_selected
    ):
        record_id = str(manifest_row["record_id"])
        _require(
            record_id in test_by_id,
            f"{record_id}: verification record is absent from formal test predictions",
        )
        formal_row = test_by_id[record_id]
        if "test_manifest_index" in manifest_row:
            _require(
                manifest_row.get("test_manifest_index") == test_index[record_id],
                f"{record_id}: verification test manifest index differs",
            )
        for field in ("image_path", "image_sha256"):
            _require(
                manifest_row.get(field) == formal_row.get(field),
                f"{record_id}: verification manifest differs from formal test at {field}",
            )
        for field in (
            "record_id",
            *EVIDENCE_FIELDS,
            "checkpoint_sha256",
            "scores",
            "width",
            "height",
            "aspect_ratio",
        ):
            if field in reference_row or field in formal_row:
                _require(
                    reference_row.get(field) == formal_row.get(field),
                    f"{record_id}: float32 reference differs from formal test prediction at {field}",
                )
        _, final_prediction = _prediction_vectors(
            [formal_row], thresholds, schema
        )
        expected_selected = _render_selected(
            formal_row["scores"], final_prediction[0], schema
        )
        _require(
            selected_row.get("output") == expected_selected,
            f"{record_id}: selected reference differs from formal test prediction",
        )

    score_values = 0
    for manifest_row, expected, actual in zip(manifest, reference_float, reproduced_float):
        record_id = str(manifest_row["record_id"])
        image_sha256 = manifest_row.get("image_sha256")
        _require(_is_sha256(image_sha256), f"{record_id}: reproduction image SHA256 is invalid")
        _require(
            expected.get("image_sha256") == image_sha256
            and actual.get("image_sha256") == image_sha256,
            f"{record_id}: reproduction image SHA256 differs",
        )
        expected_scores = _validate_scores(expected.get("scores"), record_id=record_id)
        actual_scores = _validate_scores(actual.get("scores"), record_id=record_id)
        _require(
            actual_scores == expected_scores,
            f"{record_id}: reproduced float32 scores differ from candidate reference",
        )
        score_values += len(actual_scores)

    for manifest_row, expected, actual in zip(manifest, reference_selected, reproduced_selected):
        record_id = str(manifest_row["record_id"])
        _require(
            actual.get("image_sha256") == manifest_row.get("image_sha256"),
            f"{record_id}: selected reproduction image SHA256 differs",
        )
        _require(
            actual.get("output") == expected.get("output"),
            f"{record_id}: reproduced selected output differs from candidate reference",
        )

    return {
        "complete": True,
        "records": expected_records,
        "score_values": score_values,
        "probabilities_exact": True,
        "selected_outputs_exact": True,
        "image_sha256s_exact": True,
        "candidate_hashes_verified": len(candidate_files),
        "candidate_weights_sha256": result["candidate_weights_sha256"],
        "commands": list(commands),
        "environment": dict(environment),
        "result_sha256": sha256_file(result_path),
        "reproduced_float32_sha256": sha256_file(output_paths["reproduced_float32_path"]),
        "reproduced_selected_only_sha256": sha256_file(output_paths["reproduced_selected_only_path"]),
    }


def audit_packaged_output_modes(
    posttrain_dir: Path | str,
    schema: Mapping[str, Any],
    *,
    expected_records: int = 32,
) -> dict[str, Any]:
    """Verify the two customer-facing modes regenerated from replayed float32."""

    root = Path(posttrain_dir)
    mode_dir = root / "final_mode_verification"
    summary_path = mode_dir / "final_mode_verification.json"
    confidence_path = mode_dir / "reproduced_32_selected_with_confidence.jsonl"
    all_scores_path = mode_dir / "reproduced_32_all_scores.jsonl"
    summary = _load_json(summary_path, name="final mode verification summary")
    _require(isinstance(summary, Mapping), "final mode verification must be an object")
    reproduction = _load_json(
        root / "reproduction_result.json", name="posttrain reproduction result"
    )
    _require(isinstance(reproduction, Mapping), "reproduction result must be an object")
    candidate = root.parent / "delivery_candidate"
    candidate_infer = candidate / "infer.py"
    _require(candidate_infer.is_file(), "candidate infer.py is missing")
    _require(
        summary.get("candidate_infer_sha256") == sha256_file(candidate_infer)
        == reproduction.get("candidate_infer_sha256"),
        "final mode candidate infer SHA256 binding mismatch",
    )
    reproduction_path = root / "reproduction_result.json"
    _require(
        summary.get("reproduction_result_sha256")
        == sha256_file(reproduction_path),
        "final modes are not bound to the current reproduction result",
    )
    commands = summary.get("commands")
    _require(
        isinstance(commands, list)
        and len(commands) >= 2
        and all(
            isinstance(command, str)
            and str(candidate_infer) in command
            and "--scores-json" in command
            for command in commands
        )
        and any(
            "--mode selected_with_confidence" in command
            or "--mode selected-with-confidence" in command
            for command in commands
        )
        and any(
            "--mode all_scores" in command or "--mode all-scores" in command
            for command in commands
        ),
        "final mode commands must execute candidate infer.py --scores-json for both modes",
    )
    environment = summary.get("environment")
    environment_fields = (
        "gpu",
        "cuda",
        "pytorch",
        "transformers",
        "peft",
        "safetensors",
        "pillow",
    )
    _require(
        isinstance(environment, Mapping)
        and all(isinstance(environment.get(key), str) and environment.get(key) for key in environment_fields),
        "final mode execution environment is incomplete",
    )
    float_path = Path(str(reproduction.get("reproduced_float32_path") or ""))
    selected_path = Path(str(reproduction.get("reproduced_selected_only_path") or ""))
    _require(
        summary.get("source_float32_sha256") == sha256_file(float_path)
        and summary.get("source_selected_only_sha256") == sha256_file(selected_path),
        "final modes are not bound to current replay sources",
    )
    _require(
        summary.get("selected_with_confidence_sha256") == sha256_file(confidence_path)
        and summary.get("all_scores_sha256") == sha256_file(all_scores_path),
        "final mode output SHA256 mismatch",
    )
    required_true = (
        "float32_json_roundtrip_exact",
        "selected_only_reformatted_exact",
        "selected_with_confidence_names_exact",
        "confidence_two_decimal",
        "all_scores_exactly_57",
        "all_scores_schema_order_exact",
        "all_scores_two_decimal",
        "unsupported_假两件_fixed_0.00",
    )
    _require(
        summary.get("status") == "success"
        and int(summary.get("records", -1)) == expected_records
        and int(summary.get("score_values", -1)) == expected_records * 57
        and all(summary.get(key) is True for key in required_true),
        "final mode verification summary is incomplete",
    )

    float_rows = _load_jsonl(float_path, name="replayed float32")
    selected_rows = _load_jsonl(selected_path, name="replayed selected-only")
    confidence_rows = _load_jsonl(confidence_path, name="replayed selected-with-confidence")
    all_score_rows = _load_jsonl(all_scores_path, name="replayed all-scores")
    _require(
        all(
            len(rows) == expected_records
            for rows in (float_rows, selected_rows, confidence_rows, all_score_rows)
        ),
        f"packaged output modes must contain exactly {expected_records} records",
    )
    ids = [row.get("record_id") for row in float_rows]
    for rows in (selected_rows, confidence_rows, all_score_rows):
        _require(
            [row.get("record_id") for row in rows] == ids,
            "packaged output mode record order differs from float32 replay",
        )
    positions = {tag: index for index, tag in enumerate(schema["labels"])}
    for float_row, selected_row, confidence_row, all_score_row in zip(
        float_rows, selected_rows, confidence_rows, all_score_rows
    ):
        record_id = str(float_row.get("record_id"))
        scores = _validate_scores(float_row.get("scores"), record_id=record_id)
        image_sha256 = float_row.get("image_sha256")
        image_path = float_row.get("image_path")
        _require(
            _is_sha256(image_sha256)
            and confidence_row.get("image_sha256") == image_sha256
            and all_score_row.get("image_sha256") == image_sha256
            and confidence_row.get("image_path") == image_path
            and all_score_row.get("image_path") == image_path,
            f"{record_id}: packaged mode image evidence differs",
        )
        selected = selected_row.get("output")
        _require(
            isinstance(selected, Mapping) and tuple(selected) == EXPECTED_CATEGORIES,
            f"{record_id}: replayed selected-only structure is invalid",
        )
        confidence = confidence_row.get("output")
        _require(
            isinstance(confidence, Mapping) and tuple(confidence) == EXPECTED_CATEGORIES,
            f"{record_id}: packaged confidence categories are invalid",
        )
        confidence_names: dict[str, list[str]] = {}
        for category, items in confidence.items():
            _require(isinstance(items, list), f"{record_id}: confidence values must be arrays")
            confidence_names[category] = []
            for item in items:
                _require(
                    isinstance(item, Mapping) and tuple(item) == ("name", "confidence"),
                    f"{record_id}: confidence item must contain name and confidence",
                )
                tag = item["name"]
                value = item["confidence"]
                _require(tag in positions, f"{record_id}: unknown confidence tag: {tag}")
                _require(
                    isinstance(value, str)
                    and TWO_DECIMAL_RE.fullmatch(value) is not None
                    and value == f"{scores[positions[tag]]:.2f}",
                    f"{record_id}: packaged confidence differs from float32 score",
                )
                confidence_names[category].append(tag)
        _require(
            confidence_names == selected,
            f"{record_id}: selected-with-confidence names differ from selected-only",
        )
        output = all_score_row.get("output")
        _require(
            isinstance(output, Mapping) and tuple(output) == ("scores",),
            f"{record_id}: all-scores output wrapper is invalid",
        )
        score_map = output["scores"]
        _require(
            isinstance(score_map, Mapping) and list(score_map) == schema["labels"],
            f"{record_id}: all-scores must contain 57 schema-ordered labels",
        )
        for index, tag in enumerate(schema["labels"]):
            value = score_map[tag]
            expected = "0.00" if tag == UNSUPPORTED_TAG else f"{scores[index]:.2f}"
            _require(
                isinstance(value, str)
                and TWO_DECIMAL_RE.fullmatch(value) is not None
                and value == expected,
                f"{record_id}: packaged all-score differs at {tag}",
            )
    return {
        "complete": True,
        "records": expected_records,
        "score_values": expected_records * 57,
        "valid_records": expected_records * 2,
        "available_modes": ["selected_with_confidence", "all_scores"],
        "selected_with_confidence_names_exact": True,
        "confidence_two_decimal": True,
        "all_scores_exactly_57": True,
        "all_scores_schema_order_exact": True,
        "all_scores_two_decimal": True,
        "unsupported_假两件_fixed_0.00": True,
        "sealed_delivery_verification_status": summary.get(
            "sealed_delivery_verification_status"
        ),
        "sealed_delivery_customer_ready": summary.get(
            "sealed_delivery_customer_ready"
        ),
        "summary_sha256": sha256_file(summary_path),
        "candidate_infer_sha256": sha256_file(candidate_infer),
        "reproduction_result_sha256": sha256_file(reproduction_path),
        "commands": list(commands),
        "environment": dict(environment),
        "selected_with_confidence_sha256": sha256_file(confidence_path),
        "all_scores_sha256": sha256_file(all_scores_path),
    }


def _validate_threshold_contract(
    path: Path,
    schema: Mapping[str, Any],
    *,
    validation_manifest_sha256: str,
    expected_validation_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _load_json(path, name="frozen thresholds")
    _require(isinstance(payload, dict), "thresholds must be an object")
    _require(
        payload.get("schema_sha256") == schema.get("schema_sha256"),
        "threshold schema SHA256 mismatch",
    )
    _require(
        payload.get("schema_version") == schema.get("schema_version"),
        "threshold schema version mismatch",
    )
    _require(
        payload.get("validation_manifest_sha256") == validation_manifest_sha256,
        "threshold validation manifest SHA256 mismatch",
    )
    _require(
        int(payload.get("calibration_records", -1)) == expected_validation_count,
        "threshold calibration count mismatch",
    )
    _require(
        _is_sha256(payload.get("checkpoint_sha256")),
        "threshold checkpoint SHA256 is invalid",
    )
    _thresholds_map(payload, schema)
    labels = payload["labels"]
    fallback = [
        tag
        for tag in schema["labels"]
        if schema["label_training_modes"][tag] == "pu"
        and labels[tag].get("status") == "fallback_insufficient_support"
    ]
    return payload, {
        "pu_fallback_count": len(fallback),
        "pu_fallback_labels": fallback,
        "thresholds_sha256": sha256_file(path),
    }


def _validate_threshold_support(
    thresholds: Mapping[str, Any],
    validation_rows: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        math.isclose(
            float(thresholds.get("fallback_threshold", -1.0)),
            0.5,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "frozen fallback threshold must be exactly 0.5",
    )
    entries = thresholds.get("labels")
    _require(isinstance(entries, Mapping), "threshold label entries are missing")
    fallback_labels: list[str] = []
    support_by_label: dict[str, dict[str, int]] = {}
    for index, tag in enumerate(schema["labels"]):
        mode = schema["label_training_modes"][tag]
        entry = entries[tag]
        _require(
            isinstance(entry, Mapping) and entry.get("mode") == mode,
            f"threshold mode mismatch at {tag}",
        )
        status = entry.get("status")
        if mode == "unsupported":
            _require(
                status == "disabled_unsupported"
                and entry.get("threshold") is None
                and entry.get("support") == {},
                f"unsupported threshold contract differs at {tag}",
            )
            continue
        threshold = entry.get("threshold")
        _require(
            isinstance(threshold, (int, float))
            and not isinstance(threshold, bool)
            and math.isfinite(float(threshold))
            and 0.0 <= float(threshold) <= 1.0,
            f"threshold is invalid at {tag}",
        )
        if mode == "pn":
            expected_support = {
                "known_positive": sum(
                    bool(row["known_mask"][index])
                    and float(row["labels"][index]) == 1.0
                    for row in validation_rows
                ),
                "known_negative": sum(
                    bool(row["known_mask"][index])
                    and float(row["labels"][index]) == 0.0
                    for row in validation_rows
                ),
            }
            _require(
                status in {"calibrated", "fallback_insufficient_support"},
                f"PN threshold status is invalid at {tag}",
            )
        else:
            expected_support = {
                "positive": sum(
                    bool(row["pu_positive_mask"][index])
                    for row in validation_rows
                ),
                "unlabeled": sum(
                    not bool(row["known_mask"][index])
                    and not bool(row["pu_positive_mask"][index])
                    for row in validation_rows
                ),
            }
            _require(
                status == "fallback_insufficient_support",
                f"all 20 PU labels must use fallback_insufficient_support: {tag}",
            )
            _require(
                math.isclose(float(threshold), 0.5, rel_tol=0.0, abs_tol=0.0),
                f"PU fallback threshold must be exactly 0.5 at {tag}",
            )
            fallback_labels.append(tag)
        _require(
            entry.get("support") == expected_support,
            f"threshold validation support differs at {tag}",
        )
        support_by_label[tag] = expected_support
    _require(
        len(fallback_labels) == EXPECTED_MODE_COUNTS["pu"],
        "all 20 PU labels must use fallback threshold 0.5",
    )
    return {
        "pu_fallback_count": len(fallback_labels),
        "pu_fallback_labels": fallback_labels,
        "support_by_label": support_by_label,
    }


def _require_close(actual: Any, expected: float, name: str) -> None:
    _require(
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12),
        f"reported metric differs from independent recomputation: {name}",
    )


def _reported_validation_values(report: Mapping[str, Any]) -> dict[str, float]:
    try:
        metrics = report["validation_metrics"]
        final = metrics["final_format"]
        overall = final["overall_36pn"]
        dictionary = metrics["dictionary_all_positive"]["final_format"]
        return {
            "known_micro_f1": float(overall["micro"]["f1"]),
            "jd23_micro_f1": float(final["jd23_clean"]["micro"]["f1"]),
            "macro_f1": float(overall["macro"]["f1_both_class_labels"]),
            "dictionary_positive_macro_recall": float(dictionary["macro_positive_recall"]),
            "trusted_negative_specificity": float(
                overall.get("trusted_negatives", overall["micro"])["specificity"]
            ),
            "json_validity_rate": 1.0,
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditContractError("validation metrics are incomplete") from exc


def _require_same_scalar(actual: Any, expected: Any, name: str) -> None:
    if expected is None:
        _require(actual is None, f"reported value differs at {name}")
    elif isinstance(expected, bool):
        _require(actual is expected, f"reported value differs at {name}")
    elif isinstance(expected, (int, float)):
        _require_close(actual, float(expected), name)
    else:
        _require(actual == expected, f"reported value differs at {name}")


def _crosscheck_pn_slice(
    actual: Any, expected: Mapping[str, Any], *, name: str
) -> None:
    _require(isinstance(actual, Mapping), f"reported PN slice is missing: {name}")
    _require_same_scalar(actual.get("record_count"), expected["record_count"], f"{name}.record_count")
    count_metric_keys = (
        "tp",
        "fp",
        "fn",
        "tn",
        "known_cells",
        "known_positive",
        "known_negative",
        "precision",
        "recall",
        "f1",
        "specificity",
        "fpr",
        "accuracy",
    )
    actual_micro = actual.get("micro")
    _require(isinstance(actual_micro, Mapping), f"reported micro metrics are missing: {name}")
    for key in count_metric_keys:
        _require_same_scalar(actual_micro.get(key), expected["micro"][key], f"{name}.micro.{key}")
    macro_keys = (
        "precision",
        "recall",
        "f1",
        "specificity",
        "fpr",
        "accuracy",
        "labels_evaluated",
        "labels_with_both_classes",
        "f1_both_class_labels",
        "positive_labels_evaluated",
        "positive_labels_macro_recall",
    )
    actual_macro = actual.get("macro")
    _require(isinstance(actual_macro, Mapping), f"reported macro metrics are missing: {name}")
    for key in macro_keys:
        _require_same_scalar(actual_macro.get(key), expected["macro"][key], f"{name}.macro.{key}")
    actual_per_label = actual.get("per_label")
    expected_per_label = expected["per_label"]
    _require(
        isinstance(actual_per_label, Mapping)
        and list(actual_per_label) == list(expected_per_label),
        f"reported per-label inventory differs: {name}",
    )
    for tag, expected_item in expected_per_label.items():
        actual_item = actual_per_label[tag]
        _require(isinstance(actual_item, Mapping), f"reported per-label metrics are invalid: {name}.{tag}")
        for key in count_metric_keys:
            _require_same_scalar(actual_item.get(key), expected_item[key], f"{name}.per_label.{tag}.{key}")
    for key in ("exact_match", "exact_match_rows"):
        _require_same_scalar(actual.get(key), expected[key], f"{name}.{key}")
    trusted = actual.get("trusted_negatives")
    _require(isinstance(trusted, Mapping), f"reported trusted negatives are missing: {name}")
    for key in ("tn", "fp", "specificity", "fpr"):
        _require_same_scalar(trusted.get(key), expected["trusted_negatives"][key], f"{name}.trusted_negatives.{key}")


def _crosscheck_pu_view(
    actual: Any, expected: Mapping[str, Any], *, name: str
) -> None:
    _require(isinstance(actual, Mapping), f"reported PU metrics are missing: {name}")
    actual_per_label = actual.get("per_label")
    expected_per_label = expected["per_label"]
    _require(
        isinstance(actual_per_label, Mapping)
        and list(actual_per_label) == list(expected_per_label),
        f"reported PU label inventory differs: {name}",
    )
    scalar_keys = (
        "positive_count",
        "unlabeled_count",
        "positive_recall",
        "positive_vs_unlabeled_concordance",
        "all_coverage",
        "unlabeled_coverage",
        "selected_positive_count",
        "selected_unlabeled_count",
        "threshold",
        "calibration_status",
    )
    for tag, expected_item in expected_per_label.items():
        actual_item = actual_per_label[tag]
        _require(isinstance(actual_item, Mapping), f"reported PU label metrics are invalid: {tag}")
        for key in scalar_keys:
            _require_same_scalar(actual_item.get(key), expected_item[key], f"{name}.{tag}.{key}")
        for group in ("positive_score_quantiles", "unlabeled_score_quantiles"):
            actual_quantiles = actual_item.get(group)
            _require(isinstance(actual_quantiles, Mapping), f"reported PU quantiles are missing: {name}.{tag}.{group}")
            for key in ("p10", "p50", "p90"):
                _require_same_scalar(actual_quantiles.get(key), expected_item[group][key], f"{name}.{tag}.{group}.{key}")
    summary = actual.get("summary")
    _require(isinstance(summary, Mapping), f"reported PU summary is missing: {name}")
    for key, expected_value in expected["summary"].items():
        _require_same_scalar(summary.get(key), expected_value, f"{name}.summary.{key}")


def _crosscheck_view(actual: Any, expected: Mapping[str, Any], *, name: str) -> None:
    _require(isinstance(actual, Mapping), f"reported prediction view is missing: {name}")
    for slice_name in ("overall_36pn", "jd23_clean", "dictionary_pn_clean"):
        _crosscheck_pn_slice(
            actual.get(slice_name), expected[slice_name], name=f"{name}.{slice_name}"
        )
    mixed = actual.get("mixed_exact_audit")
    _require(isinstance(mixed, Mapping), f"reported mixed audit is missing: {name}")
    for key, expected_value in expected["mixed_exact_audit"].items():
        _require_same_scalar(mixed.get(key), expected_value, f"{name}.mixed_exact_audit.{key}")
    _crosscheck_pu_view(actual.get("pu"), expected["pu"], name=f"{name}.pu")


def _crosscheck_dictionary_view(
    actual: Any, expected: Mapping[str, Any], *, name: str
) -> None:
    _require(isinstance(actual, Mapping), f"reported dictionary metrics are missing: {name}")
    for key in (
        "dictionary_records",
        "labels_with_positive_support",
        "macro_positive_recall",
        "micro_positive_recall",
        "positive_support",
        "selected_positive",
    ):
        _require_same_scalar(actual.get(key), expected[key], f"{name}.{key}")
    actual_per_label = actual.get("per_label")
    _require(
        isinstance(actual_per_label, Mapping)
        and list(actual_per_label) == list(expected["per_label"]),
        f"reported dictionary label inventory differs: {name}",
    )
    for tag, expected_item in expected["per_label"].items():
        item = actual_per_label[tag]
        _require(isinstance(item, Mapping), f"reported dictionary item is invalid: {name}.{tag}")
        for key in ("mode", "supervision", "positive_support", "selected_positive", "recall"):
            _require_same_scalar(item.get(key), expected_item[key], f"{name}.{tag}.{key}")


def _crosscheck_format_loss(actual: Any, expected: Mapping[str, Any]) -> None:
    _require(isinstance(actual, Mapping), "reported format_constraint_loss is missing")
    for group in ("raw_multi_selected_records", "observed_multi_positive_records"):
        _require(actual.get(group) == expected[group], f"reported format loss differs at {group}")
    suppressed = actual.get("formatter_suppressed_predictions")
    _require(isinstance(suppressed, Mapping), "reported formatter suppression is missing")
    _require(
        suppressed.get("by_subcategory") == expected["formatter_suppressed_predictions"]["by_subcategory"]
        and suppressed.get("by_tag") == expected["formatter_suppressed_predictions"]["by_tag"],
        "reported formatter suppression inventory differs",
    )
    _require_same_scalar(
        suppressed.get("total"),
        expected["formatter_suppressed_predictions"]["total"],
        "format_constraint_loss.suppressed.total",
    )
    for key in (
        "observed_positive_count",
        "oracle_final_recall_ceiling",
        "contract_forced_false_negatives",
    ):
        _require_same_scalar(actual.get(key), expected[key], f"format_constraint_loss.{key}")
    hit_rate = actual.get("subcategory_hit_rate")
    _require(isinstance(hit_rate, Mapping) and list(hit_rate) == list(expected["subcategory_hit_rate"]), "reported subcategory hit-rate inventory differs")
    for subcategory, expected_item in expected["subcategory_hit_rate"].items():
        item = hit_rate[subcategory]
        for key, expected_value in expected_item.items():
            _require_same_scalar(item.get(key), expected_value, f"format_constraint_loss.{subcategory}.{key}")


def _crosscheck_report(
    report_path: Path,
    *,
    schema_path: Path,
    schema: Mapping[str, Any],
    validation_manifest_path: Path,
    test_manifest_path: Path,
    validation_predictions_path: Path,
    test_predictions_path: Path,
    thresholds_path: Path,
    validation_metrics: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
    classification: Mapping[str, Any],
    expected_validation_count: int,
    expected_test_count: int,
    threshold_audit: Mapping[str, Any],
) -> dict[str, Any]:
    report = _load_json(report_path, name="evaluation report")
    _require(isinstance(report, Mapping), "evaluation report must be an object")
    for split, expected in (
        ("validation", expected_validation_count),
        ("test", expected_test_count),
    ):
        value = report.get(split)
        _require(
            isinstance(value, Mapping)
            and value.get("complete") is True
            and int(value.get("expected", -1)) == expected
            and int(value.get("predicted", -1)) == expected,
            f"evaluation {split} coverage is incomplete",
        )
    provenance = report.get("provenance")
    _require(isinstance(provenance, Mapping), "evaluation provenance is missing")
    expected_provenance = {
        "schema_sha256": schema["schema_sha256"],
        "schema_file_sha256": sha256_file(schema_path),
        "validation_manifest_sha256": sha256_file(validation_manifest_path),
        "test_manifest_sha256": sha256_file(test_manifest_path),
        "validation_predictions_sha256": sha256_file(validation_predictions_path),
        "predictions_sha256": sha256_file(test_predictions_path),
        "thresholds_sha256": sha256_file(thresholds_path),
        "checkpoint_sha256": str(
            _load_json(thresholds_path, name="frozen thresholds")["checkpoint_sha256"]
        ),
    }
    for key, expected in expected_provenance.items():
        _require(
            provenance.get(key) == expected,
            f"evaluation provenance mismatch at {key}",
        )
    reported_classification = report.get("classification")
    _require(
        isinstance(reported_classification, Mapping)
        and reported_classification.get("verdict") == classification["verdict"],
        "evaluation verdict differs from independent classification",
    )
    _require(
        report.get("status") == classification["verdict"],
        "evaluation report status differs from independent classification",
    )
    reported_values = reported_classification.get("values")
    _require(isinstance(reported_values, Mapping), "evaluation performance values are missing")
    for key, expected in test_metrics["values"].items():
        _require_close(reported_values.get(key), float(expected), f"test.{key}")
    _require(
        reported_classification.get("success_gates")
        == classification["success_gates"],
        "evaluation success gates differ from independent classification",
    )
    performance = report.get("performance")
    _require(
        isinstance(performance, Mapping)
        and performance.get("verdict_basis") == "final_format",
        "evaluation performance contract is incomplete",
    )
    for view_name, expected_values in (
        ("raw_thresholded", test_metrics["raw_values"]),
        ("final_format", test_metrics["values"]),
        ("final_minus_raw", test_metrics["performance"]["final_minus_raw"]),
    ):
        actual_values = performance.get(view_name)
        _require(
            isinstance(actual_values, Mapping),
            f"evaluation performance view is missing: {view_name}",
        )
        for key, expected in expected_values.items():
            _require_close(actual_values.get(key), float(expected), f"performance.{view_name}.{key}")
    _crosscheck_view(
        report.get("raw_thresholded"),
        test_metrics["raw_thresholded"],
        name="test.raw_thresholded",
    )
    _crosscheck_view(
        report.get("final_format"),
        test_metrics["final_format"],
        name="test.final_format",
    )
    dictionary_views = report.get("dictionary_all_positive")
    _require(
        isinstance(dictionary_views, Mapping),
        "evaluation dictionary views are missing",
    )
    for view_name in ("raw_thresholded", "final_format"):
        _crosscheck_dictionary_view(
            dictionary_views.get(view_name),
            test_metrics["dictionary_all_positive_views"][view_name],
            name=f"test.dictionary_all_positive.{view_name}",
        )
    _crosscheck_format_loss(
        report.get("format_constraint_loss"), test_metrics["format_constraint_loss"]
    )
    for key, expected in validation_metrics["values"].items():
        _require_close(
            _reported_validation_values(report).get(key),
            float(expected),
            f"validation.{key}",
        )
    reported_validation = report.get("validation_metrics")
    _require(
        isinstance(reported_validation, Mapping),
        "validation metric views are missing",
    )
    for view_name in ("raw_thresholded", "final_format"):
        _crosscheck_view(
            reported_validation.get(view_name),
            validation_metrics[view_name],
            name=f"validation.{view_name}",
        )
    validation_dictionary = reported_validation.get("dictionary_all_positive")
    _require(
        isinstance(validation_dictionary, Mapping),
        "validation dictionary views are missing",
    )
    for view_name in ("raw_thresholded", "final_format"):
        _crosscheck_dictionary_view(
            validation_dictionary.get(view_name),
            validation_metrics["dictionary_all_positive_views"][view_name],
            name=f"validation.dictionary_all_positive.{view_name}",
        )
    _crosscheck_format_loss(
        reported_validation.get("format_constraint_loss"),
        validation_metrics["format_constraint_loss"],
    )
    try:
        reported_per_label = report["final_format"]["overall_36pn"]["per_label"]
        reported_dictionary = report["dictionary_all_positive"]["final_format"]["per_label"]
    except (KeyError, TypeError) as exc:
        raise AuditContractError("evaluation per-label metrics are incomplete") from exc
    _require(
        isinstance(reported_per_label, Mapping)
        and set(reported_per_label) == set(test_metrics["pn_per_label"]),
        "evaluation PN per-label inventory differs",
    )
    for tag, expected in test_metrics["pn_per_label"].items():
        actual = reported_per_label[tag]
        for key in ("tp", "fp", "fn", "tn"):
            _require(
                int(actual.get(key, -1)) == int(expected[key]),
                f"evaluation per-label count differs at {tag}.{key}",
            )
    _require(
        isinstance(reported_dictionary, Mapping)
        and set(reported_dictionary) == set(test_metrics["dictionary_per_label"]),
        "evaluation dictionary per-label inventory differs",
    )
    for tag, expected in test_metrics["dictionary_per_label"].items():
        actual = reported_dictionary[tag]
        _require(
            int(actual.get("positive_support", -1)) == int(expected["positive_support"])
            and int(actual.get("selected_positive", -1))
            == int(expected["selected_positive"]),
            f"evaluation dictionary support differs at {tag}",
        )
        _require_close(actual.get("recall"), expected["recall"], f"dictionary.{tag}.recall")
    reported_threshold = report.get("threshold_calibration")
    _require(
        isinstance(reported_threshold, Mapping)
        and int(reported_threshold.get("pu_fallback_count", -1))
        == int(threshold_audit["pu_fallback_count"])
        and list(reported_threshold.get("pu_fallback_labels") or [])
        == list(threshold_audit["pu_fallback_labels"]),
        "reported PU threshold fallback differs from frozen thresholds",
    )
    cleanup = report.get("process_cleanup")
    _require(
        isinstance(cleanup, Mapping) and cleanup.get("complete") is True,
        "evaluation process cleanup is incomplete",
    )
    return {
        "complete": True,
        "status": report.get("status"),
        "evaluation_report_sha256": sha256_file(report_path),
        "provenance": expected_provenance,
    }


def _per_label_rows(
    metrics: Mapping[str, Any], schema: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tag in schema["labels"]:
        mode = schema["label_training_modes"][tag]
        if mode == "unsupported":
            continue
        dictionary = metrics["dictionary_per_label"].get(tag, {})
        if mode == "pn":
            source = metrics["pn_per_label"][tag]
            warning = ""
            if source["positive_support"] >= 20 and source["f1"] < 0.50:
                warning = "common_label_f1_below_0.50"
            row = {
                "tag": tag,
                "mode": mode,
                **{
                    key: source[key]
                    for key in (
                        "positive_support",
                        "negative_support",
                        "tp",
                        "fp",
                        "fn",
                        "tn",
                        "precision",
                        "recall",
                        "f1",
                        "specificity",
                    )
                },
                "unlabeled_support": "",
                "unlabeled_coverage": "",
                "dictionary_positive_support": dictionary.get("positive_support", 0),
                "dictionary_positive_recall": dictionary.get("recall", ""),
                "warning": warning,
            }
        else:
            source = metrics["pu_per_label"][tag]
            warning = (
                "dictionary_positive_recall_zero"
                if dictionary.get("positive_support", 0) and dictionary.get("recall") == 0
                else ""
            )
            row = {
                "tag": tag,
                "mode": mode,
                "positive_support": source["positive_support"],
                "negative_support": "",
                "tp": "",
                "fp": "",
                "fn": "",
                "tn": "",
                "precision": "",
                "recall": source["positive_recall"],
                "f1": "",
                "specificity": "",
                "unlabeled_support": source["unlabeled_support"],
                "unlabeled_coverage": source["unlabeled_coverage"],
                "dictionary_positive_support": dictionary.get("positive_support", 0),
                "dictionary_positive_recall": dictionary.get("recall", ""),
                "warning": warning,
            }
        rows.append(row)
    return rows


def _supervision_counts(
    rows: Sequence[Mapping[str, Any]], schema: Mapping[str, Any]
) -> dict[str, int]:
    counts = {
        "pn_known_positive": 0,
        "pn_known_negative": 0,
        "pn_known_cells": 0,
        "pn_unknown": 0,
        "pu_positive": 0,
        "pu_unlabeled": 0,
        "unsupported_unknown": 0,
    }
    for row in rows:
        for index, tag in enumerate(schema["labels"]):
            mode = schema["label_training_modes"][tag]
            known = bool(row["known_mask"][index])
            pu = bool(row["pu_positive_mask"][index])
            if mode == "pn":
                if known:
                    counts["pn_known_cells"] += 1
                    if float(row["labels"][index]) == 1.0:
                        counts["pn_known_positive"] += 1
                    else:
                        counts["pn_known_negative"] += 1
                else:
                    counts["pn_unknown"] += 1
            elif mode == "pu":
                if pu:
                    counts["pu_positive"] += 1
                else:
                    counts["pu_unlabeled"] += 1
            else:
                counts["unsupported_unknown"] += int(not known and not pu)
    return counts


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    )
    _atomic_text(path, text)


def _validate_output_isolation(paths: AuditPaths) -> None:
    output = paths.output_dir.resolve(strict=False)
    input_roots = [
        paths.dataset_root.resolve(strict=False),
        paths.evaluation_dir.resolve(strict=False),
        paths.posttrain_dir.resolve(strict=False),
    ]
    if paths.delivery_dir is not None:
        input_roots.append(paths.delivery_dir.resolve(strict=False))
    _require(
        all(output != root and not output.is_relative_to(root) for root in input_roots),
        "output directory must be outside every input tree",
    )
    _require(
        output != paths.schema.resolve(strict=False),
        "output directory must differ from the schema input",
    )


def _clear_audit_outputs(output_dir: Path | str) -> None:
    output = Path(output_dir)
    if not output.exists():
        return
    _require(output.is_dir(), "audit output path must be a directory")
    for name in AUDIT_OUTPUT_FILES:
        path = output / name
        _require(not path.is_dir(), f"audit output file path is a directory: {name}")
        path.unlink(missing_ok=True)


def _audit_posttrain_provenance_binding(
    paths: AuditPaths,
    *,
    schema: Mapping[str, Any],
    thresholds_path: Path,
    report_path: Path,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    preflight_path = paths.posttrain_dir / "preflight_contract.json"
    final_report_path = paths.posttrain_dir / "final_report.json"
    preflight = _load_json(preflight_path, name="posttrain preflight")
    final_report = _load_json(final_report_path, name="posttrain final report")
    _require(isinstance(preflight, Mapping), "posttrain preflight must be an object")
    dataset_hashes = preflight.get("dataset_sha256")
    _require(isinstance(dataset_hashes, Mapping), "posttrain dataset hashes are missing")
    expected_dataset_hashes = {
        split: sha256_file(paths.dataset_root / f"{split}.jsonl")
        for split in ("train", "val", "test")
    }
    for split, expected in expected_dataset_hashes.items():
        _require(
            dataset_hashes.get(split) == expected,
            f"posttrain preflight {split} manifest SHA256 mismatch",
        )
    leakage_hash = sha256_file(paths.dataset_root / "leakage_check.json")
    _require(
        preflight.get("leakage_check_sha256") == leakage_hash,
        "posttrain preflight leakage SHA256 mismatch",
    )
    _require(isinstance(final_report, Mapping), "posttrain final report must be an object")
    if final_report.get("status") != "fail":
        training = final_report.get("training")
        _require(
            isinstance(training, Mapping)
            and training.get("checkpoint_sha256") == checkpoint_sha256,
            "posttrain training checkpoint SHA256 mismatch",
        )
        evaluation = final_report.get("evaluation")
        evaluation_provenance = (
            evaluation.get("provenance") if isinstance(evaluation, Mapping) else None
        )
        _require(
            isinstance(evaluation_provenance, Mapping),
            "posttrain evaluation provenance is missing",
        )
        expected_evaluation = {
            "checkpoint_sha256": checkpoint_sha256,
            "schema_sha256": schema["schema_sha256"],
            "schema_file_sha256": sha256_file(paths.schema),
            "validation_manifest_sha256": expected_dataset_hashes["val"],
            "test_manifest_sha256": expected_dataset_hashes["test"],
            "thresholds_sha256": sha256_file(thresholds_path),
        }
        report_provenance = _load_json(report_path, name="evaluation report").get(
            "provenance"
        )
        _require(
            isinstance(report_provenance, Mapping),
            "evaluation report provenance is missing",
        )
        for key, expected in expected_evaluation.items():
            if key in evaluation_provenance:
                _require(
                    evaluation_provenance.get(key) == expected,
                    f"posttrain evaluation provenance mismatch at {key}",
                )
            if key in report_provenance:
                _require(
                    report_provenance.get(key) == expected,
                    f"evaluation report provenance mismatch at {key}",
                )
    return {
        "complete": True,
        "dataset_sha256": expected_dataset_hashes,
        "leakage_check_sha256": leakage_hash,
        "preflight_sha256": sha256_file(preflight_path),
        "final_report_sha256": sha256_file(final_report_path),
    }


def audit_delivery(
    paths: AuditPaths,
    *,
    expected_validation_count: int = 5444,
    expected_test_count: int = 5441,
    expected_pn_both_class_labels: int = 36,
    expected_dictionary_supported_labels: int = 56,
) -> dict[str, Any]:
    _validate_output_isolation(paths)
    _clear_audit_outputs(paths.output_dir)
    schema = load_schema(paths.schema)
    validation_manifest = paths.dataset_root / "val.jsonl"
    test_manifest = paths.dataset_root / "test.jsonl"
    validation_predictions = paths.evaluation_dir / "validation_predictions_float32.jsonl"
    test_predictions = paths.evaluation_dir / "test_predictions_float32.jsonl"
    leakage_path = paths.dataset_root / "leakage_check.json"
    thresholds_path = paths.evaluation_dir / "thresholds.json"
    report_path = paths.evaluation_dir / "evaluation_report.json"
    leakage = validate_leakage(leakage_path)
    thresholds, threshold_audit = _validate_threshold_contract(
        thresholds_path,
        schema,
        validation_manifest_sha256=sha256_file(validation_manifest),
        expected_validation_count=expected_validation_count,
    )
    checkpoint_sha256 = str(thresholds["checkpoint_sha256"])
    _, validation_rows = load_split_evidence(
        validation_manifest,
        validation_predictions,
        schema,
        expected_validation_count,
        expected_checkpoint_sha256=checkpoint_sha256,
    )
    threshold_audit.update(
        _validate_threshold_support(thresholds, validation_rows, schema)
    )
    _, test_rows = load_split_evidence(
        test_manifest,
        test_predictions,
        schema,
        expected_test_count,
        expected_checkpoint_sha256=checkpoint_sha256,
    )
    output_modes = audit_output_modes(
        paths.evaluation_dir,
        test_rows,
        thresholds,
        schema,
        require_complete=True,
    )
    validation_metrics = recompute_metrics(validation_rows, thresholds, schema)
    test_metrics = recompute_metrics(
        test_rows,
        thresholds,
        schema,
        json_validity_rate=float(output_modes["json_validity_rate"]),
    )
    _require(
        int(test_metrics["overall_36pn"]["labels_with_both_classes"])
        == expected_pn_both_class_labels,
        "PN labels-with-both-classes support count mismatch",
    )
    _require(
        int(test_metrics["dictionary_all_positive"]["labels_with_positive_support"])
        == expected_dictionary_supported_labels,
        "dictionary labels-with-positive-support count mismatch",
    )
    classification = classify_performance(test_metrics["values"])
    report_audit = _crosscheck_report(
        report_path,
        schema_path=paths.schema,
        schema=schema,
        validation_manifest_path=validation_manifest,
        test_manifest_path=test_manifest,
        validation_predictions_path=validation_predictions,
        test_predictions_path=test_predictions,
        thresholds_path=thresholds_path,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        classification=classification,
        expected_validation_count=expected_validation_count,
        expected_test_count=expected_test_count,
        threshold_audit=threshold_audit,
    )
    posttrain_binding = _audit_posttrain_provenance_binding(
        paths,
        schema=schema,
        thresholds_path=thresholds_path,
        report_path=report_path,
        checkpoint_sha256=checkpoint_sha256,
    )
    posttrain = audit_posttrain(paths.posttrain_dir)
    _require(
        not (
            classification["verdict"] != "success"
            and (posttrain["status"] == "success" or posttrain["customer_ready"])
        ),
        "posttrain overclaims the independently measured performance",
    )
    reproduction: dict[str, Any] | None = None
    packaged_output_modes: dict[str, Any] = {
        "complete": False,
        "available_modes": [],
        "reason": "reproduction_result_unavailable",
    }
    if posttrain["reproduction_result_sha256"] is not None:
        reproduction = audit_reproduction_bundle(
            paths.posttrain_dir,
            paths.posttrain_dir.parent / "delivery_candidate",
            evaluation_dir=paths.evaluation_dir,
            test_rows=test_rows,
            thresholds=thresholds,
            schema=schema,
            expected_records=min(32, expected_test_count),
        )
        candidate_verification = _load_json(
            paths.posttrain_dir.parent / "delivery_candidate" / "VERIFICATION.json",
            name="candidate verification",
        )
        candidate_provenance = (
            candidate_verification.get("provenance")
            if isinstance(candidate_verification, Mapping)
            else None
        )
        _require(
            isinstance(candidate_provenance, Mapping),
            "candidate provenance is missing",
        )
        expected_candidate_provenance = {
            "checkpoint_sha256": checkpoint_sha256,
            "schema_file_sha256": sha256_file(paths.schema),
            "thresholds_sha256": sha256_file(thresholds_path),
            "metrics_sha256": sha256_file(report_path),
            "test_manifest_sha256": sha256_file(test_manifest),
        }
        for key, expected in expected_candidate_provenance.items():
            _require(
                candidate_provenance.get(key) == expected,
                f"candidate provenance mismatch at {key}",
            )
        mode_summary_path = (
            paths.posttrain_dir
            / "final_mode_verification"
            / "final_mode_verification.json"
        )
        if mode_summary_path.is_file():
            packaged_output_modes = audit_packaged_output_modes(
                paths.posttrain_dir,
                schema,
                expected_records=min(32, expected_test_count),
            )
        else:
            packaged_output_modes = {
                "complete": False,
                "available_modes": [],
                "reason": "final_mode_verification_missing",
            }
    sealed: dict[str, Any] | None = None
    if paths.delivery_dir is not None:
        sealed = verify_sealed_inventory(paths.delivery_dir)
        verification_path = paths.delivery_dir / "VERIFICATION.json"
        _require(
            verification_path.is_file(),
            "sealed VERIFICATION.json is missing",
        )
        verification = _load_json(verification_path, name="sealed verification")
        _require(isinstance(verification, Mapping), "sealed verification is invalid")
        sealed_status = verification.get("status")
        _require(
            sealed_status in {"success", "partial"},
            "sealed status is invalid",
        )
        _require(
            not (
                classification["verdict"] != "success"
                and (
                    sealed_status == "success"
                    or verification.get("customer_ready") is True
                )
            ),
            "sealed delivery overclaims the independently measured performance",
        )
        _require(
            sealed_status == posttrain["status"],
            "sealed status differs from posttrain status",
        )
        if sealed_status == "success":
            _require(
                verification.get("customer_ready") is True,
                "successful sealed delivery is not customer ready",
            )
        sealed["status"] = verification.get("status")
        sealed["customer_ready"] = bool(verification.get("customer_ready"))
        if packaged_output_modes["complete"]:
            _require(
                packaged_output_modes.get("sealed_delivery_verification_status")
                == sealed["status"]
                and packaged_output_modes.get("sealed_delivery_customer_ready")
                == sealed["customer_ready"],
                "final mode summary differs from sealed delivery status",
            )
        if reproduction is not None:
            sealed_provenance = verification.get("provenance")
            _require(
                verification.get("result_sha256")
                == posttrain["reproduction_result_sha256"]
                and isinstance(sealed_provenance, Mapping)
                and sealed_provenance.get("weights_sha256")
                == reproduction["candidate_weights_sha256"],
                "sealed delivery reproduction or candidate binding mismatch",
            )
    if classification["verdict"] == "fail" or posttrain["status"] == "fail":
        final_verdict = "fail"
    elif (
        classification["verdict"] == "partial"
        or posttrain["status"] == "partial"
        or (sealed is not None and sealed.get("status") == "partial")
        or not packaged_output_modes["complete"]
    ):
        final_verdict = "partial"
    else:
        final_verdict = "success"
    representatives = select_representatives(test_rows, thresholds, schema, count=6)
    representative_outcomes = Counter(
        str(row["outcome"]) for row in representatives
    )
    warnings = [
        "PU confidence scores are sigmoid model scores and are not calibrated probabilities."
    ]
    if threshold_audit["pu_fallback_count"]:
        warnings.append(
            f"{threshold_audit['pu_fallback_count']} PU thresholds use fallback 0.5 because validation support is insufficient."
        )
    if (
        representative_outcomes.get("success", 0) != 3
        or representative_outcomes.get("error", 0) != 3
    ):
        warnings.append(
            "representative 3+3 requirement was not met: "
            f"success={representative_outcomes.get('success', 0)}, "
            f"error={representative_outcomes.get('error', 0)}; missing outcomes were not backfilled."
        )
    audit = {
        "audit_version": "bosideng-unified57-final-audit-v1",
        "integrity": {"passed": True},
        "verdict": final_verdict,
        "performance_verdict": classification["verdict"],
        "customer_ready": final_verdict == "success"
        and posttrain["customer_ready"]
        and (sealed is None or sealed.get("customer_ready") is True),
        "counts": {
            "validation": len(validation_rows),
            "test": len(test_rows),
            "pn_labels_with_both_classes": test_metrics["overall_36pn"][
                "labels_with_both_classes"
            ],
            "dictionary_labels_with_positive_support": test_metrics[
                "dictionary_all_positive"
            ]["labels_with_positive_support"],
        },
        "values": classification["values"],
        "raw_values": test_metrics["raw_values"],
        "performance": test_metrics["performance"],
        "success_gates": classification["success_gates"],
        "validation_values": validation_metrics["values"],
        "recomputed_metrics": {
            "raw_thresholded": test_metrics["raw_thresholded"],
            "final_format": test_metrics["final_format"],
            "dictionary_all_positive": test_metrics[
                "dictionary_all_positive_views"
            ],
            "format_constraint_loss": test_metrics["format_constraint_loss"],
        },
        "supervision_counts": {
            "validation": _supervision_counts(validation_rows, schema),
            "test": _supervision_counts(test_rows, schema),
        },
        "threshold_calibration": threshold_audit,
        "data_contract": {
            "schema_file_sha256": sha256_file(paths.schema),
            "schema_sha256": schema["schema_sha256"],
            "validation_manifest_sha256": sha256_file(validation_manifest),
            "test_manifest_sha256": sha256_file(test_manifest),
            "leakage_check_sha256": sha256_file(leakage_path),
            "leakage_passed": leakage["passed"],
        },
        "evaluation_report": report_audit,
        "output_modes": output_modes,
        "packaged_output_modes": packaged_output_modes,
        "posttrain": posttrain,
        "posttrain_provenance": posttrain_binding,
        "reproduction_32": reproduction,
        "sealed_delivery": sealed,
        "representative_count": len(representatives),
        "representative_outcomes": {
            "success": representative_outcomes.get("success", 0),
            "error": representative_outcomes.get("error", 0),
        },
        "warnings": warnings,
    }
    per_label = _per_label_rows(test_metrics, schema)
    write_audit_reports(paths.output_dir, audit, per_label)
    _atomic_text(
        paths.output_dir / "output_modes_audit.json",
        json.dumps(
            {
                "evaluation": output_modes,
                "packaged_reproduction": packaged_output_modes,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )
    _atomic_jsonl(paths.output_dir / "representative_selection.jsonl", representatives)
    return audit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--posttrain-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--delivery-dir", type=Path)
    parser.add_argument("--expected-validation-count", type=int, default=5444)
    parser.add_argument("--expected-test-count", type=int, default=5441)
    parser.add_argument("--expected-pn-both-class-labels", type=int, default=36)
    parser.add_argument("--expected-dictionary-supported-labels", type=int, default=56)
    args = parser.parse_args(argv)
    if (
        args.expected_validation_count <= 0
        or args.expected_test_count <= 0
        or args.expected_pn_both_class_labels <= 0
        or args.expected_dictionary_supported_labels <= 0
    ):
        parser.error("expected counts must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = AuditPaths(
        schema=args.schema,
        dataset_root=args.dataset_root,
        evaluation_dir=args.evaluation_dir,
        posttrain_dir=args.posttrain_dir,
        output_dir=args.output_dir,
        delivery_dir=args.delivery_dir,
    )
    try:
        _validate_output_isolation(paths)
    except AuditContractError:
        return 2
    try:
        audit = audit_delivery(
            paths,
            expected_validation_count=args.expected_validation_count,
            expected_test_count=args.expected_test_count,
            expected_pn_both_class_labels=args.expected_pn_both_class_labels,
            expected_dictionary_supported_labels=args.expected_dictionary_supported_labels,
        )
    except (AuditContractError, OSError, ValueError, TypeError, KeyError) as exc:
        failure = {
            "audit_version": "bosideng-unified57-final-audit-v1",
            "exit_code": 2,
            "integrity": {"passed": False},
            "verdict": "integrity_error",
            "customer_ready": False,
            "counts": {
                "validation": args.expected_validation_count,
                "test": args.expected_test_count,
            },
            "values": {},
            "success_gates": {},
            "warnings": [],
            "errors": [str(exc)],
        }
        try:
            _clear_audit_outputs(paths.output_dir)
            write_audit_reports(paths.output_dir, failure, [])
        except OSError:
            pass
        return 2
    return 0 if audit["verdict"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
