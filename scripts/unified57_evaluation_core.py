"""Schema-driven evaluation contracts for the Bosideng Unified57 model.

The module deliberately contains no model or DDP code.  It turns archived
float32 scores into calibrated predictions, honest PN/PU metrics, the strict
product JSON contract, durable prediction shards, and deterministic delivery
verification records.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import re
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

# Canonical order from scripts/unified56_contract.py.  Kept local so the
# evaluator remains a standalone delivery file on remote nodes.
JD23_TAGS = (
    "长款", "中款", "短款",
    "H型", "O型", "X型", "A型", "宽松",
    "连帽", "毛领", "立领", "翻领", "无领",
    "压胶充绒", "压胶袋盖", "压胶门襟",
    "平行绗线", "菱形绗线", "葫芦型绗线", "反光条",
    "按扣", "腰带", "插肩袖",
)


EXPECTED_MODE_COUNTS = {"pn": 36, "pu": 20, "unsupported": 1}
EXPECTED_CATEGORIES = ("局部结构", "廓形", "工艺", "面辅料")
UNSUPPORTED_TAG = "假两件"
TWO_DECIMAL_RE = re.compile(r"(?:0\.\d{2}|1\.00)\Z")


def validate_schema(schema: Mapping[str, Any]) -> dict[str, int]:
    labels = schema.get("labels")
    modes = schema.get("label_training_modes")
    if not isinstance(labels, list) or len(labels) != 57 or len(set(labels)) != 57:
        raise ValueError("schema must contain exactly 57 unique labels")
    if schema.get("num_labels") != 57:
        raise ValueError("schema num_labels must be 57")
    if not isinstance(modes, Mapping) or list(modes) != labels:
        raise ValueError("label_training_modes must exactly follow label order")
    counts = Counter(modes.values())
    actual = {key: counts.get(key, 0) for key in EXPECTED_MODE_COUNTS}
    if actual != EXPECTED_MODE_COUNTS or set(counts) != set(EXPECTED_MODE_COUNTS):
        raise ValueError("schema must have exactly 36 PN / 20 PU / 1 unsupported")
    if schema.get("unsupported_labels") != [UNSUPPORTED_TAG] or modes.get(UNSUPPORTED_TAG) != "unsupported":
        raise ValueError("假两件 must be the sole unsupported label")

    categories = schema.get("semantic_categories")
    if not isinstance(categories, Mapping) or tuple(categories) != EXPECTED_CATEGORIES:
        raise ValueError("semantic_categories must contain the four ordered categories")
    seen: list[str] = []
    subcategory_count = 0
    for subcategories in categories.values():
        if not isinstance(subcategories, Mapping):
            raise ValueError("every category must map subcategories to tag lists")
        subcategory_count += len(subcategories)
        for tags in subcategories.values():
            if not isinstance(tags, list) or not tags:
                raise ValueError("each subcategory must contain at least one tag")
            seen.extend(tags)
    if subcategory_count != 20 or Counter(seen) != Counter(labels):
        raise ValueError("semantic_categories must contain 20 subcategories covering all 57 labels once")
    return actual


def _index(schema: Mapping[str, Any]) -> dict[str, int]:
    return {name: index for index, name in enumerate(schema["labels"])}


def _validate_vector(name: str, values: Sequence[Any], length: int = 57) -> None:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} values")


def _validate_row(row: Mapping[str, Any], schema: Mapping[str, Any], *, require_scores: bool = True) -> None:
    for key in ("labels", "known_mask", "pu_positive_mask"):
        _validate_vector(key, row.get(key), len(schema["labels"]))
    if require_scores:
        _validate_vector("scores", row.get("scores"), len(schema["labels"]))
        for value in row["scores"]:
            if not math.isfinite(float(value)):
                raise ValueError("scores must be finite")
    for known, pu in zip(row["known_mask"], row["pu_positive_mask"]):
        if bool(known) and bool(pu):
            raise ValueError("known_mask and pu_positive_mask must be disjoint")
    expected_hash = schema.get("schema_sha256")
    if row.get("schema_sha256", expected_hash) != expected_hash:
        raise ValueError("row schema_sha256 mismatch")
    expected_version = schema.get("schema_version")
    if row.get("schema_version", expected_version) != expected_version:
        raise ValueError("row schema_version mismatch")


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _threshold_item(thresholds: Mapping[str, Any], tag: str) -> Mapping[str, Any]:
    labels = thresholds.get("labels", thresholds)
    item = labels[tag]
    return item if isinstance(item, Mapping) else {"threshold": item}


def _f1(tp: int, fp: int, fn: int) -> float:
    return _safe_ratio(2 * tp, 2 * tp + fp + fn)


def _candidate_thresholds(scores: Iterable[float], fallback: float) -> list[float]:
    return sorted({float(fallback), *(float(score) for score in scores)})


def _count_greater_equal(sorted_scores: Sequence[float], threshold: float) -> int:
    """Count scores >= threshold in O(log N) after one ascending sort."""

    return len(sorted_scores) - bisect.bisect_left(sorted_scores, threshold)


def calibrate_thresholds(
    rows: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
    fallback: float = 0.5,
) -> dict[str, Any]:
    """Fit validation-only thresholds under the 36 PN / 20 PU contract."""

    validate_schema(schema)
    if not 0.0 <= fallback <= 1.0:
        raise ValueError("fallback must lie in [0, 1]")
    for item in rows:
        _validate_row(item, schema)

    output: OrderedDict[str, Any] = OrderedDict()
    for index, tag in enumerate(schema["labels"]):
        mode = schema["label_training_modes"][tag]
        if mode == "unsupported":
            output[tag] = {
                "mode": mode,
                "threshold": None,
                "method": "disabled",
                "status": "disabled_unsupported",
                "support": {},
            }
            continue

        if mode == "pn":
            observed = [
                (float(item["scores"][index]), int(float(item["labels"][index]) == 1.0))
                for item in rows
                if bool(item["known_mask"][index])
            ]
            positive_scores = sorted(score for score, truth in observed if truth == 1)
            negative_scores = sorted(score for score, truth in observed if truth == 0)
            support = {
                "known_positive": len(positive_scores),
                "known_negative": len(negative_scores),
            }
            if not positive_scores or not negative_scores:
                threshold = float(fallback)
                status = "fallback_insufficient_support"
            else:
                candidates = _candidate_thresholds((score for score, _ in observed), fallback)

                def key(threshold_value: float) -> tuple[float, float, float]:
                    tp = _count_greater_equal(positive_scores, threshold_value)
                    fp = _count_greater_equal(negative_scores, threshold_value)
                    fn = len(positive_scores) - tp
                    return (_f1(tp, fp, fn), -abs(threshold_value - fallback), threshold_value)

                threshold = max(candidates, key=key)
                status = "calibrated"
            output[tag] = {
                "mode": mode,
                "threshold": threshold,
                "method": "observed_pn_f1",
                "status": status,
                "support": support,
            }
            continue

        positive_rows = [item for item in rows if bool(item["pu_positive_mask"][index])]
        unlabeled_rows = [
            item
            for item in rows
            if not bool(item["known_mask"][index]) and not bool(item["pu_positive_mask"][index])
        ]
        support = {"positive": len(positive_rows), "unlabeled": len(unlabeled_rows)}
        if len(positive_rows) < 5 or len(unlabeled_rows) < 50:
            threshold = float(fallback)
            status = "fallback_insufficient_support"
        else:
            positive_scores = sorted(float(item["scores"][index]) for item in positive_rows)
            unlabeled_scores = sorted(float(item["scores"][index]) for item in unlabeled_rows)
            candidates = _candidate_thresholds((*positive_scores, *unlabeled_scores), fallback)

            def key(threshold_value: float) -> tuple[float, float, float, float]:
                recall = _safe_ratio(
                    _count_greater_equal(positive_scores, threshold_value),
                    len(positive_scores),
                )
                coverage = _safe_ratio(
                    _count_greater_equal(unlabeled_scores, threshold_value),
                    len(unlabeled_scores),
                )
                return (recall - coverage, recall, -abs(threshold_value - fallback), threshold_value)

            threshold = max(candidates, key=key)
            status = "calibrated"
        output[tag] = {
            "mode": mode,
            "threshold": threshold,
            "method": "positive_minus_unlabeled_coverage",
            "status": status,
            "support": support,
        }

    return {
        "schema_version": schema["schema_version"],
        "schema_sha256": schema["schema_sha256"],
        "fallback_threshold": float(fallback),
        "labels": output,
    }


def raw_predictions(
    scores: Sequence[float], thresholds: Mapping[str, Any], schema: Mapping[str, Any]
) -> list[int]:
    validate_schema(schema)
    _validate_vector("scores", scores, len(schema["labels"]))
    if any(not math.isfinite(float(score)) for score in scores):
        raise ValueError("scores must be finite")
    result: list[int] = []
    for index, tag in enumerate(schema["labels"]):
        mode = schema["label_training_modes"][tag]
        if mode == "unsupported":
            result.append(0)
            continue
        threshold = _threshold_item(thresholds, tag).get("threshold")
        if threshold is None:
            raise ValueError(f"supported tag {tag} has no threshold")
        result.append(int(float(scores[index]) >= float(threshold)))
    return result


def final_format_predictions(
    scores: Sequence[float], thresholds: Mapping[str, Any], schema: Mapping[str, Any]
) -> list[int]:
    raw = raw_predictions(scores, thresholds, schema)
    result = [0] * len(raw)
    indices = _index(schema)
    for subcategories in schema["semantic_categories"].values():
        for tags in subcategories.values():
            candidates = [indices[tag] for tag in tags if raw[indices[tag]]]
            if not candidates:
                continue
            winner = max(candidates, key=lambda idx: (float(scores[idx]), -idx))
            result[winner] = 1
    result[indices[UNSUPPORTED_TAG]] = 0
    return result


def _binary_counts(truth: Sequence[int], prediction: Sequence[int]) -> dict[str, int]:
    tp = sum(t == 1 and p == 1 for t, p in zip(truth, prediction))
    fp = sum(t == 0 and p == 1 for t, p in zip(truth, prediction))
    fn = sum(t == 1 and p == 0 for t, p in zip(truth, prediction))
    tn = sum(t == 0 and p == 0 for t, p in zip(truth, prediction))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _counts_metrics(counts: Mapping[str, int]) -> dict[str, Any]:
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    return {
        **counts,
        "known_cells": tp + fp + fn + tn,
        "known_positive": tp + fn,
        "known_negative": tn + fp,
        "precision": _safe_ratio(tp, tp + fp),
        "recall": _safe_ratio(tp, tp + fn),
        "f1": _f1(tp, fp, fn),
        "specificity": _safe_ratio(tn, tn + fp),
        "fpr": _safe_ratio(fp, fp + tn),
        "accuracy": _safe_ratio(tp + tn, tp + fp + fn + tn),
    }


def evaluate_pn_slice(
    rows: Sequence[Mapping[str, Any]],
    binary_rows: Sequence[Sequence[int]],
    label_indices: Sequence[int],
) -> dict[str, Any]:
    if len(rows) != len(binary_rows):
        raise ValueError("rows and binary_rows must have equal length")
    per_label: OrderedDict[str, Any] = OrderedDict()
    totals = {key: 0 for key in ("tp", "fp", "fn", "tn")}
    exact_hits = 0
    exact_rows = 0

    for label_index in label_indices:
        truth: list[int] = []
        predicted: list[int] = []
        for item, binary in zip(rows, binary_rows):
            if bool(item["known_mask"][label_index]):
                truth.append(int(float(item["labels"][label_index]) == 1.0))
                predicted.append(int(binary[label_index]))
        metrics = _counts_metrics(_binary_counts(truth, predicted))
        per_label[str(label_index)] = metrics
        for key in totals:
            totals[key] += metrics[key]

    for item, binary in zip(rows, binary_rows):
        observed = [index for index in label_indices if bool(item["known_mask"][index])]
        if observed:
            exact_rows += 1
            exact_hits += all(
                int(float(item["labels"][index]) == 1.0) == int(binary[index]) for index in observed
            )

    eligible = [metrics for metrics in per_label.values() if metrics["known_cells"]]
    macro_keys = ("precision", "recall", "f1", "specificity", "fpr", "accuracy")
    macro = {
        key: _safe_ratio(sum(item[key] for item in eligible), len(eligible)) for key in macro_keys
    }
    macro["labels_evaluated"] = len(eligible)
    both_class = [
        item for item in eligible if item["known_positive"] > 0 and item["known_negative"] > 0
    ]
    positive_labels = [item for item in eligible if item["known_positive"] > 0]
    macro["labels_with_both_classes"] = len(both_class)
    macro["f1_both_class_labels"] = _safe_ratio(
        sum(item["f1"] for item in both_class), len(both_class)
    )
    macro["positive_labels_evaluated"] = len(positive_labels)
    macro["positive_labels_macro_recall"] = _safe_ratio(
        sum(item["recall"] for item in positive_labels), len(positive_labels)
    )
    micro = _counts_metrics(totals)
    return {
        "record_count": len(rows),
        "micro": micro,
        "macro": macro,
        "per_label": per_label,
        "exact_match": _safe_ratio(exact_hits, exact_rows),
        "exact_match_rows": exact_rows,
        "trusted_negatives": {
            "tn": micro["tn"],
            "fp": micro["fp"],
            "specificity": micro["specificity"],
            "fpr": micro["fpr"],
        },
    }


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


def evaluate_pu_label(
    positive_scores: Sequence[float], unlabeled_scores: Sequence[float], threshold: float
) -> dict[str, Any]:
    positives = [float(value) for value in positive_scores]
    unlabeled = [float(value) for value in unlabeled_scores]
    if any(not math.isfinite(value) for value in (*positives, *unlabeled)):
        raise ValueError("PU scores must be finite")
    sorted_unlabeled = sorted(unlabeled)
    concordant = 0.0
    for score in positives:
        lower = bisect.bisect_left(sorted_unlabeled, score)
        upper = bisect.bisect_right(sorted_unlabeled, score)
        concordant += lower + 0.5 * (upper - lower)
    positive_selected = sum(score >= threshold for score in positives)
    unlabeled_selected = sum(score >= threshold for score in unlabeled)
    total = len(positives) + len(unlabeled)
    return {
        "positive_count": len(positives),
        "unlabeled_count": len(unlabeled),
        "positive_recall": _safe_ratio(positive_selected, len(positives)),
        "positive_vs_unlabeled_concordance": _safe_ratio(
            concordant, len(positives) * len(unlabeled)
        ),
        "positive_score_quantiles": {
            "p10": _quantile(positives, 0.10),
            "p50": _quantile(positives, 0.50),
            "p90": _quantile(positives, 0.90),
        },
        "unlabeled_score_quantiles": {
            "p10": _quantile(unlabeled, 0.10),
            "p50": _quantile(unlabeled, 0.50),
            "p90": _quantile(unlabeled, 0.90),
        },
        "all_coverage": _safe_ratio(positive_selected + unlabeled_selected, total),
        "unlabeled_coverage": _safe_ratio(unlabeled_selected, len(unlabeled)),
        "selected_positive_count": positive_selected,
        "selected_unlabeled_count": unlabeled_selected,
        "threshold": float(threshold),
    }


def _pu_metrics_for_view(
    rows: Sequence[Mapping[str, Any]],
    binary_rows: Sequence[Sequence[int]],
    thresholds: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    per_label: OrderedDict[str, Any] = OrderedDict()
    total_p = total_u = selected_p = selected_u = 0
    concordances: list[float] = []
    for index, tag in enumerate(schema["labels"]):
        if schema["label_training_modes"][tag] != "pu":
            continue
        positives = [float(item["scores"][index]) for item in rows if bool(item["pu_positive_mask"][index])]
        unlabeled = [
            float(item["scores"][index])
            for item in rows
            if not bool(item["known_mask"][index]) and not bool(item["pu_positive_mask"][index])
        ]
        base = evaluate_pu_label(
            positives, unlabeled, float(_threshold_item(thresholds, tag)["threshold"])
        )
        # Coverage and P recall must describe this prediction view after formatter suppression.
        positive_flags = [
            int(binary[index]) for item, binary in zip(rows, binary_rows) if bool(item["pu_positive_mask"][index])
        ]
        unlabeled_flags = [
            int(binary[index])
            for item, binary in zip(rows, binary_rows)
            if not bool(item["known_mask"][index]) and not bool(item["pu_positive_mask"][index])
        ]
        base["selected_positive_count"] = sum(positive_flags)
        base["selected_unlabeled_count"] = sum(unlabeled_flags)
        base["positive_recall"] = _safe_ratio(sum(positive_flags), len(positive_flags))
        base["unlabeled_coverage"] = _safe_ratio(sum(unlabeled_flags), len(unlabeled_flags))
        base["all_coverage"] = _safe_ratio(
            sum(positive_flags) + sum(unlabeled_flags), len(positive_flags) + len(unlabeled_flags)
        )
        base["calibration_status"] = _threshold_item(thresholds, tag).get("status")
        per_label[tag] = base
        total_p += len(positive_flags)
        total_u += len(unlabeled_flags)
        selected_p += sum(positive_flags)
        selected_u += sum(unlabeled_flags)
        if positives and unlabeled:
            concordances.append(base["positive_vs_unlabeled_concordance"])

    return {
        "per_label": per_label,
        "summary": {
            "positive_count": total_p,
            "unlabeled_count": total_u,
            "support_weighted_positive_recall": _safe_ratio(selected_p, total_p),
            "macro_positive_vs_unlabeled_concordance": _safe_ratio(
                sum(concordances), len(concordances)
            ),
            "micro_all_coverage": _safe_ratio(selected_p + selected_u, total_p + total_u),
            "micro_unlabeled_coverage": _safe_ratio(selected_u, total_u),
        },
    }


def _slice_metrics(
    rows: Sequence[Mapping[str, Any]],
    binaries: Sequence[Sequence[int]],
    pn_indices: Sequence[int],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    def named(report: dict[str, Any]) -> dict[str, Any]:
        report["per_label"] = OrderedDict(
            (schema["labels"][int(index)], metrics)
            for index, metrics in report["per_label"].items()
        )
        return report

    overall = named(evaluate_pn_slice(rows, binaries, pn_indices))
    indices = _index(schema)
    if len(JD23_TAGS) != 23 or any(tag not in indices for tag in JD23_TAGS):
        raise ValueError("canonical JD23 tag contract is incompatible with Unified57 schema")
    jd23_indices = [indices[tag] for tag in JD23_TAGS]
    if any(schema["label_training_modes"][tag] != "pn" for tag in JD23_TAGS):
        raise ValueError("all canonical JD23 tags must use PN supervision")

    def subset(predicate: Callable[[Mapping[str, Any]], bool]) -> tuple[list[Mapping[str, Any]], list[Sequence[int]]]:
        pairs = [(item, binary) for item, binary in zip(rows, binaries) if predicate(item)]
        return [item for item, _ in pairs], [binary for _, binary in pairs]

    jd_rows, jd_binary = subset(lambda item: item.get("sources") == ["jd_complete23"])
    dict_rows, dict_binary = subset(lambda item: item.get("sources") == ["dictionary_v4"])
    mixed_rows, _ = subset(
        lambda item: set(item.get("sources", [])) == {"jd_complete23", "dictionary_v4"}
    )
    mixed_known = sum(bool(item["known_mask"][index]) for item in mixed_rows for index in pn_indices)
    mixed_positive = sum(
        bool(item["known_mask"][index]) and float(item["labels"][index]) == 1.0
        for item in mixed_rows
        for index in pn_indices
    )
    return {
        "overall_36pn": overall,
        "jd23_clean": named(evaluate_pn_slice(jd_rows, jd_binary, jd23_indices)),
        "dictionary_pn_clean": named(evaluate_pn_slice(dict_rows, dict_binary, pn_indices)),
        "mixed_exact_audit": {
            "record_count": len(mixed_rows),
            "known_cells": mixed_known,
            "known_positive": mixed_positive,
            "known_negative": mixed_known - mixed_positive,
            "fraction_of_overall_known_cells": _safe_ratio(
                mixed_known, overall["micro"]["known_cells"]
            ),
        },
    }


def _format_constraint_loss(
    rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Sequence[int]],
    final_rows: Sequence[Sequence[int]],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    indices = _index(schema)
    raw_multi: OrderedDict[str, int] = OrderedDict()
    observed_multi: OrderedDict[str, int] = OrderedDict()
    suppressed_by_subcategory: OrderedDict[str, int] = OrderedDict()
    suppressed_by_tag = Counter()
    hit_by_subcategory: OrderedDict[str, Any] = OrderedDict()
    observed_total = oracle_retained = forced_fn = 0

    for subcategories in schema["semantic_categories"].values():
        for subcategory, tags in subcategories.items():
            tag_indices = [indices[tag] for tag in tags if tag != UNSUPPORTED_TAG]
            raw_multi[subcategory] = 0
            observed_multi[subcategory] = 0
            suppressed_by_subcategory[subcategory] = 0
            hit_denominator = hit_numerator = 0
            for item, raw, final in zip(rows, raw_rows, final_rows):
                raw_count = sum(raw[index] for index in tag_indices)
                if raw_count > 1:
                    raw_multi[subcategory] += 1
                for index in tag_indices:
                    if raw[index] and not final[index]:
                        suppressed_by_subcategory[subcategory] += 1
                        suppressed_by_tag[schema["labels"][index]] += 1
                observed = [
                    index
                    for index in tag_indices
                    if (
                        bool(item["known_mask"][index]) and float(item["labels"][index]) == 1.0
                    )
                    or bool(item["pu_positive_mask"][index])
                ]
                if observed:
                    hit_denominator += 1
                    hit_numerator += any(final[index] for index in observed)
                    observed_total += len(observed)
                    oracle_retained += 1
                    forced_fn += max(0, len(observed) - 1)
                    if len(observed) > 1:
                        observed_multi[subcategory] += 1
            hit_by_subcategory[subcategory] = {
                "hit": hit_numerator,
                "observed_records": hit_denominator,
                "rate": _safe_ratio(hit_numerator, hit_denominator),
            }
    return {
        "raw_multi_selected_records": raw_multi,
        "formatter_suppressed_predictions": {
            "total": sum(suppressed_by_subcategory.values()),
            "by_subcategory": suppressed_by_subcategory,
            "by_tag": OrderedDict(
                (tag, suppressed_by_tag.get(tag, 0)) for tag in schema["labels"] if tag != UNSUPPORTED_TAG
            ),
        },
        "observed_multi_positive_records": observed_multi,
        "observed_positive_count": observed_total,
        "oracle_final_recall_ceiling": _safe_ratio(oracle_retained, observed_total),
        "contract_forced_false_negatives": forced_fn,
        "subcategory_hit_rate": hit_by_subcategory,
    }


def evaluate_views(
    rows: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any], schema: Mapping[str, Any]
) -> dict[str, Any]:
    validate_schema(schema)
    for item in rows:
        _validate_row(item, schema)
    raw_rows = [raw_predictions(item["scores"], thresholds, schema) for item in rows]
    final_rows = [final_format_predictions(item["scores"], thresholds, schema) for item in rows]
    pn_indices = [
        index
        for index, tag in enumerate(schema["labels"])
        if schema["label_training_modes"][tag] == "pn"
    ]
    raw_report = _slice_metrics(rows, raw_rows, pn_indices, schema)
    final_report = _slice_metrics(rows, final_rows, pn_indices, schema)
    raw_report["pu"] = _pu_metrics_for_view(rows, raw_rows, thresholds, schema)
    final_report["pu"] = _pu_metrics_for_view(rows, final_rows, thresholds, schema)
    return {
        "raw_thresholded": raw_report,
        "final_format": final_report,
        "format_constraint_loss": _format_constraint_loss(rows, raw_rows, final_rows, schema),
        "unsupported": {UNSUPPORTED_TAG: {"status": "unsupported", "selected_count": 0}},
    }


def render_all_scores(row: Mapping[str, Any], schema: Mapping[str, Any]) -> dict[str, Any]:
    validate_schema(schema)
    scores = row.get("scores")
    _validate_vector("scores", scores, len(schema["labels"]))
    if any(not math.isfinite(float(score)) for score in scores):
        raise ValueError("scores must be finite")
    output: OrderedDict[str, Any] = OrderedDict()
    for key in ("record_id", "image_path", "image_sha256", "source", "sources"):
        if key in row:
            output[key] = row[key]
    visible: OrderedDict[str, str] = OrderedDict()
    for index, tag in enumerate(schema["labels"]):
        value = 0.0 if tag == UNSUPPORTED_TAG else min(1.0, max(0.0, float(scores[index])))
        visible[tag] = f"{value:.2f}"
    output["scores"] = visible
    return output


def validate_all_scores(payload: Mapping[str, Any], schema: Mapping[str, Any]) -> Mapping[str, Any]:
    validate_schema(schema)
    scores = payload.get("scores")
    if not isinstance(scores, Mapping) or list(scores) != schema["labels"]:
        raise ValueError("scores must contain exactly 57 keys in schema order")
    for tag, value in scores.items():
        if not isinstance(value, str) or not TWO_DECIMAL_RE.fullmatch(value):
            raise ValueError(f"score for {tag} must be a two-decimal string in [0,1]")
    if scores[UNSUPPORTED_TAG] != "0.00":
        raise ValueError("假两件 all-score must be 0.00")
    return payload


def render_selected_only(
    scores: Sequence[float], thresholds: Mapping[str, Any], schema: Mapping[str, Any]
) -> dict[str, list[str]]:
    final = final_format_predictions(scores, thresholds, schema)
    indices = _index(schema)
    payload: OrderedDict[str, list[str]] = OrderedDict()
    for category, subcategories in schema["semantic_categories"].items():
        selected = [
            tag
            for tags in subcategories.values()
            for tag in tags
            if tag != UNSUPPORTED_TAG and final[indices[tag]]
        ]
        payload[category] = sorted(selected, key=indices.__getitem__)
    return payload


def render_selected_with_confidence(
    scores: Sequence[float], thresholds: Mapping[str, Any], schema: Mapping[str, Any]
) -> dict[str, list[dict[str, str]]]:
    """Render the same final winners with user-facing two-decimal confidence."""

    selected = render_selected_only(scores, thresholds, schema)
    indices = _index(schema)
    payload: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for category, tags in selected.items():
        payload[category] = [
            {
                "name": tag,
                "confidence": f"{min(1.0, max(0.0, float(scores[indices[tag]]))):.2f}",
            }
            for tag in tags
        ]
    return payload


def validate_selected_only(
    payload: Mapping[str, Any], schema: Mapping[str, Any]
) -> Mapping[str, Any]:
    validate_schema(schema)
    if not isinstance(payload, Mapping) or tuple(payload) != EXPECTED_CATEGORIES:
        raise ValueError("selected output must contain the four categories in order")
    indices = _index(schema)
    seen: set[str] = set()
    for category, selected in payload.items():
        if not isinstance(selected, list) or any(not isinstance(tag, str) for tag in selected):
            raise ValueError("every category value must be a string array")
        if selected != sorted(selected, key=lambda tag: indices.get(tag, 10**9)):
            raise ValueError("selected tags must follow schema order")
        for tag in selected:
            if tag in seen:
                raise ValueError("selected tag must not be repeated")
            if tag == UNSUPPORTED_TAG:
                raise ValueError("unsupported tag must not be selected")
            if tag not in indices:
                raise ValueError(f"unknown selected tag: {tag}")
            seen.add(tag)
        for subcategory, allowed in schema["semantic_categories"][category].items():
            if sum(tag in allowed for tag in selected) > 1:
                raise ValueError(f"subcategory {subcategory} contains more than one selected tag")
        allowed_category = {
            tag for tags in schema["semantic_categories"][category].values() for tag in tags
        }
        if not set(selected) <= allowed_category:
            raise ValueError(f"tag assigned to wrong category {category}")
    return payload


def validate_selected_with_confidence(
    payload: Mapping[str, Any], schema: Mapping[str, Any]
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping) or tuple(payload) != EXPECTED_CATEGORIES:
        raise ValueError("selected-with-confidence output must contain the four categories in order")
    names: OrderedDict[str, list[str]] = OrderedDict()
    for category, values in payload.items():
        if not isinstance(values, list):
            raise ValueError("selected-with-confidence category values must be arrays")
        names[category] = []
        for item in values:
            if not isinstance(item, Mapping) or tuple(item) != ("name", "confidence"):
                raise ValueError("selected-with-confidence item must contain name and confidence")
            name = item["name"]
            confidence = item["confidence"]
            if not isinstance(name, str):
                raise ValueError("selected-with-confidence name must be a string")
            if not isinstance(confidence, str) or not TWO_DECIMAL_RE.fullmatch(confidence):
                raise ValueError("selected-with-confidence confidence must be a two-decimal string")
            names[category].append(name)
    validate_selected_only(names, schema)
    return payload


class BufferedPredictionShard:
    """Append-only JSONL shard with durable-offset crash recovery.

    Data and sidecar are synced only at the record/time boundaries.  Recovery
    truncates any non-durable tail before parsing, so partial JSON cannot leak
    into a resumed evaluation.
    """

    def __init__(
        self,
        path: str | Path,
        metadata: Mapping[str, Any],
        *,
        sync_every_records: int = 1000,
        sync_every_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if sync_every_records <= 0 or sync_every_seconds <= 0:
            raise ValueError("sync intervals must be positive")
        self.path = Path(path)
        self.sidecar_path = self.path.with_name(self.path.name + ".progress.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = json.loads(json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True))
        reserved = {
            "version",
            "metadata",
            "durable_records",
            "durable_offset",
            "next_local_index",
            "last_record_id",
            "complete",
        }
        if reserved & set(self.metadata):
            raise ValueError("prediction shard metadata uses a reserved key")
        self.sync_every_records = int(sync_every_records)
        self.sync_every_seconds = float(sync_every_seconds)
        self.clock = clock
        self.data_sync_count = 0
        self.durable_records = 0
        self.durable_offset = 0
        self.next_local_index = 0
        self._records = 0
        self._seen: set[str] = set()
        self._last_sync_time = float(clock())
        self._complete = False

        if self.sidecar_path.exists():
            progress = json.loads(self.sidecar_path.read_text(encoding="utf-8"))
            if progress.get("metadata") != self.metadata:
                raise ValueError("prediction shard metadata mismatch")
            self.durable_records = int(progress["durable_records"])
            self.durable_offset = int(progress["durable_offset"])
            self.next_local_index = int(progress["next_local_index"])
            self._complete = bool(progress.get("complete", False))
            if not self.path.exists() or self.path.stat().st_size < self.durable_offset:
                raise ValueError("prediction shard is shorter than durable offset")
            with self.path.open("r+b") as handle:
                handle.truncate(self.durable_offset)
            self._load_durable_rows()
        elif self.path.exists() and self.path.stat().st_size:
            # No sidecar means no byte was ever declared durable.  A process can
            # die after buffered writes and before its first sync; recovery is
            # therefore a safe truncate-to-zero, not a parse attempt.
            with self.path.open("r+b") as handle:
                handle.truncate(0)
        else:
            self.path.touch()

        self._records = self.durable_records
        self._handle = self.path.open("a", encoding="utf-8", newline="")

    def _load_durable_rows(self) -> None:
        count = 0
        last_id = None
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError("durable shard contains invalid JSON") from error
                record_id = item.get("record_id")
                if not isinstance(record_id, str) or not record_id:
                    raise ValueError("durable shard row lacks record_id")
                if record_id in self._seen:
                    raise ValueError("duplicate record_id in durable shard")
                self._seen.add(record_id)
                last_id = record_id
                count += 1
        if count != self.durable_records:
            raise ValueError("durable record count mismatch")
        progress = json.loads(self.sidecar_path.read_text(encoding="utf-8"))
        if progress.get("last_record_id") != last_id:
            raise ValueError("durable last_record_id mismatch")

    def append_batch(
        self, rows: Sequence[Mapping[str, Any]], next_local_index: int
    ) -> int:
        if self._handle.closed:
            raise ValueError("prediction shard is closed")
        if self._complete:
            raise ValueError("completed prediction shard cannot be appended")
        expected_next = self.next_local_index + len(rows)
        if int(next_local_index) != expected_next:
            raise ValueError(
                f"next_local_index must advance contiguously to {expected_next}, got {next_local_index}"
            )
        pending_ids: set[str] = set()
        for item in rows:
            record_id = item.get("record_id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError("prediction row requires a non-empty record_id")
            if record_id in self._seen or record_id in pending_ids:
                raise ValueError(f"duplicate record_id: {record_id}")
            pending_ids.add(record_id)
        for item in rows:
            self._handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._seen.update(pending_ids)
        self._records += len(rows)
        self.next_local_index = int(next_local_index)
        if (
            self._records - self.durable_records >= self.sync_every_records
            or float(self.clock()) - self._last_sync_time >= self.sync_every_seconds
        ):
            self.sync()
        return self.durable_records

    def _write_sidecar(self, *, complete: bool) -> dict[str, Any]:
        payload = {
            "version": 1,
            **self.metadata,
            "metadata": self.metadata,
            "durable_records": self.durable_records,
            "durable_offset": self.durable_offset,
            "next_local_index": self.next_local_index,
            "last_record_id": None,
            "complete": bool(complete),
        }
        # Sets do not preserve file order. Read only the final line for the authoritative id.
        if self.durable_records:
            with self.path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                end = handle.tell()
                position = end - 1
                while position > 0:
                    handle.seek(position)
                    if handle.read(1) == b"\n" and position < end - 1:
                        position += 1
                        break
                    position -= 1
                handle.seek(max(0, position))
                last_line = handle.readline().decode("utf-8")
            payload["last_record_id"] = json.loads(last_line)["record_id"]
        temp = self.sidecar_path.with_name(self.sidecar_path.name + ".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, self.sidecar_path)
        try:
            directory_fd = os.open(self.path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
        return payload

    def sync(self) -> dict[str, Any]:
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self.data_sync_count += 1
        self.durable_offset = self._handle.tell()
        self.durable_records = self._records
        self._last_sync_time = float(self.clock())
        return self._write_sidecar(complete=False)

    def close(self, *, complete: bool) -> dict[str, Any]:
        if self._handle.closed:
            return json.loads(self.sidecar_path.read_text(encoding="utf-8"))
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self.data_sync_count += 1
        self.durable_offset = self._handle.tell()
        self.durable_records = self._records
        payload = self._write_sidecar(complete=complete)
        self._handle.close()
        self._complete = bool(complete)
        return payload

    def __enter__(self) -> "BufferedPredictionShard":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close(complete=exc_type is None)


def select_verification_records(
    rows: Sequence[Mapping[str, Any]], count: int = 32, seed: int = 20260717
) -> list[Mapping[str, Any]]:
    """Pure deterministic sample selection, independent of input ordering."""

    if count <= 0 or len(rows) < count:
        raise ValueError("verification selection requires at least count rows")
    ids = [item.get("record_id") for item in rows]
    if any(not isinstance(record_id, str) or not record_id for record_id in ids):
        raise ValueError("verification rows require record_id")
    if len(set(ids)) != len(ids):
        raise ValueError("verification rows contain duplicate record_id")

    def key(item: Mapping[str, Any]) -> tuple[str, str]:
        record_id = str(item["record_id"])
        digest = hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest()
        return digest, record_id

    ordered = sorted(rows, key=key)
    selected: list[Mapping[str, Any]] = []
    selected_ids: set[str] = set()

    def has_pn_positive(item: Mapping[str, Any]) -> bool:
        return any(
            bool(known) and float(label) == 1.0
            for known, label in zip(item.get("known_mask", []), item.get("labels", []))
        )

    def has_pu_positive(item: Mapping[str, Any]) -> bool:
        return any(bool(value) for value in item.get("pu_positive_mask", []))

    def aspect_bucket(item: Mapping[str, Any]) -> str | None:
        ratio = item.get("aspect_ratio", item.get("aspect"))
        if ratio is None:
            width = item.get("width", item.get("image_width"))
            height = item.get("height", item.get("image_height"))
            if width is None or height is None or float(height) <= 0:
                return None
            ratio = float(width) / float(height)
        ratio = float(ratio)
        if ratio < 0.8:
            return "portrait"
        if ratio > 1.25:
            return "landscape"
        return "square"

    strata = [
        lambda item: item.get("sources") == ["jd_complete23"],
        lambda item: item.get("sources") == ["dictionary_v4"],
        lambda item: set(item.get("sources", [])) == {"jd_complete23", "dictionary_v4"},
        has_pn_positive,
        has_pu_positive,
        lambda item: aspect_bucket(item) == "portrait",
        lambda item: aspect_bucket(item) == "square",
        lambda item: aspect_bucket(item) == "landscape",
    ]
    for predicate in strata:
        candidate = next(
            (item for item in ordered if item["record_id"] not in selected_ids and predicate(item)), None
        )
        if candidate is not None:
            selected.append(candidate)
            selected_ids.add(str(candidate["record_id"]))
    for item in ordered:
        if len(selected) == count:
            break
        if item["record_id"] not in selected_ids:
            selected.append(item)
            selected_ids.add(str(item["record_id"]))
    return selected


def verify_reproduction(
    reference: Sequence[Mapping[str, Any]],
    reproduced: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    validate_schema(schema)
    if len(reference) != 32 or len(reproduced) != 32:
        raise ValueError("verification requires exactly 32 reference and reproduced records")
    reference_ids = [item.get("record_id") for item in reference]
    reproduced_ids = [item.get("record_id") for item in reproduced]
    if any(not isinstance(record_id, str) or not record_id for record_id in reference_ids):
        raise ValueError("reference verification record ids must be non-empty strings")
    if any(not isinstance(record_id, str) or not record_id for record_id in reproduced_ids):
        raise ValueError("reproduced verification record ids must be non-empty strings")
    if len(set(reference_ids)) != 32:
        raise ValueError("reference verification record ids must be unique")
    reproduced_by_id = {item.get("record_id"): item for item in reproduced}
    if len(reproduced_by_id) != 32:
        raise ValueError("reproduced verification record ids must be unique")
    max_delta = 0.0
    probabilities_exact = True
    selected_exact = True
    for expected in reference:
        record_id = expected.get("record_id")
        if record_id not in reproduced_by_id:
            raise ValueError(f"missing reproduced record: {record_id}")
        actual = reproduced_by_id[record_id]
        _validate_vector("reference scores", expected.get("scores"), 57)
        _validate_vector("reproduced scores", actual.get("scores"), 57)
        for expected_score, actual_score in zip(expected["scores"], actual["scores"]):
            if not math.isfinite(float(expected_score)) or not math.isfinite(float(actual_score)):
                raise ValueError("verification scores must be finite")
            delta = abs(float(expected_score) - float(actual_score))
            max_delta = max(max_delta, delta)
            probabilities_exact = probabilities_exact and delta == 0.0
        expected_selected = expected.get("selected", expected.get("selected_only", expected.get("output")))
        actual_selected = actual.get("selected", actual.get("selected_only", actual.get("output")))
        validate_selected_only(expected_selected, schema)
        validate_selected_only(actual_selected, schema)
        selected_exact = selected_exact and expected_selected == actual_selected
    return {
        "records": 32,
        "score_values": 32 * 57,
        "probabilities_exact": probabilities_exact,
        "max_abs_score_delta": max_delta,
        "selected_outputs_exact": selected_exact,
    }


__all__ = [
    "BufferedPredictionShard",
    "calibrate_thresholds",
    "evaluate_pn_slice",
    "evaluate_pu_label",
    "evaluate_views",
    "final_format_predictions",
    "raw_predictions",
    "render_all_scores",
    "render_selected_only",
    "render_selected_with_confidence",
    "select_verification_records",
    "validate_all_scores",
    "validate_schema",
    "validate_selected_only",
    "validate_selected_with_confidence",
    "verify_reproduction",
]
