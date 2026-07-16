from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.audit_unified57_final_delivery import (
    AuditContractError,
    AuditPaths,
    audit_delivery,
    audit_output_modes,
    audit_packaged_output_modes,
    audit_posttrain,
    audit_reproduction_bundle,
    classify_performance,
    load_schema,
    load_split_evidence,
    main,
    recompute_metrics,
    select_representatives,
    validate_leakage,
    verify_sealed_inventory,
    write_audit_reports,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "configs" / "bosideng_unified57_schema.json"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _base_row(schema: dict, record_id: str) -> dict:
    labels = [0.0] * 57
    known = [0] * 57
    pu = [0] * 57
    pn_index = schema["labels"].index("连帽")
    labels[pn_index] = 1.0
    known[pn_index] = 1
    return {
        "record_id": record_id,
        "image_path": f"/frozen/{record_id}.jpg",
        "image_sha256": hashlib.sha256(record_id.encode()).hexdigest(),
        "source": "jd_complete23",
        "sources": ["jd_complete23"],
        "schema_version": schema["schema_version"],
        "schema_sha256": schema["schema_sha256"],
        "labels": labels,
        "known_mask": known,
        "pu_positive_mask": pu,
    }


def _evidence(tmp_path: Path, count: int = 3) -> tuple[dict, Path, Path, list[dict], list[dict]]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    manifest_rows = [_base_row(schema, f"r{index}") for index in range(count)]
    prediction_rows = []
    for row in manifest_rows:
        prediction = copy.deepcopy(row)
        prediction["scores"] = [0.1] * 57
        prediction["scores"][schema["labels"].index("连帽")] = 0.9
        prediction["checkpoint_sha256"] = "c" * 64
        prediction_rows.append(prediction)
    manifest = tmp_path / "test.jsonl"
    predictions = tmp_path / "test_predictions_float32.jsonl"
    _write_jsonl(manifest, manifest_rows)
    _write_jsonl(predictions, prediction_rows)
    return schema, manifest, predictions, manifest_rows, prediction_rows


def test_load_schema_accepts_production_contract() -> None:
    schema = load_schema(SCHEMA_PATH)
    assert len(schema["labels"]) == 57
    assert list(schema["label_training_modes"].values()).count("pn") == 36
    assert list(schema["label_training_modes"].values()).count("pu") == 20
    assert schema["unsupported_labels"] == ["假两件"]


def test_load_split_accepts_exact_ordered_evidence(tmp_path: Path) -> None:
    schema, manifest, predictions, manifest_rows, prediction_rows = _evidence(tmp_path)
    loaded_manifest, loaded_predictions = load_split_evidence(
        manifest,
        predictions,
        schema,
        expected_count=3,
    )
    assert loaded_manifest == manifest_rows
    assert loaded_predictions == prediction_rows


def test_load_split_rejects_unknown_mask_violation(tmp_path: Path) -> None:
    schema, manifest, predictions, manifest_rows, prediction_rows = _evidence(tmp_path)
    pu_index = schema["labels"].index("前门襟")
    manifest_rows[0]["labels"][pu_index] = 1.0
    prediction_rows[0]["labels"][pu_index] = 1.0
    _write_jsonl(manifest, manifest_rows)
    _write_jsonl(predictions, prediction_rows)
    with pytest.raises(AuditContractError, match="unknown cell"):
        load_split_evidence(manifest, predictions, schema, expected_count=3)


def test_load_split_rejects_prediction_order_drift(tmp_path: Path) -> None:
    schema, manifest, predictions, _manifest_rows, prediction_rows = _evidence(tmp_path)
    _write_jsonl(predictions, list(reversed(prediction_rows)))
    with pytest.raises(AuditContractError, match="manifest order"):
        load_split_evidence(manifest, predictions, schema, expected_count=3)


def test_load_split_rejects_prediction_evidence_drift(tmp_path: Path) -> None:
    schema, manifest, predictions, _manifest_rows, prediction_rows = _evidence(tmp_path)
    prediction_rows[0]["sources"] = ["dictionary_v4"]
    _write_jsonl(predictions, prediction_rows)
    with pytest.raises(AuditContractError, match="evidence differs at sources"):
        load_split_evidence(manifest, predictions, schema, expected_count=3)


def test_load_split_rejects_invalid_image_sha256(tmp_path: Path) -> None:
    schema, manifest, predictions, manifest_rows, prediction_rows = _evidence(tmp_path)
    manifest_rows[0]["image_sha256"] = "bad"
    prediction_rows[0]["image_sha256"] = "bad"
    _write_jsonl(manifest, manifest_rows)
    _write_jsonl(predictions, prediction_rows)
    with pytest.raises(AuditContractError, match="image_sha256"):
        load_split_evidence(manifest, predictions, schema, expected_count=3)


def test_validate_leakage_requires_empty_cross_split_lists(tmp_path: Path) -> None:
    leakage_path = tmp_path / "leakage_check.json"
    leakage_path.write_text(
        json.dumps(
            {
                "passed": True,
                "cross_split_components": [],
                "cross_split_exact_phash": [],
                "cross_split_sha256": [],
            }
        ),
        encoding="utf-8",
    )
    assert validate_leakage(leakage_path)["passed"] is True
    payload = json.loads(leakage_path.read_text())
    payload["cross_split_sha256"] = [{"sha256": "bad"}]
    leakage_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AuditContractError, match="visual leakage"):
        validate_leakage(leakage_path)


def _thresholds(schema: dict, value: float = 0.5) -> dict:
    return {
        "schema_version": schema["schema_version"],
        "schema_sha256": schema["schema_sha256"],
        "checkpoint_sha256": "c" * 64,
        "validation_manifest_sha256": "v" * 64,
        "fallback_threshold": 0.5,
        "calibration_records": 4,
        "labels": {
            tag: {
                "mode": schema["label_training_modes"][tag],
                "threshold": None if tag == "假两件" else value,
                "status": (
                    "disabled_unsupported"
                    if tag == "假两件"
                    else "fallback_insufficient_support"
                    if schema["label_training_modes"][tag] == "pu"
                    else "calibrated"
                ),
            }
            for tag in schema["labels"]
        },
    }


def _metric_rows(schema: dict) -> list[dict]:
    pn = schema["labels"].index("连帽")
    pu = schema["labels"].index("前门襟")

    def item(record_id: str, sources: list[str]) -> dict:
        row = _base_row(schema, record_id)
        row["source"] = "+".join(sources)
        row["sources"] = sources
        row["labels"] = [0.0] * 57
        row["known_mask"] = [0] * 57
        row["pu_positive_mask"] = [0] * 57
        row["scores"] = [0.1] * 57
        return row

    jd_positive = item("jd-positive", ["jd_complete23"])
    jd_positive["labels"][pn] = 1.0
    jd_positive["known_mask"][pn] = 1
    jd_positive["scores"][pn] = 0.9
    jd_negative = item("jd-negative", ["jd_complete23"])
    jd_negative["known_mask"][pn] = 1
    dictionary_pn = item("dict-pn", ["dictionary_v4"])
    dictionary_pn["labels"][pn] = 1.0
    dictionary_pn["known_mask"][pn] = 1
    dictionary_pn["scores"][pn] = 0.9
    dictionary_pu = item("dict-pu", ["dictionary_v4"])
    dictionary_pu["labels"][pu] = 1.0
    dictionary_pu["pu_positive_mask"][pu] = 1
    dictionary_pu["scores"][pu] = 0.9
    # This high unknown score proves unknown PN cells stay outside all counts.
    dictionary_pu["scores"][pn] = 0.99
    return [jd_positive, jd_negative, dictionary_pn, dictionary_pu]


def test_recompute_metrics_masks_unknown_and_recomputes_six_values() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    result = recompute_metrics(_metric_rows(schema), _thresholds(schema), schema)
    values = result["values"]
    assert values == {
        "known_micro_f1": 1.0,
        "jd23_micro_f1": 1.0,
        "macro_f1": 1.0,
        "dictionary_positive_macro_recall": 1.0,
        "trusted_negative_specificity": 1.0,
        "json_validity_rate": 1.0,
    }
    hood = result["pn_per_label"]["连帽"]
    assert hood["tp"] == 2
    assert hood["tn"] == 1
    assert hood["known_cells"] == 3
    assert result["dictionary_per_label"]["前门襟"]["positive_support"] == 1


def test_classify_performance_distinguishes_success_partial_and_fail() -> None:
    success = {
        "known_micro_f1": 0.88,
        "jd23_micro_f1": 0.88,
        "macro_f1": 0.75,
        "dictionary_positive_macro_recall": 0.85,
        "trusted_negative_specificity": 0.90,
        "json_validity_rate": 1.0,
    }
    assert classify_performance(success)["verdict"] == "success"
    partial = {**success, "known_micro_f1": 0.84, "jd23_micro_f1": 0.10}
    assert classify_performance(partial)["verdict"] == "partial"
    failed = {**partial, "known_micro_f1": 0.81}
    assert classify_performance(failed)["verdict"] == "fail"


def test_write_partial_audit_reports_is_complete_and_atomic(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    audit = {
        "audit_version": "bosideng-unified57-final-audit-v1",
        "integrity": {"passed": True},
        "verdict": "partial",
        "values": {
            "known_micro_f1": 0.84,
            "jd23_micro_f1": 0.80,
            "macro_f1": 0.70,
            "dictionary_positive_macro_recall": 0.75,
            "trusted_negative_specificity": 0.95,
            "json_validity_rate": 1.0,
        },
        "success_gates": {"known_micro_f1": False},
        "counts": {"validation": 4, "test": 4},
        "warnings": ["PU confidence scores are not calibrated probabilities."],
    }
    per_label = [
        {
            "tag": "连帽",
            "mode": "pn",
            "positive_support": 2,
            "negative_support": 2,
            "tp": 1,
            "fp": 0,
            "fn": 1,
            "tn": 2,
            "precision": 1.0,
            "recall": 0.5,
            "f1": 2 / 3,
            "specificity": 1.0,
        }
    ]
    write_audit_reports(output, audit, per_label)
    assert json.loads((output / "acceptance_audit.json").read_text())["verdict"] == "partial"
    assert "连帽" in (output / "per_label_metrics.csv").read_text(encoding="utf-8")
    report = (output / "FINAL_REPORT.md").read_text(encoding="utf-8")
    assert "partial" in report
    assert "0.8400" in report
    assert not list(output.glob("*.tmp"))


def _selected_output(row: dict, schema: dict) -> dict:
    hood = schema["labels"].index("连帽")
    placket = schema["labels"].index("前门襟")
    return {
        "局部结构": [
            tag
            for tag, index in (("连帽", hood), ("前门襟", placket))
            if row["scores"][index] >= 0.5
        ],
        "廓形": [],
        "工艺": [],
        "面辅料": [],
    }


def _write_output_modes(directory: Path, rows: list[dict], schema: dict) -> None:
    selected = []
    confidence = []
    all_scores = []
    for row in rows:
        names = _selected_output(row, schema)
        positions = {tag: index for index, tag in enumerate(schema["labels"])}
        selected.append({"record_id": row["record_id"], "output": names})
        confidence.append(
            {
                "record_id": row["record_id"],
                "output": {
                    category: [
                        {
                            "name": tag,
                            "confidence": f"{row['scores'][positions[tag]]:.2f}",
                        }
                        for tag in tags
                    ]
                    for category, tags in names.items()
                },
            }
        )
        all_scores.append(
            {
                "record_id": row["record_id"],
                "scores": {
                    tag: (
                        "0.00" if tag == "假两件" else f"{row['scores'][index]:.2f}"
                    )
                    for index, tag in enumerate(schema["labels"])
                },
            }
        )
    _write_jsonl(directory / "test_selected_only.jsonl", selected)
    _write_jsonl(directory / "test_selected_with_confidence.jsonl", confidence)
    _write_jsonl(directory / "test_all_scores.jsonl", all_scores)


def test_three_output_modes_are_strict_and_cross_consistent(tmp_path: Path) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    rows = _metric_rows(schema)
    _write_output_modes(tmp_path, rows, schema)
    result = audit_output_modes(tmp_path, rows, _thresholds(schema), schema)
    assert result["available_modes"] == [
        "selected_only",
        "selected_with_confidence",
        "all_scores",
    ]
    assert result["records_by_mode"] == {
        "selected_only": 4,
        "selected_with_confidence": 4,
        "all_scores": 4,
    }
    assert result["json_validity_rate"] == 1.0
    assert result["cross_mode_consistent"] is True


def test_output_mode_rejects_bad_two_decimal_score(tmp_path: Path) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    rows = _metric_rows(schema)
    _write_output_modes(tmp_path, rows, schema)
    path = tmp_path / "test_all_scores.jsonl"
    payload = [json.loads(line) for line in path.read_text().splitlines()]
    payload[0]["scores"]["连帽"] = "0.900"
    _write_jsonl(path, payload)
    with pytest.raises(AuditContractError, match="two-decimal"):
        audit_output_modes(tmp_path, rows, _thresholds(schema), schema)


def test_output_mode_rejects_unknown_confidence_tag_as_contract_error(tmp_path: Path) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    rows = _metric_rows(schema)
    _write_output_modes(tmp_path, rows, schema)
    path = tmp_path / "test_selected_with_confidence.jsonl"
    payload = [json.loads(line) for line in path.read_text().splitlines()]
    payload[0]["output"]["局部结构"][0]["name"] = "不存在标签"
    _write_jsonl(path, payload)
    with pytest.raises(AuditContractError, match="unknown confidence tag"):
        audit_output_modes(tmp_path, rows, _thresholds(schema), schema)


def test_representatives_select_three_successes_and_three_errors_stably() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    pn = schema["labels"].index("连帽")
    rows: list[dict] = []
    sources = [
        ["jd_complete23"],
        ["dictionary_v4"],
        ["jd_complete23", "dictionary_v4"],
    ]
    for index in range(8):
        row = copy.deepcopy(_metric_rows(schema)[0])
        row["record_id"] = f"representative-{index}"
        row["sources"] = sources[index % len(sources)]
        row["source"] = "+".join(row["sources"])
        row["image_sha256"] = hashlib.sha256(row["record_id"].encode()).hexdigest()
        if index >= 4:
            row["scores"][pn] = 0.1
        rows.append(row)
    first = select_representatives(rows, _thresholds(schema), schema, count=6)
    second = select_representatives(
        list(reversed(rows)), _thresholds(schema), schema, count=6
    )
    assert [row["record_id"] for row in first] == [row["record_id"] for row in second]
    assert Counter(row["outcome"] for row in first) == {"success": 3, "error": 3}
    assert {row["source_bucket"] for row in first} == {"jd", "dictionary", "mixed"}
    assert all("pn_errors" in row and "selected" in row for row in first)


def test_representatives_do_not_backfill_missing_errors_with_successes() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    rows = []
    for index in range(8):
        row = copy.deepcopy(_metric_rows(schema)[0])
        row["record_id"] = f"all-success-{index}"
        row["image_sha256"] = hashlib.sha256(row["record_id"].encode()).hexdigest()
        rows.append(row)
    selected = select_representatives(rows, _thresholds(schema), schema, count=6)
    assert len(selected) == 3
    assert Counter(row["outcome"] for row in selected) == {"success": 3}


def _write_sealed_inventory(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_config.json").write_text('{"labels":57}\n', encoding="utf-8")
    (root / "weights.bin").write_bytes(b"frozen-weights")
    lines = []
    for path in sorted(root.iterdir()):
        if path.name == "SHA256SUMS":
            continue
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    (root / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


def test_sealed_inventory_verifies_every_declared_file(tmp_path: Path) -> None:
    sealed = tmp_path / "sealed"
    _write_sealed_inventory(sealed)
    result = verify_sealed_inventory(sealed)
    assert result["complete"] is True
    assert result["verified_files"] == 2
    assert len(result["sha256s_file_sha256"]) == 64


def test_sealed_inventory_rejects_tampering_and_extra_files(tmp_path: Path) -> None:
    sealed = tmp_path / "sealed"
    _write_sealed_inventory(sealed)
    (sealed / "model_config.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(AuditContractError, match="checksum mismatch"):
        verify_sealed_inventory(sealed)
    _write_sealed_inventory(sealed)
    (sealed / "undeclared.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(AuditContractError, match="inventory differs"):
        verify_sealed_inventory(sealed)


def test_delivery_audit_requires_sealed_verification_json(tmp_path: Path) -> None:
    paths, count, dictionary_labels = _write_full_audit_fixture(tmp_path)
    (paths.delivery_dir / "VERIFICATION.json").unlink()
    _write_sealed_inventory(paths.delivery_dir)
    with pytest.raises(AuditContractError, match="sealed VERIFICATION"):
        audit_delivery(
            paths,
            expected_validation_count=count,
            expected_test_count=count,
            expected_pn_both_class_labels=1,
            expected_dictionary_supported_labels=dictionary_labels,
        )


def _partial_metric_rows(schema: dict) -> list[dict]:
    hood = schema["labels"].index("连帽")
    placket = schema["labels"].index("前门襟")
    rows: list[dict] = []
    for index in range(50):
        row = _base_row(schema, f"jd-{index:02d}")
        row["labels"] = [0.0] * 57
        row["known_mask"] = [0] * 57
        row["pu_positive_mask"] = [0] * 57
        row["known_mask"][hood] = 1
        truth = index < 25
        row["labels"][hood] = float(truth)
        predicted = index < 21 or 25 <= index < 29
        row["scores"] = [0.1] * 57
        row["scores"][hood] = 0.9 if predicted else 0.1
        rows.append(row)
    dictionary_pn = copy.deepcopy(rows[0])
    dictionary_pn["record_id"] = "dictionary-pn"
    dictionary_pn["source"] = "dictionary_v4"
    dictionary_pn["sources"] = ["dictionary_v4"]
    dictionary_pn["image_sha256"] = hashlib.sha256(b"dictionary-pn").hexdigest()
    dictionary_pu = copy.deepcopy(rows[0])
    dictionary_pu["record_id"] = "dictionary-pu"
    dictionary_pu["source"] = "dictionary_v4"
    dictionary_pu["sources"] = ["dictionary_v4"]
    dictionary_pu["image_sha256"] = hashlib.sha256(b"dictionary-pu").hexdigest()
    dictionary_pu["labels"] = [0.0] * 57
    dictionary_pu["known_mask"] = [0] * 57
    dictionary_pu["pu_positive_mask"] = [0] * 57
    dictionary_pu["labels"][placket] = 1.0
    dictionary_pu["pu_positive_mask"][placket] = 1
    dictionary_pu["scores"] = [0.1] * 57
    dictionary_pu["scores"][placket] = 0.9
    rows.extend([dictionary_pn, dictionary_pu])
    return rows


def _evaluator_pn(report: dict) -> dict:
    return {
        "micro": {
            **report["overall_36pn"]["micro"],
            "known_positive": report["overall_36pn"]["micro"]["positive_support"],
            "known_negative": report["overall_36pn"]["micro"]["negative_support"],
        },
        "macro": {
            "f1_both_class_labels": report["overall_36pn"]["macro_f1"],
            "labels_with_both_classes": report["overall_36pn"][
                "labels_with_both_classes"
            ],
        },
        "per_label": {
            tag: {
                **values,
                "known_positive": values["positive_support"],
                "known_negative": values["negative_support"],
            }
            for tag, values in report["pn_per_label"].items()
        },
        "trusted_negatives": {
            "specificity": report["overall_36pn"]["micro"]["specificity"]
        },
    }


def _write_full_audit_fixture(
    tmp_path: Path, *, partial: bool = False, failed: bool = False
) -> tuple[AuditPaths, int, int]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    rows = _partial_metric_rows(schema) if partial or failed else _metric_rows(schema)
    if failed:
        hood = schema["labels"].index("连帽")
        for row in rows:
            row["scores"][hood] = 0.1
    count = len(rows)
    dataset = tmp_path / "dataset"
    evaluation = tmp_path / "evaluation"
    posttrain = tmp_path / "posttrain"
    candidate = tmp_path / "delivery_candidate"
    output = tmp_path / "audit"
    sealed = tmp_path / "sealed"
    manifest_rows = []
    for row in rows:
        manifest = copy.deepcopy(row)
        manifest.pop("scores")
        manifest_rows.append(manifest)
    prediction_rows = copy.deepcopy(rows)
    for row in prediction_rows:
        row["checkpoint_sha256"] = "c" * 64
    _write_jsonl(dataset / "val.jsonl", manifest_rows)
    _write_jsonl(dataset / "test.jsonl", manifest_rows)
    _write_jsonl(dataset / "train.jsonl", manifest_rows)
    _write_jsonl(evaluation / "validation_predictions_float32.jsonl", prediction_rows)
    _write_jsonl(evaluation / "test_predictions_float32.jsonl", prediction_rows)
    leakage = {
        "passed": True,
        "cross_split_components": [],
        "cross_split_exact_phash": [],
        "cross_split_sha256": [],
    }
    (dataset / "leakage_check.json").write_text(json.dumps(leakage), encoding="utf-8")
    thresholds = _thresholds(schema)
    thresholds["calibration_records"] = count
    thresholds["validation_manifest_sha256"] = hashlib.sha256(
        (dataset / "val.jsonl").read_bytes()
    ).hexdigest()
    for index, tag in enumerate(schema["labels"]):
        mode = schema["label_training_modes"][tag]
        if mode == "pn":
            thresholds["labels"][tag]["support"] = {
                "known_positive": sum(
                    bool(row["known_mask"][index]) and row["labels"][index] == 1.0
                    for row in prediction_rows
                ),
                "known_negative": sum(
                    bool(row["known_mask"][index]) and row["labels"][index] == 0.0
                    for row in prediction_rows
                ),
            }
        elif mode == "pu":
            thresholds["labels"][tag]["support"] = {
                "positive": sum(
                    bool(row["pu_positive_mask"][index]) for row in prediction_rows
                ),
                "unlabeled": sum(
                    not bool(row["known_mask"][index])
                    and not bool(row["pu_positive_mask"][index])
                    for row in prediction_rows
                ),
            }
        else:
            thresholds["labels"][tag]["support"] = {}
    (evaluation / "thresholds.json").write_text(
        json.dumps(thresholds, ensure_ascii=False), encoding="utf-8"
    )
    _write_output_modes(evaluation, prediction_rows, schema)
    recomputed = recompute_metrics(prediction_rows, thresholds, schema)
    classification = classify_performance(recomputed["values"])
    dictionary = recomputed["dictionary_all_positive_views"]
    evaluation_report = {
        "status": classification["verdict"],
        "provenance": {
            "schema_sha256": schema["schema_sha256"],
            "schema_file_sha256": hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest(),
            "checkpoint_sha256": "c" * 64,
            "validation_manifest_sha256": hashlib.sha256(
                (dataset / "val.jsonl").read_bytes()
            ).hexdigest(),
            "test_manifest_sha256": hashlib.sha256(
                (dataset / "test.jsonl").read_bytes()
            ).hexdigest(),
            "validation_predictions_sha256": hashlib.sha256(
                (evaluation / "validation_predictions_float32.jsonl").read_bytes()
            ).hexdigest(),
            "predictions_sha256": hashlib.sha256(
                (evaluation / "test_predictions_float32.jsonl").read_bytes()
            ).hexdigest(),
            "thresholds_sha256": hashlib.sha256(
                (evaluation / "thresholds.json").read_bytes()
            ).hexdigest(),
        },
        "validation": {"expected": count, "predicted": count, "complete": True},
        "test": {"expected": count, "predicted": count, "complete": True},
        "classification": classification,
        "raw_thresholded": recomputed["raw_thresholded"],
        "final_format": recomputed["final_format"],
        "format_constraint_loss": recomputed["format_constraint_loss"],
        "dictionary_all_positive": dictionary,
        "performance": recomputed["performance"],
        "output_quality": {"json_validity_rate": 1.0},
        "validation_metrics": {
            "raw_thresholded": recomputed["raw_thresholded"],
            "final_format": recomputed["final_format"],
            "format_constraint_loss": recomputed["format_constraint_loss"],
            "dictionary_all_positive": dictionary,
        },
        "threshold_calibration": {
            "pu_fallback_count": 20,
            "pu_fallback_labels": [
                tag
                for tag in schema["labels"]
                if schema["label_training_modes"][tag] == "pu"
            ],
        },
        "process_cleanup": {"complete": True},
    }
    (evaluation / "evaluation_report.json").write_text(
        json.dumps(evaluation_report, ensure_ascii=False), encoding="utf-8"
    )
    preflight = {
        "dataset_sha256": {
            "train": hashlib.sha256((dataset / "train.jsonl").read_bytes()).hexdigest(),
            "val": hashlib.sha256((dataset / "val.jsonl").read_bytes()).hexdigest(),
            "test": hashlib.sha256((dataset / "test.jsonl").read_bytes()).hexdigest(),
        },
        "leakage_check_sha256": hashlib.sha256(
            (dataset / "leakage_check.json").read_bytes()
        ).hexdigest(),
        "leakage_passed": True,
        "leakage_counts": {
            "cross_split_components": 0,
            "cross_split_exact_phash": 0,
            "cross_split_sha256": 0,
        },
    }
    posttrain.mkdir(parents=True)
    (posttrain / "preflight_contract.json").write_text(json.dumps(preflight), encoding="utf-8")
    candidate.mkdir(parents=True)
    for name in ("lora_and_classifier.safetensors", "model_config.json", "infer.py"):
        (candidate / name).write_bytes(f"synthetic-{name}".encode())
    verification = candidate / "verification"
    verification.mkdir()
    reproduction_count = min(32, count)
    verification_manifest = [
        {
            "record_id": row["record_id"],
            "image_path": row["image_path"],
            "image_sha256": row["image_sha256"],
        }
        for row in manifest_rows[:reproduction_count]
    ]
    reference_float = prediction_rows[:reproduction_count]
    selected_rows = [
        json.loads(line)
        for line in (evaluation / "test_selected_only.jsonl").read_text().splitlines()
    ][:reproduction_count]
    reproduced_selected_rows = []
    image_sha_by_id = {
        row["record_id"]: row["image_sha256"] for row in verification_manifest
    }
    for row in selected_rows:
        item = copy.deepcopy(row)
        item["image_sha256"] = image_sha_by_id[item["record_id"]]
        reproduced_selected_rows.append(item)
    _write_jsonl(verification / "verification_32_manifest.jsonl", verification_manifest)
    _write_jsonl(verification / "reference_32_float32.jsonl", reference_float)
    _write_jsonl(verification / "reference_32_selected_only.jsonl", selected_rows)
    evaluation_verification = evaluation / "verification"
    evaluation_verification.mkdir()
    _write_jsonl(
        evaluation_verification / "verification_32_manifest.jsonl",
        verification_manifest,
    )
    _write_jsonl(
        evaluation_verification / "reference_32_float32.jsonl", reference_float
    )
    _write_jsonl(
        evaluation_verification / "reference_32_selected_only.jsonl", selected_rows
    )
    (candidate / "VERIFICATION.json").write_text(
        json.dumps(
            {
                "status": "pending_reproduction",
                "provenance": {
                    "weights_sha256": hashlib.sha256(
                        (candidate / "lora_and_classifier.safetensors").read_bytes()
                    ).hexdigest(),
                    "checkpoint_sha256": "c" * 64,
                    "schema_file_sha256": hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest(),
                    "thresholds_sha256": hashlib.sha256(
                        (evaluation / "thresholds.json").read_bytes()
                    ).hexdigest(),
                    "metrics_sha256": hashlib.sha256(
                        (evaluation / "evaluation_report.json").read_bytes()
                    ).hexdigest(),
                    "test_manifest_sha256": hashlib.sha256(
                        (dataset / "test.jsonl").read_bytes()
                    ).hexdigest(),
                },
                "references": {
                    name: hashlib.sha256((verification / name).read_bytes()).hexdigest()
                    for name in (
                        "verification_32_manifest.jsonl",
                        "reference_32_float32.jsonl",
                        "reference_32_selected_only.jsonl",
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    reproduced_float = posttrain / "reproduced_32_float32.jsonl"
    reproduced_selected = posttrain / "reproduced_32_selected_only.jsonl"
    _write_jsonl(reproduced_float, reference_float)
    _write_jsonl(reproduced_selected, reproduced_selected_rows)
    reproduction = {
        "candidate_weights_sha256": hashlib.sha256(
            (candidate / "lora_and_classifier.safetensors").read_bytes()
        ).hexdigest(),
        "candidate_model_config_sha256": hashlib.sha256(
            (candidate / "model_config.json").read_bytes()
        ).hexdigest(),
        "candidate_infer_sha256": hashlib.sha256(
            (candidate / "infer.py").read_bytes()
        ).hexdigest(),
        "reproduced_float32_path": str(reproduced_float),
        "reproduced_float32_sha256": hashlib.sha256(reproduced_float.read_bytes()).hexdigest(),
        "reproduced_selected_only_path": str(reproduced_selected),
        "reproduced_selected_only_sha256": hashlib.sha256(
            reproduced_selected.read_bytes()
        ).hexdigest(),
        "commands": [
            f"python {candidate / 'infer.py'} --mode verification_float32",
            f"python {candidate / 'infer.py'} --mode selected_only",
        ],
        "environment": {
            key: "synthetic"
            for key in (
                "gpu",
                "cuda",
                "pytorch",
                "transformers",
                "peft",
                "safetensors",
                "pillow",
            )
        },
    }
    (posttrain / "reproduction_result.json").write_text(
        json.dumps(reproduction), encoding="utf-8"
    )
    final_mode_dir = posttrain / "final_mode_verification"
    final_mode_dir.mkdir()
    confidence_reference = [
        json.loads(line)
        for line in (evaluation / "test_selected_with_confidence.jsonl")
        .read_text()
        .splitlines()
    ][:reproduction_count]
    all_scores_reference = [
        json.loads(line)
        for line in (evaluation / "test_all_scores.jsonl").read_text().splitlines()
    ][:reproduction_count]
    confidence_reproduced = []
    all_scores_reproduced = []
    manifest_by_id = {row["record_id"]: row for row in manifest_rows}
    for row in confidence_reference:
        evidence = manifest_by_id[row["record_id"]]
        confidence_reproduced.append(
            {
                "record_id": row["record_id"],
                "image_path": evidence["image_path"],
                "image_sha256": evidence["image_sha256"],
                "output": row["output"],
            }
        )
    for row in all_scores_reference:
        evidence = manifest_by_id[row["record_id"]]
        all_scores_reproduced.append(
            {
                "record_id": row["record_id"],
                "image_path": evidence["image_path"],
                "image_sha256": evidence["image_sha256"],
                "output": {"scores": row["scores"]},
            }
        )
    confidence_path = final_mode_dir / "reproduced_32_selected_with_confidence.jsonl"
    all_scores_path = final_mode_dir / "reproduced_32_all_scores.jsonl"
    _write_jsonl(confidence_path, confidence_reproduced)
    _write_jsonl(all_scores_path, all_scores_reproduced)
    final_mode_summary = {
        "status": "success",
        "records": reproduction_count,
        "score_values": reproduction_count * 57,
        "source_float32_sha256": hashlib.sha256(reproduced_float.read_bytes()).hexdigest(),
        "source_selected_only_sha256": hashlib.sha256(
            reproduced_selected.read_bytes()
        ).hexdigest(),
        "float32_json_roundtrip_exact": True,
        "selected_only_reformatted_exact": True,
        "selected_with_confidence_names_exact": True,
        "confidence_two_decimal": True,
        "all_scores_exactly_57": True,
        "all_scores_schema_order_exact": True,
        "all_scores_two_decimal": True,
        "unsupported_假两件_fixed_0.00": True,
        "selected_with_confidence_sha256": hashlib.sha256(
            confidence_path.read_bytes()
        ).hexdigest(),
        "all_scores_sha256": hashlib.sha256(all_scores_path.read_bytes()).hexdigest(),
        "candidate_infer_sha256": hashlib.sha256(
            (candidate / "infer.py").read_bytes()
        ).hexdigest(),
        "reproduction_result_sha256": hashlib.sha256(
            (posttrain / "reproduction_result.json").read_bytes()
        ).hexdigest(),
        "commands": [
            f"python {candidate / 'infer.py'} --scores-json scores.json --mode selected_with_confidence",
            f"python {candidate / 'infer.py'} --scores-json scores.json --mode all_scores",
        ],
        "environment": reproduction["environment"],
        "sealed_delivery_verification_status": classification["verdict"],
        "sealed_delivery_customer_ready": classification["verdict"] == "success",
    }
    (final_mode_dir / "final_mode_verification.json").write_text(
        json.dumps(final_mode_summary, ensure_ascii=False), encoding="utf-8"
    )
    final_report = {
        "status": classification["verdict"],
        "customer_ready": classification["verdict"] == "success",
        "completed_before_deadline": True,
        "training": {"checkpoint_sha256": "c" * 64},
        "evaluation": {
            "status": classification["verdict"],
            "provenance": evaluation_report["provenance"],
        },
    }
    (posttrain / "final_report.json").write_text(json.dumps(final_report), encoding="utf-8")
    sealed.mkdir(parents=True, exist_ok=True)
    (sealed / "VERIFICATION.json").write_text(
        json.dumps(
            {
                "status": classification["verdict"],
                "customer_ready": classification["verdict"] == "success",
                "result_sha256": hashlib.sha256(
                    (posttrain / "reproduction_result.json").read_bytes()
                ).hexdigest(),
                "provenance": {
                    "weights_sha256": hashlib.sha256(
                        (candidate / "lora_and_classifier.safetensors").read_bytes()
                    ).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )
    _write_sealed_inventory(sealed)
    return (
        AuditPaths(
            schema=SCHEMA_PATH,
            dataset_root=dataset,
            evaluation_dir=evaluation,
            posttrain_dir=posttrain,
            output_dir=output,
            delivery_dir=sealed,
        ),
        count,
        recomputed["dictionary_all_positive"]["labels_with_positive_support"],
    )


def test_audit_posttrain_verifies_reproduction_hashes(tmp_path: Path) -> None:
    paths, _count, _dictionary_labels = _write_full_audit_fixture(tmp_path)
    result = audit_posttrain(paths.posttrain_dir)
    assert result["complete"] is True
    assert result["status"] == "success"
    assert result["reproduction_outputs_verified"] == 2


def test_audit_posttrain_accepts_fail_report_without_reproduction(tmp_path: Path) -> None:
    paths, _count, _dictionary_labels = _write_full_audit_fixture(tmp_path)
    (paths.posttrain_dir / "final_report.json").write_text(
        json.dumps(
            {
                "status": "fail",
                "customer_ready": False,
                "stage": "evaluate",
                "error": "synthetic failure",
            }
        ),
        encoding="utf-8",
    )
    (paths.posttrain_dir / "reproduction_result.json").unlink()
    result = audit_posttrain(paths.posttrain_dir)
    assert result["complete"] is True
    assert result["status"] == "fail"
    assert result["reproduction_outputs_verified"] == 0
    assert result["reproduction_result_sha256"] is None


def test_reproduction_bundle_verifies_candidate_references_and_exact_outputs(
    tmp_path: Path,
) -> None:
    paths, count, _dictionary_labels = _write_full_audit_fixture(tmp_path)
    result = audit_reproduction_bundle(
        paths.posttrain_dir,
        paths.posttrain_dir.parent / "delivery_candidate",
        evaluation_dir=paths.evaluation_dir,
        test_rows=[
            json.loads(line)
            for line in (
                paths.evaluation_dir / "test_predictions_float32.jsonl"
            ).read_text().splitlines()
        ],
        thresholds=json.loads(
            (paths.evaluation_dir / "thresholds.json").read_text()
        ),
        schema=load_schema(paths.schema),
        expected_records=min(32, count),
    )
    assert result["records"] == min(32, count)
    assert result["score_values"] == min(32, count) * 57
    assert result["probabilities_exact"] is True
    assert result["selected_outputs_exact"] is True
    assert result["candidate_hashes_verified"] == 3


def test_reproduction_bundle_rejects_float_score_drift(tmp_path: Path) -> None:
    paths, count, _dictionary_labels = _write_full_audit_fixture(tmp_path)
    reproduced = paths.posttrain_dir / "reproduced_32_float32.jsonl"
    rows = [json.loads(line) for line in reproduced.read_text().splitlines()]
    rows[0]["scores"][0] = 0.123456
    _write_jsonl(reproduced, rows)
    result_path = paths.posttrain_dir / "reproduction_result.json"
    result = json.loads(result_path.read_text())
    result["reproduced_float32_sha256"] = hashlib.sha256(
        reproduced.read_bytes()
    ).hexdigest()
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(AuditContractError, match="float32 scores differ"):
        audit_reproduction_bundle(
            paths.posttrain_dir,
            paths.posttrain_dir.parent / "delivery_candidate",
            evaluation_dir=paths.evaluation_dir,
            test_rows=[
                json.loads(line)
                for line in (
                    paths.evaluation_dir / "test_predictions_float32.jsonl"
                ).read_text().splitlines()
            ],
            thresholds=json.loads(
                (paths.evaluation_dir / "thresholds.json").read_text()
            ),
            schema=load_schema(paths.schema),
            expected_records=min(32, count),
        )


def test_reproduction_bundle_rejects_self_signed_reference_drift_from_evaluation(
    tmp_path: Path,
) -> None:
    paths, count, dictionary_labels = _write_full_audit_fixture(tmp_path)
    candidate = paths.posttrain_dir.parent / "delivery_candidate"
    reference_path = candidate / "verification" / "reference_32_float32.jsonl"
    reference = [json.loads(line) for line in reference_path.read_text().splitlines()]
    reference[0]["scores"][0] = 0.22
    _write_jsonl(reference_path, reference)
    candidate_verification_path = candidate / "VERIFICATION.json"
    candidate_verification = json.loads(candidate_verification_path.read_text())
    candidate_verification["references"]["reference_32_float32.jsonl"] = hashlib.sha256(
        reference_path.read_bytes()
    ).hexdigest()
    candidate_verification_path.write_text(
        json.dumps(candidate_verification), encoding="utf-8"
    )
    reproduced_path = paths.posttrain_dir / "reproduced_32_float32.jsonl"
    _write_jsonl(reproduced_path, reference)
    reproduction_path = paths.posttrain_dir / "reproduction_result.json"
    reproduction = json.loads(reproduction_path.read_text())
    reproduction["reproduced_float32_sha256"] = hashlib.sha256(
        reproduced_path.read_bytes()
    ).hexdigest()
    reproduction_path.write_text(json.dumps(reproduction), encoding="utf-8")
    with pytest.raises(AuditContractError, match="formal evaluation verification"):
        audit_delivery(
            paths,
            expected_validation_count=count,
            expected_test_count=count,
            expected_pn_both_class_labels=1,
            expected_dictionary_supported_labels=dictionary_labels,
        )


def test_packaged_output_modes_verify_all_32_replayed_records(tmp_path: Path) -> None:
    paths, count, _dictionary_labels = _write_full_audit_fixture(tmp_path)
    schema = load_schema(paths.schema)
    result = audit_packaged_output_modes(
        paths.posttrain_dir,
        schema,
        expected_records=min(32, count),
    )
    assert result["complete"] is True
    assert result["records"] == min(32, count)
    assert result["valid_records"] == min(32, count) * 2
    assert result["all_scores_schema_order_exact"] is True
    assert result["selected_with_confidence_names_exact"] is True


def test_packaged_output_modes_reject_wrong_all_scores_inventory(tmp_path: Path) -> None:
    paths, count, _dictionary_labels = _write_full_audit_fixture(tmp_path)
    schema = load_schema(paths.schema)
    mode_dir = paths.posttrain_dir / "final_mode_verification"
    path = mode_dir / "reproduced_32_all_scores.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["output"]["scores"].pop("连帽")
    _write_jsonl(path, rows)
    summary_path = mode_dir / "final_mode_verification.json"
    summary = json.loads(summary_path.read_text())
    summary["all_scores_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary_path.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(AuditContractError, match="57 schema-ordered"):
        audit_packaged_output_modes(
            paths.posttrain_dir,
            schema,
            expected_records=min(32, count),
        )


def test_packaged_output_modes_require_candidate_scores_json_execution_binding(
    tmp_path: Path,
) -> None:
    paths, count, _dictionary_labels = _write_full_audit_fixture(tmp_path)
    summary_path = (
        paths.posttrain_dir
        / "final_mode_verification"
        / "final_mode_verification.json"
    )
    summary = json.loads(summary_path.read_text())
    summary["commands"] = ["echo fabricated"]
    summary_path.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(AuditContractError, match="scores-json"):
        audit_packaged_output_modes(
            paths.posttrain_dir,
            load_schema(paths.schema),
            expected_records=min(32, count),
        )


def test_full_success_audit_writes_all_artifacts(tmp_path: Path) -> None:
    paths, count, dictionary_labels = _write_full_audit_fixture(tmp_path)
    audit = audit_delivery(
        paths,
        expected_validation_count=count,
        expected_test_count=count,
        expected_pn_both_class_labels=1,
        expected_dictionary_supported_labels=dictionary_labels,
    )
    assert audit["verdict"] == "success"
    assert audit["integrity"]["passed"] is True
    assert audit["sealed_delivery"]["complete"] is True
    assert audit["output_modes"]["complete"] is True
    assert audit["raw_values"] == recompute_metrics(
        [
            json.loads(line)
            for line in (
                paths.evaluation_dir / "test_predictions_float32.jsonl"
            ).read_text().splitlines()
        ],
        json.loads((paths.evaluation_dir / "thresholds.json").read_text()),
        load_schema(paths.schema),
    )["raw_values"]
    assert audit["supervision_counts"]["test"] == {
        "pn_known_positive": 2,
        "pn_known_negative": 1,
        "pn_known_cells": 3,
        "pn_unknown": 141,
        "pu_positive": 1,
        "pu_unlabeled": 79,
        "unsupported_unknown": 4,
    }
    assert (paths.output_dir / "acceptance_audit.json").is_file()
    assert (paths.output_dir / "per_label_metrics.csv").is_file()
    assert (paths.output_dir / "FINAL_REPORT.md").is_file()
    assert (paths.output_dir / "output_modes_audit.json").is_file()
    assert len((paths.output_dir / "representative_selection.jsonl").read_text().splitlines()) == 3
    assert audit["representative_outcomes"] == {"success": 3, "error": 0}
    assert any("representative" in warning for warning in audit["warnings"])


def test_audit_rejects_raw_view_pu_or_format_metric_drift(tmp_path: Path) -> None:
    paths, count, dictionary_labels = _write_full_audit_fixture(tmp_path)
    report_path = paths.evaluation_dir / "evaluation_report.json"
    report = json.loads(report_path.read_text())
    report["raw_thresholded"]["pu"]["summary"][
        "micro_unlabeled_coverage"
    ] += 0.01
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(AuditContractError, match="test.raw_thresholded.pu.summary"):
        audit_delivery(
            paths,
            expected_validation_count=count,
            expected_test_count=count,
            expected_pn_both_class_labels=1,
            expected_dictionary_supported_labels=dictionary_labels,
        )


def test_success_metrics_with_partial_posttrain_returns_partial_not_integrity(
    tmp_path: Path,
) -> None:
    paths, count, dictionary_labels = _write_full_audit_fixture(tmp_path)
    post_report = json.loads((paths.posttrain_dir / "final_report.json").read_text())
    post_report["status"] = "partial"
    post_report["customer_ready"] = False
    (paths.posttrain_dir / "final_report.json").write_text(
        json.dumps(post_report), encoding="utf-8"
    )
    verification_path = paths.delivery_dir / "VERIFICATION.json"
    verification = json.loads(verification_path.read_text())
    verification["status"] = "partial"
    verification["customer_ready"] = False
    verification_path.write_text(json.dumps(verification), encoding="utf-8")
    mode_summary_path = (
        paths.posttrain_dir
        / "final_mode_verification"
        / "final_mode_verification.json"
    )
    mode_summary = json.loads(mode_summary_path.read_text())
    mode_summary["sealed_delivery_verification_status"] = "partial"
    mode_summary["sealed_delivery_customer_ready"] = False
    mode_summary_path.write_text(
        json.dumps(mode_summary, ensure_ascii=False), encoding="utf-8"
    )
    _write_sealed_inventory(paths.delivery_dir)
    audit = audit_delivery(
        paths,
        expected_validation_count=count,
        expected_test_count=count,
        expected_pn_both_class_labels=1,
        expected_dictionary_supported_labels=dictionary_labels,
    )
    assert audit["performance_verdict"] == "success"
    assert audit["verdict"] == "partial"
    assert audit["integrity"]["passed"] is True
    assert audit["customer_ready"] is False


def test_audit_rejects_output_directory_inside_any_input_tree(tmp_path: Path) -> None:
    paths, count, dictionary_labels = _write_full_audit_fixture(tmp_path)
    unsafe = AuditPaths(
        schema=paths.schema,
        dataset_root=paths.dataset_root,
        evaluation_dir=paths.evaluation_dir,
        posttrain_dir=paths.posttrain_dir,
        output_dir=paths.dataset_root / "audit-output",
        delivery_dir=paths.delivery_dir,
    )
    with pytest.raises(AuditContractError, match="outside every input tree"):
        audit_delivery(
            unsafe,
            expected_validation_count=count,
            expected_test_count=count,
            expected_pn_both_class_labels=1,
            expected_dictionary_supported_labels=dictionary_labels,
        )
    assert not unsafe.output_dir.exists()


def test_audit_rejects_prediction_checkpoint_drift(tmp_path: Path) -> None:
    paths, count, dictionary_labels = _write_full_audit_fixture(tmp_path)
    prediction_path = paths.evaluation_dir / "test_predictions_float32.jsonl"
    predictions = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    predictions[0]["checkpoint_sha256"] = "d" * 64
    _write_jsonl(prediction_path, predictions)
    with pytest.raises(AuditContractError, match="checkpoint"):
        audit_delivery(
            paths,
            expected_validation_count=count,
            expected_test_count=count,
            expected_pn_both_class_labels=1,
            expected_dictionary_supported_labels=dictionary_labels,
        )


def test_audit_rejects_posttrain_preflight_manifest_drift(tmp_path: Path) -> None:
    paths, count, dictionary_labels = _write_full_audit_fixture(tmp_path)
    preflight_path = paths.posttrain_dir / "preflight_contract.json"
    preflight = json.loads(preflight_path.read_text())
    preflight["dataset_sha256"]["test"] = "d" * 64
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    with pytest.raises(AuditContractError, match="preflight.*test"):
        audit_delivery(
            paths,
            expected_validation_count=count,
            expected_test_count=count,
            expected_pn_both_class_labels=1,
            expected_dictionary_supported_labels=dictionary_labels,
        )


def test_audit_requires_all_20_pu_fallback_thresholds_and_exact_support(
    tmp_path: Path,
) -> None:
    paths, count, dictionary_labels = _write_full_audit_fixture(tmp_path)
    threshold_path = paths.evaluation_dir / "thresholds.json"
    thresholds = json.loads(threshold_path.read_text())
    thresholds["labels"]["前门襟"]["threshold"] = 0.4
    threshold_path.write_text(json.dumps(thresholds, ensure_ascii=False), encoding="utf-8")
    evaluation_report_path = paths.evaluation_dir / "evaluation_report.json"
    report = json.loads(evaluation_report_path.read_text())
    report["provenance"]["thresholds_sha256"] = hashlib.sha256(
        threshold_path.read_bytes()
    ).hexdigest()
    evaluation_report_path.write_text(
        json.dumps(report, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(AuditContractError, match="PU fallback threshold"):
        audit_delivery(
            paths,
            expected_validation_count=count,
            expected_test_count=count,
            expected_pn_both_class_labels=1,
            expected_dictionary_supported_labels=dictionary_labels,
        )


def test_audit_rejects_candidate_and_sealed_reproduction_binding_drift(
    tmp_path: Path,
) -> None:
    paths, count, dictionary_labels = _write_full_audit_fixture(tmp_path)
    candidate_verification_path = (
        paths.posttrain_dir.parent / "delivery_candidate" / "VERIFICATION.json"
    )
    candidate_verification = json.loads(candidate_verification_path.read_text())
    candidate_verification["provenance"]["checkpoint_sha256"] = "d" * 64
    candidate_verification_path.write_text(
        json.dumps(candidate_verification), encoding="utf-8"
    )
    with pytest.raises(AuditContractError, match="candidate.*checkpoint"):
        audit_delivery(
            paths,
            expected_validation_count=count,
            expected_test_count=count,
            expected_pn_both_class_labels=1,
            expected_dictionary_supported_labels=dictionary_labels,
        )


def test_audit_rejects_sealed_reproduction_result_binding_drift(tmp_path: Path) -> None:
    paths, count, dictionary_labels = _write_full_audit_fixture(tmp_path)
    verification_path = paths.delivery_dir / "VERIFICATION.json"
    verification = json.loads(verification_path.read_text())
    verification["result_sha256"] = "d" * 64
    verification_path.write_text(json.dumps(verification), encoding="utf-8")
    _write_sealed_inventory(paths.delivery_dir)
    with pytest.raises(AuditContractError, match="sealed delivery reproduction"):
        audit_delivery(
            paths,
            expected_validation_count=count,
            expected_test_count=count,
            expected_pn_both_class_labels=1,
            expected_dictionary_supported_labels=dictionary_labels,
        )


def test_cli_partial_returns_one_and_still_writes_all_reports(tmp_path: Path) -> None:
    paths, count, dictionary_labels = _write_full_audit_fixture(tmp_path, partial=True)
    code = main(
        [
            "--schema",
            str(paths.schema),
            "--dataset-root",
            str(paths.dataset_root),
            "--evaluation-dir",
            str(paths.evaluation_dir),
            "--posttrain-dir",
            str(paths.posttrain_dir),
            "--output-dir",
            str(paths.output_dir),
            "--delivery-dir",
            str(paths.delivery_dir),
            "--expected-validation-count",
            str(count),
            "--expected-test-count",
            str(count),
            "--expected-pn-both-class-labels",
            "1",
            "--expected-dictionary-supported-labels",
            str(dictionary_labels),
        ]
    )
    assert code == 1
    audit = json.loads((paths.output_dir / "acceptance_audit.json").read_text())
    assert audit["verdict"] == "partial"
    assert (paths.output_dir / "per_label_metrics.csv").is_file()
    assert (paths.output_dir / "FINAL_REPORT.md").is_file()


def test_cli_true_fail_without_reproduction_returns_one_and_complete_reports(
    tmp_path: Path,
) -> None:
    paths, count, dictionary_labels = _write_full_audit_fixture(tmp_path, failed=True)
    (paths.posttrain_dir / "reproduction_result.json").unlink()
    code = main(
        [
            "--schema",
            str(paths.schema),
            "--dataset-root",
            str(paths.dataset_root),
            "--evaluation-dir",
            str(paths.evaluation_dir),
            "--posttrain-dir",
            str(paths.posttrain_dir),
            "--output-dir",
            str(paths.output_dir),
            "--expected-validation-count",
            str(count),
            "--expected-test-count",
            str(count),
            "--expected-pn-both-class-labels",
            "1",
            "--expected-dictionary-supported-labels",
            str(dictionary_labels),
        ]
    )
    assert code == 1
    audit = json.loads((paths.output_dir / "acceptance_audit.json").read_text())
    assert audit["verdict"] == "fail"
    assert audit["performance_verdict"] == "fail"
    assert audit["posttrain"]["reproduction_outputs_verified"] == 0
    assert (paths.output_dir / "per_label_metrics.csv").is_file()
    assert (paths.output_dir / "FINAL_REPORT.md").is_file()
    assert (paths.output_dir / "output_modes_audit.json").is_file()
    assert (paths.output_dir / "representative_selection.jsonl").is_file()


def test_cli_mask_integrity_error_returns_two_and_writes_diagnostics(tmp_path: Path) -> None:
    paths, count, dictionary_labels = _write_full_audit_fixture(tmp_path)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    (paths.output_dir / "output_modes_audit.json").write_text(
        '{"stale":true}', encoding="utf-8"
    )
    (paths.output_dir / "representative_selection.jsonl").write_text(
        '{"stale":true}\n', encoding="utf-8"
    )
    manifest_path = paths.dataset_root / "val.jsonl"
    prediction_path = paths.evaluation_dir / "validation_predictions_float32.jsonl"
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    predictions = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    pu_index = json.loads(SCHEMA_PATH.read_text())["labels"].index("前门襟")
    manifest[0]["labels"][pu_index] = 1.0
    predictions[0]["labels"][pu_index] = 1.0
    _write_jsonl(manifest_path, manifest)
    _write_jsonl(prediction_path, predictions)
    thresholds_path = paths.evaluation_dir / "thresholds.json"
    thresholds = json.loads(thresholds_path.read_text())
    thresholds["validation_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    thresholds_path.write_text(json.dumps(thresholds), encoding="utf-8")
    code = main(
        [
            "--schema",
            str(paths.schema),
            "--dataset-root",
            str(paths.dataset_root),
            "--evaluation-dir",
            str(paths.evaluation_dir),
            "--posttrain-dir",
            str(paths.posttrain_dir),
            "--output-dir",
            str(paths.output_dir),
            "--expected-validation-count",
            str(count),
            "--expected-test-count",
            str(count),
            "--expected-pn-both-class-labels",
            "1",
            "--expected-dictionary-supported-labels",
            str(dictionary_labels),
        ]
    )
    assert code == 2
    audit = json.loads((paths.output_dir / "acceptance_audit.json").read_text())
    assert audit["exit_code"] == 2
    assert audit["verdict"] == "integrity_error"
    assert "unknown cell" in audit["errors"][0]
    assert (paths.output_dir / "per_label_metrics.csv").is_file()
    assert "unknown cell" in (paths.output_dir / "FINAL_REPORT.md").read_text()
    assert not (paths.output_dir / "output_modes_audit.json").exists()
    assert not (paths.output_dir / "representative_selection.jsonl").exists()


def test_cli_type_error_in_untrusted_json_returns_two_with_diagnostics(
    tmp_path: Path,
) -> None:
    paths, count, dictionary_labels = _write_full_audit_fixture(tmp_path)
    threshold_path = paths.evaluation_dir / "thresholds.json"
    thresholds = json.loads(threshold_path.read_text())
    thresholds["calibration_records"] = None
    threshold_path.write_text(json.dumps(thresholds), encoding="utf-8")
    code = main(
        [
            "--schema",
            str(paths.schema),
            "--dataset-root",
            str(paths.dataset_root),
            "--evaluation-dir",
            str(paths.evaluation_dir),
            "--posttrain-dir",
            str(paths.posttrain_dir),
            "--output-dir",
            str(paths.output_dir),
            "--expected-validation-count",
            str(count),
            "--expected-test-count",
            str(count),
            "--expected-pn-both-class-labels",
            "1",
            "--expected-dictionary-supported-labels",
            str(dictionary_labels),
        ]
    )
    assert code == 2
    audit = json.loads((paths.output_dir / "acceptance_audit.json").read_text())
    assert audit["verdict"] == "integrity_error"
    assert audit["exit_code"] == 2
