import json
from pathlib import Path

import pytest

from scripts.unified57_evaluation_core import (
    BufferedPredictionShard,
    calibrate_thresholds,
    evaluate_pn_slice,
    evaluate_pu_label,
    evaluate_views,
    final_format_predictions,
    raw_predictions,
    render_all_scores,
    render_selected_only,
    select_verification_records,
    validate_all_scores,
    validate_schema,
    validate_selected_only,
    verify_reproduction,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "configs/bosideng_unified57_schema.json").read_text())
INDEX = {name: i for i, name in enumerate(SCHEMA["labels"])}


def vector(values=None, fill=0.0):
    out = [fill] * 57
    for name, value in (values or {}).items():
        out[INDEX[name]] = value
    return out


def thresholds(value=0.5):
    labels = {}
    for name in SCHEMA["labels"]:
        mode = SCHEMA["label_training_modes"][name]
        labels[name] = {
            "mode": mode,
            "threshold": None if mode == "unsupported" else value,
            "status": "disabled_unsupported" if mode == "unsupported" else "calibrated",
        }
    return {"labels": labels}


def row(*, scores=None, labels=None, known=None, pu=None, sources=None, record_id="r"):
    return {
        "record_id": record_id,
        "image_path": f"/{record_id}.jpg",
        "image_sha256": "a" * 64,
        "source": (sources or ["jd_complete23"])[0],
        "sources": sources or ["jd_complete23"],
        "scores": scores or vector(),
        "labels": labels or vector(),
        "known_mask": known or vector(),
        "pu_positive_mask": pu or vector(),
        "schema_version": SCHEMA["schema_version"],
        "schema_sha256": SCHEMA["schema_sha256"],
    }


def test_schema_requires_exact_36_20_1_contract():
    result = validate_schema(SCHEMA)
    assert result == {"pn": 36, "pu": 20, "unsupported": 1}
    bad = json.loads(json.dumps(SCHEMA))
    bad["label_training_modes"]["无袖"] = "pu"
    with pytest.raises(ValueError, match="36 PN / 20 PU / 1 unsupported"):
        validate_schema(bad)


def test_pn_calibration_uses_only_known_cells_and_stable_ties():
    rows = [
        row(scores=vector({"连帽": 0.9}), labels=vector({"连帽": 1}), known=vector({"连帽": 1})),
        row(scores=vector({"连帽": 0.4}), labels=vector({"连帽": 0}), known=vector({"连帽": 1})),
        row(scores=vector({"连帽": 0.99})),  # unknown: must be ignored
    ]
    payload = calibrate_thresholds(rows, SCHEMA)
    hood = payload["labels"]["连帽"]
    assert hood["method"] == "observed_pn_f1"
    assert hood["support"] == {"known_positive": 1, "known_negative": 1}
    assert hood["threshold"] == 0.5


def test_calibration_falls_back_when_support_is_insufficient():
    payload = calibrate_thresholds(
        [row(scores=vector({"连帽": 0.9}), labels=vector({"连帽": 1}), known=vector({"连帽": 1}))],
        SCHEMA,
    )
    assert payload["labels"]["连帽"]["threshold"] == 0.5
    assert payload["labels"]["连帽"]["status"] == "fallback_insufficient_support"


def test_pu_calibration_optimizes_positive_minus_unlabeled_coverage():
    rows = []
    for i, score in enumerate([0.9, 0.85, 0.8, 0.75, 0.7]):
        rows.append(row(record_id=f"p{i}", scores=vector({"织带": score}), pu=vector({"织带": 1})))
    for i in range(50):
        rows.append(row(record_id=f"u{i}", scores=vector({"织带": 0.1 + i / 100})))
    item = calibrate_thresholds(rows, SCHEMA)["labels"]["织带"]
    assert item["method"] == "positive_minus_unlabeled_coverage"
    assert item["status"] == "calibrated"
    assert item["threshold"] == 0.7


def test_raw_keeps_multi_positive_and_final_uses_stable_schema_tie():
    scores = vector({"压胶充绒": 0.9, "压胶袋盖": 0.9})
    raw = raw_predictions(scores, thresholds(), SCHEMA)
    final = final_format_predictions(scores, thresholds(), SCHEMA)
    assert raw[INDEX["压胶充绒"]] == raw[INDEX["压胶袋盖"]] == 1
    assert final[INDEX["压胶充绒"]] == 1
    assert final[INDEX["压胶袋盖"]] == 0
    assert final[INDEX["假两件"]] == 0


def test_pn_metrics_ignore_unknown_and_report_specificity_exact_match():
    rows = [
        row(labels=vector({"连帽": 1}), known=vector({"连帽": 1})),
        row(labels=vector({"连帽": 0}), known=vector({"连帽": 1})),
        row(labels=vector({"连帽": 0}), known=vector()),
    ]
    binary = [vector({"连帽": 1}), vector({"连帽": 1}), vector({"连帽": 1})]
    report = evaluate_pn_slice(rows, binary, [INDEX["连帽"]])
    assert report["micro"]["tp"] == 1
    assert report["micro"]["fp"] == 1
    assert report["micro"]["known_cells"] == 2
    assert report["micro"]["specificity"] == 0.0
    assert report["micro"]["accuracy"] == 0.5
    assert report["exact_match"] == 0.5


def test_pu_metrics_never_claim_precision_f1_or_accuracy():
    report = evaluate_pu_label([0.9, 0.6], [0.7, 0.2], 0.5)
    assert report["positive_recall"] == 1.0
    assert report["positive_vs_unlabeled_concordance"] == 0.75
    assert report["unlabeled_coverage"] == 0.5
    forbidden = {"precision", "f1", "specificity", "accuracy", "negative_support", "roc_auc"}
    assert forbidden.isdisjoint(report)


def test_views_report_contract_forced_false_negative_and_clean_slices():
    truth = vector({"压胶充绒": 1, "压胶袋盖": 1})
    known = vector({"压胶充绒": 1, "压胶袋盖": 1})
    rows = [
        row(scores=vector({"压胶充绒": 0.8, "压胶袋盖": 0.9}), labels=truth, known=known),
        row(record_id="d", sources=["dictionary_v4"], scores=vector({"连帽": 0.8}), labels=vector({"连帽": 1}), known=vector({"连帽": 1})),
        row(record_id="m", sources=["jd_complete23", "dictionary_v4"], scores=vector({"连帽": 0.8}), labels=vector({"连帽": 1}), known=vector({"连帽": 1})),
    ]
    report = evaluate_views(rows, thresholds(), SCHEMA)
    assert report["format_constraint_loss"]["contract_forced_false_negatives"] == 1
    assert report["format_constraint_loss"]["oracle_final_recall_ceiling"] == pytest.approx(3 / 4)
    assert report["raw_thresholded"]["jd23_clean"]["record_count"] == 1
    assert report["raw_thresholded"]["dictionary_pn_clean"]["record_count"] == 1
    assert report["raw_thresholded"]["mixed_exact_audit"]["record_count"] == 1


def test_renderers_enforce_57_two_decimal_scores_and_four_categories():
    scores = vector({"连帽": 0.876, "假两件": 0.99, "H型": 0.501})
    all_scores = render_all_scores(row(scores=scores), SCHEMA)
    assert list(all_scores["scores"]) == SCHEMA["labels"]
    assert all_scores["scores"]["连帽"] == "0.88"
    assert all_scores["scores"]["假两件"] == "0.00"
    assert validate_all_scores(all_scores, SCHEMA) is all_scores
    selected = render_selected_only(scores, thresholds(), SCHEMA)
    assert list(selected) == ["局部结构", "廓形", "工艺", "面辅料"]
    assert selected["局部结构"] == ["连帽"]
    assert selected["廓形"] == ["H型"]
    assert validate_selected_only(selected, SCHEMA) is selected


def test_selected_validator_rejects_two_tags_from_same_subcategory():
    payload = {"局部结构": [], "廓形": [], "工艺": ["压胶充绒", "压胶袋盖"], "面辅料": []}
    with pytest.raises(ValueError, match="subcategory"):
        validate_selected_only(payload, SCHEMA)


def test_buffered_shard_syncs_at_1000_and_recovers_durable_offset(tmp_path):
    path = tmp_path / "rank00.part.jsonl"
    meta = {"split": "test", "rank": 0, "world_size": 8, "schema_sha256": SCHEMA["schema_sha256"]}
    shard = BufferedPredictionShard(path, meta)
    shard.append_batch([{"record_id": f"r{i}"} for i in range(1000)], 1000)
    assert shard.durable_records == 1000
    assert shard.data_sync_count == 1
    durable = shard.durable_offset
    shard._handle.write('{"record_id":"tail"}\n')
    shard._handle.flush()
    shard._handle.close()
    resumed = BufferedPredictionShard(path, meta)
    assert path.stat().st_size == durable
    assert resumed.next_local_index == 1000
    resumed.close(complete=True)


def test_buffered_shard_time_sync_and_metadata_drift(tmp_path):
    now = [0.0]
    path = tmp_path / "rank00.part.jsonl"
    meta = {"split": "val", "rank": 0}
    shard = BufferedPredictionShard(path, meta, clock=lambda: now[0])
    shard.append_batch([{"record_id": "a"}], 1)
    assert shard.durable_records == 0
    now[0] = 30.0
    shard.append_batch([{"record_id": "b"}], 2)
    assert shard.durable_records == 2
    shard.close(complete=False)
    with pytest.raises(ValueError, match="metadata"):
        BufferedPredictionShard(path, {"split": "test", "rank": 0})


def test_verification_selection_is_deterministic_and_comparator_requires_32x57():
    rows = [row(record_id=f"r{i}", sources=["dictionary_v4"] if i % 2 else ["jd_complete23"]) for i in range(40)]
    first = select_verification_records(rows, count=32, seed=20260717)
    second = select_verification_records(list(reversed(rows)), count=32, seed=20260717)
    assert [r["record_id"] for r in first] == [r["record_id"] for r in second]
    reference = [{"record_id": r["record_id"], "scores": vector(), "selected": {"局部结构": [], "廓形": [], "工艺": [], "面辅料": []}} for r in first]
    report = verify_reproduction(reference, json.loads(json.dumps(reference)), SCHEMA)
    assert report == {
        "records": 32,
        "score_values": 1824,
        "probabilities_exact": True,
        "max_abs_score_delta": 0.0,
        "selected_outputs_exact": True,
    }
