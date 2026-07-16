import hashlib
import json
from pathlib import Path

import pytest


SCHEMA_PATH = Path("configs/bosideng_unified57_schema.json")


def _schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _row(record_id: str, score: float = 0.9):
    schema = _schema()
    labels = [0.0] * 57
    known = [0] * 57
    pu = [0] * 57
    labels[0] = 1.0
    known[0] = 1
    return {
        "record_id": record_id,
        "image_path": f"images/{record_id}.jpg",
        "image_sha256": "a" * 64,
        "sources": ["jd_complete23"],
        "schema_version": schema["schema_version"],
        "schema_sha256": schema["schema_sha256"],
        "labels": labels,
        "known_mask": known,
        "pu_positive_mask": pu,
        "scores": [score] + [0.1] * 56,
    }


def test_load_manifest_locks_sha_and_stride_partition(tmp_path):
    from scripts.evaluate_unified57_multilabel import (
        load_manifest,
        partition_records,
        sha256_file,
    )

    manifest = tmp_path / "val.jsonl"
    rows = [_row(f"r{index}") for index in range(9)]
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    digest = sha256_file(manifest)
    loaded = load_manifest(manifest, _schema(), expected_sha256=digest)
    assert [row["record_id"] for row in partition_records(loaded, 2, 4)] == ["r2", "r6"]
    with pytest.raises(ValueError, match="SHA256"):
        load_manifest(manifest, _schema(), expected_sha256="0" * 64)


def test_threshold_file_is_frozen_to_checkpoint_and_validation(tmp_path):
    from scripts.evaluate_unified57_multilabel import freeze_thresholds, load_frozen_thresholds

    schema = _schema()
    rows = [_row(f"r{index}", 0.9 if index % 2 else 0.1) for index in range(60)]
    for index, row in enumerate(rows):
        row["labels"][0] = float(index % 2)
    path = tmp_path / "thresholds.json"
    payload = freeze_thresholds(
        path,
        rows,
        schema,
        checkpoint_sha256="1" * 64,
        validation_manifest_sha256="2" * 64,
    )
    assert payload["checkpoint_sha256"] == "1" * 64
    assert payload["validation_manifest_sha256"] == "2" * 64
    frozen_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_frozen_thresholds(
        path,
        schema,
        checkpoint_sha256="1" * 64,
        validation_manifest_sha256="2" * 64,
        expected_threshold_sha256=frozen_sha,
    )
    assert loaded == payload
    with pytest.raises(ValueError, match="checkpoint"):
        load_frozen_thresholds(
            path,
            schema,
            checkpoint_sha256="3" * 64,
            validation_manifest_sha256="2" * 64,
        )


def test_merge_requires_every_expected_record_and_preserves_manifest_order(tmp_path):
    from scripts.evaluate_unified57_multilabel import merge_prediction_shards

    expected = [_row("r0"), _row("r1"), _row("r2")]
    shard0 = tmp_path / "rank0.jsonl"
    shard1 = tmp_path / "rank1.jsonl"
    shard0.write_text(json.dumps(expected[0]) + "\n" + json.dumps(expected[2]) + "\n")
    shard1.write_text(json.dumps(expected[1]) + "\n")
    merged = merge_prediction_shards(expected, [shard0, shard1], tmp_path / "merged.jsonl")
    assert [row["record_id"] for row in merged] == ["r0", "r1", "r2"]
    shard1.write_text("")
    with pytest.raises(RuntimeError, match="incomplete"):
        merge_prediction_shards(expected, [shard0, shard1], tmp_path / "bad.jsonl")


def test_render_delivery_outputs_masks_unsupported_and_builds_references(tmp_path):
    from scripts.evaluate_unified57_multilabel import write_delivery_outputs

    schema = _schema()
    rows = [_row(f"r{index}") for index in range(40)]
    thresholds = {
        "schema_version": schema["schema_version"],
        "schema_sha256": schema["schema_sha256"],
        "labels": {
            tag: {
                "mode": schema["label_training_modes"][tag],
                "threshold": None if tag == "假两件" else 0.5,
                "status": "disabled_unsupported" if tag == "假两件" else "calibrated",
            }
            for tag in schema["labels"]
        },
    }
    report = write_delivery_outputs(rows, thresholds, schema, tmp_path)
    assert report["output_quality"]["json_validity_rate"] == 1.0
    assert report["verification"]["records"] == 32
    assert "raw_thresholded" in report["performance"]
    assert "final_format" in report["performance"]
    assert isinstance(report["representative_6"], list)
    assert len(report["representative_6"]) == 6
    first = json.loads((tmp_path / "test_all_scores.jsonl").read_text().splitlines()[0])
    assert first["scores"]["假两件"] == "0.00"
    verification_manifest = json.loads(
        (tmp_path / "verification" / "verification_32_manifest.jsonl").read_text().splitlines()[0]
    )
    assert set(verification_manifest) == {
        "record_id", "test_manifest_index", "image_path", "image_sha256",
        "source", "sources", "selection_bucket",
    }
    float_reference = json.loads(
        (tmp_path / "verification" / "reference_32_float32.jsonl").read_text().splitlines()[0]
    )
    assert {"scores", "labels", "known_mask", "pu_positive_mask", "schema_sha256", "checkpoint_sha256"} <= set(float_reference)
    selected_reference = json.loads(
        (tmp_path / "verification" / "reference_32_selected_only.jsonl").read_text().splitlines()[0]
    )
    assert set(selected_reference) == {"record_id", "output"}
    assert len((tmp_path / "verification" / "reference_32_float32.jsonl").read_text().splitlines()) == 32
    assert len((tmp_path / "representative6" / "manifest.jsonl").read_text().splitlines()) == 6
    representative = json.loads(
        (tmp_path / "representative6" / "manifest.jsonl").read_text().splitlines()[0]
    )
    assert {"truth_summary", "pn_errors", "pu_positive_hits"} <= set(representative)


def test_dictionary_macro_recall_combines_pn_and_pu_positives():
    from scripts.evaluate_unified57_multilabel import evaluate_dictionary_positive_recall

    schema = _schema()
    rows = [_row("pn"), _row("pu")]
    rows[0]["sources"] = ["dictionary_v4"]
    rows[1]["sources"] = ["dictionary_v4"]
    rows[1]["known_mask"] = [0] * 57
    rows[1]["labels"] = [0.0] * 57
    rows[1]["pu_positive_mask"] = [0] * 57
    pu_index = schema["labels"].index("前门襟")
    rows[1]["labels"][pu_index] = 1.0
    rows[1]["pu_positive_mask"][pu_index] = 1
    rows[1]["scores"][pu_index] = 0.9
    thresholds = {
        "labels": {
            tag: {"threshold": None if tag == "假两件" else 0.5}
            for tag in schema["labels"]
        }
    }
    report = evaluate_dictionary_positive_recall(rows, thresholds, schema)
    assert report["labels_with_positive_support"] == 2
    assert report["macro_positive_recall"] == 1.0
    assert report["per_label"]["前门襟"]["supervision"] == "pu_positive_mask"


def test_parse_args_requires_explicit_hashes_and_two_manifests(tmp_path):
    from scripts.evaluate_unified57_multilabel import parse_args

    argv = [
        "--model", "/model", "--model-config-sha256", "e" * 64,
        "--schema", "/schema.json", "--schema-file-sha256", "a" * 64,
        "--checkpoint", "/latest.pt", "--checkpoint-sha256", "b" * 64,
        "--validation-manifest", "/val.jsonl", "--validation-manifest-sha256", "c" * 64,
        "--test-manifest", "/test.jsonl", "--test-manifest-sha256", "d" * 64,
        "--expected-trainable-manifest-sha256", "f" * 64,
        "--base-artifact-manifest-sha256", "9" * 64,
        "--output-dir", str(tmp_path), "--wall-clock-seconds", "2700",
        "--image-cache-root", "/tmp/cache",
    ]
    args = parse_args(argv)
    assert args.expected_world_size == 8
    assert args.model_config_sha256 == "e" * 64
    assert args.test_manifest_sha256 == "d" * 64
    assert args.expected_trainable_manifest_sha256 == "f" * 64
    assert args.image_cache_root == Path("/tmp/cache")
    with pytest.raises(SystemExit):
        parse_args(["--model", "/model"])
