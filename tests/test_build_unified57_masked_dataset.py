import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import build_unified57_masked_dataset as builder


UNIFIED56_TAGS = builder.UNIFIED56_TAGS
encode_jd_record = builder.encode_jd_record


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "configs" / "bosideng_unified57_schema.json"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _phash(value: int) -> str:
    return f"{value:016x}"


def _raw_jd(record_id: str, sha256: str, phash64: str, **selected):
    row = {
        "record_id": record_id,
        "source": "jd_complete23",
        "image_path": f"/images/{record_id}.jpg",
        "sha256": sha256,
        "phash64": phash64,
        "长度分类": [],
        "主廓形": [],
        "帽类结构": [],
        "工艺标签": [],
        "闭合方式": [],
        "其他结构": [],
        "肩袖结构": [],
    }
    row.update(selected)
    return row


def _encoded_jd(record_id: str, sha256: str, phash64: str, **selected):
    raw = _raw_jd(record_id, sha256, phash64, **selected)
    encoded = encode_jd_record(raw)
    return {
        "record_id": record_id,
        "source": "jd_complete23",
        "image_path": raw["image_path"],
        "image_sha256": sha256,
        "phash64": phash64,
        **encoded,
    }


def _dictionary(record_id: str, tag: str, sha256: str, phash64: str):
    return {
        "record_id": record_id,
        "canonical_tag": tag,
        "relative_path": f"images/{tag}/{record_id}.jpg",
        "sha256": sha256,
        "phash64": phash64,
        "known_positive_only": True,
        # The real 32 no-sleeve rows retain this legacy false value. The
        # unified57 builder must select by canonical_tag, not this flag.
        "trainable_canonical56": tag != "无袖",
    }


def _by_tag(row):
    schema = builder.load_schema(SCHEMA_PATH)
    return {
        tag: (
            row["labels"][index],
            row["known_mask"][index],
            row["pu_positive_mask"][index],
        )
        for index, tag in enumerate(schema["labels"])
    }


def _build(jd=(), dictionary=(), **kwargs):
    return builder.build_dataset(
        list(jd),
        list(dictionary),
        schema_path=SCHEMA_PATH,
        **kwargs,
    )


def test_same_visual_dictionary_rows_aggregate_all_positive_labels():
    same_sha = _sha("same-bytes")
    result = _build(
        dictionary=[
            _dictionary("d-h", "H型", same_sha, _phash(0)),
            _dictionary("d-long", "长款", same_sha, _phash(15)),
        ]
    )

    assert len(result["records"]) == 1
    row = result["records"][0]
    states = _by_tag(row)
    assert states["H型"] == (1.0, 1, 0)
    assert states["长款"] == (1.0, 1, 0)
    assert set(row["source_record_ids"]) == {"d-h", "d-long"}
    assert row["binding_count"] == 2
    assert row["dictionary_binding_count"] == 2
    assert row["jd_binding_count"] == 0


def test_jd_raw_or_encoded_rows_open_exactly_23_pn_dimensions():
    raw = _raw_jd(
        "jd-raw",
        _sha("jd-raw"),
        _phash(0x1000),
        主廓形=["H型"],
        工艺标签=["压胶门襟"],
        帽类结构=["连帽"],
    )
    encoded = _encoded_jd(
        "jd-encoded",
        _sha("jd-encoded"),
        _phash(0xF000000000000000),
        长度分类=["短款"],
        帽类结构=["连帽"],
    )

    result = _build(jd=[raw, encoded])

    assert len(result["records"]) == 2
    for row in result["records"]:
        assert sum(row["known_mask"]) == 23
        assert sum(row["pu_positive_mask"]) == 0
        assert _by_tag(row)["无袖"] == (0.0, 0, 0)


def test_safe_exclusions_do_not_treat_bent_sleeve_or_taped_tags_as_mutually_exclusive():
    result = _build(
        dictionary=[
            _dictionary("shoulder", "插肩袖", _sha("shoulder"), _phash(0)),
            _dictionary("bent", "弯袖", _sha("bent"), _phash(0xFF00000000000000)),
            _dictionary("taped", "压胶充绒", _sha("taped"), _phash(0x00FF000000000000)),
            _dictionary("front", "前门襟", _sha("front"), _phash(0x0000FF0000000000)),
        ]
    )
    rows = {row["source_record_ids"][0]: row for row in result["records"]}

    shoulder = _by_tag(rows["shoulder"])
    assert shoulder["插肩袖"] == (1.0, 1, 0)
    assert shoulder["正肩袖"] == (0.0, 1, 0)
    assert shoulder["落肩袖"] == (0.0, 1, 0)
    assert shoulder["弯袖"] == (0.0, 0, 0)

    bent = _by_tag(rows["bent"])
    assert bent["弯袖"] == (1.0, 1, 0)
    assert bent["插肩袖"] == (0.0, 0, 0)
    assert bent["正肩袖"] == (0.0, 0, 0)
    assert bent["落肩袖"] == (0.0, 0, 0)

    taped = _by_tag(rows["taped"])
    assert taped["压胶充绒"] == (1.0, 1, 0)
    assert taped["压胶袋盖"] == (0.0, 0, 0)
    assert taped["压胶门襟"] == (0.0, 0, 0)

    front = _by_tag(rows["front"])
    assert front["前门襟"] == (1.0, 0, 1)
    assert front["双拉链"] == (0.0, 0, 0)


def test_positive_wins_same_label_but_safe_group_multi_positive_is_unknown():
    positive_sha = _sha("positive-wins")
    conflict_sha = _sha("safe-conflict")
    result = _build(
        jd=[
            _encoded_jd("jd-negative", positive_sha, _phash(0), 工艺标签=[]),
            _encoded_jd("jd-a", conflict_sha, _phash(0xFFFF), 主廓形=["A型"]),
        ],
        dictionary=[
            _dictionary("dict-positive", "压胶门襟", positive_sha, _phash(0)),
            _dictionary("dict-h", "H型", conflict_sha, _phash(0xFFFF)),
        ],
    )
    rows = {frozenset(row["source_record_ids"]): row for row in result["records"]}

    positive_wins = _by_tag(rows[frozenset({"jd-negative", "dict-positive"})])
    assert positive_wins["压胶门襟"] == (1.0, 1, 0)
    mixed_row = rows[frozenset({"jd-negative", "dict-positive"})]
    assert mixed_row["binding_count"] == 2
    assert mixed_row["dictionary_binding_count"] == 1
    assert mixed_row["jd_binding_count"] == 1

    pure_conflict = _by_tag(rows[frozenset({"jd-a", "dict-h"})])
    for tag in ("H型", "O型", "X型", "A型", "茧型", "箱型"):
        assert pure_conflict[tag] == (0.0, 0, 0)
    assert any(
        conflict["kind"] == "mutually_exclusive_positive_conflict"
        and set(conflict["positive_tags"]) == {"H型", "A型"}
        for conflict in result["conflicts"]
    )


def test_known_not_hooded_implies_two_negatives_without_positive_reverse_inference():
    result = _build(
        jd=[
            _encoded_jd("not-hooded", _sha("not-hooded"), _phash(0), 帽类结构=[]),
            _encoded_jd(
                "hooded",
                _sha("hooded"),
                _phash(0xFF00000000000000),
                帽类结构=["连帽"],
            ),
        ]
    )
    rows = {row["source_record_ids"][0]: _by_tag(row) for row in result["records"]}

    assert rows["not-hooded"]["连帽"] == (0.0, 1, 0)
    assert rows["not-hooded"]["拆卸帽"] == (0.0, 1, 0)
    assert rows["not-hooded"]["帽口抽绳"] == (0.0, 1, 0)
    assert rows["hooded"]["连帽"] == (1.0, 1, 0)
    assert rows["hooded"]["拆卸帽"] == (0.0, 0, 0)
    assert rows["hooded"]["帽口抽绳"] == (0.0, 0, 0)


def test_sleeveless_and_known_sleeve_positive_have_strict_negative_implications():
    result = _build(
        dictionary=[
            _dictionary("sleeveless", "无袖", _sha("sleeveless-rule"), _phash(0)),
            _dictionary("raglan", "插肩袖", _sha("raglan-rule"), _phash(0xFFFF000000000000)),
        ]
    )
    rows = {row["source_record_ids"][0]: _by_tag(row) for row in result["records"]}

    assert rows["sleeveless"]["无袖"] == (1.0, 1, 0)
    for tag in (
        "弯袖",
        "插肩袖",
        "正肩袖",
        "落肩袖",
        "防风袖口",
        "罗纹袖口",
        "袖袢",
    ):
        assert rows["sleeveless"][tag] == (0.0, 1, 0)
    assert rows["raglan"]["插肩袖"] == (1.0, 1, 0)
    assert rows["raglan"]["无袖"] == (0.0, 1, 0)


def test_exact_phash_merges_supervision_and_distance_two_only_joins_split_component():
    exact = _phash(0)
    near = _phash(0b11)
    result = _build(
        dictionary=[
            _dictionary("exact-a", "弯袖", _sha("exact-a"), exact),
            _dictionary("exact-b", "袖标", _sha("exact-b"), exact),
            _dictionary("near", "腰带", _sha("near"), near),
        ]
    )

    assert len(result["records"]) == 2
    exact_row = next(row for row in result["records"] if len(row["source_record_ids"]) == 2)
    near_row = next(row for row in result["records"] if row is not exact_row)
    exact_states = _by_tag(exact_row)
    near_states = _by_tag(near_row)
    assert exact_states["弯袖"] == (1.0, 1, 0)
    assert exact_states["袖标"] == (1.0, 0, 1)
    assert exact_states["腰带"] == (0.0, 0, 0)
    assert near_states["腰带"] == (1.0, 1, 0)
    assert exact_row["visual_group_id"] != near_row["visual_group_id"]
    assert exact_row["visual_component_id"] == near_row["visual_component_id"]
    assert exact_row["split"] == near_row["split"]


def test_representative_path_sha_and_phash_come_from_one_member():
    representative = _dictionary("a", "前门襟", "f" * 64, "f" * 16)
    same_phash = _dictionary("b", "袖标", "0" * 64, "f" * 16)
    same_sha = _dictionary("c", "贴袋", "f" * 64, "0" * 16)

    result = _build(dictionary=[representative, same_phash, same_sha])

    assert len(result["records"]) == 1
    row = result["records"][0]
    assert row["image_path"].endswith("images/前门襟/a.jpg")
    assert row["image_sha256"] == "f" * 64
    assert row["phash64"] == "f" * 16
    assert row["image_sha256s"] == ["0" * 64, "f" * 64]
    assert row["exact_phashes"] == ["0" * 16, "f" * 16]


def _identity_only_dictionary(record_id="invalid-mode"):
    return {
        "record_id": record_id,
        "relative_path": f"images/{record_id}.jpg",
        "sha256": _sha(record_id),
        "phash64": _phash(0),
    }


@pytest.mark.parametrize("label_value", [0.0, 1.0])
def test_explicit_known_mask_rejects_pure_pu_label(label_value):
    schema = builder.load_schema(SCHEMA_PATH)
    index = schema["labels"].index("前门襟")
    row = _identity_only_dictionary(f"pu-as-pn-{int(label_value)}")
    row["labels"] = [0.0] * 57
    row["known_mask"] = [0] * 57
    row["labels"][index] = label_value
    row["known_mask"][index] = 1

    with pytest.raises(ValueError, match="前门襟.*mode=pu"):
        _build(dictionary=[row])


def test_pn_negative_tags_rejects_pure_pu_label():
    row = _identity_only_dictionary("pu-negative-tag")
    row["pn_negative_tags"] = ["前门襟"]

    with pytest.raises(ValueError, match="前门襟.*mode=pu"):
        _build(dictionary=[row])


def test_pu_positive_mask_rejects_pn_label():
    schema = builder.load_schema(SCHEMA_PATH)
    index = schema["labels"].index("H型")
    row = _identity_only_dictionary("pn-as-pu")
    row["pu_positive_mask"] = [0] * 57
    row["pu_positive_mask"][index] = 1

    with pytest.raises(ValueError, match="H型.*mode=pn"):
        _build(dictionary=[row])


@pytest.mark.parametrize("evidence_field", ["pn_negative_tags", "pu_positive_tags"])
def test_unsupported_label_rejects_all_explicit_supervision(evidence_field):
    row = _identity_only_dictionary(f"unsupported-{evidence_field}")
    row[evidence_field] = ["假两件"]

    with pytest.raises(ValueError, match="假两件.*mode=unsupported"):
        _build(dictionary=[row])


def test_train_val_test_have_no_visual_component_crossing():
    rows = []
    for component_index in range(12):
        repeated_byte = component_index + 1
        base = int.from_bytes(bytes([repeated_byte]) * 8, "big")
        rows.extend(
            [
                _dictionary(
                    f"c{component_index}-a",
                    "弯袖",
                    _sha(f"c{component_index}-a"),
                    _phash(base),
                ),
                _dictionary(
                    f"c{component_index}-b",
                    "弯袖",
                    _sha(f"c{component_index}-b"),
                    _phash(base ^ 1),
                ),
            ]
        )

    result = _build(dictionary=rows, seed=20260717)

    owners = {}
    for row in result["records"]:
        owner = owners.setdefault(row["visual_component_id"], row["split"])
        assert owner == row["split"]
    assert {row["split"] for row in result["records"]} == {"train", "val", "test"}
    assert result["leakage_check"]["passed"] is True
    assert result["leakage_check"]["cross_split_components"] == []


def _long_tail_stratification_rows():
    rare_tags = ("H型", "立领", "合体", "前门襟", "插袋", "袖标")
    tags = [tag for tag in rare_tags for _ in range(3)] + ["织带"] * 82
    rows = []
    for index, tag in enumerate(tags):
        # Repeating one distinct byte eight times gives every two fixture
        # pHashes a Hamming distance of at least 8, so all 100 rows are
        # independent visual components.
        phash_value = int.from_bytes(bytes([index]) * 8, "big")
        rows.append(
            _dictionary(
                f"strat-{index:03d}",
                tag,
                _sha(f"strat-{index:03d}"),
                _phash(phash_value),
            )
        )
    return rows


def test_long_tail_stratification_preserves_positive_support_and_global_ratios():
    result = _build(dictionary=_long_tail_stratification_rows(), seed=20260717)
    records = result["records"]
    schema = builder.load_schema(SCHEMA_PATH)

    component_positive_splits = {
        tag: {
            row["split"]
            for row in records
            if row["labels"][schema["labels"].index(tag)] == 1.0
        }
        for tag in schema["labels"]
    }
    component_positive_counts = {
        tag: sum(
            row["labels"][schema["labels"].index(tag)] == 1.0
            for row in records
        )
        for tag in schema["labels"]
    }
    for tag, count in component_positive_counts.items():
        if schema["label_training_modes"][tag] != "unsupported" and count >= 3:
            assert component_positive_splits[tag] == {"train", "val", "test"}, tag

    split_counts = {split: sum(row["split"] == split for row in records) for split in ("train", "val", "test")}
    assert abs(split_counts["train"] - len(records) * 0.8) <= len(records) * 0.02
    assert abs(split_counts["val"] - len(records) * 0.1) <= len(records) * 0.02
    assert abs(split_counts["test"] - len(records) * 0.1) <= len(records) * 0.02
    assert result["leakage_check"]["passed"] is True
    assert result["leakage_check"]["cross_split_components"] == []


def test_split_capacity_uses_ratio_tolerance_to_fit_ten_disjoint_long_tails():
    rare_tags = (
        "H型",
        "立领",
        "合体",
        "长款",
        "压胶门襟",
        "前门襟",
        "插袋",
        "袖标",
        "魔术贴",
        "D字扣",
    )
    tags = [tag for tag in rare_tags for _ in range(3)] + ["织带"] * 70
    rows = []
    for index, tag in enumerate(tags):
        phash_value = int.from_bytes(bytes([index]) * 8, "big")
        rows.append(
            _dictionary(
                f"capacity-{index:03d}",
                tag,
                _sha(f"capacity-{index:03d}"),
                _phash(phash_value),
            )
        )

    result = _build(dictionary=rows, seed=20260717)
    schema = builder.load_schema(SCHEMA_PATH)
    split_counts = {
        split: sum(row["split"] == split for row in result["records"])
        for split in ("train", "val", "test")
    }

    assert 78 <= split_counts["train"] <= 82
    assert 8 <= split_counts["val"] <= 12
    assert 8 <= split_counts["test"] <= 12
    for tag in (*rare_tags, "织带"):
        tag_index = schema["labels"].index(tag)
        assert {
            row["split"]
            for row in result["records"]
            if row["labels"][tag_index] == 1.0
        } == {"train", "val", "test"}
    assert result["leakage_check"]["passed"] is True


def test_summary_reports_each_labels_supervision_counts_by_split():
    result = _build(dictionary=_long_tail_stratification_rows(), seed=20260717)

    for tag, totals in result["summary"]["per_label"].items():
        by_split = totals["by_split"]
        assert set(by_split) == {"train", "val", "test"}
        for field in ("pn_positive", "pn_negative", "pu_positive", "unknown"):
            assert sum(by_split[split][field] for split in by_split) == totals[field], (
                tag,
                field,
            )
        assert set(totals["positive_components_by_split"]) == {
            "train",
            "val",
            "test",
        }
        assert (
            sum(totals["positive_components_by_split"].values())
            == totals["positive_components"]
        )


def test_all_vectors_are_57_and_schema_hash_matches_rows():
    schema = builder.load_schema(SCHEMA_PATH)
    result = _build(
        dictionary=[
            _dictionary("sleeveless", "无袖", _sha("sleeveless"), _phash(0))
        ]
    )

    assert schema["labels"] == [*UNIFIED56_TAGS, "无袖"]
    assert schema["num_labels"] == 57
    assert schema["schema_sha256"] == builder.compute_schema_sha256(schema)
    assert schema["label_training_modes"]["H型"] == "pn"
    assert schema["label_training_modes"]["压胶门襟"] == "pn"
    assert schema["label_training_modes"]["弯袖"] == "pn"
    assert schema["label_training_modes"]["前门襟"] == "pu"
    assert schema["label_training_modes"]["假两件"] == "unsupported"
    assert list(schema["label_training_modes"].values()).count("pn") == 36
    assert list(schema["label_training_modes"].values()).count("pu") == 20
    row = result["records"][0]
    assert len(row["labels"]) == 57
    assert len(row["known_mask"]) == 57
    assert len(row["pu_positive_mask"]) == 57
    assert row["schema_sha256"] == schema["schema_sha256"]
    assert _by_tag(row)["无袖"] == (1.0, 1, 0)


def test_cli_writes_schema_bound_splits_and_audits(tmp_path):
    jd_path = tmp_path / "jd_enriched.jsonl"
    dictionary_path = tmp_path / "dictionary.jsonl"
    dictionary_root = tmp_path / "dictionary"
    output_dir = tmp_path / "output"
    dictionary_root.mkdir()
    jd_path.write_text(
        json.dumps(
            _raw_jd("jd", _sha("jd"), _phash(0), 主廓形=["H型"]),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    dictionary_path.write_text(
        json.dumps(
            _dictionary("dict", "无袖", _sha("dict"), _phash(0xFFFF)),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = builder.main(
        [
            "--jd-enriched",
            str(jd_path),
            "--dictionary-manifest",
            str(dictionary_path),
            "--dictionary-root",
            str(dictionary_root),
            "--schema",
            str(SCHEMA_PATH),
            "--output-dir",
            str(output_dir),
            "--seed",
            "20260717",
        ]
    )

    assert exit_code == 0
    assert {path.name for path in output_dir.iterdir()} == {
        "train.jsonl",
        "val.jsonl",
        "test.jsonl",
        "dataset_summary.json",
        "conflicts.csv",
        "leakage_check.json",
    }
    output_rows = []
    for split in ("train", "val", "test"):
        output_rows.extend(
            json.loads(line)
            for line in (output_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        )
    assert len(output_rows) == 2
    assert all(len(row["pu_positive_mask"]) == 57 for row in output_rows)


def test_script_help_is_self_contained_for_the_three_file_delivery(tmp_path):
    standalone = tmp_path / "build_unified57_masked_dataset.py"
    shutil.copy2(ROOT / "scripts" / "build_unified57_masked_dataset.py", standalone)

    completed = subprocess.run(
        [sys.executable, str(standalone), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--jd-enriched" in completed.stdout
