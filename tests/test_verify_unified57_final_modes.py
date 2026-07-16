from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pytest

from scripts.verify_unified57_final_modes import (
    FinalModeVerificationError,
    main,
    verify_final_modes,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "configs" / "bosideng_unified57_schema.json"
CATEGORIES = ("局部结构", "廓形", "工艺", "面辅料")
TWO_DECIMAL = re.compile(r"(?:0\.\d{2}|1\.00)\Z")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


FAKE_INFER = r'''#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
CATEGORIES = ("局部结构", "廓形", "工艺", "面辅料")

parser = argparse.ArgumentParser()
parser.add_argument("--scores-json", type=Path, required=True)
parser.add_argument("--mode", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
config = json.loads((PACKAGE_DIR / "model_config.json").read_text(encoding="utf-8"))
scores = json.loads(args.scores_json.read_text(encoding="utf-8"))["scores"]
labels = config["tag_order"]
selected = config["selected_tags"]
fault = config.get("fault")
if args.mode == "selected_with_confidence":
    output = {
        category: [
            {"name": tag, "confidence": f"{scores[labels.index(tag)]:.2f}"}
            for tag in selected[category]
        ]
        for category in CATEGORIES
    }
    if fault == "confidence_name":
        output[CATEGORIES[0]][0]["name"] = labels[-2]
    if fault == "confidence_value":
        output[CATEGORIES[0]][0]["confidence"] = "0.01"
    if fault == "bad_confidence_decimal":
        output[CATEGORIES[0]][0]["confidence"] = "0.9"
elif args.mode == "all_scores":
    items = [
        (tag, "0.00" if tag == "假两件" else f"{score:.2f}")
        for tag, score in zip(labels, scores)
    ]
    if fault == "wrong_order":
        items[0], items[1] = items[1], items[0]
    output = {"scores": dict(items)}
    if fault == "bad_decimal":
        output["scores"][labels[0]] = "0.9"
    if fault == "unsupported_nonzero":
        output["scores"]["假两件"] = "0.42"
else:
    raise SystemExit("unexpected mode")
args.output.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
with (PACKAGE_DIR / "invocations.jsonl").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"mode": args.mode, "scores": scores}) + "\n")
'''


class Fixture:
    def __init__(self, root: Path, *, fault: str | None = None) -> None:
        self.root = root
        self.delivery = root / "delivery_candidate"
        self.posttrain = root / "posttrain"
        self.output = root / "final_mode_verification"
        self.delivery.mkdir(parents=True)
        self.posttrain.mkdir(parents=True)
        self.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.labels = self.schema["labels"]
        assert len(self.labels) == 57 and "假两件" in self.labels
        self.selected_tags = {
            "局部结构": [self.labels[0]],
            "廓形": [self.labels[20]],
            "工艺": [self.labels[30]],
            "面辅料": [self.labels[40]],
        }
        _write_json(
            self.delivery / "model_config.json",
            {
                "tag_order": self.labels,
                "selected_tags": self.selected_tags,
                "fault": fault,
            },
        )
        (self.delivery / "infer.py").write_text(FAKE_INFER, encoding="utf-8")
        os.chmod(self.delivery / "infer.py", 0o755)
        (self.delivery / "lora_and_classifier.safetensors").write_bytes(b"weights")

        self.float_path = self.posttrain / "reproduced_32_float32.jsonl"
        self.selected_path = self.posttrain / "reproduced_32_selected_only.jsonl"
        self.float_rows: list[dict] = []
        self.selected_rows: list[dict] = []
        for index in range(32):
            scores = [((index * 7 + position * 3) % 101) / 100 for position in range(57)]
            scores[self.labels.index("假两件")] = 0.91
            evidence = {
                "record_id": f"verification:{index:02d}",
                "image_path": f"/dataset/images/{index:02d}.jpg",
                "image_sha256": f"{index + 1:064x}",
            }
            self.float_rows.append({**evidence, "scores": scores})
            self.selected_rows.append(
                {
                    **evidence,
                    "output": {
                        category: list(tags)
                        for category, tags in self.selected_tags.items()
                    },
                }
            )
        _write_jsonl(self.float_path, self.float_rows)
        _write_jsonl(self.selected_path, self.selected_rows)
        self.reproduction_path = self.posttrain / "reproduction_result.json"
        self.rewrite_reproduction()

    def rewrite_reproduction(self) -> None:
        _write_json(
            self.reproduction_path,
            {
                "candidate_weights_sha256": _sha256(
                    self.delivery / "lora_and_classifier.safetensors"
                ),
                "candidate_model_config_sha256": _sha256(
                    self.delivery / "model_config.json"
                ),
                "candidate_infer_sha256": _sha256(self.delivery / "infer.py"),
                "reproduced_float32_path": str(self.float_path),
                "reproduced_float32_sha256": _sha256(self.float_path),
                "reproduced_selected_only_path": str(self.selected_path),
                "reproduced_selected_only_sha256": _sha256(self.selected_path),
                "commands": ["real gpu reproduction"],
                "environment": {
                    "gpu": "NVIDIA H20",
                    "cuda": "12.6",
                    "pytorch": "2.7.1+cu126",
                    "transformers": "4.57.1",
                    "peft": "0.17.1",
                    "safetensors": "0.7.0",
                    "pillow": "12.1.1",
                },
            },
        )


@pytest.fixture
def fixture(tmp_path: Path) -> Fixture:
    return Fixture(tmp_path)


def test_runs_candidate_infer_for_32_rows_and_atomically_writes_three_files(
    fixture: Fixture,
) -> None:
    result = verify_final_modes(fixture.delivery, fixture.posttrain, fixture.output)

    assert result["status"] == "success"
    assert result["records"] == 32
    assert result["score_values"] == 1824
    assert result["candidate_infer_sha256"] == _sha256(fixture.delivery / "infer.py")
    assert result["reproduction_result_sha256"] == _sha256(
        fixture.reproduction_path
    )
    assert result["source_float32_sha256"] == _sha256(fixture.float_path)
    assert result["source_selected_only_sha256"] == _sha256(fixture.selected_path)
    assert list(result["environment"]) == [
        "gpu",
        "cuda",
        "pytorch",
        "transformers",
        "peft",
        "safetensors",
        "pillow",
    ]
    assert len(result["commands"]) == 64
    assert all(str(fixture.delivery / "infer.py") in command for command in result["commands"])
    assert sum("--mode selected_with_confidence" in command for command in result["commands"]) == 32
    assert sum("--mode all_scores" in command for command in result["commands"]) == 32

    assert sorted(path.name for path in fixture.output.iterdir()) == [
        "final_mode_verification.json",
        "reproduced_32_all_scores.jsonl",
        "reproduced_32_selected_with_confidence.jsonl",
    ]
    summary = json.loads(
        (fixture.output / "final_mode_verification.json").read_text(encoding="utf-8")
    )
    assert summary == result
    invocations = [
        json.loads(line)
        for line in (fixture.delivery / "invocations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(invocations) == 64


def test_outputs_bind_evidence_and_enforce_all_cross_mode_values(
    fixture: Fixture,
) -> None:
    verify_final_modes(fixture.delivery, fixture.posttrain, fixture.output)
    confidence_rows = [
        json.loads(line)
        for line in (
            fixture.output / "reproduced_32_selected_with_confidence.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    all_score_rows = [
        json.loads(line)
        for line in (fixture.output / "reproduced_32_all_scores.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(confidence_rows) == len(all_score_rows) == 32
    for source, selected, confidence, all_scores in zip(
        fixture.float_rows, fixture.selected_rows, confidence_rows, all_score_rows
    ):
        evidence = {key: source[key] for key in ("record_id", "image_path", "image_sha256")}
        assert {key: confidence[key] for key in evidence} == evidence
        assert {key: all_scores[key] for key in evidence} == evidence
        score_map = all_scores["output"]["scores"]
        assert list(score_map) == fixture.labels
        assert len(score_map) == 57
        assert all(TWO_DECIMAL.fullmatch(value) for value in score_map.values())
        assert score_map["假两件"] == "0.00"
        stripped = {
            category: [item["name"] for item in items]
            for category, items in confidence["output"].items()
        }
        assert stripped == selected["output"]
        for items in confidence["output"].values():
            for item in items:
                assert item["confidence"] == score_map[item["name"]]


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        ("infer.py", "candidate_infer_sha256"),
        ("model_config.json", "candidate_model_config_sha256"),
        ("lora_and_classifier.safetensors", "candidate_weights_sha256"),
    ],
)
def test_rejects_candidate_or_sealed_package_hash_drift_before_execution(
    fixture: Fixture, artifact: str, message: str
) -> None:
    with (fixture.delivery / artifact).open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(FinalModeVerificationError, match=message):
        verify_final_modes(fixture.delivery, fixture.posttrain, fixture.output)

    assert not fixture.output.exists()
    assert not (fixture.delivery / "invocations.jsonl").exists()


def test_rejects_reproduction_input_hash_drift_before_execution(fixture: Fixture) -> None:
    with fixture.float_path.open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(FinalModeVerificationError, match="reproduced_float32_sha256"):
        verify_final_modes(fixture.delivery, fixture.posttrain, fixture.output)

    assert not fixture.output.exists()
    assert not (fixture.delivery / "invocations.jsonl").exists()


@pytest.mark.parametrize("result_sha256", ["f" * 64, None])
def test_sealed_delivery_must_bind_the_current_reproduction_result(
    fixture: Fixture, result_sha256: str | None
) -> None:
    verification = {
        "status": "success",
        "customer_ready": True,
    }
    if result_sha256 is not None:
        verification["result_sha256"] = result_sha256
    _write_json(
        fixture.delivery / "VERIFICATION.json",
        verification,
    )

    with pytest.raises(FinalModeVerificationError, match="sealed.*reproduction"):
        verify_final_modes(fixture.delivery, fixture.posttrain, fixture.output)

    assert not fixture.output.exists()
    assert not (fixture.delivery / "invocations.jsonl").exists()


def test_rejects_selected_only_reformat_mismatch_without_publishing(
    fixture: Fixture,
) -> None:
    fixture.selected_rows[0]["output"]["局部结构"] = []
    _write_jsonl(fixture.selected_path, fixture.selected_rows)
    fixture.rewrite_reproduction()

    with pytest.raises(FinalModeVerificationError, match="selected-only"):
        verify_final_modes(fixture.delivery, fixture.posttrain, fixture.output)

    assert not fixture.output.exists()


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("wrong_order", "schema order"),
        ("bad_decimal", "two-decimal"),
        ("unsupported_nonzero", "假两件"),
        ("confidence_name", "selected-only"),
        ("confidence_value", "all_scores"),
        ("bad_confidence_decimal", "two-decimal"),
    ],
)
def test_rejects_invalid_package_mode_contracts_atomically(
    tmp_path: Path, fault: str, message: str
) -> None:
    fixture = Fixture(tmp_path, fault=fault)

    with pytest.raises(FinalModeVerificationError, match=message):
        verify_final_modes(fixture.delivery, fixture.posttrain, fixture.output)

    assert not fixture.output.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("31_rows", "exactly 32"),
        ("56_scores", "exactly 57"),
        ("missing_environment", "environment.*pillow"),
    ],
)
def test_rejects_incomplete_32_by_57_reproduction_contract(
    fixture: Fixture, mutation: str, message: str
) -> None:
    reproduction = json.loads(fixture.reproduction_path.read_text(encoding="utf-8"))
    if mutation == "31_rows":
        _write_jsonl(fixture.float_path, fixture.float_rows[:-1])
        reproduction["reproduced_float32_sha256"] = _sha256(fixture.float_path)
    elif mutation == "56_scores":
        fixture.float_rows[0]["scores"] = fixture.float_rows[0]["scores"][:-1]
        _write_jsonl(fixture.float_path, fixture.float_rows)
        reproduction["reproduced_float32_sha256"] = _sha256(fixture.float_path)
    else:
        del reproduction["environment"]["pillow"]
    _write_json(fixture.reproduction_path, reproduction)

    with pytest.raises(FinalModeVerificationError, match=message):
        verify_final_modes(fixture.delivery, fixture.posttrain, fixture.output)

    assert not fixture.output.exists()


def test_cli_returns_zero_and_replaces_an_existing_output_as_one_bundle(
    fixture: Fixture,
) -> None:
    fixture.output.mkdir()
    (fixture.output / "stale.txt").write_text("stale", encoding="utf-8")

    assert (
        main(
            [
                "--delivery-dir",
                str(fixture.delivery),
                "--posttrain-dir",
                str(fixture.posttrain),
                "--output-dir",
                str(fixture.output),
                "--python-executable",
                sys.executable,
            ]
        )
        == 0
    )
    assert not (fixture.output / "stale.txt").exists()
    assert (fixture.output / "final_mode_verification.json").is_file()
