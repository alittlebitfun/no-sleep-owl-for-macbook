from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from scripts import package_unified57_delivery as delivery


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "configs" / "bosideng_unified57_schema.json"
FINAL_PROMPT_PATH = Path(
    "/Users/Zhuanz1/.codex/attachments/"
    "ea08ce0b-f36f-472a-8eb0-82ac537aaa77/pasted-text.txt"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class PackageFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.checkpoint = root / "formal_v3.pt"
        self.thresholds = root / "thresholds.json"
        self.metrics = root / "evaluation_report.json"
        self.predictions = root / "test_predictions_float32.jsonl"
        self.verification = root / "verification_input"
        self.output = root / "delivery"
        self.verification.mkdir()
        self._write_checkpoint()
        self._write_thresholds()
        self._write_prediction_and_verification_rows()
        self._write_metrics()

    def _write_checkpoint(self) -> None:
        state = {
            (
                f"backbone.base_model.model.layers.{index:03d}."
                "self_attn.q_proj.lora_A.default.weight"
            ): torch.full((1, 1), float(index), dtype=torch.float32)
            for index in range(288)
        }
        state["classifier.weight"] = torch.arange(57 * 3, dtype=torch.float32).reshape(
            57, 3
        )
        state["classifier.bias"] = torch.arange(57, dtype=torch.float32)
        payload = {
            "format_version": 3,
            "tag_order": self.schema["labels"],
            "schema_sha256": self.schema["schema_sha256"],
            "manifest_sha256": "1" * 64,
            "mask_contract_version": "unified57_known_pu_positive_disjoint_v1",
            "pu_output_semantics": "uncalibrated_confidence",
            "loss_contract": {
                "version": "unified57_pn_sample_mean_bce_pairwise_pu_v1",
                "pn": "per-sample masked BCE over known_mask cells",
                "pu": "FP32 positive-vs-unlabeled pairwise hinge ranking",
                "unknown_is_negative": False,
                "pu_output_semantics": "uncalibrated_confidence",
            },
            "model": state,
            "trainable_names": sorted(state),
            "initialization_audit": {
                "mode": "aggregate18_v2_weight_transfer",
                "source_checkpoint_sha256": "2" * 64,
            },
            "run_contract": {
                "base_model": "/models/Qwen3-VL-8B-Instruct",
                "base_model_config_sha256": "3" * 64,
                "image_max_pixels": 112896,
                "lora_rank": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "head_dropout": 0.1,
                "dtype": "bfloat16",
                "vision_prompt_sha256": hashlib.sha256(
                    delivery.MODEL_INPUT_PROMPT.encode("utf-8")
                ).hexdigest(),
            },
        }
        torch.save(payload, self.checkpoint)

    def _write_thresholds(self) -> None:
        labels: dict[str, dict] = {}
        for tag in self.schema["labels"]:
            mode = self.schema["label_training_modes"][tag]
            disabled = mode == "unsupported"
            labels[tag] = {
                "mode": mode,
                "threshold": None if disabled else 0.5,
                "method": (
                    "disabled"
                    if disabled
                    else "observed_pn_f1"
                    if mode == "pn"
                    else "positive_minus_unlabeled_coverage"
                ),
                "status": ("disabled_unsupported" if disabled else "calibrated"),
                "support": {},
            }
        _write_json(
            self.thresholds,
            {
                "schema_version": self.schema["schema_version"],
                "schema_sha256": self.schema["schema_sha256"],
                "checkpoint_sha256": _sha256(self.checkpoint),
                "validation_manifest_sha256": "4" * 64,
                "fallback_threshold": 0.5,
                "labels": labels,
            },
        )

    def _write_prediction_and_verification_rows(self) -> None:
        float_rows: list[dict] = []
        manifests: list[dict] = []
        selected: list[dict] = []
        for index in range(32):
            record_id = f"verification:{index:02d}"
            scores = [((index + label_index) % 100) / 100 for label_index in range(57)]
            float_rows.append(
                {
                    "record_id": record_id,
                    "image_path": f"images/{index:02d}.jpg",
                    "image_sha256": f"{index + 10:064x}",
                    "source": "jd_complete23",
                    "sources": ["jd_complete23"],
                    "scores": scores,
                    "labels": [0.0] * 57,
                    "known_mask": [0] * 57,
                    "pu_positive_mask": [0] * 57,
                    "schema_version": self.schema["schema_version"],
                    "schema_sha256": self.schema["schema_sha256"],
                    "checkpoint_sha256": _sha256(self.checkpoint),
                }
            )
            manifests.append(
                {
                    "record_id": record_id,
                    "test_manifest_index": index,
                    "image_path": f"images/{index:02d}.jpg",
                    "image_sha256": f"{index + 10:064x}",
                    "source": "jd_complete23",
                    "sources": ["jd_complete23"],
                    "selection_bucket": "jd_only_pn",
                }
            )
            selected.append(
                {
                    "record_id": record_id,
                    "output": delivery.render_selected_only(
                        scores,
                        {
                            tag: (None if tag == "假两件" else 0.5)
                            for tag in self.schema["labels"]
                        },
                        self.schema,
                    ),
                }
            )
        _write_jsonl(self.predictions, float_rows)
        _write_jsonl(self.verification / "verification_32_manifest.jsonl", manifests)
        _write_jsonl(self.verification / "reference_32_float32.jsonl", float_rows)
        _write_jsonl(self.verification / "reference_32_selected_only.jsonl", selected)

    def _write_metrics(self) -> None:
        _write_json(
            self.metrics,
            {
                "status": "success",
                "provenance": {
                    "schema_sha256": self.schema["schema_sha256"],
                    "checkpoint_sha256": _sha256(self.checkpoint),
                    "thresholds_sha256": _sha256(self.thresholds),
                    "predictions_sha256": _sha256(self.predictions),
                    "validation_manifest_sha256": "4" * 64,
                    "test_manifest_sha256": "5" * 64,
                },
                "output_quality": {"json_validity_rate": 1.0},
                "reproduction_32": {
                    "records": 32,
                    "score_values": 1824,
                    "probabilities_exact": True,
                    "max_abs_score_delta": 0.0,
                    "selected_outputs_exact": True,
                },
            },
        )

    def build(self) -> dict:
        return delivery.build_delivery_package(
            checkpoint_path=self.checkpoint,
            schema_path=SCHEMA_PATH,
            thresholds_path=self.thresholds,
            metrics_path=self.metrics,
            predictions_path=self.predictions,
            verification_dir=self.verification,
            final_prompt_path=FINAL_PROMPT_PATH,
            output_dir=self.output,
        )


@pytest.fixture
def package(tmp_path: Path) -> PackageFixture:
    fixture = PackageFixture(tmp_path)
    fixture.build()
    return fixture


def test_builds_lightweight_split_safetensors_package(package: PackageFixture) -> None:
    expected = {
        "README.md",
        "SHA256SUMS",
        "VERIFICATION.json",
        "final_prompt.txt",
        "infer.py",
        "label_schema.json",
        "lora_and_classifier.safetensors",
        "model_config.json",
        "requirements.txt",
        "tests/test_decode_equivalence.py",
        "thresholds.json",
        "verification/reference_32_float32.jsonl",
        "verification/reference_32_selected_only.jsonl",
        "verification/verification_32_manifest.jsonl",
    }
    actual = {
        path.relative_to(package.output).as_posix()
        for path in package.output.rglob("*")
        if path.is_file()
    }
    assert actual == expected
    assert not any("base" in path.name.lower() for path in package.output.rglob("*"))

    weights = load_file(str(package.output / "lora_and_classifier.safetensors"))
    assert len(weights) == 290
    assert len([name for name in weights if "lora_" in name]) == 288
    assert {name for name in weights if name.startswith("classifier.")} == {
        "classifier.weight",
        "classifier.bias",
    }
    assert weights["classifier.weight"].shape == (57, 3)
    assert weights["classifier.bias"].shape == (57,)

    config = json.loads(
        (package.output / "model_config.json").read_text(encoding="utf-8")
    )
    assert config["base_model"]["included"] is False
    assert config["checkpoint"]["sha256"] == _sha256(package.checkpoint)
    assert config["schema"]["sha256"] == package.schema["schema_sha256"]
    assert config["schema"]["file_sha256"] == _sha256(SCHEMA_PATH)
    assert config["model_input_prompt"] == delivery.MODEL_INPUT_PROMPT
    assert (
        config["model_input_prompt_sha256"]
        == hashlib.sha256(delivery.MODEL_INPUT_PROMPT.encode("utf-8")).hexdigest()
    )
    assert config["product_prompt_sha256"] == delivery.FINAL_PROMPT_SHA256
    assert config["image_max_pixels"] == 112896
    assert config["weights"]["tensor_count"] == 290
    assert config["lora"]["tensor_count"] == 288
    assert config["classifier"]["num_labels"] == 57
    assert config["unsupported_labels"] == ["假两件"]
    assert _sha256(package.output / "label_schema.json") == _sha256(SCHEMA_PATH)
    assert _sha256(package.output / "thresholds.json") == _sha256(package.thresholds)
    assert _sha256(package.output / "final_prompt.txt") == _sha256(FINAL_PROMPT_PATH)


def test_render_modes_follow_schema_and_unsupported_contract() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    thresholds = {tag: (None if tag == "假两件" else 0.5) for tag in schema["labels"]}
    scores = [0.01] * 57
    scores[schema["labels"].index("连帽")] = 0.8
    scores[schema["labels"].index("拆卸帽")] = 0.9
    scores[schema["labels"].index("立领")] = 0.75
    scores[schema["labels"].index("假两件")] = 0.99
    scores[schema["labels"].index("压胶充绒")] = 0.8
    scores[schema["labels"].index("压胶袋盖")] = 0.8

    all_scores = delivery.render_all_scores(scores, schema)
    assert list(all_scores) == schema["labels"]
    assert len(all_scores) == 57
    assert all(
        re.fullmatch(r"(?:0\.\d{2}|1\.00)", value) for value in all_scores.values()
    )
    assert all_scores["假两件"] == "0.00"

    selected = delivery.render_selected_only(scores, thresholds, schema)
    assert list(selected) == ["局部结构", "廓形", "工艺", "面辅料"]
    assert selected["局部结构"] == ["拆卸帽", "立领"]
    assert selected["工艺"] == ["压胶充绒"]
    assert "假两件" not in sum(selected.values(), [])

    with_confidence = delivery.render_selected_with_confidence(
        scores, thresholds, schema
    )
    assert list(with_confidence) == list(selected)
    assert with_confidence["局部结构"] == [
        {"name": "拆卸帽", "confidence": "0.90"},
        {"name": "立领", "confidence": "0.75"},
    ]
    assert with_confidence["工艺"] == [{"name": "压胶充绒", "confidence": "0.80"}]


def test_reuses_training_v3_loader_and_fails_closed_on_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = PackageFixture(tmp_path)
    calls = 0
    original = delivery.load_v3_model_state_for_inference

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(delivery, "load_v3_model_state_for_inference", counted)
    fixture.build()
    assert calls == 1

    broken_root = tmp_path / "broken"
    broken_root.mkdir()
    broken = PackageFixture(broken_root)
    payload = json.loads(broken.thresholds.read_text(encoding="utf-8"))
    payload["checkpoint_sha256"] = "f" * 64
    _write_json(broken.thresholds, payload)
    with pytest.raises(ValueError, match="checkpoint file SHA256"):
        broken.build()


def test_checksums_and_verification32_references_are_complete(
    package: PackageFixture,
) -> None:
    checksum_lines = (
        (package.output / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    )
    listed = {}
    for line in checksum_lines:
        digest, relative = line.split("  ", 1)
        listed[relative] = digest
    expected_paths = sorted(
        path.relative_to(package.output).as_posix()
        for path in package.output.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    assert list(listed) == expected_paths
    assert all(
        listed[relative] == _sha256(package.output / relative)
        for relative in expected_paths
    )

    verification = json.loads(
        (package.output / "VERIFICATION.json").read_text(encoding="utf-8")
    )
    assert verification["records"] == 32
    assert verification["score_values"] == 1824
    assert verification["probabilities_exact"] is True
    assert verification["max_abs_score_delta"] == 0.0
    assert verification["selected_outputs_exact"] is True
    assert verification["references"]["predictions_sha256"] == _sha256(
        package.predictions
    )


def test_rejects_verification_reference_drift_from_frozen_predictions(
    tmp_path: Path,
) -> None:
    fixture = PackageFixture(tmp_path)
    reference = fixture.verification / "reference_32_float32.jsonl"
    rows = [
        json.loads(line) for line in reference.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["scores"][0] = 0.99
    _write_jsonl(reference, rows)
    with pytest.raises(
        ValueError, match="verification float32 differs from predictions"
    ):
        fixture.build()


def test_generated_infer_exposes_all_three_modes_without_loading_base(
    package: PackageFixture,
) -> None:
    infer_path = package.output / "infer.py"
    spec = importlib.util.spec_from_file_location(
        "generated_unified57_infer", infer_path
    )
    assert spec is not None and spec.loader is not None
    infer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infer)

    config = json.loads(
        (package.output / "model_config.json").read_text(encoding="utf-8")
    )
    thresholds = json.loads(
        (package.output / "thresholds.json").read_text(encoding="utf-8")
    )
    scores = [0.0] * 57
    scores[config["tag_order"].index("H型")] = 0.91
    scores[config["tag_order"].index("假两件")] = 1.0

    selected = infer.format_scores(scores, "selected_only", config, thresholds)
    confidence = infer.format_scores(
        scores, "selected_with_confidence", config, thresholds
    )
    all_scores = infer.format_scores(scores, "all_scores", config, thresholds)
    assert selected == {
        "局部结构": [],
        "廓形": ["H型"],
        "工艺": [],
        "面辅料": [],
    }
    assert confidence["廓形"] == [{"name": "H型", "confidence": "0.91"}]
    assert all_scores["scores"]["假两件"] == "0.00"


def test_generated_decode_and_single_weights_contract(
    package: PackageFixture, tmp_path: Path
) -> None:
    test_path = package.output / "tests" / "test_decode_equivalence.py"
    spec = importlib.util.spec_from_file_location("generated_decode_test", test_path)
    assert spec is not None and spec.loader is not None
    generated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generated)
    generated.test_decode_equivalence_at_336_square_budget(tmp_path)
    generated.test_single_safetensors_has_exact_trainable_contract()
