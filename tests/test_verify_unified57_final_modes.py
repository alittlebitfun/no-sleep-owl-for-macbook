from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import shlex
import sys
from pathlib import Path

import pytest

from scripts import verify_unified57_final_modes as verifier
from scripts.audit_unified57_final_delivery import audit_packaged_output_modes
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


def _write_sha256s(root: Path) -> None:
    paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (root / "SHA256SUMS").write_text(
        "".join(f"{_sha256(root / relative)}  {relative}\n" for relative in paths),
        encoding="utf-8",
    )


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _tree_inventory(root: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            result[relative] = {"type": "directory"}
        elif path.is_file():
            result[relative] = {"type": "file", "sha256": _sha256(path)}
        else:
            result[relative] = {"type": "special"}
    return result


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
if fault == "mutate_package":
    (PACKAGE_DIR / "runtime-mutation.txt").write_text("mutated", encoding="utf-8")
if fault == "mutate_empty_dir":
    (PACKAGE_DIR / "runtime-empty-dir").mkdir(exist_ok=True)
if fault == "mutate_source":
    with Path(config["mutation_target"]).open("a", encoding="utf-8") as handle:
        handle.write("\n")
'''


class Fixture:
    def __init__(self, root: Path, *, fault: str | None = None) -> None:
        self.root = root
        self.delivery = root / "delivery_candidate"
        self.posttrain = root / "posttrain"
        self.output = self.posttrain / "final_mode_verification"
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
                "mutation_target": str(
                    self.posttrain / "reproduced_32_float32.jsonl"
                ),
            },
        )
        (self.delivery / "infer.py").write_text(FAKE_INFER, encoding="utf-8")
        os.chmod(self.delivery / "infer.py", 0o755)
        (self.delivery / "lora_and_classifier.safetensors").write_bytes(b"weights")
        self.provenance = {
            "checkpoint_sha256": "1" * 64,
            "schema_sha256": self.schema["schema_sha256"],
            "schema_file_sha256": "2" * 64,
            "thresholds_sha256": "3" * 64,
            "final_prompt_sha256": "4" * 64,
            "weights_sha256": _sha256(
                self.delivery / "lora_and_classifier.safetensors"
            ),
            "metrics_sha256": "5" * 64,
            "test_manifest_sha256": "6" * 64,
            "trainable_manifest_sha256": "7" * 64,
            "base_artifact_manifest_sha256": "8" * 64,
        }
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
        self.verification_dir = self.delivery / "verification"
        self.verification_dir.mkdir()
        self.verification_manifest_path = (
            self.verification_dir / "verification_32_manifest.jsonl"
        )
        self.reference_float_path = (
            self.verification_dir / "reference_32_float32.jsonl"
        )
        self.reference_selected_path = (
            self.verification_dir / "reference_32_selected_only.jsonl"
        )
        _write_jsonl(
            self.verification_manifest_path,
            [
                {
                    "record_id": row["record_id"],
                    "image_path": row["image_path"],
                    "image_sha256": row["image_sha256"],
                    "test_manifest_index": index,
                }
                for index, row in enumerate(self.float_rows)
            ],
        )
        _write_jsonl(self.reference_float_path, self.float_rows)
        _write_jsonl(
            self.reference_selected_path,
            [
                {"record_id": row["record_id"], "output": row["output"]}
                for row in self.selected_rows
            ],
        )
        self.reference_hashes = {
            path.name: _sha256(path)
            for path in (
                self.verification_manifest_path,
                self.reference_float_path,
                self.reference_selected_path,
            )
        }
        _write_json(
            self.delivery / "VERIFICATION.json",
            {
                "status": "pending_reproduction",
                "evaluation_status": "success",
                "customer_ready": False,
                "internal_use_only": True,
                "provenance": self.provenance,
                "references": self.reference_hashes,
            },
        )
        self.reproduction_path = self.posttrain / "reproduction_result.json"
        self.rewrite_reproduction()

    def make_sealed(self) -> Path:
        sealed = self.root / "sealed_delivery"
        shutil.copytree(self.delivery, sealed)
        reproduction = json.loads(
            self.reproduction_path.read_text(encoding="utf-8")
        )
        reproduction_32 = {
            "records": 32,
            "score_values": 1824,
            "probabilities_exact": True,
            "max_abs_score_delta": 0.0,
            "selected_outputs_exact": True,
            "selected_mismatch_records": 0,
            "image_sha256s_exact": True,
            "commands": reproduction["commands"],
            "environment": reproduction["environment"],
            "result_sha256": _sha256(self.reproduction_path),
            "reproduced_float32_sha256": reproduction[
                "reproduced_float32_sha256"
            ],
            "reproduced_selected_only_sha256": reproduction[
                "reproduced_selected_only_sha256"
            ],
        }
        _write_json(
            sealed / "VERIFICATION.json",
            {
                "status": "success",
                "evaluation_status": "success",
                "customer_ready": True,
                "internal_use_only": False,
                "provenance": dict(self.provenance),
                "references": dict(self.reference_hashes),
                **reproduction_32,
                "reproduction_32": reproduction_32,
            },
        )
        _write_sha256s(sealed)
        return sealed

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
                "commands": [
                    (
                        f"{sys.executable} {self.delivery / 'infer.py'} "
                        f"--verification-manifest {self.verification_manifest_path} "
                        "--mode verification_float32"
                    ),
                    (
                        f"{sys.executable} {self.delivery / 'infer.py'} "
                        f"--verification-manifest {self.verification_manifest_path} "
                        "--mode selected_only"
                    ),
                ],
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
    candidate_before = _tree_hashes(fixture.delivery)
    result = verify_final_modes(fixture.delivery, fixture.posttrain, fixture.output)

    assert result["status"] == "success"
    assert result["records"] == 32
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
    assert _tree_hashes(fixture.delivery) == candidate_before


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


def test_reproduction_ids_must_be_unique_and_match_candidate_reference_order(
    fixture: Fixture,
) -> None:
    fixture.float_rows[1]["record_id"] = fixture.float_rows[0]["record_id"]
    fixture.selected_rows[1]["record_id"] = fixture.selected_rows[0]["record_id"]
    _write_jsonl(fixture.float_path, fixture.float_rows)
    _write_jsonl(fixture.selected_path, fixture.selected_rows)
    fixture.rewrite_reproduction()

    with pytest.raises(FinalModeVerificationError, match="unique.*record_id"):
        verify_final_modes(fixture.delivery, fixture.posttrain, fixture.output)


def test_candidate_reference_hash_and_float_content_are_acceptance_roots(
    fixture: Fixture,
) -> None:
    reference_rows = [dict(row) for row in fixture.float_rows]
    reference_rows[0] = {
        **reference_rows[0],
        "scores": [0.42, *reference_rows[0]["scores"][1:]],
    }
    _write_jsonl(fixture.reference_float_path, reference_rows)
    candidate_verification = json.loads(
        (fixture.delivery / "VERIFICATION.json").read_text(encoding="utf-8")
    )
    candidate_verification["references"][fixture.reference_float_path.name] = _sha256(
        fixture.reference_float_path
    )
    _write_json(fixture.delivery / "VERIFICATION.json", candidate_verification)

    with pytest.raises(FinalModeVerificationError, match="candidate float reference"):
        verify_final_modes(fixture.delivery, fixture.posttrain, fixture.output)


def test_candidate_reference_file_hash_drift_is_rejected(fixture: Fixture) -> None:
    with fixture.reference_selected_path.open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(FinalModeVerificationError, match="candidate reference.*SHA256"):
        verify_final_modes(fixture.delivery, fixture.posttrain, fixture.output)


def test_original_reproduction_commands_must_cover_frozen_manifest_and_modes(
    fixture: Fixture,
) -> None:
    reproduction = json.loads(fixture.reproduction_path.read_text(encoding="utf-8"))
    reproduction["commands"] = ["python unrelated.py"]
    _write_json(fixture.reproduction_path, reproduction)

    with pytest.raises(FinalModeVerificationError, match="reproduction commands"):
        verify_final_modes(fixture.delivery, fixture.posttrain, fixture.output)


def test_reproduction_command_substrings_do_not_substitute_for_semantic_argv(
    fixture: Fixture,
) -> None:
    reproduction = json.loads(fixture.reproduction_path.read_text(encoding="utf-8"))
    reproduction["commands"] = [
        (
            f"echo {fixture.delivery / 'infer.py'} --verification-manifest "
            f"{fixture.verification_manifest_path} --mode verification_float32"
        ),
        (
            f"echo {fixture.delivery / 'infer.py'} --verification-manifest "
            f"{fixture.verification_manifest_path} --mode selected_only"
        ),
    ]
    _write_json(fixture.reproduction_path, reproduction)

    with pytest.raises(FinalModeVerificationError, match="reproduction commands"):
        verify_final_modes(fixture.delivery, fixture.posttrain, fixture.output)


@pytest.mark.parametrize("symlink_parent", [False, True])
def test_output_resolved_path_must_not_overlap_candidate_or_inputs(
    fixture: Fixture, symlink_parent: bool
) -> None:
    if symlink_parent:
        alias = fixture.root / "candidate-alias"
        alias.symlink_to(fixture.delivery, target_is_directory=True)
        output = alias / "nested-output"
    else:
        output = fixture.delivery / "nested-output"

    with pytest.raises(
        FinalModeVerificationError, match="^output path overlaps protected input"
    ):
        verify_final_modes(fixture.delivery, fixture.posttrain, output)

    assert not (fixture.delivery / "nested-output").exists()


@pytest.mark.parametrize("protected", ["reproduced_file", "posttrain_ancestor"])
def test_existing_output_overlap_is_reported_before_overwrite_refusal(
    fixture: Fixture, protected: str
) -> None:
    output = fixture.float_path if protected == "reproduced_file" else fixture.posttrain

    with pytest.raises(
        FinalModeVerificationError, match="^output path overlaps protected input"
    ):
        verify_final_modes(fixture.delivery, fixture.posttrain, output)


def test_scratch_is_outside_publish_tree_and_bundle_has_three_regular_files(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[Path] = []
    original = verifier.tempfile.mkdtemp

    def recording_mkdtemp(*args, **kwargs):
        path = Path(original(*args, **kwargs))
        created.append(path)
        return str(path)

    monkeypatch.setattr(verifier.tempfile, "mkdtemp", recording_mkdtemp)
    verify_final_modes(fixture.delivery, fixture.posttrain, fixture.output)

    assert len(created) == 2
    assert all(first not in second.parents for first in created for second in created if first != second)
    files = list(fixture.output.iterdir())
    assert len(files) == 3
    assert all(path.is_file() and not path.is_symlink() for path in files)


@pytest.mark.parametrize("fault", ["mutate_package", "mutate_empty_dir"])
def test_candidate_package_mutation_during_formatting_is_detected(
    tmp_path: Path, fault: str
) -> None:
    fixture = Fixture(tmp_path, fault=fault)

    with pytest.raises(FinalModeVerificationError, match="candidate package changed"):
        verify_final_modes(fixture.delivery, fixture.posttrain, fixture.output)

    assert not fixture.output.exists()


def test_reproduction_source_mutation_during_formatting_is_detected(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path, fault="mutate_source")

    with pytest.raises(FinalModeVerificationError, match="source inputs changed"):
        verify_final_modes(fixture.delivery, fixture.posttrain, fixture.output)

    assert not fixture.output.exists()


def test_execution_hash_ledger_is_complete_and_durable(fixture: Fixture) -> None:
    sealed = fixture.make_sealed()
    expected_candidate = _tree_hashes(fixture.delivery)
    expected_sealed = _tree_hashes(sealed)
    expected_candidate_entries = _tree_inventory(fixture.delivery)
    expected_sealed_entries = _tree_inventory(sealed)

    result = verify_final_modes(
        fixture.delivery,
        fixture.posttrain,
        fixture.output,
        sealed_delivery_dir=sealed,
    )

    interpreter = result["python_executable"]
    assert Path(interpreter["path"]).is_file()
    assert interpreter["sha256"] == _sha256(Path(interpreter["path"]))
    assert result["candidate_inventory"]["files"] == expected_candidate
    assert result["sealed_inventory"]["files"] == expected_sealed
    assert result["candidate_inventory"]["entries"] == expected_candidate_entries
    assert result["sealed_inventory"]["entries"] == expected_sealed_entries
    assert result["candidate_package_unchanged"] is True
    assert result["sealed_package_unchanged"] is True
    assert len(result["score_payloads"]) == 32
    assert sum(len(row["payload"]["scores"]) for row in result["score_payloads"]) == 1824
    assert len(result["execution_ledger"]) == 64
    assert result["commands"] == [
        shlex.join(entry["argv"]) for entry in result["execution_ledger"]
    ]
    payload_by_id = {row["record_id"]: row for row in result["score_payloads"]}
    for entry in result["execution_ledger"]:
        assert isinstance(entry["argv"], list) and entry["argv"][0] == interpreter["path"]
        assert entry["python_executable_sha256"] == interpreter["sha256"]
        assert entry["return_code"] == 0
        assert re.fullmatch(r"[0-9a-f]{64}", entry["input_sha256"])
        assert re.fullmatch(r"[0-9a-f]{64}", entry["output_sha256"])
        assert entry["input_sha256"] == payload_by_id[entry["record_id"]]["sha256"]
    assert {
        "candidate_VERIFICATION.json",
        "sealed_VERIFICATION.json",
        "sealed_SHA256SUMS",
        "verification_32_manifest.jsonl",
        "reference_32_float32.jsonl",
        "reference_32_selected_only.jsonl",
    }.issubset(result["input_sha256"])


def test_self_mutating_python_executable_is_rejected(fixture: Fixture) -> None:
    wrapper = fixture.root / "format-python-wrapper"
    replacement = (
        "#!/bin/sh\n"
        "# mutated interpreter wrapper\n"
        f"exec {shlex.quote(sys.executable)} \"$@\"\n"
    )
    wrapper.write_text(
        "#!/bin/sh\n"
        f"printf '%s' {shlex.quote(replacement)} > \"$0.tmp\"\n"
        "chmod +x \"$0.tmp\"\n"
        "mv \"$0.tmp\" \"$0\"\n"
        f"exec {shlex.quote(sys.executable)} \"$@\"\n",
        encoding="utf-8",
    )
    os.chmod(wrapper, 0o755)

    with pytest.raises(FinalModeVerificationError, match="python executable changed"):
        verify_final_modes(
            fixture.delivery,
            fixture.posttrain,
            fixture.output,
            python_executable=str(wrapper),
        )

    assert not fixture.output.exists()


def test_malformed_unhashable_tag_order_is_a_controlled_contract_error(
    fixture: Fixture,
) -> None:
    config = json.loads(
        (fixture.delivery / "model_config.json").read_text(encoding="utf-8")
    )
    config["tag_order"][0] = {"unhashable": True}
    _write_json(fixture.delivery / "model_config.json", config)
    reproduction = json.loads(fixture.reproduction_path.read_text(encoding="utf-8"))
    reproduction["candidate_model_config_sha256"] = _sha256(
        fixture.delivery / "model_config.json"
    )
    _write_json(fixture.reproduction_path, reproduction)

    with pytest.raises(FinalModeVerificationError, match="tag_order"):
        verify_final_modes(fixture.delivery, fixture.posttrain, fixture.output)


def test_final_mode_bundle_passes_the_full_independent_auditor(
    fixture: Fixture,
) -> None:
    verify_final_modes(fixture.delivery, fixture.posttrain, fixture.output)

    audited = audit_packaged_output_modes(
        fixture.posttrain, fixture.schema, expected_records=32
    )

    assert audited["complete"] is True
    assert audited["score_values"] == 1824


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


def test_rejects_reproduction_input_hash_drift_before_execution(fixture: Fixture) -> None:
    with fixture.float_path.open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(FinalModeVerificationError, match="reproduced_float32_sha256"):
        verify_final_modes(fixture.delivery, fixture.posttrain, fixture.output)

    assert not fixture.output.exists()


@pytest.mark.parametrize("result_sha256", ["f" * 64, None])
def test_sealed_delivery_must_bind_the_current_reproduction_result(
    fixture: Fixture, result_sha256: str | None
) -> None:
    sealed = fixture.make_sealed()
    verification = json.loads(
        (sealed / "VERIFICATION.json").read_text(encoding="utf-8")
    )
    if result_sha256 is not None:
        verification["result_sha256"] = result_sha256
    else:
        del verification["result_sha256"]
    _write_json(sealed / "VERIFICATION.json", verification)
    _write_sha256s(sealed)

    with pytest.raises(FinalModeVerificationError, match="sealed.*reproduction"):
        verify_final_modes(
            fixture.delivery,
            fixture.posttrain,
            fixture.output,
            sealed_delivery_dir=sealed,
        )

    assert not fixture.output.exists()


def test_candidate_executes_formats_while_sealed_dir_only_supplies_metadata(
    fixture: Fixture,
) -> None:
    sealed = fixture.make_sealed()
    candidate_before = _tree_hashes(fixture.delivery)
    sealed_before = _tree_hashes(sealed)

    result = verify_final_modes(
        fixture.delivery,
        fixture.posttrain,
        fixture.output,
        sealed_delivery_dir=sealed,
    )

    assert result["sealed_delivery_verification_status"] == "success"
    assert result["sealed_delivery_customer_ready"] is True
    assert all(str(fixture.delivery / "infer.py") in command for command in result["commands"])
    assert all(str(sealed / "infer.py") not in command for command in result["commands"])
    assert _tree_hashes(fixture.delivery) == candidate_before
    assert _tree_hashes(sealed) == sealed_before


@pytest.mark.parametrize("relationship", ["identical", "nested", "symlink_alias"])
def test_candidate_and_sealed_directories_must_not_overlap(
    fixture: Fixture, relationship: str
) -> None:
    if relationship == "identical":
        sealed = fixture.delivery
    elif relationship == "symlink_alias":
        alias = fixture.root / "candidate-sealed-alias"
        alias.symlink_to(fixture.delivery, target_is_directory=True)
        sealed = alias
    else:
        source = fixture.make_sealed()
        sealed = fixture.delivery / "sealed_nested"
        shutil.move(source, sealed)

    with pytest.raises(
        FinalModeVerificationError, match="candidate.*sealed.*overlap"
    ):
        verify_final_modes(
            fixture.delivery,
            fixture.posttrain,
            fixture.output,
            sealed_delivery_dir=sealed,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_sha256s", "SHA256SUMS"),
        ("extra_file", "inventory"),
        ("empty_directory", "inventory"),
        ("nested_sha256s", "inventory"),
        ("tampered_file", "checksum mismatch"),
    ],
)
def test_sealed_delivery_requires_exact_checksum_inventory_before_metadata(
    fixture: Fixture, mutation: str, message: str
) -> None:
    sealed = fixture.make_sealed()
    if mutation == "missing_sha256s":
        (sealed / "SHA256SUMS").unlink()
    elif mutation == "extra_file":
        (sealed / "unlisted.txt").write_text("unlisted", encoding="utf-8")
    elif mutation == "empty_directory":
        (sealed / "unlisted-empty-directory").mkdir()
    elif mutation == "nested_sha256s":
        (sealed / "nested").mkdir()
        (sealed / "nested" / "SHA256SUMS").write_text(
            "unlisted nested checksum", encoding="utf-8"
        )
    else:
        with (sealed / "model_config.json").open("ab") as handle:
            handle.write(b"tampered")

    with pytest.raises(FinalModeVerificationError, match=message):
        verify_final_modes(
            fixture.delivery,
            fixture.posttrain,
            fixture.output,
            sealed_delivery_dir=sealed,
        )

    assert not fixture.output.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("status", "sealed.*status"),
        ("customer_ready", "sealed.*customer_ready"),
        ("provenance", "sealed.*provenance"),
        ("reproduction_sha", "sealed.*reproduced_float32_sha256"),
    ],
)
def test_sealed_status_and_provenance_are_unconditionally_bound(
    fixture: Fixture, mutation: str, message: str
) -> None:
    sealed = fixture.make_sealed()
    verification = json.loads(
        (sealed / "VERIFICATION.json").read_text(encoding="utf-8")
    )
    if mutation == "status":
        verification["status"] = "pending_reproduction"
    elif mutation == "customer_ready":
        verification["customer_ready"] = False
    elif mutation == "provenance":
        verification["provenance"]["weights_sha256"] = "f" * 64
    else:
        verification["reproduced_float32_sha256"] = "f" * 64
    _write_json(sealed / "VERIFICATION.json", verification)
    _write_sha256s(sealed)

    with pytest.raises(FinalModeVerificationError, match=message):
        verify_final_modes(
            fixture.delivery,
            fixture.posttrain,
            fixture.output,
            sealed_delivery_dir=sealed,
        )

    assert not fixture.output.exists()


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("image_sha256s_exact", False),
        ("selected_mismatch_records", 1),
        ("max_abs_score_delta", 0.25),
    ],
)
def test_sealed_reproduction_status_fields_equal_independently_derived_truth(
    fixture: Fixture, field: str, bad_value: object
) -> None:
    sealed = fixture.make_sealed()
    verification = json.loads(
        (sealed / "VERIFICATION.json").read_text(encoding="utf-8")
    )
    verification[field] = bad_value
    verification["reproduction_32"][field] = bad_value
    _write_json(sealed / "VERIFICATION.json", verification)
    _write_sha256s(sealed)

    with pytest.raises(FinalModeVerificationError, match="derived reproduction"):
        verify_final_modes(
            fixture.delivery,
            fixture.posttrain,
            fixture.output,
            sealed_delivery_dir=sealed,
        )


@pytest.mark.parametrize("field", ["evaluation_status", "references"])
def test_sealed_candidate_origin_metadata_is_exactly_preserved(
    fixture: Fixture, field: str
) -> None:
    sealed = fixture.make_sealed()
    verification = json.loads(
        (sealed / "VERIFICATION.json").read_text(encoding="utf-8")
    )
    verification[field] = "fail" if field == "evaluation_status" else {"fake": "0" * 64}
    _write_json(sealed / "VERIFICATION.json", verification)
    _write_sha256s(sealed)

    with pytest.raises(FinalModeVerificationError, match="sealed candidate metadata"):
        verify_final_modes(
            fixture.delivery,
            fixture.posttrain,
            fixture.output,
            sealed_delivery_dir=sealed,
        )


def test_sealed_artifacts_must_be_byte_equal_to_executed_candidate(
    fixture: Fixture,
) -> None:
    sealed = fixture.make_sealed()
    with (sealed / "infer.py").open("ab") as handle:
        handle.write(b"\n# sealed drift\n")
    _write_sha256s(sealed)

    with pytest.raises(FinalModeVerificationError, match="sealed.*infer"):
        verify_final_modes(
            fixture.delivery,
            fixture.posttrain,
            fixture.output,
            sealed_delivery_dir=sealed,
        )


@pytest.mark.parametrize("artifact", ["verification/reference_32_float32.jsonl", "README.md"])
def test_sealed_non_metadata_tree_must_exactly_copy_candidate(
    fixture: Fixture, artifact: str
) -> None:
    if artifact == "README.md":
        (fixture.delivery / artifact).write_text("candidate readme\n", encoding="utf-8")
    sealed = fixture.make_sealed()
    with (sealed / artifact).open("a", encoding="utf-8") as handle:
        handle.write("sealed drift\n")
    _write_sha256s(sealed)

    with pytest.raises(FinalModeVerificationError, match="sealed package content"):
        verify_final_modes(
            fixture.delivery,
            fixture.posttrain,
            fixture.output,
            sealed_delivery_dir=sealed,
        )


@pytest.mark.parametrize("mutation", ["drops_candidate_field", "adds_sealed_field"])
def test_sealed_verification_is_exact_candidate_plus_reproduction(
    fixture: Fixture, mutation: str
) -> None:
    candidate_verification = json.loads(
        (fixture.delivery / "VERIFICATION.json").read_text(encoding="utf-8")
    )
    if mutation == "drops_candidate_field":
        candidate_verification["timing"] = {"total_seconds": 12.5}
        _write_json(fixture.delivery / "VERIFICATION.json", candidate_verification)
    sealed = fixture.make_sealed()
    if mutation == "adds_sealed_field":
        sealed_verification = json.loads(
            (sealed / "VERIFICATION.json").read_text(encoding="utf-8")
        )
        sealed_verification["unbound_metadata"] = {"trusted": True}
        _write_json(sealed / "VERIFICATION.json", sealed_verification)
        _write_sha256s(sealed)

    with pytest.raises(FinalModeVerificationError, match="exact candidate plus reproduction"):
        verify_final_modes(
            fixture.delivery,
            fixture.posttrain,
            fixture.output,
            sealed_delivery_dir=sealed,
        )



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


def test_cli_returns_zero_and_publishes_a_new_bundle(
    fixture: Fixture,
) -> None:
    sealed = fixture.make_sealed()

    assert (
        main(
            [
                "--candidate-dir",
                str(fixture.delivery),
                "--sealed-delivery-dir",
                str(sealed),
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
    assert (fixture.output / "final_mode_verification.json").is_file()


def test_existing_output_is_refused_without_overwrite(fixture: Fixture) -> None:
    fixture.output.mkdir()
    stale = fixture.output / "stale.txt"
    stale.write_text("stale", encoding="utf-8")

    with pytest.raises(FinalModeVerificationError, match="already exists"):
        verify_final_modes(fixture.delivery, fixture.posttrain, fixture.output)

    assert stale.read_text(encoding="utf-8") == "stale"
    assert sorted(path.name for path in fixture.output.iterdir()) == ["stale.txt"]
