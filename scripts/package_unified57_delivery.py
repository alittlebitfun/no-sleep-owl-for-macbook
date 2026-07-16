#!/usr/bin/env python3
"""Build a provenance-locked, base-model-free Unified57 delivery package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from safetensors.torch import save_file
from torch import nn

try:
    from scripts.train_unified57_qwen3vl_multilabel import (
        MASK_CONTRACT_VERSION,
        PU_OUTPUT_SEMANTICS,
        VISION_PROMPT,
        load_unified57_schema,
        load_v3_model_state_for_inference,
    )
except ModuleNotFoundError:  # Support direct execution from scripts/.
    from train_unified57_qwen3vl_multilabel import (  # type: ignore[no-redef]
        MASK_CONTRACT_VERSION,
        PU_OUTPUT_SEMANTICS,
        VISION_PROMPT,
        load_unified57_schema,
        load_v3_model_state_for_inference,
    )


PACKAGE_VERSION = "unified57-lightweight-v1"
MODEL_INPUT_PROMPT = VISION_PROMPT
FINAL_PROMPT_SHA256 = "ef6c147c99851496ee5d2154f341bb7d7fc13e717a8ec805686688af6c83216a"
EXPECTED_SCHEMA_FILE_SHA256 = (
    "43620d06b5db44f667803038b5039732bd70140c8522e70cc04158b51aed3a9a"
)
EXPECTED_SCHEMA_SHA256 = (
    "71371493ccac8d8fd31cc84fafe9c2d9ee84ef646815da1e38bdbe8e25aa2e7c"
)
EXPECTED_LORA_TENSORS = 288
EXPECTED_IMAGE_MAX_PIXELS = 336 * 336
CATEGORY_ORDER = ("局部结构", "廓形", "工艺", "面辅料")
VERIFICATION_FILENAMES = (
    "verification_32_manifest.jsonl",
    "reference_32_float32.jsonl",
    "reference_32_selected_only.jsonl",
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path | str, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read {name}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _load_jsonl(path: Path | str, *, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{name} line {line_number} is not an object")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read {name}: {path}") from exc
    if not rows:
        raise ValueError(f"{name} must contain at least one record")
    return rows


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _require_equal(actual: object, expected: object, *, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} mismatch: expected={expected!r}, actual={actual!r}")


def _score_vector(scores: Sequence[float], schema: Mapping[str, Any]) -> list[float]:
    labels = list(schema["labels"])
    if isinstance(scores, (str, bytes)) or len(scores) != len(labels):
        raise ValueError("scores must contain exactly 57 values")
    result: list[float] = []
    for index, value in enumerate(scores):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"score {index} must be numeric")
        resolved = float(value)
        if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
            raise ValueError(f"score {index} must be finite and within [0, 1]")
        result.append(resolved)
    return result


def _threshold_map(
    thresholds: Mapping[str, Any], schema: Mapping[str, Any]
) -> dict[str, float | None]:
    labels = list(schema["labels"])
    source: Mapping[str, Any]
    nested = thresholds.get("labels")
    if isinstance(nested, Mapping):
        source = nested
    else:
        source = thresholds
    if list(source) != labels:
        raise ValueError("threshold label keys must match schema order exactly")
    result: dict[str, float | None] = {}
    for tag in labels:
        item = source[tag]
        value = item.get("threshold") if isinstance(item, Mapping) else item
        if value is None:
            result[tag] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"threshold for {tag} must be numeric or null")
        resolved = float(value)
        if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
            raise ValueError(f"threshold for {tag} must be within [0, 1]")
        result[tag] = resolved
    return result


def render_all_scores(
    scores: Sequence[float], schema: Mapping[str, Any]
) -> dict[str, str]:
    """Render exactly 57 user-facing two-decimal scores in schema order."""
    values = _score_vector(scores, schema)
    result: dict[str, str] = {}
    for tag, value in zip(schema["labels"], values):
        result[tag] = "0.00" if tag == "假两件" else f"{value:.2f}"
    return result


def _selected_winners(
    scores: Sequence[float],
    thresholds: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, list[tuple[str, float]]]:
    values = _score_vector(scores, schema)
    resolved_thresholds = _threshold_map(thresholds, schema)
    labels = list(schema["labels"])
    label_index = {tag: index for index, tag in enumerate(labels)}
    semantic = schema.get("semantic_categories")
    if not isinstance(semantic, Mapping) or tuple(semantic) != CATEGORY_ORDER:
        raise ValueError("semantic category order must be 局部结构/廓形/工艺/面辅料")
    result: dict[str, list[tuple[str, float]]] = {
        category: [] for category in CATEGORY_ORDER
    }
    for category in CATEGORY_ORDER:
        subcategories = semantic[category]
        if not isinstance(subcategories, Mapping):
            raise ValueError(f"semantic category {category} must contain subcategories")
        for tags in subcategories.values():
            candidates: list[tuple[str, float]] = []
            for tag in tags:
                threshold = resolved_thresholds[tag]
                index = label_index[tag]
                if (
                    tag != "假两件"
                    and threshold is not None
                    and values[index] >= threshold
                ):
                    candidates.append((tag, values[index]))
            if candidates:
                winner = max(
                    candidates,
                    key=lambda item: (item[1], -label_index[item[0]]),
                )
                result[category].append(winner)
        result[category].sort(key=lambda item: label_index[item[0]])
    return result


def render_selected_only(
    scores: Sequence[float],
    thresholds: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, list[str]]:
    return {
        category: [tag for tag, _score in winners]
        for category, winners in _selected_winners(scores, thresholds, schema).items()
    }


def render_selected_with_confidence(
    scores: Sequence[float],
    thresholds: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, list[dict[str, str]]]:
    return {
        category: [
            {"name": tag, "confidence": f"{score:.2f}"} for tag, score in winners
        ]
        for category, winners in _selected_winners(scores, thresholds, schema).items()
    }


def _validate_selected_only(payload: object, schema: Mapping[str, Any]) -> dict:
    if not isinstance(payload, Mapping) or tuple(payload) != CATEGORY_ORDER:
        raise ValueError(
            "selected-only output must contain the four categories in order"
        )
    labels = list(schema["labels"])
    index = {tag: position for position, tag in enumerate(labels)}
    category_for_tag = {
        tag: category
        for category, subcategories in schema["semantic_categories"].items()
        for tags in subcategories.values()
        for tag in tags
    }
    subcategory_for_tag = {
        tag: (category, subcategory)
        for category, subcategories in schema["semantic_categories"].items()
        for subcategory, tags in subcategories.items()
        for tag in tags
    }
    seen: set[str] = set()
    seen_subcategories: set[tuple[str, str]] = set()
    normalized: dict[str, list[str]] = {}
    for category in CATEGORY_ORDER:
        tags = payload[category]
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError(
                f"selected-only category {category} must be a tag string array"
            )
        if tags != sorted(tags, key=index.__getitem__):
            raise ValueError(
                f"selected-only category {category} is not in schema order"
            )
        for tag in tags:
            if tag not in index or category_for_tag[tag] != category:
                raise ValueError(f"selected-only tag {tag!r} is in the wrong category")
            if tag == "假两件":
                raise ValueError("假两件 is unsupported and cannot be selected")
            if tag in seen:
                raise ValueError(f"selected-only tag {tag!r} is duplicated")
            subcategory = subcategory_for_tag[tag]
            if subcategory in seen_subcategories:
                raise ValueError(
                    f"selected-only subcategory {subcategory!r} has two tags"
                )
            seen.add(tag)
            seen_subcategories.add(subcategory)
        normalized[category] = list(tags)
    return normalized


def _validate_schema(path: Path) -> tuple[dict[str, Any], str]:
    file_sha256 = sha256_file(path)
    _require_equal(
        file_sha256,
        EXPECTED_SCHEMA_FILE_SHA256,
        name="schema file SHA256",
    )
    schema = load_unified57_schema(path)
    _require_equal(
        schema.get("schema_sha256"), EXPECTED_SCHEMA_SHA256, name="schema_sha256"
    )
    _require_equal(schema.get("num_labels"), 57, name="schema num_labels")
    _require_equal(
        tuple(schema.get("semantic_categories") or ()),
        CATEGORY_ORDER,
        name="category order",
    )
    if sum(len(value) for value in schema["semantic_categories"].values()) != 20:
        raise ValueError("schema must contain exactly 20 semantic subcategories")
    return schema, file_sha256


class _TrainableStateCarrier(nn.Module):
    """Tiny parameter carrier that lets the training loader validate/export state."""

    def __init__(self, state: Mapping[str, object]) -> None:
        super().__init__()
        self._external_names: list[str] = []
        self._slot_names: list[str] = []
        for index, (name, value) in enumerate(state.items()):
            if not isinstance(name, str) or not torch.is_tensor(value):
                raise ValueError("checkpoint model state must map names to tensors")
            if not value.is_floating_point():
                raise ValueError(
                    f"checkpoint trainable tensor {name} must be floating point"
                )
            slot = f"slot_{index:04d}"
            self.register_parameter(
                slot,
                nn.Parameter(torch.empty_like(value, device="cpu"), requires_grad=True),
            )
            self._external_names.append(name)
            self._slot_names.append(slot)

    def named_parameters(  # type: ignore[override]
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterable[tuple[str, nn.Parameter]]:
        del prefix, recurse, remove_duplicate
        for external_name, slot_name in zip(self._external_names, self._slot_names):
            yield external_name, getattr(self, slot_name)


def _load_checkpoint_state(
    checkpoint_path: Path,
    *,
    schema: Mapping[str, Any],
    expected_checkpoint_sha256: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], str]:
    checkpoint_sha256 = sha256_file(checkpoint_path)
    _require_equal(
        checkpoint_sha256,
        expected_checkpoint_sha256,
        name="checkpoint file SHA256",
    )
    try:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"unable to load v3 checkpoint: {checkpoint_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("v3 checkpoint must contain a dictionary")
    state = raw.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("v3 checkpoint model mapping is missing")

    _require_equal(raw.get("format_version"), 3, name="checkpoint format_version")
    _require_equal(raw.get("tag_order"), schema["labels"], name="checkpoint tag_order")
    _require_equal(
        raw.get("schema_sha256"),
        schema["schema_sha256"],
        name="checkpoint schema_sha256",
    )
    _require_sha256(raw.get("manifest_sha256"), name="checkpoint manifest_sha256")
    _require_equal(
        raw.get("mask_contract_version"),
        MASK_CONTRACT_VERSION,
        name="checkpoint mask_contract_version",
    )
    _require_equal(
        raw.get("pu_output_semantics"),
        PU_OUTPUT_SEMANTICS,
        name="checkpoint pu_output_semantics",
    )
    loss_contract = raw.get("loss_contract")
    if not isinstance(loss_contract, Mapping):
        raise ValueError("checkpoint loss_contract is required")
    _require_equal(
        loss_contract.get("version"),
        "unified57_pn_sample_mean_bce_pairwise_pu_v1",
        name="checkpoint loss_contract version",
    )
    if not loss_contract.get("pn") or not loss_contract.get("pu"):
        raise ValueError("checkpoint loss_contract must describe PN and PU objectives")
    if loss_contract.get("unknown_is_negative") not in (None, False):
        raise ValueError("checkpoint loss_contract cannot treat unknown as negative")
    if loss_contract.get("pu_output_semantics") not in (None, PU_OUTPUT_SEMANTICS):
        raise ValueError("checkpoint loss_contract PU output semantics drifted")
    initialization = raw.get("initialization_audit")
    if not isinstance(initialization, Mapping) or not initialization.get("mode"):
        raise ValueError("checkpoint initialization_audit with mode is required")
    run_contract = raw.get("run_contract")
    if not isinstance(run_contract, Mapping):
        raise ValueError("checkpoint run_contract is required")
    _require_equal(
        run_contract.get("image_max_pixels"),
        EXPECTED_IMAGE_MAX_PIXELS,
        name="checkpoint image_max_pixels",
    )
    _require_equal(run_contract.get("lora_rank"), 16, name="checkpoint lora_rank")
    _require_equal(run_contract.get("lora_alpha"), 32, name="checkpoint lora_alpha")
    _require_equal(
        run_contract.get("lora_dropout"), 0.05, name="checkpoint lora_dropout"
    )
    _require_equal(
        run_contract.get("head_dropout"), 0.1, name="checkpoint head_dropout"
    )
    _require_equal(run_contract.get("dtype"), "bfloat16", name="checkpoint dtype")
    _require_equal(
        run_contract.get("vision_prompt_sha256"),
        hashlib.sha256(MODEL_INPUT_PROMPT.encode("utf-8")).hexdigest(),
        name="checkpoint vision_prompt_sha256",
    )
    if not run_contract.get("base_model"):
        raise ValueError("checkpoint run_contract base_model is required")
    _require_sha256(
        run_contract.get("base_model_config_sha256"),
        name="checkpoint base_model_config_sha256",
    )

    carrier = _TrainableStateCarrier(state)
    loaded_payload = load_v3_model_state_for_inference(
        checkpoint_path,
        model=carrier,
        expected_tag_order=schema["labels"],
        expected_schema_sha256=schema["schema_sha256"],
    )
    loaded = {
        name: parameter.detach().cpu().contiguous().clone()
        for name, parameter in carrier.named_parameters()
    }
    lora = {name: value for name, value in loaded.items() if "lora_" in name}
    classifier = {
        name: loaded[name]
        for name in ("classifier.weight", "classifier.bias")
        if name in loaded
    }
    unexpected = set(loaded) - set(lora) - set(classifier)
    if unexpected:
        raise ValueError(
            "checkpoint contains unexpected trainable tensors: "
            + ", ".join(sorted(unexpected)[:5])
        )
    if len(lora) != EXPECTED_LORA_TENSORS:
        raise ValueError(
            f"checkpoint must contain exactly 288 LoRA tensors; found {len(lora)}"
        )
    if set(classifier) != {"classifier.weight", "classifier.bias"}:
        raise ValueError(
            "checkpoint must contain classifier.weight and classifier.bias"
        )
    weight = classifier["classifier.weight"]
    bias = classifier["classifier.bias"]
    if weight.ndim != 2 or weight.shape[0] != 57 or weight.shape[1] <= 0:
        raise ValueError("classifier.weight must have shape [57, hidden_size]")
    if tuple(bias.shape) != (57,):
        raise ValueError("classifier.bias must have shape [57]")
    _require_equal(
        set(loaded_payload.get("trainable_names") or []),
        set(loaded),
        name="loaded checkpoint trainable_names",
    )
    return dict(loaded_payload), {**lora, **classifier}, checkpoint_sha256


def _validate_thresholds(
    path: Path,
    *,
    schema: Mapping[str, Any],
    checkpoint_sha256: str,
) -> tuple[dict[str, Any], str, dict[str, float | None]]:
    payload = _load_json(path, name="thresholds")
    _require_equal(
        payload.get("schema_version"),
        schema["schema_version"],
        name="thresholds schema_version",
    )
    _require_equal(
        payload.get("schema_sha256"),
        schema["schema_sha256"],
        name="thresholds schema_sha256",
    )
    _require_equal(
        payload.get("checkpoint_sha256"),
        checkpoint_sha256,
        name="thresholds checkpoint_sha256",
    )
    _require_sha256(
        payload.get("validation_manifest_sha256"),
        name="thresholds validation_manifest_sha256",
    )
    _require_equal(payload.get("fallback_threshold"), 0.5, name="fallback_threshold")
    resolved = _threshold_map(payload, schema)
    labels = payload["labels"]
    for tag in schema["labels"]:
        item = labels[tag]
        if not isinstance(item, Mapping):
            raise ValueError(f"threshold entry for {tag} must be an object")
        mode = schema["label_training_modes"][tag]
        _require_equal(item.get("mode"), mode, name=f"threshold mode for {tag}")
        if mode == "unsupported":
            _require_equal(item.get("threshold"), None, name=f"threshold for {tag}")
            _require_equal(item.get("method"), "disabled", name=f"method for {tag}")
            _require_equal(
                item.get("status"), "disabled_unsupported", name=f"status for {tag}"
            )
        elif item.get("status") not in {"calibrated", "fallback_insufficient_support"}:
            raise ValueError(f"threshold status for {tag} is invalid")
    return payload, sha256_file(path), resolved


def _validate_predictions(
    path: Path,
    *,
    schema: Mapping[str, Any],
    checkpoint_sha256: str,
) -> tuple[list[dict[str, Any]], str]:
    rows = _load_jsonl(path, name="test predictions")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"test prediction row {index} lacks record_id")
        if record_id in seen:
            raise ValueError(f"duplicate prediction record_id: {record_id}")
        seen.add(record_id)
        _require_equal(
            row.get("schema_version"),
            schema["schema_version"],
            name=f"{record_id} schema_version",
        )
        _require_equal(
            row.get("schema_sha256"),
            schema["schema_sha256"],
            name=f"{record_id} schema_sha256",
        )
        if "checkpoint_sha256" in row:
            _require_equal(
                row["checkpoint_sha256"],
                checkpoint_sha256,
                name=f"{record_id} checkpoint_sha256",
            )
        if not isinstance(row.get("image_path"), str) or not row["image_path"]:
            raise ValueError(f"{record_id}: image_path is required")
        _require_sha256(row.get("image_sha256"), name=f"{record_id} image_sha256")
        if not isinstance(row.get("source"), str) or not row["source"]:
            raise ValueError(f"{record_id}: source is required")
        if not isinstance(row.get("sources"), list) or not all(
            isinstance(source, str) and source for source in row["sources"]
        ):
            raise ValueError(f"{record_id}: sources must be a non-empty string array")
        _score_vector(row.get("scores") or [], schema)
        vectors: dict[str, list[int | float]] = {}
        for field in ("labels", "known_mask", "pu_positive_mask"):
            value = row.get(field)
            if not isinstance(value, list) or len(value) != 57:
                raise ValueError(f"{record_id}: {field} must contain exactly 57 values")
            if any(item not in (0, 1, 0.0, 1.0, False, True) for item in value):
                raise ValueError(f"{record_id}: {field} must be binary")
            vectors[field] = value
        if any(
            bool(known) and bool(pu)
            for known, pu in zip(vectors["known_mask"], vectors["pu_positive_mask"])
        ):
            raise ValueError(f"{record_id}: known_mask overlaps pu_positive_mask")
    return rows, sha256_file(path)


def _validate_metrics(
    path: Path,
    *,
    schema: Mapping[str, Any],
    checkpoint_sha256: str,
    thresholds_sha256: str,
    predictions_sha256: str,
    validation_manifest_sha256: str,
) -> tuple[dict[str, Any], str]:
    payload = _load_json(path, name="evaluation metrics")
    _require_equal(payload.get("status"), "success", name="evaluation status")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("evaluation metrics provenance is required")
    expected = {
        "schema_sha256": schema["schema_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "thresholds_sha256": thresholds_sha256,
        "predictions_sha256": predictions_sha256,
        "validation_manifest_sha256": validation_manifest_sha256,
    }
    for key, value in expected.items():
        _require_equal(provenance.get(key), value, name=f"evaluation {key}")
    _require_sha256(
        provenance.get("test_manifest_sha256"), name="evaluation test_manifest_sha256"
    )
    output_quality = payload.get("output_quality")
    if (
        not isinstance(output_quality, Mapping)
        or output_quality.get("json_validity_rate") != 1.0
    ):
        raise ValueError("evaluation JSON validity must be exactly 100%")
    reproduction = payload.get("reproduction_32")
    if not isinstance(reproduction, Mapping):
        raise ValueError("evaluation reproduction_32 gate is required")
    gates = {
        "records": 32,
        "score_values": 1824,
        "probabilities_exact": True,
        "max_abs_score_delta": 0.0,
        "selected_outputs_exact": True,
    }
    for key, expected_value in gates.items():
        _require_equal(
            reproduction.get(key), expected_value, name=f"reproduction_32 {key}"
        )
    return payload, sha256_file(path)


def _validate_verification_references(
    directory: Path,
    *,
    schema: Mapping[str, Any],
    thresholds: Mapping[str, float | None],
    checkpoint_sha256: str,
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    paths = {name: directory / name for name in VERIFICATION_FILENAMES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError("missing verification32 references: " + ", ".join(missing))
    manifest = _load_jsonl(
        paths[VERIFICATION_FILENAMES[0]], name="verification32 manifest"
    )
    floats = _load_jsonl(
        paths[VERIFICATION_FILENAMES[1]], name="verification32 float32"
    )
    selected = _load_jsonl(
        paths[VERIFICATION_FILENAMES[2]], name="verification32 selected-only"
    )
    if len(manifest) != 32 or len(floats) != 32 or len(selected) != 32:
        raise ValueError(
            "all verification32 references must contain exactly 32 records"
        )
    ids = [row.get("record_id") for row in manifest]
    if any(not isinstance(record_id, str) or not record_id for record_id in ids):
        raise ValueError("verification32 manifest has an invalid record_id")
    if len(set(ids)) != 32:
        raise ValueError("verification32 manifest record_id values must be unique")
    if [row.get("record_id") for row in floats] != ids:
        raise ValueError("verification32 float32 order differs from its manifest")
    if [row.get("record_id") for row in selected] != ids:
        raise ValueError("verification32 selected-only order differs from its manifest")

    predictions_by_id = {row["record_id"]: row for row in predictions}
    for index, (manifest_row, float_row, selected_row) in enumerate(
        zip(manifest, floats, selected)
    ):
        record_id = ids[index]
        for field in (
            "test_manifest_index",
            "image_path",
            "image_sha256",
            "source",
            "sources",
            "selection_bucket",
        ):
            if field not in manifest_row:
                raise ValueError(f"{record_id}: verification manifest lacks {field}")
        if not isinstance(manifest_row["test_manifest_index"], int) or isinstance(
            manifest_row["test_manifest_index"], bool
        ):
            raise ValueError(f"{record_id}: test_manifest_index must be an integer")
        _require_sha256(manifest_row["image_sha256"], name=f"{record_id} image_sha256")
        if record_id not in predictions_by_id:
            raise ValueError(
                f"{record_id}: verification row is absent from predictions"
            )
        prediction = predictions_by_id[record_id]
        scores = _score_vector(float_row.get("scores") or [], schema)
        if scores != _score_vector(prediction.get("scores") or [], schema):
            raise ValueError(
                f"{record_id}: verification float32 differs from predictions"
            )
        for field in ("image_path", "image_sha256", "source", "sources"):
            _require_equal(
                float_row.get(field),
                manifest_row.get(field),
                name=f"{record_id} {field}",
            )
        for field in (
            "image_path",
            "image_sha256",
            "source",
            "sources",
            "labels",
            "known_mask",
            "pu_positive_mask",
        ):
            _require_equal(
                float_row.get(field),
                prediction.get(field),
                name=f"{record_id} frozen {field}",
            )
        _require_equal(
            float_row.get("schema_version"),
            schema["schema_version"],
            name=f"{record_id} schema_version",
        )
        _require_equal(
            float_row.get("schema_sha256"),
            schema["schema_sha256"],
            name=f"{record_id} schema_sha256",
        )
        if "checkpoint_sha256" in float_row:
            _require_equal(
                float_row["checkpoint_sha256"],
                checkpoint_sha256,
                name=f"{record_id} checkpoint_sha256",
            )
        output = _validate_selected_only(selected_row.get("output"), schema)
        expected_output = render_selected_only(scores, thresholds, schema)
        _require_equal(
            output, expected_output, name=f"{record_id} selected-only reference"
        )

    return {
        "records": 32,
        "score_values": 32 * 57,
        "paths": paths,
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
    }


def _requirements_text() -> str:
    return (
        """torch>=2.6\ntransformers>=4.57\npeft>=0.17\nsafetensors>=0.5\nPillow>=11\n"""
    )


def _readme_text(config: Mapping[str, Any]) -> str:
    base_model = config["base_model"]["identifier"]
    return f"""# Bosideng Unified57 lightweight delivery

This package contains 288 LoRA tensors, the 57-label classifier head, frozen
thresholds, inference code, and frozen verification32 references. The
Qwen3-VL base model is an external dependency and is not included.

## Required base model

`{base_model}`

Use base-model bytes whose `config.json` SHA-256 matches `model_config.json`.
The classifier input prompt stays fixed to the training prompt. `final_prompt.txt`
is the product taxonomy and output contract.

## Inference

```bash
python infer.py --base-model /path/to/Qwen3-VL-8B-Instruct --image image.jpg
```

The default `selected_only` mode emits a strict four-category JSON object with
tag-name strings. Additional modes are `selected_with_confidence`, `all_scores`,
and `verification_float32`.

- `selected_with_confidence` uses the same winners as `selected_only` and adds
  two-decimal confidence strings.
- `all_scores` emits all 57 tags in schema order with two-decimal strings;
  `假两件` is fixed to `0.00` because its training mode is unsupported.
- `verification_float32` exposes unrounded 57-head values only for reproduction.

The 20 PU-label values have `uncalibrated_confidence` semantics. The package
does not claim calibrated probabilities for those labels.

## Verification references

The three files under `verification/` reference external test images by path and
SHA-256. Reproduction requires the exact matching image bytes. `SHA256SUMS`
authenticates every package file other than itself.
"""


INFER_SOURCE = r'''#!/usr/bin/env python3
"""Unified57 strict, confidence, all-score, and verification inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
CATEGORY_ORDER = ("局部结构", "廓形", "工艺", "面辅料")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _threshold_map(payload, config):
    source = payload.get("labels", payload)
    labels = config["tag_order"]
    if list(source) != labels:
        raise ValueError("threshold order differs from model tag order")
    result = {}
    for tag in labels:
        item = source[tag]
        result[tag] = item.get("threshold") if isinstance(item, dict) else item
    return result


def _validated_scores(scores, config):
    if not isinstance(scores, (list, tuple)) or len(scores) != 57:
        raise ValueError("scores must contain exactly 57 values")
    result = []
    for value in scores:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("scores must be numeric")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("scores must be finite and within [0, 1]")
        result.append(value)
    if len(config["tag_order"]) != 57:
        raise ValueError("config must contain 57 tags")
    return result


def _winners(scores, config, thresholds):
    scores = _validated_scores(scores, config)
    threshold_map = _threshold_map(thresholds, config)
    labels = config["tag_order"]
    positions = {tag: index for index, tag in enumerate(labels)}
    result = {category: [] for category in CATEGORY_ORDER}
    semantic = config["semantic_categories"]
    if tuple(semantic) != CATEGORY_ORDER:
        raise ValueError("semantic category order drifted")
    for category in CATEGORY_ORDER:
        for tags in semantic[category].values():
            eligible = []
            for tag in tags:
                threshold = threshold_map[tag]
                index = positions[tag]
                if tag != "假两件" and threshold is not None and scores[index] >= float(threshold):
                    eligible.append((tag, scores[index]))
            if eligible:
                result[category].append(
                    max(eligible, key=lambda item: (item[1], -positions[item[0]]))
                )
        result[category].sort(key=lambda item: positions[item[0]])
    return result


def format_scores(scores, mode, config, thresholds):
    scores = _validated_scores(scores, config)
    if mode == "all_scores":
        return {
            "scores": {
                tag: ("0.00" if tag == "假两件" else f"{score:.2f}")
                for tag, score in zip(config["tag_order"], scores)
            }
        }
    if mode == "verification_float32":
        return {"scores": scores}
    winners = _winners(scores, config, thresholds)
    if mode == "selected_only":
        return {
            category: [tag for tag, _score in rows]
            for category, rows in winners.items()
        }
    if mode == "selected_with_confidence":
        return {
            category: [
                {"name": tag, "confidence": f"{score:.2f}"}
                for tag, score in rows
            ]
            for category, rows in winners.items()
        }
    raise ValueError(f"unsupported mode: {mode}")


def _verify_artifacts(config):
    expected = {
        "label_schema.json": config["schema"]["file_sha256"],
        "thresholds.json": config["thresholds"]["sha256"],
        "final_prompt.txt": config["product_prompt_sha256"],
        "lora_and_classifier.safetensors": config["weights"]["sha256"],
    }
    for name, expected_sha in expected.items():
        actual = sha256_file(PACKAGE_DIR / name)
        if actual != expected_sha:
            raise ValueError(f"{name} SHA256 mismatch")


def collect_images(inputs):
    paths = []
    for source in inputs:
        source = Path(source)
        if source.is_dir():
            paths.extend(
                path for path in sorted(source.rglob("*"))
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        elif source.is_file():
            paths.append(source)
        else:
            raise FileNotFoundError(source)
    if not paths:
        raise ValueError("no supported images found")
    return paths


def decode_resized_rgb(path, max_pixels):
    from PIL import Image

    with Image.open(path) as source:
        width, height = source.size
        if max_pixels > 0 and width * height > max_pixels:
            scale = math.sqrt(max_pixels / float(width * height))
            target = (max(1, int(width * scale)), max(1, int(height * scale)))
            if source.format in {"JPEG", "MPO"}:
                try:
                    source.draft("RGB", target)
                except (AttributeError, OSError):
                    pass
        else:
            target = (width, height)
        image = source.convert("RGB")
        if image.size != target:
            image = image.resize(target, Image.Resampling.LANCZOS)
        return image.copy()


def build_model(base_model, config, device):
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file
    from torch import nn
    from transformers import AutoModelForImageTextToText

    expected_base_config_sha = config["base_model"].get("config_sha256")
    if expected_base_config_sha:
        base_config_path = Path(base_model) / "config.json"
        if not base_config_path.is_file():
            raise ValueError(
                "a local base-model directory with config.json is required for SHA verification"
            )
        if sha256_file(base_config_path) != expected_base_config_sha:
            raise ValueError("base-model config.json SHA256 mismatch")
    backbone = AutoModelForImageTextToText.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    backbone.config.use_cache = False
    lora = config["lora"]
    backbone = get_peft_model(
        backbone,
        LoraConfig(
            r=int(lora["rank"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(lora["target_modules"]),
        ),
    )

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.dropout = nn.Dropout(float(config["classifier"]["dropout"]))
            self.classifier = nn.Linear(
                int(config["classifier"]["hidden_size"]),
                int(config["classifier"]["num_labels"]),
            )

        def forward(self, **inputs):
            attention_mask = inputs.get("attention_mask")
            inputs["output_hidden_states"] = True
            inputs["return_dict"] = True
            inputs["use_cache"] = False
            outputs = self.backbone(**inputs)
            hidden = outputs.hidden_states[-1]
            if attention_mask is None:
                pooled = hidden[:, -1, :]
            else:
                positions = torch.arange(
                    attention_mask.shape[1], device=attention_mask.device
                ).unsqueeze(0)
                last = positions.masked_fill(~attention_mask.bool(), 0).max(dim=1).values
                pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
            return self.classifier(self.dropout(pooled))

    model = Classifier()
    weights = load_file(
        str(PACKAGE_DIR / "lora_and_classifier.safetensors"), device="cpu"
    )
    expected_lora = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    }
    expected_classifier = {"classifier.weight", "classifier.bias"}
    lora_weights = {name: value for name, value in weights.items() if "lora_" in name}
    classifier_weights = {
        name: weights[name] for name in expected_classifier if name in weights
    }
    if len(weights) != 290:
        raise RuntimeError("delivery weights must contain exactly 290 tensors")
    if set(lora_weights) != expected_lora:
        raise RuntimeError("LoRA tensor names differ from the reconstructed model")
    if set(classifier_weights) != expected_classifier:
        raise RuntimeError("classifier tensor names differ from the reconstructed model")
    model.load_state_dict({**lora_weights, **classifier_weights}, strict=False)
    return model.to(device).eval()


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model")
    parser.add_argument("--image", action="append", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--mode",
        choices=(
            "selected_only",
            "selected_with_confidence",
            "all_scores",
            "verification_float32",
        ),
        default="selected_only",
    )
    parser.add_argument(
        "--scores-json",
        type=Path,
        help="Format a JSON 57-score vector without loading the base model.",
    )
    args = parser.parse_args(argv)
    if args.scores_json is None and (not args.base_model or not args.image):
        parser.error("--base-model and --image are required for model inference")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    config = load_json(PACKAGE_DIR / "model_config.json")
    thresholds = load_json(PACKAGE_DIR / "thresholds.json")
    _verify_artifacts(config)
    if args.scores_json is not None:
        raw = json.loads(args.scores_json.read_text(encoding="utf-8"))
        scores = raw.get("scores") if isinstance(raw, dict) else raw
        output = format_scores(scores, args.mode, config, thresholds)
        text = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    else:
        import torch
        from transformers import AutoProcessor

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        paths = collect_images(args.image)
        device = torch.device(args.device)
        processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
        model = build_model(args.base_model, config, device)
        rows = []
        for start in range(0, len(paths), args.batch_size):
            path_batch = paths[start : start + args.batch_size]
            images = [
                decode_resized_rgb(path, int(config["image_max_pixels"]))
                for path in path_batch
            ]
            inputs = processor(
                images=images,
                text=[config["model_input_prompt"]] * len(images),
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(**inputs)
            for path, scores in zip(path_batch, torch.sigmoid(logits.float()).cpu().tolist()):
                rows.append(
                    {
                        "image": str(path),
                        "output": format_scores(scores, args.mode, config, thresholds),
                    }
                )
        if len(rows) == 1:
            text = json.dumps(rows[0]["output"], ensure_ascii=False, indent=2) + "\n"
        else:
            text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


DECODE_TEST_SOURCE = r"""from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

from PIL import Image
from safetensors.torch import load_file


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_infer():
    spec = importlib.util.spec_from_file_location(
        "unified57_delivery_infer", PACKAGE_DIR / "infer.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decode_equivalence_at_336_square_budget(tmp_path):
    infer = _load_infer()
    config = json.loads((PACKAGE_DIR / "model_config.json").read_text(encoding="utf-8"))
    source_path = tmp_path / "source.png"
    source = Image.new("RGB", (1000, 500))
    source.putdata(
        [
            ((x * 7) % 256, (y * 11) % 256, ((x + y) * 13) % 256)
            for y in range(500)
            for x in range(1000)
        ]
    )
    source.save(source_path)
    max_pixels = int(config["image_max_pixels"])
    scale = math.sqrt(max_pixels / float(1000 * 500))
    target = (max(1, int(1000 * scale)), max(1, int(500 * scale)))
    expected = source.resize(target, Image.Resampling.LANCZOS)
    actual = infer.decode_resized_rgb(source_path, max_pixels)
    assert actual.size == target
    assert actual.tobytes() == expected.tobytes()


def test_single_safetensors_has_exact_trainable_contract():
    config = json.loads((PACKAGE_DIR / "model_config.json").read_text(encoding="utf-8"))
    weights = load_file(str(PACKAGE_DIR / "lora_and_classifier.safetensors"))
    assert len(weights) == 290
    assert len([name for name in weights if "lora_" in name]) == 288
    assert weights["classifier.weight"].shape == (
        57,
        int(config["classifier"]["hidden_size"]),
    )
    assert tuple(weights["classifier.bias"].shape) == (57,)
"""


def _write_checksums(package_dir: Path) -> None:
    files = sorted(
        path.relative_to(package_dir).as_posix()
        for path in package_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (package_dir / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(package_dir / relative)}  {relative}\n" for relative in files
        ),
        encoding="utf-8",
    )


def build_delivery_package(
    *,
    checkpoint_path: Path | str,
    schema_path: Path | str,
    thresholds_path: Path | str,
    metrics_path: Path | str,
    predictions_path: Path | str,
    verification_dir: Path | str,
    final_prompt_path: Path | str,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Validate all frozen inputs and atomically create the lightweight package."""
    checkpoint_path = Path(checkpoint_path)
    schema_path = Path(schema_path)
    thresholds_path = Path(thresholds_path)
    metrics_path = Path(metrics_path)
    predictions_path = Path(predictions_path)
    verification_dir = Path(verification_dir)
    final_prompt_path = Path(final_prompt_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    schema, schema_file_sha256 = _validate_schema(schema_path)
    prompt_sha256 = sha256_file(final_prompt_path)
    _require_equal(prompt_sha256, FINAL_PROMPT_SHA256, name="final prompt SHA256")
    threshold_preview = _load_json(thresholds_path, name="thresholds")
    declared_checkpoint_sha256 = _require_sha256(
        threshold_preview.get("checkpoint_sha256"),
        name="thresholds checkpoint_sha256",
    )
    checkpoint, state, checkpoint_sha256 = _load_checkpoint_state(
        checkpoint_path,
        schema=schema,
        expected_checkpoint_sha256=declared_checkpoint_sha256,
    )
    thresholds_payload, thresholds_sha256, threshold_values = _validate_thresholds(
        thresholds_path,
        schema=schema,
        checkpoint_sha256=checkpoint_sha256,
    )
    predictions, predictions_sha256 = _validate_predictions(
        predictions_path,
        schema=schema,
        checkpoint_sha256=checkpoint_sha256,
    )
    metrics, metrics_sha256 = _validate_metrics(
        metrics_path,
        schema=schema,
        checkpoint_sha256=checkpoint_sha256,
        thresholds_sha256=thresholds_sha256,
        predictions_sha256=predictions_sha256,
        validation_manifest_sha256=thresholds_payload["validation_manifest_sha256"],
    )
    verification = _validate_verification_references(
        verification_dir,
        schema=schema,
        thresholds=threshold_values,
        checkpoint_sha256=checkpoint_sha256,
        predictions=predictions,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        lora = {name: value for name, value in state.items() if "lora_" in name}
        classifier = {
            name: state[name] for name in ("classifier.weight", "classifier.bias")
        }
        save_file(
            {**lora, **classifier},
            str(temporary / "lora_and_classifier.safetensors"),
        )
        shutil.copyfile(schema_path, temporary / "label_schema.json")
        shutil.copyfile(thresholds_path, temporary / "thresholds.json")
        shutil.copyfile(final_prompt_path, temporary / "final_prompt.txt")
        (temporary / "infer.py").write_text(INFER_SOURCE, encoding="utf-8")
        os.chmod(temporary / "infer.py", 0o755)
        package_tests = temporary / "tests"
        package_tests.mkdir()
        (package_tests / "test_decode_equivalence.py").write_text(
            DECODE_TEST_SOURCE, encoding="utf-8"
        )
        (temporary / "requirements.txt").write_text(
            _requirements_text(), encoding="utf-8"
        )
        target_verification = temporary / "verification"
        target_verification.mkdir()
        for name, source in verification["paths"].items():
            shutil.copyfile(source, target_verification / name)

        run_contract = checkpoint["run_contract"]
        weights_sha256 = sha256_file(temporary / "lora_and_classifier.safetensors")
        config: dict[str, Any] = {
            "package_version": PACKAGE_VERSION,
            "base_model": {
                "identifier": run_contract["base_model"],
                "config_sha256": run_contract.get("base_model_config_sha256"),
                "processor_identifier": run_contract["base_model"],
                "included": False,
            },
            "checkpoint": {
                "format_version": 3,
                "sha256": checkpoint_sha256,
                "manifest_sha256": checkpoint["manifest_sha256"],
                "mask_contract_version": checkpoint["mask_contract_version"],
                "loss_contract": checkpoint["loss_contract"],
                "initialization_audit": checkpoint["initialization_audit"],
            },
            "schema": {
                "version": schema["schema_version"],
                "sha256": schema["schema_sha256"],
                "file_sha256": schema_file_sha256,
                "filename": "label_schema.json",
            },
            "tag_order": schema["labels"],
            "semantic_categories": schema["semantic_categories"],
            "label_training_modes": schema["label_training_modes"],
            "unsupported_labels": schema["unsupported_labels"],
            "pu_output_semantics": PU_OUTPUT_SEMANTICS,
            "image_max_pixels": EXPECTED_IMAGE_MAX_PIXELS,
            "inference_dtype": run_contract.get("dtype", "bfloat16"),
            "model_input_prompt": MODEL_INPUT_PROMPT,
            "model_input_prompt_sha256": hashlib.sha256(
                MODEL_INPUT_PROMPT.encode("utf-8")
            ).hexdigest(),
            "product_prompt_sha256": prompt_sha256,
            "weights": {
                "filename": "lora_and_classifier.safetensors",
                "sha256": weights_sha256,
                "tensor_count": 290,
            },
            "lora": {
                "weights_filename": "lora_and_classifier.safetensors",
                "tensor_count": len(lora),
                "rank": run_contract["lora_rank"],
                "alpha": run_contract["lora_alpha"],
                "dropout": run_contract.get("lora_dropout", 0.05),
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "task_type": "CAUSAL_LM",
            },
            "classifier": {
                "weights_filename": "lora_and_classifier.safetensors",
                "tensor_count": 2,
                "num_labels": 57,
                "hidden_size": int(classifier["classifier.weight"].shape[1]),
                "dropout": run_contract.get("head_dropout", 0.1),
            },
            "thresholds": {
                "filename": "thresholds.json",
                "sha256": thresholds_sha256,
                "validation_manifest_sha256": thresholds_payload[
                    "validation_manifest_sha256"
                ],
            },
            "evaluation": {
                "metrics_sha256": metrics_sha256,
                "predictions_sha256": predictions_sha256,
                "test_manifest_sha256": metrics["provenance"]["test_manifest_sha256"],
                "status": metrics["status"],
            },
            "output_modes": {
                "default": "selected_only",
                "supported": [
                    "selected_only",
                    "selected_with_confidence",
                    "all_scores",
                    "verification_float32",
                ],
            },
        }
        _write_json(temporary / "model_config.json", config)
        verification_payload = {
            "status": "success",
            "records": 32,
            "score_values": 1824,
            "probabilities_exact": True,
            "max_abs_score_delta": 0.0,
            "selected_outputs_exact": True,
            "provenance": {
                "checkpoint_sha256": checkpoint_sha256,
                "schema_sha256": schema["schema_sha256"],
                "schema_file_sha256": schema_file_sha256,
                "thresholds_sha256": thresholds_sha256,
                "final_prompt_sha256": prompt_sha256,
                "weights_sha256": weights_sha256,
                "metrics_sha256": metrics_sha256,
            },
            "references": {
                "predictions_sha256": predictions_sha256,
                **verification["sha256"],
            },
            "environment": metrics.get("environment", {}),
            "external_images_required": True,
        }
        _write_json(temporary / "VERIFICATION.json", verification_payload)
        (temporary / "README.md").write_text(_readme_text(config), encoding="utf-8")
        _write_checksums(temporary)
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "output_dir": str(output_dir),
        "checkpoint_sha256": checkpoint_sha256,
        "schema_sha256": schema["schema_sha256"],
        "thresholds_sha256": thresholds_sha256,
        "predictions_sha256": predictions_sha256,
        "lora_tensors": 288,
        "classifier_tensors": 2,
        "verification_records": 32,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--verification-dir", type=Path, required=True)
    parser.add_argument("--final-prompt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_delivery_package(
        checkpoint_path=args.checkpoint,
        schema_path=args.schema,
        thresholds_path=args.thresholds,
        metrics_path=args.metrics,
        predictions_path=args.predictions,
        verification_dir=args.verification_dir,
        final_prompt_path=args.final_prompt,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
