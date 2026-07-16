#!/usr/bin/env python3
"""Build a leakage-safe unified57 PN/PU multilabel dataset.

The builder keeps three independent 57-dimensional tensors:

* ``labels`` stores positive targets (unknown values retain the conventional 0).
* ``known_mask`` selects trustworthy positive/negative (PN) BCE positions.
* ``pu_positive_mask`` selects positive-only dictionary observations for PU loss.

Exact SHA256 or exact 64-bit pHash identities merge supervision.  pHash
Hamming-distance components at distance <= 2 are used only as split atoms.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

try:
    from scripts.unified56_contract import (
        ALIASES,
        JD23_TAGS,
        UNIFIED56_TAGS,
        encode_jd_record,
    )
except ModuleNotFoundError:  # Support ``python scripts/build_...py``.
    try:
        from unified56_contract import (  # type: ignore[no-redef]
            ALIASES,
            JD23_TAGS,
            UNIFIED56_TAGS,
            encode_jd_record,
        )
    except ModuleNotFoundError:
        # The task is delivered as exactly three committed files.  Keep a
        # compact compatibility copy of the existing contract so the builder
        # remains executable when that reference module is absent.
        UNIFIED56_TAGS = (
            "连帽",
            "拆卸帽",
            "毛领",
            "立领",
            "翻领",
            "无领",
            "弯袖",
            "插肩袖",
            "正肩袖",
            "落肩袖",
            "防风袖口",
            "罗纹袖口",
            "袖袢",
            "前门襟",
            "双拉链",
            "暗门襟",
            "插袋",
            "贴袋",
            "立体袋",
            "假两件",
            "腰带",
            "袖标",
            "H型",
            "O型",
            "X型",
            "A型",
            "茧型",
            "箱型",
            "合体",
            "宽松",
            "长款",
            "中长款",
            "中款",
            "短款",
            "压胶充绒",
            "压胶袋盖",
            "压胶门襟",
            "平行绗线",
            "菱形绗线",
            "葫芦型绗线",
            "反光条",
            "帽口抽绳",
            "下摆抽绳",
            "腰部抽绳",
            "按扣",
            "魔术贴",
            "D字扣",
            "防水拉链",
            "树脂拉链",
            "织带",
            "光泽面料",
            "哑光面料",
            "自然光面料",
            "金属感面料",
            "硬壳面料",
            "软壳面料",
        )
        _JD23_FIELDS = {
            "长度分类": ("长款", "中款", "短款"),
            "主廓形": ("H型", "O型", "X型", "A型", "宽松"),
            "帽类结构": ("连帽", "毛领", "立领", "翻领", "无领"),
            "工艺标签": (
                "压胶充绒",
                "压胶袋盖",
                "压胶门襟",
                "平行绗线",
                "菱形绗线",
                "葫芦型绗线",
                "反光条",
            ),
            "闭合方式": ("按扣",),
            "其他结构": ("腰带",),
            "肩袖结构": ("插肩袖",),
        }
        JD23_TAGS = tuple(
            tag for tags in _JD23_FIELDS.values() for tag in tags
        )
        ALIASES = {
            "葫芦形绗线": "葫芦型绗线",
            "绗线反光条": "反光条",
            "纤线反光条": "反光条",
            "袖拌": "袖袢",
            "罗纹袖": "罗纹袖口",
        }

        def encode_jd_record(record):  # type: ignore[no-redef]
            selected = set()
            for field_name in _JD23_FIELDS:
                values = record.get(field_name, [])
                if isinstance(values, str):
                    values = [values]
                if not isinstance(values, list):
                    continue
                selected.update(
                    ALIASES.get(value.strip(), value.strip())
                    for value in values
                    if isinstance(value, str) and value.strip()
                )
            known = set(JD23_TAGS)
            return {
                "labels": [
                    1.0 if tag in selected else 0.0 for tag in UNIFIED56_TAGS
                ],
                "known_mask": [
                    1 if tag in known else 0 for tag in UNIFIED56_TAGS
                ],
            }


DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "bosideng_unified57_schema.json"
)
DEFAULT_SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLIT_RATIO_TOLERANCE = 0.02
PHASH_DISTANCE_THRESHOLD = 2

EXPECTED_SAFE_GROUPS = {
    "领型": ("立领", "翻领", "无领"),
    "肩部结构": ("插肩袖", "正肩袖", "落肩袖"),
    "外廓形": ("H型", "O型", "X型", "A型", "茧型", "箱型"),
    "松量": ("合体", "宽松"),
    "衣长": ("长款", "中长款", "中款", "短款"),
}

SLEEVE_AND_CUFF_TAGS = (
    "弯袖",
    "插肩袖",
    "正肩袖",
    "落肩袖",
    "防风袖口",
    "罗纹袖口",
    "袖袢",
)


@dataclass
class _EvidenceRow:
    key: str
    record_id: str
    source: str
    image_path: str
    sha256: str
    phash64: str
    pn_positive: set[str] = field(default_factory=set)
    pu_positive: set[str] = field(default_factory=set)
    pn_negative: set[str] = field(default_factory=set)


class _DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


def compute_schema_sha256(schema: Mapping[str, object]) -> str:
    """Hash canonical schema JSON while excluding its stored digest field."""
    payload = {key: value for key, value in schema.items() if key != "schema_sha256"}
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_schema(path: Path | str = DEFAULT_SCHEMA_PATH) -> dict:
    """Load and strictly validate the fixed unified57 schema contract."""
    schema_path = Path(path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if not isinstance(schema, dict):
        raise ValueError("schema must be a JSON object")
    expected_labels = [*UNIFIED56_TAGS, "无袖"]
    if schema.get("labels") != expected_labels:
        raise ValueError("schema labels must preserve canonical56 and append 无袖")
    if schema.get("num_labels") != 57 or len(set(expected_labels)) != 57:
        raise ValueError("schema must contain 57 unique labels")
    stored_hash = schema.get("schema_sha256")
    actual_hash = compute_schema_sha256(schema)
    if stored_hash != actual_hash:
        raise ValueError(
            f"schema SHA256 mismatch: stored={stored_hash!r}, computed={actual_hash}"
        )
    raw_groups = schema.get("safe_mutual_exclusion_groups")
    normalized_groups = {
        name: tuple(tags) for name, tags in (raw_groups or {}).items()
    }
    if normalized_groups != EXPECTED_SAFE_GROUPS:
        raise ValueError("schema safe mutual-exclusion groups differ from contract")
    modes = schema.get("label_training_modes")
    if not isinstance(modes, dict) or set(modes) != set(expected_labels):
        raise ValueError("schema label_training_modes must cover all 57 labels")
    if set(modes.values()) - {"pn", "pu", "unsupported"}:
        raise ValueError("schema contains an unsupported label training mode")
    mode_counts = Counter(modes.values())
    if mode_counts != {"pn": 36, "pu": 20, "unsupported": 1}:
        raise ValueError("schema must define exactly 36 PN, 20 PU, and 1 unsupported label")
    if schema.get("unsupported_labels") != ["假两件"]:
        raise ValueError("假两件 must remain the sole unsupported label")
    return schema


def _normalize_tag(tag: object, labels: set[str]) -> str:
    if not isinstance(tag, str) or not tag.strip():
        raise ValueError(f"tag must be a non-empty string: {tag!r}")
    normalized = ALIASES.get(tag.strip(), tag.strip())
    if normalized not in labels:
        raise ValueError(f"tag is outside unified57 schema: {normalized}")
    return normalized


def _normalize_sha256(value: object) -> str | None:
    if value is None or not str(value).strip():
        return None
    normalized = str(value).strip().lower()
    if len(normalized) != 64:
        raise ValueError(f"SHA256 must contain 64 hex characters: {value!r}")
    int(normalized, 16)
    return normalized


def _normalize_phash64(value: object) -> str | None:
    if value is None or not str(value).strip():
        return None
    if isinstance(value, int):
        if not 0 <= value < (1 << 64):
            raise ValueError(f"pHash integer is outside uint64: {value}")
        return f"{value:016x}"
    normalized = str(value).strip().lower()
    if normalized.startswith("0x"):
        normalized = normalized[2:]
    if len(normalized) > 16:
        raise ValueError(f"pHash must contain at most 16 hex characters: {value!r}")
    int(normalized, 16)
    return normalized.zfill(16)


def _value_from(record: Mapping[str, object], names: Sequence[str]) -> object | None:
    for name in names:
        value = record.get(name)
        if value is not None and str(value).strip():
            return value
    return None


def _record_id(record: Mapping[str, object]) -> str:
    value = _value_from(record, ("record_id", "id"))
    if value is None:
        raise ValueError("input row is missing record_id/id")
    return str(value).strip()


@lru_cache(maxsize=None)
def _file_sha256(path_string: str) -> str:
    digest = hashlib.sha256()
    with Path(path_string).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def _image_phash64(path_string: str) -> str:
    try:
        from scripts.enrich_jd_image_metadata import phash64
    except ModuleNotFoundError:
        from enrich_jd_image_metadata import phash64  # type: ignore[no-redef]
    return _normalize_phash64(phash64(Path(path_string))) or ""


@lru_cache(maxsize=None)
def _verify_image(path_string: str) -> None:
    from PIL import Image

    with Image.open(path_string) as image:
        image.verify()


def _identity(
    record: Mapping[str, object],
    *,
    root: Path | None,
    validate_images: bool,
) -> tuple[str, str, str]:
    path_value = _value_from(
        record,
        ("image_path", "local_path", "path", "relative_path", "image"),
    )
    if path_value is None:
        raise ValueError(f"{_record_id(record)} is missing an image path")
    image_path = Path(str(path_value))
    if not image_path.is_absolute() and root is not None:
        image_path = root / image_path
    image_path_string = str(image_path)
    sha256 = _normalize_sha256(
        _value_from(record, ("image_sha256", "sha256", "image_hash"))
    )
    phash64 = _normalize_phash64(
        _value_from(
            record,
            ("phash64", "image_phash", "phash", "perceptual_hash"),
        )
    )
    if validate_images:
        if not image_path.is_file():
            raise FileNotFoundError(f"image does not exist: {image_path}")
        _verify_image(image_path_string)
    if sha256 is None:
        if not image_path.is_file():
            raise ValueError(
                f"{_record_id(record)} needs SHA256 metadata or a readable image"
            )
        sha256 = _file_sha256(image_path_string)
    if phash64 is None:
        if not image_path.is_file():
            raise ValueError(
                f"{_record_id(record)} needs pHash metadata or a readable image"
            )
        phash64 = _image_phash64(image_path_string)
    return image_path_string, sha256, phash64


def _coerce_binary_vector(
    values: object,
    *,
    name: str,
    width: int,
    integer: bool,
) -> list[int] | list[float]:
    if not isinstance(values, list) or len(values) != width:
        raise ValueError(f"{name} must be a list with width {width}")
    result: list[int] | list[float]
    if integer:
        result = []
        for value in values:
            if value not in (0, 1, False, True):
                raise ValueError(f"{name} must contain only 0/1 values")
            result.append(int(value))
    else:
        result = []
        for value in values:
            if value not in (0, 1, 0.0, 1.0, False, True):
                raise ValueError(f"{name} must contain only binary values")
            result.append(float(value))
    return result


def _normalize_jd_row(
    record: Mapping[str, object],
    *,
    ordinal: int,
    schema: Mapping[str, object],
    validate_images: bool,
) -> _EvidenceRow:
    record_id = _record_id(record)
    labels_order = list(schema["labels"])
    labels_value = record.get("labels")
    mask_value = record.get("known_mask")
    if (labels_value is None) != (mask_value is None):
        raise ValueError(f"{record_id}: labels and known_mask must appear together")
    if labels_value is None:
        encoded = encode_jd_record(dict(record))
        labels = list(encoded["labels"]) + [0.0]
        known_mask = list(encoded["known_mask"]) + [0]
    else:
        if not isinstance(labels_value, list):
            raise ValueError(f"{record_id}: labels must be a list")
        width = len(labels_value)
        if width not in (56, 57):
            raise ValueError(f"{record_id}: JD vectors must contain 56 or 57 values")
        labels = list(
            _coerce_binary_vector(
                labels_value, name=f"{record_id}.labels", width=width, integer=False
            )
        )
        known_mask = list(
            _coerce_binary_vector(
                mask_value,
                name=f"{record_id}.known_mask",
                width=width,
                integer=True,
            )
        )
        if width == 56:
            labels.append(0.0)
            known_mask.append(0)
    expected_known = [int(tag in set(JD23_TAGS)) for tag in labels_order]
    if known_mask != expected_known:
        raise ValueError(f"{record_id}: JD known_mask must select exactly the JD23 tags")
    for index, known in enumerate(known_mask):
        if not known and labels[index] != 0.0:
            raise ValueError(f"{record_id}: unknown JD positions cannot carry labels")
    supplied_pu = record.get("pu_positive_mask")
    if supplied_pu is not None:
        pu_width = len(supplied_pu) if isinstance(supplied_pu, list) else -1
        pu_values = _coerce_binary_vector(
            supplied_pu,
            name=f"{record_id}.pu_positive_mask",
            width=pu_width,
            integer=True,
        )
        if pu_width == 56:
            pu_values.append(0)
        if len(pu_values) != 57 or any(pu_values):
            raise ValueError(f"{record_id}: JD rows cannot contain PU positives")
    image_path, sha256, phash64 = _identity(
        record, root=None, validate_images=validate_images
    )
    pn_positive = {
        tag
        for tag, label, known in zip(labels_order, labels, known_mask)
        if known and label == 1.0
    }
    pn_negative = {
        tag
        for tag, label, known in zip(labels_order, labels, known_mask)
        if known and label == 0.0
    }
    return _EvidenceRow(
        key=f"jd_complete23:{record_id}:{ordinal}",
        record_id=record_id,
        source="jd_complete23",
        image_path=image_path,
        sha256=sha256,
        phash64=phash64,
        pn_positive=pn_positive,
        pn_negative=pn_negative,
    )


def _as_tag_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, (str, int)):
        return [value]
    if isinstance(value, list):
        return value
    raise ValueError(f"tag collection must be a string or list: {value!r}")


def _validate_evidence_training_modes(
    record_id: str,
    schema: Mapping[str, object],
    *,
    pn_positive: set[str],
    pn_negative: set[str],
    pu_positive: set[str],
) -> None:
    modes = schema["label_training_modes"]
    for tag in sorted(pn_positive | pn_negative):
        mode = modes[tag]
        if mode != "pn":
            raise ValueError(
                f"{record_id}: {tag} has mode={mode} and cannot enter PN evidence"
            )
    for tag in sorted(pu_positive):
        mode = modes[tag]
        if mode != "pu":
            raise ValueError(
                f"{record_id}: {tag} has mode={mode} and cannot enter PU evidence"
            )


def _normalize_dictionary_row(
    record: Mapping[str, object],
    *,
    ordinal: int,
    schema: Mapping[str, object],
    root: Path | None,
    validate_images: bool,
) -> _EvidenceRow:
    record_id = _record_id(record)
    labels_order = list(schema["labels"])
    labels_set = set(labels_order)
    unsupported = set(schema["unsupported_labels"])
    pn_positive: set[str] = set()
    pu_positive: set[str] = set()
    pn_negative: set[str] = set()

    canonical_values: list[object] = []
    if record.get("canonical_tags") is not None:
        canonical_values.extend(_as_tag_list(record.get("canonical_tags")))
    elif record.get("canonical_tag") is not None:
        canonical_values.extend(_as_tag_list(record.get("canonical_tag")))
    elif record.get("label") is not None:
        canonical_values.extend(_as_tag_list(record.get("label")))
    elif record.get("remote_tag") is not None:
        canonical_values.extend(_as_tag_list(record.get("remote_tag")))

    safe_by_tag = {
        tag: tuple(group)
        for group in EXPECTED_SAFE_GROUPS.values()
        for tag in group
    }
    for raw_tag in canonical_values:
        tag = _normalize_tag(raw_tag, labels_set)
        if tag in unsupported:
            raise ValueError(f"{record_id}: unsupported label cannot be a positive: {tag}")
        mode = schema["label_training_modes"][tag]
        if mode == "pn":
            pn_positive.add(tag)
            if tag in safe_by_tag:
                pn_negative.update(set(safe_by_tag[tag]) - {tag})
        elif mode == "pu":
            pu_positive.add(tag)
        else:
            raise ValueError(f"{record_id}: unsupported label cannot be a positive: {tag}")

    explicit_labels = record.get("labels")
    explicit_known = record.get("known_mask")
    if (explicit_labels is None) != (explicit_known is None):
        raise ValueError(f"{record_id}: labels and known_mask must appear together")
    if explicit_labels is not None:
        if not isinstance(explicit_labels, list):
            raise ValueError(f"{record_id}: labels must be a list")
        width = len(explicit_labels)
        if width not in (56, 57):
            raise ValueError(f"{record_id}: dictionary vectors need width 56 or 57")
        vector_labels = list(
            _coerce_binary_vector(
                explicit_labels,
                name=f"{record_id}.labels",
                width=width,
                integer=False,
            )
        )
        vector_known = list(
            _coerce_binary_vector(
                explicit_known,
                name=f"{record_id}.known_mask",
                width=width,
                integer=True,
            )
        )
        if width == 56:
            vector_labels.append(0.0)
            vector_known.append(0)
        pn_positive.update(
            tag
            for tag, label, known in zip(labels_order, vector_labels, vector_known)
            if known and label == 1.0
        )
        pn_negative.update(
            tag
            for tag, label, known in zip(labels_order, vector_labels, vector_known)
            if known and label == 0.0
        )

    explicit_pu = record.get("pu_positive_mask")
    if explicit_pu is not None:
        if not isinstance(explicit_pu, list):
            raise ValueError(f"{record_id}: pu_positive_mask must be a list")
        width = len(explicit_pu)
        if width not in (56, 57):
            raise ValueError(f"{record_id}: PU vector needs width 56 or 57")
        vector_pu = list(
            _coerce_binary_vector(
                explicit_pu,
                name=f"{record_id}.pu_positive_mask",
                width=width,
                integer=True,
            )
        )
        if width == 56:
            vector_pu.append(0)
        pu_positive.update(
            tag for tag, selected in zip(labels_order, vector_pu) if selected
        )

    for raw_tag in _as_tag_list(record.get("pn_positive_tags")):
        pn_positive.add(_normalize_tag(raw_tag, labels_set))
    for raw_tag in _as_tag_list(record.get("pu_positive_tags")):
        pu_positive.add(_normalize_tag(raw_tag, labels_set))
    for raw_tag in _as_tag_list(record.get("pn_negative_tags")):
        pn_negative.add(_normalize_tag(raw_tag, labels_set))

    _validate_evidence_training_modes(
        record_id,
        schema,
        pn_positive=pn_positive,
        pn_negative=pn_negative,
        pu_positive=pu_positive,
    )
    if not (pn_positive or pu_positive or pn_negative):
        raise ValueError(f"{record_id}: dictionary row has no usable supervision")
    image_path, sha256, phash64 = _identity(
        record, root=root, validate_images=validate_images
    )
    return _EvidenceRow(
        key=f"dictionary_v4:{record_id}:{ordinal}",
        record_id=record_id,
        source="dictionary_v4",
        image_path=image_path,
        sha256=sha256,
        phash64=phash64,
        pn_positive=pn_positive,
        pu_positive=pu_positive,
        pn_negative=pn_negative,
    )


def _merge_identity_rows(
    rows: Sequence[Mapping[str, object]],
    identities: Sequence[Mapping[str, object]] | None,
) -> list[dict]:
    if identities is None:
        return [dict(row) for row in rows]
    identity_by_id: dict[str, Mapping[str, object]] = {}
    for identity in identities:
        record_id = _record_id(identity)
        if record_id in identity_by_id:
            raise ValueError(f"duplicate JD identity record_id: {record_id}")
        identity_by_id[record_id] = identity
    merged_rows = []
    for row in rows:
        record_id = _record_id(row)
        if record_id not in identity_by_id:
            raise ValueError(f"JD identity manifest is missing record_id: {record_id}")
        merged = dict(identity_by_id[record_id])
        for key, value in row.items():
            if value is not None and (not isinstance(value, str) or value.strip()):
                merged[key] = value
        merged_rows.append(merged)
    return merged_rows


def _group_exact_identities(rows: Sequence[_EvidenceRow]) -> list[list[_EvidenceRow]]:
    dsu = _DisjointSet(len(rows))
    sha_owner: dict[str, int] = {}
    phash_owner: dict[str, int] = {}
    for index, row in enumerate(rows):
        previous = sha_owner.setdefault(row.sha256, index)
        dsu.union(previous, index)
        previous = phash_owner.setdefault(row.phash64, index)
        dsu.union(previous, index)
    groups: dict[int, list[_EvidenceRow]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[dsu.find(index)].append(row)
    return [
        sorted(group, key=lambda item: item.key)
        for group in sorted(groups.values(), key=lambda items: min(row.key for row in items))
    ]


def _state_dict(labels: Sequence[str]) -> dict[str, tuple[float, int, int]]:
    return {tag: (0.0, 0, 0) for tag in labels}


def _aggregate_exact_group(
    members: Sequence[_EvidenceRow],
    *,
    schema: Mapping[str, object],
) -> tuple[dict, list[dict]]:
    label_order = list(schema["labels"])
    pn_positive = set().union(*(member.pn_positive for member in members))
    pu_positive = set().union(*(member.pu_positive for member in members))
    pn_negative = set().union(*(member.pn_negative for member in members))
    all_positive = pn_positive | pu_positive
    signature = "\n".join(sorted(member.key for member in members))
    visual_group_id = (
        "exact:"
        + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:20]
    )
    source_record_ids = [member.record_id for member in members]
    sources = sorted({member.source for member in members})
    conflicts: list[dict] = []
    conflicted_tags: set[str] = set()

    for group_name, group_tags in EXPECTED_SAFE_GROUPS.items():
        positive_tags = sorted(all_positive.intersection(group_tags))
        if len(positive_tags) >= 2:
            conflicted_tags.update(group_tags)
            conflicts.append(
                {
                    "kind": "mutually_exclusive_positive_conflict",
                    "visual_group_id": visual_group_id,
                    "group": group_name,
                    "label": "",
                    "positive_tags": positive_tags,
                    "source_record_ids": source_record_ids,
                    "detail": "safe-group multi-positive supervision downgraded to unknown",
                }
            )

    states = _state_dict(label_order)
    for tag in label_order:
        if tag in conflicted_tags:
            continue
        has_pn_positive = tag in pn_positive
        has_pu_positive = tag in pu_positive
        has_negative = tag in pn_negative
        if has_pn_positive:
            states[tag] = (1.0, 1, 0)
        elif has_pu_positive:
            states[tag] = (1.0, 0, 1)
        elif has_negative:
            states[tag] = (0.0, 1, 0)
        if (has_pn_positive or has_pu_positive) and has_negative:
            conflicts.append(
                {
                    "kind": "positive_overrode_negative",
                    "visual_group_id": visual_group_id,
                    "group": "",
                    "label": tag,
                    "positive_tags": [tag],
                    "source_record_ids": source_record_ids,
                    "detail": "credible positive retained for the same label",
                }
            )

    derived_negatives: dict[str, list[str]] = defaultdict(list)
    if states["连帽"] == (0.0, 1, 0):
        for tag in ("拆卸帽", "帽口抽绳"):
            derived_negatives[tag].append("known 连帽=0")
    if states["无袖"][0] == 1.0:
        for tag in SLEEVE_AND_CUFF_TAGS:
            derived_negatives[tag].append("无袖=1")
    if any(states[tag] == (1.0, 1, 0) for tag in SLEEVE_AND_CUFF_TAGS):
        derived_negatives["无袖"].append("known sleeve/cuff=1")

    for tag, reasons in sorted(derived_negatives.items()):
        if tag in conflicted_tags:
            continue
        if states[tag][0] == 1.0:
            conflicts.append(
                {
                    "kind": "positive_overrode_implication_negative",
                    "visual_group_id": visual_group_id,
                    "group": "",
                    "label": tag,
                    "positive_tags": [tag],
                    "source_record_ids": source_record_ids,
                    "detail": "; ".join(reasons),
                }
            )
            continue
        states[tag] = (0.0, 1, 0)

    for tag in schema["unsupported_labels"]:
        states[tag] = (0.0, 0, 0)

    representative = min(
        members,
        key=lambda member: (
            0 if member.source == "jd_complete23" else 1,
            member.record_id,
            member.image_path,
        ),
    )
    labels = [states[tag][0] for tag in label_order]
    known_mask = [states[tag][1] for tag in label_order]
    pu_positive_mask = [states[tag][2] for tag in label_order]
    if any(known and pu for known, pu in zip(known_mask, pu_positive_mask)):
        raise RuntimeError("known_mask and pu_positive_mask overlap")
    row = {
        "record_id": "unified57:"
        + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:24],
        "source": "+".join(sources),
        "sources": sources,
        "source_record_ids": source_record_ids,
        "binding_count": len(members),
        "dictionary_binding_count": sum(
            member.source == "dictionary_v4" for member in members
        ),
        "jd_binding_count": sum(
            member.source == "jd_complete23" for member in members
        ),
        "image_path": representative.image_path,
        "image_sha256": representative.sha256,
        "image_sha256s": sorted({member.sha256 for member in members}),
        "phash64": representative.phash64,
        "exact_phashes": sorted({member.phash64 for member in members}),
        "visual_group_id": visual_group_id,
        "labels": labels,
        "known_mask": known_mask,
        "pu_positive_mask": pu_positive_mask,
        "schema_version": schema["schema_version"],
        "schema_sha256": schema["schema_sha256"],
        "conflicted_tags": sorted(conflicted_tags),
    }
    return row, conflicts


def _phash_components(rows: list[dict]) -> dict:
    dsu = _DisjointSet(len(rows))
    owners: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        for phash in row["exact_phashes"]:
            owners[phash].append(index)
    for indexes in owners.values():
        for index in indexes[1:]:
            dsu.union(indexes[0], index)

    # Four exact bands guarantee a shared band for every pair with <=2 bit
    # differences.  Exact Hamming distance is checked before union.
    band_width = 16
    band_mask = (1 << band_width) - 1
    band_indexes: list[dict[int, list[str]]] = [defaultdict(list) for _ in range(4)]
    candidate_pairs = 0
    accepted_pairs = 0
    for phash in sorted(owners):
        value = int(phash, 16)
        candidates: set[str] = set()
        for band_index in range(4):
            band_value = (value >> (band_index * band_width)) & band_mask
            candidates.update(band_indexes[band_index][band_value])
        for other in sorted(candidates):
            candidate_pairs += 1
            if (value ^ int(other, 16)).bit_count() <= PHASH_DISTANCE_THRESHOLD:
                accepted_pairs += 1
                for left in owners[phash]:
                    for right in owners[other]:
                        dsu.union(left, right)
        for band_index in range(4):
            band_value = (value >> (band_index * band_width)) & band_mask
            band_indexes[band_index][band_value].append(phash)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[dsu.find(index)].append(index)
    for indexes in components.values():
        signature = "\n".join(sorted(rows[index]["visual_group_id"] for index in indexes))
        component_id = (
            "phash2:"
            + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:20]
        )
        for index in indexes:
            rows[index]["visual_component_id"] = component_id
            rows[index]["group_id"] = component_id
    return {
        "threshold": PHASH_DISTANCE_THRESHOLD,
        "unique_phashes": len(owners),
        "candidate_pairs": candidate_pairs,
        "accepted_pairs": accepted_pairs,
        "components": len(components),
    }


def _validate_split_ratios(split_ratios: Mapping[str, float]) -> dict[str, float]:
    if set(split_ratios) != {"train", "val", "test"}:
        raise ValueError("split_ratios must contain train, val, and test")
    normalized = {
        name: float(split_ratios[name]) for name in ("train", "val", "test")
    }
    if any(value <= 0.0 for value in normalized.values()):
        raise ValueError("split ratios must be positive")
    total = sum(normalized.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1.0")
    return normalized


def _integer_split_targets(
    total: int,
    ratios: Mapping[str, float],
) -> dict[str, int]:
    splits = tuple(ratios)
    fractional = {split: total * ratios[split] for split in splits}
    targets = {split: math.floor(fractional[split]) for split in splits}
    leftovers = total - sum(targets.values())
    split_order = {split: index for index, split in enumerate(splits)}
    ranked = sorted(
        splits,
        key=lambda split: (
            fractional[split] - math.floor(fractional[split]),
            ratios[split],
            -split_order[split],
        ),
        reverse=True,
    )
    for split in ranked[:leftovers]:
        targets[split] += 1
    if total >= len(splits):
        for missing_split in (split for split in splits if targets[split] == 0):
            donor = max(
                (split for split in splits if targets[split] > 1),
                key=lambda split: (
                    targets[split] - fractional[split],
                    targets[split],
                    ratios[split],
                    -split_order[split],
                ),
            )
            targets[donor] -= 1
            targets[missing_split] = 1
    return targets


def _assign_splits(
    rows: list[dict],
    *,
    label_order: Sequence[str],
    split_ratios: Mapping[str, float],
    seed: int,
) -> None:
    ratios = _validate_split_ratios(split_ratios)
    components: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        components[row["visual_component_id"]].append(row)
    if not rows:
        return
    component_labels = {
        component_id: {
            tag
            for row in component_rows
            for tag, value in zip(label_order, row["labels"])
            if value == 1.0
        }
        for component_id, component_rows in components.items()
    }
    label_component_totals = Counter(
        tag for labels in component_labels.values() for tag in labels
    )
    label_targets = {
        tag: _integer_split_targets(total, ratios)
        for tag, total in label_component_totals.items()
    }
    rng = random.Random(seed)
    tie_breakers = {component_id: rng.random() for component_id in sorted(components)}

    def component_rarity(item: tuple[str, list[dict]]) -> tuple:
        component_id, component_rows = item
        positives = component_labels[component_id]
        rarest = min(
            (label_component_totals[tag] for tag in positives),
            default=len(rows) + 1,
        )
        return (
            rarest,
            -len(positives),
            -len(component_rows),
            tie_breakers[component_id],
            component_id,
        )

    target_rows = {name: len(rows) * ratio for name, ratio in ratios.items()}
    max_component_size = max(
        len(component_rows) for component_rows in components.values()
    )
    allowed_row_error = max(
        math.ceil(len(rows) * SPLIT_RATIO_TOLERANCE),
        max_component_size,
    )
    lower_rows = {
        split: max(0, math.ceil(target_rows[split] - allowed_row_error))
        for split in ratios
    }
    upper_rows = {
        split: math.floor(target_rows[split] + allowed_row_error)
        for split in ratios
    }
    assigned_rows = Counter()
    assigned_label_components = {
        tag: Counter() for tag in label_component_totals
    }
    split_order = {"train": 0, "val": 1, "test": 2}
    for component_id, component_rows in sorted(
        components.items(), key=component_rarity
    ):
        positives = component_labels[component_id]
        component_size = len(component_rows)
        remaining_after_assignment = (
            len(rows) - sum(assigned_rows.values()) - component_size
        )

        def preserves_ratio_bounds(split: str) -> bool:
            if assigned_rows[split] + component_size > upper_rows[split]:
                return False
            required_for_lower_bounds = sum(
                max(
                    lower_rows[candidate]
                    - assigned_rows[candidate]
                    - (component_size if candidate == split else 0),
                    0,
                )
                for candidate in ratios
            )
            return required_for_lower_bounds <= remaining_after_assignment

        eligible_splits = [
            split
            for split in ratios
            if preserves_ratio_bounds(split)
        ]
        if not eligible_splits:
            raise RuntimeError(
                "no split can accept visual component within ratio tolerance: "
                + json.dumps(
                    {
                        "component_id": component_id,
                        "component_size": component_size,
                        "assigned_rows": dict(assigned_rows),
                        "lower_rows": lower_rows,
                        "upper_rows": upper_rows,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )

        def split_score(split: str) -> tuple:
            quota_need = sum(
                assigned_label_components[tag][split] < label_targets[tag][split]
                for tag in positives
            )
            label_deficit = sum(
                max(
                    label_targets[tag][split]
                    - assigned_label_components[tag][split],
                    0,
                )
                / max(label_targets[tag][split], 1)
                for tag in positives
                if label_targets[tag][split] > 0
            )
            row_deficit = (
                target_rows[split] - assigned_rows[split]
            ) / max(target_rows[split], 1.0)
            return (
                quota_need,
                label_deficit,
                row_deficit,
                ratios[split],
                -split_order[split],
            )

        owner = max(
            eligible_splits,
            key=split_score,
        )
        assigned_rows[owner] += component_size
        for tag in positives:
            assigned_label_components[tag][owner] += 1
        for row in component_rows:
            row["split"] = owner

    missing_support = {
        tag: [
            split
            for split in ratios
            if assigned_label_components[tag][split] == 0
        ]
        for tag, total in label_component_totals.items()
        if total >= len(ratios)
        and any(assigned_label_components[tag][split] == 0 for split in ratios)
    }
    if missing_support:
        raise RuntimeError(
            "split stratification left evaluable labels without positive support: "
            + json.dumps(missing_support, ensure_ascii=False, sort_keys=True)
        )

    ratio_errors = {
        split: {
            "actual": assigned_rows[split],
            "target": target_rows[split],
        }
        for split in ratios
        if abs(assigned_rows[split] - target_rows[split]) > allowed_row_error
    }
    if ratio_errors:
        raise RuntimeError(
            "component-level split ratios exceed tolerance: "
            + json.dumps(ratio_errors, ensure_ascii=False, sort_keys=True)
        )


def _leakage_check(rows: Sequence[dict]) -> dict:
    def crossings(field: str, *, many: bool) -> list[dict]:
        owners: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            values = row[field] if many else [row[field]]
            for value in values:
                owners[str(value)].add(row["split"])
        return [
            {field: value, "splits": sorted(splits)}
            for value, splits in sorted(owners.items())
            if len(splits) > 1
        ]

    component_crossings = crossings("visual_component_id", many=False)
    sha_crossings = crossings("image_sha256s", many=True)
    phash_crossings = crossings("exact_phashes", many=True)
    return {
        "passed": not (component_crossings or sha_crossings or phash_crossings),
        "pHash_hamming_threshold": PHASH_DISTANCE_THRESHOLD,
        "cross_split_components": component_crossings,
        "cross_split_sha256": sha_crossings,
        "cross_split_exact_phash": phash_crossings,
    }


def _summary(
    rows: Sequence[dict],
    *,
    schema: Mapping[str, object],
    jd_input_rows: int,
    dictionary_input_rows: int,
    exact_input_rows: int,
    phash_statistics: Mapping[str, object],
    conflicts: Sequence[dict],
    leakage: Mapping[str, object],
    seed: int,
) -> dict:
    per_label = {}
    supervision_fields = ("pn_positive", "pn_negative", "pu_positive", "unknown")
    for index, tag in enumerate(schema["labels"]):
        counts = Counter()
        split_counts = {
            split: Counter() for split in ("train", "val", "test")
        }
        positive_components_by_split = {
            split: set() for split in ("train", "val", "test")
        }
        for row in rows:
            label = row["labels"][index]
            known = row["known_mask"][index]
            pu = row["pu_positive_mask"][index]
            if known:
                field = "pn_positive" if label == 1.0 else "pn_negative"
            elif pu:
                if label != 1.0:
                    raise RuntimeError("PU mask selected a non-positive label")
                field = "pu_positive"
            else:
                field = "unknown"
            counts[field] += 1
            split_counts[row["split"]][field] += 1
            if label == 1.0:
                positive_components_by_split[row["split"]].add(
                    row["visual_component_id"]
                )
        component_counts = {
            split: len(component_ids)
            for split, component_ids in positive_components_by_split.items()
        }
        per_label[tag] = {
            "training_mode": schema["label_training_modes"][tag],
            **{field: counts[field] for field in supervision_fields},
            "by_split": {
                split: {
                    field: split_counts[split][field]
                    for field in supervision_fields
                }
                for split in ("train", "val", "test")
            },
            "positive_components": sum(component_counts.values()),
            "positive_components_by_split": component_counts,
        }
    return {
        "schema_version": schema["schema_version"],
        "schema_sha256": schema["schema_sha256"],
        "num_labels": 57,
        "seed": seed,
        "inputs": {
            "jd_rows": jd_input_rows,
            "dictionary_rows": dictionary_input_rows,
            "total_evidence_rows": exact_input_rows,
        },
        "output_records": len(rows),
        "splits": dict(sorted(Counter(row["split"] for row in rows).items())),
        "sources": dict(sorted(Counter(row["source"] for row in rows).items())),
        "exact_identity_groups": len(rows),
        "exact_rows_merged": exact_input_rows - len(rows),
        "phash_components": dict(phash_statistics),
        "conflicts": dict(sorted(Counter(row["kind"] for row in conflicts).items())),
        "leakage_passed": bool(leakage["passed"]),
        "per_label": per_label,
        "supervision_contract": {
            "pn": "known_mask=1",
            "pu_positive": "known_mask=0 and pu_positive_mask=1",
            "unknown": "known_mask=0 and pu_positive_mask=0",
            "unknown_is_never_negative": True,
        },
    }


def build_dataset(
    jd_records: Sequence[Mapping[str, object]],
    dictionary_records: Sequence[Mapping[str, object]],
    *,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
    dictionary_root: Path | str | None = None,
    identity_records: Sequence[Mapping[str, object]] | None = None,
    split_ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
    seed: int = 20260717,
    validate_images: bool = False,
) -> dict:
    """Build records and audits from encoded/raw JD plus dictionary evidence."""
    schema = load_schema(schema_path)
    root = Path(dictionary_root) if dictionary_root is not None else None
    merged_jd = _merge_identity_rows(jd_records, identity_records)
    evidence: list[_EvidenceRow] = []
    seen_keys: set[tuple[str, str]] = set()
    for ordinal, row in enumerate(merged_jd):
        record_id = _record_id(row)
        key = ("jd_complete23", record_id)
        if key in seen_keys:
            raise ValueError(f"duplicate JD record_id: {record_id}")
        seen_keys.add(key)
        evidence.append(
            _normalize_jd_row(
                row,
                ordinal=ordinal,
                schema=schema,
                validate_images=validate_images,
            )
        )
    for ordinal, row in enumerate(dictionary_records):
        record_id = _record_id(row)
        key = ("dictionary_v4", record_id)
        if key in seen_keys:
            raise ValueError(f"duplicate dictionary record_id: {record_id}")
        seen_keys.add(key)
        evidence.append(
            _normalize_dictionary_row(
                row,
                ordinal=ordinal,
                schema=schema,
                root=root,
                validate_images=validate_images,
            )
        )
    exact_groups = _group_exact_identities(evidence)
    output_rows: list[dict] = []
    conflicts: list[dict] = []
    for members in exact_groups:
        row, group_conflicts = _aggregate_exact_group(members, schema=schema)
        output_rows.append(row)
        conflicts.extend(group_conflicts)
    phash_statistics = _phash_components(output_rows)
    _assign_splits(
        output_rows,
        label_order=schema["labels"],
        split_ratios=split_ratios,
        seed=seed,
    )
    output_rows.sort(key=lambda row: row["record_id"])
    leakage = _leakage_check(output_rows)
    if not leakage["passed"]:
        raise RuntimeError("visual leakage detected after component-level splitting")
    summary = _summary(
        output_rows,
        schema=schema,
        jd_input_rows=len(jd_records),
        dictionary_input_rows=len(dictionary_records),
        exact_input_rows=len(evidence),
        phash_statistics=phash_statistics,
        conflicts=conflicts,
        leakage=leakage,
        seed=seed,
    )
    return {
        "records": output_rows,
        "conflicts": sorted(
            conflicts,
            key=lambda row: (
                row["visual_group_id"],
                row["kind"],
                row.get("group", ""),
                row.get("label", ""),
            ),
        ),
        "leakage_check": leakage,
        "summary": summary,
    }


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            rows.append(row)
    return rows


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_outputs(result: Mapping[str, object], output_dir: Path | str) -> None:
    """Write split JSONL plus summary/conflict/leakage audit artifacts."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    records = list(result["records"])
    for split in ("train", "val", "test"):
        with (destination / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in records:
                if row["split"] == split:
                    handle.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
    _write_json(destination / "dataset_summary.json", result["summary"])
    _write_json(destination / "leakage_check.json", result["leakage_check"])
    conflict_fields = (
        "kind",
        "visual_group_id",
        "group",
        "label",
        "positive_tags",
        "source_record_ids",
        "detail",
    )
    with (destination / "conflicts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=conflict_fields)
        writer.writeheader()
        for conflict in result["conflicts"]:
            writer.writerow(
                {
                    field: json.dumps(conflict.get(field), ensure_ascii=False)
                    if isinstance(conflict.get(field), list)
                    else conflict.get(field, "")
                    for field in conflict_fields
                }
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build unified57 masked PN/PU train/val/test JSONL files."
    )
    jd_group = parser.add_mutually_exclusive_group(required=True)
    jd_group.add_argument(
        "--jd-enriched",
        type=Path,
        help="Raw enriched JD23 JSONL with aggregate fields, SHA256, and phash64.",
    )
    jd_group.add_argument(
        "--jd-manifest",
        type=Path,
        help="Pre-encoded 56/57-dimensional JD labels and known_mask JSONL.",
    )
    parser.add_argument(
        "--jd-identity-manifest",
        type=Path,
        help="Optional record_id join supplying image_path/SHA256/phash64 to --jd-manifest.",
    )
    parser.add_argument("--dictionary-manifest", type=Path, required=True)
    parser.add_argument("--dictionary-root", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--validate-images",
        action="store_true",
        help="Open and verify every image in addition to identity metadata checks.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.jd_enriched is not None and args.jd_identity_manifest is not None:
        raise ValueError("--jd-identity-manifest is only valid with --jd-manifest")
    jd_path = args.jd_enriched or args.jd_manifest
    jd_records = _read_jsonl(jd_path)
    dictionary_records = _read_jsonl(args.dictionary_manifest)
    identity_records = (
        _read_jsonl(args.jd_identity_manifest)
        if args.jd_identity_manifest is not None
        else None
    )
    result = build_dataset(
        jd_records,
        dictionary_records,
        schema_path=args.schema,
        dictionary_root=args.dictionary_root,
        identity_records=identity_records,
        seed=args.seed,
        validate_images=args.validate_images,
    )
    write_outputs(result, args.output_dir)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
