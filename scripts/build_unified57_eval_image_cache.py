#!/usr/bin/env python3
"""Build and validate a lossless local image cache for Unified57 evaluation.

The cache stores the exact RGB pixels produced by the training decoder at the
frozen 336-square pixel budget.  It never replaces source paths or source
SHA-256 values in prediction evidence; evaluation uses it only as a faster
pixel source.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import functools
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

CACHE_VERSION = "bosideng_unified57_eval_rgb_cache_v1"
DEFAULT_IMAGE_MAX_PIXELS = 336 * 336
EXPECTED_SPLIT_COUNTS = {"validation": 5444, "test": 5441}
TRAINER_PATH = Path(__file__).with_name("train_unified57_qwen3vl_multilabel.py")


@functools.lru_cache(maxsize=1)
def _trainer_source() -> tuple[str, ast.Module]:
    source = TRAINER_PATH.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(TRAINER_PATH))


@functools.lru_cache(maxsize=1)
def _training_decoder_source() -> str:
    source, tree = _trainer_source()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == (
            "_decode_resized_rgb"
        ):
            segment = ast.get_source_segment(source, node)
            if segment:
                return segment
    raise RuntimeError("training decoder source was not found")


@functools.lru_cache(maxsize=1)
def _training_decoder():
    tree = ast.parse(_training_decoder_source(), filename=str(TRAINER_PATH))
    namespace: dict[str, Any] = {"Image": Image, "math": math}
    exec(compile(tree, str(TRAINER_PATH), "exec"), namespace)
    return namespace["_decode_resized_rgb"]


def _decode_resized_rgb(source: Image.Image, image_max_pixels: int) -> Image.Image:
    """Execute the exact decoder function extracted from the frozen trainer."""

    return _training_decoder()(source, image_max_pixels)


@functools.lru_cache(maxsize=1)
def training_vision_prompt() -> str:
    _, tree = _trainer_source()
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "VISION_PROMPT" for target in targets):
            continue
        value = ast.literal_eval(node.value)
        if isinstance(value, str) and value:
            return value
    raise RuntimeError("training VISION_PROMPT source was not found")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@functools.lru_cache(maxsize=1)
def decoder_contract_sha256() -> str:
    source = _training_decoder_source().encode("utf-8")
    return hashlib.sha256(source).hexdigest()


def rgb_sha256(image: Image.Image) -> str:
    if image.mode != "RGB":
        raise ValueError("cached image contract requires RGB mode")
    return hashlib.sha256(image.tobytes()).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read {name}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _load_jsonl(path: Path, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{name} line {line_number} must be an object")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read {name}: {path}") from exc
    return rows


def _resolved_source_path(record: Mapping[str, Any], manifest_parent: Path) -> Path:
    value = record.get("image_path") or record.get("local_image_path")
    if not isinstance(value, str) or not value:
        raise ValueError(f"record {record.get('record_id')} has no local image path")
    path = Path(value)
    return path if path.is_absolute() else manifest_parent / path


def _entry_key(split: str, record_id: str) -> str:
    return hashlib.sha256(f"{split}\0{record_id}".encode("utf-8")).hexdigest()[:32]


def _validate_entry(
    row: Mapping[str, Any],
    *,
    record: Mapping[str, Any],
    split: str,
    manifest_parent: Path,
    image_max_pixels: int,
    verify_file: bool,
) -> dict[str, Any]:
    record_id = str(record.get("record_id") or "")
    expected_source = _resolved_source_path(record, manifest_parent)
    expected = {
        "version": CACHE_VERSION,
        "split": split,
        "record_id": record_id,
        "original_image_path": str(expected_source),
        "original_image_sha256": record.get("image_sha256"),
        "image_max_pixels": image_max_pixels,
        "decoder_contract_sha256": decoder_contract_sha256(),
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"cache entry {record_id} mismatch at {key}")
    cache_path = Path(str(row.get("cache_path") or ""))
    if not cache_path.is_file():
        raise ValueError(f"cache file missing for {record_id}")
    for key in ("original_width", "original_height", "cached_width", "cached_height"):
        if not isinstance(row.get(key), int) or int(row[key]) <= 0:
            raise ValueError(f"cache entry {record_id} has invalid {key}")
    for key in ("cache_file_size", "cache_file_mtime_ns"):
        if not isinstance(row.get(key), int) or int(row[key]) <= 0:
            raise ValueError(f"cache entry {record_id} has invalid {key}")
    if not isinstance(row.get("cache_file_sha256"), str) or not isinstance(
        row.get("rgb_sha256"), str
    ):
        raise ValueError(f"cache entry {record_id} lacks hashes")
    cache_stat = cache_path.stat()
    if cache_stat.st_size != row["cache_file_size"]:
        raise ValueError(f"cache file size mismatch for {record_id}")
    if cache_stat.st_mtime_ns != row["cache_file_mtime_ns"]:
        raise ValueError(f"cache file mtime mismatch for {record_id}")
    if verify_file:
        if sha256_file(cache_path) != row["cache_file_sha256"]:
            raise ValueError(f"cache file SHA256 mismatch for {record_id}")
        with Image.open(cache_path) as cached_source:
            cached = cached_source.convert("RGB")
        if list(cached.size) != [row["cached_width"], row["cached_height"]]:
            raise ValueError(f"cache dimensions mismatch for {record_id}")
        if rgb_sha256(cached) != row["rgb_sha256"]:
            raise ValueError(f"cache RGB mismatch for {record_id}")
    return dict(row)


def build_cache_entry(
    *,
    record: Mapping[str, Any],
    split: str,
    manifest_parent: Path,
    cache_root: Path,
    image_max_pixels: int,
) -> dict[str, Any]:
    record_id = str(record.get("record_id") or "")
    image_sha = record.get("image_sha256")
    if not record_id or not isinstance(image_sha, str) or len(image_sha) != 64:
        raise ValueError("cache source record requires record_id and image_sha256")
    key = _entry_key(split, record_id)
    entry_path = cache_root / "entries" / split / f"{key}.json"
    if entry_path.is_file():
        try:
            return _validate_entry(
                _load_json(entry_path, "cache entry"),
                record=record,
                split=split,
                manifest_parent=manifest_parent,
                image_max_pixels=image_max_pixels,
                verify_file=True,
            )
        except ValueError:
            entry_path.unlink(missing_ok=True)

    source_path = _resolved_source_path(record, manifest_parent)
    actual_source_sha = sha256_file(source_path)
    if actual_source_sha != image_sha:
        raise ValueError(f"source image SHA256 mismatch for {record_id}")
    with Image.open(source_path) as source:
        original_width, original_height = source.size
        decoded = _decode_resized_rgb(source, image_max_pixels)
    pixel_sha = rgb_sha256(decoded)
    cache_path = cache_root / "images" / split / f"{key}.ppm"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    try:
        decoded.save(temporary, format="PPM")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, cache_path)
    finally:
        temporary.unlink(missing_ok=True)
    with Image.open(cache_path) as cached_source:
        cached = cached_source.convert("RGB")
    if cached.size != decoded.size or cached.tobytes() != decoded.tobytes():
        cache_path.unlink(missing_ok=True)
        raise ValueError(f"lossless cache verification failed for {record_id}")
    cache_stat = cache_path.stat()
    row = {
        "version": CACHE_VERSION,
        "split": split,
        "record_id": record_id,
        "original_image_path": str(source_path),
        "original_image_sha256": image_sha,
        "original_width": int(original_width),
        "original_height": int(original_height),
        "cache_path": str(cache_path),
        "cache_file_sha256": sha256_file(cache_path),
        "cache_file_size": int(cache_stat.st_size),
        "cache_file_mtime_ns": int(cache_stat.st_mtime_ns),
        "cached_width": int(cached.width),
        "cached_height": int(cached.height),
        "rgb_sha256": pixel_sha,
        "image_max_pixels": int(image_max_pixels),
        "decoder_contract_sha256": decoder_contract_sha256(),
    }
    _atomic_json(entry_path, row)
    return row


def _worker(payload: tuple[dict[str, Any], str, str, str, int]) -> dict[str, Any]:
    record, split, manifest_parent, cache_root, image_max_pixels = payload
    return build_cache_entry(
        record=record,
        split=split,
        manifest_parent=Path(manifest_parent),
        cache_root=Path(cache_root),
        image_max_pixels=image_max_pixels,
    )


def _source_rows(
    validation_manifest: Path, test_manifest: Path
) -> list[tuple[dict[str, Any], str, Path]]:
    result: list[tuple[dict[str, Any], str, Path]] = []
    for split, path in (("validation", validation_manifest), ("test", test_manifest)):
        rows = _load_jsonl(path, f"{split} manifest")
        expected = EXPECTED_SPLIT_COUNTS[split]
        if len(rows) != expected:
            raise ValueError(f"{split} manifest must contain {expected} records")
        result.extend((row, split, path.parent) for row in rows)
    return result


def build_cache(
    *,
    validation_manifest: Path,
    test_manifest: Path,
    validation_manifest_sha256: str,
    test_manifest_sha256: str,
    cache_root: Path,
    image_max_pixels: int = DEFAULT_IMAGE_MAX_PIXELS,
    workers: int = 24,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    actual_validation_sha = sha256_file(validation_manifest)
    actual_test_sha = sha256_file(test_manifest)
    if actual_validation_sha != validation_manifest_sha256:
        raise ValueError("validation manifest SHA256 mismatch")
    if actual_test_sha != test_manifest_sha256:
        raise ValueError("test manifest SHA256 mismatch")
    sources = _source_rows(validation_manifest, test_manifest)
    cache_root.mkdir(parents=True, exist_ok=True)
    complete_path = cache_root / "complete.json"
    if complete_path.is_file():
        try:
            validate_cache(
                validation_manifest=validation_manifest,
                test_manifest=test_manifest,
                validation_manifest_sha256=validation_manifest_sha256,
                test_manifest_sha256=test_manifest_sha256,
                cache_root=cache_root,
                image_max_pixels=image_max_pixels,
            )
            return _load_json(complete_path, "cache complete marker")
        except (OSError, ValueError):
            pass
    # A builder crash must never leave a stale marker that makes an incomplete
    # cache look usable. Verified per-record sidecars remain as resume keys.
    complete_path.unlink(missing_ok=True)
    (cache_root / "runtime_validated.json").unlink(missing_ok=True)
    payloads = [
        (dict(record), split, str(parent), str(cache_root), image_max_pixels)
        for record, split, parent in sources
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(_worker, payloads, chunksize=4))
    manifest_path = cache_root / "cache_manifest.jsonl"
    _atomic_jsonl(manifest_path, rows)
    complete = {
        "version": CACHE_VERSION,
        "status": "complete",
        "record_count": len(rows),
        "split_counts": dict(EXPECTED_SPLIT_COUNTS),
        "validation_manifest_sha256": actual_validation_sha,
        "test_manifest_sha256": actual_test_sha,
        "image_max_pixels": image_max_pixels,
        "decoder_contract_sha256": decoder_contract_sha256(),
        "cache_manifest_sha256": sha256_file(manifest_path),
        "completed_at_unix": time.time(),
    }
    _atomic_json(cache_root / "complete.json", complete)
    return complete


def validate_cache(
    *,
    validation_manifest: Path,
    test_manifest: Path,
    validation_manifest_sha256: str,
    test_manifest_sha256: str,
    cache_root: Path,
    image_max_pixels: int = DEFAULT_IMAGE_MAX_PIXELS,
) -> dict[str, Any]:
    complete_path = cache_root / "complete.json"
    manifest_path = cache_root / "cache_manifest.jsonl"
    complete = _load_json(complete_path, "cache complete marker")
    expected_complete = {
        "version": CACHE_VERSION,
        "status": "complete",
        "record_count": sum(EXPECTED_SPLIT_COUNTS.values()),
        "split_counts": EXPECTED_SPLIT_COUNTS,
        "validation_manifest_sha256": validation_manifest_sha256,
        "test_manifest_sha256": test_manifest_sha256,
        "image_max_pixels": image_max_pixels,
        "decoder_contract_sha256": decoder_contract_sha256(),
    }
    for key, value in expected_complete.items():
        if complete.get(key) != value:
            raise ValueError(f"cache complete marker mismatch at {key}")
    if sha256_file(validation_manifest) != validation_manifest_sha256:
        raise ValueError("validation manifest SHA256 mismatch")
    if sha256_file(test_manifest) != test_manifest_sha256:
        raise ValueError("test manifest SHA256 mismatch")
    manifest_sha = sha256_file(manifest_path)
    if complete.get("cache_manifest_sha256") != manifest_sha:
        raise ValueError("cache manifest SHA256 mismatch")
    source_rows = _source_rows(validation_manifest, test_manifest)
    rows = _load_jsonl(manifest_path, "cache manifest")
    if len(rows) != len(source_rows):
        raise ValueError("cache manifest record count mismatch")
    seen: set[str] = set()
    for row, (record, split, parent) in zip(rows, source_rows):
        record_id = str(record["record_id"])
        if record_id in seen:
            raise ValueError(f"duplicate cache record_id {record_id}")
        seen.add(record_id)
        _validate_entry(
            row,
            record=record,
            split=split,
            manifest_parent=parent,
            image_max_pixels=image_max_pixels,
            verify_file=True,
        )
    validated = {
        "version": CACHE_VERSION,
        "status": "validated",
        "record_count": len(rows),
        "cache_manifest_sha256": manifest_sha,
        "complete_marker_sha256": sha256_file(complete_path),
        "image_max_pixels": image_max_pixels,
        "decoder_contract_sha256": decoder_contract_sha256(),
        "validated_at_unix": time.time(),
    }
    _atomic_json(cache_root / "runtime_validated.json", validated)
    return validated


def load_validated_cache_overlay(
    *,
    cache_root: Path,
    validation_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    validation_manifest_sha256: str,
    test_manifest_sha256: str,
    image_max_pixels: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    complete_path = cache_root / "complete.json"
    manifest_path = cache_root / "cache_manifest.jsonl"
    validated_path = cache_root / "runtime_validated.json"
    complete = _load_json(complete_path, "cache complete marker")
    validated = _load_json(validated_path, "cache runtime validation")
    expected = {
        "version": CACHE_VERSION,
        "status": "complete",
        "record_count": len(validation_rows) + len(test_rows),
        "validation_manifest_sha256": validation_manifest_sha256,
        "test_manifest_sha256": test_manifest_sha256,
        "image_max_pixels": image_max_pixels,
        "decoder_contract_sha256": decoder_contract_sha256(),
    }
    for key, value in expected.items():
        if complete.get(key) != value:
            raise ValueError(f"cache overlay complete marker mismatch at {key}")
    manifest_sha = sha256_file(manifest_path)
    if complete.get("cache_manifest_sha256") != manifest_sha:
        raise ValueError("cache overlay manifest SHA256 mismatch")
    if validated.get("status") != "validated" or validated.get(
        "cache_manifest_sha256"
    ) != manifest_sha:
        raise ValueError("cache overlay lacks current runtime validation")
    if validated.get("complete_marker_sha256") != sha256_file(complete_path):
        raise ValueError("cache overlay validation marker is stale")
    source_by_id: dict[str, tuple[Mapping[str, Any], str]] = {}
    for split, source_rows in (
        ("validation", validation_rows),
        ("test", test_rows),
    ):
        for source in source_rows:
            record_id = str(source["record_id"])
            if record_id in source_by_id:
                raise ValueError(f"duplicate source record_id {record_id}")
            source_by_id[record_id] = (source, split)
    rows = _load_jsonl(manifest_path, "cache manifest")
    if len(rows) != len(source_by_id):
        raise ValueError("cache overlay record count mismatch")
    overlay: dict[str, dict[str, Any]] = {}
    for row in rows:
        record_id = str(row.get("record_id") or "")
        source_item = source_by_id.get(record_id)
        if source_item is None or record_id in overlay:
            raise ValueError(f"cache overlay has unexpected record {record_id}")
        source, split = source_item
        expected_row = {
            "version": CACHE_VERSION,
            "split": split,
            "image_max_pixels": image_max_pixels,
            "decoder_contract_sha256": decoder_contract_sha256(),
        }
        for key, value in expected_row.items():
            if row.get(key) != value:
                raise ValueError(f"cache overlay row mismatch at {key} for {record_id}")
        if row.get("original_image_sha256") != source.get("image_sha256"):
            raise ValueError(f"cache overlay source SHA mismatch for {record_id}")
        cache_path = Path(str(row.get("cache_path") or ""))
        if not cache_path.is_file():
            raise ValueError(f"cache overlay file missing for {record_id}")
        cache_stat = cache_path.stat()
        if cache_stat.st_size != row.get("cache_file_size"):
            raise ValueError(f"cache overlay file size mismatch for {record_id}")
        if cache_stat.st_mtime_ns != row.get("cache_file_mtime_ns"):
            raise ValueError(f"cache overlay file mtime mismatch for {record_id}")
        overlay[record_id] = dict(row)
    return overlay, {
        "cache_manifest_sha256": manifest_sha,
        "complete_marker_sha256": sha256_file(complete_path),
        "record_count": len(overlay),
        "image_max_pixels": image_max_pixels,
        "decoder_contract_sha256": decoder_contract_sha256(),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest-sha256", required=True)
    parser.add_argument("--test-manifest-sha256", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--image-max-pixels", type=int, default=DEFAULT_IMAGE_MAX_PIXELS)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args(argv)
    if args.image_max_pixels <= 0 or args.workers <= 0:
        parser.error("image-max-pixels and workers must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    kwargs = {
        "validation_manifest": args.validation_manifest,
        "test_manifest": args.test_manifest,
        "validation_manifest_sha256": args.validation_manifest_sha256,
        "test_manifest_sha256": args.test_manifest_sha256,
        "cache_root": args.cache_root,
        "image_max_pixels": args.image_max_pixels,
    }
    result = (
        build_cache(**kwargs, workers=args.workers)
        if args.command == "build"
        else validate_cache(**kwargs)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
