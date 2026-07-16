from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

from scripts.build_unified57_eval_image_cache import (
    CACHE_VERSION,
    _decode_resized_rgb,
    _atomic_json,
    _atomic_jsonl,
    _validate_entry,
    build_cache_entry,
    decoder_contract_sha256,
    load_validated_cache_overlay,
    rgb_sha256,
    sha256_file,
    training_vision_prompt,
)
from scripts.evaluate_unified57_multilabel import EvaluationCollator
from scripts import build_unified57_eval_image_cache as cache_module


def _record(path: Path, record_id: str) -> dict:
    return {
        "record_id": record_id,
        "image_path": str(path),
        "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize("suffix", [".jpg", ".png"])
def test_cache_entry_is_bit_exact_to_training_decoder(
    tmp_path: Path, suffix: str
) -> None:
    source = tmp_path / f"odd{suffix}"
    image = Image.new("RGB", (1001, 503))
    image.putdata(
        [
            ((x * 7) % 256, (y * 11) % 256, ((x + y) * 13) % 256)
            for y in range(503)
            for x in range(1001)
        ]
    )
    save_kwargs = {"quality": 93} if suffix == ".jpg" else {}
    image.save(source, **save_kwargs)
    record = _record(source, f"record-{suffix[1:]}")

    row = build_cache_entry(
        record=record,
        split="validation",
        manifest_parent=tmp_path,
        cache_root=tmp_path / "cache",
        image_max_pixels=336 * 336,
    )

    with Image.open(source) as opened:
        expected = _decode_resized_rgb(opened, 336 * 336)
    with Image.open(row["cache_path"]) as opened:
        actual = opened.convert("RGB")
    assert actual.size == expected.size
    assert actual.tobytes() == expected.tobytes()
    assert row["rgb_sha256"] == rgb_sha256(expected)
    assert row["original_width"] == 1001
    assert row["original_height"] == 503


def test_cache_entry_resume_rejects_corrupt_cached_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (32, 17), (10, 20, 30)).save(source)
    record = _record(source, "record-1")
    row = build_cache_entry(
        record=record,
        split="test",
        manifest_parent=tmp_path,
        cache_root=tmp_path / "cache",
        image_max_pixels=336 * 336,
    )
    cache_path = Path(row["cache_path"])
    cache_path.write_bytes(cache_path.read_bytes() + b"corrupt")

    rebuilt = build_cache_entry(
        record=record,
        split="test",
        manifest_parent=tmp_path,
        cache_root=tmp_path / "cache",
        image_max_pixels=336 * 336,
    )

    assert sha256_file(rebuilt["cache_path"]) == rebuilt["cache_file_sha256"]
    with Image.open(rebuilt["cache_path"]) as opened:
        rebuilt_image = opened.convert("RGB")
    with Image.open(source) as opened:
        expected = _decode_resized_rgb(opened, 336 * 336)
    assert rebuilt_image.tobytes() == expected.tobytes()
    assert rebuilt["cache_file_sha256"] == row["cache_file_sha256"]


def test_validated_overlay_preserves_original_evidence_and_dimensions(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    records = []
    rows = []
    for split, color in (("validation", (1, 2, 3)), ("test", (4, 5, 6))):
        source = tmp_path / f"{split}.png"
        Image.new("RGB", (77, 41), color).save(source)
        record = _record(source, f"record-{split}")
        records.append(record)
        rows.append(
            build_cache_entry(
                record=record,
                split=split,
                manifest_parent=tmp_path,
                cache_root=cache_root,
                image_max_pixels=336 * 336,
            )
        )
    manifest = cache_root / "cache_manifest.jsonl"
    _atomic_jsonl(manifest, rows)
    complete = {
        "version": CACHE_VERSION,
        "status": "complete",
        "record_count": 2,
        "validation_manifest_sha256": "a" * 64,
        "test_manifest_sha256": "b" * 64,
        "image_max_pixels": 336 * 336,
        "decoder_contract_sha256": decoder_contract_sha256(),
        "cache_manifest_sha256": sha256_file(manifest),
    }
    _atomic_json(cache_root / "complete.json", complete)
    _atomic_json(
        cache_root / "runtime_validated.json",
        {
            "version": CACHE_VERSION,
            "status": "validated",
            "cache_manifest_sha256": sha256_file(manifest),
            "complete_marker_sha256": sha256_file(cache_root / "complete.json"),
        },
    )

    overlay, provenance = load_validated_cache_overlay(
        cache_root=cache_root,
        validation_rows=[records[0]],
        test_rows=[records[1]],
        validation_manifest_sha256="a" * 64,
        test_manifest_sha256="b" * 64,
        image_max_pixels=336 * 336,
    )

    for record in records:
        cached = overlay[record["record_id"]]
        assert cached["original_image_path"] == record["image_path"]
        assert cached["original_image_sha256"] == record["image_sha256"]
        assert cached["original_width"] == 77
        assert cached["original_height"] == 41
    assert provenance["record_count"] == 2

    cache_path = Path(rows[0]["cache_path"])
    payload = cache_path.read_bytes()
    cache_path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    stat = cache_path.stat()
    os.utime(
        cache_path,
        ns=(stat.st_atime_ns, int(rows[0]["cache_file_mtime_ns"]) + 1),
    )
    with pytest.raises(ValueError, match="mtime mismatch"):
        load_validated_cache_overlay(
            cache_root=cache_root,
            validation_rows=[records[0]],
            test_rows=[records[1]],
            validation_manifest_sha256="a" * 64,
            test_manifest_sha256="b" * 64,
            image_max_pixels=336 * 336,
        )


def test_entry_contract_rejects_decoder_or_source_drift(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (11, 13), (7, 8, 9)).save(source)
    record = _record(source, "record-drift")
    row = build_cache_entry(
        record=record,
        split="validation",
        manifest_parent=tmp_path,
        cache_root=tmp_path / "cache",
        image_max_pixels=336 * 336,
    )
    drifted = dict(row)
    drifted["decoder_contract_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="decoder_contract_sha256"):
        _validate_entry(
            drifted,
            record=record,
            split="validation",
            manifest_parent=tmp_path,
            image_max_pixels=336 * 336,
            verify_file=False,
        )
    changed_record = dict(record, image_sha256="f" * 64)
    with pytest.raises(ValueError, match="original_image_sha256"):
        _validate_entry(
            row,
            record=changed_record,
            split="validation",
            manifest_parent=tmp_path,
            image_max_pixels=336 * 336,
            verify_file=False,
        )


def test_collator_uses_cache_without_opening_original_source(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (73, 41), (12, 34, 56)).save(source)
    record = _record(source, "record-cached-collator")
    row = build_cache_entry(
        record=record,
        split="validation",
        manifest_parent=tmp_path,
        cache_root=tmp_path / "cache",
        image_max_pixels=336 * 336,
    )
    source.unlink()

    class Processor:
        def __call__(self, *, images, text, padding, return_tensors):
            assert len(images) == len(text) == 1
            assert images[0].mode == "RGB"
            assert padding is True
            assert return_tensors == "pt"
            return {"pixel_bytes": images[0].tobytes()}

    output = EvaluationCollator(
        Processor(),
        tmp_path,
        336 * 336,
        image_cache={record["record_id"]: row},
        split="validation",
    )([record])
    metadata = output["metadata"][0]
    assert metadata["image_path"] == str(source)
    assert metadata["image_sha256"] == record["image_sha256"]
    assert metadata["width"] == 73
    assert metadata["height"] == 41
    assert "cache_path" not in metadata


def test_launcher_validates_complete_cache_before_torchrun() -> None:
    launcher = Path("scripts/launch_unified57_eval_node1.sh").read_text(
        encoding="utf-8"
    )
    validation = launcher.index("build_unified57_eval_image_cache.py\" validate")
    torchrun = launcher.index("-m torch.distributed.run")
    assert validation < torchrun
    assert "--image-cache-root" in launcher


def test_decoder_and_prompt_are_extracted_from_frozen_trainer_source() -> None:
    trainer = Path("scripts/train_unified57_qwen3vl_multilabel.py").read_text(
        encoding="utf-8"
    )
    assert "def _decode_resized_rgb" in trainer
    assert training_vision_prompt() == (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "识别图中服装的可见结构、工艺和面辅料属性。<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_full_builder_validation_and_resume_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation_image = tmp_path / "validation.png"
    test_image = tmp_path / "test.png"
    Image.new("RGB", (91, 47), (1, 2, 3)).save(validation_image)
    Image.new("RGB", (67, 103), (4, 5, 6)).save(test_image)
    validation_manifest = tmp_path / "val.jsonl"
    test_manifest = tmp_path / "test.jsonl"
    _atomic_jsonl(validation_manifest, [_record(validation_image, "validation-1")])
    _atomic_jsonl(test_manifest, [_record(test_image, "test-1")])
    validation_sha = sha256_file(validation_manifest)
    test_sha = sha256_file(test_manifest)
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(
        cache_module,
        "EXPECTED_SPLIT_COUNTS",
        {"validation": 1, "test": 1},
    )
    monkeypatch.setattr(
        cache_module.concurrent.futures,
        "ProcessPoolExecutor",
        ThreadPoolExecutor,
    )

    complete = cache_module.build_cache(
        validation_manifest=validation_manifest,
        test_manifest=test_manifest,
        validation_manifest_sha256=validation_sha,
        test_manifest_sha256=test_sha,
        cache_root=cache_root,
        workers=2,
    )
    assert complete["record_count"] == 2
    validated = cache_module.validate_cache(
        validation_manifest=validation_manifest,
        test_manifest=test_manifest,
        validation_manifest_sha256=validation_sha,
        test_manifest_sha256=test_sha,
        cache_root=cache_root,
    )
    assert validated["record_count"] == 2
    first_complete_sha = sha256_file(cache_root / "complete.json")
    complete = cache_module.build_cache(
        validation_manifest=validation_manifest,
        test_manifest=test_manifest,
        validation_manifest_sha256=validation_sha,
        test_manifest_sha256=test_sha,
        cache_root=cache_root,
        workers=2,
    )
    assert complete["record_count"] == 2
    assert sha256_file(cache_root / "complete.json") == first_complete_sha

    first = json.loads(
        (cache_root / "cache_manifest.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    Path(first["cache_path"]).write_bytes(b"corrupt")
    complete = cache_module.build_cache(
        validation_manifest=validation_manifest,
        test_manifest=test_manifest,
        validation_manifest_sha256=validation_sha,
        test_manifest_sha256=test_sha,
        cache_root=cache_root,
        workers=2,
    )
    assert complete["record_count"] == 2
    assert not (cache_root / "runtime_validated.json").exists()
