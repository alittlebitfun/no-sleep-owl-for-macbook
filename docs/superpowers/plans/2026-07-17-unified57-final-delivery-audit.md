# Unified57 Final Delivery Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone read-only auditor that independently verifies the frozen Unified57 dataset, evaluation, post-training, and optional sealed-delivery claims and always writes a reproducible acceptance bundle for completed performance audits.

**Architecture:** One standard-library Python CLI owns strict input validation, independent metric recomputation, optional artifact audits, and atomic report writing. Tests build compact 57-label fixtures from the production schema and call public functions plus the CLI entrypoint without GPU, model, network, or remote state.

**Tech Stack:** Python 3.10+ standard library, pytest, JSON/JSONL/CSV/Markdown.

## Global Constraints

- Never write below schema, dataset, evaluation, post-training, or sealed-delivery input paths.
- Unknown cells are excluded from PN metrics and trusted negatives.
- Exit `0` means full success, `1` means completed partial/fail with all reports present, and `2` means contract/integrity failure with diagnostic reports present when the output directory is writable.
- Do not modify training, evaluator, supervisor, package, or active remote state.
- Production defaults are validation `5444` and test `5441`; tests pass explicit small counts.

---

### Task 1: Frozen contract and ordered evidence validation

**Files:**
- Create: `tests/test_audit_unified57_final_delivery.py`
- Create: `scripts/audit_unified57_final_delivery.py`

**Interfaces:**
- Produces: `AuditPaths`, `AuditContractError`, `load_schema(path)`, `load_split_evidence(manifest_path, prediction_path, schema, expected_count)`, and `validate_leakage(path)`.
- Consumes: frozen JSON/JSONL paths and the production schema shape.

- [ ] **Step 1: Write failing tests for valid evidence, mask violation, and record-order drift**

```python
def test_load_split_rejects_unknown_mask_violation(fixture):
    fixture.test_manifest_rows[0]["labels"][fixture.pu_index] = 1
    fixture.rewrite()
    with pytest.raises(AuditContractError, match="unknown cell"):
        load_split_evidence(
            fixture.test_manifest, fixture.test_predictions,
            fixture.schema, expected_count=fixture.test_count,
        )

def test_load_split_rejects_prediction_order_drift(fixture):
    fixture.reverse_prediction_order()
    with pytest.raises(AuditContractError, match="manifest order"):
        load_split_evidence(
            fixture.test_manifest, fixture.test_predictions,
            fixture.schema, expected_count=fixture.test_count,
        )
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `PYTHONPATH=. pytest -q tests/test_audit_unified57_final_delivery.py -k 'unknown_mask or order_drift'`

Expected: collection/import failure because `scripts.audit_unified57_final_delivery` does not exist.

- [ ] **Step 3: Implement strict schema, mask, count, order, equality, and leakage validation**

```python
class AuditContractError(ValueError):
    pass

def load_split_evidence(manifest_path, prediction_path, schema, expected_count):
    manifest = load_jsonl(manifest_path)
    predictions = load_jsonl(prediction_path)
    require(len(manifest) == expected_count, "unexpected manifest count")
    require([r["record_id"] for r in predictions] ==
            [r["record_id"] for r in manifest],
            "predictions differ from manifest order")
    for truth, scored in zip(manifest, predictions):
        validate_masks(truth, schema)
        for field in EVIDENCE_FIELDS:
            require(scored.get(field) == truth.get(field),
                    f"prediction evidence differs at {field}")
        validate_scores(scored["scores"], 57)
    return manifest, predictions
```

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run: `PYTHONPATH=. pytest -q tests/test_audit_unified57_final_delivery.py -k 'unknown_mask or order_drift or valid_evidence'`

Expected: all selected tests pass.

### Task 2: Independent metrics, six gates, and complete reports

**Files:**
- Modify: `tests/test_audit_unified57_final_delivery.py`
- Modify: `scripts/audit_unified57_final_delivery.py`

**Interfaces:**
- Produces: `recompute_split_metrics(rows, thresholds, schema)`, `classify(values)`, `audit_delivery(paths, expected_validation_count, expected_test_count)`, and atomic report writers.
- Consumes: validated manifest/prediction rows and threshold entries.

- [ ] **Step 1: Write failing success and partial tests**

```python
def test_success_audit_recomputes_all_six_gates_and_writes_reports(fixture):
    result = audit_delivery(fixture.paths, fixture.val_count, fixture.test_count)
    assert result["verdict"] == "success"
    assert all(result["success_gates"].values())
    assert (fixture.output / "acceptance_audit.json").is_file()
    assert (fixture.output / "per_label_metrics.csv").is_file()
    assert (fixture.output / "FINAL_REPORT.md").is_file()

def test_partial_exit_one_still_writes_complete_reports(fixture):
    fixture.make_test_known_micro_f1(0.84)
    exit_code = main(fixture.cli_args())
    assert exit_code == 1
    assert json.loads((fixture.output / "acceptance_audit.json").read_text())["verdict"] == "partial"
    assert (fixture.output / "FINAL_REPORT.md").is_file()
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `PYTHONPATH=. pytest -q tests/test_audit_unified57_final_delivery.py -k 'success_audit or partial_exit'`

Expected: failures because metric and report functions are absent.

- [ ] **Step 3: Implement independent raw/final predictions, PN/JD23/dictionary metrics, gate classification, CSV, JSON, and Markdown**

```python
SUCCESS_THRESHOLDS = {
    "known_micro_f1": 0.88,
    "jd23_micro_f1": 0.88,
    "macro_f1": 0.75,
    "dictionary_positive_macro_recall": 0.85,
    "trusted_negative_specificity": 0.90,
    "json_validity_rate": 1.0,
}

def classify(values):
    gates = {name: values[name] >= threshold
             for name, threshold in SUCCESS_THRESHOLDS.items()}
    gates["json_validity_rate"] = values["json_validity_rate"] == 1.0
    verdict = "success" if all(gates.values()) else (
        "partial" if values["known_micro_f1"] >= 0.82 and
        gates["json_validity_rate"] else "fail"
    )
    return verdict, gates
```

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run: `PYTHONPATH=. pytest -q tests/test_audit_unified57_final_delivery.py -k 'success_audit or partial_exit or per_label'`

Expected: all selected tests pass and no warnings.

### Task 3: Three output modes and representative metadata

**Files:**
- Modify: `tests/test_audit_unified57_final_delivery.py`
- Modify: `scripts/audit_unified57_final_delivery.py`

**Interfaces:**
- Produces: `audit_output_modes(...)` and `select_representatives(...)`.
- Consumes: test predictions, thresholds, schema, and any available evaluator/posttrain mode JSONL files.

- [ ] **Step 1: Write failing cross-mode and representative tests**

```python
def test_three_modes_are_strict_and_consistent(fixture):
    result = audit_output_modes(fixture.evaluation, fixture.rows,
                                fixture.thresholds, fixture.schema)
    assert result["available_modes"] == [
        "selected_only", "selected_with_confidence", "all_scores"
    ]
    assert result["json_validity_rate"] == 1.0
    assert result["cross_mode_consistent"] is True

def test_representatives_select_successes_and_errors_deterministically(fixture):
    first = select_representatives(fixture.rows, fixture.thresholds, fixture.schema)
    second = select_representatives(list(reversed(fixture.rows)),
                                    fixture.thresholds, fixture.schema)
    assert [r["record_id"] for r in first] == [r["record_id"] for r in second]
    assert {r["outcome"] for r in first} == {"success", "error"}
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `PYTHONPATH=. pytest -q tests/test_audit_unified57_final_delivery.py -k 'three_modes or representatives'`

Expected: failures because mode and representative functions are absent.

- [ ] **Step 3: Implement strict format parsing, score consistency, subcategory exclusivity, and stable selection**

```python
def stable_key(row):
    return hashlib.sha256(("20260717:" + row["record_id"]).encode()).hexdigest()

def select_representatives(rows, thresholds, schema):
    annotated = [representative_metadata(row, thresholds, schema) for row in rows]
    successes = sorted((r for r in annotated if r["outcome"] == "success"), key=stable_key)
    errors = sorted((r for r in annotated if r["outcome"] == "error"), key=stable_key)
    return select_with_source_coverage(successes, 3) + select_with_source_coverage(errors, 3)
```

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run: `PYTHONPATH=. pytest -q tests/test_audit_unified57_final_delivery.py -k 'three_modes or representatives'`

Expected: all selected tests pass.

#### Final-mode execution evidence contract

`posttrain/final_mode_verification/final_mode_verification.json` must bind the
format-only outputs to the current candidate bytes and replay result. In
addition to the format booleans and output hashes, it must contain:

- `candidate_infer_sha256`: SHA256 of `delivery_candidate/infer.py`;
- `reproduction_result_sha256`: SHA256 of the current
  `posttrain/reproduction_result.json`;
- `commands`: recorded successful commands that invoke that exact candidate
  `infer.py` through `--scores-json`, covering both
  `--mode selected_with_confidence` and `--mode all_scores`;
- `environment`: non-empty `gpu`, `cuda`, `pytorch`, `transformers`, `peft`,
  `safetensors`, and `pillow` strings.

The auditor also requires candidate verification references to be byte-equal
to `evaluation/verification/`, then independently binds every selected record,
image SHA, float32 score vector, and selected-only output to the frozen full
test predictions. Self-signed candidate references are not an acceptance root.

### Task 4: Posttrain and sealed-inventory audit, CLI hardening, regression

**Files:**
- Modify: `tests/test_audit_unified57_final_delivery.py`
- Modify: `scripts/audit_unified57_final_delivery.py`

**Interfaces:**
- Produces: `audit_posttrain(path)`, `verify_sealed_inventory(path)`, `parse_args(argv)`, and `main(argv=None) -> int`.
- Consumes: posttrain final/reproduction JSON and optional sealed `SHA256SUMS`.

- [ ] **Step 1: Write failing sealed inventory success and tamper tests**

```python
def test_sealed_inventory_verifies_every_file(fixture):
    result = verify_sealed_inventory(fixture.sealed)
    assert result["complete"] is True
    assert result["verified_files"] > 0

def test_sealed_inventory_rejects_tampering(fixture):
    (fixture.sealed / "model_config.json").write_text("tampered")
    with pytest.raises(AuditContractError, match="checksum mismatch"):
        verify_sealed_inventory(fixture.sealed)
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `PYTHONPATH=. pytest -q tests/test_audit_unified57_final_delivery.py -k 'sealed_inventory'`

Expected: failures because sealed validation is absent.

- [ ] **Step 3: Implement exact SHA256 inventory equality, posttrain evidence, CLI error reporting, and final atomic writes**

```python
def verify_sealed_inventory(root):
    declared = parse_sha256s(root / "SHA256SUMS")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.name != "SHA256SUMS"}
    require(set(declared) == actual, "sealed inventory differs from files")
    for relative, digest in declared.items():
        require(sha256_file(root / relative) == digest,
                f"sealed checksum mismatch: {relative}")
    return {"complete": True, "verified_files": len(declared)}
```

- [ ] **Step 4: Run focused and complete tests**

Run:

```bash
PYTHONPATH=. pytest -q tests/test_audit_unified57_final_delivery.py
PYTHONPATH=. pytest -q tests/test_unified57_evaluation_core.py tests/test_evaluate_unified57_multilabel.py tests/test_package_unified57_delivery.py tests/test_supervise_unified57_posttrain.py
python3 -m py_compile scripts/audit_unified57_final_delivery.py
```

Expected: all audit tests pass; existing Unified57 tests pass in the configured project runtime; py_compile exits `0`.

- [ ] **Step 5: Commit only audit-owned files**

```bash
git add \
  docs/superpowers/plans/2026-07-17-unified57-final-delivery-audit.md \
  scripts/audit_unified57_final_delivery.py \
  tests/test_audit_unified57_final_delivery.py
git commit -m "feat: add independent unified57 delivery audit"
```
