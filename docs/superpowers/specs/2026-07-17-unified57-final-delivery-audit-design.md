# Unified57 Final Delivery Audit Design

## Objective

Add one standalone, read-only command-line auditor for the frozen Unified57
dataset, evaluation, post-training, and optional sealed-delivery artifacts. It
must independently verify integrity and metric claims, then write a complete
acceptance bundle even when the measured model verdict is `partial` or `fail`.

## Interface

The command accepts explicit paths for the schema, dataset root, evaluation
directory, post-training directory, output directory, and an optional sealed
delivery directory. Frozen production defaults remain 5,444 validation records
and 5,441 test records; count overrides exist only to support small deterministic
fixtures and alternate frozen contracts.

Exit codes:

- `0`: all integrity checks and all six success gates pass.
- `1`: the audit completed and reports were written, but performance is partial
  or fail.
- `2`: an input contract, integrity, mask, order, hash, or sealed-inventory
  validation failed. The auditor still writes the audit and Markdown report when
  the output directory is usable.

## Read-only boundary

The auditor never writes below any input directory, never imports model runtime
code, never allocates a GPU, and never invokes remote commands. Outputs are
written atomically below the caller-supplied output directory.

## Validation pipeline

1. Validate the exact 57-label, 36 PN / 20 PU / 1 unsupported schema.
2. Load validation and test manifests plus float32 predictions as ordered JSONL.
3. Require exact record counts, unique identifiers, identical manifest order,
   and equality of truth/mask/source/image evidence between each prediction and
   its frozen manifest row.
4. Enforce disjoint masks and strict unknown semantics:
   - known cells occur only on PN labels;
   - PU positives occur only on PU labels and have label `1`;
   - cells with neither mask have neutral label `0`;
   - unsupported `假两件` is never supervised.
5. Validate the leakage audit and its empty cross-split collision lists.
6. Validate threshold schema/checkpoint/validation contracts.
7. Independently reproduce raw thresholding, final one-tag-per-subcategory
   formatting, all PN counts, JD23 clean metrics, dictionary explicit-positive
   recall, trusted-negative specificity, per-label support, and the six delivery
   gates. Unknown and PU-unlabeled cells never enter PN accuracy/F1/specificity.
8. Cross-check recomputed values against evaluation_report.json with tight
   floating-point tolerance.
9. Validate every available output-mode JSONL for record coverage, ordering,
   schema, two-decimal formatting, mutual exclusion, unsupported score, and
   consistency with frozen float32 scores.
10. Validate required post-training reports and, when supplied, the sealed
    delivery `SHA256SUMS` inventory and hashes.
11. Deterministically select metadata for three success and three error examples,
    preferring JD, dictionary, and mixed source coverage. Selection does not
    affect metrics and never copies or modifies source images.

## Outputs

- `acceptance_audit.json`: complete machine-readable checks, recomputed metrics,
  gate results, provenance, warnings, and final verdict.
- `per_label_metrics.csv`: one row per supported label with mode, support,
  confusion counts when valid, recall, F1, specificity, PU diagnostics, and gate
  warning flags.
- `FINAL_REPORT.md`: concise Chinese handoff report with counts, six metrics,
  support caveats, integrity status, output-mode status, and remaining risks.
- `output_modes_audit.json`: emitted when any output-mode artifacts exist.
- `representative_selection.jsonl`: deterministic success/error metadata.

## Acceptance semantics

The six frozen success thresholds are known-PN micro F1 >= 0.88, JD23 micro F1
>= 0.88, PN macro F1 >= 0.75, dictionary explicit-positive macro recall >=
0.85, trusted-negative specificity >= 0.90, and strict JSON validity exactly
1.0. A performance result is `partial` when known-PN micro F1 >= 0.82 and JSON
validity is 1.0; otherwise it is `fail`.

All 20 PU validation supports are below the current calibration minimum. The
auditor must report fallback thresholds and explicitly state that confidence
scores are model scores, not calibrated probabilities.

## Tests

Focused synthetic fixtures cover:

- a complete `success` audit and a complete `partial` audit;
- strict unknown/mask violations;
- prediction record-order drift;
- all three final output formats and cross-mode consistency;
- sealed delivery inventory success and tampering;
- complete report generation for exit code `1`.
