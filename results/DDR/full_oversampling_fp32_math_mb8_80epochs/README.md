# DDR — Full Oversampling FP32 Math Recovery

## Status
- Training: **80/80 epochs completed**
- Best checkpoint: **epoch 69**
- Best validation macro-F1: **0.653069**
- Best validation accuracy: **0.785942**
- Final test: **COMPLETED** from the epoch-69 best checkpoint; see `final_test_summary.json` and `FINAL_TEST_README.md`

## Experiment provenance
The original FP32 training completed epochs 1–42.
Following a numerical failure, a diagnostic recovery fork continued
epochs 43–80 using Math SDPA, micro-batch 8, and gradient accumulation 2.

This is a **mixed-protocol recovery campaign**, not a fully homogeneous
80-epoch training run. See `FORK_PROVENANCE.json`.

## Files
- `config.json`: run configuration
- `protocol.json`: recovery protocol
- `FORK_PROVENANCE.json`: fork lineage
- `history.jsonl`: all 80 epoch metrics
- `summary.json`: training and validation summary
- `checkpoint_manifest.json`: SHA-256 checkpoint identities
- `*.png`: training and validation graphs

## Interpretation
`history.jsonl` and the validation entries in `summary.json` are training/validation metrics.
The separate `final_test_summary.json` contains DDR final-test metrics: 1,252 test images,
975 correct, accuracy 0.778754, and macro-F1 0.631981. Its confusion matrix
and per-class report are in `confusion_matrix.png` and `per_class_metrics.csv`.
Per-image predictions remain on the server.

Checkpoint binaries are stored outside GitHub.
No raw fundus images or patient data are included.
