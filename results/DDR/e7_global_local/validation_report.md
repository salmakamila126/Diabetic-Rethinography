# E7 Global/Local ConvNeXt-Tiny on DDR

**PARTIAL / EARLY STOPPED AT EPOCH 10.** The 20-epoch run stopped during epoch 11. Ten completed epochs are recorded. Validation macro-F1 selected epoch 2 from the recorded interval. Final test remains `NOT_ACCESSED`.

[Research notebook](../../../notebooks/novelty/E7_DDR.ipynb) · [Audited aggregate evidence](validation_evidence.json) · [Full epoch CSV](history.csv) · [E7 training protocol](../../../novelty_integrated/e7/README.md)

## Research question and protocol

E7 tests whether a global view and two detail views change grading performance. One shared ConvNeXt-Tiny receives a 384×384 letterboxed image and two 224×224 local FOV crops. A learned gate fuses the pooled global and mean local features. The model warm-started from DDR E5 epoch 18 (`7dd7488013b7af5ad4565052b47d203c8077d8203dd5520d5b20b0fe592fbe00`). Seed 42, B1 training-only augmentation, raw selected weights, BF16, AdamW, effective batch 32 and the categorical plus 0.25 ordinal BCE objective are recorded in the evidence. Class reweighting was scheduled for epoch 11 and is inactive in all ten completed epochs.

Frozen support is 10,018 training and 1,252 validation images. Validation grades 0–4 contain 626, 63, 448, 24 and 91 images. Selection maximizes validation macro-F1, then minimizes validation categorical CE, then favors the earlier epoch. Training and validation used different loss definitions. See the epoch CSV for full-precision values and per-epoch resources.

## Selected validation checkpoint

| Frozen DDR validation | E5 | E6 | E7 (partial) |
| --- | --- | --- | --- |
| Epochs completed | 20 | 20 | 10 |
| Selected epoch | 18 | 12 | 2 |
| Accuracy | 0.823482428 | 0.797923323 | 0.833865815 |
| Macro-F1 | 0.653199134 | 0.662455052 | 0.668562794 |
| Weighted-F1 | 0.818187567 | 0.812289836 | 0.825592959 |
| QWK | 0.857161823 | 0.849553202 | 0.852150858 |
| Grade MAE | 0.266773163 | 0.283546326 | 0.261182109 |
| Grade 1 recall | 0.190476 | 0.365079 | 0.174603 |
| Grade 3 recall | 0.541667 | 0.458333 | 0.458333 |


E7 has the highest macro-F1 and accuracy of the three selected checkpoints in this comparison. E5 retains higher QWK. E7 grade 1 recall is 11/63, below both E5 and E6. Grade 3 recall is 11/24, tied with E6 and below E5. Grade 1 has the lowest E7 F1. These small class supports limit interpretation.

| Grade | Support | Precision | Recall | F1 | Overgraded | Undergraded |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 626 | 0.894984 | 0.912141 | 0.903481 | 55 | 0 |
| 1 | 63 | 0.343750 | 0.174603 | 0.231579 | 33 | 19 |
| 2 | 448 | 0.778234 | 0.845982 | 0.810695 | 9 | 60 |
| 3 | 24 | 0.687500 | 0.458333 | 0.550000 | 2 | 11 |
| 4 | 91 | 0.911392 | 0.791209 | 0.847059 | 0 | 19 |


Of 1,252 validation images, 1,044 were graded correctly. 99 were overgraded and 109 were undergraded. For true grade 1, 19 predictions were grade 0 and 33 were grade 2. For true grade 3, 11 predictions were grade 2 and 2 were grade 4. These counts describe the recorded grading errors.

Training total loss falls from 0.2979 to 0.0380. Validation categorical CE is 0.6243 at the selected epoch and 1.0780 at epoch 10. Macro-F1 at epoch 10 is 0.651676. The comparison uses one validation-selected checkpoint per method, one run per method, and unequal training budgets. It provides no causal estimate of multi-view fusion, confirmed novelty, or independent final-test performance.

## Figure index

- Train total objective and validation categorical CE by epoch. [SVG](../../../docs/figures/ddr_e7_losses.svg) · [PNG](../../../docs/figures/ddr_e7_losses.png)
- Train and validation macro-F1, accuracy and QWK by epoch. [SVG](../../../docs/figures/ddr_e7_epoch_scores.svg) · [PNG](../../../docs/figures/ddr_e7_epoch_scores.png)
- Best-checkpoint counts and true-grade row percentages. [SVG](../../../docs/figures/ddr_e7_confusion.svg) · [PNG](../../../docs/figures/ddr_e7_confusion.png)
- Precision, recall, F1 and direction of errors by true grade. [SVG](../../../docs/figures/ddr_e7_grade_errors.svg) · [PNG](../../../docs/figures/ddr_e7_grade_errors.png)
- Best-checkpoint comparison on the same DDR validation split. [SVG](../../../docs/figures/ddr_e7_e5_e6_e7_comparison.svg) · [PNG](../../../docs/figures/ddr_e7_e5_e6_e7_comparison.png)

Lines connect only the ten recorded epoch values. The vertical marker shows epoch 2. The normalized matrix divides each row by its true-grade support. The PNG and SVG versions use the same plotted data and embed the same machine-readable figure payload.

From the repository root, render all figures into a fresh directory with Matplotlib 3.11.2 installed:

```bash
python -m docs.e7_ddr_publication --figures ../e7-ddr-rendered
```

## Provenance and limitations

- Original aggregate archive SHA-256 `1b1d0b82779071552051bc3c01b478c9c4b77a50b58d2d90060b88765c9d8d85`. Its seven member hashes are retained in the evidence.
- Training source revision `b7c42b6a3560d233960707390119e05731d635c3`. All 18 recorded source files matched both that Git revision and the local source during import.
- Frozen manifest SHA-256 `7bc3e3916e012bc46fcd9cd69aa469e0daf3229e1273f1d8ecb6054899bf6eb1`. Sidecar and ordered membership digests match the published DDR E5/E6 receipts. The ZIP contains aggregate evidence and no manifest bytes or checkpoint binaries.
- `best.pt` · epoch 2 · 334,173,979 bytes · `ecfed1989c4f6bd89737942796c73e75257b628fd79ad37fee344d340ee1323f`
- `last.pt` · epoch 10 · 334,194,907 bytes · `68f277406b8d04a9a0df4069150de747aeeed961af82d12566489d6d30508710`

The checkpoint SHA-256 values are those recorded in the exported summaries. The original binaries remain external. Private cache and dataset paths were removed from the public record. The original ten history objects were retained unchanged. No recovery, backend parity or latency result is included for E7 DDR.
