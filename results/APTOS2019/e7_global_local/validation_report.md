# E7 Global/Local ConvNeXt-Tiny on APTOS2019

**Status: VALIDATION_COMPLETE.** The original run completed 20 epochs and selected epoch 19. The 366-image validation split supplies every result on this page. The run records no final-test access.

## Research question and architecture

E7 asks whether combining the full retinal field with localized views changes grading performance. One shared HF ConvNeXt V1 processes a full 384×384 letterbox and two 224×224 letterboxes from overlapping left/right crops. Each crop spans 55% of estimated field width and height, centered vertically. All views use EXIF-corrected RGB and official rescale/normalization with processor resize disabled. The gate combines global and mean local embeddings with learned sigmoid weights. A categorical head supplies grade predictions and an ordered-threshold head supplies auxiliary training supervision. The learned gate and geometric crop positions do not localize verified lesions.

The recorded 27,826,283 trainable parameters agree with the E5 trainable count plus 1,537 fusion-gate parameters. Backbone parameters are shared across all three views. [Model](../../../novelty_integrated/e7/model.py) · [Geometry](../../../novelty_integrated/e7/geometry.py) · [Runner and protocol](../../../novelty_integrated/e7/README.md) · [Notebook](../../../notebooks/novelty/E7_APTOS2019.ipynb).

## Training and validation protocol

E7 started from APTOS E5 epoch 16 with strict backbone/head transfer. The E5 checkpoint SHA-256 is `c44a88429f070f1eddc516ca9ece029c9f3a9d51b5624905bfcba20b5f6e45b4`. The new gate was initialized by the E7 model constructor. The run used seed 42, BF16, full fine-tuning, micro-batch 4 and accumulation 8 (effective batch 32, final window 18). AdamW used backbone LR 1e−5, head LR 1e−3, weight decay 0.01 and gradient clipping at 1.0. No scheduler or EMA was used.

The training objective is categorical CE plus 0.25 × ordinal BCE. Train-only inverse-square-root weights affect categorical CE from epoch 11, with weights normalized to mean one per training example. Validation loss is ordinary categorical CE. Validation is deterministic. B1 augmentation applies only to training and precedes view generation. The recorded training cache path points to the E6 decoded-RGB cache; validation has a separate E7 decoded-RGB cache. Checkpoint selection uses raw weights, highest validation macro-F1, lower validation CE for a tie, then the earlier epoch.

The frozen APTOS manifest SHA-256 is `73d1546e3c2fd2bf9e3d0fd10af914cb523d28e4bc8e1f25212af454d4be8854`. The manifest contains 2,930 train and 366 validation rows, with 2,832 and 351 SHA groups respectively. The local manifest and sidecar copies matched the recorded hashes, grade support, and train/validation group separation. No images were opened in this audit. The training record reports source-image verification.

## Complete epoch history

The [audited JSON](validation_evidence.json) preserves all recorded fields and numerical precision, including 40 train/validation confusion matrices, grade statistics and resource records. The [derived CSV](history.csv) flattens those same 20 source JSON rows. The ZIP did not contain the original runner CSV.

### Train scores

| Epoch | Total loss | Categorical | Ordinal BCE | Accuracy | Macro-F1 | Weighted-F1 | QWK | Grade MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.35295 | 0.30945 | 0.17403 | 0.87986 | 0.77922 | 0.87712 | 0.91953 | 0.16314 |
| 2 | 0.27349 | 0.23374 | 0.15899 | 0.90887 | 0.83060 | 0.90705 | 0.94273 | 0.12082 |
| 3 | 0.23012 | 0.19236 | 0.15102 | 0.93072 | 0.87277 | 0.93010 | 0.95634 | 0.09181 |
| 4 | 0.20544 | 0.16858 | 0.14745 | 0.93788 | 0.88352 | 0.93723 | 0.96245 | 0.08157 |
| 5 | 0.17764 | 0.14180 | 0.14335 | 0.94812 | 0.90271 | 0.94758 | 0.97015 | 0.06621 |
| 6 | 0.14768 | 0.11310 | 0.13832 | 0.96212 | 0.93113 | 0.96202 | 0.97930 | 0.04778 |
| 7 | 0.11994 | 0.08690 | 0.13217 | 0.97372 | 0.95189 | 0.97358 | 0.98456 | 0.03413 |
| 8 | 0.11352 | 0.08142 | 0.12840 | 0.97031 | 0.94505 | 0.97018 | 0.98202 | 0.03857 |
| 9 | 0.09785 | 0.06687 | 0.12391 | 0.97986 | 0.96042 | 0.97985 | 0.98739 | 0.02696 |
| 10 | 0.09121 | 0.06139 | 0.11926 | 0.98123 | 0.96450 | 0.98116 | 0.98824 | 0.02526 |
| 11 | 0.10739 | 0.07921 | 0.11272 | 0.98191 | 0.96461 | 0.98190 | 0.98964 | 0.02321 |
| 12 | 0.10298 | 0.07466 | 0.11329 | 0.98225 | 0.96579 | 0.98225 | 0.98897 | 0.02355 |
| 13 | 0.08639 | 0.05963 | 0.10702 | 0.98498 | 0.97172 | 0.98501 | 0.99118 | 0.01945 |
| 14 | 0.08881 | 0.06242 | 0.10557 | 0.98498 | 0.96940 | 0.98500 | 0.99150 | 0.01911 |
| 15 | 0.08655 | 0.06162 | 0.09970 | 0.98430 | 0.97125 | 0.98431 | 0.99036 | 0.02082 |
| 16 | 0.07564 | 0.05174 | 0.09558 | 0.98635 | 0.97176 | 0.98635 | 0.99190 | 0.01775 |
| 17 | 0.08795 | 0.06410 | 0.09542 | 0.98703 | 0.97352 | 0.98703 | 0.99263 | 0.01672 |
| 18 | 0.08438 | 0.06118 | 0.09277 | 0.98635 | 0.97326 | 0.98635 | 0.99209 | 0.01775 |
| 19 | 0.09027 | 0.06830 | 0.08789 | 0.98805 | 0.97534 | 0.98804 | 0.99270 | 0.01570 |
| 20 | 0.09020 | 0.06931 | 0.08356 | 0.98567 | 0.97041 | 0.98568 | 0.99161 | 0.01877 |

### Validation scores

| Epoch | Categorical CE | Accuracy | Macro-F1 | Weighted-F1 | QWK | Grade MAE |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.51794 | 0.81694 | 0.67262 | 0.81869 | 0.90400 | 0.22404 |
| 2 | 0.51373 | 0.81694 | 0.66462 | 0.81822 | 0.89789 | 0.23497 |
| 3 | 0.57575 | 0.82514 | 0.65128 | 0.81431 | 0.90278 | 0.21858 |
| 4 | 0.54137 | 0.83060 | 0.68624 | 0.82893 | 0.91102 | 0.20765 |
| 5 | 0.58260 | 0.80874 | 0.64089 | 0.80593 | 0.90280 | 0.23224 |
| 6 | 0.60226 | 0.82514 | 0.65031 | 0.81844 | 0.90847 | 0.21311 |
| 7 | 0.64533 | 0.82514 | 0.66463 | 0.82240 | 0.91003 | 0.21311 |
| 8 | 0.63820 | 0.83060 | 0.66965 | 0.82772 | 0.89618 | 0.22131 |
| 9 | 0.65669 | 0.81148 | 0.64729 | 0.81101 | 0.90492 | 0.22951 |
| 10 | 0.69420 | 0.81694 | 0.63338 | 0.81118 | 0.90486 | 0.22404 |
| 11 | 0.73487 | 0.83060 | 0.67348 | 0.82967 | 0.90779 | 0.21311 |
| 12 | 0.76798 | 0.81967 | 0.67506 | 0.82425 | 0.90837 | 0.21858 |
| 13 | 0.84671 | 0.82240 | 0.64827 | 0.81808 | 0.89736 | 0.22404 |
| 14 | 0.81368 | 0.82787 | 0.67206 | 0.82647 | 0.89615 | 0.21858 |
| 15 | 0.88898 | 0.82787 | 0.67457 | 0.82773 | 0.90197 | 0.21585 |
| 16 | 0.87240 | 0.83607 | 0.67546 | 0.83353 | 0.90434 | 0.20765 |
| 17 | 0.84547 | 0.83880 | 0.68131 | 0.83633 | 0.91208 | 0.20219 |
| 18 | 0.93040 | 0.81694 | 0.65422 | 0.81746 | 0.89447 | 0.24044 |
| 19 | 0.90345 | 0.83607 | 0.69649 | 0.83679 | 0.91269 | 0.20219 |
| 20 | 0.89353 | 0.84153 | 0.69076 | 0.83889 | 0.91436 | 0.19672 |

### Recorded resources

| Epoch | Train s | Validation s | Train images/s | Val images/s | GPU allocated B | GPU reserved B | Peak process RSS B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 778.93254 | 43.84373 | 3.76156 | 8.34783 | 1846094848 | 2034237440 | 2079842304 |
| 2 | 714.33775 | 52.80073 | 4.10170 | 6.93172 | 1846095360 | 2034237440 | 2175299584 |
| 3 | 776.57600 | 62.41222 | 3.77297 | 5.86424 | 1846095360 | 2034237440 | 2198048768 |
| 4 | 820.34739 | 69.84256 | 3.57166 | 5.24036 | 1846095360 | 2034237440 | 2201214976 |
| 5 | 924.33497 | 81.96025 | 3.16985 | 4.46558 | 1846095360 | 2034237440 | 2424455168 |
| 6 | 998.81871 | 68.86946 | 2.93347 | 5.31440 | 1846095360 | 2034237440 | 2424455168 |
| 7 | 781.74283 | 48.11213 | 3.74804 | 7.60723 | 1846095360 | 2034237440 | 2424455168 |
| 8 | 705.38395 | 58.19941 | 4.15377 | 6.28872 | 1846095360 | 2034237440 | 2424455168 |
| 9 | 711.84285 | 68.83685 | 4.11608 | 5.31692 | 1846095360 | 2034237440 | 2424455168 |
| 10 | 718.05370 | 68.84580 | 4.08047 | 5.31623 | 1846095360 | 2034237440 | 2428882944 |
| 11 | 697.14119 | 65.23211 | 4.20288 | 5.61073 | 1846095360 | 2034237440 | 2428882944 |
| 12 | 707.24757 | 69.47576 | 4.14282 | 5.26802 | 1846095360 | 2034237440 | 2428882944 |
| 13 | 688.13961 | 65.81456 | 4.25786 | 5.56108 | 1846095360 | 2034237440 | 2431250432 |
| 14 | 664.29093 | 69.53495 | 4.41072 | 5.26354 | 1846095360 | 2034237440 | 2445496320 |
| 15 | 691.89327 | 67.76054 | 4.23476 | 5.40137 | 1846095360 | 2034237440 | 2445496320 |
| 16 | 704.16873 | 75.25186 | 4.16093 | 4.86367 | 1846095360 | 2034237440 | 2445496320 |
| 17 | 775.24842 | 71.43953 | 3.77943 | 5.12321 | 1846095360 | 2034237440 | 2445496320 |
| 18 | 688.83116 | 76.04167 | 4.25358 | 4.81315 | 1846095360 | 2034237440 | 2445496320 |
| 19 | 831.80906 | 66.98229 | 3.52244 | 5.46413 | 1846095360 | 2034237440 | 2445975552 |
| 20 | 791.68081 | 53.13851 | 3.70099 | 6.88766 | 1846095360 | 2034237440 | 2458243072 |

Summed epoch times were 15,170.82 s training and 1,304.39 s validation (4.58 h combined). Peaks were 1.719 GiB GPU allocated, 1.895 GiB GPU reserved and 2.289 GiB process RSS. These are logged training resource records. The ZIP has no CPU or GPU utilization series or deployment latency measurement.

## Selected checkpoint and grading errors

| Metric | Epoch 19 |
| --- | --- |
| Accuracy | 0.836065574 |
| Macro-F1 | 0.696492253 |
| Weighted-F1 | 0.836788255 |
| QWK | 0.912686517 |
| Grade MAE | 0.202185792 |
| Validation CE | 0.903450064 |
| Correct | 306 |
| Incorrect | 60 |
| Undergraded | 31 |
| Overgraded | 29 |

True grades are rows and predictions are columns.

| True grade | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| 0 | 175 | 5 | 0 | 0 | 0 |
| 1 | 1 | 24 | 10 | 0 | 0 |
| 2 | 0 | 12 | 80 | 5 | 4 |
| 3 | 0 | 0 | 7 | 8 | 5 |
| 4 | 0 | 2 | 6 | 3 | 19 |

Row-normalized fractions

| True grade | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.97222 | 0.02778 | 0.00000 | 0.00000 | 0.00000 |
| 1 | 0.02857 | 0.68571 | 0.28571 | 0.00000 | 0.00000 |
| 2 | 0.00000 | 0.11881 | 0.79208 | 0.04950 | 0.03960 |
| 3 | 0.00000 | 0.00000 | 0.35000 | 0.40000 | 0.25000 |
| 4 | 0.00000 | 0.06667 | 0.20000 | 0.10000 | 0.63333 |

| Grade | Support | Precision | Recall | F1 | Under | Over |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 180 | 0.99432 | 0.97222 | 0.98315 | 0 | 5 |
| 1 | 35 | 0.55814 | 0.68571 | 0.61538 | 1 | 10 |
| 2 | 101 | 0.77670 | 0.79208 | 0.78431 | 12 | 9 |
| 3 | 20 | 0.50000 | 0.40000 | 0.44444 | 7 | 5 |
| 4 | 30 | 0.67857 | 0.63333 | 0.65517 | 11 | 0 |

Grade 1 recall was 24/35. Ten grade-1 images were graded 2 and one was graded 0. Grade 3 recall was 8/20, the lowest of the five grades. Seven grade-3 images were graded 2 and five were graded 4. Of 60 errors, 48 were one grade away and 12 were at least two grades away. The aggregate evidence does not identify clinical causes of these errors.

Validation CE reached its minimum at epoch 2 (0.513727). Epoch 19 was selected on macro-F1 under the original rule. Train loss and validation CE measure different objectives.

## APTOS2019 comparison with E5 and E6

| Selected validation metric | E5 | E6 | E7 |
| --- | --- | --- | --- |
| Best epoch | 16 | 15 | 19 |
| Macro-F1 | 0.705791432 | 0.679204179 | 0.696492253 |
| Accuracy | 0.830601093 | 0.789617486 | 0.836065574 |
| QWK | 0.899241254 | 0.880384339 | 0.912686517 |
| Grade MAE | 0.221311475 | 0.273224044 | 0.202185792 |
| Weighted-F1 | 0.833080466 | 0.795890536 | 0.836788255 |

E5 achieved the highest macro-F1. E7 improved accuracy, weighted-F1 and QWK and reduced grade MAE relative to E5, while reducing grade-1 and grade-3 recall. E6 trailed E5 on the listed aggregate metrics. The selected checkpoints share the frozen validation set. The comparison uses a single seed and the same validation data for checkpoint selection, with no significance test. E7 warm-started from E5 and adds three-view inference, so optimization budget and computation differ. These results do not establish a gain attributable to the views or clinical superiority. [E5 validation report](../e5_integrated/validation_report.md) · [E6 validation report](../e6_ldam_ordinal_adjacent/validation_report.md).

## Figure index

| Panel | Evidence shown |
| --- | --- |
| [PNG](../../../docs/figures/aptos2019_e7_epoch_scores.png) / [SVG](../../../docs/figures/aptos2019_e7_epoch_scores.svg) | All five metrics for train and validation across 20 epochs, with epoch 19 marked. |
| [PNG](../../../docs/figures/aptos2019_e7_losses.png) / [SVG](../../../docs/figures/aptos2019_e7_losses.svg) | Training total, weighted or unweighted categorical term, ordinal BCE, and validation categorical CE. |
| [PNG](../../../docs/figures/aptos2019_e7_resources.png) / [SVG](../../../docs/figures/aptos2019_e7_resources.svg) | Epoch times, throughput, GPU allocator peaks and process RSS high-water mark. |
| [PNG](../../../docs/figures/aptos2019_e7_confusion.png) / [SVG](../../../docs/figures/aptos2019_e7_confusion.svg) | Epoch-19 validation confusion in counts and true-grade row percentages. |
| [PNG](../../../docs/figures/aptos2019_e7_grade_errors.png) / [SVG](../../../docs/figures/aptos2019_e7_grade_errors.svg) | Grade-wise precision, recall, F1, and under/overgraded counts with support. |
| [PNG](../../../docs/figures/aptos2019_e7_e5_e6_e7_comparison.png) / [SVG](../../../docs/figures/aptos2019_e7_e5_e6_e7_comparison.svg) | Selected validation checkpoints on the same APTOS2019 split, including minority-grade recall. |

## Provenance and limitations

The run was archived under `foundation_e7/APTOS2019/run/train` relative to the experiment root. All 18 recorded source digests match Git commit `b7c42b6a3560d233960707390119e05731d635c3` and the local source. The transfer SHA-256 is `f60524e43ca2dd594526332368e895fb4ab746321b2a55f57d9e0ddc3b2e431d`. The JSON records the five original member hashes, the processor revision and hashes, cache report digests and runtime settings. Absolute dataset roots were omitted and cache roots converted to paths relative to the experiment root.

| Checkpoint | Bytes | SHA-256 |
| --- | --- | --- |
| best.pt | 334209883 | 5ef9816b9c7511a2b0347f7967543f04d70a77c9d0de630fd4116f714c38d11c |
| last.pt | 334211995 | 428920a006d8666e19c74850ce9ad9912b30356e120636fce3afceeb62b5f940 |

The archived sidecars match the values independently hashed on HPC and supplied by the operator. The ZIP contains no binaries, source images, row-level predictions, or original smoke/benchmark reports. No E7 checkpoint was loaded or exported, and no latency benchmark was run here. DDR completion remains unverified.
