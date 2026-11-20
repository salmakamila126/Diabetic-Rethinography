# E5 APTOS2019 frozen validation

2,930 training and 366 validation images. The run completed 20 epochs. The selected checkpoint was epoch **16**.

[E5–E6 comparison and figure index](../../../docs/E5_E6_VALIDATION.md) · [Full aggregate evidence, configuration, and hashes](validation_evidence.json)

The [separately documented historical final test](README.md) retains its original scope. This report uses training and validation exports.

## Run configuration and provenance

| Field | Recorded value |
| --- | --- |
| Run identity | foundation_e5/APTOS2019/integrated_seed42_b32_20_cached/run |
| Architecture | ConvNeXt-Tiny with five-class categorical and ordered-threshold ordinal heads |
| Foundation model / revision | facebook/convnext-tiny-224 / 6166b7613034066690a621d8bf25ffdf181a34f0 |
| Seed / epochs | 42 / 20 |
| Micro-batch × accumulation → effective batch | 32 × 1 → 32 |
| Optimizer / LR backbone / LR head / decay / clip | AdamW / 1e-05 / 0.001 / 0.01 / 1.0 |
| Precision / device / scheduler | bf16 / cuda / none in runner |
| Data transport, train | B1 EXIF-transposed RGB then fresh train-only augmentation then official processor |
| Data transport, validation | verified official float32 tensor cache |
| Workers / persistent / prefetch / pin | 2 / True / 2 / True |
| Train support grades 0–4 | [1445, 297, 798, 154, 236] |
| Validation support grades 0–4 | [180, 35, 101, 20, 30] |
| Checkpoint selection | max validation macro-F1; tie lower validation CE; tie earliest epoch |
| Parameter counts | backbone_total: 27,820,128, backbone_trainable: 27,820,128, head_total: 4,618, total_trainable: 27,824,746 |
| Versions | torch 2.7.1+cu126, transformers 4.57.3, pillow 11.1.0, numpy 1.26.4 |
| Source commit | 4c9ab9d403ec1d9384ef86dc6de14378b68a77fe |
| Frozen manifest SHA-256 | 73d1546e3c2fd2bf9e3d0fd10af914cb523d28e4bc8e1f25212af454d4be8854 |
| Manifest sidecar SHA-256 | d3e367237f1a3d43ed9948cf40c6065d9ba2756e402742b7cb253bb1be44a657 |
| Ordered train/validation identity SHA-256 | 487cfa60222c08d4a2b6b2b48b05f89fea0f0d3f7285d924059939966a12dae0 |
| Processor SHA-256 / effective processor SHA-256 | fd575b890da5a949493e1d1e7a70bfcb9e4b99fe444004d2dbfa253add254741 / dd1700334d237761807778002bac2692337179f1ff24e0519fd67e74090bd5fd |
| Official weight SHA-256 | 1fe15b2eba9e355de14c4e244e6365bf61ffef84da7d3442c0afbb2d8564f477 |
| categorical | ordinary CE; train-only inverse-sqrt weights from epoch 11 if deferred |
| deferred weighting | True |
| ordinal | True |
| ordinal coefficient | 0.25000 |
| validation loss | ordinary categorical CE for every arm |
| weight start epoch | 11 |
| weights | [0.6927309254961216, 1.5279891665791683, 0.9321741471935598, 2.121964623980555, 1.7141250365704452] |
| Source best.pt | c44a88429f070f1eddc516ca9ece029c9f3a9d51b5624905bfcba20b5f6e45b4 (334,177,899 bytes) |
| Source last.pt | 8682dc83f230c7bdd03ae674cdcb2a586b50728d7a33c66f8e3b450467614f4e (334,186,411 bytes) |


Processor, augmentation, optimizer, class balance, source hashes, and the complete transport configuration are in the JSON. The original run stores configuration in `run.json`. The supplied checkpoint sidecars match the recorded checkpoint digests and byte sizes.

## Best checkpoint metrics

| Metric | Value | Source |
| --- | --- | --- |
| Accuracy | 0.83060 | Recorded |
| Macro-F1 | 0.70579 | Recorded |
| Weighted-F1 | 0.83308 | Recorded |
| QWK | 0.89924 | Recorded |
| Grade MAE | 0.22131 | Recorded |
| Categorical CE | 0.58854 | Recorded |
| Macro precision | 0.70638 | Matrix-derived |
| Macro recall | 0.71240 | Matrix-derived |
| correct | 304 | Recorded |
| wrong | 62 | Recorded |
| undergraded | 32 | Recorded |
| overgraded | 30 | Recorded |
| mean overgrade magnitude | 1.23333 | Recorded |
| mean undergrade magnitude | 1.37500 | Recorded |
| high confidence wrong | 23 | Recorded |
| high confidence wrong rate all | 0.06284 | Recorded |
| high confidence wrong fraction of wrong | 0.37097 | Recorded |
| Adjacent mistakes (absolute grade difference 1) | 45 | Matrix-derived |
| Non-adjacent mistakes (absolute grade difference ≥2) | 17 | Matrix-derived |


### Confusion matrix

Rows are true grades and columns are predicted grades. Counts include the diagonal. Percentages are normalized within each true grade.

| True grade | 0 | 1 | 2 | 3 | 4 | Support |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 174 | 6 | 0 | 0 | 0 | 180 |
| 1 | 1 | 26 | 8 | 0 | 0 | 35 |
| 2 | 0 | 13 | 76 | 5 | 7 | 101 |
| 3 | 0 | 0 | 6 | 10 | 4 | 20 |
| 4 | 0 | 2 | 8 | 2 | 18 | 30 |


| True grade | 0 (%) | 1 (%) | 2 (%) | 3 (%) | 4 (%) |
| --- | --- | --- | --- | --- | --- |
| 0 | 96.67 | 3.33 | 0.00 | 0.00 | 0.00 |
| 1 | 2.86 | 74.29 | 22.86 | 0.00 | 0.00 |
| 2 | 0.00 | 12.87 | 75.25 | 4.95 | 6.93 |
| 3 | 0.00 | 0.00 | 30.00 | 50.00 | 20.00 |
| 4 | 0.00 | 6.67 | 26.67 | 6.67 | 60.00 |


Largest off-diagonal cells (true→predicted) were 2→1 (13), 4→2 (8), 1→2 (8), 2→4 (7). The grade 1 row was `[1, 26, 8, 0, 0]`; the grade 3 row was `[0, 0, 6, 10, 4]`.

### Per-grade metrics

Support, TP, precision, recall, and F1 are recorded and independently recomputed from the matrix. FP and FN are matrix-derived. Per-grade MAE was recorded in the CSV.

| Grade | Support | TP | FP | FN | Precision | Recall | F1 | Grade MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 180 | 174 | 1 | 6 | 0.99429 | 0.96667 | 0.98028 | 0.03333 |
| 1 | 35 | 26 | 21 | 9 | 0.55319 | 0.74286 | 0.63415 | 0.25714 |
| 2 | 101 | 76 | 22 | 25 | 0.77551 | 0.75248 | 0.76382 | 0.31683 |
| 3 | 20 | 10 | 7 | 10 | 0.58824 | 0.50000 | 0.54054 | 0.50000 |
| 4 | 30 | 18 | 11 | 12 | 0.62069 | 0.60000 | 0.61017 | 0.80000 |


## All 20 epochs

Markdown rounds values for readability. The JSON preserves the precision and full field coverage of the export. Per-grade MAE fields removed from the JSON by the export filter are preserved separately from the CSV.

### Training scores

| Epoch | Total loss | Categorical | Ordinal | Accuracy | Macro-F1 | Weighted-F1 | QWK | Grade MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.89716 | 0.81704 | 0.32048 | 0.70444 | 0.36052 | 0.64146 | 0.58996 | 0.52082 |
| 2 | 0.60857 | 0.54784 | 0.24291 | 0.79249 | 0.49678 | 0.75218 | 0.81474 | 0.30307 |
| 3 | 0.53077 | 0.47472 | 0.22420 | 0.82253 | 0.61312 | 0.80223 | 0.85750 | 0.25119 |
| 4 | 0.48459 | 0.43050 | 0.21638 | 0.83140 | 0.65416 | 0.81875 | 0.87518 | 0.23379 |
| 5 | 0.44948 | 0.39767 | 0.20725 | 0.84369 | 0.67494 | 0.83279 | 0.88667 | 0.21672 |
| 6 | 0.41515 | 0.36536 | 0.19914 | 0.86348 | 0.72679 | 0.85708 | 0.90313 | 0.18874 |
| 7 | 0.38212 | 0.33404 | 0.19231 | 0.87577 | 0.75309 | 0.87060 | 0.91618 | 0.16860 |
| 8 | 0.35165 | 0.30513 | 0.18607 | 0.88601 | 0.77681 | 0.88244 | 0.92619 | 0.15290 |
| 9 | 0.32047 | 0.27521 | 0.18104 | 0.89522 | 0.79601 | 0.89201 | 0.92690 | 0.14573 |
| 10 | 0.28248 | 0.23936 | 0.17249 | 0.91365 | 0.82543 | 0.91096 | 0.94461 | 0.11536 |
| 11 | 0.34527 | 0.30338 | 0.16759 | 0.92389 | 0.85795 | 0.92363 | 0.95094 | 0.10273 |
| 12 | 0.31301 | 0.27194 | 0.16427 | 0.93106 | 0.86748 | 0.93111 | 0.95847 | 0.09044 |
| 13 | 0.27317 | 0.23316 | 0.16007 | 0.94505 | 0.89915 | 0.94522 | 0.96563 | 0.07338 |
| 14 | 0.24154 | 0.20257 | 0.15584 | 0.95597 | 0.91876 | 0.95609 | 0.97745 | 0.05358 |
| 15 | 0.21979 | 0.18189 | 0.15162 | 0.95768 | 0.92482 | 0.95777 | 0.97515 | 0.05461 |
| 16 | 0.18386 | 0.14699 | 0.14749 | 0.96894 | 0.94216 | 0.96893 | 0.98473 | 0.03754 |
| 17 | 0.16289 | 0.12690 | 0.14395 | 0.97509 | 0.95355 | 0.97519 | 0.98624 | 0.03140 |
| 18 | 0.13954 | 0.10510 | 0.13778 | 0.97918 | 0.96051 | 0.97922 | 0.98741 | 0.02730 |
| 19 | 0.13155 | 0.09817 | 0.13352 | 0.97952 | 0.96007 | 0.97951 | 0.98820 | 0.02594 |
| 20 | 0.12153 | 0.08873 | 0.13120 | 0.97645 | 0.95575 | 0.97652 | 0.98766 | 0.02935 |


### Validation scores

| Epoch | CE | Ordinal | Accuracy | Macro-F1 | Weighted-F1 | QWK | Grade MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.62426 | 0.24906 | 0.75956 | 0.42866 | 0.71252 | 0.78828 | 0.34153 |
| 2 | 0.56488 | 0.23472 | 0.76503 | 0.44423 | 0.72328 | 0.81134 | 0.32240 |
| 3 | 0.51851 | 0.22620 | 0.79235 | 0.54920 | 0.76704 | 0.85890 | 0.27322 |
| 4 | 0.49417 | 0.21963 | 0.81694 | 0.63301 | 0.80682 | 0.88760 | 0.23770 |
| 5 | 0.51171 | 0.21510 | 0.81148 | 0.61345 | 0.79747 | 0.88152 | 0.24044 |
| 6 | 0.49071 | 0.21075 | 0.81421 | 0.63504 | 0.80398 | 0.88749 | 0.23497 |
| 7 | 0.49273 | 0.20854 | 0.81967 | 0.66421 | 0.81665 | 0.89313 | 0.22951 |
| 8 | 0.47721 | 0.20471 | 0.82240 | 0.66003 | 0.81722 | 0.89701 | 0.22404 |
| 9 | 0.48682 | 0.20172 | 0.82514 | 0.68179 | 0.82564 | 0.88984 | 0.23224 |
| 10 | 0.49042 | 0.19954 | 0.81967 | 0.66923 | 0.81919 | 0.89917 | 0.22404 |
| 11 | 0.53907 | 0.20021 | 0.81421 | 0.69235 | 0.81926 | 0.89251 | 0.24317 |
| 12 | 0.51501 | 0.19546 | 0.83333 | 0.69970 | 0.83434 | 0.90609 | 0.21038 |
| 13 | 0.53696 | 0.19695 | 0.83880 | 0.70422 | 0.83908 | 0.90711 | 0.20492 |
| 14 | 0.52513 | 0.19516 | 0.81694 | 0.67693 | 0.81937 | 0.89625 | 0.23497 |
| 15 | 0.55184 | 0.19212 | 0.82787 | 0.70131 | 0.83078 | 0.89722 | 0.22678 |
| 16 | 0.58854 | 0.19052 | 0.83060 | 0.70579 | 0.83308 | 0.89924 | 0.22131 |
| 17 | 0.58019 | 0.18832 | 0.82787 | 0.70208 | 0.83059 | 0.89502 | 0.22951 |
| 18 | 0.59348 | 0.18869 | 0.80328 | 0.66112 | 0.80739 | 0.88282 | 0.26503 |
| 19 | 0.62824 | 0.19094 | 0.81967 | 0.68835 | 0.82198 | 0.89096 | 0.24044 |
| 20 | 0.64636 | 0.18617 | 0.82240 | 0.69402 | 0.82572 | 0.89404 | 0.23497 |


### Selection, imbalance schedule, and errors

HC wrong requires an incorrect argmax and maximum raw softmax probability ≥0.90.

| Epoch | Best through epoch | Updated | Weights active | Optimizer steps | Backbone LR | Head LR | Train correct | Train wrong | Train over | Train under | Train HC wrong | Val correct | Val wrong | Val over | Val under | Val HC wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | True | False | 92 | 0.00001 | 0.00100 | 2064 | 866 | 238 | 628 | 11 | 278 | 88 | 24 | 64 | 0 |
| 2 | 2 | True | False | 92 | 0.00001 | 0.00100 | 2322 | 608 | 162 | 446 | 12 | 280 | 86 | 29 | 57 | 1 |
| 3 | 3 | True | False | 92 | 0.00001 | 0.00100 | 2410 | 520 | 155 | 365 | 8 | 290 | 76 | 33 | 43 | 1 |
| 4 | 4 | True | False | 92 | 0.00001 | 0.00100 | 2436 | 494 | 171 | 323 | 8 | 299 | 67 | 32 | 35 | 2 |
| 5 | 4 | False | False | 92 | 0.00001 | 0.00100 | 2472 | 458 | 165 | 293 | 11 | 297 | 69 | 27 | 42 | 10 |
| 6 | 6 | True | False | 92 | 0.00001 | 0.00100 | 2530 | 400 | 165 | 235 | 11 | 298 | 68 | 32 | 36 | 10 |
| 7 | 7 | True | False | 92 | 0.00001 | 0.00100 | 2566 | 364 | 153 | 211 | 10 | 300 | 66 | 32 | 34 | 14 |
| 8 | 7 | False | False | 92 | 0.00001 | 0.00100 | 2596 | 334 | 138 | 196 | 5 | 301 | 65 | 33 | 32 | 11 |
| 9 | 9 | True | False | 92 | 0.00001 | 0.00100 | 2623 | 307 | 129 | 178 | 10 | 302 | 64 | 31 | 33 | 13 |
| 10 | 9 | False | False | 92 | 0.00001 | 0.00100 | 2677 | 253 | 107 | 146 | 11 | 300 | 66 | 31 | 35 | 15 |
| 11 | 11 | True | True | 92 | 0.00001 | 0.00100 | 2707 | 223 | 107 | 116 | 7 | 298 | 68 | 37 | 31 | 17 |
| 12 | 12 | True | True | 92 | 0.00001 | 0.00100 | 2728 | 202 | 100 | 102 | 11 | 305 | 61 | 30 | 31 | 20 |
| 13 | 13 | True | True | 92 | 0.00001 | 0.00100 | 2769 | 161 | 76 | 85 | 4 | 307 | 59 | 26 | 33 | 23 |
| 14 | 13 | False | True | 92 | 0.00001 | 0.00100 | 2801 | 129 | 70 | 59 | 12 | 299 | 67 | 36 | 31 | 19 |
| 15 | 13 | False | True | 92 | 0.00001 | 0.00100 | 2806 | 124 | 56 | 68 | 4 | 303 | 63 | 32 | 31 | 23 |
| 16 | 16 | True | True | 92 | 0.00001 | 0.00100 | 2839 | 91 | 40 | 51 | 5 | 304 | 62 | 30 | 32 | 23 |
| 17 | 16 | False | True | 92 | 0.00001 | 0.00100 | 2857 | 73 | 39 | 34 | 6 | 303 | 63 | 32 | 31 | 22 |
| 18 | 16 | False | True | 92 | 0.00001 | 0.00100 | 2869 | 61 | 35 | 26 | 7 | 294 | 72 | 43 | 29 | 25 |
| 19 | 16 | False | True | 92 | 0.00001 | 0.00100 | 2870 | 60 | 26 | 34 | 10 | 300 | 66 | 35 | 31 | 27 |
| 20 | 16 | False | True | 92 | 0.00001 | 0.00100 | 2861 | 69 | 38 | 31 | 9 | 301 | 65 | 34 | 31 | 27 |


### Resources

| Epoch | Train s | Val s | Train images/s | Val images/s | Train input s | Val input s | GPU allocated B | GPU reserved B | Process peak RSS B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 559.68691 | 0.64834 | 5.23507 | 564.52050 | 550.06126 | 0.30575 | 2476794880 | 2745171968 | 1738670080 |
| 2 | 550.02425 | 0.45380 | 5.32704 | 806.52079 | 542.00940 | 0.24144 | 2476794880 | 2745171968 | 1988501504 |
| 3 | 562.83936 | 0.43346 | 5.20575 | 844.37497 | 554.71584 | 0.21092 | 2476794880 | 2745171968 | 2028228608 |
| 4 | 582.83110 | 0.46230 | 5.02719 | 791.69016 | 574.87519 | 0.25942 | 2476794880 | 2745171968 | 2032386048 |
| 5 | 577.98957 | 0.47998 | 5.06930 | 762.52632 | 569.73449 | 0.24128 | 2476794880 | 2745171968 | 2032586752 |
| 6 | 559.41336 | 0.45603 | 5.23763 | 802.58278 | 551.27283 | 0.24413 | 2476794880 | 2745171968 | 2032586752 |
| 7 | 571.94099 | 0.49047 | 5.12291 | 746.22090 | 563.79605 | 0.26737 | 2476794880 | 2745171968 | 2032586752 |
| 8 | 830.08175 | 0.46060 | 3.52977 | 794.60818 | 821.87706 | 0.18814 | 2476794880 | 2745171968 | 2032586752 |
| 9 | 859.80906 | 0.43765 | 3.40773 | 836.27652 | 851.64171 | 0.22045 | 2476794880 | 2745171968 | 2032586752 |
| 10 | 970.25210 | 0.41776 | 3.01983 | 876.09953 | 961.96148 | 0.21285 | 2476794880 | 2745171968 | 2037899264 |
| 11 | 1031.22158 | 0.43570 | 2.84129 | 840.03079 | 1023.05122 | 0.23745 | 2476794880 | 2747269120 | 2037899264 |
| 12 | 815.97859 | 0.41133 | 3.59078 | 889.79784 | 808.05534 | 0.19980 | 2476794880 | 2747269120 | 2037899264 |
| 13 | 634.30505 | 0.42521 | 4.61923 | 860.74343 | 626.50333 | 0.21538 | 2476794880 | 2747269120 | 2037899264 |
| 14 | 662.64190 | 0.54987 | 4.42169 | 665.61647 | 654.81811 | 0.33034 | 2476794880 | 2747269120 | 2037899264 |
| 15 | 750.50838 | 0.45137 | 3.90402 | 810.86289 | 742.55664 | 0.23624 | 2476794880 | 2747269120 | 2037899264 |
| 16 | 702.05785 | 0.47894 | 4.17345 | 764.18037 | 693.66216 | 0.24514 | 2476794880 | 2747269120 | 2037899264 |
| 17 | 746.19027 | 0.53141 | 3.92661 | 688.73121 | 737.97153 | 0.29567 | 2476794880 | 2747269120 | 2037899264 |
| 18 | 921.87255 | 0.51575 | 3.17831 | 709.64658 | 913.37703 | 0.28155 | 2476794880 | 2747269120 | 2037899264 |
| 19 | 740.55845 | 0.41787 | 3.95647 | 875.86756 | 732.03029 | 0.19087 | 2476794880 | 2747269120 | 2037899264 |
| 20 | 795.79206 | 0.63273 | 3.68187 | 578.44160 | 780.12693 | 0.39973 | 2476794880 | 2747269120 | 2040758272 |


Recorded train/validation phases sum to 4.01 h and 0.16 min. These totals exclude any time outside the recorded phases. RSS is the parent process peak and does not account for all worker memory. Shared compute and distinct input transport confound throughput.

### Recorded input cache counters

| Epoch | train rgb hits | train rgb misses | validation cache reads or rgb hits | validation rgb misses |
| --- | --- | --- | --- | --- |
| 1 | 0 | 2930 | 366 | 0 |
| 2 | 0 | 2930 | 366 | 0 |
| 3 | 0 | 2930 | 366 | 0 |
| 4 | 1 | 2929 | 366 | 0 |
| 5 | 0 | 2930 | 366 | 0 |
| 6 | 0 | 2930 | 366 | 0 |
| 7 | 0 | 2930 | 366 | 0 |
| 8 | 0 | 2930 | 366 | 0 |
| 9 | 0 | 2930 | 366 | 0 |
| 10 | 0 | 2930 | 366 | 0 |
| 11 | 0 | 2930 | 366 | 0 |
| 12 | 1 | 2929 | 366 | 0 |
| 13 | 0 | 2930 | 366 | 0 |
| 14 | 0 | 2930 | 366 | 0 |
| 15 | 0 | 2930 | 366 | 0 |
| 16 | 0 | 2930 | 366 | 0 |
| 17 | 0 | 2930 | 366 | 0 |
| 18 | 0 | 2930 | 366 | 0 |
| 19 | 0 | 2930 | 366 | 0 |
| 20 | 0 | 2930 | 366 | 0 |


## Figures

[APTOS2019 figures and captions](../../../docs/E5_E6_VALIDATION.md#aptos2019-figure-suite)

Recorded statistics are descriptive for this frozen validation split. The report contains no sample-level predictions.
