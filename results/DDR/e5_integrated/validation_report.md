# E5 DDR frozen validation

10,018 training and 1,252 validation images. The run completed 20 epochs. The selected checkpoint was epoch **18**.

[E5–E6 comparison and figure index](../../../docs/E5_E6_VALIDATION.md) · [Full aggregate evidence, configuration, and hashes](validation_evidence.json)

## Run configuration and provenance

| Field | Recorded value |
| --- | --- |
| Run identity | foundation_e5/DDR/integrated_seed42_b32_20_cached/run_retry1 |
| Architecture | ConvNeXt-Tiny with five-class categorical and ordered-threshold ordinal heads |
| Foundation model / revision | facebook/convnext-tiny-224 / 6166b7613034066690a621d8bf25ffdf181a34f0 |
| Seed / epochs | 42 / 20 |
| Micro-batch × accumulation → effective batch | 32 × 1 → 32 |
| Optimizer / LR backbone / LR head / decay / clip | AdamW / 1e-05 / 0.001 / 0.01 / 1.0 |
| Precision / device / scheduler | bf16 / cuda / none in runner |
| Data transport, train | B1 EXIF-transposed RGB then fresh train-only augmentation then official processor |
| Data transport, validation | verified official float32 tensor cache |
| Workers / persistent / prefetch / pin | 2 / True / 2 / True |
| Train support grades 0–4 | [5014, 504, 3581, 189, 730] |
| Validation support grades 0–4 | [626, 63, 448, 24, 91] |
| Checkpoint selection | max validation macro-F1; tie lower validation CE; tie earliest epoch |
| Parameter counts | backbone_total: 27,820,128, backbone_trainable: 27,820,128, head_total: 4,618, total_trainable: 27,824,746 |
| Versions | torch 2.7.1+cu126, transformers 4.57.3, pillow 11.1.0, numpy 1.26.4 |
| Source commit | 4c9ab9d403ec1d9384ef86dc6de14378b68a77fe |
| Frozen manifest SHA-256 | 7bc3e3916e012bc46fcd9cd69aa469e0daf3229e1273f1d8ecb6054899bf6eb1 |
| Manifest sidecar SHA-256 | 56270be9a922565e7b4f4bc192fafcc6ebb53110904b3942703f273aa20ae08c |
| Ordered train/validation identity SHA-256 | b3dc145d12e4554f9210f581f04d6cb06b7ecceb5812b793c5ba9793d244503b |
| Processor SHA-256 / effective processor SHA-256 | fd575b890da5a949493e1d1e7a70bfcb9e4b99fe444004d2dbfa253add254741 / dd1700334d237761807778002bac2692337179f1ff24e0519fd67e74090bd5fd |
| Official weight SHA-256 | 1fe15b2eba9e355de14c4e244e6365bf61ffef84da7d3442c0afbb2d8564f477 |
| categorical | ordinary CE; train-only inverse-sqrt weights from epoch 11 if deferred |
| deferred weighting | True |
| ordinal | True |
| ordinal coefficient | 0.25000 |
| validation loss | ordinary categorical CE for every arm |
| weight start epoch | 11 |
| weights | [0.7297673426617228, 2.301766797811434, 0.8635240327782341, 3.7587694410119923, 1.912560400091831] |
| Source best.pt | 7dd7488013b7af5ad4565052b47d203c8077d8203dd5520d5b20b0fe592fbe00 (334,190,379 bytes) |
| Source last.pt | 384f8112c8e503e509f65409a187115bdeb403baf55cc2ecdd7737a8b3d70cb8 (334,195,499 bytes) |


Processor, augmentation, optimizer, class balance, source hashes, and the complete transport configuration are in the JSON. The original run stores configuration in `run.json`. The supplied checkpoint sidecars match the recorded checkpoint digests and byte sizes.

## Best checkpoint metrics

| Metric | Value | Source |
| --- | --- | --- |
| Accuracy | 0.82348 | Recorded |
| Macro-F1 | 0.65320 | Recorded |
| Weighted-F1 | 0.81819 | Recorded |
| QWK | 0.85716 | Recorded |
| Grade MAE | 0.26677 | Recorded |
| Categorical CE | 0.88551 | Recorded |
| Macro precision | 0.66214 | Matrix-derived |
| Macro recall | 0.64699 | Matrix-derived |
| correct | 1031 | Recorded |
| wrong | 221 | Recorded |
| undergraded | 135 | Recorded |
| overgraded | 86 | Recorded |
| mean overgrade magnitude | 1.45349 | Recorded |
| mean undergrade magnitude | 1.54815 | Recorded |
| high confidence wrong | 118 | Recorded |
| high confidence wrong rate all | 0.09425 | Recorded |
| high confidence wrong fraction of wrong | 0.53394 | Recorded |
| Adjacent mistakes (absolute grade difference 1) | 109 | Matrix-derived |
| Non-adjacent mistakes (absolute grade difference ≥2) | 112 | Matrix-derived |


### Confusion matrix

Rows are true grades and columns are predicted grades. Counts include the diagonal. Percentages are normalized within each true grade.

| True grade | 0 | 1 | 2 | 3 | 4 | Support |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 588 | 10 | 28 | 0 | 0 | 626 |
| 1 | 24 | 12 | 26 | 0 | 1 | 63 |
| 2 | 58 | 26 | 346 | 9 | 9 | 448 |
| 3 | 0 | 0 | 8 | 13 | 3 | 24 |
| 4 | 0 | 0 | 16 | 3 | 72 | 91 |


| True grade | 0 (%) | 1 (%) | 2 (%) | 3 (%) | 4 (%) |
| --- | --- | --- | --- | --- | --- |
| 0 | 93.93 | 1.60 | 4.47 | 0.00 | 0.00 |
| 1 | 38.10 | 19.05 | 41.27 | 0.00 | 1.59 |
| 2 | 12.95 | 5.80 | 77.23 | 2.01 | 2.01 |
| 3 | 0.00 | 0.00 | 33.33 | 54.17 | 12.50 |
| 4 | 0.00 | 0.00 | 17.58 | 3.30 | 79.12 |


Largest off-diagonal cells (true→predicted) were 2→0 (58), 0→2 (28), 2→1 (26), 1→2 (26). The grade 1 row was `[24, 12, 26, 0, 1]`; the grade 3 row was `[0, 0, 8, 13, 3]`.

### Per-grade metrics

Support, TP, precision, recall, and F1 are recorded and independently recomputed from the matrix. FP and FN are matrix-derived. Per-grade MAE was recorded in the CSV.

| Grade | Support | TP | FP | FN | Precision | Recall | F1 | Grade MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 626 | 588 | 82 | 38 | 0.87761 | 0.93930 | 0.90741 | 0.10543 |
| 1 | 63 | 12 | 36 | 51 | 0.25000 | 0.19048 | 0.21622 | 0.84127 |
| 2 | 448 | 346 | 78 | 102 | 0.81604 | 0.77232 | 0.79358 | 0.37723 |
| 3 | 24 | 13 | 12 | 11 | 0.52000 | 0.54167 | 0.53061 | 0.45833 |
| 4 | 91 | 72 | 13 | 19 | 0.84706 | 0.79121 | 0.81818 | 0.38462 |


## All 20 epochs

Markdown rounds values for readability. The JSON preserves the precision and full field coverage of the export. Per-grade MAE fields removed from the JSON by the export filter are preserved separately from the CSV.

### Training scores

| Epoch | Total loss | Categorical | Ordinal | Accuracy | Macro-F1 | Weighted-F1 | QWK | Grade MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.82985 | 0.75045 | 0.31761 | 0.71731 | 0.39632 | 0.68527 | 0.65184 | 0.51338 |
| 2 | 0.64164 | 0.57582 | 0.26328 | 0.79048 | 0.48687 | 0.76267 | 0.78493 | 0.35476 |
| 3 | 0.57136 | 0.51136 | 0.24002 | 0.81473 | 0.54849 | 0.79008 | 0.81853 | 0.30675 |
| 4 | 0.51637 | 0.46105 | 0.22125 | 0.83280 | 0.58992 | 0.81082 | 0.84113 | 0.27261 |
| 5 | 0.46370 | 0.41268 | 0.20408 | 0.84708 | 0.61520 | 0.82722 | 0.86259 | 0.24286 |
| 6 | 0.42013 | 0.37249 | 0.19055 | 0.86634 | 0.65173 | 0.84949 | 0.88892 | 0.20383 |
| 7 | 0.38302 | 0.33863 | 0.17756 | 0.88012 | 0.68824 | 0.86730 | 0.90452 | 0.17868 |
| 8 | 0.34644 | 0.30479 | 0.16657 | 0.89050 | 0.71252 | 0.87956 | 0.91693 | 0.15871 |
| 9 | 0.29861 | 0.26041 | 0.15277 | 0.90627 | 0.75371 | 0.89812 | 0.93097 | 0.13366 |
| 10 | 0.27245 | 0.23576 | 0.14673 | 0.91415 | 0.78035 | 0.90852 | 0.93944 | 0.11978 |
| 11 | 0.31967 | 0.28521 | 0.13781 | 0.92364 | 0.83292 | 0.92480 | 0.94955 | 0.10371 |
| 12 | 0.27233 | 0.23981 | 0.13011 | 0.93252 | 0.85205 | 0.93372 | 0.95759 | 0.08924 |
| 13 | 0.22551 | 0.19436 | 0.12459 | 0.94779 | 0.89163 | 0.94830 | 0.96460 | 0.07117 |
| 14 | 0.18554 | 0.15624 | 0.11721 | 0.95638 | 0.91216 | 0.95700 | 0.97153 | 0.05899 |
| 15 | 0.16161 | 0.13396 | 0.11059 | 0.96796 | 0.93221 | 0.96816 | 0.97904 | 0.04332 |
| 16 | 0.12772 | 0.10161 | 0.10443 | 0.97315 | 0.95065 | 0.97327 | 0.98069 | 0.03813 |
| 17 | 0.10823 | 0.08359 | 0.09858 | 0.97814 | 0.96005 | 0.97819 | 0.98480 | 0.03055 |
| 18 | 0.08932 | 0.06665 | 0.09070 | 0.98543 | 0.97418 | 0.98547 | 0.98977 | 0.02046 |
| 19 | 0.06522 | 0.04446 | 0.08305 | 0.98902 | 0.98241 | 0.98903 | 0.99120 | 0.01657 |
| 20 | 0.06583 | 0.04540 | 0.08175 | 0.98732 | 0.97960 | 0.98735 | 0.99028 | 0.01867 |


### Validation scores

| Epoch | CE | Ordinal | Accuracy | Macro-F1 | Weighted-F1 | QWK | Grade MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.64305 | 0.28388 | 0.76518 | 0.45140 | 0.73452 | 0.73979 | 0.40735 |
| 2 | 0.59197 | 0.26685 | 0.78355 | 0.52183 | 0.75671 | 0.77373 | 0.37141 |
| 3 | 0.56192 | 0.24688 | 0.79153 | 0.55431 | 0.76907 | 0.79053 | 0.35543 |
| 4 | 0.55320 | 0.23850 | 0.79952 | 0.56200 | 0.77545 | 0.80439 | 0.33946 |
| 5 | 0.56805 | 0.25010 | 0.80032 | 0.60272 | 0.78626 | 0.80730 | 0.33147 |
| 6 | 0.57908 | 0.23048 | 0.80112 | 0.59471 | 0.78950 | 0.80846 | 0.32588 |
| 7 | 0.54177 | 0.21364 | 0.82827 | 0.58217 | 0.80545 | 0.83797 | 0.27875 |
| 8 | 0.54130 | 0.21175 | 0.82827 | 0.62035 | 0.81259 | 0.83933 | 0.27716 |
| 9 | 0.55605 | 0.21084 | 0.82668 | 0.62088 | 0.81010 | 0.83847 | 0.27955 |
| 10 | 0.60349 | 0.20568 | 0.82748 | 0.60148 | 0.80554 | 0.83219 | 0.28355 |
| 11 | 0.64273 | 0.20063 | 0.77875 | 0.64018 | 0.79537 | 0.84312 | 0.30751 |
| 12 | 0.64361 | 0.20588 | 0.80112 | 0.63216 | 0.80181 | 0.83266 | 0.30272 |
| 13 | 0.66981 | 0.21059 | 0.81949 | 0.64173 | 0.81179 | 0.84587 | 0.28035 |
| 14 | 0.69924 | 0.20089 | 0.80911 | 0.64416 | 0.81119 | 0.84945 | 0.28115 |
| 15 | 0.74815 | 0.19873 | 0.81310 | 0.64321 | 0.80911 | 0.83217 | 0.29153 |
| 16 | 0.83041 | 0.20315 | 0.78994 | 0.62871 | 0.79842 | 0.83642 | 0.30431 |
| 17 | 0.84322 | 0.20497 | 0.81709 | 0.63610 | 0.81346 | 0.85242 | 0.27316 |
| 18 | 0.88551 | 0.20443 | 0.82348 | 0.65320 | 0.81819 | 0.85716 | 0.26677 |
| 19 | 0.97169 | 0.20688 | 0.80751 | 0.63211 | 0.80340 | 0.83872 | 0.29633 |
| 20 | 1.05798 | 0.20853 | 0.80990 | 0.64021 | 0.80672 | 0.83381 | 0.29792 |


### Selection, imbalance schedule, and errors

HC wrong requires an incorrect argmax and maximum raw softmax probability ≥0.90.

| Epoch | Best through epoch | Updated | Weights active | Optimizer steps | Backbone LR | Head LR | Train correct | Train wrong | Train over | Train under | Train HC wrong | Val correct | Val wrong | Val over | Val under | Val HC wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | True | False | 314 | 0.00001 | 0.00100 | 7186 | 2832 | 873 | 1959 | 47 | 958 | 294 | 91 | 203 | 22 |
| 2 | 2 | True | False | 314 | 0.00001 | 0.00100 | 7919 | 2099 | 791 | 1308 | 135 | 981 | 271 | 86 | 185 | 31 |
| 3 | 3 | True | False | 314 | 0.00001 | 0.00100 | 8162 | 1856 | 744 | 1112 | 174 | 991 | 261 | 120 | 141 | 28 |
| 4 | 4 | True | False | 314 | 0.00001 | 0.00100 | 8343 | 1675 | 696 | 979 | 162 | 1001 | 251 | 99 | 152 | 45 |
| 5 | 5 | True | False | 314 | 0.00001 | 0.00100 | 8486 | 1532 | 623 | 909 | 171 | 1002 | 250 | 149 | 101 | 41 |
| 6 | 5 | False | False | 314 | 0.00001 | 0.00100 | 8679 | 1339 | 574 | 765 | 152 | 1003 | 249 | 149 | 100 | 46 |
| 7 | 5 | False | False | 314 | 0.00001 | 0.00100 | 8817 | 1201 | 535 | 666 | 156 | 1037 | 215 | 85 | 130 | 69 |
| 8 | 8 | True | False | 314 | 0.00001 | 0.00100 | 8921 | 1097 | 459 | 638 | 123 | 1037 | 215 | 107 | 108 | 58 |
| 9 | 9 | True | False | 314 | 0.00001 | 0.00100 | 9079 | 939 | 419 | 520 | 113 | 1035 | 217 | 88 | 129 | 78 |
| 10 | 9 | False | False | 314 | 0.00001 | 0.00100 | 9158 | 860 | 352 | 508 | 105 | 1036 | 216 | 102 | 114 | 91 |
| 11 | 11 | True | True | 314 | 0.00001 | 0.00100 | 9253 | 765 | 369 | 396 | 57 | 975 | 277 | 149 | 128 | 56 |
| 12 | 11 | False | True | 314 | 0.00001 | 0.00100 | 9342 | 676 | 307 | 369 | 53 | 1003 | 249 | 135 | 114 | 67 |
| 13 | 13 | True | True | 314 | 0.00001 | 0.00100 | 9495 | 523 | 243 | 280 | 55 | 1026 | 226 | 105 | 121 | 92 |
| 14 | 14 | True | True | 314 | 0.00001 | 0.00100 | 9581 | 437 | 213 | 224 | 38 | 1013 | 239 | 108 | 131 | 82 |
| 15 | 14 | False | True | 314 | 0.00001 | 0.00100 | 9697 | 321 | 151 | 170 | 42 | 1018 | 234 | 116 | 118 | 100 |
| 16 | 14 | False | True | 314 | 0.00001 | 0.00100 | 9749 | 269 | 126 | 143 | 42 | 989 | 263 | 138 | 125 | 106 |
| 17 | 14 | False | True | 314 | 0.00001 | 0.00100 | 9799 | 219 | 94 | 125 | 32 | 1023 | 229 | 100 | 129 | 116 |
| 18 | 18 | True | True | 314 | 0.00001 | 0.00100 | 9872 | 146 | 62 | 84 | 32 | 1031 | 221 | 86 | 135 | 118 |
| 19 | 18 | False | True | 314 | 0.00001 | 0.00100 | 9908 | 110 | 49 | 61 | 17 | 1011 | 241 | 131 | 110 | 135 |
| 20 | 18 | False | True | 314 | 0.00001 | 0.00100 | 9891 | 127 | 67 | 60 | 29 | 1014 | 238 | 142 | 96 | 146 |


### Resources

| Epoch | Train s | Val s | Train images/s | Val images/s | Train input s | Val input s | GPU allocated B | GPU reserved B | Process peak RSS B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2318.29141 | 1.53584 | 4.32129 | 815.18853 | 2287.90009 | 0.69343 | 2476794880 | 2766143488 | 1797210112 |
| 2 | 2211.79078 | 1.26833 | 4.52936 | 987.12503 | 2183.64725 | 0.68430 | 2476794880 | 2766143488 | 2050965504 |
| 3 | 2221.25528 | 1.39383 | 4.51006 | 898.24216 | 2192.42222 | 0.74363 | 2476794880 | 2766143488 | 2050965504 |
| 4 | 2433.91342 | 1.85163 | 4.11601 | 676.16050 | 2404.31857 | 1.09796 | 2476794880 | 2766143488 | 2050965504 |
| 5 | 2512.16049 | 1.20232 | 3.98780 | 1041.32057 | 2477.07898 | 0.31219 | 2476794880 | 2766143488 | 2050965504 |
| 6 | 1962.60269 | 1.08575 | 5.10445 | 1153.12249 | 1930.36722 | 0.47471 | 2476794880 | 2766143488 | 2050965504 |
| 7 | 2017.30948 | 1.25446 | 4.96602 | 998.03901 | 1987.32247 | 0.58160 | 2476794880 | 2766143488 | 2050965504 |
| 8 | 2373.69463 | 1.11952 | 4.22042 | 1118.33779 | 2340.38765 | 0.55700 | 2476794880 | 2766143488 | 2050965504 |
| 9 | 2073.23600 | 1.19235 | 4.83206 | 1050.02473 | 2043.95158 | 0.51926 | 2476794880 | 2766143488 | 2050965504 |
| 10 | 2108.87800 | 1.20577 | 4.75039 | 1038.34197 | 2041.37659 | 0.47867 | 2476794880 | 2766143488 | 2050965504 |
| 11 | 1965.97779 | 1.06470 | 5.09568 | 1175.91828 | 1932.63650 | 0.49425 | 2476794880 | 2766143488 | 2050965504 |
| 12 | 1961.59943 | 1.16874 | 5.10706 | 1071.24331 | 1931.41828 | 0.46473 | 2476794880 | 2766143488 | 2050965504 |
| 13 | 2018.59599 | 1.12172 | 4.96286 | 1116.14094 | 1990.19040 | 0.56323 | 2476794880 | 2766143488 | 2050965504 |
| 14 | 1958.41414 | 1.10076 | 5.11536 | 1137.39994 | 1934.28376 | 0.50881 | 2476794880 | 2766143488 | 2050965504 |
| 15 | 2058.65622 | 1.16221 | 4.86628 | 1077.25560 | 2033.16979 | 0.52619 | 2476794880 | 2766143488 | 2050965504 |
| 16 | 2250.58807 | 1.59592 | 4.45128 | 784.49885 | 2193.82740 | 0.17183 | 2476794880 | 2766143488 | 2055729152 |
| 17 | 2516.03726 | 1.87455 | 3.98166 | 667.89416 | 2389.10152 | 0.19743 | 2476794880 | 2766143488 | 2057560064 |
| 18 | 2409.23090 | 1.61236 | 4.15817 | 776.50267 | 2325.68319 | 0.82991 | 2476794880 | 2766143488 | 2057560064 |
| 19 | 2366.17485 | 1.95354 | 4.23384 | 640.88744 | 2282.92685 | 0.09878 | 2476794880 | 2766143488 | 2057560064 |
| 20 | 2463.44112 | 1.18378 | 4.06667 | 1057.62513 | 2383.92719 | 0.58035 | 2476794880 | 2766143488 | 2057560064 |


Recorded train/validation phases sum to 12.28 h and 0.45 min. These totals exclude any time outside the recorded phases. RSS is the parent process peak and does not account for all worker memory. Shared compute and distinct input transport confound throughput.

### Recorded input cache counters

| Epoch | train rgb hits | train rgb misses | validation cache reads or rgb hits | validation rgb misses |
| --- | --- | --- | --- | --- |
| 1 | 0 | 10018 | 1252 | 0 |
| 2 | 0 | 10018 | 1252 | 0 |
| 3 | 0 | 10018 | 1252 | 0 |
| 4 | 0 | 10018 | 1252 | 0 |
| 5 | 0 | 10018 | 1252 | 0 |
| 6 | 0 | 10018 | 1252 | 0 |
| 7 | 0 | 10018 | 1252 | 0 |
| 8 | 0 | 10018 | 1252 | 0 |
| 9 | 0 | 10018 | 1252 | 0 |
| 10 | 0 | 10018 | 1252 | 0 |
| 11 | 0 | 10018 | 1252 | 0 |
| 12 | 0 | 10018 | 1252 | 0 |
| 13 | 0 | 10018 | 1252 | 0 |
| 14 | 0 | 10018 | 1252 | 0 |
| 15 | 0 | 10018 | 1252 | 0 |
| 16 | 0 | 10018 | 1252 | 0 |
| 17 | 0 | 10018 | 1252 | 0 |
| 18 | 0 | 10018 | 1252 | 0 |
| 19 | 0 | 10018 | 1252 | 0 |
| 20 | 0 | 10018 | 1252 | 0 |


## Figures

[DDR figures and captions](../../../docs/E5_E6_VALIDATION.md#ddr-figure-suite)

Recorded statistics are descriptive for this frozen validation split. The report contains no sample-level predictions.
