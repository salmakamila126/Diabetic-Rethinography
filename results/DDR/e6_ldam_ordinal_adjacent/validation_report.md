# E6 DDR frozen validation

10,018 training and 1,252 validation images. The run completed 20 epochs. The selected checkpoint was epoch **12**.

[E5–E6 comparison and figure index](../../../docs/E5_E6_VALIDATION.md) · [Full aggregate evidence, configuration, and hashes](validation_evidence.json)

## Run configuration and provenance

| Field | Recorded value |
| --- | --- |
| Run identity | foundation_e6/DDR/ldam_ordinal_adjacent_seed42_b32_20/run |
| Architecture | ConvNeXt-Tiny with five-class categorical and ordered-threshold ordinal heads |
| Foundation model / revision | facebook/convnext-tiny-224 / 6166b7613034066690a621d8bf25ffdf181a34f0 |
| Seed / epochs | 42 / 20 |
| Micro-batch × accumulation → effective batch | 32 × 1 → 32 |
| Optimizer / LR backbone / LR head / decay / clip | AdamW / 1e-05 / 0.001 / 0.01 / 1.0 |
| Precision / device / scheduler | bf16 / cuda / none in runner |
| Data transport, train | verified mmap decoded RGB, stochastic B1 augmentation each epoch |
| Data transport, validation | uncached official processor |
| Workers / persistent / prefetch / pin | 2 / True / 2 / True |
| Train support grades 0–4 | [5014, 504, 3581, 189, 730] |
| Validation support grades 0–4 | [626, 63, 448, 24, 91] |
| Checkpoint selection | max validation macro-F1; tie lower validation CE; tie earliest epoch |
| Parameter counts | backbone_total: 27,820,128, backbone_trainable: 27,820,128, head_total: 4,618, total_trainable: 27,824,746 |
| Versions | torch 2.7.1+cu126, transformers 4.57.3, pillow 11.1.0, numpy 2.5.5 |
| Source commit | b3dd3ca01936f7d173cb646721271703b4e0308d |
| Frozen manifest SHA-256 | 7bc3e3916e012bc46fcd9cd69aa469e0daf3229e1273f1d8ecb6054899bf6eb1 |
| Manifest sidecar SHA-256 | 56270be9a922565e7b4f4bc192fafcc6ebb53110904b3942703f273aa20ae08c |
| Ordered train/validation identity SHA-256 | b3dc145d12e4554f9210f581f04d6cb06b7ecceb5812b793c5ba9793d244503b |
| Processor SHA-256 / effective processor SHA-256 | fd575b890da5a949493e1d1e7a70bfcb9e4b99fe444004d2dbfa253add254741 / dd1700334d237761807778002bac2692337179f1ff24e0519fd67e74090bd5fd |
| Official weight SHA-256 | 1fe15b2eba9e355de14c4e244e6365bf61ffef84da7d3442c0afbb2d8564f477 |
| adjacent coefficient | 0.10000 |
| adjacent pairs | [[0, 1], [1, 2]] |
| categorical | LDAM true-class margin scaled by 30 |
| effective number beta | 0.99990 |
| effective number weights | [0.45916403715459936, 3.683643705967695, 0.601521412230104, 9.67042083743533, 2.5718320197641646] |
| ldam margins | [0.22031261885883618, 0.39127114501832183, 0.23965383061583065, 0.5, 0.3566602752086369] |
| ordinal | True |
| ordinal coefficient | 0.25000 |
| validation loss | ordinary categorical CE on raw logits |
| weight start epoch | 11 |
| Source best.pt | 7e4bf74ced0daec7f078e7e441551ce40c96c2816eff990d3aafa860f7113df5 (334,176,555 bytes) |
| Source last.pt | a4fc9147b72df31e338391da5a3e8861b19f4ed6e9daec07955c9d7259b3bedd (334,197,547 bytes) |


Processor, augmentation, optimizer, class balance, source hashes, and the complete transport configuration are in the JSON. The original run stores configuration in `run.json`. The supplied checkpoint sidecars match the recorded checkpoint digests and byte sizes.

## Best checkpoint metrics

| Metric | Value | Source |
| --- | --- | --- |
| Accuracy | 0.79792 | Recorded |
| Macro-F1 | 0.66246 | Recorded |
| Weighted-F1 | 0.81229 | Recorded |
| QWK | 0.84955 | Recorded |
| Grade MAE | 0.28355 | Recorded |
| Categorical CE | 0.92770 | Recorded |
| Macro precision | 0.70334 | Matrix-derived |
| Macro recall | 0.65202 | Matrix-derived |
| correct | 999 | Recorded |
| wrong | 253 | Recorded |
| undergraded | 137 | Recorded |
| overgraded | 116 | Recorded |
| mean overgrade magnitude | 1.36207 | Recorded |
| mean undergrade magnitude | 1.43796 | Recorded |
| high confidence wrong | 0 | Recorded |
| high confidence wrong rate all | 0.00000 | Recorded |
| high confidence wrong fraction of wrong | 0.00000 | Recorded |
| Adjacent mistakes (absolute grade difference 1) | 158 | Matrix-derived |
| Non-adjacent mistakes (absolute grade difference ≥2) | 95 | Matrix-derived |


### Confusion matrix

Rows are true grades and columns are predicted grades. Counts include the diagonal. Percentages are normalized within each true grade.

| True grade | 0 | 1 | 2 | 3 | 4 | Support |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 548 | 51 | 27 | 0 | 0 | 626 |
| 1 | 18 | 23 | 21 | 0 | 1 | 63 |
| 2 | 37 | 52 | 345 | 1 | 13 | 448 |
| 3 | 0 | 0 | 11 | 11 | 2 | 24 |
| 4 | 2 | 2 | 13 | 2 | 72 | 91 |


| True grade | 0 (%) | 1 (%) | 2 (%) | 3 (%) | 4 (%) |
| --- | --- | --- | --- | --- | --- |
| 0 | 87.54 | 8.15 | 4.31 | 0.00 | 0.00 |
| 1 | 28.57 | 36.51 | 33.33 | 0.00 | 1.59 |
| 2 | 8.26 | 11.61 | 77.01 | 0.22 | 2.90 |
| 3 | 0.00 | 0.00 | 45.83 | 45.83 | 8.33 |
| 4 | 2.20 | 2.20 | 14.29 | 2.20 | 79.12 |


Largest off-diagonal cells (true→predicted) were 2→1 (52), 0→1 (51), 2→0 (37), 0→2 (27). The grade 1 row was `[18, 23, 21, 0, 1]`; the grade 3 row was `[0, 0, 11, 11, 2]`.

### Per-grade metrics

Support, TP, precision, recall, and F1 are recorded and independently recomputed from the matrix. FP and FN are matrix-derived. Per-grade MAE was recorded in the CSV.

| Grade | Support | TP | FP | FN | Precision | Recall | F1 | Grade MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 626 | 548 | 57 | 78 | 0.90579 | 0.87540 | 0.89033 | 0.16773 |
| 1 | 63 | 23 | 105 | 40 | 0.17969 | 0.36508 | 0.24084 | 0.66667 |
| 2 | 448 | 345 | 72 | 103 | 0.82734 | 0.77009 | 0.79769 | 0.34152 |
| 3 | 24 | 11 | 3 | 13 | 0.78571 | 0.45833 | 0.57895 | 0.54167 |
| 4 | 91 | 72 | 16 | 19 | 0.81818 | 0.79121 | 0.80447 | 0.46154 |


## All 20 epochs

Markdown rounds values for readability. The JSON preserves the precision and full field coverage of the export. Per-grade MAE fields removed from the JSON by the export filter are preserved separately from the CSV.

### Training scores

| Epoch | Total loss | Categorical | Ordinal | Adjacent | Accuracy | Macro-F1 | Weighted-F1 | QWK | Grade MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 5.41681 | 5.33292 | 0.33068 | 0.01219 | 0.72070 | 0.42990 | 0.69502 | 0.67141 | 0.50629 |
| 2 | 4.22970 | 4.15990 | 0.27477 | 0.01103 | 0.79088 | 0.50742 | 0.76591 | 0.79010 | 0.35167 |
| 3 | 3.68641 | 3.62191 | 0.25430 | 0.00925 | 0.82002 | 0.57122 | 0.79724 | 0.82579 | 0.29796 |
| 4 | 3.39620 | 3.33558 | 0.23872 | 0.00942 | 0.83839 | 0.61806 | 0.81764 | 0.84771 | 0.26383 |
| 5 | 3.01807 | 2.96146 | 0.22309 | 0.00837 | 0.85855 | 0.66482 | 0.84035 | 0.87408 | 0.22430 |
| 6 | 2.77082 | 2.71728 | 0.21094 | 0.00800 | 0.86983 | 0.68844 | 0.85248 | 0.88840 | 0.20274 |
| 7 | 2.51320 | 2.46230 | 0.20052 | 0.00770 | 0.88491 | 0.71341 | 0.86940 | 0.90521 | 0.17409 |
| 8 | 2.35124 | 2.30211 | 0.19342 | 0.00767 | 0.89209 | 0.73135 | 0.87845 | 0.91723 | 0.15851 |
| 9 | 2.16336 | 2.11641 | 0.18492 | 0.00713 | 0.90188 | 0.76514 | 0.89122 | 0.92369 | 0.14454 |
| 10 | 1.98938 | 1.94464 | 0.17627 | 0.00678 | 0.90777 | 0.78125 | 0.89910 | 0.93156 | 0.13236 |
| 11 | 3.86740 | 3.82164 | 0.17542 | 0.01908 | 0.86684 | 0.76787 | 0.88327 | 0.92814 | 0.16161 |
| 12 | 3.29699 | 3.25386 | 0.16559 | 0.01737 | 0.88551 | 0.79570 | 0.89726 | 0.93856 | 0.13975 |
| 13 | 2.58446 | 2.54233 | 0.16298 | 0.01387 | 0.91136 | 0.84006 | 0.91812 | 0.94888 | 0.11210 |
| 14 | 2.15646 | 2.11530 | 0.15929 | 0.01339 | 0.91855 | 0.85863 | 0.92521 | 0.95239 | 0.10321 |
| 15 | 1.90468 | 1.86498 | 0.15505 | 0.00943 | 0.93282 | 0.87860 | 0.93646 | 0.95713 | 0.08904 |
| 16 | 1.63517 | 1.59626 | 0.15171 | 0.00984 | 0.94091 | 0.89968 | 0.94388 | 0.96089 | 0.07956 |
| 17 | 1.30988 | 1.26850 | 0.16302 | 0.00626 | 0.95318 | 0.92336 | 0.95471 | 0.96755 | 0.06488 |
| 18 | 1.20638 | 1.16772 | 0.15240 | 0.00561 | 0.95957 | 0.93403 | 0.96062 | 0.97198 | 0.05620 |
| 19 | 0.87478 | 0.83758 | 0.14694 | 0.00468 | 0.96736 | 0.94944 | 0.96788 | 0.97424 | 0.04861 |
| 20 | 0.79557 | 0.75969 | 0.14212 | 0.00346 | 0.97235 | 0.95983 | 0.97265 | 0.97781 | 0.04172 |


### Validation scores

| Epoch | CE | Ordinal | Accuracy | Macro-F1 | Weighted-F1 | QWK | Grade MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.24744 | 0.29075 | 0.76597 | 0.44972 | 0.73568 | 0.74293 | 0.40735 |
| 2 | 1.24883 | 0.27688 | 0.78115 | 0.54220 | 0.75223 | 0.76772 | 0.38099 |
| 3 | 1.18240 | 0.25784 | 0.79473 | 0.55091 | 0.77004 | 0.79379 | 0.34744 |
| 4 | 1.16527 | 0.24854 | 0.81470 | 0.60171 | 0.79118 | 0.82772 | 0.30911 |
| 5 | 1.22255 | 0.25277 | 0.78275 | 0.59450 | 0.76902 | 0.77805 | 0.37939 |
| 6 | 1.14300 | 0.23237 | 0.82029 | 0.63011 | 0.80228 | 0.83060 | 0.29712 |
| 7 | 1.07092 | 0.22856 | 0.82748 | 0.58454 | 0.80301 | 0.83778 | 0.28115 |
| 8 | 1.09109 | 0.22357 | 0.82268 | 0.59727 | 0.80379 | 0.83669 | 0.28914 |
| 9 | 1.06393 | 0.21686 | 0.82029 | 0.63642 | 0.80755 | 0.83057 | 0.29073 |
| 10 | 1.02214 | 0.21331 | 0.82188 | 0.62110 | 0.80565 | 0.83373 | 0.28834 |
| 11 | 1.04915 | 0.22090 | 0.75639 | 0.62548 | 0.78134 | 0.82237 | 0.34105 |
| 12 | 0.92770 | 0.21309 | 0.79792 | 0.66246 | 0.81229 | 0.84955 | 0.28355 |
| 13 | 0.90848 | 0.21711 | 0.80192 | 0.63240 | 0.80636 | 0.84523 | 0.29153 |
| 14 | 0.93429 | 0.21634 | 0.78514 | 0.62241 | 0.79534 | 0.82473 | 0.32348 |
| 15 | 0.87684 | 0.21773 | 0.77236 | 0.64760 | 0.79045 | 0.82196 | 0.32348 |
| 16 | 0.89507 | 0.21274 | 0.74042 | 0.62498 | 0.77338 | 0.83337 | 0.34185 |
| 17 | 0.74521 | 0.20695 | 0.82029 | 0.65917 | 0.81797 | 0.84081 | 0.27716 |
| 18 | 0.76362 | 0.21513 | 0.79313 | 0.64458 | 0.80547 | 0.85367 | 0.28594 |
| 19 | 0.75326 | 0.21368 | 0.81310 | 0.65178 | 0.81554 | 0.85068 | 0.27796 |
| 20 | 0.79428 | 0.22402 | 0.76837 | 0.63511 | 0.77671 | 0.79881 | 0.35543 |


### Selection, imbalance schedule, and errors

HC wrong requires an incorrect argmax and maximum raw softmax probability ≥0.90.

| Epoch | Best through epoch | Updated | Weights active | Optimizer steps | Backbone LR | Head LR | Train correct | Train wrong | Train over | Train under | Train HC wrong | Val correct | Val wrong | Val over | Val under | Val HC wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | True | False | 314 | 0.00001 | 0.00100 | 7220 | 2798 | 1033 | 1765 | 0 | 959 | 293 | 88 | 205 | 0 |
| 2 | 2 | True | False | 314 | 0.00001 | 0.00100 | 7923 | 2095 | 796 | 1299 | 0 | 978 | 274 | 52 | 222 | 0 |
| 3 | 3 | True | False | 314 | 0.00001 | 0.00100 | 8215 | 1803 | 700 | 1103 | 0 | 995 | 257 | 92 | 165 | 0 |
| 4 | 4 | True | False | 314 | 0.00001 | 0.00100 | 8399 | 1619 | 640 | 979 | 0 | 1020 | 232 | 103 | 129 | 0 |
| 5 | 4 | False | False | 314 | 0.00001 | 0.00100 | 8601 | 1417 | 581 | 836 | 0 | 980 | 272 | 199 | 73 | 0 |
| 6 | 6 | True | False | 314 | 0.00001 | 0.00100 | 8714 | 1304 | 568 | 736 | 0 | 1027 | 225 | 120 | 105 | 0 |
| 7 | 6 | False | False | 314 | 0.00001 | 0.00100 | 8865 | 1153 | 513 | 640 | 0 | 1036 | 216 | 72 | 144 | 0 |
| 8 | 6 | False | False | 314 | 0.00001 | 0.00100 | 8937 | 1081 | 447 | 634 | 0 | 1030 | 222 | 119 | 103 | 0 |
| 9 | 9 | True | False | 314 | 0.00001 | 0.00100 | 9035 | 983 | 420 | 563 | 0 | 1027 | 225 | 80 | 145 | 0 |
| 10 | 9 | False | False | 314 | 0.00001 | 0.00100 | 9094 | 924 | 378 | 546 | 0 | 1029 | 223 | 101 | 122 | 0 |
| 11 | 9 | False | True | 314 | 0.00001 | 0.00100 | 8684 | 1334 | 720 | 614 | 0 | 947 | 305 | 162 | 143 | 0 |
| 12 | 12 | True | True | 314 | 0.00001 | 0.00100 | 8871 | 1147 | 591 | 556 | 0 | 999 | 253 | 116 | 137 | 0 |
| 13 | 12 | False | True | 314 | 0.00001 | 0.00100 | 9130 | 888 | 454 | 434 | 0 | 1004 | 248 | 113 | 135 | 0 |
| 14 | 12 | False | True | 314 | 0.00001 | 0.00100 | 9202 | 816 | 427 | 389 | 0 | 983 | 269 | 148 | 121 | 0 |
| 15 | 12 | False | True | 314 | 0.00001 | 0.00100 | 9345 | 673 | 351 | 322 | 0 | 967 | 285 | 157 | 128 | 0 |
| 16 | 12 | False | True | 314 | 0.00001 | 0.00100 | 9426 | 592 | 289 | 303 | 0 | 927 | 325 | 194 | 131 | 0 |
| 17 | 12 | False | True | 314 | 0.00001 | 0.00100 | 9549 | 469 | 240 | 229 | 0 | 1027 | 225 | 102 | 123 | 0 |
| 18 | 12 | False | True | 314 | 0.00001 | 0.00100 | 9613 | 405 | 200 | 205 | 0 | 993 | 259 | 102 | 157 | 0 |
| 19 | 12 | False | True | 314 | 0.00001 | 0.00100 | 9691 | 327 | 161 | 166 | 0 | 1018 | 234 | 107 | 127 | 0 |
| 20 | 12 | False | True | 314 | 0.00001 | 0.00100 | 9741 | 277 | 136 | 141 | 0 | 962 | 290 | 198 | 92 | 0 |


### Resources

| Epoch | Train s | Val s | Train images/s | Val images/s | Train input s | Val input s | GPU allocated B | GPU reserved B | Process peak RSS B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2320.03220 | 79.51553 | 4.31804 | 15.74535 | 2272.00740 | 78.68700 | 2476697088 | 2766143488 | 2063015936 |
| 2 | 2473.32797 | 180.95697 | 4.05041 | 6.91877 | 2351.00192 | 179.07556 | 2476697088 | 2766143488 | 2312577024 |
| 3 | 2531.89892 | 104.63009 | 3.95671 | 11.96597 | 2495.50605 | 103.76265 | 2476697088 | 2766143488 | 2312577024 |
| 4 | 2405.89451 | 93.57878 | 4.16394 | 13.37910 | 2336.68716 | 92.01306 | 2476697088 | 2766143488 | 2312577024 |
| 5 | 2185.04931 | 80.15095 | 4.58479 | 15.62053 | 2024.19283 | 78.75251 | 2476697088 | 2766143488 | 2452889600 |
| 6 | 2491.99389 | 94.35279 | 4.02007 | 13.26935 | 2265.69603 | 91.70298 | 2476697088 | 2766143488 | 2472001536 |
| 7 | 2393.91932 | 294.07786 | 4.18477 | 4.25738 | 1977.11439 | 290.50042 | 2476697088 | 2766143488 | 2479697920 |
| 8 | 2054.05902 | 93.24588 | 4.87717 | 13.42687 | 1630.89791 | 89.73633 | 2476697088 | 2766143488 | 2479697920 |
| 9 | 1933.27698 | 88.50480 | 5.18188 | 14.14613 | 1489.88200 | 84.71402 | 2476697088 | 2766143488 | 2483195904 |
| 10 | 2153.36076 | 101.82572 | 4.65226 | 12.29552 | 1599.11306 | 98.63027 | 2476697088 | 2766143488 | 2483195904 |
| 11 | 1937.72677 | 81.09418 | 5.16998 | 15.43884 | 1437.48910 | 77.93563 | 2476697088 | 2766143488 | 2503827456 |
| 12 | 1914.46902 | 80.94867 | 5.23278 | 15.46659 | 1470.91263 | 77.54434 | 2476697088 | 2766143488 | 2503827456 |
| 13 | 1948.63010 | 76.85409 | 5.14105 | 16.29061 | 1495.15431 | 73.60887 | 2476697088 | 2766143488 | 2506739712 |
| 14 | 1930.53909 | 73.45689 | 5.18922 | 17.04401 | 1509.32867 | 70.25593 | 2476697088 | 2766143488 | 2506739712 |
| 15 | 1958.07646 | 81.61088 | 5.11625 | 15.34109 | 1510.33545 | 78.48333 | 2476697088 | 2766143488 | 2523885568 |
| 16 | 1913.29649 | 73.99041 | 5.23599 | 16.92111 | 1465.20531 | 70.81054 | 2476697088 | 2766143488 | 2532528128 |
| 17 | 1900.64200 | 68.97761 | 5.27085 | 18.15082 | 1476.45803 | 65.96960 | 2476697088 | 2766143488 | 2532528128 |
| 18 | 1912.55595 | 70.88606 | 5.23802 | 17.66215 | 1488.25971 | 67.44017 | 2476697088 | 2766143488 | 2532528128 |
| 19 | 1946.40686 | 72.29377 | 5.14692 | 17.31823 | 1540.20033 | 68.94527 | 2476697088 | 2766143488 | 2533150720 |
| 20 | 1941.65956 | 70.64767 | 5.15950 | 17.72174 | 1546.80375 | 67.57256 | 2476697088 | 2766143488 | 2533150720 |


Recorded train/validation phases sum to 11.74 h and 32.69 min. These totals exclude any time outside the recorded phases. RSS is the parent process peak and does not account for all worker memory. Shared compute and distinct input transport confound throughput.

Per-epoch cache counters are absent in this run's resource export.

## Figures

[DDR figures and captions](../../../docs/E5_E6_VALIDATION.md#ddr-figure-suite)

Recorded statistics are descriptive for this frozen validation split. The report contains no sample-level predictions.
