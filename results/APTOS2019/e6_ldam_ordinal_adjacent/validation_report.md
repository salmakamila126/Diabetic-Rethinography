# E6 APTOS2019 frozen validation

2,930 training and 366 validation images. The run completed 20 epochs. The selected checkpoint was epoch **15**.

[E5–E6 comparison and figure index](../../../docs/E5_E6_VALIDATION.md) · [Full aggregate evidence, configuration, and hashes](validation_evidence.json)

## Run configuration and provenance

| Field | Recorded value |
| --- | --- |
| Run identity | foundation_e6/APTOS2019/ldam_ordinal_adjacent_seed42_b32_20/run |
| Architecture | ConvNeXt-Tiny with five-class categorical and ordered-threshold ordinal heads |
| Foundation model / revision | facebook/convnext-tiny-224 / 6166b7613034066690a621d8bf25ffdf181a34f0 |
| Seed / epochs | 42 / 20 |
| Micro-batch × accumulation → effective batch | 32 × 1 → 32 |
| Optimizer / LR backbone / LR head / decay / clip | AdamW / 1e-05 / 0.001 / 0.01 / 1.0 |
| Precision / device / scheduler | bf16 / cuda / none in runner |
| Data transport, train | verified mmap decoded RGB, stochastic B1 augmentation each epoch |
| Data transport, validation | uncached official processor |
| Workers / persistent / prefetch / pin | 2 / True / 2 / True |
| Train support grades 0–4 | [1445, 297, 798, 154, 236] |
| Validation support grades 0–4 | [180, 35, 101, 20, 30] |
| Checkpoint selection | max validation macro-F1; tie lower validation CE; tie earliest epoch |
| Parameter counts | backbone_total: 27,820,128, backbone_trainable: 27,820,128, head_total: 4,618, total_trainable: 27,824,746 |
| Versions | torch 2.7.1+cu126, transformers 4.57.3, pillow 11.1.0, numpy 2.5.5 |
| Source commit | b3dd3ca01936f7d173cb646721271703b4e0308d |
| Frozen manifest SHA-256 | 73d1546e3c2fd2bf9e3d0fd10af914cb523d28e4bc8e1f25212af454d4be8854 |
| Manifest sidecar SHA-256 | d3e367237f1a3d43ed9948cf40c6065d9ba2756e402742b7cb253bb1be44a657 |
| Ordered train/validation identity SHA-256 | 487cfa60222c08d4a2b6b2b48b05f89fea0f0d3f7285d924059939966a12dae0 |
| Processor SHA-256 / effective processor SHA-256 | fd575b890da5a949493e1d1e7a70bfcb9e4b99fe444004d2dbfa253add254741 / dd1700334d237761807778002bac2692337179f1ff24e0519fd67e74090bd5fd |
| Official weight SHA-256 | 1fe15b2eba9e355de14c4e244e6365bf61ffef84da7d3442c0afbb2d8564f477 |
| adjacent coefficient | 0.10000 |
| adjacent pairs | [[0, 1], [1, 2]] |
| categorical | LDAM true-class margin scaled by 30 |
| effective number beta | 0.99990 |
| effective number weights | [0.42294645697553074, 1.9445937634128903, 0.7419297601954513, 3.723667786455057, 2.4398018397740713] |
| ldam margins | [0.2856822270564808, 0.42428831583697174, 0.3313976784419964, 0.5, 0.44938873728640827] |
| ordinal | True |
| ordinal coefficient | 0.25000 |
| validation loss | ordinary categorical CE on raw logits |
| weight start epoch | 11 |
| Source best.pt | f03acdc4fc0325f0c7556d5eafec8a2e91ba8b69d733ddb28d86593f64ea221e (334,177,515 bytes) |
| Source last.pt | 225bfcab05c0cde51e5d771f1b113f2f8086eb4ad8f1d22884055cf5763e290b (334,188,395 bytes) |


Processor, augmentation, optimizer, class balance, source hashes, and the complete transport configuration are in the JSON. The original run stores configuration in `run.json`. The supplied checkpoint sidecars match the recorded checkpoint digests and byte sizes.

## Best checkpoint metrics

| Metric | Value | Source |
| --- | --- | --- |
| Accuracy | 0.78962 | Recorded |
| Macro-F1 | 0.67920 | Recorded |
| Weighted-F1 | 0.79589 | Recorded |
| QWK | 0.88038 | Recorded |
| Grade MAE | 0.27322 | Recorded |
| Categorical CE | 0.81660 | Recorded |
| Macro precision | 0.67302 | Matrix-derived |
| Macro recall | 0.71263 | Matrix-derived |
| correct | 289 | Recorded |
| wrong | 77 | Recorded |
| undergraded | 42 | Recorded |
| overgraded | 35 | Recorded |
| mean overgrade magnitude | 1.31429 | Recorded |
| mean undergrade magnitude | 1.28571 | Recorded |
| high confidence wrong | 0 | Recorded |
| high confidence wrong rate all | 0.00000 | Recorded |
| high confidence wrong fraction of wrong | 0.00000 | Recorded |
| Adjacent mistakes (absolute grade difference 1) | 56 | Matrix-derived |
| Non-adjacent mistakes (absolute grade difference ≥2) | 21 | Matrix-derived |


### Confusion matrix

Rows are true grades and columns are predicted grades. Counts include the diagonal. Percentages are normalized within each true grade.

| True grade | 0 | 1 | 2 | 3 | 4 | Support |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 172 | 8 | 0 | 0 | 0 | 180 |
| 1 | 1 | 28 | 6 | 0 | 0 | 35 |
| 2 | 1 | 25 | 58 | 6 | 11 | 101 |
| 3 | 0 | 0 | 4 | 12 | 4 | 20 |
| 4 | 0 | 2 | 7 | 2 | 19 | 30 |


| True grade | 0 (%) | 1 (%) | 2 (%) | 3 (%) | 4 (%) |
| --- | --- | --- | --- | --- | --- |
| 0 | 95.56 | 4.44 | 0.00 | 0.00 | 0.00 |
| 1 | 2.86 | 80.00 | 17.14 | 0.00 | 0.00 |
| 2 | 0.99 | 24.75 | 57.43 | 5.94 | 10.89 |
| 3 | 0.00 | 0.00 | 20.00 | 60.00 | 20.00 |
| 4 | 0.00 | 6.67 | 23.33 | 6.67 | 63.33 |


Largest off-diagonal cells (true→predicted) were 2→1 (25), 2→4 (11), 0→1 (8), 4→2 (7). The grade 1 row was `[1, 28, 6, 0, 0]`; the grade 3 row was `[0, 0, 4, 12, 4]`.

### Per-grade metrics

Support, TP, precision, recall, and F1 are recorded and independently recomputed from the matrix. FP and FN are matrix-derived. Per-grade MAE was recorded in the CSV.

| Grade | Support | TP | FP | FN | Precision | Recall | F1 | Grade MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 180 | 172 | 2 | 8 | 0.98851 | 0.95556 | 0.97175 | 0.04444 |
| 1 | 35 | 28 | 35 | 7 | 0.44444 | 0.80000 | 0.57143 | 0.20000 |
| 2 | 101 | 58 | 17 | 43 | 0.77333 | 0.57426 | 0.65909 | 0.54455 |
| 3 | 20 | 12 | 8 | 8 | 0.60000 | 0.60000 | 0.60000 | 0.40000 |
| 4 | 30 | 19 | 15 | 11 | 0.55882 | 0.63333 | 0.59375 | 0.73333 |


## All 20 epochs

Markdown rounds values for readability. The JSON preserves the precision and full field coverage of the export. Per-grade MAE fields removed from the JSON by the export filter are preserved separately from the CSV.

### Training scores

| Epoch | Total loss | Categorical | Ordinal | Adjacent | Accuracy | Macro-F1 | Weighted-F1 | QWK | Grade MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 7.05602 | 6.96965 | 0.33854 | 0.01731 | 0.71809 | 0.43449 | 0.68428 | 0.66472 | 0.47031 |
| 2 | 5.16134 | 5.09440 | 0.26269 | 0.01265 | 0.81092 | 0.60424 | 0.79450 | 0.83604 | 0.27918 |
| 3 | 4.51901 | 4.45688 | 0.24399 | 0.01131 | 0.83891 | 0.68623 | 0.83112 | 0.88122 | 0.22696 |
| 4 | 4.07273 | 4.01403 | 0.23096 | 0.00962 | 0.86075 | 0.73505 | 0.85510 | 0.89458 | 0.19898 |
| 5 | 3.68664 | 3.63026 | 0.22165 | 0.00972 | 0.87474 | 0.76842 | 0.87223 | 0.91494 | 0.17201 |
| 6 | 3.19431 | 3.14013 | 0.21361 | 0.00776 | 0.90000 | 0.81414 | 0.89756 | 0.92798 | 0.13993 |
| 7 | 2.82355 | 2.77117 | 0.20695 | 0.00634 | 0.91160 | 0.83703 | 0.91022 | 0.93621 | 0.12389 |
| 8 | 2.47384 | 2.42311 | 0.20050 | 0.00602 | 0.92218 | 0.85676 | 0.92146 | 0.94781 | 0.10683 |
| 9 | 2.02879 | 1.97933 | 0.19609 | 0.00438 | 0.93788 | 0.88505 | 0.93714 | 0.95709 | 0.08635 |
| 10 | 1.68220 | 1.63488 | 0.18803 | 0.00303 | 0.95154 | 0.90897 | 0.95117 | 0.96853 | 0.06621 |
| 11 | 2.70339 | 2.65742 | 0.18255 | 0.00331 | 0.94949 | 0.90750 | 0.94992 | 0.96351 | 0.07167 |
| 12 | 2.24575 | 2.20020 | 0.18064 | 0.00390 | 0.96143 | 0.93003 | 0.96169 | 0.97730 | 0.05017 |
| 13 | 1.97296 | 1.92909 | 0.17426 | 0.00306 | 0.96382 | 0.93235 | 0.96420 | 0.97797 | 0.04778 |
| 14 | 1.59039 | 1.54476 | 0.18137 | 0.00294 | 0.96758 | 0.93971 | 0.96781 | 0.97830 | 0.04403 |
| 15 | 1.53473 | 1.49038 | 0.17613 | 0.00317 | 0.97201 | 0.94893 | 0.97223 | 0.98237 | 0.03720 |
| 16 | 1.34914 | 1.30618 | 0.17073 | 0.00275 | 0.97543 | 0.95403 | 0.97557 | 0.98393 | 0.03311 |
| 17 | 1.22134 | 1.17460 | 0.18596 | 0.00248 | 0.98157 | 0.96487 | 0.98167 | 0.98828 | 0.02457 |
| 18 | 1.09133 | 1.04821 | 0.17151 | 0.00239 | 0.97986 | 0.96323 | 0.97999 | 0.98826 | 0.02594 |
| 19 | 1.37389 | 1.33415 | 0.15813 | 0.00206 | 0.98020 | 0.96256 | 0.98025 | 0.98985 | 0.02457 |
| 20 | 1.07317 | 1.03223 | 0.16257 | 0.00299 | 0.97850 | 0.96024 | 0.97863 | 0.98705 | 0.02799 |


### Validation scores

| Epoch | CE | Ordinal | Accuracy | Macro-F1 | Weighted-F1 | QWK | Grade MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.19095 | 0.26514 | 0.79781 | 0.59014 | 0.77791 | 0.82888 | 0.28962 |
| 2 | 1.17838 | 0.24535 | 0.80601 | 0.65051 | 0.80243 | 0.86013 | 0.26776 |
| 3 | 1.07873 | 0.23688 | 0.79508 | 0.56368 | 0.77289 | 0.84874 | 0.27596 |
| 4 | 1.02054 | 0.23039 | 0.79235 | 0.64032 | 0.79111 | 0.86057 | 0.28962 |
| 5 | 1.00167 | 0.22686 | 0.80055 | 0.62471 | 0.79405 | 0.87121 | 0.26776 |
| 6 | 0.97240 | 0.22072 | 0.80601 | 0.64584 | 0.80022 | 0.87215 | 0.25956 |
| 7 | 0.98156 | 0.21849 | 0.78962 | 0.63956 | 0.79271 | 0.86501 | 0.27869 |
| 8 | 0.87457 | 0.21785 | 0.80055 | 0.59950 | 0.78977 | 0.86020 | 0.27869 |
| 9 | 0.92041 | 0.21679 | 0.80328 | 0.65946 | 0.80663 | 0.86644 | 0.27596 |
| 10 | 0.88494 | 0.21235 | 0.78689 | 0.64199 | 0.78833 | 0.86147 | 0.29235 |
| 11 | 0.89179 | 0.21257 | 0.79508 | 0.66376 | 0.80049 | 0.87471 | 0.27596 |
| 12 | 0.88985 | 0.21070 | 0.77869 | 0.64191 | 0.78644 | 0.85866 | 0.30874 |
| 13 | 0.82184 | 0.21059 | 0.79508 | 0.62735 | 0.79652 | 0.86446 | 0.28689 |
| 14 | 0.85387 | 0.21085 | 0.78689 | 0.66686 | 0.79363 | 0.85942 | 0.29781 |
| 15 | 0.81660 | 0.21074 | 0.78962 | 0.67920 | 0.79589 | 0.88038 | 0.27322 |
| 16 | 0.75069 | 0.20819 | 0.80328 | 0.63894 | 0.80136 | 0.87953 | 0.25410 |
| 17 | 0.79362 | 0.20536 | 0.78962 | 0.65383 | 0.79473 | 0.86746 | 0.28142 |
| 18 | 0.71837 | 0.20765 | 0.77596 | 0.59838 | 0.77710 | 0.84061 | 0.33333 |
| 19 | 0.71879 | 0.21216 | 0.78689 | 0.63020 | 0.78794 | 0.86246 | 0.29781 |
| 20 | 0.74053 | 0.20655 | 0.77049 | 0.62216 | 0.77816 | 0.85362 | 0.32240 |


### Selection, imbalance schedule, and errors

HC wrong requires an incorrect argmax and maximum raw softmax probability ≥0.90.

| Epoch | Best through epoch | Updated | Weights active | Optimizer steps | Backbone LR | Head LR | Train correct | Train wrong | Train over | Train under | Train HC wrong | Val correct | Val wrong | Val over | Val under | Val HC wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | True | False | 92 | 0.00001 | 0.00100 | 2104 | 826 | 286 | 540 | 0 | 292 | 74 | 30 | 44 | 0 |
| 2 | 2 | True | False | 92 | 0.00001 | 0.00100 | 2376 | 554 | 193 | 361 | 0 | 295 | 71 | 32 | 39 | 0 |
| 3 | 2 | False | False | 92 | 0.00001 | 0.00100 | 2458 | 472 | 178 | 294 | 0 | 291 | 75 | 33 | 42 | 0 |
| 4 | 2 | False | False | 92 | 0.00001 | 0.00100 | 2522 | 408 | 164 | 244 | 0 | 290 | 76 | 44 | 32 | 0 |
| 5 | 2 | False | False | 92 | 0.00001 | 0.00100 | 2563 | 367 | 147 | 220 | 0 | 293 | 73 | 41 | 32 | 0 |
| 6 | 2 | False | False | 92 | 0.00001 | 0.00100 | 2637 | 293 | 120 | 173 | 0 | 295 | 71 | 38 | 33 | 0 |
| 7 | 2 | False | False | 92 | 0.00001 | 0.00100 | 2671 | 259 | 99 | 160 | 0 | 289 | 77 | 33 | 44 | 0 |
| 8 | 2 | False | False | 92 | 0.00001 | 0.00100 | 2702 | 228 | 102 | 126 | 0 | 293 | 73 | 39 | 34 | 0 |
| 9 | 9 | True | False | 92 | 0.00001 | 0.00100 | 2748 | 182 | 74 | 108 | 0 | 294 | 72 | 40 | 32 | 0 |
| 10 | 9 | False | False | 92 | 0.00001 | 0.00100 | 2788 | 142 | 62 | 80 | 0 | 288 | 78 | 45 | 33 | 0 |
| 11 | 11 | True | True | 92 | 0.00001 | 0.00100 | 2782 | 148 | 81 | 67 | 0 | 291 | 75 | 37 | 38 | 0 |
| 12 | 11 | False | True | 92 | 0.00001 | 0.00100 | 2817 | 113 | 59 | 54 | 0 | 285 | 81 | 48 | 33 | 0 |
| 13 | 11 | False | True | 92 | 0.00001 | 0.00100 | 2824 | 106 | 63 | 43 | 0 | 291 | 75 | 42 | 33 | 0 |
| 14 | 14 | True | True | 92 | 0.00001 | 0.00100 | 2835 | 95 | 54 | 41 | 0 | 288 | 78 | 41 | 37 | 0 |
| 15 | 15 | True | True | 92 | 0.00001 | 0.00100 | 2848 | 82 | 47 | 35 | 0 | 289 | 77 | 35 | 42 | 0 |
| 16 | 15 | False | True | 92 | 0.00001 | 0.00100 | 2858 | 72 | 33 | 39 | 0 | 294 | 72 | 34 | 38 | 0 |
| 17 | 15 | False | True | 92 | 0.00001 | 0.00100 | 2876 | 54 | 30 | 24 | 0 | 289 | 77 | 38 | 39 | 0 |
| 18 | 15 | False | True | 92 | 0.00001 | 0.00100 | 2871 | 59 | 31 | 28 | 0 | 284 | 82 | 53 | 29 | 0 |
| 19 | 15 | False | True | 92 | 0.00001 | 0.00100 | 2872 | 58 | 26 | 32 | 0 | 288 | 78 | 48 | 30 | 0 |
| 20 | 15 | False | True | 92 | 0.00001 | 0.00100 | 2867 | 63 | 31 | 32 | 0 | 282 | 84 | 47 | 37 | 0 |


### Resources

| Epoch | Train s | Val s | Train images/s | Val images/s | Train input s | Val input s | GPU allocated B | GPU reserved B | Process peak RSS B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 416.91209 | 34.56896 | 7.02786 | 10.58753 | 387.66998 | 34.14598 | 2476697088 | 2745171968 | 1830780928 |
| 2 | 431.05104 | 32.12958 | 6.79734 | 11.39137 | 420.54748 | 31.91892 | 2476697088 | 2745171968 | 1841750016 |
| 3 | 423.99349 | 34.63224 | 6.91048 | 10.56819 | 415.27363 | 34.43743 | 2476697088 | 2745171968 | 2034569216 |
| 4 | 421.34356 | 32.95024 | 6.95395 | 11.10766 | 412.27433 | 32.76005 | 2476697088 | 2745171968 | 2085351424 |
| 5 | 373.23037 | 27.38498 | 7.85038 | 13.36499 | 352.34302 | 26.92527 | 2476697088 | 2745171968 | 2085351424 |
| 6 | 394.95485 | 29.43795 | 7.41857 | 12.43293 | 358.91892 | 29.00111 | 2476697088 | 2745171968 | 2085351424 |
| 7 | 410.54528 | 27.29946 | 7.13685 | 13.40686 | 372.55636 | 26.76751 | 2476697088 | 2745171968 | 2085351424 |
| 8 | 413.91893 | 30.85509 | 7.07868 | 11.86190 | 376.96224 | 30.21092 | 2476697088 | 2745171968 | 2086166528 |
| 9 | 512.74987 | 29.74381 | 5.71429 | 12.30508 | 477.46870 | 29.09149 | 2476697088 | 2745171968 | 2086166528 |
| 10 | 520.24740 | 29.65635 | 5.63194 | 12.34137 | 484.37559 | 29.40392 | 2476697088 | 2745171968 | 2086166528 |
| 11 | 481.54836 | 28.85411 | 6.08454 | 12.68450 | 471.40258 | 28.57634 | 2476697088 | 2745171968 | 2086166528 |
| 12 | 509.16332 | 25.75842 | 5.75454 | 14.20895 | 500.29854 | 25.57613 | 2476697088 | 2745171968 | 2086166528 |
| 13 | 380.66031 | 26.47700 | 7.69715 | 13.82332 | 373.56551 | 26.25696 | 2476697088 | 2745171968 | 2086166528 |
| 14 | 401.42516 | 26.69550 | 7.29899 | 13.71017 | 390.32592 | 26.47107 | 2476697088 | 2745171968 | 2086166528 |
| 15 | 380.04784 | 26.93331 | 7.70956 | 13.58912 | 367.41796 | 26.69785 | 2476697088 | 2745171968 | 2086342656 |
| 16 | 425.13381 | 26.65462 | 6.89195 | 13.73120 | 412.44380 | 26.41094 | 2476697088 | 2745171968 | 2092666880 |
| 17 | 519.80047 | 28.71713 | 5.63678 | 12.74501 | 508.69758 | 28.47862 | 2476697088 | 2745171968 | 2092666880 |
| 18 | 532.12494 | 26.06915 | 5.50623 | 14.03959 | 522.58398 | 25.85047 | 2476697088 | 2745171968 | 2092666880 |
| 19 | 364.75143 | 26.38926 | 8.03287 | 13.86928 | 353.62728 | 26.00250 | 2476697088 | 2745171968 | 2092666880 |
| 20 | 382.09539 | 26.78573 | 7.66824 | 13.66399 | 342.49765 | 26.37524 | 2476697088 | 2745171968 | 2092666880 |


Recorded train/validation phases sum to 2.42 h and 9.63 min. These totals exclude any time outside the recorded phases. RSS is the parent process peak and does not account for all worker memory. Shared compute and distinct input transport confound throughput.

Per-epoch cache counters are absent in this run's resource export.

## Figures

[APTOS2019 figures and captions](../../../docs/E5_E6_VALIDATION.md#aptos2019-figure-suite)

Recorded statistics are descriptive for this frozen validation split. The report contains no sample-level predictions.
