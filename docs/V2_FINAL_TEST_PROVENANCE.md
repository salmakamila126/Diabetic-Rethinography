# Historical V2 BG10 80-epoch final-test JSON provenance

**Verification date:** 2026-11-13. The reference SHA-256 values below were recorded from the original GPU-server JSON files on that date and supplied by the researcher. SHA-256 of each local file was recalculated byte for byte and matched its corresponding reference value. The JSON files were not modified and final-test inference was not rerun.

| Dataset | Run ID | Original server result path | Local aggregate result | Reference and observed SHA-256 | Checkpoint selected from validation | Test count |
| --- | --- | --- | --- | --- | ---: | ---: |
| APTOS2019 | `aptos_v2_bg10_stable_80ep_seed42` | `~/Kel_13/checkpoints/aptos_v2_bg10_stable_80ep_seed42/final_test_v2_80ep.json` | [`results/APTOS2019/v2_bg10_historical_80epochs/final_test_v2_80ep.json`](../results/APTOS2019/v2_bg10_historical_80epochs/final_test_v2_80ep.json) | `5db8e70f6c8ef78d0f9047f08f6ba1bcdae4c97c6d6b39c33ba58e8792b28e3c` | 69 | 366 |
| DDR | `ddr_v2_fp32_weighted_fp64clip_seed42` | `~/Kel_13/checkpoints/ddr_v2_fp32_weighted_fp64clip_seed42/final_test_v2_80ep.json` | [`results/DDR/v2_bg10_historical_80epochs/final_test_v2_80ep.json`](../results/DDR/v2_bg10_historical_80epochs/final_test_v2_80ep.json) | `95b0a47ef36626e23e6721ebb1bfeacd6265a39939457ff37ae5115ca58381d6` | 72 | 1252 |

The server path layout follows the run directories in [`evaluation.py`](../reproduction/cadvit/evaluation.py) and the documented artifact root `~/Kel_13`; the supplied hash record establishes identity of the file bytes, not an independent path or execution log audit. Both JSON files have internally consistent sample counts, confusion-matrix diagonals, and accuracies. The files report APTOS2019 accuracy `0.7896174863387978`, macro-F1 `0.6049524682606406`, and DDR accuracy `0.8043130990415336`, macro-F1 `0.6202869401176525`.

**Verification levels are separate:**

1. **Available final-test result:** the local JSON exists.
2. **Internally validated result and byte identity:** the matrix agrees with reported count/accuracy, and its SHA-256 matches the researcher-provided server hash.
3. **Fully verified execution lineage:** **not established locally**. Checkpoint binaries, frozen manifest contents, source/runtime execution record, and DDR checkpoint-transfer evidence are unavailable here. Matching result-file hashes does not independently prove checkpoint lineage, manifest integrity, or which evaluator executed.

These V2 BG10 results are distinct from historical V1, APTOS `balanced_v1`, and the DDR FP32 Math recovery final test. The V2 DDR training history combines AMP, weighted FP32, and FP64 gradient-norm recovery phases; the score must not be treated as an unchanged 80-epoch protocol.
