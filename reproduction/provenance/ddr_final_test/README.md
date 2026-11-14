# DDR final-test evaluator source provenance

`eval_ddr_recovery_once_executed.py` is a **byte-for-byte archive** of the evaluator downloaded from the GPU server. Its SHA-256 matches the server-recorded value:

`423e38af208fb6a2c0b4a6c2634c325a28981f356c8da2258a1e1d4e45e98f04`

The script's explicit test-entry command is `python eval_ddr_recovery_once_executed.py --execute-final-test`. The exact historical shell command and execution log are not present locally; this command describes the source interface and **must not be run again**. The archived source is associated with the documented final test. Matching its hash does not, by itself, cryptographically prove that this file was the one executed on the server.

| Evidence | Recorded value |
| --- | --- |
| Server result | `~/Kel_13/experiments/cadvit_tables_1_5_balanced_fp32_math_mb8_diag/ddr_full/test_metrics.json` |
| Server `test_metrics.json` SHA-256 | `4575f79eb536999c3ee0a9cdfb4a52dd01cf613020a1174556563df2367e1ea3` |
| Selected `best.pt` | Epoch 69 (checkpoint field `epoch=68`); SHA-256 `84e3272d3f10d9179fd2dd9e2799b640962e1185981856f87bcb3876b9deace0` |
| Manifest SHA-256 | `7bc3e3916e012bc46fcd9cd69aa469e0daf3229e1273f1d8ecb6054899bf6eb1` |
| Local aggregate result | [`final_test_summary.json`](../../../results/DDR/full_oversampling_fp32_math_mb8_80epochs/final_test_summary.json) |

The training campaign is **mixed-protocol**: epochs 1–42 used FP32 with micro-batch 16; the diagnostic recovery fork used FP32 Math SDPA, micro-batch 8, and accumulation 2 for epochs 43–80. The test source selects the checkpoint from validation macro-F1, validates the frozen manifest, evaluates the test split without augmentation, and writes `test_metrics.json` with create-only semantics. Per-image predictions remain on the server and are not archived here.

**Audit limits.** The local bundle does not contain the server `test_metrics.json`, checkpoint binaries, manifest, or execution transcript. Its recorded result hash therefore cannot be recomputed locally. The source checks checkpoint SHA-256 against `checkpoint_manifest.json` only if that file is present in its server repository path. The script hashes files under `~/Kel_13/Diabetic-Rethinography-v2`, while Python imports depend on the invocation environment; the exact imported module paths require the server execution record to confirm. Do not change the archived Python file; document any later corrections separately.

The independently written `reproduction/cadvit/ddr_recovery_final_test.py` is a distinct, untracked draft adapter. It has a different CLI, evaluation micro-batch, and output layout, and is **not evidence of the server evaluation**. Exclude that draft from the final commit unless separately reviewed.
