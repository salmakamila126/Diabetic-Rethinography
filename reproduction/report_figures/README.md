# Static report figures

These files make historic outputs visible in the static [V1](../experiments/01_experiment_initial_cadvit_v1.ipynb), [V2](../experiments/02_experiment_bg10_cadvit_v2.ipynb), and [balanced/recovery](../experiments/03_experiment_balanced_training_and_ddr_recovery.ipynb) experiment records. The [main report](../../CAD_ViT.ipynb) is a shorter overview. The source notebook remains byte-for-byte preserved in [`reproduction/archive/`](../archive/).

| File | Provenance |
| --- | --- |
| `historical_v2_aptos_training_curves.png` | Exact decoded PNG from archived notebook cell 73, output 2; SHA-256 `839fec9b25cdcdd3e9ae0d6508452f4a6ef0cb5483d91eacd24ee5670852f77d` |
| `historical_v2_ddr_training_curves.png` | Exact decoded PNG from cell 73, output 4; SHA-256 `77563a76a17cd224bfa288097659d32b6fd4e6f02b11e8f9f88de48bd38d67e1` |
| `historical_v1_aptos_confusion_matrix.png` | Exact decoded PNG from cell 88, output 0; SHA-256 `7c627dec9e7a6ae78dbbc16fffb94ee2665f39331b322cbaec6279d4173f4077` |
| `historical_v1_ddr_confusion_matrix.png` | Exact decoded PNG from cell 88, output 1; SHA-256 `41ca8e04b68871d4ab5e6c74b652a75b6ac67323e25e19140b9561ec02c0ae2f` |
| `aptos_balanced_80_epoch_curves.svg` | New static four-panel visualization of the 80 `train_loss`, `val_loss`, `val_accuracy`, and `val_macro_f1` values in [`history.jsonl`](../../results/APTOS2019/full_oversampling_amp_fp16_80epochs/history.jsonl), SHA-256 `7ce8490772e1c37967c7418a52b2391fce1f6636ad6d798e52e6bb29150ccdcf`. The dashed line marks the validation-selected epoch 24. No training or inference was run. |

The historical PNGs preserve the original figure bytes; the APTOS SVG is a derived visualization and is not an original training artifact. DDR recovery's current PNG curves remain under `results/DDR/full_oversampling_fp32_math_mb8_80epochs/` and are linked directly from Experiment 3.
