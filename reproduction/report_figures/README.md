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

## DDR real-image visualization transfer

These **nine unique PNGs and two metadata JSON files** were copied byte-for-byte from the locally verified `cadvit_real_visualizations.zip`. The archive SHA-256 is `3d5802a2953b78ee24f28fd74ae75b3e4906107862553d0472c9f42df0636127`, matching the server value supplied by the researcher. The PNGs were inspected visually; no model execution or final-test inference was performed during integration. The transfer hash establishes archive byte identity with that recorded value; it does not independently prove how each panel was generated. The JSON files preserve the server-reported per-image provenance for Grad-CAM and retrospective error examples.

| Transferred file | SHA-256 |
| --- | --- |
| `DDR_train_grade_gallery.png` | `a2c9517dbeb6f2d70d7b815f8d0d86395e028e3cccea948816271ad76554bf16` |
| `DDR_preprocessing_grade0.png` | `175bdfffec2da4b0cac07029265f514074983683c4840421f8832457ebde00a0` |
| `DDR_preprocessing_grade1.png` | `43a7a7a715fd6ee97ff886be2977436e2e56a42265312b92abf8c1c8b7ab377e` |
| `DDR_preprocessing_grade2.png` | `3b581d93a40c684b7fc7a3cb9ed1d3ad90db68798a4d664e074421f03db50f89` |
| `DDR_preprocessing_grade3.png` | `1c54065c04524f4ba124b197887403ffddb172135a0fee41f3a90472c5ef371c` |
| `DDR_preprocessing_grade4.png` | `67e61f0ab112f4a42696f84e6cfe12d11480e76835b132895a60e0a520bf6974` |
| `DDR_augmentation_gallery.png` | `2ece8522c22b92d1836d3896af0e268645a4c089eb5f634d9c2af5dd25578d48` |
| `DDR_gradcam_validation.png` | `8f6420520f2edfb3112b66a4f888d48af1cf4d4bee6ed54a482b2d364ac3fcd8` |
| `DDR_gradcam_metadata.json` | `b7b666ad6e1455324f0dc650cbe4ca0aaa9759653f527ca83627ff4fdb92ab24` |
| `DDR_error_gallery.png` | `468db9ef5bbf20f4408387ed301876201c8101d19a5f082b2f2e2bf9932a000f` |
| `DDR_error_metadata.json` | `d27f8e3f3d56f46de26eb0ebb3985c374c42ce02406e2cae6731d6a9ac4b2619` |

The images contain adapted OIA-DDR fundus data. Changes include selection into grade galleries, crop/resize/blur and BG10 comparison panels, geometric augmentation, Grad-CAM overlays, and a retrospective error gallery. They are distributed for **non-commercial research under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**. Credit: OIA-DDR dataset by Tao Li, Yingqi Gao, Kai Wang, Song Guo, Hanruo Liu, and Hong Kang; cite *Diagnostic Assessment of Deep Learning Algorithms for Diabetic Retinopathy Screening*, *Information Sciences* 501 (2019), 511–522, [DOI 10.1016/j.ins.2019.06.011](https://doi.org/10.1016/j.ins.2019.06.011). See the [dataset maintainers' license statement](https://github.com/nkicsl/DDR-dataset/blob/master/README.md#license). The CC license applies to these derived image panels; it does not relicense the repository code.

## APTOS2019 local-only visualization transfer

Nine PNGs and two metadata JSON files from the same verified ZIP were copied byte-for-byte into this directory for local research review. They are ignored by `.gitignore` and must not be staged or published while permission to share APTOS images with collaborators remains unconfirmed. The [original Kaggle dataset](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) is subject to competition rules; academic or non-commercial purpose alone does not establish redistribution rights. The relative Markdown image links in notebooks 01, 07, and 08 render without a kernel **only on a local copy containing these files**. They do not render on GitHub or a fresh clone. The transfer ZIP SHA-256 is `3d5802a2953b78ee24f28fd74ae75b3e4906107862553d0472c9f42df0636127`. This confirms archive byte identity with the researcher-supplied server hash, not independent execution or checkpoint provenance.

| Local-only transferred file | SHA-256 |
| --- | --- |
| `APTOS2019_train_grade_gallery.png` | `3730e5cb02fe5e1cc12e8f7aa34faa43458d8d9078c2622c269fc991f6d07194` |
| `APTOS2019_preprocessing_grade0.png` | `e81e3ac4fe6a6e99cbeacd39bd6f5a09736c9327fa666593856a84f50ece27d3` |
| `APTOS2019_preprocessing_grade1.png` | `89bbaf061299c8a206b05a9d6c8caa039d2694d470149747144ad3054e341623` |
| `APTOS2019_preprocessing_grade2.png` | `81f8753bcf39941ce08202d3585ece31dfd46e53f0a140b10bdfcde4296e1593` |
| `APTOS2019_preprocessing_grade3.png` | `aefe3aa2af7fcf78b3181b1bb434971aac217c4d62b4fadf8230b9d266bb83fa` |
| `APTOS2019_preprocessing_grade4.png` | `f0c7f821e30cb2403968b6d7c34557fa5f1cef437c8a9745bc4e18fe83098ed3` |
| `APTOS2019_augmentation_gallery.png` | `1099cc2e39f74c18baa1e180fd52982c54894ce6dfad0a227e5819db85df764c` |
| `APTOS2019_gradcam_validation.png` | `072cd6deb0d89ee00444f9e92a1995779ee2fb27d22170fd360067eb60ab0ac8` |
| `APTOS2019_gradcam_metadata.json` | `3a782e0c1fd1fe63ba3b1b9b75c07f159db5ef8d7a23e29aeca03a9d420a4119` |
| `APTOS2019_error_gallery.png` | `cfda5ba216e3575ab642e0b6087f453980019d9cc3b12ea5d451a85ae7c45663` |
| `APTOS2019_error_metadata.json` | `bfe82e33e68a80bcc2c7f3e6cd9499a5ddd9f80c2f5bfabe348f1910500ec2bf` |

The metadata reports validation Grad-CAM for the APTOS balanced checkpoint selected at epoch 24 and a retrospective gallery from saved final-test predictions. The checkpoint, manifest, and raw per-image result are not present locally, so their contents and lineage cannot be independently verified from these files alone. No training or final-test inference was rerun for this transfer.
