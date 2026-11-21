# CAD-ViT and ConvNeXt experiment inventory

Reviewed `salma` at **027e586f607bde4877b395ca566082c713340e8b** before the notebook migration. The optimization investigation reads `origin/novelty` at **528f6558c7596c2424fe4f428428a93c74dd8c14** as a separate historical source. The [notebook index](../notebooks/README.md) is the entry point. This audit used committed aggregate files and notebook source and stored outputs. It did not inspect live HPC jobs.

## Notebook migration and archival record

| Former active entry | Disposition and evidence |
| --- | --- |
| [Root CAD_ViT_Novelty.ipynb at the reviewed commit](https://github.com/salmakamila126/Diabetic-Rethinography/blob/027e586f607bde4877b395ca566082c713340e8b/CAD_ViT_Novelty.ipynb) | The first 21 cells describe the CAD-ViT P1–P4 prototype, including the architecture probe, parameter table, ablations, cache audit, provisional resource snapshots and guarded KC commands. The [prototype protocol](CADVIT_NOVELTY_PROTOCOL.md) carries the method, controls, hashes and execution notes. Its archived notebook retains the original cells and historical status. The remaining 18 cells cover the same ten B0–B4 runs as the dataset-specific notebooks. |
| [01 CAD-ViT reproduction snapshot](https://github.com/salmakamila126/Diabetic-Rethinography/blob/027e586f607bde4877b395ca566082c713340e8b/notebooks/01_CAD_ViT_Reproduction.ipynb) | [Moved reproduction notebook](../notebooks/reproduction/CAD_ViT_Reproduction.ipynb), with its code, tables and original visuals retained. The [105-cell reproduction archive](../reproduction/archive/CAD_ViT_original_105_cells.ipynb) and the three [historical campaigns](../reproduction/experiments/) remain the reproduction record. |
| [02 B0–B4 combined snapshot](https://github.com/salmakamila126/Diabetic-Rethinography/blob/027e586f607bde4877b395ca566082c713340e8b/notebooks/02_ConvNeXt_B0_B4_Validation.ipynb) | Superseded by ten active dataset-specific notebook entries. APTOS B4 contains Q and R. DDR B4 explains the inapplicable arm without fabricating a run. |
| [03 E5 APTOS snapshot](https://github.com/salmakamila126/Diabetic-Rethinography/blob/027e586f607bde4877b395ca566082c713340e8b/notebooks/03_ConvNeXt_E5_APTOS2019.ipynb) and [03 E6 protocol snapshot](https://github.com/salmakamila126/Diabetic-Rethinography/blob/027e586f607bde4877b395ca566082c713340e8b/notebooks/03_ConvNeXt_E6_Boundary_Aware.ipynb) | Replaced as active entries by per-dataset notebooks based on the published validation evidence. The [historical APTOS E5 final test](../results/APTOS2019/e5_integrated/test_summary.json) stays independent and retains original bytes. |
| E7 APTOS2019 and DDR | New protocol notebooks for the published runner. No locally verifiable completed result or checkpoint receipt was found for either dataset. Their current progress on HPC is unknown. |

The CAD-ViT candidate (`reproduction/cadvit`) and the CAD-ViT P1–P4 prototype (`novelty/cadvit_*`) are separate from the ConvNeXt experiments. The ordered 80-epoch APTOS balanced and DDR recovery histories stay in their original result directories. P1–P4 prior art, ablations, original-checkpoint transfer and failed or incomplete pilot context remain traceable through the protocol and pinned snapshot. PReOF E0–E3 remain recorded under their existing [phase documentation](PReOF_PHASE1.md), [E2 report](PReOF_E2.md) and [E3 report](PReOF_E3_FOUNDATION_FEASIBILITY.md).

## B0–B4 mapping

The [validation evidence](../novelty/convnext_b0_b4/validation_evidence.json) names each historical run, pretrained revision and processor, checkpoint hash, frozen manifest, configuration, epoch history and figure hash. Source and checkpoint hashes have different roles. A matching source blob identifies the main module. It does not establish byte equivalence of every historical dependency or device setting.

| Arm | Question and actual source | APTOS2019 validation | DDR validation | Reproducibility |
| --- | --- | --- | --- | --- |
| B0 | Full fine-tuning, official input transform, ordinary CE. Main source SHA matches [de4699e](https://github.com/salmakamila126/Diabetic-Rethinography/blob/de4699e7162b461195f87924d5f7ca39d2313cdb/novelty/foundation_adaptation.py). | 20 epochs, best 17, macro-F1 0.673686 | 20 epochs, best 8, 0.619817 | Current [source](../novelty/foundation_adaptation.py) exposes B0 for new runs. The historical logs do not record train accuracy, train F1 or train confusion per epoch. |
| B1 | B0 plus conservative train-only retinal augmentation. Main source SHA matches [3a563c6](https://github.com/salmakamila126/Diabetic-Rethinography/blob/3a563c67fef82454ce2ed34858615bd456424a13/novelty/foundation_adaptation.py). | 20 epochs, best 16, 0.701124 | 20 epochs, best 11, 0.652146 | Tracked CLI. The augmentation implementation fixes order and seed rules. |
| B2 | Unaugmented B0 plus train-only weighted categorical CE. Same main source commit as B1. | 20 epochs, best 7, 0.659987 | 20 epochs, best 5, 0.623084 | Tracked CLI. Weights are inverse square roots normalized to mean one per training image. |
| B3 | B1 plus grade 3 target fraction 0.10 with proportional replacement of other grades. | 20 epochs, best 20, 0.700624 | 20 epochs, best 11, 0.651006 | Historical runner and sampler hashes are in JSON. No matching tracked runner source was found. |
| B4-Q / B4-R | APTOS label-conflict quarantine vs class-matched whole-group removal control. Both remove 50 train rows and leave all 366 validation rows. | Q best 11, 0.688319. R best 9, 0.707897. 20 epochs each. | Inapplicable. DDR's audited training manifest has no conflicting-label exact SHA groups. | An external runner, selection artifacts and audited validation cache are required. The B1 dependency matches 3a563c6, while the B4-specific logic remains external. |

The validation split is reused for checkpoint selection and comparison. B1 improved macro-F1 over B0 in both observed runs. B3 did not improve that aggregate over B1, and B4-Q did not outperform B4-R. Neither the run order nor reuse of B1 augmentation implies a monotonic improvement or a common initialization from the preceding arm. See the [controls and historical lineage](FOUNDATION_TRAINING_READINESS.md).

## E5, E6 and E7 mapping

| Experiment | Authentic implementation and inputs | APTOS2019 | DDR |
| --- | --- | --- | --- |
| E5 integrated | [Runner](../novelty_integrated/e5/runner.py), [model](../novelty_integrated/e5/model.py), [objective](../novelty_integrated/e5/objective.py). HF ConvNeXt V1 with five categorical outputs plus ordered ordinal thresholds. Ordinal BCE weight 0.25 and inverse-square-root class weighting from epoch 11. | [20 epochs, best 16](../results/APTOS2019/e5_integrated/validation_report.md). Macro-F1 0.705791. Separate locked final-test evidence exists. | [20 epochs, best 18](../results/DDR/e5_integrated/validation_report.md). Macro-F1 0.653199. |
| E6 boundary aware | [Runner](../novelty_integrated/e6/runner.py), [objective](../novelty_integrated/e6/objective.py), [decoded input cache](../novelty_integrated/e6/decoded_cache.py). Same head family as E5, with LDAM, deferred effective-number weights and adjacent boundary treatment. | [20 epochs, best 15](../results/APTOS2019/e6_ldam_ordinal_adjacent/validation_report.md). Macro-F1 0.679204. | [20 epochs, best 12](../results/DDR/e6_ldam_ordinal_adjacent/validation_report.md). Macro-F1 0.662455. |
| E7 global/local | [Runner](../novelty_integrated/e7/runner.py), [fusion model](../novelty_integrated/e7/model.py), [geometry](../novelty_integrated/e7/geometry.py). One shared backbone, global 384 px plus two local 224 px views. Intended dataset-matched verified E5 warm start, gated fusion and E5-style objective. Published source and CPU checks establish an implementation. | [Protocol and evidence status](../notebooks/novelty/E7_APTOS2019.ipynb), no verified dataset results in checkout. | [Protocol and evidence status](../notebooks/novelty/E7_DDR.ipynb), no verified dataset results in checkout. |

Published E5/E6 configurations identify the exact run revisions and all four best and last checkpoint digests with sidecar evidence. Original binaries and frozen image manifests remain external. For E7 the established E5 checkpoint digest identifies the intended initialization, not an E7 training outcome. Relative manifest references are preserved. Training commands are guarded, paths are supplied externally, and reports remain independently readable.

## Inventory of histories and figures

| Evidence family | Available epoch history | Real saved figures |
| --- | --- | --- |
| CAD-ViT reproduction | V1 and V2 historical history remains in the campaign archives. Balanced APTOS and DDR recovery each have 80 original epoch rows in `results/`. | The [figure audit](FIGURE_AUDIT.md) inventories historical aggregates. Use the reproduction notebook for the APTOS balanced curve, DDR recovery validation macro-F1 and train loss, and original DDR final-test confusion matrix. The named APTOS image panels require locally available, authorized files. |
| ConvNeXt B0–B4 | 200 original epoch rows across ten runs. | 10 learning-curve SVGs and 10 confusion-matrix PNGs. All filenames and SHA-256 values are in the [source JSON](../novelty/convnext_b0_b4/validation_evidence.json). Both are saved in notebook display outputs and linked directly. |
| E5/E6 | 80 original epoch rows across four runs. Per-grade MAE recovered from original CSV supplements. | [13 PNG/SVG pairs](E5_E6_VALIDATION.md#figure-index) cover curves, objectives, resources, raw and normalized confusion matrices, grade metrics, direction and inter-method differences. Each E5/E6 notebook embeds the relevant published dataset confusion and score panels and links the figure suite. The older E5 APTOS final-test figure remains separate in its original folder. |
| E7 | No verified epoch history in Git for either dataset. | None verified. Geometry diagrams or synthetic checks would not represent measured performance. |
| Optimization branch | Historical stored output in each pinned notebook. | Those notebooks and outputs remain on `novelty`. No new optimization figure or result was generated or imported. |

## Optimization compatibility and proposed protocol

Read the [source audit and protocol](OPTIMIZATION_COMPATIBILITY.md) for specific backend and pruning compatibility per experiment. The first eligible optimization candidate should use its own verified original checkpoint and a reproducibly fixed non-test subset. Full validation metrics cannot stand in for measurements on that subset.
