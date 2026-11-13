# CAD-ViT collaborator guide

## Start here

This repository documents a **partial, paper-guided reproduction** of CAD-ViT for five-grade diabetic retinopathy classification. It is a research baseline, not the authors' original implementation or a clinically validated system. Open the [main research overview](../CAD_ViT.ipynb) first: it compares the six final-test results and links the three detailed records, [V1](../reproduction/experiments/01_experiment_initial_cadvit_v1.ipynb), [V2 BG10](../reproduction/experiments/02_experiment_bg10_cadvit_v2.ipynb), and [balanced/recovery](../reproduction/experiments/03_experiment_balanced_training_and_ddr_recovery.ipynb). All four notebooks are static Markdown and render without a kernel or GPU. The [105-cell original notebook](../reproduction/archive/CAD_ViT_original_105_cells.ipynb) is an immutable evidence snapshot; its [archive note](../reproduction/archive/README.md) records the byte-level hash.

Do not use **Run All** on the archive. Its V1/V2 cells can mount storage, create manifests, write EDA CSVs/checkpoints, train, or evaluate a frozen test split. Opening the report or archive does not run cells.

## Completed experiments and evidence

| Generation | Dataset and run | Training / preprocessing | Final-test evidence | Remaining limit |
| --- | --- | --- | --- | --- |
| V1 historical | APTOS `aptos_main_80ep_seed42`; DDR `ddr_main_80ep_seed42` | Notebook V1 model, preprocessing, split, and runner | Final-test metrics/matrices preserved in archive and main report | Original test JSON and checkpoints external |
| V2 BG10 historical | APTOS `aptos_v2_bg10_stable_80ep_seed42`; DDR `ddr_v2_fp32_weighted_fp64clip_seed42` | 80-epoch histories in archive/main; DDR has three recovery phases | [APTOS JSON](../results/APTOS2019/v2_bg10_historical_80epochs/final_test_v2_80ep.json) and [DDR JSON](../results/DDR/v2_bg10_historical_80epochs/final_test_v2_80ep.json), each matching the recorded server SHA-256 | Checkpoint, manifest, and execution lineage not fully verified; see [V2 provenance](V2_FINAL_TEST_PROVENANCE.md) |
| Balanced campaign | APTOS `balanced_v1/aptos_r16` | V1 preprocessing, train-only oversampling, AMP FP16; best validation epoch 24 | [Training and final-test artifacts](../results/APTOS2019/full_oversampling_amp_fp16_80epochs/) | Manifest/checkpoint binaries external |
| FP32 Math recovery | DDR `cadvit_tables_1_5_balanced_fp32_math_mb8_diag/ddr_full` | V1 preprocessing, train-only oversampling; epoch 1–42 and 43–80 use different protocols; best validation epoch 69 | [Training and final-test artifacts](../results/DDR/full_oversampling_fp32_math_mb8_80epochs/), [archived evaluator source](../reproduction/provenance/ddr_final_test/README.md) | Raw per-image predictions, manifest, checkpoint, and exact recovery runner source external |

The V2 65/30-epoch runs, failed DDR AMP run, and FP32 pilot runs remain separate historical/diagnostic evidence in the archive. Do not merge their curves, metrics, or checkpoint claims with the six final-test rows in the main report. Validation selects a checkpoint; a final test reports its frozen split once. A matching JSON hash proves byte identity with the recorded file, not the checkpoint lineage or execution path.

## Which code is authoritative?

- [`reproduction/cadvit/model.py`](../reproduction/cadvit/model.py) is the reusable candidate architecture. Its model definitions were extracted with AST equivalence from original notebook cells 53/55, and are protected by [`tests/test_model.py`](../tests/test_model.py). It has about 5.28 million parameters versus about 7.26 million in the paper.
- [`reproduction/cadvit/preprocessing.py`](../reproduction/cadvit/preprocessing.py) contains the reusable V1 and V2 BG10 transforms. Four key definitions match original notebook cell 41 by AST and pixel regression tests in [`tests/test_preprocessing.py`](../tests/test_preprocessing.py).
- [`reproduction/cadvit/experiments.py`](../reproduction/cadvit/experiments.py) owns modular campaign protocols, split loaders, training, checkpoint selection, and explicit training/test CLI commands. It is **not** an interchangeable copy of the original notebook's V1/V2 runners. The local file hash matches the APTOS balanced config, but **does not match the recorded DDR Math recovery runner hash**. Preserve this as an unresolved source-version issue; do not silently change a source hash.
- [`reproduction/cadvit/evaluation.py`](../reproduction/cadvit/evaluation.py) provides the separately guarded V2 80-epoch evaluator and five-class metrics. Original notebook cells 85/97 targeted different legacy guards. The two V2 result hashes are documented in [V2 provenance](V2_FINAL_TEST_PROVENANCE.md).
- [`visualization.py`](../reproduction/cadvit/visualization.py), [`preprocessing_audit.py`](../reproduction/cadvit/preprocessing_audit.py), and [`interpretability.py`](../reproduction/cadvit/interpretability.py) support analysis and figures. The [eight numbered thematic notebooks](../reproduction/notebooks/) use these modules for focused inspection; their code is separate from the static experiment records.
- The [archived original notebook](../reproduction/archive/CAD_ViT_original_105_cells.ipynb) is authoritative for the *historical notebook implementation and stored outputs*, not for the later modular DDR recovery runner. The [DDR evaluator archive](../reproduction/provenance/ddr_final_test/README.md) preserves source associated with the documented recovery test, with its own execution-provenance limits. An untracked draft adapter at `reproduction/cadvit/ddr_recovery_final_test.py` was written independently and is not the server evaluator.

Similar function names do not imply identical split policy, sampler, preprocessing, checkpoint format, or execution guard. Check the experiment row, config and source hash before using code to interpret an older result.

## Inspect without a GPU

1. Read [CAD_ViT.ipynb](../CAD_ViT.ipynb) in GitHub for the concise overview. The four complete 80-row tables, result matrices, and figures render statically in the [V2](../reproduction/experiments/02_experiment_bg10_cadvit_v2.ipynb) and [balanced/recovery](../reproduction/experiments/03_experiment_balanced_training_and_ddr_recovery.ipynb) records; V1's available logs are incomplete and are described in its [own record](../reproduction/experiments/01_experiment_initial_cadvit_v1.ipynb).
2. Read the local JSON/CSV/PNG artifacts under [`results/`](../results/). The two V2 JSON files are currently untracked and must be included in a reviewed commit for fresh-clone access. For example, `python -m json.tool results/DDR/full_oversampling_fp32_math_mb8_80epochs/final_test_summary.json` formats the aggregate DDR summary; it does not import PyTorch or run inference.
3. Use [04_classification_evaluation.ipynb](../reproduction/notebooks/04_classification_evaluation.ipynb) for paper and campaign metric inspection and [07_error_analysis.ipynb](../reproduction/notebooks/07_error_analysis.ipynb) for aggregate DDR errors. Some optional cells need external artifact paths. Per-image DDR recovery predictions remain on the server, so no local test-image gallery is claimed.
4. If needed, inspect the archive's saved output. Do not execute its training or final-test cells.

There is no pinned environment/lock file in this repository. If running the numbered notebooks, record the Python and package versions used; treat missing optional dependencies as an environment issue rather than modifying historical outputs.

## Folder and filename guide

| Location | Meaning |
| --- | --- |
| [`CAD_ViT.ipynb`](../CAD_ViT.ipynb) | Short research overview and six-run comparison; open first. |
| [`reproduction/experiments/`](../reproduction/experiments/) | Three static, dataset-separated experiment records; four full 80-epoch tables and final-test matrices live in V2 and balanced/recovery records. |
| [`reproduction/notebooks/`](../reproduction/notebooks/) | Eight existing thematic notebooks with optional code for a specific method or analysis. `preprocessing_analysis.ipynb` is pre-existing separate work and is not part of the numbered set. |
| [`reproduction/archive/`](../reproduction/archive/) | Byte-preserved original 105-cell notebook and reference mirrors for its relative result links. |
| [`reproduction/cadvit/`](../reproduction/cadvit/) | Modular Python implementation and analysis helpers. The untracked DDR draft adapter is not the executed server source. |
| [`reproduction/provenance/`](../reproduction/provenance/) | Archived external evaluator source and execution caveats. |
| [`reproduction/report_figures/`](../reproduction/report_figures/) | Saved or derived figures used by the static reports; its README states each figure's origin. |
| [`results/`](../results/) | Aggregate campaign artifacts. Names such as `full_oversampling_fp32_math_mb8_80epochs` are descriptive run snapshots; retain them for provenance and existing links. |

Legacy filenames, notebook numbers, and result directory names are retained because reports, hashes, and external run records refer to them. New EC²-DR work should use new descriptive paths rather than renaming this baseline.

## New EC²-DR research

Start from a clean working tree on a separate branch, for example `git switch -c research/ec2-dr-prototype`. Keep a new package/protocol and artifact directory for EC²-DR; do not modify baseline `model.py`, `preprocessing.py`, frozen manifests, checkpoints, or historical result files merely to develop the prototype. The method specification for EC²-DR is not implemented here yet.

Before a comparable experiment, freeze the dataset version, manifest bytes/hash and split membership, preprocessing version, metric definitions, validation checkpoint rule, seed, compute precision, and baseline checkpoint identity. Put training artifacts outside Git, give each new run a unique ID, and change one research factor at a time when attributing an effect. Use validation for design and checkpoint selection; evaluate the final test once after decisions are frozen. Record source-file hashes, protocol changes, and any interrupted/recovery phases rather than reusing a baseline label.

Review [the paper experiment plan](PAPER_EXPERIMENT_REPRODUCTION_PLAN.md), [architecture fidelity audit](ARCHITECTURE_FIDELITY_AUDIT.md), and [figure audit](FIGURE_AUDIT.md) for methodological caveats. Some older documents retain historic section references; the archived notebook preserves that numbering.
