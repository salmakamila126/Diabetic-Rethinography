# Optimization notebooks

The [research notebooks](../README.md) document original runs. This directory holds independent optimization protocols and views of external campaign results. The [HPC guide](../../novelty_integrated/optimization/README.md#staged-hpc-campaign) supplies the audit, preparation, isolated measurement and explicit recovery commands.

| Original checkpoint | APTOS2019 | DDR |
| --- | --- | --- |
| B0 | [Notebook](B0_APTOS2019.ipynb) | [Notebook](B0_DDR.ipynb) |
| B1 | [Notebook](B1_APTOS2019.ipynb) | [Notebook](B1_DDR.ipynb) |
| B2 | [Notebook](B2_APTOS2019.ipynb) | [Notebook](B2_DDR.ipynb) |
| B3 | [Notebook](B3_APTOS2019.ipynb) | [Notebook](B3_DDR.ipynb) |
| B4-Q | [Notebook](B4-Q_APTOS2019.ipynb) | Not applicable |
| B4-R | [Notebook](B4-R_APTOS2019.ipynb) | Not applicable |
| E5 | [Priority 1](E5_APTOS2019.ipynb) | [Priority 4](E5_DDR.ipynb) |
| E6 | [Priority 2](E6_APTOS2019.ipynb) | [Priority 5](E6_DDR.ipynb) |
| E7 | [Priority 3](E7_APTOS2019.ipynb) | [Partial training evidence, optimization not run](E7_DDR.ipynb) |

Each Jupyter notebook defaults to read-only. With an external campaign path set, it displays only completed reports that match the job, checkpoint and frozen manifest. GitHub shows the static rationale, receipt and `NOT_RUN` state until measured, sanitized evidence is available for a separate publication review.

Regenerate the templates with `python notebooks/build_optimization_notebooks.py` from the repository root. The generator needs `nbformat` and the repository's Python environment. It reads published receipts and performs no training, export or inference.
