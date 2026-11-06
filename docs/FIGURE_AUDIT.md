# CAD-ViT Figure 1: preprocessing audit

The source is the user-provided PDF of [CAD-ViT, IEEE Access 2025](https://doi.org/10.1109/ACCESS.2025.3572455), Section III.A and Figure 1. The PDF is outside this repository. Section III.A describes cropping black regions, resizing images to 512×512, and Gaussian blur. It discusses area interpolation, but does not provide the exact OpenCV call or every numeric parameter used by the authors. Figure 1 compares original and processed fundus photographs; visual resemblance alone cannot establish identical code or parameters.

## Views and provenance

| View | Implementation | Interpretation |
| --- | --- | --- |
| `original` | EXIF-transposed RGB rendering of the selected source image | Display baseline; no raw file is changed. |
| `paper_described` | Historical V1 operationalization of the paper's crop → resize → blur sequence | Uses the explicit paper steps plus notebook assumptions. It is deliberately pixel-identical to `historical_v1`. |
| `historical_v1` | `preprocess_fundus()` from `reproduction/cadvit/preprocessing.py` | Exact extraction of the historical notebook function. |
| `historical_v2_bg10` | `preprocess_fundus_v2()` from the same module | V1 followed by `clip(4I − 4GaussianBlur(I, σ=10) + 128, 0, 255)` in float32. This is an experimental extension, not a paper claim. |

The historical code uses EXIF orientation, RGB conversion, grayscale threshold **10** for cropping, `cv2.INTER_AREA` to resize to **512×512**, and Gaussian blur **5×5, σ=1**. Only the target size and operation sequence are explicit in CAD-ViT Section III.A. The remaining numeric choices and exact library calls are implementation assumptions. See [PAPER_FIDELITY_MATRIX.md](PAPER_FIDELITY_MATRIX.md).

## Running the notebook safely

Open `reproduction/notebooks/preprocessing_analysis.ipynb`. Set `CAD_VIT_AUDIT_IMAGE` to a selected fundus image and `CAD_VIT_AUDIT_SPLIT` to `train` or `audit`. For `train`, also set `CAD_VIT_AUDIT_MANIFEST` and `CAD_VIT_AUDIT_DATASET_ROOT`; the notebook checks that the image occurs only in a manifest row with `split=train`. For `audit`, choose an independent image that is not part of validation or test. Without a manifest, membership of an independent audit image cannot be proven automatically. The notebook rejects a declared test/validation split and performs no training or test evaluation.

`CAD_VIT_AUDIT_OUTPUT` optionally selects an output directory. By default, derived files are written under ignored `preprocessing_audit_outputs/`:

- Four PNG variants and `comparison.png`.
- `rgb_histograms.png` plus `rgb_histograms.csv` (256 counts per RGB channel and view).
- `metrics.csv` (one row per view, including the source image SHA-256, split declaration, and EXIF orientation).

The input image is opened read-only. Damaged files raise a clear error. An all-dark image keeps the historical crop error rather than receiving a fabricated processed image. The EXIF orientation tag is recorded in the in-memory result and applied to the displayed original as well as both historical pipelines.

## Metric definitions

- **Brightness:** mean of `0.299R + 0.587G + 0.114B`, with channels in 0–255.
- **Contrast:** population standard deviation (`ddof=0`) of the same luminance.
- **Mean RGB distance from original:** Euclidean distance between the two global mean RGB vectors, in channel-value units. Cropping can change this value without any color transform.
- **Pixel MAE from V1:** mean absolute RGB pixel difference from `historical_v1`, calculated only for aligned 512×512 processed views; the original has no pixel MAE entry.
- **RGB histogram:** number of pixels at each integer intensity 0–255 for each channel, without density normalization.

The histogram PNG scales each panel to its own maximum count for readability; compare absolute counts using the CSV rather than the line heights across panels.

Interpret border removal, smoothing, and possible V2 halos only from the actual selected image. Neither one-image appearance nor histogram similarity establishes equal lesion preservation, equivalent model performance, or exact fidelity to the authors' preprocessing. No real-image audit result is asserted until the notebook is run on a permitted image.
