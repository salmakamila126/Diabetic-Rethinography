# CAD-ViT preprocessing fidelity matrix

Primary source: [CAD-ViT, IEEE Access 2025, DOI 10.1109/ACCESS.2025.3572455](https://doi.org/10.1109/ACCESS.2025.3572455), Section III.A and Figure 1, checked against the user-provided PDF. The PDF itself is not in this repository. The table distinguishes text supported by that source from the historical notebook and this audit's operationalization.

| Component | Paper evidence | Historical V1 | Historical V2 BG10 | Audit status |
| --- | --- | --- | --- | --- |
| Crop black region | Section III.A describes removal of black image areas; Figure 1 illustrates border removal. | Grayscale mask `gray > 10`, then bounding rectangle. | Inherits V1. | Crop is **paper-described**; threshold **10** is an **implementation assumption**. |
| Resize | Section III.A states uniform **512×512**; area interpolation is discussed. | `cv2.resize(..., (512, 512), interpolation=cv2.INTER_AREA)`. | Inherits V1. | Target dimensions are **explicit in paper**. The exact OpenCV call is an **implementation assumption**; do not infer author code from the discussion of interpolation. |
| Gaussian blur | Section III.A describes smoothing with Gaussian blur. | `cv2.GaussianBlur` with **5×5, σX=σY=1**. | Applies V1 first. | Blur is **paper-described**; kernel and sigma are **implementation assumptions**. |
| EXIF and RGB | No specific EXIF or color-library implementation identified in Section III.A. | `ImageOps.exif_transpose(image).convert("RGB")`. | Inherits V1. | **Historical implementation detail**, not an explicit paper parameter. |
| Ben Graham enhancement | Not stated as a CAD-ViT preprocessing step in Section III.A. | Not present. | Float32 background blur **σ=10**, then `clip(4I − 4B + 128, 0, 255).astype(uint8)`. | **Experimental extension V2**. Its sigma and coefficients are historical V2 settings, not CAD-ViT author settings. |
| `paper_described` audit view | Crop → resize → blur and 512×512 are supported; full numeric implementation is unavailable. | Same code and configuration as this audit view. | Separate extension view. | **Operationalized from paper plus V1 assumptions**. `paper_described` and `historical_v1` must be exactly equal; no artificial difference is introduced. |

## Still unverified

- Author source code, exact crop rule, Gaussian kernel/sigma, interpolation call, EXIF handling, and complete color-space details are not established by Section III.A/Figure 1.
- A side-by-side resemblance to Figure 1 is descriptive. It cannot prove method identity, lesion preservation, or equal downstream classification.
- No actual train or independent audit image has been processed by this new audit notebook yet. Output tables and figures remain empty until the user supplies an allowed image; validation/test images must not be used for exploratory tuning.
