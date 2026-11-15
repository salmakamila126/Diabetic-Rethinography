# E3 frozen-model enhancement sensitivity and foundation-model feasibility

**Status (15 November 2026):** E3 code and synthetic tests are ready for a manual KC validation pilot. No E3 inference or real E3 result has been run here. E0, E1, and E2 findings in this section are operator-reported from KC: E0 did not show convincing incremental error-detection AUPRC from cross-preprocessing response; E1 showed large preprocessing-mismatch degradation; E2 showed substantial representation drift but no positive AUPRC interval in the 16 primary all-layer comparisons. The decision remains **NO-GO** for full PReOF, drift-guided fusion, and selective alignment on current evidence.

## 1. Fixed E3 protocol

[`novelty/preof_e3.py`](../novelty/preof_e3.py) imports the reproduction code without modifying it. It accepts only rows from the SHA-256-verified frozen **validation** manifest and complete Phase 1 prediction cache. For APTOS2019 these contain 366 validation images, and for DDR 1,252. The four historical checkpoints are V1 and V2 for each dataset; Phase 1 `checked_models` verifies their byte hashes, checkpoint records, model/preprocessing source, and historical V2 notebook definitions before loading. No test manifest or image is opened.

The historical [`preprocess_fundus`](../reproduction/cadvit/preprocessing.py) applies EXIF transpose, RGB conversion, grayscale threshold `>10` bounding-box crop, resize to 512×512 with OpenCV `INTER_AREA`, then Gaussian blur `(5,5)` with `sigmaX=sigmaY=1.0`. It produces `uint8` RGB. Historical V2 `preprocess_fundus_v2` starts with that exact V1 result, casts to `float32`, computes Gaussian background with `sigmaX=sigmaY=10`, then `clip(4*pixels - 4*background + 128, 0, 255).astype(uint8)`. The final cast **truncates** fractional values. The V1/V2 checkpoints have different training protocols, so comparisons *between* them do not isolate an enhancement effect.

E3 uses Phase 1's intact V1 and BG10 `.npy` arrays. The cache records bind each array to source-image SHA-256, source/config/library hashes and a pixel digest. E3 requires valid cache hits and never repairs or writes to these caches. All new variants are generated in memory, one image at a time; only predictions, metadata, and aggregate outputs persist outside Git. Inputs to each fixed model are copied `uint8 RGB → CHW float32 /255`, identical to Phase 1. Models run in `eval()` and `torch.inference_mode()`; there is no augmentation, optimizer, gradient, or batch-normalization update.

| Family | Prespecified conditions | Exact pixel operation after historical V1 base |
| --- | --- | --- |
| RGB interpolation | `v1` (α=0), `mix025`, `mix050`, `mix075`, `v2_bg10` (α=1) | Blend cached V1/V2 arrays in `float32` with `(1−α)V1 + αV2`, apply `np.rint` (ties to even), clip to `[0,255]`, cast `uint8`. Endpoints reuse original cached arrays directly. |
| Ben Graham | `bg5`, `v2_bg10`, `bg20` | Apply historical `4*V1 − 4*GaussianBlur(V1, σ) + 128` in `float32`, clip, then truncate to `uint8`, with σ=5/10/20. BG10 is the cached historical V2 endpoint. |
| LAB-CLAHE | `clahe2`, `clahe4` | Convert cached V1 RGB `uint8` to OpenCV LAB, apply CLAHE only to L using tile grid 8×8 and clip limit 2/4, convert LAB to RGB `uint8`. |
| Gamma | `gamma075`, `gamma125` | Compute `255*(V1/255)^γ` in floating point for γ=.75/1.25, `np.rint`, clip, cast `uint8`. γ<1 brightens by this definition. |

There are **11 distinct input conditions**; BG10 is listed in both the interpolation and sigma families but processed only once. The four model/endpoint logits and probabilities per image come from the verified Phase 1 cache. The pilot re-runs the four endpoint forwards on its first image and checks cached logits at `atol=rtol=1e-4` to detect model/device/transport mismatches. It also recomputes BG10 from V1 and checks pixel equality with the cached V2 on each selected image. These checks are numerical and provenance guards, not evidence of clinical equivalence. A failed check stops the run; do not relax tolerance without investigation. The remaining nine condition images per model require new frozen inference. This is sensitivity to inputs at inference time, **not** training-time augmentation robustness.

### Pilot selection and analysis

The default pilot ranks whole `group_id` values separately for each true grade using `SHA-256(seed|dataset|grade|group_id)`, selects up to **8 groups per grade**, unions them, then preserves the manifest row order. A group containing multiple grades remains intact, so the number of images per grade is not fixed in advance. The actual group IDs, five grade supports, image count, seed, and quota are saved in `metadata.json`; analysis checks selection again against the frozen manifest. The deterministic pilot is a **selected subset**, not a representative random sample of the full validation cohort. The smoke procedure can set `--groups-per-grade 1` in a separate output directory.

For each model/condition, E3 reports accuracy, five-grade macro-F1 (absent grades contribute zero), quadratic weighted kappa, grade MAE, confusion matrix, overgrading/undergrading, per-grade support and errors, and wrong predictions with maximum softmax probability ≥0.9. Each condition is paired against that **same model's native input** (V1 model/V1 pixels; V2 model/BG10 pixels). It reports native→condition prediction-transition matrix, correct→wrong and wrong→correct counts, changes in ordinal absolute error and grade direction, all four ordinal-boundary crossings, plus percentile 95% intervals for accuracy and MAE differences from 500 group-level paired bootstrap draws. The bootstrap is descriptive for the selected groups, and many correlated conditions are explored. SVGs show interpolation response at the five prespecified α values and accuracy across the BG, CLAHE, and gamma strength settings; the JSON/Markdown contain all metrics. No “best” condition is selected as a confirmed result. A later confirmation requires a separately prespecified cohort/plan and targeted prior-art review.

The program writes `metadata.json`, resumable `rows.jsonl`, `analysis.json`, `report.md`, `interpolation_response.svg`, and `strength_response.svg` to an **outside-repository** E3 directory. Resume validates metadata and every saved row, including the original endpoint logits, before continuing. If a cache entry is absent/corrupt, investigate Phase 1 provenance; E3 does not regenerate it. A complete pilot exits without inference on resume. Analysis requires a complete pilot and reruns no model. The JSON contains machine-readable results and source/cache hashes. Approximate new prediction payload is `N × 2 models × 9 conditions × (5 logits + 5 probabilities) × 8 bytes ≈ 1.44 kB/image` before JSON overhead; no image variant is stored.

## 2. Manual KC commands

Run only in a reviewed **novelty** checkout on KC, after inspecting the existing cache paths. No command below trains or accesses final test. Both datasets use the same read-only preprocessing root; use distinct output directories. The `--trust-checkpoints` flag is required because project checkpoints use PyTorch pickle loading after exact SHA-256 checks.

```bash
ROOT="$HOME/Kel_13"
REPO="$ROOT/Diabetic-Rethinography"
PY="$ROOT/.venv/bin/python"
V2_SOURCE="$ROOT/Diabetic-Rethinography-v2/CAD_ViT.ipynb"
PREPROCESS_CACHE="$ROOT/preof_phase1/preprocessing"
cd "$REPO"
test "$(git branch --show-current)" = novelty || { echo 'Use novelty checkout'; exit 1; }
test -f "$V2_SOURCE" && test -f novelty/preof_e3.py || { echo 'Required source missing'; exit 1; }
```

**Smoke (separate output from the pilot):**

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -m novelty.preof_e3 pilot --dataset APTOS2019 --artifact-root "$ROOT" --prediction-cache-dir "$ROOT/preof_phase1/aptos_full" --preprocess-cache-root "$PREPROCESS_CACHE" --v2-source "$V2_SOURCE" --trust-checkpoints --device cuda:0 --groups-per-grade 1 --output-dir "$ROOT/preof_e3/aptos_smoke"
CUDA_VISIBLE_DEVICES=0 "$PY" -m novelty.preof_e3 pilot --dataset DDR --artifact-root "$ROOT" --prediction-cache-dir "$ROOT/preof_phase1/ddr_full" --preprocess-cache-root "$PREPROCESS_CACHE" --v2-source "$V2_SOURCE" --trust-checkpoints --device cuda:0 --groups-per-grade 1 --output-dir "$ROOT/preof_e3/ddr_smoke"
```

Inspect the selected group IDs/support and endpoint checks. Then run the **grouped, grade-stratified validation pilots** manually:

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -m novelty.preof_e3 pilot --dataset APTOS2019 --artifact-root "$ROOT" --prediction-cache-dir "$ROOT/preof_phase1/aptos_full" --preprocess-cache-root "$PREPROCESS_CACHE" --v2-source "$V2_SOURCE" --trust-checkpoints --device cuda:0 --groups-per-grade 8 --seed 42 --output-dir "$ROOT/preof_e3/aptos_pilot"
CUDA_VISIBLE_DEVICES=0 "$PY" -m novelty.preof_e3 pilot --dataset DDR --artifact-root "$ROOT" --prediction-cache-dir "$ROOT/preof_phase1/ddr_full" --preprocess-cache-root "$PREPROCESS_CACHE" --v2-source "$V2_SOURCE" --trust-checkpoints --device cuda:0 --groups-per-grade 8 --seed 42 --output-dir "$ROOT/preof_e3/ddr_pilot"
```

If interrupted, rerun the **identical** pilot command with `--resume`. Once complete, CPU analysis runs without checkpoint loading:

```bash
"$PY" -m novelty.preof_e3 analyze --artifact-root "$ROOT" --prediction-cache-dir "$ROOT/preof_phase1/aptos_full" --output-dir "$ROOT/preof_e3/aptos_pilot" --bootstrap 500 --seed 42
"$PY" -m novelty.preof_e3 analyze --artifact-root "$ROOT" --prediction-cache-dir "$ROOT/preof_phase1/ddr_full" --output-dir "$ROOT/preof_e3/ddr_pilot" --bootstrap 500 --seed 42
```

No full-validation E3 expansion is authorized by this pilot protocol. Review pilot grade support, uncertainty, source hashes, and preprocessing endpoint checks before defining any wider run. If the KC source/cache/checkpoints are unavailable, stop and report that blocker; do not reconstruct the caches from test data.

## 3. Foundation-model feasibility audit (no download or training)

| Model | Access and license | Official loading/preprocessing | Frozen vector and planning size |
| --- | --- | --- | --- |
| [`facebook/dinov3-vits16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) | Gated on Hugging Face; the account must agree to share information and accept the [DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md). Confirm institutional permission and model access before any weight retrieval. | Meta/HF recommend `AutoImageProcessor` with `AutoModel`. [Transformers DINOv3 documentation](https://huggingface.co/docs/transformers/model_doc/dinov3) describes RGB conversion, 224² resize, rescale and ImageNet normalization for its processor defaults. The **model-specific gated processor config has not been inspected**; pin revision and inspect it after access before use. ViT-S/16 has 384-dimensional tokens, a class token and register tokens; compare a prespecified `pooler_output`/class representation with patch mean as a distinct analysis. | Model card lists ~21.6M parameters (about 82 MiB at FP32 for weights alone). One 384-dimensional `float32` pooled vector = 1,536 bytes/image. |
| [`facebook/convnext-tiny-224`](https://huggingface.co/facebook/convnext-tiny-224) | Public model card states Apache-2.0; keep license/attribution with the pinned revision. | The [model-specific processor config](https://huggingface.co/facebook/convnext-tiny-224/blob/main/preprocessor_config.json) specifies size 224, `crop_pct=.875`, bicubic (`resample=3`), ImageNet mean/std, and resize/normalization. Inspect the loaded `AutoImageProcessor` to resolve defaults not explicit in the file, especially rescaling and crop execution. Use `AutoModel` or the backbone of `AutoModelForImageClassification`; do not reuse its ImageNet classifier as a DR classifier. | [Transformers ConvNeXt config](https://huggingface.co/docs/transformers/model_doc/convnext) has final hidden width 768; global average pooling yields 768 `float32` values = 3,072 bytes/image. The [Meta ConvNeXt release](https://github.com/facebookresearch/ConvNeXt/blob/main/README.md) lists ConvNeXt-T at about 28M parameters. |

The [Meta DINOv3 repository](https://github.com/facebookresearch/dinov3/blob/main/README.md) documents its model-access process and a Transformers path (version ≥4.56 for native support). The proposed dependencies are PyTorch, compatible `transformers`, Pillow and the Hugging Face authentication client for gated DINOv3 access; `safetensors`/`accelerate` may be required by the chosen loader/device mapping. No dependency was installed here. Verify exact versions and model compatibility on KC in a separate environment before use.

**VRAM:** weight storage alone is not a runtime memory estimate. DINOv3 ViT-S is roughly **82 MiB** in FP32 parameters; 28M ConvNeXt-T parameters require roughly **107 MiB** in FP32. These are lower bounds before feature maps, attention workspaces, CUDA context and allocator overhead; the exact loaded parameter counts and activation peaks must be measured after access. For DINOv3 at 224², patch tokens number `14×14=196` plus special tokens; attention activations scale quadratically with token count. At 512², patch tokens rise to `32×32=1024`, so do not transfer a 224² batch-size assumption to 512². Provisionally plan **at least 2 GiB free VRAM for batch-1 224² inference**, then measure rather than treating this as a guarantee. A safe deployment preflight is batch 1, `eval()` + `inference_mode()`, then `torch.cuda.reset_peak_memory_stats()` / `max_memory_allocated()` and incremental batches, with model revision, dtype, processor config and device recorded. No measured KC VRAM claim is made here.

**Controlled frozen-feature comparison design:** use the existing frozen **training and validation** manifests; verify group separation and image hashes. Extract train and validation embeddings using the exact pinned official processor for each model at 224², no random augmentation, and frozen weights. Keep this arm separate from any CAD-ViT 512² preprocessing arm; otherwise a change in crop/resize and backbone is confounded. Fit one prespecified regularized 5-class linear classifier on **training embeddings only**. Select regularization and any class weighting with grouped inner cross-validation on training groups only, then refit on all training data. Evaluate once on the untouched validation split with accuracy, macro-F1, QWK, MAE and per-grade support/confusion; compare descriptively with the same-manifest CAD-ViT native results. No test-set use or validation fitting. Cache per-image vectors outside Git with dataset/manifest/split/image hashes, model revision and weight digest, processor JSON/digest, library versions, dtype, pooling specification, embedding shape, and vector checksums. One 384+768 pooled pair needs 4,608 raw bytes/image; actual `.npy`/metadata overhead is additional. Training-set disk size is `4,608 × N_train` bytes if both are stored; validation-only APTOS+DDR (1,618 images) would be about 7.11 MiB before overhead.

**Possible later PEFT, not implemented:** LoRA/adapters on selected DINOv3 attention/MLP projections and adapters or selected pointwise ConvNeXt blocks would require exact module-name inspection, train-only tuning, a separate confirmation plan, and comparison against the frozen linear probe. Their trainable parameter count and VRAM cannot be specified before selecting modules and ranks. Neither linear probing nor PEFT, nor combining them with E3 enhancement, constitutes a novelty claim by itself. Any candidate methodological contribution needs a targeted prior-art audit first.

## 4. Interpretation limits and decisions

E3 can estimate how fixed models' **validation predictions** respond to controlled pixel transformations. It cannot identify lesion loss, a clinical mechanism, or training-time enhancement robustness. LAB-CLAHE and gamma are distinct transformations, not monotone points on the same Ben Graham scale. The five interpolation points are pixel blends of two complete preprocessing outputs, not intermediate settings inside the historical pipeline. The pilot has selected-group sampling, limited minority-grade support, correlated conditions, and possible model/device numerical differences; report support and intervals alongside every apparent improvement. Results from this pilot are **exploratory**. A GO/NO-GO for a new method requires prespecified independent confirmation and prior-art review. Foundation-model access, exact processor config, measured VRAM, train-manifest integrity, and dataset sharing permissions are unresolved feasibility gates; no weights were accessed here.
