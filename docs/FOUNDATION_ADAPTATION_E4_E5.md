# E4/E5 — controlled foundation adaptation and preprocessing sensitivity

**Status:** implementation and synthetic tests only. No foundation-model weight, KC dataset, or GPU was accessed locally. The operator reports E3 completed on KC with 43 APTOS2019 and 40 DDR validation images across the prespecified 11 conditions; those pilot observations are exploratory. They do not justify E3 expansion, PReOF fusion, or a methodological novelty claim.

## Model and dependency audit

The implementation is [`novelty/foundation_adaptation.py`](../novelty/foundation_adaptation.py). It accepts only a **local, full-commit pinned Hugging Face snapshot** outside Git. The script checks the snapshot directory name, `config.json`, `preprocessor_config.json`, and the SHA-256 of the single official weight file before loading. `AutoModel` and `AutoImageProcessor` are called with `local_files_only=True` and `trust_remote_code=False`; the processor uses `use_fast=False` to keep its implementation choice fixed. The declared revision is recorded, but a directory name plus file hashes alone do not prove Hub lineage; retain the `hf download --revision` receipt and Hub commit URL. No script path invokes `snapshot_download`, HTTP, installation, or a test-set evaluator.

| Short name | Official model | Access/license | Weights and processor | Audited shape at 224² |
| --- | --- | --- | --- | --- |
| `dinov3_vits16` | [Meta DINOv3 ViT-S/16](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) | **Gated**; access request and [DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md). | `model.safetensors`; the gated model-specific processor config cannot be read anonymously. Use the exact downloaded config; `AutoImageProcessor` output and effective settings are recorded at smoke/extraction. [DINOv3 processor documentation](https://huggingface.co/docs/transformers/v4.57.3/model_doc/dinov3) gives the library behavior, not a substitute for the gated config. | `pixel_values [B,3,224,224]`, 1 CLS + 4 registers + 196 patches, `last_hidden_state [B,201,384]`, CLS `pooler_output [B,384]`. |
| `convnext_tiny` | [Meta ConvNeXt Tiny ImageNet-1k](https://huggingface.co/facebook/convnext-tiny-224) | Public, Apache-2.0. | `pytorch_model.bin` is a legacy pickle weight: explicit `--trust-legacy-bin` after hash check. Its [processor config](https://huggingface.co/facebook/convnext-tiny-224/blob/6166b7613034066690a621d8bf25ffdf181a34f0/preprocessor_config.json) records size 224, crop fraction .875, bicubic resampling, ImageNet mean/std, and normalization; the loaded processor records any implicit defaults. | `pixel_values [B,3,224,224]`, final map `[B,768,7,7]`, normalized global-average `pooler_output [B,768]`. |
| `dinov3_convnext_tiny` (optional) | [Meta DINOv3 ConvNeXt Tiny LVD-1689M](https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m) | **Gated**, DINOv3 license. Distilled using the DINOv3 protocol; a useful pretraining comparator, not an architecture-only comparison. | `model.safetensors`; gated processor config needs access. Frozen probing is supported; partial block selection is deliberately disabled until its loaded module tree is audited. | Expected 768-dimensional pooled vector; actual hidden shape/processor behavior must pass smoke. |

The public ConvNeXt revision visible in the [official Hub commit](https://huggingface.co/facebook/convnext-tiny-224/commit/6166b7613034066690a621d8bf25ffdf181a34f0) is `6166b7613034066690a621d8bf25ffdf181a34f0`. **The two gated DINOv3 full commit IDs and their processor bytes remain unverified without authorized access**; obtain and pin them manually before any run. The [Meta DINOv3 model card](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) lists ~21.6M ViT-S parameters; the [DINOv3 ConvNeXt card](https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m) lists ~27.8M; the [Meta ConvNeXt release](https://github.com/facebookresearch/ConvNeXt/blob/main/README.md) lists ConvNeXt-T near 28M. The code measures exact loaded and trainable counts during smoke and stores them with run summaries.

`transformers==4.57.3` is deliberately pinned. Its [DINOv3 implementation](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/dinov3_vit/modeling_dinov3_vit.py) exposes twelve blocks as `backbone.layer[0:12]`, final `norm`, and CLS `pooler_output`. The ViT path requests eager attention for a fixed implementation choice. The [ConvNeXt implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/convnext/modeling_convnext.py) exposes `encoder.stages`, three blocks in the final stage, and final `layernorm`; the code checks that layout at runtime. A future library version may have different module names, so it is rejected until re-audited. PyTorch 2.7.1 CUDA wheels are documented in the [official prior-version installation guide](https://pytorch.org/get-started/previous-versions/). The new environment must be separate from reproduction's `.venv`.

## Shared train/validation design and primary full fine-tuning

The existing SHA-256-pinned APTOS2019 and DDR CSV manifests are parsed only for `train` and `val` records; the entire file is hashed as bytes, while test labels and images are not used. Metadata root/seed, row validity, unique IDs/paths, train/validation group separation, content-hash separation, grade support and **every train/validation image SHA-256** are checked before work. Images are decoded with PIL, EXIF-transposed, converted to RGB, then passed to each model's pinned **official processor**. No CAD-ViT crop/blur, BG enhancement, gamma, CLAHE, random flip, or other enhancement augmentation is used in the initial baseline. The effective processor dictionary and hashes are recorded. The one trainable head is `Linear(d,5)` with ordinary cross entropy; no ImageNet classifier is reused.

`extract` uses a frozen backbone in `eval()`/`inference_mode()`, writes only the official `pooler_output` as `float32`, and never stores full feature maps. One `features.npy` memmap contains train rows followed by validation rows. The append-only `rows.jsonl` binds split, image, grade, group, source-image hash, and vector hash to each row. `metadata.json` binds manifest, model revision/weight/config, processor, runtime, feature dimension, and implementation hash. Resume validates the entire saved prefix and fails on corruption. **Only `mode=frozen` consumes this cache.** Partial fine-tuning reads and reprocesses train/validation images each epoch, so it cannot silently use stale frozen embeddings.

**Primary protocol: conventional full fine-tuning.** For the two primary backbones, `mode=full` loads the hash-checked original official pretrained weights, discards any pretrained classifier, initializes a seeded five-class `Linear(d,5)` head, and makes **every backbone and head parameter trainable**. The backbone uses training mode during each train epoch and eval mode for validation. AdamW has separate backbone and head parameter groups with configurable learning rates. Full mode reads images through the official processor on every epoch and rejects any frozen-feature-cache argument. The optional DINOv3 ConvNeXt comparator is excluded from full tuning pending its own audit.

The retained secondary partial mode unfreezes ViT-S **blocks 10–11 plus final norm**, or ConvNeXt-T **the last two residual blocks of stage 3 plus final layer norm**. All earlier backbone parameters remain frozen and the backbone stays in eval behavior while gradients flow through the selected blocks. Frozen linear probing and its persistent cache also remain available. Neither secondary path is required before full tuning. `peft` is an explicit unimplemented extension point: LoRA/adapters need a separate reviewed protocol.

**Primary full-tuning budget:** seed 42, **20 epochs**, effective batch **32**, AdamW, weight decay .01, head LR 1e-3, backbone LR 1e-5, gradient clip 1.0, no early stopping or enhancement augmentation. Prespecified micro-batches are **1, 4, 8, 16, 32**, paired with accumulation **32, 8, 4, 2, 1** respectively; the code rejects other full-mode pairs. Each incomplete last window is divided by its actual number of images, so every image has equal gradient weight. Each epoch records all optimizer window image counts, including the final window. The same seed, loss, manifest, evaluation and selection rule apply across batch-size trials, but changed micro-batch can still affect numerical order and training behavior; select one feasible size from resource benchmarking before the run and report it.

CUDA BF16 is supported only if PyTorch reports it on the selected GPU. FP16 uses gradient scaling; FP32 remains available. The two-step disposable smoke at each batch size measures peak allocated/reserved VRAM **after optimizer state exists**, free/total device memory, and model-step throughput excluding image decoding and processor time; it saves no modified weights. Training requires a matching smoke report for the chosen micro-batch. On shared 12 GB RTX 3060 GPUs, inspect free memory before and during training; a successful benchmark does not guarantee immunity to later contention. Non-finite loss or gradients stop the run. The full-mode checkpoint stores **all** fine-tuned backbone and head parameters, optimizer, scaler, RNG and history in epoch-level `last.pt` and validation-selected `best.pt`, with SHA-256 sidecars outside Git. Explicit trusted resume restores that state. Frozen and partial modes retain the same 20-epoch defaults but are secondary protocols.

**Checkpoint selection:** maximize five-grade **validation macro-F1**, tie-break by lower validation cross entropy, then earliest epoch. No checkpoint is chosen from final test or E3 sensitivity. Report per epoch: train/validation loss, accuracy, macro-F1, weighted-F1, QWK, grade MAE, per-grade recall/support, true-row confusion matrix, wall time, images/s, and peak GPU allocated/reserved bytes. Total/trainable parameter counts are recorded at smoke and in the completed run summary. The final summary points to the selected epoch and says `NOT_ACCESSED` for final test. This reuses historical validation data for tuning and is **not independent confirmation**.

## E5 later sensitivity on the E3 pilot

`sensitivity` is a separate **manual** command after a run has a validation-selected best checkpoint. It verifies the exact E3 pilot `metadata.json` and `rows.jsonl`, including image order, grade, group and hash, and evaluates only those same validation images. It reports an official raw-image processor anchor plus all 11 E3 variants for the **fixed trained checkpoint**. The E3 arrays come from the read-only Phase 1 preprocessing cache; intermediate variants are transient via [`preof_e3.py`](../novelty/preof_e3.py). This does not extend E3's cohort and does not retrain. The raw official arm and the E3 variant arms start from different upstream preprocessing, so their contrast cannot isolate one parameter. The selected validation subset was already involved in checkpoint selection; E5 is exploratory. Phase 1 cache keys include NumPy/OpenCV/Pillow versions: the isolated environment must match the cache versions for E5 or fail closed. Do not regenerate or alter Phase 1 caches to make E5 pass.

## Storage and remaining confounders

The frozen manifests imply 3,296 APTOS and 11,270 DDR **train+validation** images. Raw pooled-vector payload per image is 1,536 bytes for ViT-S and 3,072 bytes for either ConvNeXt Tiny. Both primary model caches across both datasets need about **64.0 MiB** of vector payload, plus `.npy` header, row index and metadata. The optional comparator adds about **42.7 MiB**. Hub snapshots require roughly 86 MB for ViT-S, 114 MB for ImageNet ConvNeXt-T, and 111 MB for DINOv3 ConvNeXt-T weights before configs and cache overhead, based on official Hub file listings. Partial checkpoints include the full backbone, selected-block optimizer state, and RNG; reserve several hundred MB per run for `last.pt` and `best.pt`, then verify actual KC sizes. None belongs in Git.

Backbones differ in architecture, pretraining data/objective (notably ImageNet-1k versus LVD-1689M), model size, official resize/crop, pooling, and weight format. The shared five-class head, manifest, seed, epoch/step policy, loss and reporting narrow but do not erase these confounders. Frozen and partial modes also differ in precision path and amount of trainable capacity. Small validation minority-grade support and multiple exploratory comparisons limit inference. Neither backbone substitution, conventional fine-tuning, PEFT, nor enhancement augmentation is a novelty claim. A proposed contribution needs a targeted prior-art audit and separate prespecified confirmation.

## Manual KC setup and commands

Run only after reviewing the code on the KC novelty checkout and confirming dataset/license access. The following setup is **manual**; it does not run from this repository. Choose the CUDA wheel matching the KC driver; the shown CUDA 12.6 wheel is an [official PyTorch 2.7.1 option](https://pytorch.org/get-started/previous-versions/). If the driver cannot support it, stop and select a compatible official wheel before proceeding.

```bash
ROOT="$HOME/Kel_13"
REPO="$ROOT/Diabetic-Rethinography"
cd "$REPO"
test "$(git branch --show-current)" = novelty || { echo 'Use the novelty checkout'; exit 1; }
python3 -m venv "$ROOT/.venv_foundation"
PY="$ROOT/.venv_foundation/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
"$PY" -m pip install transformers==4.57.3 numpy==1.26.4 Pillow==11.1.0 huggingface_hub==0.36.0 safetensors==0.5.3
"$PY" -m pip freeze > "$ROOT/foundation_dependency_freeze.txt"
"$PY" -c 'import torch,transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)'
```

Get authorized DINOv3 access through the model page, review its license, then authenticate manually. The gated revision may require a browser or authenticated Git credential; `git ls-remote` retrieves **commit metadata only**, not weights. If it fails, obtain the full commit SHA from the authorized Hub page and set the variable explicitly. The public ConvNeXt revision below is already identified; check that it still matches the intended snapshot.

```bash
HF="$ROOT/.venv_foundation/bin/hf"
"$HF" auth login
VIT_REV=$(git ls-remote https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m HEAD | awk '{print $1}')
CONV_REV=6166b7613034066690a621d8bf25ffdf181a34f0
test "${#VIT_REV}" -eq 40 || { echo 'Set an authorized full DINOv3 commit SHA'; exit 1; }
VIT_DIR=$("$HF" download facebook/dinov3-vits16-pretrain-lvd1689m --revision "$VIT_REV" --cache-dir "$ROOT/hf_foundation")
CONV_DIR=$("$HF" download facebook/convnext-tiny-224 --revision "$CONV_REV" --cache-dir "$ROOT/hf_foundation")
VIT_SHA=$(sha256sum "$VIT_DIR/model.safetensors" | awk '{print $1}')
CONV_SHA=$(sha256sum "$CONV_DIR/pytorch_model.bin" | awk '{print $1}')
printf '%s %s %s\n' "$VIT_REV" "$VIT_DIR" "$VIT_SHA" "$CONV_REV" "$CONV_DIR" "$CONV_SHA" > "$ROOT/foundation_download_receipts.txt"
```

For each primary model, set the variables below (the second line is the ConvNeXt alternative), then repeat for `APTOS2019` and `DDR`. `preflight` hashes every train/validation image without loading a model. The example uses BF16; if the selected GPU fails the BF16 smoke, record that outcome and use a separately named FP16 or FP32 protocol rather than silently changing precision. For ConvNeXt, `LOAD_FLAG` explicitly permits the verified legacy `.bin` weight.

```bash
MODEL_KIND=dinov3_vits16; MODEL_DIR="$VIT_DIR"; REV="$VIT_REV"; WEIGHT_SHA="$VIT_SHA"; LOAD_FLAG=""
# ConvNeXt alternative:
# MODEL_KIND=convnext_tiny; MODEL_DIR="$CONV_DIR"; REV="$CONV_REV"; WEIGHT_SHA="$CONV_SHA"; LOAD_FLAG="--trust-legacy-bin"
DATASET=APTOS2019
# Repeat the complete procedure with DATASET=DDR.
BASE="$ROOT/foundation_e4/$DATASET/$MODEL_KIND"
"$PY" -m novelty.foundation_adaptation preflight --dataset "$DATASET" --artifact-root "$ROOT" --model-kind "$MODEL_KIND" --model-dir "$MODEL_DIR" --revision "$REV" --weight-sha256 "$WEIGHT_SHA"
```

Run the batch-1 smoke first, then benchmark the remaining prespecified sizes. Each size starts from the **original official snapshot** in a fresh process, takes two disposable forward/backward/AdamW steps, and writes a separate JSON. An out-of-memory size can fail without altering earlier reports; investigate failures before selecting a feasible micro-batch. No benchmark weights are saved.

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -m novelty.foundation_adaptation smoke --dataset "$DATASET" --artifact-root "$ROOT" --model-kind "$MODEL_KIND" --model-dir "$MODEL_DIR" --revision "$REV" --weight-sha256 "$WEIGHT_SHA" $LOAD_FLAG --mode full --batch-size 1 --device cuda:0 --precision bf16 --seed 42 --output-dir "$BASE/full_smoke_b1"
for BS in 4 8 16 32; do
  CUDA_VISIBLE_DEVICES=0 "$PY" -m novelty.foundation_adaptation smoke --dataset "$DATASET" --artifact-root "$ROOT" --model-kind "$MODEL_KIND" --model-dir "$MODEL_DIR" --revision "$REV" --weight-sha256 "$WEIGHT_SHA" $LOAD_FLAG --mode full --batch-size "$BS" --device cuda:0 --precision bf16 --seed 42 --output-dir "$BASE/full_smoke_b$BS" || echo "Batch $BS did not pass; inspect the error and GPU state"
done
```

In a **separate KC terminal**, monitor the shared GPU during smoke and training:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv -l 2
```

Choose a passing micro-batch after reviewing peak reserved bytes and available shared memory. The following full-mode command uses `MICRO=4` only as a **command example**; set it to a measured passing size before the run. The accumulation factor keeps the nominal batch at 32. The final short window is recorded separately and is not padded or dropped. Full fine-tuning never reads `frozen_vectors`.

```bash
MICRO=4
ACCUM=$((32 / MICRO))
test -f "$BASE/full_smoke_b$MICRO/smoke.json" || { echo 'Matching smoke report required'; exit 1; }
CUDA_VISIBLE_DEVICES=0 "$PY" -m novelty.foundation_adaptation train --dataset "$DATASET" --artifact-root "$ROOT" --model-kind "$MODEL_KIND" --model-dir "$MODEL_DIR" --revision "$REV" --weight-sha256 "$WEIGHT_SHA" $LOAD_FLAG --mode full --device cuda:0 --precision bf16 --seed 42 --smoke-report "$BASE/full_smoke_b$MICRO/smoke.json" --output-dir "$BASE/full_seed42_b$MICRO" --epochs 20 --micro-batch "$MICRO" --accumulation "$ACCUM" --head-lr 0.001 --backbone-lr 0.00001
```

For an interrupted run, repeat that **exact training command**, adding `--resume --trust-run-checkpoint`. Keep `MICRO`, `ACCUM`, the pinned snapshot, precision, learning rates, and all other flags unchanged. The resume guard checks run metadata and checkpoint hashes. Example:

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -m novelty.foundation_adaptation train --dataset "$DATASET" --artifact-root "$ROOT" --model-kind "$MODEL_KIND" --model-dir "$MODEL_DIR" --revision "$REV" --weight-sha256 "$WEIGHT_SHA" $LOAD_FLAG --mode full --device cuda:0 --precision bf16 --seed 42 --smoke-report "$BASE/full_smoke_b$MICRO/smoke.json" --output-dir "$BASE/full_seed42_b$MICRO" --epochs 20 --micro-batch "$MICRO" --accumulation "$ACCUM" --head-lr 0.001 --backbone-lr 0.00001 --resume --trust-run-checkpoint
```

The retained **secondary** modes can still be run under their own output directories. Frozen probing first extracts verified embeddings; partial tuning uses images directly and rejects that feature cache. They are not prerequisites for full tuning.

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -m novelty.foundation_adaptation smoke --dataset "$DATASET" --artifact-root "$ROOT" --model-kind "$MODEL_KIND" --model-dir "$MODEL_DIR" --revision "$REV" --weight-sha256 "$WEIGHT_SHA" $LOAD_FLAG --mode frozen --batch-size 1 --device cuda:0 --precision auto --output-dir "$BASE/smoke_frozen"
CUDA_VISIBLE_DEVICES=0 "$PY" -m novelty.foundation_adaptation smoke --dataset "$DATASET" --artifact-root "$ROOT" --model-kind "$MODEL_KIND" --model-dir "$MODEL_DIR" --revision "$REV" --weight-sha256 "$WEIGHT_SHA" $LOAD_FLAG --mode partial --batch-size 1 --device cuda:0 --precision auto --output-dir "$BASE/smoke_partial"
CUDA_VISIBLE_DEVICES=0 "$PY" -m novelty.foundation_adaptation extract --dataset "$DATASET" --artifact-root "$ROOT" --model-kind "$MODEL_KIND" --model-dir "$MODEL_DIR" --revision "$REV" --weight-sha256 "$WEIGHT_SHA" $LOAD_FLAG --device cuda:0 --feature-cache-dir "$BASE/frozen_vectors"
CUDA_VISIBLE_DEVICES=0 "$PY" -m novelty.foundation_adaptation train --dataset "$DATASET" --artifact-root "$ROOT" --model-kind "$MODEL_KIND" --model-dir "$MODEL_DIR" --revision "$REV" --weight-sha256 "$WEIGHT_SHA" $LOAD_FLAG --mode frozen --device cuda:0 --precision auto --feature-cache-dir "$BASE/frozen_vectors" --smoke-report "$BASE/smoke_frozen/smoke.json" --output-dir "$BASE/probe_seed42" --epochs 20 --micro-batch 1 --accumulation 32 --head-lr 0.001 --backbone-lr 0.00001
CUDA_VISIBLE_DEVICES=0 "$PY" -m novelty.foundation_adaptation train --dataset "$DATASET" --artifact-root "$ROOT" --model-kind "$MODEL_KIND" --model-dir "$MODEL_DIR" --revision "$REV" --weight-sha256 "$WEIGHT_SHA" $LOAD_FLAG --mode partial --device cuda:0 --precision auto --smoke-report "$BASE/smoke_partial/smoke.json" --output-dir "$BASE/partial_seed42" --epochs 20 --micro-batch 1 --accumulation 32 --head-lr 0.001 --backbone-lr 0.00001
```

`best.pt` and `last.pt` are project-created pickle files outside Git; the explicit resume flag is restricted to the hash-checked run directory. Do not point it at an unrelated checkpoint. The operator may later run `sensitivity` on the exact E3 pilot after confirming the isolated environment's NumPy/OpenCV/Pillow versions match the Phase 1 cache. OpenCV is needed only for E5, not for the initial adaptation baseline. Inspect a cache metadata file and install the recorded OpenCV version in the isolated environment; if the NumPy or Pillow versions differ, use a compatible environment matching all three rather than altering the existing cache. This is not part of the initial baseline:

```bash
CACHE_META=$(find "$ROOT/preof_phase1/preprocessing/v1" -type f -name '*.json' -print -quit)
test -n "$CACHE_META" || { echo 'Phase 1 cache metadata missing'; exit 1; }
"$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["payload"]["libraries"])' "$CACHE_META"
CV_VER=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["payload"]["libraries"]["opencv"])' "$CACHE_META")
"$PY" -m pip install "opencv-python-headless==$CV_VER"
"$PY" -c 'import cv2,numpy,PIL; print({"opencv":cv2.__version__,"numpy":numpy.__version__,"pillow":PIL.__version__})'
```

```bash
E3_PILOT="$ROOT/preof_e3/aptos_pilot"
# For DATASET=DDR, set E3_PILOT="$ROOT/preof_e3/ddr_pilot" instead.
CUDA_VISIBLE_DEVICES=0 "$PY" -m novelty.foundation_adaptation sensitivity --dataset "$DATASET" --artifact-root "$ROOT" --model-kind "$MODEL_KIND" --model-dir "$MODEL_DIR" --revision "$REV" --weight-sha256 "$WEIGHT_SHA" $LOAD_FLAG --device cuda:0 --run-dir "$BASE/full_seed42_b$MICRO" --e3-pilot-dir "$E3_PILOT" --preprocess-cache-root "$ROOT/preof_phase1/preprocessing" --output-dir "$BASE/full_seed42_b${MICRO}_e3_sensitivity" --trust-run-checkpoint
```

The optional DINOv3 ConvNeXt comparator requires separate authorized revision/processor verification and has **frozen-probe support only**. Do not launch its partial mode until the final-block layout and fair budget are reviewed. No real model performance, VRAM value, or E5 sensitivity result is claimed in this document.
