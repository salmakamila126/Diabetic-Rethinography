# Foundation adaptation: final training readiness audit

**16 November 2026.** This is a validation-only research audit. The ConvNeXt results below were supplied by the KC operator; their checkpoint, manifest, history JSON and log bytes were **not available locally for independent verification**. They remain completed vanilla full-fine-tuning baselines. No change in this document authorizes editing those runs, accessing final test, or claiming clinical readiness. The initial audit added observability. Approved B0/B1/B2 implementation is described below; no KC experiment was run here.

## Evidence and current configuration

Both reported ConvNeXt-Tiny runs used the same frozen dataset manifests, the official 224px processor, 20 epochs, seed 42, physical/effective batch 32, BF16, ordinary five-class cross entropy, AdamW, backbone LR `1e-5`, head LR `1e-3`, weight decay `0.01`, no augmentation or scheduler, and validation macro-F1 checkpoint selection. The operator recorded runner SHA-256 `289d50684fe9d8ca06c9237d5318ebc449346a6e51f75a659f3ecfcfc0e112b1`.

| Reported validation evidence | APTOS2019 ConvNeXt | DDR ConvNeXt |
| --- | ---: | ---: |
| Best epoch | 17 | 8 |
| Best-checkpoint accuracy / macro-F1 / QWK | 0.814208 / 0.673686 / 0.889230 | 0.807508 / 0.619817 / 0.820307 |
| Train loss, first → last | 0.816191 → 0.041576 | 0.7506 → approximately 0.0007 |
| Minimum validation loss | approximately 0.500 at epoch 6 | approximately 0.5726 at epoch 4 |
| Final validation loss | 0.865305 | 1.725003 |
| Final epoch macro-F1 | not supplied | 0.591462 |
| Grade 1 F1 / recall | not supplied | 0.121951 / 0.079365 |
| Mean train throughput | 10.47 images/s | 15.17 images/s |
| Train plus validation time | 6,298 s | 14,710 s |

The divergence between very low training loss and rising validation loss is **consistent with overfitting**. It does not identify a clinical cause. DDR grade 1 is weak on the supplied metrics; its support and confusion counts must be read from the actual validation artifact before making a more precise claim. Validation loss minima precede the macro-F1-selected best epochs: these are different selection criteria. Training accuracy and macro-F1 in the revised logger are **online epoch metrics from batches seen while weights are changing**; they are not a second fixed-checkpoint train evaluation. The completed 20-epoch checkpoints must remain byte-preserved. Extending vanilla DDR merely because an 80-epoch ceiling exists is not supported by these curves.

## Controlled baseline decisions

All proposals below start from the **same official pretrained weights in new run directories**. Keep the frozen train/validation manifests, five-grade definitions, checkpoint selection, seed, batch policy and metric definitions fixed. Compare one intervention at a time against the archived vanilla run and report all grades, calibration, validation loss and resource use. Reusing validation for candidate choice makes all such comparisons exploratory; a later independent confirmation plan is required.

| Factor | Current | Controlled arm or later proposal | Evidence and trade-off |
| --- | --- | --- | --- |
| Conservative retinal augmentation | None | B1 implemented, not yet executed: train only; horizontal flip `p=0.50`; rotation `p=0.50`, angle uniform `[-7°, +7°]`; brightness `p=0.30`, factor `[0.90,1.10]`; contrast `p=0.30`, factor `[0.90,1.10]`. Apply in this order to raw RGB **before** the official processor. No vertical flip, shear, random crop, cutout, blur, hue, saturation, CLAHE or BG enhancement in the first arm. Require a field-of-view mask check with at least 99% of original visible retinal area retained; reject/resample a transform that fails. Validation remains deterministic. | The [CAD-ViT paper](https://doi.org/10.1109/ACCESS.2025.3572455) describes crop/resize/Gaussian preprocessing, while the [historical notebook record](../reproduction/experiments/01_experiment_initial_cadvit_v1.ipynb) marks its geometric augmentation settings as implementation assumptions. Retinal studies use flips/rotation/brightness changes ([fundus augmentation study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8198489/)); **the probabilities and narrow ranges here are conservative proposals**, not numbers established by that paper. Geometric operations can remove peripheral retina; color shifts can suppress faint lesions, so inspect generated examples by grade and reject failed FOV masks before training. |
| Minority-class objective | Ordinary CE | **B2 implemented:** train-only class weights `w_c ∝ 1/sqrt(n_c)`, normalized to training-sample mean 1, with explicit per-image weighted loss divided by the actual accumulation-window image count. **Later, unimplemented arm:** unweighted CE with train-only inverse-frequency sampling `p_i ∝ 1/n_{y_i}`, replacement and exactly `N_train` draws per epoch; log unique images and repeat counts. Do not combine interventions. | DDR grade 1 F1/recall are poor. [PyTorch CE](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) supports class weights; [WeightedRandomSampler](https://docs.pytorch.org/docs/stable/data.html) supports replacement sampling. Weighting changes the objective; sampling repeats minority images and may increase memorization. PyTorch's default weighted-CE mean divides by summed sample weights, so the current image-count accumulation formula must **not** be reused blindly for that default reduction. B2 implements this explicit per-image normalization. |
| Learning rate and regularization | Backbone `1e-5`, head `1e-3`, weight decay `0.01` | First compare head LR `3e-4` against `1e-3` at fixed backbone LR/20 epochs, or weight decay `0.05` against `0.01` as a distinct arm. Label smoothing `0.05` is another separate arm, only after class-wise/calibration review. | Head LR is 100× backbone LR and train loss collapses, but this does **not prove** the head LR caused overfitting. Lower LR or more decay can underfit; smoothing can impair minority sensitivity and probability calibration. Change one factor per arm. [PyTorch CE](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) defines label smoothing. |
| Schedule and duration | Constant LR; 20 epochs | Compare constant-LR 20-epoch baseline to a **new** 20-epoch 2-epoch linear warmup (0.1× to 1× each group LR) plus cosine decay to 0.1× base LR by epoch 20. Consider a distinct **maximum 80-epoch** run only if 20-epoch validation trends warrant it; for that arm predeclare 4-epoch warmup and cosine decay to 0.1× by epoch 80. Validate every epoch and retain the macro-F1-selected best. | [PyTorch CosineAnnealingLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html) is conventional and changes each optimizer group's rate. More epochs increase compute and may worsen observed overfitting. The supplied 20-epoch durations imply rough fourfold wall time before contention/cache effects; do not treat that as a measured forecast. A scheduler needs its own checkpointed state and resume test before implementation. |
| Input resolution / field of view | Official 224px; ConvNeXt `crop_pct=0.875` | First audit visible retinal-mask area before/after the exact pinned processor by dataset/grade. Then pilot **224 center crop vs 224 FOV-preserving resize/pad** to isolate crop policy. Only after that, pilot **224 vs 384** with the same FOV-preserving policy, official weight initialization and an independent memory smoke. Keep all pilots separate from the main 224 protocol. | The [official ConvNeXt processor config](https://huggingface.co/facebook/convnext-tiny-224/blob/375cd43761c9deb0e5877f0343fba4a6918e16b8/preprocessor_config.json) specifies `crop_pct=0.875`; exact effective transform must be read from the pinned KC snapshot. Higher-resolution retinal studies report possible gains ([resolution study](https://pmc.ncbi.nlm.nih.gov/articles/PMC6656880/)), but resolution, crop policy, VRAM and compute can confound the comparison. DINOv3's gated processor and 384px behavior require their own audit; do not silently change either model's main protocol. |

**Decision order:** verify original histories/support and review the B1 gallery; manually compare B0/B1/B2 20-epoch arms; consider guarded extension only after reviewing validation grade metrics and losses. Sampling, LR/scheduling and resolution changes require separate review. A favorable validation result among many arms is exploratory, not independent confirmation.

## Cache, throughput and storage audit

The existing uncached path decodes, verifies SHA-256, and invokes the deterministic official processor on train and validation images **every epoch**. Startup also hashes every train/validation source image. The new official-input cache avoids repeated decoding/processing for a frozen deterministic processor, but each warm read verifies both the `.npy` file hash and tensor hash, so it can still be I/O bound. Its current `float32 [3,224,224]` payload is 602,112 bytes/image: about 1.85 GiB APTOS and 6.32 GiB DDR **per model**, about 16.34 GiB for both models/datasets before sidecars and checkpoints. With **102 GB reported free on a 95%-utilized shared volume**, prepare only the selected model/dataset cache after measuring uncached and warm throughput; do not create speculative duplicates.

A lossless `uint8` *pre-normalization* tensor would be 150,528 bytes/image, about 4.09 GiB for all four model/dataset caches, but it is **not implemented**. Quantizing an already normalized `float32` tensor is lossy. A future uint8 transport must reproduce the exact official resize/crop/rescale/normalization order and pass byte-exact output equivalence on **every image** before replacing the float32 cache. Memory mapping can lower page-cache pressure, but `torch.stack` still forms a batch and per-file hashing still reads the bytes; test it before adopting it. Share a cache only when model processor config/effective settings, manifest, library versions and **all tensor hashes** match, and enforce immutable metadata with one writer and concurrent read-only consumers.

Random train augmentation **cannot be frozen into the current final-tensor cache** without removing between-epoch variation. B1 transforms raw RGB each epoch before the official processor; no stochastic output is cached. A matching official-input cache may serve validation only. Keep validation on the exact deterministic cached or uncached input. Do not reuse frozen backbone embeddings in full tuning.

The current generator loads and hashes synchronously with no worker prefetch, pinned-memory staging or asynchronous host-to-GPU copy. Before changing it, compare **end-to-end** train/validation seconds and images/s for one throwaway, uniquely named, one-epoch uncached run and one cached run on the same model, dataset, GPU and batch size. Record CPU time, shared GPU contention, cache cold/warm rates, peak VRAM, disk bytes and full library versions. A DataLoader/prefetch or hash-policy change requires ordering, tensor-equality and RNG/resume tests; fewer hashes trade speed for weaker in-run corruption detection. No throughput gain is claimed in advance.

## Readiness matrix

| Dataset / backbone | Evidence now | Technical readiness | Scientific next step |
| --- | --- | --- | --- |
| APTOS / ConvNeXt-Tiny | Completed vanilla 20-epoch operator report; best macro-F1 0.673686 at epoch 17; rising validation loss | Baseline preserved. New runner needs a fresh matching smoke because its source hash changed. | B1 after the required manual gallery and FOV review. Do not relabel old run or automatically extend it. |
| DDR / ConvNeXt-Tiny | Completed vanilla 20-epoch operator report; best macro-F1 0.619817 at epoch 8; grade 1 F1 0.121951 | Baseline preserved; strong overfitting warning. | Check grade support/confusion, then compare B1 and B2 separately. Vanilla 80-epoch extension has no current performance rationale. |
| APTOS / DINOv3 ViT-S/16 | No real KC training result provided | Gated revision/processor, BF16 gradient coverage and 12 GB fit remain to be smoke-verified. | Technical smoke and FOV audit first; new 20-epoch vanilla run only after manual review. |
| DDR / DINOv3 ViT-S/16 | No real KC training result provided | Same gate, with longer DDR runtime/storage exposure. | Technical smoke and warm-cache benchmark first; no claimed comparative performance. |

The [existing E4/E5 protocol](FOUNDATION_ADAPTATION_E4_E5.md) provides exact pinned-snapshot, smoke, cache, 80-epoch and guarded-continuation commands. An original 20-epoch run with a different runner hash, incomplete summary, mismatched smoke, missing checkpoint sidecars, optimizer or RNG is **incompatible with automatic continuation**. The two actual KC ConvNeXt run directories were not connected here, so their individual compatibility cannot yet be certified. The revised runner has a different implementation SHA-256 from the original 20-epoch source; its smoke reports and official-input-cache metadata are source-bound. Do not reuse an old smoke report or cache under the revised source, and do not delete an old cache while an existing run uses it.

## Manual KC verification (no automatic execution)

Use the variables and pinned local snapshot setup in the [E4/E5 protocol](FOUNDATION_ADAPTATION_E4_E5.md). These commands inspect train/validation artifacts or run engineering-only, uniquely named benchmarks; they never request final test. Do not use the revised runner to resume a still-running legacy 20-epoch process.

```bash
# Read-only source and completed-run metadata review; set these to actual KC paths.
APTOS_RUN="$ROOT/foundation_e4/APTOS2019/convnext_tiny/full_seed42_b32"
DDR_RUN="$ROOT/foundation_e4/DDR/convnext_tiny/full_seed42_b32"
"$PY" - "$APTOS_RUN" "$DDR_RUN" <<'PY'
import json, pathlib, sys
for name in sys.argv[1:]:
    root = pathlib.Path(name)
    run = json.loads((root / "run.json").read_text())
    history = json.loads((root / "history.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    print(root, "epochs=", len(history), "best=", summary["best"]["epoch"],
          "source_sha256=", run["implementation_sha256"],
          "last_validation=", history[-1]["validation"])
PY
# Inspect exact processor settings from the local pinned snapshot and smoke.
"$PY" - "$MODEL_DIR" "$BASE/full_smoke_b32/smoke.json" <<'PY'
import json, pathlib, sys
print(json.loads((pathlib.Path(sys.argv[1])/"preprocessor_config.json").read_text()))
print(json.loads(pathlib.Path(sys.argv[2]).read_text())["effective_processor"])
PY
```

After preparing and verifying a **single** selected cache using the existing protocol, compare one-epoch end-to-end measurements in new, clearly labeled engineering directories. These runs update model weights for timing only and are **not** scientific baselines; review available disk/GPU resources first. The revised source requires a new matching smoke report. Preserve `RUN_UNCACHED` and `RUN_CACHED` until the timing/metric comparison is reviewed.

```bash
RUN_UNCACHED="$BASE/engineering_uncached_1ep_b$MICRO"
RUN_CACHED="$BASE/engineering_cached_1ep_b$MICRO"
CUDA_VISIBLE_DEVICES=0 "$PY" -u -m novelty.foundation_adaptation train --dataset "$DATASET" --artifact-root "$ROOT" --model-kind "$MODEL_KIND" --model-dir "$MODEL_DIR" --revision "$REV" --weight-sha256 "$WEIGHT_SHA" $LOAD_FLAG --mode full --device cuda:0 --precision bf16 --seed 42 --smoke-report "$BASE/full_smoke_b${MICRO}_80/smoke.json" --output-dir "$RUN_UNCACHED" --epochs 1 --micro-batch "$MICRO" --accumulation "$ACCUM" --head-lr 0.001 --backbone-lr 0.00001
CUDA_VISIBLE_DEVICES=0 "$PY" -u -m novelty.foundation_adaptation train --dataset "$DATASET" --artifact-root "$ROOT" --model-kind "$MODEL_KIND" --model-dir "$MODEL_DIR" --revision "$REV" --weight-sha256 "$WEIGHT_SHA" $LOAD_FLAG --mode full --device cuda:0 --precision bf16 --seed 42 --smoke-report "$BASE/full_smoke_b${MICRO}_80/smoke.json" --input-cache-dir "$CACHE" --output-dir "$RUN_CACHED" --epochs 1 --micro-batch "$MICRO" --accumulation "$ACCUM" --head-lr 0.001 --backbone-lr 0.00001
"$PY" - "$RUN_UNCACHED" "$RUN_CACHED" <<'PY'
import json, pathlib, sys
for name in sys.argv[1:]:
    root = pathlib.Path(name)
    row = json.loads((root/"history.json").read_text())[0]
    print(root, "train_s=", row["resource"]["train_seconds"],
          "validation_s=", row["resource"]["validation_seconds"],
          "train_images_per_second=", row["resource"]["train_images_per_second"],
          "peak_reserved_bytes=", row["resource"]["gpu_peak_reserved_bytes"])
PY
```

For a new, approved training run, launch `python -u -m novelty.foundation_adaptation train ...` **inside tmux** with a unique output directory. Interactive terminals display throttled tqdm bars with the separately installed `tqdm==4.70.1` package ([PyPI release](https://pypi.org/project/tqdm/4.70.1/)); redirected sessions show sparse plain progress lines. The optional package is listed in the isolated KC setup command, but was not installed locally in this audit. `training.log`, `history.json`, `history.csv` and the final `learning_curves.svg` persist in the output. All timestamps in the log are UTC. Monitor live without another model pass:

```bash
tmux new -s foundation_adaptation
# In another terminal, after setting RUN to the active output directory:
tail -f "$RUN/training.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv -l 2
```

B1 augmentation and B2 weighted CE were approved and implemented after this initial audit. Sampling, LR/schedule changes, label smoothing and resolution changes remain unimplemented.

## Approved B0/B1/B2 protocol and KC commands

B0 keeps ordinary CE and deterministic official inputs. B1 applies training-only horizontal flip (p=.50), rotation (p=.50, uniform ±7 degrees), brightness (p=.30, factor .90–1.10), then contrast (p=.30, factor .90–1.10) before the official processor. The RNG is keyed by seed, epoch and image ID. Run and smoke metadata bind the augmentation module SHA-256, exact protocol and Pillow version. The gallery selects evenly spaced examples from distinct training groups per grade. Bicubic rotation with black fill tries up to eight angles; if a candidate cannot retain 99% of the original max-channel >8 visible-area **proxy**, rotation is skipped. This proxy is not lesion/FOV segmentation. Inspect each grade's gallery for cropping, lesion visibility, color and background artifacts. B2 uses deterministic B0 inputs and weighted CE from training counts only. Validation uses B0's exact deterministic processor, ordinary CE and unchanged metrics for every arm.

For B2, let n_c be grade-c train support, N the train count and a_c=n_c^-0.5. Then w_c=a_c/(sum_j n_j*a_j/N), so the mean training-sample weight is one. Each optimizer accumulation window of m actual images uses sum_i(w_yi*CE_i)/m, including the incomplete final window. This differs from PyTorch's weighted-CE mean denominator sum_i w_yi. Weighted training loss is not numerically comparable to B0/B1 ordinary training loss.

Use the pinned snapshot and isolated variables in the [E4/E5 setup](FOUNDATION_ADAPTATION_E4_E5.md). Repeat the block for APTOS2019 and DDR, and for convnext_tiny and dinov3_vits16; set MODEL_DIR, REV, WEIGHT_SHA and LOAD_FLAG to the actual verified local model snapshot each time. ConvNeXt uses LOAD_FLAG=--trust-legacy-bin only after pickle weight hash verification; DINOv3 uses LOAD_FLAG empty. Use a separate BASE per model/dataset. All output directories are outside Git; no test split is opened. The gallery contains real training images, and APTOS redistribution permission is unconfirmed: keep it internal and outside Git.

~~~bash
DATASET=APTOS2019
MODEL_KIND=convnext_tiny
MICRO=4
ACCUM=$((32 / MICRO))
BASE="$ROOT/foundation_e4/$DATASET/$MODEL_KIND"
CACHE="$ROOT/foundation_inputs/$DATASET/$MODEL_KIND/b$MICRO"
mkdir -p "$BASE"
df -h "$ROOT"
"$PY" -m novelty.foundation_adaptation preflight --dataset "$DATASET" --artifact-root "$ROOT" --model-kind "$MODEL_KIND" --model-dir "$MODEL_DIR" --revision "$REV" --weight-sha256 "$WEIGHT_SHA"
"$PY" -m novelty.foundation_adaptation prepare-input-cache --dataset "$DATASET" --artifact-root "$ROOT" --model-kind "$MODEL_KIND" --model-dir "$MODEL_DIR" --revision "$REV" --weight-sha256 "$WEIGHT_SHA" --batch-size "$MICRO" --input-cache-dir "$CACHE"
"$PY" -m novelty.foundation_adaptation prepare-input-cache --dataset "$DATASET" --artifact-root "$ROOT" --model-kind "$MODEL_KIND" --model-dir "$MODEL_DIR" --revision "$REV" --weight-sha256 "$WEIGHT_SHA" --batch-size "$MICRO" --input-cache-dir "$CACHE" --resume --verify-equivalence
"$PY" -m novelty.foundation_adaptation inspect-augmentation --dataset "$DATASET" --artifact-root "$ROOT" --model-kind "$MODEL_KIND" --model-dir "$MODEL_DIR" --revision "$REV" --weight-sha256 "$WEIGHT_SHA" --seed 42 --epoch 1 --samples-per-grade 3 --output-dir "$BASE/B1_gallery_seed42_e1"
for ARM in B0 B1 B2; do
  CUDA_VISIBLE_DEVICES=0 "$PY" -u -m novelty.foundation_adaptation smoke --dataset "$DATASET" --artifact-root "$ROOT" --model-kind "$MODEL_KIND" --model-dir "$MODEL_DIR" --revision "$REV" --weight-sha256 "$WEIGHT_SHA" $LOAD_FLAG --mode full --experiment "$ARM" --batch-size "$MICRO" --device cuda:0 --precision bf16 --seed 42 --output-dir "$BASE/"$ARM"_smoke_b"$MICRO
done
~~~

After visual review and a passing smoke, manually launch separate 20-epoch runs. B0/B2 may use the *same* deterministic source-bound cache read-only; B1 decodes and augments train each epoch and should omit the cache unless an already matching cache is useful for validation. No stochastic augmented tensor is persisted. Each run begins from official pretrained weights in a unique directory.

~~~bash
run_arm () {
  local ARM="$1"; shift
  CUDA_VISIBLE_DEVICES=0 "$PY" -u -m novelty.foundation_adaptation train --dataset "$DATASET" --artifact-root "$ROOT" --model-kind "$MODEL_KIND" --model-dir "$MODEL_DIR" --revision "$REV" --weight-sha256 "$WEIGHT_SHA" $LOAD_FLAG --mode full --experiment "$ARM" --device cuda:0 --precision bf16 --seed 42 --smoke-report "$BASE/"$ARM"_smoke_b"$MICRO"/smoke.json" "$@" --output-dir "$BASE/"$ARM"_seed42_b"$MICRO"_20" --epochs 20 --micro-batch "$MICRO" --accumulation "$ACCUM" --head-lr 0.001 --backbone-lr 0.00001 --weight-decay 0.01
}
# Archive the exact runner before any 20-epoch arm for a later guarded extension.
SOURCE_COPY="$ROOT/foundation_sources/foundation_adaptation_B0_B1_B2.py"
mkdir -p "$(dirname "$SOURCE_COPY")"
if [ ! -e "$SOURCE_COPY" ]; then cp novelty/foundation_adaptation.py "$SOURCE_COPY"; else cmp novelty/foundation_adaptation.py "$SOURCE_COPY"; fi
sha256sum "$SOURCE_COPY"
AUG_SOURCE_COPY="$ROOT/foundation_sources/foundation_augmentation_B1.py"
if [ ! -e "$AUG_SOURCE_COPY" ]; then cp novelty/foundation_augmentation.py "$AUG_SOURCE_COPY"; else cmp novelty/foundation_augmentation.py "$AUG_SOURCE_COPY"; fi
sha256sum "$AUG_SOURCE_COPY"
run_arm B0 --input-cache-dir "$CACHE"
run_arm B1
run_arm B2 --input-cache-dir "$CACHE"
~~~

For a **manual end-to-end benchmark**, use two distinct one-epoch B0 engineering output directories with the same smoke/seed/model/GPU/batch and comparable server load; add --input-cache-dir "$CACHE" only to the cached run. These do perform optimizer updates and are not scientific results. Read history.json resource fields: train/validation wall seconds and images/s, synchronous train/validation input seconds, process CPU seconds, process-lifetime peak RSS and peak VRAM. Read cache summary.json for cold/warm images/s and disk bytes. Input timing adds no per-batch CUDA synchronization. No cache speedup can be claimed until KC measurement.

For a guarded **20-to-80 extension**, archive the exact novelty/foundation_adaptation.py bytes outside Git when starting the 20-epoch run and keep the matching smoke/cache. Then use the same training arguments in a fresh output directory with --epochs 80 --extend-from-run OLD_RUN --source-file SOURCE_COPY --source-smoke-report OLD_SMOKE --trust-run-checkpoint. B1/B2 source hash, augmentation/weight formula, manifest, cache identity, optimizer and RNG must match; retain the B1 augmentation-module copy and use the exact matching checkout if either source module later changes; any scientific change is rejected. The original 20-epoch run stays byte-unchanged. The [legacy B0 continuation example](FOUNDATION_ADAPTATION_E4_E5.md) remains separate.

**Storage:** a full float32 official-input cache is about 1.85 GiB APTOS or 6.32 GiB DDR per model, before metadata. B0/B2 share a matching cache; B1 needs no new train cache. Best/last full-model checkpoints add substantial disk per arm. Check df -h and du -sh "$BASE" before running multiple arms on the 95%-utilized KC volume. Gallery PNG/JSON must remain in restricted external storage.

### Exact KC benchmark and extension commands

These are manual actions only. The benchmark runs one optimizer epoch twice in new engineering directories; do not launch while another run is using the GPU. Save the runner copy **before** starting any 20-epoch arm, so the later extension can verify source identity.

~~~bash
benchmark_arm () {
  local LABEL="$1"; shift
  CUDA_VISIBLE_DEVICES=0 "$PY" -u -m novelty.foundation_adaptation train --dataset "$DATASET" --artifact-root "$ROOT" --model-kind "$MODEL_KIND" --model-dir "$MODEL_DIR" --revision "$REV" --weight-sha256 "$WEIGHT_SHA" $LOAD_FLAG --mode full --experiment B0 --device cuda:0 --precision bf16 --seed 42 --smoke-report "$BASE/B0_smoke_b"$MICRO"/smoke.json" "$@" --output-dir "$BASE/engineering_B0_"$LABEL"_1ep_b"$MICRO --epochs 1 --micro-batch "$MICRO" --accumulation "$ACCUM" --head-lr 0.001 --backbone-lr 0.00001 --weight-decay 0.01
}
benchmark_arm uncached
benchmark_arm cached --input-cache-dir "$CACHE"
"$PY" - "$BASE/engineering_B0_uncached_1ep_b"$MICRO "$BASE/engineering_B0_cached_1ep_b"$MICRO <<'PY'
import json, pathlib, sys
for name in sys.argv[1:]:
    row=json.loads((pathlib.Path(name)/"history.json").read_text())[0]
    print(name, row["resource"])
PY
~~~

After a completed 20-epoch arm passes history/checkpoint review, set ARM to that *same* arm and run its isolated extension. Repeat an interrupted extension with the same arguments plus --resume. For B0/B2 use the cache argument; for B1 omit it as in the first 20 epochs.

~~~bash
ARM=B1
OLD_RUN="$BASE/"$ARM"_seed42_b"$MICRO"_20"
EXTENDED="$BASE/"$ARM"_seed42_b"$MICRO"_80_extension"
extend_arm () {
  CUDA_VISIBLE_DEVICES=0 "$PY" -u -m novelty.foundation_adaptation train --dataset "$DATASET" --artifact-root "$ROOT" --model-kind "$MODEL_KIND" --model-dir "$MODEL_DIR" --revision "$REV" --weight-sha256 "$WEIGHT_SHA" $LOAD_FLAG --mode full --experiment "$ARM" --device cuda:0 --precision bf16 --seed 42 --smoke-report "$BASE/"$ARM"_smoke_b"$MICRO"/smoke.json" "$@" --output-dir "$EXTENDED" --epochs 80 --micro-batch "$MICRO" --accumulation "$ACCUM" --head-lr 0.001 --backbone-lr 0.00001 --weight-decay 0.01 --extend-from-run "$OLD_RUN" --source-file "$SOURCE_COPY" --source-smoke-report "$BASE/"$ARM"_smoke_b"$MICRO"/smoke.json" --trust-run-checkpoint
}
extend_arm
# For B0/B2: extend_arm --input-cache-dir "$CACHE"
# On interruption: extend_arm --resume
# For B0/B2 interruption: extend_arm --input-cache-dir "$CACHE" --resume
~~~
