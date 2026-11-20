# E5: exploratory integrated ConvNeXt-Tiny pilot

> **20 Nov 2026 — Status terbaru E5 APTOS2019:** training 20 epoch selesai; validation macro-F1 0.705791, best epoch 16; one-time frozen final test 366 images selesai dengan macro-F1 0.640456, accuracy 0.825137, QWK 0.891194. Bukti: [E5 APTOS results](../../results/APTOS2019/e5_integrated/README.md), [notebook analisis](../../notebooks/03_ConvNeXt_E5_APTOS2019.ipynb). DDR E5 final belum terdokumentasi. Kalimat lama di bawah yang menyatakan 'belum training' adalah snapshot perencanaan sebelum eksperimen, bukan status terkini.


**Status:** implementation and synthetic CPU checks only. No APTOS2019 or DDR
E5 training, validation inference, or final-test access has been run here.
`integrated` is the first planned pilot; `control`, `ordinal`, and `deferred`
are the three controlled comparison arms. These are proposed adaptations, not
a reproduction of Stelter et al.'s distribution-based loss or a novelty claim.

## Scientific protocol

All arms start in a fresh process from the same pinned official
`facebook/convnext-tiny-224` pretrained snapshot, load its official processor
through the existing `novelty.foundation_adaptation` functions, and fully tune
the backbone. A new five-class linear head supplies every reported prediction.
The model revision, weight SHA-256, model-config SHA-256, and processor-config
SHA-256 are locked to the matching APTOS/DDR B1 evidence in `config.py`; a
different otherwise valid ConvNeXt snapshot is rejected.
APTOS2019 and DDR use separate frozen `seed42` train/validation manifests.
The existing manifest loader checks the CSV hash, sidecar, group separation,
train/validation image SHA-256, and expected validation count. Test rows are
skipped before their image paths or labels are used. No E5 command accepts a
test split. Run directories, logs, and checkpoints must be outside Git.
The earlier manifest audit reported that 392 DDR rows assigned to the frozen
**validation** split originated from DDR's official directory named `test`.
This runner follows the frozen manifest `split`, matching B1; it never opens
rows assigned to the frozen `split=test`. The original DDR naming/protocol
distinction must remain explicit when interpreting results.
If the unchanged sidecar still records a Comm1 absolute dataset root, E5
requires `--manifest-source-root` equal to that exact recorded value. It maps
relative CSV image paths under the new `--artifact-root/datasets/<dataset>`
and checks every selected image hash there. The original CSV and JSON bytes
are never rewritten; their hashes and both roots are recorded in `run.json`.

Every arm uses B1's *training-only* raw-RGB augmentation in its exact order:
horizontal flip (p=0.50), bicubic rotation (p=0.50, within ±7 degrees and the
existing visible-area proxy), brightness (p=0.30, factor 0.90–1.10), contrast
(p=0.30, factor 0.90–1.10). The official processor follows augmentation.
Validation uses the same official processor without augmentation. Neither
stochastic augmentation nor validation images are resampled or relabeled.
The default implementation still uses uncached input for both splits. The
optional E5 transport keeps B1's exact transformation order: frozen source
SHA-256 check, EXIF transpose and RGB conversion, **fresh keyed B1 augmentation
each epoch**, then the pinned official processor. A bounded per-worker RAM LRU
can retain decoded RGB pixels; it never retains augmented training tensors.
Cache hits check source size and modification time, while every miss recomputes
the frozen source hash. The cache is lost on process exit, so every new run
verifies the sources again. An in-place alteration that restores both stat
fields is outside this non-adversarial cache check; the full manifest SHA audit
at run start remains mandatory. A separate, val-only disk cache uses the
existing `foundation_input_cache.InputCache` format: exact float32 3×224×224
tensors, source/processor/library/code metadata, per-entry tensor and file
SHA-256, and an ordered entry digest. Creation compares batch, singleton,
and round-trip tensors exactly. Existing incomplete or changed caches fail
closed; use a new external directory after a source or code change. No B0–B4
cache or dataset directory is written.

The optional DataLoader keeps the original `default_rng(seed+epoch)` batch
order and B1's image-keyed augmentation draws across worker settings. Worker
count, prefetch, pinning, RAM limit, and cache identity are recorded in
`run.json`, `smoke.json`, and checkpoint provenance; changing them prevents
guarded resume. `--decoded-cache-mib` limits **each train worker** (or the
main process with zero workers); total image-cache RAM is at most that limit
times `max(1,num_workers)`, plus processor/worker overhead and prefetched
batches. Start small on shared HPC. The disk cache estimate is
`validation_rows × (3×224×224×4 + 4096)` bytes, approximately 212 MiB for
APTOS2019 (366 rows) and 724 MiB for DDR (1,252 rows). The builder checks
the filesystem containing the requested cache root, its explicit size cap,
and a configurable free-space reserve before writing. It never accesses
manifest `split=test` rows.
The loader records per-epoch train RGB hit/miss counts and validation cache
reads/misses; a cached validation read still verifies its entry SHA-256.

For pooled feature `h ∈ R^768`, categorical logits are `z = W_c h + b_c`.
The categorical head has `768×5+5 = 3,845` parameters; the optional
ordinal head adds `768+1+1+3 = 773`. The complete backbone and trainable
counts are recorded from the actually loaded official model in smoke and
run summaries, rather than inferred from a different ConvNeXt package.
The optional auxiliary head uses severity `s = w_o^T h + b_o` and four
thresholds `τ_0 = a`, `τ_k = τ_(k−1) + softplus(d_k) + 1e−4` for `k=1,2,3`.
Its threshold logits are `s−τ_k`, with targets `1[y>k]`. Thus thresholds stay
strictly ordered and cumulative probabilities cannot reverse order. Thresholds
are initialized from **training-label priors** only. This auxiliary target is
derived from the same grade label; it supplies no new clinical annotation.

For train class counts `n_c`, `w_c = n_c^(−1/2) /
(sum_j n_j*n_j^(−1/2)/N_train)`. Only the `deferred` and `integrated` arms use
these weights, beginning **epoch 11**. All arms use ordinary CE in epochs
1–10. The per-image-mean training objective is

`L = mean_i[a_epoch(y_i) * CE(z_i,y_i)] + 0.25 * mean_(i,k) BCEWithLogits(s_i−τ_k,1[y_i>k])`

with the second term absent in `control` and `deferred`. A gradient
accumulation window divides each micro-batch sum by the *actual window image
count*, including the final short window. Validation loss and checkpoint
tie-breaking use **ordinary categorical CE** in every arm. Ordinal validation
loss is diagnostic. AdamW uses backbone LR `1e-5`, head LR `1e-3`, weight decay
`0.01`, gradient clipping norm `1`, effective batch `32`, seed `42`, BF16 on
supported CUDA devices, and a locked 20-epoch pilot. No scheduler, sampling,
new processor, resolution change, or adapter is introduced. Best checkpoint:
maximum validation macro-F1, then lower validation categorical CE, then
earliest epoch. The categorical head alone supplies predictions and metrics.

| Arm | Ordinal auxiliary | Deferred class weights |
| --- | --- | --- |
| `control` | No | No |
| `ordinal` | Yes | No |
| `deferred` | No | Epochs 11–20 |
| `integrated` | Yes | Epochs 11–20 |

The integrated pilot tests whether ordinal structure and late minority
weighting improve the existing five-grade classifier. It does not establish
better grade-1/3 recall, clinical interpretability, or novelty. B2's
always-on weighting had mixed validation outcomes, and small minority-grade
support limits precision. Run the other arms under the same E5 source for
controlled ablation; archived B1 remains historical context. Reusing the
already-tuned validation splits yields exploratory evidence, not independent
confirmation. A maximum 50-epoch study would be a separately prespecified
protocol, not a silent continuation of this 20-epoch pilot.

Prior art: [ordinal DR grading and distribution-based losses](https://proceedings.mlr.press/v307/stelter26a.html),
[rank-consistent ordinal heads](https://arxiv.org/abs/1901.07884),
[deferred reweighting for imbalanced recognition](https://proceedings.neurips.cc/paper/2019/hash/621461af90cadfdaf0e8d4cc25129f91-Abstract.html),
and [ConvNeXt for DR classification](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1812599/full).
The specific E5 switch epoch, auxiliary coefficient, and combination remain
project hypotheses. E5 implements neither the Stelter loss nor the exact CORAL
training objective; those works establish relevant ordinal prior art.

## Run provenance and outputs

`run.json` records arm, manifest/image-order metadata and hashes, official
snapshot revision, model and processor hashes, effective processor, source
hashes, seed, library versions, optimizer settings, and final-test status.
`smoke.json` must match this exact protocol before `train` starts. Every
completed epoch writes `last.pt`, validation-selected `best.pt`, SHA-256
sidecars, `history.json`, `history.csv`, and `training.log`. History includes
categorical and ordinal losses, train and validation aggregate/per-grade
metrics, QWK, MAE, confusion matrices, and resource measurements. A completed
20-epoch run writes `summary.json`. `--resume --trust-run-checkpoint` restores
model, optimizer, RNG, history, and selection after verifying hashes and exact
run metadata. Resume is at an epoch boundary; an interrupted multi-file epoch
write fails closed for manual audit.

## Engineering readiness and scientific interpretation

| Setting | Implemented value | Status for first pilot |
| --- | --- | --- |
| Pretrained model and processor | Pinned official ConvNeXt-Tiny snapshot, 224×224 | B1 inheritance; keep fixed |
| Train augmentation | B1 horizontal flip, rotation, brightness, contrast | B1 inheritance; keep fixed |
| Full tuning and categorical head | All backbone parameters plus five-class head | B1 inheritance; keep fixed |
| Optimizer | AdamW; backbone `1e-5`, head `1e-3`, decay `0.01`, clip `1` | B1 inheritance; keep fixed |
| Batch and duration | Effective 32; micro-batch 32/16/8/4/1 with matching accumulation; 20 epochs | B1 budget; choose one micro-batch for all arms after profiling |
| Mixed precision and scheduling | BF16 autocast on supported CUDA; FP32 CE/BCE; no scheduler | Keep fixed; verify actual A100 environment |
| Ordinal head | Four ordered trainable thresholds, coefficient `0.25` | E5 hypothesis, not proven optimal |
| Deferred weights | Train-only inverse-square-root class weights from epoch 11 | E5 hypothesis, not proven optimal |
| Selection | Max validation macro-F1; lower ordinary validation CE; earliest epoch | B1 inheritance; keep fixed |
| Test split | Never read in E5 development | Locked protocol |

The five-class categorical logits alone determine grades. E5's ordinal
targets reuse the same labels, so ordinal and categorical supervision overlap.
The `control`/`ordinal` and `deferred`/`integrated` comparisons isolate the
auxiliary term; `control`/`deferred` and `ordinal`/`integrated` isolate late
weighting. Historical B1 values are contextual, since their validation data
have already informed this design. The reported FP32 reference macro-F1
0.710288 (APTOS2019) and 0.653212 (DDR), and QWK 0.899672/0.846788,
remain operator-reported until their exact HPC artifacts and protocol are
cross-checked. Evaluate grade-1/3 recall with numerators and supports, not
macro-F1 alone.

The current ConvNeXt implementation pools a 768×7×7 final map to 768 features
before both heads. Whether 224×224 input or global pooling loses useful
microaneurysm or minority-grade evidence is **unmeasured here**. CRA-Net
adds category and lesion-relation attention and reports DDR/APTOS accuracy and
QWK ablations, not a directly comparable E5 macro-F1. Triple-DRNet uses
hierarchical DR/PDR/NPDR stages and attention and reports APTOS accuracy/QWK;
its training/evaluation protocol differs. Masud et al. study a custom
weighted loss, but their reported F1 is weighted average and their objective
is not E5's delayed inverse-square-root CE. SSiT uses saliency-guided retinal
self-supervised ViT pretraining and a saliency prediction task, unlike E5's
ImageNet-pretrained ConvNeXt. These are prior-art directions, not evidence
that their modules improve this frozen ConvNeXt split. See
[CRA-Net](https://doi.org/10.1016/j.compbiomed.2024.107993),
[Triple-DRNet](https://doi.org/10.1016/j.compbiomed.2023.106631),
[weighted-loss CNN](https://doi.org/10.1145/3470976), and
[SSiT](https://arxiv.org/abs/2210.10969). The supplied Alavee et al. XAI
paper concerns transfer-model comparisons and Grad-CAM, not SSiT or a direct
E5 architectural comparison.

Future separate, pre-registered comparisons, ranked by minimal intervention:

1. **Controlled higher-resolution pilot:** same backbone/objective at a new
   processor resolution, with matched training budget and measured memory and
   grade-level effects. Computational cost rises; no benefit is assumed.
2. **Spatial pooling or lightweight local feature head:** retain information
   from the final 7×7 map before global pooling. Requires isolated ablations
   against parameter-matched alternatives; may overfit minority grades.
3. **Retinal-domain representation learning:** SSiT-like domain pretraining is
   supported as prior art but needs a separately governed image corpus,
   compute budget, and leakage audit. It is not an E5 pilot extension.

Before any architecture change, audit exact prior art and define independent
confirmation. Increased GPU throughput is an engineering outcome, not a
classification improvement.

## Manual A100 HPC commands (do not run until reviewed/deployed)

These commands assume the reviewed E5 files are present in the HPC novelty
checkout. This Windows implementation has **not** been deployed to HPC. Check
the actual repository and official snapshot paths; the commands fail if they
are absent. No package installation, model download, or Git publication is
implied.

```bash
ROOT=/home/oracletiop/CAD-ViT-Research/Kel_13
REPO="$ROOT/Diabetic-Rethinography-Novelty"
REV=6166b7613034066690a621d8bf25ffdf181a34f0
WEIGHT_SHA=1fe15b2eba9e355de14c4e244e6365bf61ffef84da7d3442c0afbb2d8564f477
MODEL_DIR="$ROOT/hf_foundation/models--facebook--convnext-tiny-224/snapshots/$REV"
OUT="$ROOT/foundation_e5"
test -f "$REPO/novelty_integrated/e5/runner.py" || { echo 'E5 source missing'; exit 1; }
test -f "$MODEL_DIR/pytorch_model.bin" || { echo 'Pinned official snapshot missing; set MODEL_DIR to its verified path'; exit 1; }
test "$(sha256sum "$MODEL_DIR/pytorch_model.bin" | cut -d' ' -f1)" = "$WEIGHT_SHA" || { echo 'Official weight hash mismatch'; exit 1; }
test -f "$ROOT/cad_vit_runs/manifests/APTOS2019_seed42.csv" || exit 1
test -f "$ROOT/cad_vit_runs/manifests/DDR_seed42.csv" || exit 1
SOURCE_APTOS=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["root"])' "$ROOT/cad_vit_runs/manifests/APTOS2019_seed42.json")
SOURCE_DDR=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["root"])' "$ROOT/cad_vit_runs/manifests/DDR_seed42.json")
printf 'Frozen source roots: APTOS=%s DDR=%s\n' "$SOURCE_APTOS" "$SOURCE_DDR"
# Review the printed roots against the transferred original sidecars.
conda activate convnext-research
python -c 'import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0)); assert transformers.__version__ == "4.57.3"; assert torch.cuda.is_bf16_supported()'
nvidia-smi
df -h "$ROOT"
cd "$REPO"
```

Choose an authorized GPU with sufficient free memory and use a new output
directory. For the **integrated APTOS2019 pilot**:

```bash
CUDA_VISIBLE_DEVICES=0 python -m novelty_integrated.e5.runner smoke \
  --dataset APTOS2019 --arm integrated --artifact-root "$ROOT" \
  --manifest-source-root "$SOURCE_APTOS" \
  --model-dir "$MODEL_DIR" --revision "$REV" --weight-sha256 "$WEIGHT_SHA" \
  --trust-legacy-bin --device cuda:0 --precision bf16 --seed 42 \
  --micro-batch 32 --accumulation 1 \
  --output-dir "$OUT/APTOS2019/integrated_seed42_b32_20/smoke"
CUDA_VISIBLE_DEVICES=0 python -u -m novelty_integrated.e5.runner train \
  --dataset APTOS2019 --arm integrated --artifact-root "$ROOT" \
  --manifest-source-root "$SOURCE_APTOS" \
  --model-dir "$MODEL_DIR" --revision "$REV" --weight-sha256 "$WEIGHT_SHA" \
  --trust-legacy-bin --device cuda:0 --precision bf16 --seed 42 \
  --micro-batch 32 --accumulation 1 \
  --smoke-report "$OUT/APTOS2019/integrated_seed42_b32_20/smoke/smoke.json" \
  --output-dir "$OUT/APTOS2019/integrated_seed42_b32_20/run"
```

For the **independent integrated DDR pilot**:

```bash
CUDA_VISIBLE_DEVICES=0 python -m novelty_integrated.e5.runner smoke \
  --dataset DDR --arm integrated --artifact-root "$ROOT" \
  --manifest-source-root "$SOURCE_DDR" \
  --model-dir "$MODEL_DIR" --revision "$REV" --weight-sha256 "$WEIGHT_SHA" \
  --trust-legacy-bin --device cuda:0 --precision bf16 --seed 42 \
  --micro-batch 32 --accumulation 1 \
  --output-dir "$OUT/DDR/integrated_seed42_b32_20/smoke"
CUDA_VISIBLE_DEVICES=0 python -u -m novelty_integrated.e5.runner train \
  --dataset DDR --arm integrated --artifact-root "$ROOT" \
  --manifest-source-root "$SOURCE_DDR" \
  --model-dir "$MODEL_DIR" --revision "$REV" --weight-sha256 "$WEIGHT_SHA" \
  --trust-legacy-bin --device cuda:0 --precision bf16 --seed 42 \
  --micro-batch 32 --accumulation 1 \
  --smoke-report "$OUT/DDR/integrated_seed42_b32_20/smoke/smoke.json" \
  --output-dir "$OUT/DDR/integrated_seed42_b32_20/run"
```

To resume a previously started run, repeat its *identical* `train` command
with `--resume --trust-run-checkpoint`; never point it at a B0–B4 directory.
Subsequent ablations replace `integrated` in both `--arm` and the isolated
output path with `control`, `ordinal`, or `deferred`, and require a fresh
matching smoke. If micro-batch 32 fails the A100 smoke, use a separately
named run with `--micro-batch 16 --accumulation 2` (or another supported pair)
for **all four arms being compared**, and record that protocol choice before
training. Do not mix micro-batch protocols within a controlled comparison.

### Optional guarded cache, worker, and A100 profile path

Use this path only after the Windows E5 source has been reviewed and deployed
to the HPC checkout. It is an **alternative** to the uncached commands above.
Run from `REPO` with the `ROOT`, `REV`, `MODEL_DIR`, `WEIGHT_SHA`, `OUT`,
`SOURCE_APTOS`, and `SOURCE_DDR` variables set by the preflight block. Confirm
worker/RAM quotas and free disk capacity first. `profile` performs one
disposable train-batch optimizer update plus one validation forward; it saves
only a profile JSON, never a model. It is not a scientific pilot result.
GPU times use explicit synchronization at stage boundaries; normal training
does not synchronize for this profile. Run `nvidia-smi dmon -s ucm -d 1` in a
separate approved terminal for utilization context.

```bash
set -euo pipefail
cd "$REPO"
for DATASET in APTOS2019 DDR; do
  if [ "$DATASET" = APTOS2019 ]; then SOURCE_ROOT="$SOURCE_APTOS"; else SOURCE_ROOT="$SOURCE_DDR"; fi
  CACHE="$OUT/cache/$DATASET/official_v1_mb32"
  if [ ! -e "$CACHE" ]; then
    CUDA_VISIBLE_DEVICES='' python -m novelty_integrated.e5.runner prepare-cache \
      --dataset "$DATASET" --arm integrated --artifact-root "$ROOT" \
      --manifest-source-root "$SOURCE_ROOT" --model-dir "$MODEL_DIR" \
      --revision "$REV" --weight-sha256 "$WEIGHT_SHA" \
      --device cpu --precision fp32 --micro-batch 32 --accumulation 1 \
      --validation-cache-root "$CACHE" --cache-max-gib 2 --cache-reserve-gib 5
  else
    printf 'Existing cache requires verify-inputs: %s\n' "$CACHE"
  fi
done
```

The cache builder runs only CPU preprocessing on the frozen train/validation
manifest and writes only validation tensors. It validates every train/val
image SHA in the loader but never opens a frozen test row. The source and
processor code hashes are part of the cache identity. If the directory
already exists, the builder stops; verify it through a matching smoke or
use a fresh external cache root after auditing the failed build.

Run the following **separately** for each dataset. This loop is an exact
template for actual-image input equivalence, integrated profiling, and smoke; it does not
launch automatically when the notebook or module is imported. Check the
first profile's peak VRAM and input-wait times before training. The two-worker,
128 MiB/worker cache is a conservative starting configuration, **not** a
measured throughput optimum. Use identical flags in smoke and train because
the smoke report binds the transport configuration.

```bash
set -euo pipefail
cd "$REPO"
for DATASET in APTOS2019 DDR; do
  if [ "$DATASET" = APTOS2019 ]; then SOURCE_ROOT="$SOURCE_APTOS"; else SOURCE_ROOT="$SOURCE_DDR"; fi
  BASE="$OUT/$DATASET/integrated_seed42_b32_20_cached"
  CACHE="$OUT/cache/$DATASET/official_v1_mb32"
  COMMON=(--dataset "$DATASET" --arm integrated --artifact-root "$ROOT"
    --manifest-source-root "$SOURCE_ROOT" --model-dir "$MODEL_DIR"
    --revision "$REV" --weight-sha256 "$WEIGHT_SHA" --trust-legacy-bin
    --device cuda:0 --precision bf16 --seed 42 --micro-batch 32 --accumulation 1
    --validation-cache-root "$CACHE" --num-workers 2 --persistent-workers
    --prefetch-factor 2 --pin-memory --decoded-cache-mib 128)
  CUDA_VISIBLE_DEVICES='' python -m novelty_integrated.e5.runner verify-inputs \
    "${COMMON[@]}" --device cpu --precision fp32 \
    --output-dir "$BASE/input_preflight" --sample-per-grade 2
  test "$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$BASE/input_preflight/input_equivalence.json")" = ACTUAL_INPUT_EQUIVALENCE_PASS || exit 1
  COMMON+=(--input-preflight-report "$BASE/input_preflight/input_equivalence.json")
  CUDA_VISIBLE_DEVICES=0 python -m novelty_integrated.e5.runner profile \
    "${COMMON[@]}" --output-dir "$BASE/profile"
  CUDA_VISIBLE_DEVICES=0 python -m novelty_integrated.e5.runner smoke \
    "${COMMON[@]}" --output-dir "$BASE/smoke"
done
```

After separate training approval, run this block once with
`DATASET=APTOS2019` and separately with `DATASET=DDR`. These are independent
runs; inspect the first run before starting the second.

```bash
DATASET=APTOS2019  # change to DDR for the separate DDR run
if [ "$DATASET" = APTOS2019 ]; then SOURCE_ROOT="$SOURCE_APTOS"; else SOURCE_ROOT="$SOURCE_DDR"; fi
BASE="$OUT/$DATASET/integrated_seed42_b32_20_cached"
CACHE="$OUT/cache/$DATASET/official_v1_mb32"
COMMON=(--dataset "$DATASET" --arm integrated --artifact-root "$ROOT"
  --manifest-source-root "$SOURCE_ROOT" --model-dir "$MODEL_DIR"
  --revision "$REV" --weight-sha256 "$WEIGHT_SHA" --trust-legacy-bin
  --device cuda:0 --precision bf16 --seed 42 --micro-batch 32 --accumulation 1
  --validation-cache-root "$CACHE" --num-workers 2 --persistent-workers
  --prefetch-factor 2 --pin-memory --decoded-cache-mib 128
  --input-preflight-report "$BASE/input_preflight/input_equivalence.json")
CUDA_VISIBLE_DEVICES=0 python -u -m novelty_integrated.e5.runner train \
  "${COMMON[@]}" --smoke-report "$BASE/smoke/smoke.json" \
  --output-dir "$BASE/run" 2>&1 | tee "$BASE/training-console.log"
```

To resume, reuse
the *identical* `COMMON` arguments, output path, and smoke report and add
`--resume --trust-run-checkpoint`. `run.json` rejects changed worker/cache
settings. For controlled arm comparisons, use separate paths and matching
smoke reports for `control`, `ordinal`, and `deferred`; never reuse an E5
checkpoint as another arm's initialization. Any change to E5 source hashes
requires new cache and smoke provenance before a fresh run.

No A100 memory, throughput, cache-hit rate, or speedup has been measured by
the Windows-only CPU tests. The per-worker RAM bound excludes Python worker
overhead and prefetched tensors. `train_input_seconds` and
`validation_input_seconds` measure iterator waiting in the main process,
not total worker CPU time. The `profile` probe measures one batch only; use
full epoch logs for actual epoch throughput. Lower micro-batch with matched
accumulation preserves the nominal effective batch size but may change
BatchNorm statistics and numerical results. Choose a feasible micro-batch
once before comparing arms.
