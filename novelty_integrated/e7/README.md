# E7 global/local ConvNeXt pilot

**Status:** source implementation and synthetic CPU checks only. Neither cache has been built on HPC; no E7 GPU smoke, benchmark, training, validation result, or final-test evaluation has been performed here. The live E6 DDR status is unknown from this Windows session. Do not call the approach clinically lesion aware based on its learned gate or patch positions.

## Protocol and source audit

The frozen seed-42 APTOS2019 and DDR train/validation manifests are loaded through `novelty_integrated.e5.data.load_train_val`. That loader hashes the manifest and every selected train/validation image, enforces group and exact-content separation, and skips frozen `split=test` rows before reading their image paths or labels. The class order is `0,1,2,3,4`. Its original sidecar root must be passed as `--manifest-source-root` on a migrated host. E7 has no test command.

The official `facebook/convnext-tiny-224` snapshot is pinned to revision `6166b7613034066690a621d8bf25ffdf181a34f0`, weight SHA-256 `1fe15b2eba9e355de14c4e244e6365bf61ffef84da7d3442c0afbb2d8564f477`, processor config SHA-256 `fd575b890da5a949493e1d1e7a70bfcb9e4b99fe444004d2dbfa253add254741`, and model config SHA-256 `441f982e839a9ea96df7c1678422dc4e65cd5632731a6957f6aa03728195423d`. Source functions: `novelty.foundation_adaptation.audit_snapshot`, `load_local_processor`, `load_local_backbone`, `configure_trainable`, `_metrics`, `_best_is_new`, and atomic checkpoint helpers. E5 supplies its categorical/ordered-threshold heads, B1 train-only augmentation, ordinary categorical CE + 0.25 ordinal BCE, and train-only inverse-square-root weighting from epoch 11. E6's LDAM, adjacent-boundary objective, and oversampling are **not** used; E6's complete decoded training RGB cache may be read without writing to it.

E7 intentionally changes E5's 224-pixel view policy. It retains the full EXIF-corrected original RGB field using 384-pixel bicubic letterboxing, detects an approximate field-of-view bounding box from RGB intensity, extracts two overlapping left/right crops from the original resolution, letterboxes each to 224 pixels, then uses the *pinned slow official processor* for rescale and normalization with geometric resizing disabled. The official processor snapshot remains unchanged. These geometric choices are hypotheses, not a reproduction of the official 224-pixel center crop, and may still miss small lesions or alter apparent scale. Each crop spans 55% of FOV width and 55% of FOV height, centered vertically; the pair overlaps about 10% of FOV width. For an approximately square FOV, this gives roughly 6% more sampling density in the crop than the 384-pixel global view. Upper and lower peripheral regions are covered **only by the global view**, so the local branch cannot guarantee lesion coverage. Two 224-pixel patches cannot cover the entire 384-pixel global field at greater detail; this is a deliberate two-patch compute/coverage trade-off to audit on real images. The geometry threshold can mistake a bright border for retinal field; inspect real examples before training. Validation is deterministic. Training applies the existing conservative B1 flips/rotation/brightness/contrast before the three views are derived, with per-image, per-epoch keyed randomness.

One ConvNeXt backbone processes the global 384×384 image and the two 224×224 patches sequentially. It yields a 768-dimension pooled feature per view. A sigmoid gate over the concatenated global and mean local features mixes those two vectors before the E5 categorical and ordinal heads. The gate bias starts at -2 and its weights start at zero. This starts with roughly 12% local contribution, **not** baseline-output equivalence. No labels or validation outcomes influence patch selection. Global feature maps should be 12×12 and local maps 7×7; a real A100 smoke must verify these assumptions. The existing 224-only `foundation_adaptation.pooled` enforces 7×7 and cannot be called for the 384 view.

Preferred initialization is a *completed, checksum-verified, dataset-matched E5 integrated* `best.pt`, with strict backbone and head state transfer. The E5 source `summary.json`, training counts, frozen manifest, official model snapshot, selected epoch, and checkpoint sidecar must all match. No fallback to another checkpoint is permitted. `--init official` is an explicit separate protocol choice when no compatible completed E5 checkpoint exists; do not compare its outcomes as if it were warm-started. E6 initialization has **not** been validated or implemented. No E5/E6 checkpoint is bundled in Git.

Twenty epochs, seed 42, full fine-tuning, effective batch 32 via exact final-window gradient accumulation, AdamW backbone/head rates `1e-5`/`1e-3`, weight decay 0.01, no scheduler, BF16 on supported CUDA or FP32, clipping norm 1.0. Validation uses ordinary categorical CE and categorical-head predictions. Best checkpoint is maximum validation macro-F1; tie lower validation CE; remaining tie retains earlier epoch. Raw versus optional EMA selection is **prespecified**. Per-epoch history includes train/validation aggregate and grade-wise metrics, confusion matrices, ordinal training loss, times, RSS, peak VRAM, and cache provenance. Checkpoints include model, optimizer, EMA when enabled, RNG state, complete history, and record. Resume requires an exact record and verified best/last checkpoint sidecars; it never silently restarts.

`cache.py` can open a complete E6 train RGB cache **read-only** if its metadata and all entry hashes match; otherwise build an independent E7 train cache under `foundation_e7/cache`. Validation always uses a new isolated E7 RGB cache containing the frozen validation images only. It reuses E6's exact per-entry storage/checksum format (so entry sidecars carry the E6 entry schema) while the validation *root metadata* carries a separate E7 validation schema and split/order identity. Both cache builders use one writer lock, per-entry source and tensor digests, atomic same-filesystem writes, row/order metadata, and disk budget/reserve guards. No stochastic augmented image is cached. The cache stores decoded original RGB, not 384/224 tensors; geometric transforms and official normalization still run on each access. Cache preparation is CPU-only but may read/hash many images and consume substantial I/O. The small `verify-inputs` check compares direct and cached RGB plus tensors exactly for representative grade-stratified samples at training epochs 1/2 and deterministic validation. It does not prove whole-dataset input equivalence. Every selected train/validation source hash is rechecked before a run.

Optional `inference.py` implements aligned probability ensembles, predetermined horizontal/vertical whole-image TTA, and a tiny candidate-set temperature fit for a separate non-test calibration split. None is enabled in default training or checkpoint selection. TTA may change metrics/latency and requires its own validation protocol. The calibration helper rejects `split=val` and `split=test` and requires sample IDs disjoint from supplied forbidden validation/test IDs; no independent calibration subset is currently committed. A calibration manifest and group-disjointness audit are required before use. Ensemble inputs must match dataset, manifest, ordered IDs, class order, split and checkpoint identities. E6 DDR must be completed and separately verified before it can participate.

## HPC deployment and execution

### Deploy the reviewed salma commit

Run in the authorized JupyterLab terminal or on a machine with confirmed SSH access. Before deployment, inspect the active jobs and available resources. An allocated GPU must be available under lab policy. Free VRAM alone does not establish availability.

~~~bash
set -euo pipefail
ROOT="$HOME/CAD-ViT-Research/Kel_13"
BASE="$ROOT/Diabetic-Rethinography"
E7_REPO="$ROOT/Diabetic-Rethinography-E7"
E7_OUT="$ROOT/foundation_e7"
REV=6166b7613034066690a621d8bf25ffdf181a34f0
MODEL_DIR="$ROOT/hf_foundation/models--facebook--convnext-tiny-224/snapshots/$REV"
test "$(id -un)" = oracletiop
git -C "$BASE" remote get-url origin
git -C "$BASE" worktree list
ls -ld "$ROOT" "$BASE" "$E7_REPO" "$E7_OUT" 2>/dev/null || true
ps -eo user,pid,%cpu,%mem,etime,args | grep -E '[n]ovelty_integrated.e[567].runner' || true
tmux list-sessions 2>/dev/null || true
nvidia-smi --query-gpu=index,uuid,name,memory.free,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
free -h
df -h "$ROOT"
quota -s || true
cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/cpu.max 2>/dev/null || true

read -r -p 'Reviewed E7 commit SHA on salma: ' E7_SHA
[[ "$E7_SHA" =~ ^[0-9a-f]{40}$ ]]
test "$(git -C "$BASE" ls-remote --exit-code origin refs/heads/salma | cut -f1)" = "$E7_SHA"
git -C "$BASE" fetch --no-tags origin salma
test "$(git -C "$BASE" rev-parse FETCH_HEAD)" = "$E7_SHA"
if [ -e "$E7_REPO" ] || [ -L "$E7_REPO" ]; then
  test ! -L "$E7_REPO"
  test -O "$E7_REPO"
  test "$(git -C "$E7_REPO" rev-parse --show-toplevel)" = "$E7_REPO"
else
  git -C "$BASE" worktree add --detach "$E7_REPO" "$E7_SHA"
fi
test "$(git -C "$E7_REPO" rev-parse HEAD)" = "$E7_SHA"
test -z "$(git -C "$E7_REPO" status --porcelain)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate convnext-research
cd "$E7_REPO"
export PYTHONUNBUFFERED=1
python -c 'import torch,transformers,PIL,numpy; print(torch.__version__,transformers.__version__,PIL.__version__,numpy.__version__); assert transformers.__version__=="4.57.3"'
CUDA_VISIBLE_DEVICES='' python -m unittest tests.test_e7_integrated tests.test_e7_cache tests.test_e5_integrated tests.test_e5_input_pipeline tests.test_e6_integrated tests.test_e6_decoded_cache
python -m novelty_integrated.e7.runner --help
sha256sum "$MODEL_DIR/pytorch_model.bin" "$MODEL_DIR/preprocessor_config.json" "$MODEL_DIR/config.json"
~~~

### Prepare each dataset independently

In the same shell, set `DATASET=APTOS2019` or `DATASET=DDR` and run the following block once for that dataset. Listing completed-run candidates is read-only. Enter the matching completed E5 integrated run directory and its independently recorded checkpoint SHA. The source runner verifies the summary, sidecar, dataset, frozen split, official snapshot and strict state-dict compatibility. Choose the completed E6 full-resolution training cache for the same dataset.

~~~bash
read -r -p 'Dataset (APTOS2019 or DDR): ' DATASET
case "$DATASET" in
  APTOS2019) SESSION=e7-aptos ;;
  DDR) SESSION=e7-ddr ;;
  *) exit 1 ;;
esac
SOURCE_ROOT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["root"])' "$ROOT/cad_vit_runs/manifests/${DATASET}_seed42.json")
find "$ROOT/foundation_e5/$DATASET" -type f -name summary.json -print
read -r -p 'Completed E5 integrated run directory: ' E5_RUN
read -r -p 'Recorded best.pt SHA-256: ' E5_SHA
[[ "$E5_SHA" =~ ^[0-9a-f]{64}$ ]]
test "$(sha256sum "$E5_RUN/best.pt" | cut -d' ' -f1)" = "$E5_SHA"
python - "$E5_RUN" "$DATASET" "$E5_SHA" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
summary = json.loads((run / "summary.json").read_text())
sidecar = json.loads((run / "best.pt.sha256.json").read_text())
assert summary["status"] == "VALIDATION_COMPLETE" and summary["epochs_completed"] == 20
assert summary["record"]["dataset"] == sys.argv[2] and summary["record"]["arm"] == "integrated"
assert sidecar == {"sha256": sys.argv[3], "bytes": (run / "best.pt").stat().st_size}
print("Selected E5 epoch:", summary["best"]["epoch"])
PY
TRAIN_CACHE="$ROOT/foundation_e6/cache/$DATASET/decoded_rgb_train_v1"
VAL_CACHE="$E7_OUT/cache/$DATASET/decoded_rgb_val_v1"
RUN_BASE="$E7_OUT/$DATASET/run"
test -f "$TRAIN_CACHE/summary.json"
test ! -e "$TRAIN_CACHE/builder.lock"
python -m json.tool "$TRAIN_CACHE/summary.json"
COMMON=(--dataset "$DATASET" --artifact-root "$ROOT" --manifest-source-root "$SOURCE_ROOT"
  --model-dir "$MODEL_DIR" --train-cache-root "$TRAIN_CACHE" --val-cache-root "$VAL_CACHE"
  --reuse-e6-train --init e5 --e5-checkpoint "$E5_RUN/best.pt"
  --e5-checkpoint-sha256 "$E5_SHA" --trust-e5-checkpoint --trust-legacy-bin
  --micro-batch 4 --accumulation 8 --num-workers 2 --persistent-workers --prefetch-factor 2 --pin-memory)
E7=(python -u -m novelty_integrated.e7.runner)
CUDA_VISIBLE_DEVICES='' "${E7[@]}" estimate-cache "${COMMON[@]}" --device cpu --precision fp32
~~~

Review the validation size estimate, quota, RAM, disk free space and concurrent I/O before proceeding. With E6 cache reuse, the training estimate describes its decoded footprint and does not represent new training-cache growth. An absent or incompatible E6 cache blocks that dataset's reuse path. Choose a separate E7 train cache without the reuse flag only if the full extra footprint and shared I/O have been budgeted. Confirm that the chosen limits below fit the estimate plus the expected checkpoints and a shared-disk reserve.

~~~bash
read -r -p 'Approved validation cache cap in GiB: ' VAL_CAP
read -r -p 'Required shared disk reserve in GiB: ' RESERVE
CUDA_VISIBLE_DEVICES='' "${E7[@]}" prepare-cache "${COMMON[@]}"   --device cpu --precision fp32 --val-cache-max-gib "$VAL_CAP" --reserve-gib "$RESERVE"
CUDA_VISIBLE_DEVICES='' "${E7[@]}" verify-inputs "${COMMON[@]}"   --device cpu --precision fp32 --sample-per-grade 2 --output-dir "$RUN_BASE/input_preflight"
~~~

The default input check covers 10 images per split across five grades, with two epochs of augmentation on each training sample. The report includes the number that actually changed across epochs. Complete cache entry integrity and sample-based pixel equivalence are separate checks. The benchmark includes a bounded mmap copy measurement after cache verification. It cannot be interpreted as cold-disk throughput.

### Smoke, benchmark and launch the selected dataset

Recheck GPU ownership, utilization, and E6 progress before both the benchmark and training. Enter a lab-approved device UUID/index. CUDA maps the selected device to internal device 0. Use the same batch, accumulation and platform settings for smoke, benchmark and training. If memory is insufficient, choose a smaller micro-batch and corresponding accumulation to keep 32 effective examples, then use fresh reports before starting a run. A bounded optimizer benchmark uses disposable weights.

~~~bash
read -r -p 'Approved idle GPU index or UUID: ' GPU
GPU_ARGS=(--device cuda:0 --precision bf16 --acknowledge-shared-gpu
  --input-preflight-report "$RUN_BASE/input_preflight/input_equivalence.json")
SMOKE_ARGS=(--smoke-report "$RUN_BASE/smoke/smoke.json")
BENCH_ARGS=(--benchmark-report "$RUN_BASE/benchmark/benchmark.json")
CUDA_VISIBLE_DEVICES="$GPU" "${E7[@]}" smoke "${COMMON[@]}" "${GPU_ARGS[@]}"   --output-dir "$RUN_BASE/smoke"
CUDA_VISIBLE_DEVICES="$GPU" "${E7[@]}" benchmark "${COMMON[@]}" "${GPU_ARGS[@]}" "${SMOKE_ARGS[@]}"   --bench-batches 4 --output-dir "$RUN_BASE/benchmark"
python -m json.tool "$RUN_BASE/benchmark/benchmark.json"
~~~

Review the 20-epoch time estimate with headroom for checkpoint I/O and shared load. Before launching, recheck storage for two full AdamW checkpoints plus a temporary checkpoint write. Each epoch publishes JSON and CSV history. An active E7 run owns a training lock. A stale lock after a hard interruption needs manual review of the recorded PID and checkpoint sidecars before resuming.

~~~bash
TMUX_BIN=$(command -v tmux)
TRAIN=("${E7[@]}" train "${COMMON[@]}" "${GPU_ARGS[@]}" "${SMOKE_ARGS[@]}" "${BENCH_ARGS[@]}"
  --output-dir "$RUN_BASE/train")
test ! -e "$RUN_BASE/train"
if "$TMUX_BIN" has-session -t "=$SESSION" 2>/dev/null; then
  echo "E7 session already exists: $SESSION" >&2
  exit 1
fi
# Pass the resolved Conda interpreter directly into tmux.
TRAIN[0]=$(command -v python)
printf -v TRAIN_LINE '%q ' "${TRAIN[@]}"
"$TMUX_BIN" new-session -d -c "$E7_REPO" -s "$SESSION"   "exec env CUDA_VISIBLE_DEVICES=$(printf '%q' "$GPU") $TRAIN_LINE"
"$TMUX_BIN" list-panes -t "=$SESSION" -F '#{pane_pid} #{pane_current_command}'
"$TMUX_BIN" capture-pane -p -t "=$SESSION" -S -30
test ! -f "$RUN_BASE/train/training.log" || tail -n 30 "$RUN_BASE/train/training.log"
~~~

Check the E7 log again after integrity checks and initialization. Treat training as started only when E7 logs a valid training batch or epoch. For an interrupted run with valid checkpoint digests and inactive E7 process, rebuild the same variables and launch the same command with the explicit resume flags.

~~~bash
printf -v RESUME_LINE '%q ' "${TRAIN[@]}" --resume --trust-run-checkpoint
"$TMUX_BIN" new-session -d -c "$E7_REPO" -s "$SESSION"   "exec env CUDA_VISIBLE_DEVICES=$(printf '%q' "$GPU") $RESUME_LINE"
tail -n 40 "$RUN_BASE/train/training.log"
~~~

### Validation and optional TTA or ensemble

After a completed E7 run, inspect the selected epoch directly from its summary. The separate evaluation command exports ordered validation probabilities. These CPU examples use FP32. Reproducing the training-time BF16 validation path requires an approved GPU and matching precision flags. Prediction files belong in the external E7 artifact area.

~~~bash
python -c 'import json,sys; s=json.load(open(sys.argv[1])); assert s["status"]=="VALIDATION_COMPLETE"; print(json.dumps(s["best"], indent=2))' "$RUN_BASE/train/summary.json"
BEST_SHA=$(sha256sum "$RUN_BASE/train/best.pt" | cut -d' ' -f1)
EVAL=(--device cpu --precision fp32 --run-dir "$RUN_BASE/train"
  --checkpoint-sha256 "$BEST_SHA" --trust-run-checkpoint)
CUDA_VISIBLE_DEVICES='' "${E7[@]}" evaluate "${COMMON[@]}" "${EVAL[@]}"   --tta none --output-dir "$RUN_BASE/validation"
CUDA_VISIBLE_DEVICES='' "${E7[@]}" evaluate "${COMMON[@]}" "${EVAL[@]}"   --tta horizontal --output-dir "$RUN_BASE/validation_tta_h"
~~~

The example below averages two already verified file sources with fixed equal weights. Each input must have a verified producing run and checkpoint, processor/transform identity, and aligned validation labels. Historical aggregate summaries cannot be used as probability inputs. E6 predictions enter only after the E6 result and preprocessing provenance are independently verified. Any ensemble gain observed on this validation split is exploratory.

~~~bash
P1="$RUN_BASE/validation/validation_predictions.json"
P2="$RUN_BASE/validation_tta_h/validation_predictions.json"
sha256sum "$P1" "$P2"
python - "$P1" "$P2" "$RUN_BASE/ensemble_50_50.json" <<'PY'
import json, sys
from pathlib import Path
from novelty.foundation_adaptation import _classification
from novelty.preof_phase1 import sha256
from novelty_integrated.e7.inference import ensemble

paths = [Path(name) for name in sys.argv[1:3]]
items = [json.loads(path.read_text()) for path in paths]
assert all(item.get("processor_sha256") and item.get("geometry") for item in items)
assert items[0]["labels"] == items[1]["labels"]
weights = [0.5, 0.5]
values = ensemble(items, weights)
with open(sys.argv[3], "x") as stream:
    json.dump({"split": "val", "dataset": items[0]["dataset"], "weights": weights,
               "calibration": None, "input_sha256": [sha256(path) for path in paths],
               "metrics": _classification(items[0]["labels"], values)}, stream, indent=2)
PY
~~~

Report measured results with the source commit, checkpoint hash, manifest identity, GPU/cache status and validation scope. The committed E5 APTOS evidence records validation macro-F1 0.705791 at epoch 16. Compare E7 with any verified E5/E6 validation results under compatible splits and explain differences in initialization, geometry, compute and objectives. No E7 validation gain or clinical localization has been established.
