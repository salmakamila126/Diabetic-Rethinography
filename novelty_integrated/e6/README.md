# Study 03 follow-up: E6 boundary-aware ConvNeXt-Tiny

**Status 20 November 2026:** source and CPU synthetic checks only. Cache population, actual-image equivalence, A100 benchmark, E6 training, validation and final test **NOT RUN** in this Windows session. The operator's DDR E5 grade-1 counts for epochs 6/8/9/10 are historical validation observations awaiting local HPC artifact verification; they are not E6 results. E6 is an exploratory hypothesis, not a demonstrated novelty or clinical tool.

## Files and locked protocol

- `config.py`: official ConvNeXt-Tiny revision and hashes inherited unchanged from E5, 20 epochs, seed 42, effective batch 32, backbone LR 1e-5, head LR 1e-3, AdamW weight decay 0.01, max gradient norm 1, BF16 on A100.
- `objective.py`: FP32 LDAM + delayed effective-number weights + E5 ordinal BCE + adjacent pairwise term.
- `decoded_cache.py`: train-only EXIF-corrected uint8 RGB `.npy` entries with source/file hashes, metadata, ordered digest, same-filesystem atomic writes, exclusive builder lock and read-only mmap. Complete corrupt entries fail closed; a lone partial data or sidecar entry is rebuilt by the exclusive builder. No cached augmentation or validation/test image.

`open_complete` hashes every train cache entry before a run. Worker reads then check the data/sidecar size and modification-time fingerprint and reuse that verified digest while the fingerprint is stable; no full image bytes are copied into worker RAM. An adversary able to modify content while restoring both timestamps and sizes is outside this non-adversarial storage check. Source images are SHA-verified again when each run loads the frozen manifest.
- `input_pipeline.py`: E5 `Rows`/`EpochBatches` and validation collator reused; stochastic B1 augmentation is applied *after* fresh RGB loading on every train epoch, then the official processor. The 2-worker, persistent, prefetch-2, pinned-memory settings are conservative starting values, not measured optimum.
- `runner.py`: cache estimate/build, actual-image equivalence, disposable smoke/benchmark, and guarded 20-epoch training/resume. It reads E5 validation cache only if exact E5 source, manifest, processor, batch, and ordered-entry metadata pass `validation_cache.open_complete`; otherwise omit the validation-cache option and use deterministic uncached validation. Never rewrites E5 cache.

Both datasets use the unchanged frozen seed-42 train/validation manifest and its original JSON sidecar. `data.load_train_val` verifies CSV/sidecar and every selected train/validation source hash; it skips frozen `split=test` rows before using their paths or labels. DDR official directory names do not redefine frozen split membership. `manifest-source-root` must match the sidecar's recorded original dataset root when migrating hosts. No E6 command exposes final-test evaluation.

E6-A is the implemented **independent official-pretrained initialization**. The original E5 classifier and ordinal head implementation is imported unchanged, but E6 weights are newly initialized; no E5 checkpoint is loaded. E6-B warm-start is **not enabled**: it requires a completed immutable E5 checkpoint and a separately frozen shortened protocol. Do not infer equal-budget superiority from such a future pilot.

## Exact objective

For training counts `n_c`, the true-class LDAM margin is `m_c = 0.5 n_c^(-1/4) / max_j n_j^(-1/4)`. For target `y`, categorical loss is CE of `30(z - m_y one_hot(y))`. From epoch 11, each image's categorical loss is multiplied by `w_y`, where `E(n)=(1-0.9999^n)/(1-0.9999)`, `raw_w_c=1/E(n_c)`, and `w_c=raw_w_c / [sum_j n_j raw_w_j / sum_j n_j]`; this normalization makes the mean training-image weight one. Epochs 1â€“10 are unweighted. Counts come from training rows only. The beta value and normalization are prespecified implementation assumptions, not optimized results.

The E5 ordinal head predicts `q_k=h_severity-t_k`, with strictly ordered train-prior-initialized learnable `t_k`. BCE targets are `1[y>k]`, `k=0..3`. The categorical head alone produces reported grades. For each training image with grade 0, 1 or 2, define `N_0={1}`, `N_1={0,2}`, `N_2={1}` and `a_i=mean_{j in N_y} softplus(30(z_j-z_y))/30`; for grades 3/4 let `a_i=0`. The batch mean of `a_i` is the adjacent term. Total loss is `L_LDAM-DRW + 0.25 L_ordinal + 0.1 L_adjacent`. LDAM and the adjacent term both influence categorical boundaries; ordinal BCE supplies a related severity signal. Their overlap is a limitation, and a later ablation must establish whether the adjacent term adds value. The 0.1 coefficient is an unoptimized pilot assumption. Validation loss and checkpoint tie break use ordinary CE on **raw** categorical logits. Best selection is validation macro-F1, then lower validation CE, then earliest epoch, exactly as E5.

This differs from E5's ordinary CE with deferred inverse-square-root weights. E6's loss scales may differ, so loss values are not directly comparable. The objective is based on [Cao et al. LDAM-DRW](https://proceedings.nips.cc/paper_files/paper/2019/hash/621461af90cadfdaf0e8d4cc25129f91-Abstract.html), [Cui et al. effective-number weighting](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html), and the existing E5 ordinal design. [Stelter et al.](https://proceedings.mlr.press/v307/stelter26a.html) investigate a different distribution-based ordinal loss. [Wang et al. CAD-ViT](https://doi.org/10.1109/ACCESS.2025.3572455) motivates retinal feature/context analysis, **not** this E6 objective. Adjacent pairwise discrimination and its combination with LDAM require targeted prior-art and controlled ablation before any novelty claim.

## Manual HPC deployment and execution

Run in an authorized interactive HPC terminal. The commands below **do not run automatically** and do not change active E5 source or process. Stop after each gate if a hash, resource, or E5-isolation check fails. Use the exact published `salma` commit reported in the delivery message.

```bash
set -euo pipefail
ROOT=/home/oracletiop/CAD-ViT-Research/Kel_13
BASE="$ROOT/Diabetic-Rethinography-Notebook-Review"
E6_REPO="$ROOT/Diabetic-Rethinography-E6"
E6_OUT="$ROOT/foundation_e6"
REV=6166b7613034066690a621d8bf25ffdf181a34f0
WEIGHT_SHA=1fe15b2eba9e355de14c4e244e6365bf61ffef84da7d3442c0afbb2d8564f477
MODEL_DIR="$ROOT/hf_foundation/models--facebook--convnext-tiny-224/snapshots/$REV"
EXPECTED_COMMIT=REPLACE_WITH_REPORTED_COMMIT
# Fetch updates only Git objects in the review checkout; no checkout of its active branch.
git -C "$BASE" fetch origin salma
test "$(git -C "$BASE" rev-parse origin/salma)" = "$EXPECTED_COMMIT"
test ! -e "$E6_REPO"
git -C "$BASE" worktree add -b e6-boundary-pilot "$E6_REPO" origin/salma
test "$(git -C "$E6_REPO" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$E6_REPO" status --porcelain)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate convnext-research
python -c 'import torch,transformers,PIL,numpy; print(torch.__version__,transformers.__version__,PIL.__version__,numpy.__version__); assert transformers.__version__=="4.57.3"'
test "$(sha256sum "$MODEL_DIR/pytorch_model.bin" | cut -d' ' -f1)" = "$WEIGHT_SHA"
cd "$E6_REPO"
python -m unittest tests.test_e6_integrated tests.test_e6_decoded_cache tests.test_e5_integrated tests.test_e5_input_pipeline
```

Inspect live resources and E5 ownership before any cache build or GPU job. `nvidia-smi` free VRAM does **not** prove idle compute. Do not run E6 GPU jobs while E5 or another user's workload would be affected.

```bash
free -h; df -h "$ROOT"; df -h /dev/shm; nproc
cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/cpu.max 2>/dev/null || true
ps -eo user,pid,%cpu,%mem,etime,args | grep -E '[n]ovelty_integrated.e5.runner|[n]ovelty_integrated.e6.runner'
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv
```

For each dataset, inspect the original sidecar root and estimate **actual decoded dimensions** before setting a cache budget. The cache and outputs use the independent `foundation_e6` root, never E5. `prepare-cache` is CPU-only and hashes every training image before cache admission; it does not open validation images. `verify-inputs` subsequently checks all train/validation source hashes and compares selected actual tensors. Schedule both so the shared disk is not saturated. Validation cache is optional. If E5 metadata is incompatible, remove `--validation-cache-root` from **every** command for this run and use uncached validation.

```bash
DATASET=DDR  # run APTOS2019 separately; never combine outputs
SOURCE_ROOT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["root"])' "$ROOT/cad_vit_runs/manifests/${DATASET}_seed42.json")
TRAIN_CACHE="$E6_OUT/cache/$DATASET/decoded_rgb_train_v1"
VAL_CACHE="$ROOT/foundation_e5/cache/$DATASET/official_v1_mb32"
RUN_BASE="$E6_OUT/$DATASET/ldam_ordinal_adjacent_seed42_b32_20"
COMMON=(--dataset "$DATASET" --artifact-root "$ROOT" --manifest-source-root "$SOURCE_ROOT"
  --model-dir "$MODEL_DIR" --revision "$REV" --weight-sha256 "$WEIGHT_SHA"
  --trust-legacy-bin --train-cache-root "$TRAIN_CACHE" --micro-batch 32 --accumulation 1
  --seed 42 --num-workers 2 --persistent-workers --prefetch-factor 2 --pin-memory)
# Include this only if its metadata passes E6's read-only compatibility gate:
if [ -f "$VAL_CACHE/summary.json" ]; then COMMON+=(--validation-cache-root "$VAL_CACHE"); fi
CUDA_VISIBLE_DEVICES='' python -m novelty_integrated.e6.runner estimate-cache "${COMMON[@]}" --device cpu --precision fp32
# Review estimated bytes, cgroup limits, free disk and E5 progress; choose a max GiB above the estimate.
CUDA_VISIBLE_DEVICES='' python -m novelty_integrated.e6.runner prepare-cache "${COMMON[@]}" \
  --device cpu --precision fp32 --cache-max-gib 100 --cache-reserve-gib 100
CUDA_VISIBLE_DEVICES='' python -m novelty_integrated.e6.runner verify-inputs "${COMMON[@]}" \
  --device cpu --precision fp32 --sample-per-grade 2 --output-dir "$RUN_BASE/input_preflight"
```

The fixed `100 GiB` max and reserve are **guard settings**, not measured requirements. Adjust them only after reading `estimate-cache`; if the estimate exceeds max or reserve is unavailable, stop. A cache build may continue after an interruption using the same metadata; complete corrupt entries fail closed. Run `verify-inputs` before *any* GPU command. It compares cached and uncached actual train tensors exactly at epochs 1/2 and optional validation tensors against the direct official processor.

Only after E5 is no longer contending for the selected GPU or an authorized independent compute allocation is demonstrated:

```bash
PREFLIGHT="$RUN_BASE/input_preflight/input_equivalence.json"
GPU_COMMON=("${COMMON[@]}" --device cuda:0 --precision bf16 --input-preflight-report "$PREFLIGHT")
CUDA_VISIBLE_DEVICES=0 python -m novelty_integrated.e6.runner smoke "${GPU_COMMON[@]}" \
  --output-dir "$RUN_BASE/smoke"
CUDA_VISIBLE_DEVICES=0 python -m novelty_integrated.e6.runner benchmark "${GPU_COMMON[@]}" \
  --bench-batches 8 --output-dir "$RUN_BASE/benchmark"
python -c 'import json,sys; r=json.load(open(sys.argv[1])); print("20-epoch estimate (s):",r["estimated_20_epoch_seconds"],"GPU peak bytes:",r["gpu_peak_reserved_bytes"]); assert r["estimated_20_epoch_seconds"]*1.5 <= 21600' "$RUN_BASE/benchmark/benchmark.json"
# Check E5 progress, GPU utilization, cgroup, filesystem and remaining wall-clock budget again.
CUDA_VISIBLE_DEVICES=0 python -u -m novelty_integrated.e6.runner train "${GPU_COMMON[@]}" \
  --smoke-report "$RUN_BASE/smoke/smoke.json" \
  --benchmark-report "$RUN_BASE/benchmark/benchmark.json" \
  --max-pilot-seconds 21600 --disk-reserve-gib 100 \
  --output-dir "$RUN_BASE/run" 2>&1 | tee "$RUN_BASE/training-console.log"
```

`train` rejects a mismatched smoke/benchmark record, insufficient GPU free memory, the benchmark extrapolation plus 50% margin exceeding the stated budget, and inadequate output disk reserve. The short benchmark estimates but does not certify a 20-epoch deadline: checkpoint I/O, cache coldness and shared load remain unmeasured. If the estimate fails, **do not start** the nominal 20-epoch E6-A. Any shorter pilot needs a separate frozen protocol and label. For resume, keep all arguments and paths unchanged and add `--resume --trust-run-checkpoint`; `last.pt`/`best.pt` SHA sidecars and run/history lineage are checked by the existing E5 checkpoint utilities. Never resume an E5 checkpoint as E6-A.

## Limitations and next comparison

The sparse Grade 1 support and possible label ambiguity limit inference. An improved macro-F1 may coexist with lower Grade 1 recall; report both Grade 1 and Grade 3 precision/recall/F1, support, QWK, MAE and the full validation confusion matrix. E6-A versus E5 requires matched frozen split, processor, initialization, epoch budget and selection rule; E6 changes loss scale and full cache transport. The test split remains sealed. No image-level clinical cause, lesion preservation claim, or performance improvement follows from this code alone.
