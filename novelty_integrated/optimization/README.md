# ConvNeXt inference and structured pruning

This package measures a previously selected checkpoint and its derivatives on one frozen validation sequence. The original training objective, checkpoint and historical result stay identifiable. See the [notebook index](../../notebooks/README.md), [experiment inventory](../../docs/NOTEBOOK_EXPERIMENT_AUDIT.md), and [audit of historical optimization notebooks](../../docs/OPTIMIZATION_COMPATIBILITY.md) for scientific context. The archived notebooks on `novelty` use different splits or architectures and remain historical references.

## Checkpoint eligibility

`SUPPORTED` describes an implemented adapter and published receipt. The checkpoint binary remains external, so the registry separately reports `MISSING_CHECKPOINT` until a real file is supplied and verified for an execution. `MISSING_SOURCE` means exact external source copies are required at execution. They are read for hashing only. `PENDING_TRAINING` means there is no approved completed-run receipt in this checkout, with live training status unknown. The remaining registry states `NEEDS_ADAPTER` and `UNVERIFIED` are reserved for a known incompatible architecture or evidence lacking identity, respectively.

| Arm | APTOS2019 | DDR | Construction, archive format and recovery |
| --- | --- | --- | --- |
| B0 | SUPPORTED, best 17 | SUPPORTED, best 8 | HF V1 + linear 768→5. Separate `backbone` / `head` state dicts. Historical ordinary CE, no augmentation. |
| B1 | SUPPORTED, best 16 | SUPPORTED, best 11 | Same inference topology as B0. B1 augmentation is train-only. |
| B2 | SUPPORTED, best 7 | SUPPORTED, best 5 | Same topology. Train-only weighted CE has no inference correction. |
| B3 | MISSING_SOURCE, best 20 | MISSING_SOURCE, best 11 | The checked external parallel runner uses the same linear head and validation transform. Both runner and grade-3 sampler hashes must match. |
| B4-Q | MISSING_SOURCE, best 11 | NOT_APPLICABLE | B1 checkpoint shape and `record.b4` arm/selection/cache receipts. Original APTOS validation retained. Require hashes of the external runner and common module. |
| B4-R | MISSING_SOURCE, best 9 | NOT_APPLICABLE | Independent class-matched removal control checkpoint, with the same source requirements. |
| E5 Integrated | SUPPORTED, best 16 | SUPPORTED, best 18 | HF V1 + original categorical and ordinal heads. Separate `backbone` / `heads`. Categorical-only prediction. Optional recovery uses the E5 ordinal objective and deferred weight state. |
| E6 Boundary-Aware | SUPPORTED, best 15 | SUPPORTED, best 12 | Same inference architecture as E5. Validation uses raw categorical logits. Optional recovery uses E6 LDAM, ordinal and adjacent supervision with its deferred weight state. |
| E7 Global/Local | COMPLETED, best 19; MISSING_CHECKPOINT locally | PARTIAL / EARLY STOPPED AT EPOCH 10, best 2; optimization ineligible | Both use one global 384×384 and two local 224×224 views with a shared backbone and learned fusion gate. The [audited APTOS2019 export](../../results/APTOS2019/e7_global_local/validation_report.md) identifies raw selected weights. The [DDR aggregate export](../../results/DDR/e7_global_local/validation_report.md) supplies validation evidence and checkpoint digest receipts. |

B0–B4, E5 and E6 use the same official ConvNeXt-Tiny V1 pretrained snapshot identity and slow 224 px processor. The runtime reuses `foundation_adaptation.audit_snapshot`, `load_local_backbone`, `pooled` and the official deterministic validation batch path. It checks the source hashes from E5/E6 evidence, the historical foundation checkpoint metadata, exact published checkpoint SHA-256 and project sidecar, and loads trained weights with `strict=True`. B0's historical foundation hash matches Git commit `de4699e`. B1/B2 match `3a563c6`. The current inference component uses the reviewed B1 source. B3's two copied files and B4's two copied files matched the published hashes. B4's isolated historical validation cache used direct B1-equivalent preprocessing. This framework reads verified original images and does not require those caches or train selections for inference.

The frozen CSV and original sidecar are selected separately by dataset. The APTOS2019 SHA-256 is `73d1546e3c2fd2bf9e3d0fd10af914cb523d28e4bc8e1f25212af454d4be8854`. The DDR SHA-256 is `7bc3e3916e012bc46fcd9cd69aa469e0daf3229e1273f1d8ecb6054899bf6eb1`. Grade order is 0–4. The original best checkpoint policies maximize validation macro-F1, break ties by lower validation CE and then retain the earlier epoch. The original training source and aggregate evidence are documented and linked in the [experiment inventory](../../docs/NOTEBOOK_EXPERIMENT_AUDIT.md). `python -m novelty_integrated.optimization registry` prints dataset-specific checkpoint and manifest digests.

## Benchmark protocol

- Preserve the original checkpoint and verify its hash before and after the run. The project sidecar must match both SHA-256 and file length. Trust flags authorize loading project-owned pickle after verification and the pinned official legacy snapshot. The original manifest and its sidecar stay read-only. The explicit source-root mapping uses the exact original dataset root in the sidecar.
- Use original validation order or choose `--sample-count N --sample-seed S`. Subsets rank by SHA-256 of seed, frozen image ID and image digest, select the first N, then restore manifest order. The subset must contain all five grades and divide evenly into the selected batch size. Report zero-based frozen validation positions, grade support, the ordered tensor digest, and source/processor/checkpoint hashes. No sample IDs or image paths appear in the aggregate output.
- Use identical FP32 tensors, shape, batch size and host for all candidates. Before timing, compare every backend's logits on the selected sequence against its own eager reference. The default gate requires all elements to satisfy `abs(delta) <= 1e-4 + 1e-4 * abs(reference)` plus 100% argmax agreement. Report failures and quality, and omit failed-backend timing. Pruning is allowed to alter the parent logits and is judged separately by aligned quality deltas.
- Eager inference is CPU FP32. TorchScript is traced and frozen at the exact batch shape and reloaded. ORT uses ONNX opset 17, the CPU provider, sequential execution and explicit threads. OpenVINO compiles ONNX-derived FP32 IR with the CPU latency hint, one stream and an explicit thread limit. Each is a candidate until export and numerical validation succeed for the actual checkpoint. There is no fallback to another execution provider. TorchScript is optional and deprecated on newer runtimes. [ORT threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html) · [OpenVINO CPU device](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html)
- Warm up with 20 calls by default, then run three complete passes per variant/backend/scope. Counts, timings and model inputs are fixed across comparisons. Model-only excludes preparation and includes backend API conversion/output. End-to-end includes decode, EXIF/RGB, official transforms and synchronous inference, and excludes hash checks and metrics. File caches are uncontrolled and already warmed by verification. Use the stored per-pass records, repeat in different backend orders if making comparative performance claims, and pin cores or use an isolated allocation to minimize host-load effects. This implementation times CPU only.
- Report batch and per-image median, mean and P95, summed-call throughput, per-pass wall time and raw nanosecond records. Report artifact bytes and hashes independently of the original optimizer-containing archive. For eager, bytes refer to the complete inference state dict, including retained auxiliary heads. Scripted/exported graphs can omit unused auxiliary computations. RSS snapshots and available process-lifetime high-water mark include Python, data and model loading.
- Report new FP32 quality for the selected original and each candidate on the same inputs. Historical best-validation metrics retain a separate label. All quality results use validation, which also selected the parent. Conclusions about optimization settings are exploratory unless evaluated separately on an independent approved split.

## Pruning and recovery

Hugging Face V1 ranks rows of the first expansion Linear by L1 norm, removes `floor(ratio × expansion width)` channels per block using stable index tie-breaking, and keeps corresponding columns in the second Linear. The residual path, depthwise convolution, normalization, layer scale and head widths are unchanged. E7 prunes its one shared backbone once and preserves both input tensors, all three views and the gate. The separate optional timm V2 adapter only accepts compatible channels-last GRN MLP blocks. It slices GRN scale and bias as well as the two Linear layers and cannot be applied to HF V1. GRN statistics still depend on retained channels. No historical timm checkpoint is registered. [HF 4.57.3 ConvNeXt source](https://raw.githubusercontent.com/huggingface/transformers/v4.57.3/src/transformers/models/convnext/modeling_convnext.py) · [timm V2 and GRN source](https://raw.githubusercontent.com/huggingface/pytorch-image-models/v1.0.24/timm/layers/grn.py)

Each ratio starts from a new in-memory copy of the same parent. Forward checks, parameter counts and actual-call Conv2d/Linear MACs are recorded before timing. E7 MACs count all three views, while parameter counts count shared tensors once. FLOPs use two per counted MAC and exclude normalization, activation, pooling and miscellaneous operations. For replay, apply the recorded indices through `pruning.prune_hf(original, ratio, topology=...)` before strictly loading the reported derivative `inference_state_dict.pt` with `weights_only=True`. The narrower state dict cannot be loaded directly into the original topology.

The staged recovery design follows the independent 10%, 30%, and 50% sweeps and three recovery epochs in the [archived V2 and HF V1 notebook audit](../../docs/OPTIMIZATION_COMPATIBILITY.md#structured-pruning-details). Those notebooks select recovery epochs by validation macro-F1, the V2 sweep also evaluates test, and the HF V1 sweep disables class weights throughout. Here every arm exports epoch 3, and the same frozen validation subset compares the completed arms. Test data remain sealed. The added 0% control receives the same three epochs so the recovery effect can be compared with and without pruning.

E5 uses its native categorical and ordinal objective. E6 uses its native LDAM, ordinal and adjacent-grade objective. E7 independently reuses its own `ViewCollator` and `_train_batches`, with EXIF/RGB decoding from verified training images, B1 augmentation before geometry, one 384px global view and two 224px local patches. Its optimizer includes the shared backbone, both heads and the gate. E7 starts from the archived selected raw or EMA weights. The verified APTOS2019 parent uses raw weights. When a verified parent uses EMA, recovery initializes a fresh shadow from the selected weights and exports the final shadow.

All four arms start independently from their own verified prepared state. Recovery uses fresh AdamW, original learning rates, weight decay and clipping, effective batch 32, loss weighting fixed at the selected parent epoch, and augmentation epochs that continue from that epoch. Inputs are decoded uncached. Recovery does not load training caches, train on validation rows or use validation for early stopping. B0–B4 recovery remains outside this workflow.

## Future execution commands

Use the reviewed checkout and existing research environment. Phase A does not execute the following real-checkpoint commands. The four E5/E6 paths are the original HPC run locations recorded in the published validation evidence. For an authorized relocated copy, use its actual path and preserve the same binary and sidecar. No package installation, model download or host connection is part of this workflow.

```bash
conda activate convnext-research
python -m novelty_integrated.optimization registry

ROOT="$HOME/CAD-ViT-Research/Kel_13"
SNAPSHOT="$ROOT/hf_foundation/models--facebook--convnext-tiny-224/snapshots/6166b7613034066690a621d8bf25ffdf181a34f0"
DATASET=APTOS2019                         # Or DDR, with its own checkpoint and sidecar.
ARM=E5                                  # Or E6, with its matching original checkpoint.
case "$DATASET:$ARM" in
  APTOS2019:E5) RUN=foundation_e5/APTOS2019/integrated_seed42_b32_20_cached/run ;;
  DDR:E5) RUN=foundation_e5/DDR/integrated_seed42_b32_20_cached/run_retry1 ;;
  APTOS2019:E6) RUN=foundation_e6/APTOS2019/ldam_ordinal_adjacent_seed42_b32_20/run ;;
  DDR:E6) RUN=foundation_e6/DDR/ldam_ordinal_adjacent_seed42_b32_20/run ;;
  *) printf 'Choose an eligible E5/E6 dataset and arm\n' >&2; exit 1 ;;
esac
CHECKPOINT="$ROOT/$RUN/best.pt"
CSV="$ROOT/cad_vit_runs/manifests/${DATASET}_seed42.csv"
SOURCE_ROOT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["root"])' "${CSV%.csv}.json")
OUT="$ROOT/optimization_runs"
BASE=(--dataset "$DATASET" --arm "$ARM" --artifact-root "$ROOT"
      --manifest-source-root "$SOURCE_ROOT" --model-dir "$SNAPSHOT"
      --checkpoint "$CHECKPOINT" --trust-checkpoint --trust-legacy-snapshot
      --batch-size 1 --threads 4 --warmup 20 --repeats 3)

# Phase A, read-only plan. This does not create the output directory or read data/weights.
python -m novelty_integrated.optimization run "${BASE[@]}" --output "$OUT/${DATASET}_${ARM}_backends"

# Phase B, only after authorization. Choose a fresh output and an approved CPU allocation.
mkdir -p "$OUT"
python -m novelty_integrated.optimization run "${BASE[@]}" --execute \
  --backend eager torchscript onnxruntime openvino --output "$OUT/${DATASET}_${ARM}_backends"

# Phase B, pruning without recovery. Each parent and derivative uses the same validation rows.
python -m novelty_integrated.optimization run "${BASE[@]}" --execute \
  --prune 0.1 0.3 0.5 --backend eager --output "$OUT/${DATASET}_${ARM}_pruning"

# Recovery execution uses the staged HPC commands below, which enforce the global ceiling.
```

For B3, pass `--external-source-dir` for the copied `foundation_b3_parallel_runner.py` and `sampler.py`. For B4-Q/B4-R, supply the copied `b4_runner.py` and `b4_common.py`. The approved source SHA-256 receipts are printed by the registry. For B0–B4 benchmarking, omit recovery options. Optional backends require `onnx`, `onnxruntime` and/or `openvino` in the selected environment. The authentic model and processor require `transformers==4.57.3` and local pinned weights. A missing package is recorded explicitly. Each successful run writes `plan.json`, `report.json`, `summary.csv`, `comparison.svg` and per-variant export artifacts under a fresh external directory.

## CPU checks

```bash
python -m unittest tests.test_convnext_optimization tests.test_optimization_campaign tests.test_optimization_coordination -v
```

Tests use synthetic tensors, temporary generated images and checkpoint fixtures. On Linux, the coordination tests exercise real multiprocess `flock`, including six held slots and a seventh contender. Optional backend and library-specific tests are skipped when their package is missing. A synthetic pass does not establish real-checkpoint numerical equivalence, latency or classification quality.

## Staged HPC campaign

Use this workflow for independent optimization notebooks and isolated timings. The original single-command Phase A interface above remains available for development. The staged campaign uses the same checkpoint adapters, pruning, export, numerical agreement, quality evaluation and reporting. The archived [partner experiments](../../docs/OPTIMIZATION_COMPATIBILITY.md) provide context. HF ConvNeXt V1 expansion paths and timm ConvNeXtV2 GRN paths stay distinct.

### Read-only readiness audit

From an **authorized HPC compute allocation or JupyterLab terminal**, open the review checkout with this code available. Use the existing environment. Audits read and hash original files, so avoid running them during a timing window. A directory name and matching archive receipts establish expected identity. This session cannot verify live HPC binary presence or allocated resources from Windows. Run the audit below on the host that will do the work.

```bash
conda activate convnext-research
cd "$REVIEW_CHECKOUT"                         # already-provisioned checkout
export CADVIT_ARTIFACT_ROOT="$HOME/CAD-ViT-Research/Kel_13"
export CADVIT_OPT_CAMPAIGN="$CADVIT_ARTIFACT_ROOT/optimization_runs/fp32_full_val_v1"
PYTHON="$(command -v python)"
"$PYTHON" -m novelty_integrated.optimization.hpc inventory

for job in E5_APTOS2019 E6_APTOS2019 E7_APTOS2019 E5_DDR E6_DDR; do
  "$PYTHON" -m novelty_integrated.optimization.hpc audit --job "$job" || break
done
```

The inventory is a manifest derived from the published registry. It lists per-job checkpoint path, SHA-256, best epoch, frozen manifest and validation count. The audit verifies the original binary and adjacent sidecar, code provenance, official snapshot, frozen manifest and metadata, dependency availability, CPU affinity/quota and available RAM/disk. The default backend set requires PyTorch, ONNX, ONNX Runtime and OpenVINO. An explicitly reduced set such as `--backend eager torchscript` must use a different campaign root and must remain identical across stages. `--checkpoint` and `--snapshot` accept authorized relocated files and retain all digest guards. B3/B4 also require `--external-source-dir` on every stage. E7 DDR remains ineligible for execution. Its legacy registry flag `PENDING_TRAINING` indicates that an eligible optimization checkpoint receipt is absent. Its observed research status is partial, with ten completed epochs.

### Allocations and workload

Request approved resources before launch. Set the shell variables below to the scheduler or lab allocation actually assigned, using logical CPU IDs from the affinity audit. Timing requires an exclusive set of physical CPU cores with no competing SMT siblings and an isolated storage window. The CLI checks Linux affinity, cgroup quota when exposed, free RAM and free disk. Its reservation references and flags record operator confirmation. They cannot allocate resources or enforce site-wide isolation.

| Work | Priority pairs | Parallel limit | Conservative resource gate | Measured call count |
| --- | ---: | ---: | --- | ---: |
| Export and pruning preparation | 5 | 2 on disjoint allocated CPUs | 2 CPU threads per worker, ≥6 GiB available RAM per worker with ≥12 GiB at launch for two, plan 3 GiB added per pair. The launch guard reserves for two workers (6 GiB) plus a 5 GiB free shared reserve. | None |
| Original and pruned timing | 5 | 1 | 4 isolated CPU threads, ≥6 GiB available RAM, 1 GiB extra plus reserve, CPU only | APTOS 35,136 per pair, DDR 120,192 per pair |
| Recovery | 5 eligible priority pairs, E5/E6 and completed E7 APTOS2019 | CPU 1 by default, up to 6 globally; GPU 1 | Approved resources and E7 DDR safety confirmation. CPU pinned to one or two logical CPUs on distinct physical cores per worker. GPU only with an allocated idle UUID and ≥8 GiB free VRAM as a preflight guard. | None |
| Untimed validation after recovery | 5 eligible priority pairs | 1 | Shared CPU using one logical CPU and one thread | None |
| Timing with recovered arms | 5 eligible priority pairs | 1 | Same isolated CPU policy, remeasure original and pruned controls in the same job | Twice the corresponding unrecovered counts |
| Later B0–B4 | 10 receipts | Same stage limits | Two distinct APTOS B4 arms. B3/B4 require external source receipts. | Same formula by dataset |

Timing counts assume four backends pass compatibility and parity, batch 1, four variants, two scopes and three repetitions. They exclude warm-up, quality checks, preprocessing verification and export. The five priority pairs total **345,792 timed calls**. Available host resources, wall-clock duration and output sizes on HPC are unmeasured. The resource floors are allocation policy, not measured memory demand. Apply explicit lab quota limits in addition to the filesystem free-space guard. Budget about 15 GiB for prepared outputs. Start with 23 GiB free so the last job can still meet the 11 GiB combined launch guard if four preceding pairs each consume 3 GiB. Budget recovery storage separately.

A separate fixed, grade-complete pilot can be run with `--sample-count 100` at a different campaign root. It fails before export if the chosen subset omits a grade. Use the pilot’s raw per-pass latencies to estimate full duration. Multiply each scope and variant’s measured mean call time by the full sample count and repeats, sum across scopes and backends, then add observed per-stage overhead separately. Do not substitute training throughput or historical timings from a different host.

### Untimed validation on shared CPU

With the already prepared `pilot_100_v1` campaign, use `validate` for parity and classification quality. It restores each prepared eager state, verifies the recorded artifact hashes and runs each backend on the same 100 frozen validation rows in manifest order. It uses the same preprocessing, FP32 logits, tolerances and argmax gate as measurement. E7 still receives the global image and both local views. There are no warm-up passes, latency or throughput estimates, exports or recovery. The environment and sample arguments must match preparation. The staged runner accepts the reviewed source maps of `c101b1a` (prepare) and `f406ddf` (validation) for reuse. Every other protocol field and artifact hash must match. Recovery has a separate manifest that binds both original receipts and the current recovery source. To finish an incomplete preparation from a prior revision, use that preparation revision or a new campaign root.

Use the reviewed checkout in an approved shared HPC session. Select one logical CPU permitted by the session and avoid the CPU and storage resources assigned to other lab jobs. The command below takes the first allowed CPU by default. Set `VALIDATION_CPU` beforehand if the lab has assigned another. It pins the process before Python starts, lowers priority and limits the runtime thread pools. The allocation ID is a report label for shared CPU, and the report records `cpu_mode: shared` and `exclusive_cpu_confirmed: false`.

```bash
conda activate convnext-research
cd "$REVIEW_CHECKOUT"
export CADVIT_ARTIFACT_ROOT="$HOME/CAD-ViT-Research/Kel_13"
export CADVIT_OPT_CAMPAIGN="$CADVIT_ARTIFACT_ROOT/optimization_runs/pilot_100_v1"
PYTHON="$(command -v python)"
VALIDATION_CPU="${VALIDATION_CPU:-$("$PYTHON" -c 'import os; print(min(os.sched_getaffinity(0)))')}"

validate_pilot_job() {
  taskset -c "$VALIDATION_CPU" nice -n 10 env \
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PYTHON" -m novelty_integrated.optimization.hpc validate --job "$1" \
    --sample-count 100 --shared-cpu --threads 1 --cpus "$VALIDATION_CPU" \
    --allocation-id shared-cpu-pilot100-validation
}

# One job first. Review its report before starting the remaining jobs.
validate_pilot_job E5_APTOS2019
```

```bash
# All five sequentially. The first job reuses its completed report.
failed_jobs=()
for job in E5_APTOS2019 E6_APTOS2019 E7_APTOS2019 E5_DDR E6_DDR; do
  validate_pilot_job "$job" || failed_jobs+=("$job")
done
if ((${#failed_jobs[@]})); then
  printf 'Review failed or blocked job: %s\n' "${failed_jobs[@]}" >&2
fi
```

Per job, open `<campaign>/<job>/validation/report.json` and `summary.csv`. Each table has one row per variant and backend, with argmax agreement, maximum absolute logit error, macro-F1, accuracy, QWK, and grade MAE. Inspect the JSON for the remaining parity diagnostics, frozen validation positions, input digest and both preparation and execution source hashes. Each variant has its own eager reference. Metrics on 100 validation images describe that subset. `PARITY_FAILED` still reports independently computed quality. A backend that cannot load or infer has no quality numbers. A nonzero exit marks a failed or blocked job. Audit and integrity errors block publication and are printed to the terminal. Completed reports are immutable, including those with explicit backend failures. An interrupted job can be run again with the same arguments. Validation uses the existing coordinator and per-job locks. Both timing stages still require their exclusive CPU confirmation.

### tmux launch and continuation

Run the helper below in the authorized terminal. The command arguments use the same checkout, environment, campaign path and reviewed Python interpreter. Use a different session name and log for a retry. The helper launches only the command passed to it.

```bash
logdir="$CADVIT_OPT_CAMPAIGN/logs"
mkdir -p "$logdir"
"$PYTHON" -m novelty_integrated.optimization.hpc inventory > "$logdir/jobs.json"

launch_opt() {
  local session="$1"; shift
  tmux new-session -d -s "$session" -c "$PWD" -- bash -c '
    log="$1"; shift
    "$@" >>"$log" 2>&1
  ' bash "$logdir/$session.log" env \
    CADVIT_ARTIFACT_ROOT="$CADVIT_ARTIFACT_ROOT" CADVIT_OPT_CAMPAIGN="$CADVIT_OPT_CAMPAIGN" \
    "$PYTHON" -m novelty_integrated.optimization.hpc "$@"
}

: "${PREP_ALLOCATION:?set the approved preparation allocation identifier}"
: "${PREP_CPUS_A:?set the allocated IDs for worker A}"
: "${PREP_CPUS_B:?set the distinct allocated IDs for worker B}"
launch_opt opt-prep-e5a prepare --job E5_APTOS2019 --allocation-id "$PREP_ALLOCATION" --cpus "$PREP_CPUS_A" --threads 2
launch_opt opt-prep-e6a prepare --job E6_APTOS2019 --allocation-id "$PREP_ALLOCATION" --cpus "$PREP_CPUS_B" --threads 2
```

Wait for both sessions to exit and verify their logs. Repeat for E7 APTOS2019 and E5 DDR, then E6 DDR. Never launch more than two preparation jobs. Coordinator locks prevent timing, untimed validation and preparation from overlapping recovery. Keep DDR E7 and other lab jobs undisturbed.

```bash
: "${MEASURE_ALLOCATION:?set the exclusive CPU measurement allocation identifier}"
: "${MEASURE_CPUS:?set the allocated IDs for exclusive timing}"
launch_opt opt-time-e5a measure --job E5_APTOS2019 --allocation-id "$MEASURE_ALLOCATION" \
  --cpus "$MEASURE_CPUS" --threads 4 --confirm-isolated-cpu
```

Run one timing session at a time in priority order. Confirm there are no overlapping preparations or other workloads competing for resources before each timing session. `tmux ls` and `tail -f "$logdir/opt-time-e5a.log"` show activity without touching other sessions. To run directly instead of tmux, replace `launch_opt <session>` with `"$PYTHON" -m novelty_integrated.optimization.hpc`.

### Three-epoch recovery and untimed validation

Start with the shell setup in [Untimed validation on shared CPU](#untimed-validation-on-shared-cpu) so the environment and `pilot_100_v1` path match the existing campaign. Use an updated review checkout separate from any checkout serving a live training or validation job. Review the chosen job's `validation/report.json` for `outcome: PASS`. Every unrecovered variant/backend must pass. DDR E5 and E6 can queue only after their passing reports exist. `measure` is independent of this gate.

Use resources approved for sustained recovery, including CPU, RAM and filesystem I/O that leave the existing DDR E7 job undisturbed. The GPU remains hidden for CPU recovery. GPU recovery requires a separately reserved idle GPU UUID and remains exclusive. `--max-recovery-workers` defaults to 1. Request a higher ceiling from 2 to 6 only for full four-arm CPU jobs, after the lab has approved parallel work. Each worker takes one or two logical CPUs on distinct physical cores. No two recovery workers can reserve the same core or its SMT sibling. Affinity limits where the process runs and does not establish an exclusive lab allocation.

CPU recovery leases use a private, boot-specific directory under `/var/tmp` for the research account. All participating terminals and checkouts on one host must see the same directory and working Linux `flock` semantics. This ceiling spans artifact roots and campaigns for that account on the host. The runner also holds the original artifact-root locks so that older checkouts using the same root continue to coordinate. Do not remove the lock files while jobs are active. The kernel releases slots, job locks and core leases when a process exits. The smallest ceiling requested by any active worker stays in force until that worker exits. `measure`, `measure-recovered`, GPU recovery and the untimed validation stages take the exclusive coordinator lock. Preparation and CPU recovery also exclude one another.

Admission samples host and visible cgroup CPU, memory and I/O pressure (PSI), available RAM, CPU quota, and free bytes and inodes on the recovered output filesystem. It accounts conservatively for active workers' reservations. CPU `some`, memory `full`, or I/O `full` PSI at 25%, 0.5%, or 2% over the latest 10-second window caps new admissions at one worker. At 60%, 2%, or 10%, no new worker is admitted. Missing host PSI or cgroup-v2 CPU quota telemetry also caps new admissions at one. These are admission thresholds, not measured model requirements. A worker already running can finish. Check the lab's filesystem quota, I/O policy and live training allocation separately. With defaults, five full jobs reserve 30 GiB of RAM and budget 40 GiB of output growth plus 5 GiB free on one filesystem. Six need 36 GiB and 48 GiB plus the same reserve. Admission can still defer workers as free space and available RAM change.

```bash
: "${RECOVERY_ALLOCATION:?set the separately approved recovery allocation identifier}"
: "${RECOVERY_CPU:?set one approved logical CPU for recovery}"
RECOVERY_JOB=E5_APTOS2019
recover_arm() {
  taskset -c "$RECOVERY_CPU" nice -n 10 env \
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PYTHON" -m novelty_integrated.optimization.hpc recover --job "$RECOVERY_JOB" \
    --sample-count 100 --allocation-id "$RECOVERY_ALLOCATION" --cpus "$RECOVERY_CPU" --threads 1 \
    --confirm-ddr-e7-safe --authorize-recovery --recovery-device cpu \
    --recovery-epochs 3 --recovery-micro-batch 4 --recovery-ratio "$1"
}

# Run one arm, then inspect its output before scheduling the others.
recover_arm 0.1
# Use the same command to resume an interrupted 10% arm.
# After inspection, complete the 0% control and the two other pruning ratios.
for ratio in 0 0.3 0.5; do
  recover_arm "$ratio" || break
done
```

For all five priority jobs, the launcher below starts one tmux session and log per job. Set the five CPU variables to approved logical CPU IDs on different physical cores. Leave the ceiling at 1 on a busy host or set `RECOVERY_MAX_WORKERS` only to the approved limit. Each session uses a full sequential 0%, 10%, 30%, 50% sweep. Exit 75 means a lock or resource gate deferred admission. The launcher retries that exit every 60 seconds and logs the reason. Validation or training failures stop that job and retain the log. The five jobs become eligible independently after passing `validate`.

```bash
: "${RECOVERY_ALLOCATION:?set the lab-approved recovery allocation identifier}"
: "${RECOVERY_CPU_E5_APTOS2019:?set the approved logical CPU ID}"
: "${RECOVERY_CPU_E6_APTOS2019:?set the approved logical CPU ID}"
: "${RECOVERY_CPU_E7_APTOS2019:?set the approved logical CPU ID}"
: "${RECOVERY_CPU_E5_DDR:?set the approved logical CPU ID}"
: "${RECOVERY_CPU_E6_DDR:?set the approved logical CPU ID}"
RECOVERY_MAX_WORKERS="${RECOVERY_MAX_WORKERS:-1}"
RECOVERY_QUEUE_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
jobs=(E5_APTOS2019 E6_APTOS2019 E7_APTOS2019 E5_DDR E6_DDR)
cores=("$RECOVERY_CPU_E5_APTOS2019" "$RECOVERY_CPU_E6_APTOS2019" "$RECOVERY_CPU_E7_APTOS2019"
       "$RECOVERY_CPU_E5_DDR" "$RECOVERY_CPU_E6_DDR")
logdir="$CADVIT_OPT_CAMPAIGN/logs"
mkdir -p "$logdir"

launch_recovery() {
  local job="$1" cpu="$2" session="recovery-$1-$RECOVERY_QUEUE_TAG"
  tmux new-session -d -s "$session" -c "$PWD" -- bash -c '
    log="$1"; shift
    exec >>"$log" 2>&1
    while :; do
      "$@"
      rc=$?
      printf "[%s] recovery exit=%s\n" "$(date -u +%FT%TZ)" "$rc"
      if ((rc != 75)); then exit "$rc"; fi
      sleep 60
    done
  ' bash "$logdir/$session.log" env \
    CADVIT_ARTIFACT_ROOT="$CADVIT_ARTIFACT_ROOT" CADVIT_OPT_CAMPAIGN="$CADVIT_OPT_CAMPAIGN" \
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    taskset -c "$cpu" nice -n 10 "$PYTHON" -u -m novelty_integrated.optimization.hpc recover \
    --job "$job" --sample-count 100 --allocation-id "$RECOVERY_ALLOCATION" --cpus "$cpu" --threads 1 \
    --confirm-ddr-e7-safe --authorize-recovery --recovery-device cpu --recovery-epochs 3 \
    --recovery-micro-batch 4 --max-recovery-workers "$RECOVERY_MAX_WORKERS"
}

if [[ "$RECOVERY_MAX_WORKERS" =~ ^[1-6]$ ]] && "$PYTHON" -c '
import os, sys
from novelty_integrated.optimization.coordination import physical_cores
cpus = list(map(int, sys.argv[1:]))
if not set(cpus) <= os.sched_getaffinity(0):
    raise SystemExit("At least one requested CPU is outside this session affinity")
print("Recovery physical core sibling groups:", physical_cores(cpus))
' "${cores[@]}"; then
  for i in "${!jobs[@]}"; do launch_recovery "${jobs[i]}" "${cores[i]}" || break; done
fi
```

```bash
"$PYTHON" -m novelty_integrated.optimization.hpc recovery-status
tmux ls
tail -f "$logdir/recovery-E5_APTOS2019-$RECOVERY_QUEUE_TAG.log"
```

Inspect `RECOVERY_ADMITTED` and `RECOVERY_DEFERRED` in each log for effective capacity, resource limits and conflicts. After an interruption, use a new `RECOVERY_QUEUE_TAG` and `launch_recovery <job> <same-CPU-ID>` for the affected job after its old session exits. Use the same model, recovery settings and device. The worker ceiling can change without changing the recovery protocol. Budget 12 recovery epochs per job and 60 across the five jobs. Wall time and memory use have not been measured on HPC. Checkpoints with model, optimizer, RNG and optional EMA state are stored atomically at completed epoch boundaries under `recovered/progress/`. An interrupted epoch restarts from the last complete boundary. An interrupted export can reuse the saved final epoch. Completed arms are checked and skipped. The separate `recovered/recovery_protocol.json` binds the original protocol, prepared receipts and existing validation report hashes. A verified recovery protocol from `c7e61a8` can resume with its original hashes and checkpoint. Original artifacts, validation reports and training worktrees stay read-only.

```bash
# Once the four arms are ready and all recovery workers have released the coordinator lock.
taskset -c "$VALIDATION_CPU" nice -n 10 env \
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PYTHON" -m novelty_integrated.optimization.hpc validate-recovered --job "$RECOVERY_JOB" \
  --sample-count 100 --shared-cpu --threads 1 --cpus "$VALIDATION_CPU" \
  --allocation-id shared-cpu-pilot100-validation
```

Read `validation-recovered/summary.csv` for the backend table and `validation-recovered/report.json` for matched before/after metric deltas and ranking. Each backend is compared with its own recovered eager variant on the exact same ordered validation inputs used before recovery. Rank the three pruned ratios by eager macro-F1, then lower grade MAE, then smaller ratio. The 0% control is excluded from selection. All backend parity checks must pass before the report selects a ratio. Selection from this reused subset is exploratory. `measure` and `measure-recovered` remain available separately and require an exclusive allocation.

Each pair has a `protocol.json` and independent `prepared/`, `validation/`, `recovered/`, `validation-recovered/`, `measurement/`, and `post-recovery/` outputs. Completed reports and exports are write-once. An interrupted timing job measures the pair again to keep comparisons within one allocation. Protocol changes outside the documented compatibility case require a new campaign root. Temporary or derived weights and logs remain external.

Set `CADVIT_OPT_CAMPAIGN` in JupyterLab to display reviewed reports through the [optimization notebook index](../../notebooks/optimization/README.md). The stage writes `report.json`, `summary.csv`, `comparison.svg`, and `tradeoff.svg` only after a completed measurement. No production results are included in the notebook templates.
