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
| E7 Global/Local | COMPLETED, best 19; MISSING_CHECKPOINT locally | PENDING_TRAINING | The [audited APTOS2019 export](../../results/APTOS2019/e7_global_local/validation_report.md) matches the HPC best/last digests and identifies raw selected weights. The binary must be supplied and verified before loading. Loading the real checkpoint, exporting it and benchmarking it remain NOT RUN. One global 384×384 and two local 224×224 tensors share a backbone and learned fusion gate. |

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

Recovery is separate and requires explicit authorization. It implements E5/E6 only by calling their distinct source batch trainers. It uses original train rows, B1 augmentation, fresh AdamW, original learning rates, weight decay and clipping, and effective batch 32. It freezes deferred class-weight activation at the selected parent epoch and advances augmentation seeds from that epoch. This is a new, fixed-duration recovery design, with the final recovery epoch selected without tuning on the measurement validation sequence. Record the optional 0% recovery control to distinguish recovery effects from pruning. B0–B4 and E7 recovery are unsupported. CPU is the default; a future CUDA recovery must have a separately approved allocation and pass the explicit acknowledgement gate. No inference timings include the recovery workload.

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

# Phase C, only after separate authorization. The 0% arm is a recovery control.
python -m novelty_integrated.optimization run "${BASE[@]}" --execute --authorize-recovery \
  --prune 0 0.1 0.3 0.5 --recovery-epochs 3 --recovery-micro-batch 4 \
  --backend eager --output "$OUT/${DATASET}_${ARM}_recovery"
```

For B3, pass `--external-source-dir` for the copied `foundation_b3_parallel_runner.py` and `sampler.py`. For B4-Q/B4-R, supply the copied `b4_runner.py` and `b4_common.py`. The approved source SHA-256 receipts are printed by the registry. For B0–B4 benchmarking, omit recovery options. Optional backends require `onnx`, `onnxruntime` and/or `openvino` in the selected environment. The authentic model and processor require `transformers==4.57.3` and local pinned weights. A missing package is recorded explicitly. Each successful run writes `plan.json`, `report.json`, `summary.csv`, `comparison.svg` and per-variant export artifacts under a fresh external directory.

## CPU checks

```bash
python -m unittest tests.test_convnext_optimization -v
```

Tests use synthetic tensors, temporary generated images and checkpoint fixtures. Optional backend and library-specific tests are skipped when their package is missing. A synthetic pass does not establish real-checkpoint numerical equivalence, latency or classification quality.

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

The inventory is a manifest derived from the published registry. It lists per-job checkpoint path, SHA-256, best epoch, frozen manifest and validation count. The audit verifies the original binary and adjacent sidecar, code provenance, official snapshot, frozen manifest and metadata, dependency availability, CPU affinity/quota and available RAM/disk. The default backend set requires PyTorch, ONNX, ONNX Runtime and OpenVINO. An explicitly reduced set such as `--backend eager torchscript` must use a different campaign root and must remain identical across stages. `--checkpoint` and `--snapshot` accept authorized relocated files and retain all digest guards. B3/B4 also require `--external-source-dir` on every stage. E7 DDR rejects execution until a completed receipt is integrated in the registry.

### Allocations and workload

Request approved resources before launch. Set the shell variables below to the scheduler or lab allocation actually assigned, using logical CPU IDs from the affinity audit. Timing requires an exclusive set of physical CPU cores with no competing SMT siblings and an isolated storage window. The CLI checks Linux affinity, cgroup quota when exposed, free RAM and free disk. Its reservation references and flags record operator confirmation. They cannot allocate resources or enforce site-wide isolation.

| Work | Priority pairs | Parallel limit | Conservative resource gate | Measured call count |
| --- | ---: | ---: | --- | ---: |
| Export and pruning preparation | 5 | 2 on disjoint allocated CPUs | 2 CPU threads per worker, ≥6 GiB available RAM per worker with ≥12 GiB at launch for two, plan 3 GiB added per pair. The launch guard reserves for two workers (6 GiB) plus a 5 GiB free shared reserve. | None |
| Original and pruned timing | 5 | 1 | 4 isolated CPU threads, ≥6 GiB available RAM, 1 GiB extra plus reserve, CPU only | APTOS 35,136 per pair, DDR 120,192 per pair |
| Recovery preparation | 4 eligible priority pairs, E5/E6 | 1 | Separate explicit approval. CPU default. GPU only with an allocated idle UUID, DDR E7 safety confirmed and ≥8 GiB free VRAM as a preflight guard | None |
| Timing with recovered arms | 4 eligible priority pairs | 1 | Same isolated CPU policy, remeasure original and pruned controls in the same job | Twice the corresponding unrecovered counts |
| Later B0–B4 | 10 receipts | Same stage limits | Two distinct APTOS B4 arms. B3/B4 require external source receipts. | Same formula by dataset |

Timing counts assume four backends pass compatibility and parity, batch 1, four variants, two scopes and three repetitions. They exclude warm-up, quality checks, preprocessing verification and export. The five priority pairs total **345,792 timed calls**. Available host resources, wall-clock duration and output sizes on HPC are unmeasured. The resource floors are allocation policy, not measured memory demand. Apply explicit lab quota limits in addition to the filesystem free-space guard. Budget about 15 GiB for prepared outputs. Start with 23 GiB free so the last job can still meet the 11 GiB combined launch guard if four preceding pairs each consume 3 GiB. Budget recovery storage separately.

A separate fixed, grade-complete pilot can be run with `--sample-count 100` at a different campaign root. It fails before export if the chosen subset omits a grade. Use the pilot’s raw per-pass latencies to estimate full duration. Multiply each scope and variant’s measured mean call time by the full sample count and repeats, sum across scopes and backends, then add observed per-stage overhead separately. Do not substitute training throughput or historical timings from a different host.

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

Wait for both sessions to exit and verify their logs. Repeat for E7 APTOS2019 and E5 DDR, then E6 DDR. Never launch more than two preparation jobs. Coordinator locks serialize timing and recovery and reject overlapping preparation CPUs and concurrent writers for the same job. They coordinate only processes using this campaign framework on a filesystem with working POSIX locks. Don’t use this workflow on a shared filesystem that cannot provide those locks. Keep DDR E7 and other lab jobs undisturbed.

```bash
: "${MEASURE_ALLOCATION:?set the exclusive CPU measurement allocation identifier}"
: "${MEASURE_CPUS:?set the allocated IDs for exclusive timing}"
launch_opt opt-time-e5a measure --job E5_APTOS2019 --allocation-id "$MEASURE_ALLOCATION" \
  --cpus "$MEASURE_CPUS" --threads 4 --confirm-isolated-cpu
```

Run one timing session at a time in priority order. Confirm there are no overlapping preparations or other workloads competing for resources before each timing session. `tmux ls` and `tail -f "$logdir/opt-time-e5a.log"` show activity without touching other sessions. To run directly instead of tmux, replace `launch_opt <session>` with `"$PYTHON" -m novelty_integrated.optimization.hpc`.

Recovery is opt-in and separate from measurement. Start it only after reviewing the unrecovered results and confirming it is safe for DDR E7. This command uses CPU. A GPU allocation also requires `--recovery-device cuda:0 --gpu-uuid <allocated-UUID> --confirm-ddr-e7-safe`. Verify GPU assignment with the lab and avoid a GPU shared with another job.

```bash
: "${RECOVERY_ALLOCATION:?set the separately approved recovery allocation identifier}"
: "${RECOVERY_CPUS:?set the allocated IDs for the recovery job}"
launch_opt opt-recover-e5a recover --job E5_APTOS2019 --allocation-id "$RECOVERY_ALLOCATION" \
  --cpus "$RECOVERY_CPUS" --threads 4 --authorize-recovery --recovery-epochs 3
# After this stage completes, use a separate isolated timing session.
launch_opt opt-post-e5a measure-recovered --job E5_APTOS2019 --allocation-id "$MEASURE_ALLOCATION" \
  --cpus "$MEASURE_CPUS" --threads 4 --confirm-isolated-cpu
```

Each pair has a `protocol.json` and independent `prepared/`, `measurement/`, `recovered/` and `post-recovery/` outputs. Repeat the same stage and arguments to continue after interruption. Preparation and recovery reuse completed arms after checking their artifact hashes and protocol. An incomplete arm restarts from its parent. Completed timing directories are write-once. An interrupted timing job measures the pair again to keep comparisons within one allocation. Recovery timing remeasures original, unrecovered and recovered arms together. Backend failures and missing reports remain explicit in JSON. Any changed source, dependency, selection or checkpoint requires a new campaign root. Original checkpoints stay read-only. Temporary or derived weights and logs remain external.

Set `CADVIT_OPT_CAMPAIGN` in JupyterLab to display reviewed reports through the [optimization notebook index](../../notebooks/optimization/README.md). The stage writes `report.json`, `summary.csv`, `comparison.svg`, and `tradeoff.svg` only after a completed measurement. No production results are included in the notebook templates.
