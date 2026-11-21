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
| E7 Global/Local | PENDING_TRAINING | PENDING_TRAINING | Original `GlobalLocalModel` and raw/EMA state adapter are implemented and tested synthetically. CLI execution remains blocked until a real E7 receipt is audited. One global 384×384 and two local 224×224 tensors, one shared backbone, learned fusion gate and categorical head. |

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
