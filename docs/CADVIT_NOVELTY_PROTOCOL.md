# CAD-ViT Phase 1 research prototype

Status: implementation and CPU checks only. This is an ablatable research
prototype, not a verified novel method or a performance result. The historical
reproduction under reproduction/cadvit/ is imported without modification.

## Baseline and checkpoint transfer

The baseline is CADViTCandidate in reproduction/cadvit/model.py with 5,282,101
parameters. At 256 x 256 input, the stem produces [B,32,128,128]; four
stride-two inverted residual blocks reduce the grid to [B,160,8,8]; the head
produces [B,320,8,8] and the original classifier produces five logits.
Three EnhancedMobileViTBlock instances contain the original MFB and transformer;
four InvertedResidualBlock instances contain coordinate attention.

NovelCADViT subclasses the baseline. All original state-dict tensor names,
shapes, and dtypes remain unchanged. transfer_baseline_state rejects missing,
unexpected, or mismatched original tensors, then performs a strict target load.
New tensors are initialized separately. The checkpoint loader requires a
whole-file SHA-256 and uses weights_only=True with only statically observed
metadata types allowlisted.

| Dataset | Archived checkpoint | SHA-256 observed on KC |
| --- | --- | --- |
| APTOS2019 | ~/Kel_13/experiment_archive_80ep_20261105/APTOS/best.pt | e72573f0f1c26baac53e18d950076845952da503261e212021ff6320c9ae72d5 |
| DDR | ~/Kel_13/experiment_archive_80ep_20261105/DDR/best.pt | 844665f0c597142915286b727a21f1d2223f43e0f057d083b6c1dd035c180ac1 |

These are observed file digests, not independent proof of training lineage.
The existing provenance record remains authoritative for the reproduction.

## Prespecified components

P1, local detail residual. For image I, define D = I - AvgPool3x3(I)
with boundary-aware averaging. A learned 3 x 3, stride-two projection maps D
from three channels to 32 and adds it to the unchanged stem output. The
projection starts at zero, so its initial contribution is exactly zero.
This exposes local contrast differences before later downsampling. It does
not establish that lesions or clinically meaningful detail are preserved.
Added parameters: 864.

P2, adaptive MFB. Each of the three original MFBs retains dilations
(3, 5, 7, 9), branch order and weights, concatenation, and fusion weights.
For input X, g = Linear(GAP(X)) has four logits and alpha = 4 softmax(g).
Branch i is multiplied by alpha_i before original concatenation and fusion.
The gate weight and bias start at zero, making each coefficient exactly one.
P2 adds 1,548 parameters. P2-only is also the ordinary adaptive-gating
control without ordinal boundary conditioning.

P3, supervised boundary-conditioned coordinate attention. At each of the
four coordinate-attention sites, the pre-modulation tensor U produces a scalar
s(U) = Linear(GAP(U)). Ordered thresholds are tau_0 = a and
tau_k = a + sum_{j=1}^k softplus(d_j), k=1,2,3. Local cumulative logits
z_k = s(U) - tau_k therefore have sigmoid(z_0) >= ... >= sigmoid(z_3).
A 4-to-C map of these probabilities gives
delta = tanh(Linear(sigmoid(z))). The output is original_CA(U) * (1+delta),
broadcast over height and width. The modulation map starts at zero, so
the original coordinate-attention output is initially preserved exactly.
Each site's logits receive training-only targets 1[y>k] through BCE.
Neither labels nor final predictions enter the inference graph.
P3 adds 3,860 parameters. Its learned scalar is not a validated disease
severity measure. BCE supervision gives the boundary values ordinal meaning;
the P3-only arm changes both architecture and objective. The probe-only flag keeps the same local supervision and original coordinate-attention output, while omitting the modulation map.

P4, global ordinal auxiliary objective. The original five-class classifier
remains the primary inference head. A separate four-logit head reads the
pooled 320-vector and starts with zero weight and bias. Its targets are
t_k = 1[y>k], k=0,1,2,3. P4 adds 1,284 parameters. Ordinal probabilities
never replace the five-class logits.

The exact exploratory loss is:

    L = CE(five_class_logits, y)
      + 0.20 * mean_over_four_sites(BCEWithLogits(local_boundary_logits, t)) [P3 only]
      + 0.20 * BCEWithLogits(global_ordinal_logits, t)                  [P4 only]

Each BCE uses the PyTorch mean over batch and four boundaries. The 0.20
coefficients are prespecified pilot choices, not tuned values or claims of
optimal weighting. A training protocol must log weights and active flags;
changing either creates a different scientific configuration.

## Explicit ablations

| Configuration | P1 | P2 | P3 | P4 | Total/trainable parameters |
| --- | --- | --- | --- | --- | ---: |
| baseline | no | no | no | no | 5,282,101 |
| p1_only | yes | no | no | no | 5,282,965 |
| p2_only / ordinary_gate_control | no | yes | no | no | 5,283,649 |
| p3_probe_only | no | no | supervised probe only | no | 5,282,761 |
| p3_only | no | no | probe plus modulation | no | 5,285,961 |
| p4_only | no | no | no | yes | 5,283,385 |
| p2_p3_probe | no | yes | supervised probe only | no | 5,284,309 |
| p2_p3_full | no | yes | probe plus modulation | no | 5,287,509 |
| p123 | yes | yes | probe plus modulation | no | 5,288,373 |
| p1234 | yes | yes | probe plus modulation | yes | 5,289,657 |

NoveltyFlags.to_dict() is JSON-serializable. All prototype parameters are
currently trainable; no training run is provided in this phase.

## Checks and limits

Synthetic 64 x 64 CPU input preserves the baseline's final 2 x 2 patch grid.
Tests check exact initial five-logit equivalence after strict transfer, forward
shapes for batch 1 and 2, finite outputs and loss, gradients to added modules,
ordinal encoding, independent flags, dilation rates, and both archived
checkpoints. No dataset image, validation prediction, test image, or real training is used in these architecture tests. Synthetic optimizer-step and GPU smoke checks are separate from a pilot.

The original candidate architecture is a partial paper-guided implementation:
its 5.28M parameters differ from the paper's approximately 7.26M. This
prototype inherits that fidelity limit. P1 may respond to non-lesion noise;
P2 may merely reweight known dilation branches; P3's local boundary probes
may be poorly calibrated; P4's auxiliary loss may compete with classification.
An initialization match proves compatibility, not performance or novelty.
Prior work already studies [adaptive multiscale weighting for DR](https://www.mdpi.com/2079-9292/10/12/1369)
and [cumulative ordinal thresholds in medical diagnosis](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11134/Liu_Ordinal_Regression_with_Neuron_Stick-breaking_for_Medical_Diagnosis_ECCVW_2018_paper.pdf).
A broader targeted prior-art audit and controlled validation-only pilot are
needed before a methodological novelty claim. The frozen final test must not
be used for method selection.

## Targeted prior-art screen

This screen compares the four ideas at the component level and does not
establish a new method. P1 is adjacent to published residual-attention DR
classification work ([2023 DR residual-attention study](https://onlinelibrary.wiley.com/doi/10.1155/2023/1305583));
the high-pass projection here is a testable design choice, not a new lesion
detector. P2 has direct conceptual overlap with [adaptive multiscale
weighting for DR](https://www.mdpi.com/2079-9292/10/12/1369). The original
[coordinate-attention formulation](https://openaccess.thecvf.com/content/CVPR2021/papers/Hou_Coordinate_Attention_for_Efficient_Mobile_Network_Design_CVPR_2021_paper.pdf)
already provides spatially directed feature modulation. P3 adds a supervised
local ordinal condition to that baseline, but related attention and ordinal
conditioning must be compared more thoroughly before any originality claim.
P4's cumulative targets are established in [ordinal medical-image
classification](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11134/Liu_Ordinal_Regression_with_Neuron_Stick-breaking_for_Medical_Diagnosis_ECCVW_2018_paper.pdf).
The experiment needs ordinary gating and single-component controls to
distinguish effects; combining known modules is not itself novelty.

## Verification record

On the existing KC preof-kc worktree at 3a563c6, CPU tests loaded all 285
original state tensors from each archived APTOS and DDR checkpoint with no
missing, unexpected, shape-mismatched, or dtype-mismatched originals. The full
prototype adds 33 state tensors. For synthetic 64 x 64 inputs, initial
five-class logits matched the baseline exactly (absolute tolerance zero).
Ten updated prototype CPU tests and five existing baseline model tests passed.
A synthetic batch-two GPU forward/backward smoke test on an RTX 3060 passed;
it used no optimizer step, dataset image, or experiment output. Peak allocated
GPU memory in that smoke was 67.4 MiB, which does not estimate real training
memory at the intended image resolution.


## Phase 2 scientific review and controlled ablation

The local-contrast input to P1 is a fixed high-pass residual followed by a
learned zero-initialized convolution. This provides a route for local contrast,
but does not demonstrate that retinal lesions survive preprocessing or that
the learned projection retains useful detail. Its first update receives
gradient even though its initial output is zero; the synthetic optimizer test
checks that the projection actually changes.

P2 replaces all three MFB instances while retaining each original branch
module, dilation order (3,5,7,9), concatenation, and fusion weights. P3's
ordered boundary logits are computed from the feature tensor entering each
coordinate-attention site. Labels enter only its loss. P3 modulates the
coordinate-attention output by channel; it does not change the spatial
height/width gates themselves. Describe it as boundary-conditioned feature
modulation around coordinate attention, not proven lesion attention.

P3 and P4 use the same ordinal targets at different depths. In the combined
arm their loss coefficients sum to 0.40, which may double-count the ordinal
signal. The full arm is exploratory and must not be used alone to attribute
an improvement to P3 modulation.

The primary ablation matrix is:

| Arm | Question answered |
| --- | --- |
| baseline | Unmodified checkpoint-compatible CAD-ViT continuation with new pilot protocol |
| p1_only | Effect of the local-contrast path |
| p2_only | Ordinary sample-conditioned MFB gating |
| p3_probe_only | Effect of local ordinal supervision without modulation |
| p3_only | Local supervision plus modulation without MFB gating |
| p4_only | Effect of global ordinal auxiliary loss |
| p2_p3_probe | Ordinary MFB gating plus local ordinal supervision |
| p2_p3_full | Same as preceding arm, with boundary-conditioned modulation |
| p1234 | Exploratory combined method; not a clean single-component contrast |

The direct P3 modulation contrast is p3_only versus p3_probe_only.
The key contrast beyond ordinary gating and ordinal supervision is
p2_p3_full versus p2_p3_probe. P2-only versus p2_p3_probe estimates the
effect of adding local ordinal supervision on the same gating background.
Use identical frozen manifest, initialization checkpoint, seed, microbatch,
optimizer, augmentation, and validation-selection rule within each dataset.
Keep APTOS and DDR analyses separate.

Initial five-class output equality was tested in both eval and train mode
after original-state transfer. In train mode the original BatchNorm running
statistics matched after a synthetic forward. Later optimizer updates and
new paths can change those statistics, so initial equivalence does not
guarantee equal trajectories.

## Phase 2 pilot protocol and CLI

Source: novelty/cadvit_pilot.py. It imports the frozen train/validation
manifest verifier from novelty/foundation_adaptation.py. That verifier hashes
each train and validation source image, checks group/content separation, and
skips test rows before reading their labels or paths. Only train and val
datasets can be constructed by the runner.

The selected APTOS and DDR checkpoints are the archived V2 BG10 best.pt
files listed above. The runner checks their exact SHA-256, model tensor
mapping, V2 preprocessing record, manifest hash, dataset, seed, epoch,
learning rate, and batch. DDR additionally requires the weighted-loss
recovery metadata. This is a new adaptation pilot from those checkpoints,
not a continuation of their archived optimizer states. The DDR starting
checkpoint has mixed weighted-FP32 recovery provenance; the pilot uses
ordinary five-class CE plus its prespecified auxiliaries. The protocol
difference must remain explicit.

For each image, the runner calls the existing preprocess_fundus_v2 with
PREPROCESS_CONFIG and BG_V2_CONFIG, then applies train-only horizontal flip
(p=0.5), vertical flip (p=0.5), and RandomAffine (degrees=15, shear=10,
bilinear, fill=0), then contiguous CHW float32 / 255. Validation receives
no augmentation. The geometric augmentation matches the extracted CAD-ViT
runner; an exact historical V2 augmentation source has not been verified.
No cached or newly invented preprocessing is used.

Pilot settings are seed 42, AdamW, learning rate 0.000186, weight decay
zero, FP32, gradient clip 1.0, effective batch 16, and a caller-selected
microbatch dividing 16. The runner permits 1-5 epochs only. Its final
incomplete accumulation window is normalized by its actual image count.
BatchNorm sees the physical microbatch, so the same microbatch must be
used in every ablation comparison. Training metrics are online metrics
under stochastic augmentation and changing weights; validation metrics
come from the frozen validation split and five-class primary logits.

Every epoch records loss, accuracy, macro/weighted F1, class-wise
precision/recall/F1/support/specificity, QWK, MAE, confusion matrix,
time, and peak GPU memory. Validation macro-F1 selects best.pt; exact
ties use lower primary validation CE loss, then the earlier epoch.
last.pt, best.pt, config.json, history.jsonl, and summary.json live in
independent run directories outside Git. The config includes source
commit and source-file SHA-256, manifest and sidecar hashes, verified
image counts, checkpoint hash and recovery metadata, and library versions.
Guarded resume requires matching config, checkpoint, best selection,
history, summary, and saved Python/NumPy/PyTorch/CUDA RNG states.
A crash between atomic checkpoint and history writes fails closed for
manual review.

Example manual commands on KC, from
/mnt/home/probation-03/Kel_13/Diabetic-Rethinography-Novelty:

    ROOT=/mnt/home/probation-03/Kel_13
    OUT=$ROOT/cadvit_novelty_pilots
    ../.venv/bin/python -m novelty.cadvit_pilot preflight --artifact-root "$ROOT" --output-root "$OUT" --dataset APTOS2019 --ablation p2_p3_probe --micro-batch 1 --epochs 2
    ../.venv/bin/python -m novelty.cadvit_pilot smoke --artifact-root "$ROOT" --output-root "$OUT" --dataset APTOS2019 --ablation p2_p3_probe --micro-batch 1 --epochs 2 --device cuda
    ../.venv/bin/python -m novelty.cadvit_pilot pilot --artifact-root "$ROOT" --output-root "$OUT" --dataset APTOS2019 --ablation p2_p3_probe --micro-batch 1 --epochs 2 --device cuda
    ../.venv/bin/python -m novelty.cadvit_pilot resume --artifact-root "$ROOT" --output-root "$OUT" --dataset APTOS2019 --ablation p2_p3_probe --micro-batch 1 --epochs 2 --device cuda

Repeat with --dataset DDR and a separate automatic DDR directory.
For the P3 contrast, run p2_p3_full under the same settings in another
automatic directory. Pilot/resume commands are documentation only; no
real pilot was launched in Phase 2.

The baseline analytic estimate at 512 x 512 is 8.95 GMAC per image,
excluding BatchNorm, activations, softmax, data loading, and new modules.
At the first 64 x 64 EMB grid, four 2 x 2 token streams contain 1,024
tokens each; four attention heads therefore require about 16.8 million
attention-score elements per image per layer, about 64 MiB in FP32 before
backward storage. These are estimates, not measured end-to-end VRAM or
throughput. A synthetic 512 x 512 CPU forward/backward passed in 7.75 seconds with
maximum process RSS 1,254,668 KiB (about 1.20 GiB). This is a single
measurement, not a throughput forecast. Full train/validation image-hash
preflight passed for APTOS2019 (2,930 train; 366 validation) and DDR
(10,018 train; 1,252 validation). The 512 x 512 GPU smoke remains pending:
both shared RTX 3060 devices had only about 1.7 GiB free at inspection,
so a run would have risked disrupting other users. The prior 64 x 64
synthetic GPU forward/backward passed, but does not establish 512 x 512
VRAM feasibility. The same validation split selected the starting
checkpoints, so pilot validation comparisons are exploratory rather
than independent confirmation.

The pilot baseline arm instantiates the original CADViTCandidate class,
not the novelty subclass. The other arms use NovelCADViT. After the
baseline-control change, 53 selected baseline and Phase 1/2 synthetic
tests passed. The report-integrity test module was deliberately not run
because it reads frozen final-test result artifacts; no final-test
evaluation or image access occurred here.

At the verification snapshot, the shared /mnt/home filesystem had
22 GiB free and was 99% utilized. Two FP32 model-plus-AdamW checkpoints
per pilot arm are estimated near 125-150 MiB total, plus small JSON
histories. Nine primary arms across two datasets could consume roughly
2.3-2.7 GiB; this is an estimate, not measured storage. Start with one
reviewed arm per dataset and inspect actual free space before any
pilot. No pilot run directory has been created by this phase.

## Phase 3B — deterministic V2 BG10 input cache (new protocol)

This section supersedes the *uncached runner instructions* above for new cached
pilots. The earlier incomplete APTOS pilot at
`~/Kel_13/cadvit_novelty_pilots/APTOS2019/p2_p3_probe_seed42_e2_mb1_eff16_fp32`
is preserved and must not be resumed. Cached runs use
`~/Kel_13/cadvit_novelty_pilots_cached/` and a separate external cache root
`~/Kel_13/cadvit_v2_cache/`. The runner refuses the old output root when
`--cache-mode on`.

The cache stores exactly the deterministic `preprocess_fundus_v2` result:
lossless C-contiguous `uint8` RGB [512,512,3] in uncompressed `.npy`.
No crop, blur, Ben Graham sigma, EXIF/RGB handling, normalization, or tensor
conversion is replaced. `V2FundusSplit` applies the unchanged stochastic
horizontal flip, vertical flip, and affine transform **after** cache retrieval
for train, then CHW float32 / 255. Validation has no augmentation.
`--cache-mode off` takes the original direct path.

Each namespace hashes the frozen manifest SHA, V2 configs, preprocessing and
cache implementation source, Pillow/OpenCV/NumPy versions, dataset, dtype,
shape, and format. An entry key also hashes the source-image SHA from the
verified train/validation manifest. The entry metadata records source SHA,
protocol ID, `.npy` SHA, pixel SHA, dtype, and shape. Reads validate all of
these. Missing or corrupted entries are rebuilt from a source file only after
its SHA matches the frozen manifest. A stable namespace-level Linux
`fcntl.flock` serializes writers and is released by the kernel if a process
exits; its small lock file may remain harmlessly. Data and metadata are written
with fsync and atomic replace. A torn pair is rejected and rebuilt under the
next lock; an orphan temporary file from a hard crash is not silently deleted.
No bulk array is held in RAM: entries are loaded one at a time, and DataLoader
workers remain at the original zero setting. Existing B1/foundation or reproduction
caches are neither read nor modified.

The cache identity/root and full protocol enter `config.json` and saved
checkpoint records. The runner source hash now also includes
`novelty/cadvit_input_cache.py`. Resume requires identical configuration,
code hashes, library versions, and cache identity; it cannot silently resume
the old uncached run. Preflight still checks every train/validation source
image and the pinned CAD-ViT checkpoint before cache use. No test row is
instantiated.

**Storage policy (corrected in Phase 3C).** A 786,432-byte pixel array
produces a 786,560-byte NPY file. On KC, that file allocated 790,528 bytes
and its 507-byte JSON allocated another 4,096 bytes. The old 790,528-byte
entry budget omitted the JSON block. The corrected conservative budget is
798,720 bytes per unique image, including allowance for directory/lock
overhead. At row-count worst case, APTOS needs **2.452 GiB**, DDR
**8.383 GiB**, together **10.835 GiB**. Actual frozen train/validation
manifests have 3,183 and 11,178 unique image hashes, giving conservative
**2.368 GiB** and **8.315 GiB** respectively (**10.683 GiB** combined).
An additional entry budget covers an atomic temporary write; a single
namespace lock file is shared by its entries. Existing orphan temporary files
and concurrent users consume real free space and are not assumed absent.
The **8 GiB free** reserve is an engineering policy, not a measured training
minimum. Warm-up checks before the build; every new/repair entry checks again.
A populated cache is not reserved twice. Recheck shared-disk free space
before any full build.

**Historical Phase 3B six-image APTOS subset, CPU only (3 first validation + 3 first
training rows, fixed manifest order):** direct uncached preprocessing
**1.397 s**, first cache fill including source hashing and writes **1.700 s**,
warm reads including entry integrity checks **0.00655 s**. All six cached
arrays matched direct arrays exactly. Six `.npy` files occupied
**4,719,360 bytes**. The subset IDs were validation
`0083ee8054ee`, `00a8624548a9`, `012a242ac6ff`; training
`000c1434d8d7`, `001639a390f0`, `0024cdab0c1e`.
The small, warm OS page-cache measurement is not a full-epoch throughput
forecast. Six then-current-source APTOS entries were built under namespace
`b1aa4c114a5ed26171675c52cbdec392cfb3dbad49af28e2c78831d90d715601`.
The two earlier draft-source benchmark namespaces
`7f5e969cf1f0af7c33fa08f5a9293c197cd9b47ff1dada53694eb612e9e15687`
and `aaa6a1974d2d79fff2aed4e9b06ccb0670af6e1eb04be64af9b11b27ed7a75a0`
remain intact. There are 18 small APTOS entries across three namespaces
(14,158,080 bytes of NPY files); no DDR entries were built or deleted.

**Manual KC commands, not executed here.** All commands run from the
`preof-kc` repository root. `cache-warmup` requires an explicit positive
`--warm-limit` (maximum 12,000), checks disk before writing, and reports
progress and rechecks the 8 GiB reserve before each entry write. `both` orders validation first, then training. The full APTOS
limit is 366+2,930=3,296; the full DDR limit would be 1,252+10,018=11,270.
Only run full warm-up after separate approval and fresh disk review.

```bash
cd /mnt/home/probation-03/Kel_13/Diabetic-Rethinography-Novelty
ROOT=/mnt/home/probation-03/Kel_13
OUT="$ROOT/cadvit_novelty_pilots_cached"
CACHE="$ROOT/cadvit_v2_cache"
../.venv/bin/python -m novelty.cadvit_pilot preflight --artifact-root "$ROOT" --output-root "$OUT" --cache-mode on --cache-root "$CACHE" --dataset APTOS2019 --ablation p2_p3_probe --micro-batch 1 --epochs 2
# Future full APTOS warm-up after approval; not run in Phase 3B:
../.venv/bin/python -m novelty.cadvit_pilot cache-warmup --artifact-root "$ROOT" --output-root "$OUT" --cache-mode on --cache-root "$CACHE" --warm-split both --warm-limit 3296 --dataset APTOS2019 --ablation p2_p3_probe --micro-batch 1 --epochs 2
read -r -p "Approved free GPU index (0 or 1): " GPU
CUDA_VISIBLE_DEVICES="$GPU" ../.venv/bin/python -m novelty.cadvit_pilot smoke --artifact-root "$ROOT" --output-root "$OUT" --cache-mode on --cache-root "$CACHE" --dataset APTOS2019 --ablation p2_p3_probe --micro-batch 1 --epochs 2 --device cuda
# Only after a 512x512 GPU smoke and scientific approval:
CUDA_VISIBLE_DEVICES="$GPU" ../.venv/bin/python -m novelty.cadvit_pilot pilot --artifact-root "$ROOT" --output-root "$OUT" --cache-mode on --cache-root "$CACHE" --dataset APTOS2019 --ablation p2_p3_probe --micro-batch 1 --epochs 2 --device cuda
```

The DDR interface is identical except for its own manifest/checkpoint/output
namespace. **Do not execute DDR full warm-up yet.**

```bash
cd /mnt/home/probation-03/Kel_13/Diabetic-Rethinography-Novelty
ROOT=/mnt/home/probation-03/Kel_13
OUT="$ROOT/cadvit_novelty_pilots_cached"
CACHE="$ROOT/cadvit_v2_cache"
../.venv/bin/python -m novelty.cadvit_pilot preflight --artifact-root "$ROOT" --output-root "$OUT" --cache-mode on --cache-root "$CACHE" --dataset DDR --ablation p2_p3_probe --micro-batch 1 --epochs 2
# Future DDR warm-up only after separate approval:
../.venv/bin/python -m novelty.cadvit_pilot cache-warmup --artifact-root "$ROOT" --output-root "$OUT" --cache-mode on --cache-root "$CACHE" --warm-split both --warm-limit 11270 --dataset DDR --ablation p2_p3_probe --micro-batch 1 --epochs 2
read -r -p "Approved free GPU index (0 or 1): " GPU
CUDA_VISIBLE_DEVICES="$GPU" ../.venv/bin/python -m novelty.cadvit_pilot smoke --artifact-root "$ROOT" --output-root "$OUT" --cache-mode on --cache-root "$CACHE" --dataset DDR --ablation p2_p3_probe --micro-batch 1 --epochs 2 --device cuda
# Only after a 512x512 GPU smoke and scientific approval:
CUDA_VISIBLE_DEVICES="$GPU" ../.venv/bin/python -m novelty.cadvit_pilot pilot --artifact-root "$ROOT" --output-root "$OUT" --cache-mode on --cache-root "$CACHE" --dataset DDR --ablation p2_p3_probe --micro-batch 1 --epochs 2 --device cuda
```

For a compatible interrupted *cached* run, replace `pilot` with `resume`
and keep every CLI option identical. Do not resume the uncached Phase 2 run.
No GPU smoke, training, full APTOS cache, or DDR cache build was executed
during Phase 3B.

## Phase 3C — independent CPU audit and corrections

Relative to baseline commit `e280340`, only the novelty cache module, pilot
runner, two CAD-ViT test files, this protocol, and `CAD_ViT_Novelty.ipynb`
are involved. Reproduction, foundation, manifests, checkpoints, and completed
pilot outputs are unchanged. Three cache defects were reproduced and fixed:
a valid JSON list caused an uncaught `AttributeError` instead of repair; an
abandoned exclusive lock file blocked repair after a writer crash; and the
entry disk budget omitted a JSON filesystem block. Regression tests now cover
non-object metadata, crash-released writer locking/torn publication, and
actual allocated file blocks. CPU preflight also stopped using
`torch.cuda.get_device_name`; a read-only `nvidia-smi` inventory records
GPU names without initializing CUDA.

Final-source, isolated temporary-cache audit selected the first train and
validation image in each of grades 0–4 **separately for APTOS and DDR**.
Each image's canonical path, frozen-manifest source hash, label, split,
direct V2 RGB array, cached array, and normalized float32 tensor matched.
Identical torch RNG states produced identical train tensors. Two epoch
seeds produced different augmented tensors while cached pixels remained
unchanged. Validation repeated exactly. Cache-hit processing was patched
to fail if `preprocess_fundus_v2` ran and passed without invoking it.
Missing, stale, corrupted, and torn entries rebuilt; concurrent readers
caused one preprocessing write. The temporary test caches were bounded to
10 entries per dataset and removed after audit. No final-test row was used.

| Dataset | First-per-grade train+val images | Direct V2 | Cold fill | Warm read | Exact arrays/tensors |
| --- | ---: | ---: | ---: | ---: | --- |
| APTOS2019 | 10 | 2.065 s | 2.525 s | 0.01234 s | PASS |
| DDR | 10 | 0.722 s | 1.239 s | 0.01242 s | PASS |

These are measured serial CPU timings on 10 fixed images per dataset. Cold
fill was measured after direct decoding, so the OS may already have cached
source-file pages. Warm read includes entry SHA checks and may benefit from
the OS page cache. They are not full-epoch throughput estimates. Full
manifest/source-image hashing and strict archived-checkpoint loading happen
once per CLI invocation, not once per epoch. Cache hits read and hash one
NPY per image per epoch; deterministic Gaussian blur runs only on misses.
Augmentation and tensor conversion still run per training draw.

At the Phase 3C audit snapshot, `/mnt/home` had approximately **13.94 GiB**
free and was 99% utilized. APTOS alone fits the conservative unique-image
estimate plus 8 GiB reserve at that instant, but shared capacity fell by
about 8 GiB during the audit. DDR alone needs about **16.315 GiB** including
the reserve and is **blocked** at that snapshot; both together need about
**18.683 GiB** plus transient/shared-use allowance. Full warm-up remains
unapproved and must not be started from this audit. The existing 18 small
APTOS benchmark entries under three older source-hash namespaces remain
untouched; the repaired final source uses new cache identities.


## Phase 3C cross-filesystem cache audit (conditional path)

The proposed cache root `/var/tmp/probation-03/cadvit_v2_cache` is on
`/dev/nvme0n1p2`, separate from the artifacts and cached-run output on
`/mnt/home`. At the read-only audit snapshot, the cache filesystem had
171.885 GiB free; `/mnt/home` had 13.887 GiB free. The cache guard checks
the nearest existing ancestor of the entry directory, so an absent cache
root uses the filesystem of its actual writable parent. It rejects an
unwritable or non-directory ancestor before a warm-up or pilot begins.
Data, JSON metadata, the namespace lock, and atomic temporary files are
all constructed beneath the cache root. Entry paths that escape through a
namespace symlink or cross to a nested mount are rejected. Training checkpoints and logs
remain under `OUT` on `/mnt/home`, which needs its own capacity review.

The corrected per-entry budget and 8 GiB cache-filesystem reserve remain
unchanged: 3,183 unique APTOS images require approximately 2.368 GiB,
11,178 unique DDR images approximately 8.316 GiB, and both approximately
10.684 GiB plus one 8 GiB reserve. These estimates include one temporary
entry per build and conservative metadata/directory allowance. Shared
usage and orphan files can consume additional capacity. No full cache was
built in this audit.

**Only after lab approval for this location**, use the Phase 3B command
blocks above with this one replacement:

```bash
CACHE=/var/tmp/probation-03/cadvit_v2_cache
```

Keep `ROOT` and `OUT` unchanged. Earlier command blocks retain the
historical `$ROOT/cadvit_v2_cache` value and should not be used for
cross-filesystem warm-up without this override. Confirm ownership,
retention/cleanup policy, and fresh free space before any build.
