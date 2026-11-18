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
