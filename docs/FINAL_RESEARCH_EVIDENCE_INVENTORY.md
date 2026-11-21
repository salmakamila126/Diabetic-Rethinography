# Inventaris evidence penelitian akhir

Inventaris ini membekukan bukti untuk penulisan laporan magang. Titik awal Git yang diperiksa ialah `a522a0c2dbb526a7ba5c5850dde1773736fa5af5`. Bukti baru berasal dari profil dan receipt E5 APTOS2019 yang dibaca melalui JupyterLab HPC pada 21 November 2026. [Indeks notebook](../notebooks/README.md) · [Inventaris figur dan tabel](FINAL_RESEARCH_FIGURE_INVENTORY.md)

## Cara membaca status

| Status | Arti |
| --- | --- |
| `COMPLETED_AND_VERIFIED` | Artefak agregat yang tersedia mencakup tahap yang dinyatakan dan lulus pemeriksaan angka serta provenance yang bisa dilakukan. Verifikasi binari checkpoint hanya berlaku jika dicatat oleh sumber. |
| `PARTIAL_WITH_VERIFIED_EVIDENCE` | Sebagian keluaran asli bisa diperiksa dan cakupan tahap penuh belum terbukti. |
| `IMPLEMENTED_NOT_EXECUTED` | Kode atau protokol tersedia dan belum ada laporan eksekusi untuk konfigurasi yang disebutkan. |
| `IN_PROGRESS` | Status proses aktif memerlukan observasi langsung. Snapshot recovery hanya memverifikasi berkas yang tersimpan. |
| `NOT_VERIFIED` | Bukti pelaksanaan tidak cukup atau berada di luar cakupan audit ini. |
| `NOT_APPLICABLE` | Arm tidak berlaku pada dataset atau desain yang disebutkan. |
| `NOT_MEASURED` | Kampanye yang ditinjau tidak menjalankan pengukuran tersebut. |

## REPRODUCE — CAD-ViT

Wang, Gou, dan Li (2025), IEEE Access, mendeskripsikan grading lima kelas. Bagian III dan Gambar 1–3 mendasari crop → resize 512×512 → Gaussian blur, feature fusion dengan dilatasi 3/5/7/9 dan Coordinate Attention. Tabel 1 melaporkan 7,2629 juta parameter pada rasio reduksi 16. Kandidat lokal mengikuti alur IRB/IRB+A/EMB pada tingkat diagram dan mencatat 5.282.101 parameter. Detail kanal, stride, ukuran patch, depth/head Transformer dan numerik crop/blur merupakan keputusan implementasi. V2 memakai Ben Graham BG10 sebagai ekstensi. Statistik deteksi biner di paper memakai tugas berbeda dari grading lima kelas di tabel hasil penelitian ini. [Paper asli](https://doi.org/10.1109/ACCESS.2025.3572455) · [Audit arsitektur](ARCHITECTURE_FIDELITY_AUDIT.md) · [Matriks preprocessing](PAPER_FIDELITY_MATRIX.md)

[Notebook reproduksi](../notebooks/reproduction/CAD_ViT_Reproduction.ipynb) mengindeks [arsip 105 sel](../reproduction/archive/CAD_ViT_original_105_cells.ipynb), [model kandidat](../reproduction/cadvit/model.py), [preprocessing](../reproduction/cadvit/preprocessing.py) dan [runner modular](../reproduction/cadvit/experiments.py). Runner modular mendokumentasikan kampanye balanced/recovery. V1/V2 memiliki implementasi dan keluaran historis tersendiri dalam arsip. [Laporan V1](../reproduction/experiments/01_experiment_initial_cadvit_v1.ipynb) · [Laporan V2](../reproduction/experiments/02_experiment_bg10_cadvit_v2.ipynb) · [Laporan balanced/recovery](../reproduction/experiments/03_experiment_balanced_training_and_ddr_recovery.ipynb)

| Kampanye | Dataset | Training dan checkpoint | Final test n | Accuracy | Macro-F1 | Evidence dan status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| V1 | APTOS2019 | Epoch 8 terpilih. Log lengkap tidak tersedia. | 366 | 0,724044 | 0,554058 | [Confusion tersimpan](../reproduction/report_figures/historical_v1_aptos_confusion_matrix.png) · `PARTIAL_WITH_VERIFIED_EVIDENCE` |
| V1 | DDR | Epoch 5 menurut laporan historis. Segmen resume mencatat NaN. | 1.252 | 0,431310 | 0,353457 | [Matriks dalam arsip §6](../reproduction/archive/CAD_ViT_original_105_cells.ipynb) · `PARTIAL_WITH_VERIFIED_EVIDENCE` |
| V2 awal | APTOS2019 | 65 epoch menurut catatan historis. | — | — | — | [Validasi historis dalam arsip](../reproduction/experiments/02_experiment_bg10_cadvit_v2.ipynb) · `PARTIAL_WITH_VERIFIED_EVIDENCE` |
| V2 awal | DDR | 30 epoch menurut catatan historis. | — | — | — | [Validasi historis dalam arsip](../reproduction/experiments/02_experiment_bg10_cadvit_v2.ipynb) · `PARTIAL_WITH_VERIFIED_EVIDENCE` |
| V2 BG10 | APTOS2019 | 80 epoch, best epoch 69 dengan Macro-F1 validasi 0,657127. | 366 | 0,789617 | 0,604952 | [Final-test JSON](../results/APTOS2019/v2_bg10_historical_80epochs/final_test_v2_80ep.json) · `COMPLETED_AND_VERIFIED` untuk agregat |
| V2 BG10 | DDR | 80 epoch protokol campuran, best epoch 72 dengan Macro-F1 validasi 0,690323. | 1.252 | 0,804313 | 0,620287 | [Final-test JSON](../results/DDR/v2_bg10_historical_80epochs/final_test_v2_80ep.json) · `COMPLETED_AND_VERIFIED` untuk agregat |
| Balanced oversampling | APTOS2019 | 80 epoch AMP FP16, best epoch 24 dengan Macro-F1 validasi 0,661421. | 366 | 0,732240 | 0,531529 | [History](../results/APTOS2019/full_oversampling_amp_fp16_80epochs/history.jsonl) · [Final-test JSON](../results/APTOS2019/full_oversampling_amp_fp16_80epochs/final_test_summary.json) · `COMPLETED_AND_VERIFIED` |
| Balanced oversampling dan recovery | DDR | 80 epoch, best epoch 69 dengan Macro-F1 validasi 0,653069. Protokol berubah setelah epoch 42. | 1.252 | 0,778754 | 0,631981 | [History](../results/DDR/full_oversampling_fp32_math_mb8_80epochs/history.jsonl) · [Final-test JSON](../results/DDR/full_oversampling_fp32_math_mb8_80epochs/final_test_summary.json) · `COMPLETED_AND_VERIFIED` |

Dua hash V2 dari catatan transfer mengikat byte JSON final test (Git `98828dd`); lineage transfer checkpoint DDR dan kode penulis tetap belum lengkap. Balanced APTOS mengikat epoch 24 dan `model_sha256` dalam JSON (Git `76c451f`). DDR recovery mengikat epoch 69 dan `checkpoint_sha256` dalam JSON (Git `ad9eea6`), plus hash manifest dan evaluator. Hash evaluator mendokumentasikan source terkait run. Audit itu tidak membuktikan secara independen bahwa source dieksekusi. [Provenance V2](V2_FINAL_TEST_PROVENANCE.md) · [Evaluator dan hash DDR](../reproduction/provenance/ddr_final_test/README.md)

## RETHINK — ConvNeXt-Tiny B0–B4

Semua baris di bawah adalah grading lima kelas dengan Hugging Face ConvNeXt V1 Tiny, full fine-tuning 20 epoch, seed 42, micro/effective batch 32, BF16, AdamW, backbone LR `1e-5`, head LR `1e-3`, weight decay 0,01. Setiap run memilih checkpoint pada validasi. Bukti agregat memuat 200 baris epoch dan `source_sha256.best.pt` untuk setiap run. Source asal dirujuk terpisah dari hash data dan checkpoint. [Evidence dan identitas checkpoint](../novelty/convnext_b0_b4/validation_evidence.json) · [Konfigurasi sumber B0–B2](../novelty/foundation_adaptation.py) · [Source eksternal B3/B4 dan kontrol](NOTEBOOK_EXPERIMENT_AUDIT.md#b0b4-mapping)

| Arm | Perubahan perlakuan | APTOS2019, validasi n=366 | DDR, validasi n=1.252 | Status |
| --- | --- | --- | --- | --- |
| B0 | CE biasa, tanpa augmentasi | Best 17, Macro-F1 0,673686 · [Notebook](../notebooks/baselines/B0_APTOS2019.ipynb) | Best 8, 0,619817 · [Notebook](../notebooks/baselines/B0_DDR.ipynb) | `COMPLETED_AND_VERIFIED` |
| B1 | Augmentasi konservatif train-only | Best 16, 0,701124 · [Notebook](../notebooks/baselines/B1_APTOS2019.ipynb) | Best 11, 0,652146 · [Notebook](../notebooks/baselines/B1_DDR.ipynb) | `COMPLETED_AND_VERIFIED` |
| B2 | CE berbobot inverse-square-root pada train | Best 7, 0,659987 · [Notebook](../notebooks/baselines/B2_APTOS2019.ipynb) | Best 5, 0,623084 · [Notebook](../notebooks/baselines/B2_DDR.ipynb) | `COMPLETED_AND_VERIFIED` |
| B3 | B1 + oversampling grade 3 | Best 20, 0,700624 · [Notebook](../notebooks/baselines/B3_APTOS2019.ipynb) | Best 11, 0,651006 · [Notebook](../notebooks/baselines/B3_DDR.ipynb) | `COMPLETED_AND_VERIFIED` untuk agregat |
| B4-Q | Karantina konflik label pada grup SHA identik | Best 11, 0,688319 · [Notebook](../notebooks/baselines/B4_APTOS2019.ipynb) | [Tidak berlaku](../notebooks/baselines/B4_DDR.ipynb) | APTOS `COMPLETED_AND_VERIFIED`; DDR `NOT_APPLICABLE` |
| B4-R | Kontrol penghapusan grup SHA nonkonflik dengan jumlah per grade yang sama | Best 9, 0,707897 · [Notebook](../notebooks/baselines/B4_APTOS2019.ipynb) | [Tidak berlaku](../notebooks/baselines/B4_DDR.ipynb) | APTOS `COMPLETED_AND_VERIFIED`; DDR `NOT_APPLICABLE` |

APTOS2019 memakai 2.930 baris training kecuali B4-Q/R yang mempertahankan 2.880 dan membiarkan 366 validasi utuh. DDR memakai 10.018 train dan 1.252 val. B3/B4 mempunyai receipt dan hasil agregat yang dipublikasikan, sementara source eksekusinya berada di luar Git. B1 memberi Macro-F1 lebih tinggi daripada B0 pada kedua run yang diamati. Efek itu deskriptif. B4-Q berada di bawah kontrol B4-R pada validasi APTOS. Git bukti `01334c3`, dengan basis provenance `f042551`. [Figur dan metrik per grade](FINAL_RESEARCH_FIGURE_INVENTORY.md#convnext-foundation)

## REINVENT — E5, E6 dan E7

E5 memakai ConvNeXt V1 224 px, categorical CE dengan delayed weighting berbasis train mulai epoch 11 dan ordinal BCE berbobot 0,25. E6 mempertahankan input tunggal dan memakai LDAM, delayed effective-number weighting, ordinal BCE 0,25, dan penalti batas grade 0↔1 serta 1↔2 berbobot 0,10. E7 memakai checkpoint E5 sebagai inisialisasi dan mempertahankan backbone berbagi bobot, satu view global 384 px, dua view lokal 224 px, gating, objective E5, dan pemilihan berdasarkan bobot sesuai protokol run. Ini konfigurasi berbeda, sehingga peringkat antar-arm tidak mengisolasi efek satu komponen. [E5 source](../novelty_integrated/e5/) · [E6 source](../novelty_integrated/e6/) · [E7 source](../novelty_integrated/e7/) · [Perbandingan E5/E6](E5_E6_VALIDATION.md)

| Run | Epoch selesai / rencana | Best epoch | Validasi n | Accuracy | Macro-F1 | QWK | Evidence, checkpoint dan status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E5 APTOS2019 | 20/20 | 16 | 366 | 0,830601 | 0,705791 | 0,899241 | [JSON](../results/APTOS2019/e5_integrated/validation_evidence.json) · [Notebook](../notebooks/novelty/E5_APTOS2019.ipynb) · `c44a88429f07` · `COMPLETED_AND_VERIFIED` |
| E5 DDR | 20/20 | 18 | 1.252 | 0,823482 | 0,653199 | 0,857162 | [JSON](../results/DDR/e5_integrated/validation_evidence.json) · [Notebook](../notebooks/novelty/E5_DDR.ipynb) · `7dd7488013b7` · `COMPLETED_AND_VERIFIED` |
| E6 APTOS2019 | 20/20 | 15 | 366 | 0,789617 | 0,679204 | 0,880384 | [JSON](../results/APTOS2019/e6_ldam_ordinal_adjacent/validation_evidence.json) · [Notebook](../notebooks/novelty/E6_APTOS2019.ipynb) · `f03acdc4fc03` · `COMPLETED_AND_VERIFIED` |
| E6 DDR | 20/20 | 12 | 1.252 | 0,797923 | 0,662455 | 0,849553 | [JSON](../results/DDR/e6_ldam_ordinal_adjacent/validation_evidence.json) · [Notebook](../notebooks/novelty/E6_DDR.ipynb) · `7e4bf74ced0d` · `COMPLETED_AND_VERIFIED` |
| E7 APTOS2019 | 20/20 | 19 | 366 | 0,836066 | 0,696492 | 0,912687 | [JSON](../results/APTOS2019/e7_global_local/validation_evidence.json) · [Notebook](../notebooks/novelty/E7_APTOS2019.ipynb) · `5ef9816b9c75` · `COMPLETED_AND_VERIFIED` |
| E7 DDR | **10/20** | 2 | 1.252 | 0,833866 | 0,668563 | 0,852151 | [JSON](../results/DDR/e7_global_local/validation_evidence.json) · [Notebook](../notebooks/novelty/E7_DDR.ipynb) · `ecfed1989c4f` · `PARTIAL_WITH_VERIFIED_EVIDENCE` |

Identitas lengkap checkpoint berada di `checkpoint_sidecars`. Manifest beku train/val APTOS2019 `73d1546e3c2fd2bf9e3d0fd10af914cb523d28e4bc8e1f25212af454d4be8854`; DDR `7bc3e3916e012bc46fcd9cd69aa469e0daf3229e1273f1d8ecb6054899bf6eb1`. JSON memuat konfigurasi optimizer, seed, cache, preprocessing dan versi sumber masing-masing. Git implementasi E5 `4c9ab9d`, E6 `b3dd3ca`, E7 `b7c42b6`. Git publikasi validasi E5/E6 `027e586`, E7 APTOS `0852965`, E7 DDR parsial `6617516`. [Figur dan tabel E5–E7](FINAL_RESEARCH_FIGURE_INVENTORY.md#convnext-e5e7)

**Final test E5 APTOS2019** adalah hasil historis terpisah, n=366, checkpoint epoch 16, accuracy 0,825137, Macro-F1 0,640456, QWK 0,891194. Ringkasan training merekam status saat training selesai. JSON final test mencatat evaluasi yang dilakukan kemudian. Bukti ini tidak menyediakan final test ConvNeXt lain. [Agregat historis E5](../results/APTOS2019/e5_integrated/test_summary.json)

Arm kontrol/ablation E5 selain integrated tetap `IMPLEMENTED_NOT_EXECUTED` dalam inventory publik ini. Prototipe CAD-ViT P1–P4 dan eksplorasi PReOF E0–E3 dipisahkan dari hasil ConvNeXt. [Protokol CAD-ViT P1–P4](CADVIT_NOVELTY_PROTOCOL.md) · [PReOF](PReOF_PHASE1.md)

## Optimasi — varian turunan checkpoint

| Tahap | Cakupan | Status dan batas | Evidence |
| --- | --- | --- | --- |
| Pilot sebelum recovery | E5/E6/E7 APTOS2019 serta E5/E6 DDR. Masing-masing 100 frozen val, 0/10/30/50% dan empat backend. | `COMPLETED_AND_VERIFIED` untuk agregat. Semua 60 perbandingan non-eager lulus. Semua 50% memprediksi grade 0 pada seluruh sampel. | [Protokol, hasil dan notebook](PRUNING_PILOT_RESULTS.md) · [Agregat asli, Git a522a0c](../results/optimization/pilot_100_v1/validation_evidence.json) |
| Kompleksitas | 20 prepared profiles, backbone HF ConvNeXt V1. | `COMPLETED_AND_VERIFIED` untuk receipt. MACs hanya Conv2d/Linear. Tiga view E7 berbagi bobot. | [Complexity CSV](../results/optimization/pilot_100_v1/complexity.csv) · [Verifier](pruning_pilot_supplement.py) · [Generator](pruning_pilot_publication.py) |
| E5 APTOS2019 recovery | 0%, 10%, 30%, 50% dalam protokol tiga epoch. | 10% `PARTIAL_WITH_VERIFIED_EVIDENCE` (sidecar epoch 1). Tiga arm lain `NOT_VERIFIED`. Nol arm dengan hasil pascarecovery terverifikasi. | [Audit recovery](PRUNING_PILOT_RESULTS.md#e5-aptos2019-recovery-audit) · [Receipt](../results/optimization/pilot_100_v1/e5_aptos2019_recovery_audit.json) |
| Latency dan throughput CPU | Shared-CPU FP32 pilot hanya menjalankan validasi. | `NOT_MEASURED`. MACs dan ukuran file tidak menentukan latency. | [Alokasi, protokol dan parity](../results/optimization/pilot_100_v1/validation_evidence.json) |
| Baseline optimasi mitra | ConvNeXt V2 timm dan eksperimen HF V1 historis. | Referensi metode pada branch terpisah. Hasil pilot lima job berasal dari implementasi HF V1 yang tercatat di receipt. | [Audit kompatibilitas](OPTIMIZATION_COMPATIBILITY.md) |

## Kesiapan sembilan bagian laporan

| Bagian | Bukti yang dapat dipakai | Batas yang perlu ditulis |
| --- | --- | --- |
| Latar Belakang | [Paper CAD-ViT](https://doi.org/10.1109/ACCESS.2025.3572455) dan [audit fidelity](ARCHITECTURE_FIDELITY_AUDIT.md). | Bedakan deteksi biner, grading lima kelas, dan sasaran optimasi CPU. |
| Tujuan | [Protokol eksperimen](NOTEBOOK_EXPERIMENT_AUDIT.md) dan [pilot](PRUNING_PILOT_RESULTS.md). | Hasil menguji hipotesis eksploratif dan kemampuan menjaga kualitas. |
| Dataset | [Preprocessing](PAPER_FIDELITY_MATRIX.md), [frozen split](#reinvent--e5-e6-dan-e7), dan grade support pilot. | Manifest, citra privat, dan checkpoint biner berada di luar Git. Val dipakai untuk seleksi checkpoint. |
| Metodologi & Eksperimen | [Model kandidat](../reproduction/cadvit/model.py), [indeks notebook](../notebooks/README.md), [metode optimasi](OPTIMIZATION_COMPATIBILITY.md). | Laporkan asumsi paper, perubahan protokol DDR, inisialisasi E7, dan jenis backend. |
| Hasil & Evaluasi | [Inventaris figur/tabel](FINAL_RESEARCH_FIGURE_INVENTORY.md) dan [pilot](PRUNING_PILOT_RESULTS.md). | Pisahkan full validation, subset 100 gambar, dan final test historis. |
| Diskusi | Perbedaan arsitektur, ketidakseimbangan grade, hilangnya kualitas setelah pruning, dan batas parity. | Hindari inferensi kausal klinis, klaim novelty, dan generalisasi dari sampel kecil. |
| Kesimpulan | Reproduksi parsial terdokumentasi, screening selesai, dan pre-recovery trade-off terverifikasi. | Recovery dan percepatan CPU belum terbukti. E7 DDR berhenti setelah 10 epoch. |
| Future Works | [Kebutuhan evidence lanjutan](PRUNING_PILOT_RESULTS.md#completion-status). | Rencana recovery, pengukuran terisolasi, dan konfirmasi independen tidak boleh dilaporkan sebagai hasil. |
| Output | GitHub, notebook statis, receipt, CSV, generator dan figur versi PNG/SVG. | Caption menyebut dataset, split, ukuran sampel, tahap, dan sumber bukti. |
