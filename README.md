# Reproduksi CAD-ViT: Coordinate Attention-Enhanced Vision Transformer With Dilated Feature Fusion for Diabetic Retinopathy Staging

Reproduksi penelitian [CAD-ViT (2025)](https://ieeexplore.ieee.org/document/11009024) dalam satu notebook utama, **CAD_ViT.ipynb**. Notebook menyatukan EDA, preprocessing, pengembangan model, training, dan evaluasi sebagai satu workflow penelitian tim. Output EDA historis pada §2.2 berasal dari eksekusi yang belum dapat ditautkan secara pasti ke dataset resmi dan manifest training V2; grafik APTOS2019 lokal pada §2.3 memakai pasangan label–gambar yang diverifikasi secara terpisah. Interpretasikan kedua kelompok output sesuai provenance masing-masing.

## Sumber dataset

| Dataset | Sumber resmi | Lokasi di bawah CAD_VIT_DATA_ROOT |
| --- | --- | --- |
| APTOS 2019 | [Kompetisi Kaggle APTOS 2019](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) | APTOS2019/ |
| OIA-DDR | [Repositori resmi OIA-DDR](https://github.com/nkicsl/DDR-dataset) | DDR/ |

Unduh dataset secara manual dari sumber resminya dan simpan **di luar repository**. Akses APTOS mungkin memerlukan login dan persetujuan aturan kompetisi. Notebook mencari tepat satu <code>train.csv</code> dan satu direktori <code>train_images</code> di dalam <code>APTOS2019/</code>, lalu memeriksa pasangan <code>id_code</code>–gambar dan <code>diagnosis</code> kelas 0–4. Gambar test kompetisi tidak dipakai untuk data berlabel. Jika berkas masih berupa arsip, ekstrak terlebih dahulu; notebook tidak mengunduh atau mengekstraknya.

Pada server yang diaudit, anotasi DDR berada di <code>DDR/DDR-dataset/DR_grading/train.txt</code>, <code>valid.txt</code>, dan <code>test.txt</code>; setiap baris berisi <code>filename.jpg label</code>. Notebook mencari tepat satu direktori <code>DR_grading</code> dengan ketiga anotasi tersebut dan mencocokkan nama gambar secara unik. Dari 13.673 anotasi, 12.522 label 0–4 dipakai untuk grading lima kelas (§IV.B paper); 1.151 label 5 (ungradable) tetap ada pada dataset mentah dan tidak masuk manifest eksperimen.

## Konfigurasi path pada Colab dan server Linux

Notebook membaca <code>CAD_VIT_DATA_ROOT</code> sebagai direktori yang memuat <code>APTOS2019/</code> dan <code>DDR/</code>. Variabel opsional <code>CAD_VIT_OUTPUT_ROOT</code> menunjuk lokasi keluaran EDA; jika kosong, notebook memakai <code>eda_results/</code> di direktori induk data root. Keduanya harus menunjuk ke lokasi yang dapat diakses dan, untuk keluaran, dapat ditulis oleh runtime.

Contoh Colab, jalankan sebelum sel konfigurasi:

~~~python
import os
os.environ["CAD_VIT_MOUNT_DRIVE"] = "1"
os.environ["CAD_VIT_DATA_ROOT"] = "/content/drive/MyDrive/CAD_ViT/datasets"
~~~

Contoh shell server GPU KC, sesuaikan mount dataset yang sebenarnya:

~~~bash
export CAD_VIT_DATA_ROOT=/path/to/CAD_ViT/datasets
export CAD_VIT_OUTPUT_ROOT=/path/to/CAD_ViT/eda_results
~~~

<code>CAD_VIT_MOUNT_DRIVE=1</code> hanya digunakan pada Colab untuk memasang Drive; server tidak memerlukan <code>google.colab</code>. Jangan menyimpan password SSH, token Kaggle, kunci W&B, atau kredensial lain di notebook maupun repository.

## Persiapan OIA-DDR: jalankan sekali di luar notebook training

Repositori resmi DDR menyatakan bahwa arsip terdiri atas sepuluh bagian yang perlu digabung berurutan sebelum ekstraksi. Pastikan <code>DDR-dataset.zip.001</code> sampai <code>.010</code> ada dan tidak kosong. Periksa kapasitas untuk ZIP gabungan **dan** hasil ekstraksi sebelum mulai. Jika kapasitas tidak cukup, gunakan volume lain; jangan membuat salinan tambahan dari sepuluh bagian tersebut.

Periksa bagian dan kapasitas lebih dulu:

~~~bash
set -euo pipefail
cd "$CAD_VIT_DATA_ROOT/DDR"
for part in DDR-dataset.zip.{001..010}; do
    test -s "$part" || { echo "Bagian hilang atau kosong: $part"; exit 1; }
done
du -ch DDR-dataset.zip.{001..010} | tail -1
df -h .
~~~

Perintah di atas belum menghitung ukuran hasil ekstraksi. Jika kapasitas untuk ZIP gabungan dan hasil ekstraksi belum cukup, berhenti dan gunakan volume lain. Setelah cukup, gabungkan berurutan, uji integritas, lalu **inspeksi daftar path internal**:

~~~bash
test ! -e DDR-dataset.zip || { echo "ZIP gabungan sudah ada; periksa sebelum mengulang"; exit 1; }
cat DDR-dataset.zip.{001..010} > DDR-dataset.zip
unzip -t DDR-dataset.zip
unzip -l DDR-dataset.zip
~~~

Jika uji integritas ZIP gagal, jangan gunakan hasilnya. Periksa nama folder, berkas label, serta entri path absolut atau yang memuat ../ sebelum mengekstrak. Setelah struktur dan kapasitas sesuai, jalankan sekali:

~~~bash
unzip DDR-dataset.zip -d extracted
~~~

Bandingkan checksum bagian dengan checksum resmi apabila penerbit menyediakannya; hash lokal saja tidak membuktikan keaslian unduhan. Server yang diaudit sudah memiliki <code>/mnt/home/probation-03/Kel_13/datasets/DDR/DDR-dataset/DR_grading/</code>; jangan menggabungkan atau mengekstrak ulang arsip di sana. Cell preflight hanya menginventarisasi; fungsi <code>ddr_records()</code> membaca anotasi dan memvalidasi pasangan gambar saat manifest diminta.

## Status implementasi dan final test historis V1

Notebook mencakup EDA dan preprocessing sebagai tahapan terpadu bersama model, training, evaluasi, dan checkpointing. Grafik EDA historis §2.2 merangkum subset 500 gambar bertanda APTOS dan 500 bertanda DDR, tetapi sumber file serta keanggotaannya dalam manifest V2 belum terverifikasi. EDA lokal §2.3 memverifikasi distribusi kelas dari 3.662 pasangan APTOS2019 dan meninjau karakteristik 60 gambar; tidak ada distribusi kelas DDR lokal yang diukur pada bagian itu. Pipeline DDR membaca anotasi resmi, menyimpan split bawaan untuk audit, dan membuat split reproduksi 8:1:1 berkelompok SHA-256. Audit yang diberikan pengguna melaporkan 12.422 grup unik, 97 grup duplikat, 9 grup melintasi split resmi, dan nol konflik label pada 12.522 gambar grading. Dataset server tidak diakses dari laptop saat penyusunan laporan ini. ID split penulis tidak tersedia.

Dua hasil **final test V1** berikut disalin dari JSON eksperimen aktual; angka ini berbeda dari metrik validation V2 yang dipakai untuk memilih checkpoint:

| Dataset | Run | Epoch terbaik dari validation | Jumlah gambar test | Akurasi test | F1 makro test | Specificity makro test |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| APTOS2019 | <code>aptos_main_80ep_seed42</code> | 8 | 366 | 0,724044 | 0,554058 | 0,933978 |
| DDR | <code>ddr_main_80ep_seed42</code> | 5 | 1.252 | 0,431310 | 0,353457 | 0,857445 |

Tabel final test dan dua confusion matrix V1 berbentuk heatmap berada di §6 notebook. Riwayat V1 tetap terpisah dari V2: log APTOS2019 V1 yang tersedia berisi 43 catatan epoch, sedangkan log DDR V1 yang tersedia hanya memuat resume epoch 59–69. Pada segmen DDR tersebut train/validation loss NaN; awal dan penyebabnya belum terverifikasi. Pemeriksaan checkpoint V1 yang dilaporkan peneliti belum diulang secara lokal. **Final test V2 80 epoch telah dilaporkan peneliti, tetapi artefak JSON dan confusion matrix V2 belum tersedia di workspace lokal untuk verifikasi.**

Model tetap kandidat reproduksi parsial: 5.282.101 parameter versus 7,2629 juta pada Tabel 1 paper (r=16), selisih nominal 1.980.799. AdamW, WeightedRandomSampler V1, konfigurasi internal arsitektur, dan pemilihan checkpoint dari validation macro-F1 adalah *implementation assumptions*; proses Optuna paper tidak direproduksi. **INFERRED FROM PAPER:** F1 88,4% APTOS dan 84,4% DDR pada Tabel 4–5 konsisten dengan F1 berbobot support yang dihitung dari matriks Gambar 5, tetapi paper tidak menyatakan agregasi F1 secara eksplisit. Rekonstruksi specificity DDR dari matriks paper belum cocok tepat dengan 90,3% pada Tabel 5. F1 makro dan F1 per kelas kami merupakan **SUPPLEMENTARY EVALUATION**; keduanya berbeda dari weighted loss sebagai metode optimisasi. Perhitungan F1 tambahan dari prediksi yang sudah dibekukan tidak mengubah model atau pelatihan. Gambar 5 paper memuat 367 APTOS test image, sedangkan split kami 366; keanggotaan test tidak dapat dianggap identik.

## Status V1 dan V2

**V1 — eksperimen reproduksi awal** mempertahankan preprocessing crop → resize 512×512 (INTER_AREA) → Gaussian smoothing σ=1 (nilai σ asumsi), <code>WeightedRandomSampler</code> pada train, target 80 epoch, dan ringkasan hasil historis di §5.8 dan final test V1 di bagian 6 notebook. Kode, checkpoint, serta metrik V1 tidak ditimpa.

**V2 — penyempurnaan berdasarkan literatur** mempertahankan arsitektur <code>CADViTCandidate</code> dan manifest/split V1. Setelah operasi preprocessing CAD-ViT (§III.A, Gbr. 1), V2 menambahkan Ben Graham pada RGB float32: <code>clip(4I − 4GaussianBlur(I, σ=10) + 128, 0, 255)</code>, sebelum augmentasi train dan skala tensor. Bentuk dan σ=10 didukung Pratama et al. (2026), §II.B.2, Pers. (2), Gbr. 2, dalam paper *Ben Graham Retinal Enhancement for Hierarchical Transformer-Based Diabetic Retinopathy Grading*; koefisien 4/−4 adalah *implementation assumption* peneliti, bukan nilai eksplisit CAD-ViT. V2 tidak memakai CLAHE atau <code>WeightedRandomSampler</code>. Grafik APTOS lokal baru pada §2 notebook menunjukkan halo berwarna pada tiga contoh V2; dampaknya pada klasifikasi belum terbukti.

### Eksperimen V2 selesai — 65/30 epoch

**Hasil training V2 dari log asli di luar repository:** APTOS2019 <code>aptos_v2_bg10_stable_65ep_seed42</code> lengkap epoch 1–65; best validation macro-F1 **0,650653821679372** pada epoch 65, validation accuracy 0,8142076502732241. DDR <code>ddr_v2_bg10_stable_30ep_seed42</code> lengkap epoch 1–30; best validation macro-F1 **0,5877264724487317** pada epoch 24, validation accuracy 0,8035143769968051; final epoch 30 macro-F1 0,5127013785813619. Log APTOS mencatat 10 skipped optimizer steps AMP, DDR 15; semua metrik tercatat finite. SHA-256 dan metrik historis pernah dicatat pada versi sebelumnya; ringkasannya kini di §5.8, sedangkan log asli dan riwayat Git tetap tersedia. **Ini metrik validation, bukan final test.** Peneliti melaporkan pemeriksaan checkpoint APTOS <code>best.pt</code>/<code>last.pt</code>: indeks epoch 64, 285 tensor model, nol non-finite; checkpoint tidak dibuka ulang secara lokal. File checkpoint/config DDR dilaporkan tersedia di GPU lab, tetapi isinya belum diperiksa. Final test untuk run historis V2 65/30 epoch ini tidak terdokumentasi di workspace lokal.

Durasi run historis V2, **65 epoch APTOS2019 dan 30 epoch DDR**, adalah deviasi dari 80 epoch CAD-ViT (§IV.B). Keduanya tetap hasil aktual yang terpisah dari run 80 epoch berikutnya. LR 0,000186, batch fisik 16, input 512×512, r=16, dan split 8:1:1 adalah target konfigurasi paper dalam kode. AdamW weight decay 0, CrossEntropyLoss, AMP+GradScaler, seed 42, clipping gradien norm 1, tanpa scheduler/early stopping, serta checkpoint menurut validation macro-F1 adalah *implementation assumptions*. Tidak ada Optuna ulang. Perbandingan paper, V1 final test, dan V2 validation ada di §6.8 notebook.

### V2 — hasil pelatihan aktual 80/80 epoch

`V2_EPOCHS` menetapkan **80 epoch untuk APTOS2019 dan DDR** sesuai target paper (§IV.B). Log APTOS2019 mencatat 80 epoch; validation macro-F1 terbaik **0,657127419 pada epoch 69**, accuracy **0,814207650** dan specificity **0,951646177** pada epoch itu. Rangkaian log DDR mencatat 80 epoch; validation macro-F1 terbaik **0,690323426 pada epoch 72**, accuracy **0,829872204**. Angka ini merupakan **validation**, bukan final test. Sumber dan 160 baris metrik, dua tabel lengkap, serta grafik tersimpan di §5.5 notebook; file log asli berada di luar repository. Specificity DDR pada epoch terbaik tidak tersedia pada JSONL recovery.

DDR terdiri atas tiga fase tercatat: **AMP FP16 epoch 1–59**, **weighted FP32 epoch 60–63**, serta **weighted FP32 dengan penghitungan gradient norm FP64 epoch 64–80**. Weighted loss dan FP64 clipping adalah *experimental extensions*, bukan metode CAD-ViT. Menurut riwayat terminal yang disampaikan peneliti, `last.pt` epoch 63 dan konfigurasi recovery disalin ke direktori FP64 sebelum epoch 64; log dan kode recovery konsisten dengan kelanjutan itu. Arsip audit tidak memuat transkrip pemindahan, hash transfer, manifest, atau checkpoint untuk membuktikan kesinambungan state secara langsung. Diagnostik FP32 unweighted epoch 60–62 tidak masuk kurva 80 epoch final. Karena objective loss berubah di batas fase, besaran loss lintas fase tidak boleh ditafsirkan sebagai satu kurva homogen.

**Status bukti:** kelengkapan 80+80 baris dan nilai metrik terverifikasi terhadap log yang tersedia; lineage checkpoint serta verifikasi lokal artefak final test/confusion matrix V2 historis masih **PENDING**. Hasil final test APTOS `balanced_v1` di bawah ialah kampanye berbeda dan tidak menggantikan hasil V2. Split penulis, konfigurasi internal arsitektur, dan definisi agregasi F1 paper belum dapat dipastikan. Diagram dan kurva di notebook adalah visualisasi data/protokol yang terdokumentasi, bukan bukti kesetaraan implementasi dengan penulis.

## Lingkungan dan reproduksibilitas

Semua pengembangan notebook dilakukan di laptop lokal. Dataset, manifest, checkpoint, dan log berada di luar repository. Saat notebook dijalankan secara manual, `CAD_VIT_DATA_ROOT` harus menunjuk direktori yang memuat `APTOS2019/` dan `DDR/`; `CAD_VIT_RUN_ROOT` dan `CAD_VIT_CHECKPOINT_ROOT` menunjuk lokasi keluaran terpisah dari repository. Path aktual harus diperiksa oleh operator. Codex atau agentic tools tidak dijalankan di server GPU KC.

`prepare_manifest("APTOS2019")` dan `prepare_manifest("DDR")` membaca data resmi, memvalidasi pasangan gambar-label, dan membuat split reproduksi 8:1:1 berkelompok SHA-256 dengan seed 42. Manifest yang fingerprint-nya berubah tidak ditimpa secara diam-diam. Split bawaan DDR disimpan sebagai metadata audit, sedangkan split eksperimen dibuat ulang karena ID split penulis paper tidak tersedia. Arsip DDR disiapkan terpisah; notebook tidak menggabungkan atau mengekstraknya saat training.

Kode training, resume, evaluasi, dan guard `RUN_V2_FINAL_TEST = False` tetap ada di notebook sebagai bagian workflow. Dokumentasi hasil 80 epoch di sini **tidak menjalankan ulang** kode tersebut. Sebelum menggunakan hasil final test V2 yang dilaporkan, verifikasi checkpoint, manifest, dan JSON hasil yang tepat; pemilihan checkpoint harus berdasarkan validation. Jangan menafsirkan confusion matrix V1 sebagai hasil V2.

### Klasifikasi keputusan metode

| Kategori | Contoh | Dasar atau batasan |
| --- | --- | --- |
| EXPLICIT IN PAPER | Input 512 × 512, learning rate 0,000186, batch 16, 80 epoch, split 8:1:1, Coordinate Attention r=16 | CAD-ViT (2025), §III–IV dan Tabel 1 |
| INFERRED FROM PAPER | F1 Tabel 4–5 kemungkinan berbobot support | Rekonstruksi dari matriks Gambar 5; definisi agregasi tidak dinyatakan penulis |
| IMPLEMENTATION ASSUMPTION | Kanal/expansion/patch/depth/heads internal kandidat, AdamW, seed 42, resep jumlah draw oversampling, seleksi checkpoint dari validation macro-F1, Gaussian σ=1 | Tidak dispesifikasikan cukup rinci oleh paper; dicatat dalam kode dan konfigurasi |
| EXPERIMENTAL EXTENSION | Ben Graham σ=10 pada V2, full oversampling train-only `balanced_v1`, weighted CE DDR epoch 60–80, FP64 gradient norm epoch 64–80 | Tambahan eksperimen/recovery; bukan klaim metode asli CAD-ViT |
| SUPPLEMENTARY EVALUATION | F1 makro, F1 per kelas, F1 berbobot support yang dihitung dari prediksi beku | Menambah cara melaporkan hasil tanpa mengubah pelatihan atau checkpoint |
| UNVERIFIED / PENDING | Isi manifest/checkpoint APTOS `balanced_v1`, final test V2 historis, perbandingan balancing dua kondisi, biner checkpoint/manifest dataset, prediksi per gambar, dan transkrip eksekusi DDR recovery | Artefak sumber tersebut belum tersedia untuk pemeriksaan langsung di workspace lokal; evaluator DDR yang diunduh kini diarsipkan terpisah |

Dependensi runtime tambahan yang perlu dipastikan tersedia: `torchvision` yang cocok dengan versi PyTorch, `Pillow`, `numpy`, dan `jupyter`; `torch`, `pandas`, `opencv-python`/`opencv-python-headless`, `scikit-learn`, `matplotlib`, `seaborn`, serta `wandb` juga dipakai notebook. Jangan menyimpan token W&B di notebook atau repository. Mode W&B default `offline`; mode `online` memerlukan autentikasi melalui environment atau mekanisme login manual yang aman.

## Notebook reproduksi terkontrol (hanya-baca)

`CAD_ViT.ipynb` tetap notebook historis utama dan tidak ditimpa. Delapan notebook bernomor pada `reproduction/notebooks/` menampilkan protokol dan artefak kampanye baru tanpa mengulang training/final test: `01_preprocessing`, `02_architecture`, `03_training_protocol`, `04_classification_evaluation`, `05_ablation_and_reduction`, `06_balancing_analysis`, `07_error_analysis`, dan `08_gradcam`. Audit lama `preprocessing_analysis.ipynb` tetap tersedia. Semua notebook bernomor mengimpor modul `reproduction.cadvit` yang sama dengan runner.

Notebook 04 menyimpan tiga heatmap **PAPER_REFERENCE** Gambar 4–5 dan menautkan PNG **DDR_FINAL_TEST** dari artefak lokal; notebook 06 menyimpan dua diagram batang **PAPER_REFERENCE** Gambar 6. Notebook 02 dan 08 menyimpan hasil **SYNTHETIC_TEST** CPU. Notebook 06 dapat membaca kurva dan distribusi kelas dari artefak kampanye yang fingerprint-nya cocok; notebook 04 membaca ringkasan dan menampilkan confusion matrix DDR final test secara read-only. Hasil final test APTOS `balanced_v1` tersedia sebagai JSON dan matriks angka pada §6.9 notebook utama; heatmap baru dari JSON itu belum tersimpan. Galeri kesalahan dan Grad-CAM nyata tetap **PENDING**. Path artefak dibaca melalui `CAD_VIT_ARTIFACT_ROOT`; audit gambar train/validation memerlukan `CAD_VIT_AUDIT_IMAGE`, `CAD_VIT_AUDIT_MANIFEST`, `CAD_VIT_DATA_ROOT` (root dataset tertentu), dan `CAD_VIT_AUDIT_SPLIT`. Notebook menolak gambar test untuk audit visual. Galeri kesalahan test memerlukan opt-in `CAD_VIT_SHOW_TEST_GALLERY=1` setelah evaluasi final dikunci; Grad-CAM pada gambar train/validation memerlukan `CAD_VIT_RUN_GRADCAM=1` dan checkpoint tepercaya. Inventaris grafik dan keterbatasannya ada di [audit figur](docs/FIGURE_AUDIT.md).

`paper_v1` yang telah diinisialisasi di KC adalah kontrol dengan kelas **tidak seimbang** dan tidak boleh ditimpa. Kampanye `balanced_v1` terpisah memakai oversampling deterministik hanya pada train sebagai **EXPERIMENTAL EXTENSION**; resep target jumlah draw per kelas merupakan **IMPLEMENTATION ASSUMPTION**. Paper menyebut balancing melalui augmentasi tetapi tidak memberi resep per kelas. Pemeriksaan memori sintetis, preflight manifest/dependensi, dan pilot satu epoch adalah langkah manual untuk run baru yang terpisah, setelah status artefak server diperiksa. Langkah ini tidak boleh menginisialisasi ulang run yang sudah berjalan atau diarsipkan. Perintah, matriks Gambar 1–8/Tabel 1–5, definisi metrik, dan status GO/NO-GO ada di [rencana reproduksi](docs/PAPER_EXPERIMENT_REPRODUCTION_PLAN.md). Jumlah parameter kandidat tetap 5.282.101 versus 7,2629 juta pada Tabel 1 paper; hasil baru tetap reproduksi independen, bukan kode resmi penulis.

## Kampanye terkontrol — laporan server 8 November dan artefak APTOS 10 November 2026

Tabel berikut adalah **snapshot awal**: baris APTOS berdasarkan artefak terkomit di `results/APTOS2019/full_oversampling_amp_fp16_80epochs/`, sedangkan baris DDR saat itu berasal dari keluaran terminal server yang dilaporkan peneliti. Notebook utama menempatkan snapshot di §5.7.1 dan hasil DDR recovery terbaru di §5.7.2, terpisah dari V1 serta V2 historis 80/80 epoch di §5.5. Config, JSONL, CSV, dan ringkasan test APTOS konsisten untuk run/epoch yang tercatat; berkas manifest dan checkpoint asli tidak tersedia lokal untuk pemeriksaan isi langsung. History, config, protokol, provenance fork, dan manifest checkpoint DDR recovery kini tersedia lokal; source runner, manifest dataset, dan biner checkpoint masih berada di luar workspace ini. Menampilkan `history.jsonl` saja belum memastikan kecocokan konfigurasi/protokol, SHA-256 manifest, hash sumber model dan preprocessing, serta metadata checkpoint.

| Run | Status terakhir yang dilaporkan | Validation, bukan pengujian akhir |
| --- | --- | --- |
| APTOS2019 `balanced_v1/aptos_r16` | AMP, full oversampling 7.225 draw/epoch; **80/80 epoch** menurut JSONL/CSV lokal. JSONL mencatat **26** skipped AMP updates dalam 80 epoch (23 hingga epoch 69). | Best validation epoch 24; accuracy 0,7923497267759563; macro-F1 0,6614213564213565; loss 0,6743926057398645 |
| DDR `balanced_v1` | AMP, gagal selama epoch 18; run dilaporkan diarsipkan pada experiments/archived_runs/ddr_balanced_v1_amp_failed_epoch18 | Checkpoint terbaik epoch 16; macro-F1 sekitar 56,84% |
| DDR `baseline_fp32_v2` | FP32 tanpa oversampling; pilot 3 epoch | Macro-F1 epoch 3 28,08% |
| DDR `capped_fp32_v2` | FP32 dengan cap oversampling 3×; pilot 3 epoch | Macro-F1 log sekitar 35,95%; evaluasi ulang CPU: accuracy 64,94%, macro-F1 35,91% |
| DDR `balanced_fp32_v2` | Runner `run_ddr_balanced_fp32.py` dilaporkan ada di worktree server; preflight dilaporkan lulus dan kemajuan pelatihan dilaporkan terpisah, tetapi epoch serta metrik belum diperiksa dari artefak lokal | **PENDING verifikasi**. Oversampling penuh hanya pada train: 25.070 sampel/epoch; target 80 epoch, belum diklaim selesai |

Confusion matrix DDR capped pilot yang diberikan peneliti ditampilkan di §6.3.1 notebook sebagai **validation epoch 3, reported server evidence**. Angka dan heatmap tersebut tidak menggantikan dua confusion matrix **V1 historical final test** di §6.3. Kode EDA dan audit preprocessing baru hanya membaca manifest atau gambar train/validation bila path dan opt-in diberikan; tanpa artefak ia menampilkan `PENDING`. Tidak ada training, final test, Optuna, atau pemrosesan dataset penuh yang dijalankan oleh sel baru.

### DDR 80 epoch — full oversampling FP32 Math recovery

Artefak aktual ada di [hasil DDR 80 epoch](results/DDR/full_oversampling_fp32_math_mb8_80epochs/). `history.jsonl` berisi 80 epoch berurutan; `summary.json`, `config.json`, `protocol.json`, `FORK_PROVENANCE.json`, dan `checkpoint_manifest.json` mencatat konfigurasi, asal recovery, serta identitas checkpoint. Grafik training dan validation juga ada dalam folder itu. Kampanye `cadvit_tables_1_5_balanced_fp32_math_mb8_diag/ddr_full` mengklasifikasi DR dalam lima kelas. *Full oversampling* membuat setiap kelas memiliki 5.014 draw per epoch **pada data train saja**; draw tambahan dapat mengulang gambar yang sama.

| Tahap | Konfigurasi | Hasil yang boleh dilaporkan |
| --- | --- | --- |
| Epoch 1–42 | FP32, micro-batch fisik 16 | Training/validation dari run awal |
| Epoch 43–80 | Recovery dari checkpoint epoch 42; FP32 Math SDPA, micro-batch fisik 8, akumulasi 2, batch efektif 16 | Training/validation dari fork diagnostik |
| Pemilihan model | Validation macro-F1 tertinggi, bukan accuracy atau epoch terakhir | `best.pt` epoch **69**; `last.pt` epoch **80** |
| Final test | **COMPLETED**; checkpoint epoch 69 | 1.252 gambar; matriks, tabel per kelas, dan metrik ada pada artefak lokal; prediksi per gambar tetap di server |

Training selesai **80/80 epoch** dengan **0 skipped optimizer steps** dalam history. Pada epoch pilihan 69, **validation macro-F1 0,653069359**, **accuracy 0,785942492**, dan **loss 0,790420573**. Pada epoch 80, validation accuracy **0,788338658** dan macro-F1 **0,626342701**. Angka `best_validation_accuracy` dan `best_validation_loss` dalam ringkasan adalah nilai **pada checkpoint pilihan epoch 69**, bukan nilai tertinggi/terendah dari seluruh epoch. *Training metric* menggambarkan proses belajar pada data train; *validation metric* dipakai untuk memilih checkpoint tanpa menyentuh test; *final-test metric* baru dihitung sekali pada data test setelah pilihan model dikunci. Recovery mengubah konfigurasi di tengah jalan, sehingga hasil ini **mixed-protocol** dan tidak boleh disebut satu run 80 epoch homogen atau dibandingkan langsung dengan angka test paper. Biner checkpoint dan manifest dataset tidak tersedia di laptop ini untuk audit langsung; hash checkpoint dicatat dalam `checkpoint_manifest.json` dan perlu dicocokkan lagi di server sebelum final test.

Notebook evaluasi [04_classification_evaluation.ipynb](reproduction/notebooks/04_classification_evaluation.ipynb) menampilkan matriks asli [DDR_FINAL_TEST](results/DDR/full_oversampling_fp32_math_mb8_80epochs/confusion_matrix.png), tabel per kelas, dan metrik dari [final_test_summary.json](results/DDR/full_oversampling_fp32_math_mb8_80epochs/final_test_summary.json). [per_class_metrics.csv](results/DDR/full_oversampling_fp32_math_mb8_80epochs/per_class_metrics.csv) menyediakan angka per kelas dalam bentuk tabular. Dari **1.252** gambar test, **975** benar dan **277** salah: accuracy **0,7787539936102237**, macro-F1 **0,6319812217431724**, weighted-F1 **0,8011143014317892**, precision makro **0,6214465962662282**, recall makro **0,6840525897005094**, specificity makro **0,9410946313470154**, QWK **0,8370787904943411**, dan grade MAE **0,305111821086262**. **Validation macro-F1 0,653069359** memilih checkpoint epoch 69; **final-test macro-F1 0,631981222** mengukur split test beku sekali setelah pemilihan. Test tidak digunakan untuk memilih ulang model.

Matriks berorientasi **baris = grade sebenarnya, kolom = prediksi**; jumlahnya 1.252 dan diagonalnya 975. F1 grade 1 paling rendah (**0,323144**). Grade 3 hanya memiliki 23 gambar test, 11 diprediksi benar; grade 2→3 berjumlah 16 dan 3→2 berjumlah 11. Pola agregat ini tidak membuktikan penyebab klinis atau visual. Prediksi per gambar sengaja tetap pada server sehingga galeri kesalahan lokal belum tersedia. Notebook 07 membatasi analisis kesalahan pada matriks dan tabel per kelas.

**Provenance dan batas audit lokal.** `manifest_sha256` pada ringkasan test cocok dengan `config.json`; `checkpoint_sha256` cocok dengan `checkpoint_manifest.json`. Seluruh metrik agregat, jumlah gambar, dan baris CSV dapat dihitung ulang dari matriks. Source evaluator yang diunduh dari server telah disalin **identik byte-per-byte** ke [arsip provenance DDR](reproduction/provenance/ddr_final_test/README.md); SHA-256-nya cocok dengan hash server yang dicatat. Ringkasan mencatat `source_test_metrics_sha256`, tetapi berkas `test_metrics.json` sumber, manifest, biner checkpoint, prediksi per gambar, dan transkrip eksekusi tidak tersedia lokal untuk audit langsung. Hash source yang cocok **tidak membuktikan sendiri** bahwa file itu dieksekusi di server. [ddr_recovery_final_test.py](reproduction/cadvit/ddr_recovery_final_test.py) adalah **draft adapter independen yang belum terlacak Git**, bukan evaluator asli; CLI, ukuran batch evaluasi, dan format keluarannya berbeda. Keluarkan draft tersebut dari commit final kecuali diaudit terpisah. **Jangan menjalankan ulang final test.**

**Perbandingan kontekstual dengan paper.** Tabel 4–5 CAD-ViT melaporkan accuracy **0,886** pada APTOS dan **0,848** pada DDR; final test reproduksi mencatat **0,732240** dan **0,778754**. F1 paper **0,884** dan **0,844** tidak otomatis setara dengan F1 makro kami karena agregasinya tidak dijelaskan cukup rinci. Parameter model kandidat sekitar **5,28 juta**, versus **7,26 juta** pada paper; split penulis, rincian arsitektur, Gaussian blur, balancing, dan protokol recovery DDR juga berbeda atau belum dapat dicocokkan. Ini **reproduksi parsial yang dipandu paper**, bukan implementasi identik atau bukti kesiapan klinis. Tabel berlabel `PAPER_REPORTED` dan `OUR_FINAL_TEST` serta keterbatasan lengkap ada di §6.10 dan §7.1 [notebook utama](CAD_ViT.ipynb).

### Final test APTOS2019 `balanced_v1/aptos_r16` yang terkomit

`final_test_summary.json` mencatat checkpoint **epoch 24**, dipilih berdasarkan **validation macro-F1** sebelum final test (**IMPLEMENTATION ASSUMPTION**). Pada **366** gambar test: accuracy **0,73224043715847**, precision makro **0,5325509295442328**, recall makro **0,5357451099011897**, F1 makro **0,5315285435735081**, F1 berbobot support **0,7449461138522905**, dan specificity makro **0,9360795785213256**. F1 kelas 0–4 masing-masing **0,9602272727272727; 0,48101265822784817; 0,6735751295336787; 0,1568627450980392; 0,3859649122807017**. Matriks kebingungan lengkap ada di §6.9 notebook utama. Nilai tersebut bukan hasil V1 atau V2 BG10 historis, dan tidak membuktikan kesamaan split dengan paper. `summary.json` kini menandai final test selesai karena artefak test tersedia; ini tidak menyatakan hash checkpoint/manifest sudah diverifikasi langsung. Grafik perbandingan penyeimbangan tetap **PENDING** sampai kontrol imbalanced yang sebanding tersedia.

Untuk sel analisis baru di notebook utama, set `CAD_VIT_ARTIFACT_ROOT` ke root artefak campaign **di luar repository**. Sel §2.4 membaca metadata manifest dan history tanpa membuka gambar test. Galeri train dibatasi dua gambar per grade dan maksimal 60 gambar untuk statistik resolusi; aktifkan secara sadar dengan `CAD_VIT_RENDER_TRAIN_EDA=1` serta `CAD_VIT_DATA_ROOT` yang benar. Audit visual §3.4 membutuhkan `CAD_VIT_RENDER_PREPROCESSING=1`, `CAD_VIT_AUDIT_IMAGE`, `CAD_VIT_AUDIT_MANIFEST`, `CAD_VIT_AUDIT_DATASET_ROOT`, dan `CAD_VIT_AUDIT_SPLIT=train` atau `val`. Gambar tersebut harus cocok dengan manifest dan SHA-256. Jika DDR berada pada root khusus, `CAD_VIT_DDR_ROOT` dapat menunjuk root yang sesuai dengan path relatif manifest. Jangan gunakan `Run All` untuk mengeksekusi ulang workflow historis; output lama sudah tersimpan, sedangkan sel baru dapat dijalankan secara terpisah dalam mode analisis CPU.
