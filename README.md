# CAD-ViT: Coordinate Attention-Enhanced Vision Transformer With Dilated Feature Fusion for Diabetic Retinopathy Staging

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

Contoh Colab, jalankan sebelum cell konfigurasi:

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

| Dataset | Run | Best epoch dari validation | n test | Test accuracy | Test F1 macro | Test specificity macro |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| APTOS2019 | <code>aptos_main_80ep_seed42</code> | 8 | 366 | 0,724044 | 0,554058 | 0,933978 |
| DDR | <code>ddr_main_80ep_seed42</code> | 5 | 1.252 | 0,431310 | 0,353457 | 0,857445 |

Tabel final test dan dua confusion matrix V1 berbentuk heatmap berada di §6 notebook. Riwayat V1 tetap terpisah dari V2: log APTOS2019 V1 yang tersedia berisi 43 catatan epoch, sedangkan log DDR V1 yang tersedia hanya memuat resume epoch 59–69. Pada segmen DDR tersebut train/validation loss NaN; awal dan penyebabnya belum terverifikasi. Pemeriksaan checkpoint V1 yang dilaporkan peneliti belum diulang secara lokal. **Confusion matrix dan final test V2 80 epoch masih PENDING.**

Model tetap kandidat reproduksi parsial: 5.282.101 parameter versus 7.262.900 pada Tabel 1 paper (r=16), selisih 1.980.799. AdamW, WeightedRandomSampler V1, konfigurasi internal arsitektur, dan pemilihan checkpoint dari validation macro-F1 adalah *implementation assumptions*; proses Optuna paper tidak direproduksi. Agregasi F1 paper belum dapat dipastikan identik dengan macro-F1 lima kelas notebook. JSON dan log sumber tetap di luar repository; notebook menyimpan angka, matriks, kurva, serta SHA-256 sumber untuk audit.

## Status V1 dan V2

**V1 — Original Reproduction Experiment** mempertahankan preprocessing crop → resize 512×512 (INTER_AREA) → Gaussian smoothing σ=1 (nilai σ asumsi), <code>WeightedRandomSampler</code> pada train, target 80 epoch, dan seluruh hasil historis di bagian 5–6 notebook. Kode, checkpoint, serta metrik V1 tidak ditimpa.

**V2 — Literature-Guided Refinement** mempertahankan arsitektur <code>CADViTCandidate</code> dan manifest/split V1. Setelah operasi preprocessing CAD-ViT (§III.A, Gbr. 1), V2 menambahkan Ben Graham pada RGB float32: <code>clip(4I − 4GaussianBlur(I, σ=10) + 128, 0, 255)</code>, sebelum augmentasi train dan skala tensor. Bentuk dan σ=10 didukung Pratama et al. (2026), §II.B.2, Pers. (2), Gbr. 2, dalam paper *Ben Graham Retinal Enhancement for Hierarchical Transformer-Based Diabetic Retinopathy Grading*; koefisien 4/−4 adalah *implementation assumption* peneliti, bukan nilai eksplisit CAD-ViT. V2 tidak memakai CLAHE atau <code>WeightedRandomSampler</code>. Grafik APTOS lokal baru pada §2 notebook menunjukkan halo berwarna pada tiga contoh V2; dampaknya pada klasifikasi belum terbukti.

### Completed V2 Experiments — 65/30 Epochs

**Hasil training V2 dari log asli di luar repository:** APTOS2019 <code>aptos_v2_bg10_stable_65ep_seed42</code> lengkap epoch 1–65; best validation macro-F1 **0,650653821679372** pada epoch 65, validation accuracy 0,8142076502732241. DDR <code>ddr_v2_bg10_stable_30ep_seed42</code> lengkap epoch 1–30; best validation macro-F1 **0,5877264724487317** pada epoch 24, validation accuracy 0,8035143769968051; final epoch 30 macro-F1 0,5127013785813619. Log APTOS mencatat 10 skipped optimizer steps AMP, DDR 15; semua metrik tercatat finite. SHA-256 log dan seluruh metrik per epoch tertanam di §5.7–5.8 notebook bersama grafik yang tersimpan sebagai output notebook; rendering di GitHub belum diverifikasi sebelum publikasi. **Ini metrik validation, bukan final test.** Peneliti melaporkan pemeriksaan checkpoint APTOS <code>best.pt</code>/<code>last.pt</code>: indeks epoch 64, 285 tensor model, nol non-finite; checkpoint tidak dibuka ulang secara lokal. File checkpoint/config DDR dilaporkan tersedia di GPU lab, tetapi isinya belum diperiksa. Evaluasi final test V2 masih pending.

Durasi run historis V2, **65 epoch APTOS2019 dan 30 epoch DDR**, adalah deviasi dari 80 epoch CAD-ViT (§IV.B). Keduanya tetap hasil aktual yang terpisah dari run 80 epoch berikutnya. LR 0,000186, batch fisik 16, input 512×512, r=16, dan split 8:1:1 adalah target konfigurasi paper dalam kode. AdamW weight decay 0, CrossEntropyLoss, AMP+GradScaler, seed 42, clipping gradien norm 1, tanpa scheduler/early stopping, serta checkpoint menurut validation macro-F1 adalah *implementation assumptions*. Tidak ada Optuna ulang. Perbandingan paper, V1 final test, dan V2 validation ada di §6.8 notebook.

### V2 — Hasil Training Aktual 80/80 Epochs

`V2_EPOCHS` menetapkan **80 epoch untuk APTOS2019 dan DDR** sesuai target paper (§IV.B). Log APTOS2019 mencatat 80 epoch; validation macro-F1 terbaik **0,657127419 pada epoch 69**, accuracy **0,814207650** dan specificity **0,951646177** pada epoch itu. Rangkaian log DDR mencatat 80 epoch; validation macro-F1 terbaik **0,690323426 pada epoch 72**, accuracy **0,829872204**. Angka ini merupakan **validation**, bukan final test. Sumber dan 160 baris metrik tertanam di §5.10 notebook; file log asli berada di luar repository. Specificity DDR pada epoch terbaik tidak tersedia pada JSONL recovery.

DDR terdiri atas tiga fase tercatat: **AMP FP16 epoch 1–59**, **weighted FP32 epoch 60–63**, serta **weighted FP32 dengan penghitungan gradient norm FP64 epoch 64–80**. Weighted loss dan FP64 clipping adalah *experimental extensions*, bukan metode CAD-ViT. Menurut riwayat terminal yang disampaikan peneliti, `last.pt` epoch 63 dan konfigurasi recovery disalin ke direktori FP64 sebelum epoch 64; log dan kode recovery konsisten dengan kelanjutan itu. Arsip audit tidak memuat transkrip pemindahan, hash transfer, manifest, atau checkpoint untuk membuktikan kesinambungan state secara langsung. Diagnostik FP32 unweighted epoch 60–62 tidak masuk kurva 80 epoch final. Karena objective loss berubah di batas fase, besaran loss lintas fase tidak boleh ditafsirkan sebagai satu kurva homogen.

**Status bukti:** kelengkapan 80+80 baris dan nilai metrik terverifikasi terhadap log yang tersedia; lineage checkpoint dan final test/confusion matrix V2 masih **PENDING**. Model kandidat tetap 5.282.101 parameter dibanding 7.262.900 pada Tabel 1 paper. Split penulis, konfigurasi internal arsitektur, dan definisi agregasi F1 paper belum dapat dipastikan. Diagram dan kurva di notebook adalah visualisasi data/protokol yang terdokumentasi, bukan bukti kesetaraan implementasi dengan penulis.

## Lingkungan dan reproduksibilitas

Semua pengembangan notebook dilakukan di laptop lokal. Dataset, manifest, checkpoint, dan log berada di luar repository. Saat notebook dijalankan secara manual, `CAD_VIT_DATA_ROOT` harus menunjuk direktori yang memuat `APTOS2019/` dan `DDR/`; `CAD_VIT_RUN_ROOT` dan `CAD_VIT_CHECKPOINT_ROOT` menunjuk lokasi keluaran terpisah dari repository. Path aktual harus diperiksa oleh operator. Codex atau agentic tools tidak dijalankan di server GPU KC.

`prepare_manifest("APTOS2019")` dan `prepare_manifest("DDR")` membaca data resmi, memvalidasi pasangan gambar-label, dan membuat split reproduksi 8:1:1 berkelompok SHA-256 dengan seed 42. Manifest yang fingerprint-nya berubah tidak ditimpa secara diam-diam. Split bawaan DDR disimpan sebagai metadata audit, sedangkan split eksperimen dibuat ulang karena ID split penulis paper tidak tersedia. Arsip DDR disiapkan terpisah; notebook tidak menggabungkan atau mengekstraknya saat training.

Kode training, resume, evaluasi, dan guard `RUN_V2_FINAL_TEST = False` tetap ada di notebook sebagai bagian workflow. Dokumentasi hasil 80 epoch di sini **tidak menjalankan ulang** kode tersebut. Sebelum final test V2 dilakukan di masa depan, verifikasi checkpoint dan manifest yang tepat, pilih checkpoint hanya dari validation, lalu catat hasil test sebagai analisis baru. Jangan menafsirkan confusion matrix V1 sebagai hasil V2.

### Klasifikasi keputusan metode

| Kategori | Contoh | Dasar atau batasan |
| --- | --- | --- |
| EXPLICIT IN PAPER | Input 512 x 512, learning rate 0.000186, batch 16, 80 epoch, split 8:1:1, Coordinate Attention r=16 | CAD-ViT (2025), Section III dan IV; angka perbandingan parameter pada Tabel 1 |
| IMPLEMENTATION ASSUMPTION | Kanal/expansion/patch/depth/heads internal kandidat, AdamW, seed 42, pemilihan checkpoint dari validation macro-F1, Gaussian sigma 1 | Tidak dispesifikasikan cukup rinci oleh paper; tercatat dalam kode notebook |
| EXPERIMENTAL EXTENSION | Ben Graham sigma 10 pada V2, weighted CE pada DDR epoch 60-80, FP64 gradient norm pada epoch 64-80 | Tambahan eksperimen/recovery; bukan klaim metode asli CAD-ViT |

Dependensi runtime tambahan yang perlu dipastikan tersedia: `torchvision` yang cocok dengan versi PyTorch, `Pillow`, `numpy`, dan `jupyter`; `torch`, `pandas`, `opencv-python`/`opencv-python-headless`, `scikit-learn`, `matplotlib`, `seaborn`, serta `wandb` juga dipakai notebook. Jangan menyimpan token W&B di notebook atau repository. Mode W&B default `offline`; mode `online` memerlukan autentikasi melalui environment atau mekanisme login manual yang aman.
