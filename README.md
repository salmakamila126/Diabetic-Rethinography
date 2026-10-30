# CAD-ViT: Coordinate Attention-Enhanced Vision Transformer With Dilated Feature Fusion for Diabetic Retinopathy Staging

Reproduksi penelitian [CAD-ViT (2025)](https://ieeexplore.ieee.org/document/11009024) dalam satu notebook utama, **CAD_ViT.ipynb**. Notebook mempertahankan EDA dan demonstrasi preprocessing partner. Output EDA yang tersimpan adalah hasil historis dari sumber data lama; jalankan ulang setelah validasi dataset resmi sebelum mengutipnya sebagai hasil reproduksi.

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

## Status implementasi dan hasil final test

Notebook mempertahankan EDA/preprocessing partner, model, pipeline training, evaluasi, dan checkpointing. Pipeline DDR membaca anotasi resmi, menyimpan split bawaan untuk audit, dan membuat split reproduksi 8:1:1 berkelompok SHA-256. Audit yang diberikan pengguna melaporkan 12.422 grup unik, 97 grup duplikat, 9 grup melintasi split resmi, dan nol konflik label pada 12.522 gambar grading. Dataset server tidak diakses dari laptop saat penyusunan laporan ini. ID split penulis tidak tersedia.

Dua hasil **final test** berikut disalin dari JSON eksperimen aktual; angka ini berbeda dari metrik validation yang dipakai untuk memilih checkpoint:

| Dataset | Run | Best epoch dari validation | n test | Test accuracy | Test F1 macro | Test specificity macro |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| APTOS2019 | <code>aptos_main_80ep_seed42</code> | 8 | 366 | 0,724044 | 0,554058 | 0,933978 |
| DDR | <code>ddr_main_80ep_seed42</code> | 5 | 1.252 | 0,431310 | 0,353457 | 0,857445 |

Tabel metrik final test dan dua confusion matrix berbentuk heatmap Matplotlib biru berada di bagian 6 <code>CAD_ViT.ipynb</code>. Bagian 5 memuat grafik APTOS2019 dari tepat 43 catatan epoch 1-43 dan grafik DDR dari log resume epoch 59-69. Log APTOS2019 tidak membuktikan penyelesaian 80 epoch meskipun nama run memuat <code>80ep</code>. Log metrik DDR epoch 1-58 belum ditemukan; 11 epoch resume yang tersedia menunjukkan train loss dan validation loss semuanya NaN, validation accuracy tetap 0,500000, dan validation macro-F1 tetap 0,133333. Best validation macro-F1 DDR yang tercatat di log ialah 0,365784; checkpoint terbaik untuk final test dipilih pada epoch 5. Audit checkpoint terpisah dari pengguna melaporkan 255 tensor non-finite pada <code>last.pt</code> epoch 69 dan bobot finite pada <code>best.pt</code> epoch 5; isi checkpoint tidak diuji ulang secara lokal dalam pembaruan ini. Awal dan penyebab NaN belum diketahui. Penurunan validation macro-F1 APTOS2019 tampak pada log, tetapi penyebabnya juga belum terbukti.

Model tetap kandidat reproduksi parsial: 5.282.101 parameter versus 7.262.900 pada Tabel 1 paper (r=16), selisih 1.980.799. AdamW, WeightedRandomSampler, konfigurasi internal arsitektur, dan pemilihan checkpoint dari validation macro-F1 adalah *implementation assumptions*; proses Optuna paper tidak direproduksi. Agregasi F1 paper belum dapat dipastikan identik dengan macro-F1 lima kelas notebook. JSON dan log sumber tetap di luar repository; notebook menyimpan angka, matriks, kurva, serta SHA-256 sumber untuk audit.

## Status V1 dan V2

**V1 — Original Reproduction Experiment** mempertahankan preprocessing crop → resize 512×512 (INTER_AREA) → Gaussian smoothing σ=1 (nilai σ asumsi), <code>WeightedRandomSampler</code> pada train, target 80 epoch, dan seluruh hasil historis di bagian 5–6 notebook. Kode, checkpoint, serta metrik V1 tidak ditimpa.

**V2 — Literature-Guided Refinement (Training Completed; Final Test Pending)** mempertahankan arsitektur <code>CADViTCandidate</code> dan manifest/split V1. Setelah operasi preprocessing CAD-ViT (§III.A, Gbr. 1), V2 menambahkan Ben Graham pada RGB float32: <code>clip(4I − 4GaussianBlur(I, σ=10) + 128, 0, 255)</code>, sebelum augmentasi train dan skala tensor. Bentuk dan σ=10 didukung Pratama et al. (2026), §II.B.2, Pers. (2), Gbr. 2, dalam paper *Ben Graham Retinal Enhancement for Hierarchical Transformer-Based Diabetic Retinopathy Grading*; koefisien 4/−4 adalah *implementation assumption* peneliti, bukan nilai eksplisit CAD-ViT. V2 tidak memakai CLAHE atau <code>WeightedRandomSampler</code>. Grafik APTOS lokal baru pada §2 notebook menunjukkan halo berwarna pada tiga contoh V2; dampaknya pada klasifikasi belum terbukti.

**Hasil training V2 dari log asli di luar repository:** APTOS2019 <code>aptos_v2_bg10_stable_65ep_seed42</code> lengkap epoch 1–65; best validation macro-F1 **0,650653821679372** pada epoch 65, validation accuracy 0,8142076502732241. DDR <code>ddr_v2_bg10_stable_30ep_seed42</code> lengkap epoch 1–30; best validation macro-F1 **0,5877264724487317** pada epoch 24, validation accuracy 0,8035143769968051; final epoch 30 macro-F1 0,5127013785813619. Log APTOS mencatat 10 skipped optimizer steps AMP, DDR 15; semua metrik tercatat finite. SHA-256 log dan seluruh metrik per epoch tertanam di §5.7–5.8 notebook bersama grafik yang terlihat di GitHub. **Ini metrik validation, bukan final test.** Peneliti melaporkan pemeriksaan checkpoint APTOS <code>best.pt</code>/<code>last.pt</code>: indeks epoch 64, 285 tensor model, nol non-finite; checkpoint tidak dibuka ulang secara lokal. File checkpoint/config DDR dilaporkan tersedia di GPU lab, tetapi isinya belum diperiksa. Evaluasi final test V2 masih pending.

Durasi aktual V2 **65 epoch APTOS2019 dan 30 epoch DDR** adalah deviasi karena keterbatasan sumber daya dari 80 epoch CAD-ViT (§IV.B). LR 0,000186, batch fisik 16, input 512×512, r=16, dan split 8:1:1 tetap target konfigurasi paper dalam kode. AdamW weight decay 0, CrossEntropyLoss, AMP+GradScaler, seed 42, clipping gradien norm 1, tanpa scheduler/early stopping, serta checkpoint menurut validation macro-F1 adalah *implementation assumptions*. Tidak ada Optuna ulang. Model tetap kandidat reproduksi parsial: 5.282.101 parameter vs 7.262.900 pada Tabel 1 paper. Perbandingan angka paper dan V1/V2 yang membedakan test dari validation ada di §6.8 notebook. Dua hasil final test V1 di atas tetap historis dan tidak diubah.

## Menjalankan eksperimen secara manual di GPU KC

Semua edit dilakukan di laptop lokal. Setelah notebook direview dan disalin ke server dengan cara manual yang dipilih peneliti, jalankan hanya kode notebook biasa; jangan memasang atau menjalankan Codex/agentic tools di server. Dari terminal server, folder kerja awal:

~~~bash
cd /mnt/home/probation-03/Kel_13
export CAD_VIT_DATA_ROOT=/path/ke/CAD_ViT/datasets
export CAD_VIT_RUN_ROOT=/mnt/home/probation-03/Kel_13/cad_vit_runs
export CAD_VIT_WANDB_PROJECT=CAD-ViT-reproduction
~~~

Ganti `CAD_VIT_DATA_ROOT` dengan lokasi dataset **yang benar-benar ada** dan berisi `APTOS2019/` serta `DDR/`. `CAD_VIT_RUN_ROOT` harus berada di luar repository; manifest, checkpoint, dan log W&B akan disimpan di sana. Jalankan Jupyter dari direktori clone repository yang sebenarnya di bawah `Kel_13` (nama direktori clone perlu diperiksa manual), lalu buka `CAD_ViT.ipynb`. Jangan jalankan contoh training sebelum manifest data nyata terbentuk dan diperiksa.

Untuk APTOS, gunakan `prepare_manifest("APTOS2019")` setelah preflight sukses. Untuk DDR, gunakan `prepare_manifest("DDR")` setelah direktori DR_grading resmi tersedia; CSV mapping manual tidak diperlukan. Fungsi ini mempertahankan 12.522 gambar berlabel 0–4, mencatat keanggotaan split train/valid/test resmi dalam manifest/metadata, lalu membuat split reproduksi 8:1:1 dengan seed 42 dan grup SHA-256 agar gambar identik tidak melintasi split. Manifest disimpan di luar repo dan tidak ditimpa bila fingerprint berubah.

Target paper ialah 512×512, LR 0,000186, batch 16, 80 epoch, dan attention reduction ratio 16 (§IV.B). Pada RTX 3060 12 GB, opsi `physical_batch_size=4, accumulation=4, amp=True` mempertahankan batch efektif nominal 16, tetapi bukan eksekusi yang identik dengan batch fisik 16. Catat setiap deviasi dalam konfigurasi run. Contoh pemanggilan dan smoke test data nyata ada di bagian 5 notebook. Evaluasi test dipanggil terpisah setelah pemilihan model berdasarkan validation set selesai.

Untuk V2, jalankan cell definisi secara berurutan di notebook. Contoh berikut adalah **perintah Python di cell Jupyter**, belum dijalankan pada data server:

~~~python
aptos_manifest_path = prepare_manifest("APTOS2019")
ddr_manifest_path = prepare_manifest("DDR")
preview_v2_preprocessing(aptos_manifest_path, "APTOS2019")
preview_v2_preprocessing(ddr_manifest_path, "DDR")
smoke_v2_real_batch(aptos_manifest_path, "APTOS2019")
smoke_v2_real_batch(ddr_manifest_path, "DDR")
run_experiment_v2(v2_experiment_config("APTOS2019"), aptos_manifest_path,
                  V2_RUN_NAMES["APTOS2019"])
run_experiment_v2(v2_experiment_config("DDR"), ddr_manifest_path,
                  V2_RUN_NAMES["DDR"])
~~~

Contoh pemanggilan di atas adalah dokumentasi reproduksi; kedua run V2 yang tercatat telah selesai, sehingga jangan menjalankannya ulang dengan nama run yang sama. <code>run_experiment_v2</code> default tidak melakukan resume, dan hanya menerima nama run V2 yang baru. Bila sebuah run V2 telah ada, fungsi menolak overwrite. Resume harus dipilih manual dengan <code>resume=True</code> dan record manifest/config yang sama; <code>last.pt</code> V1 DDR tidak dipakai. Jika VRAM tidak cukup untuk batch fisik 16, hentikan dan dokumentasikan deviasi sebelum mengubah konfigurasi; jalur V2 final sengaja mengunci batch fisik 16. Final test V2 belum dilakukan; verifikasi checkpoint DDR terlebih dahulu dan pertahankan test set terpisah dari pemilihan model. Fungsi <code>evaluate_test()</code> lama masih mengunci indeks epoch 39 sehingga tidak cocok untuk run 65/30; prosedur baru dengan guard <code>RUN_V2_FINAL_TEST = False</code> ada di §6.7 notebook. Jangan mengaktifkan guard sebelum checkpoint dan manifest ditinjau.


Dependensi runtime tambahan yang perlu dipastikan tersedia: `torchvision` yang cocok dengan versi PyTorch, `Pillow`, `numpy`, dan `jupyter`; `torch`, `pandas`, `opencv-python`/`opencv-python-headless`, `scikit-learn`, `matplotlib`, `seaborn`, serta `wandb` juga dipakai notebook. Jangan menyimpan token W&B di notebook atau repository. Mode W&B default `offline`; mode `online` memerlukan autentikasi melalui environment atau mekanisme login manual yang aman.
