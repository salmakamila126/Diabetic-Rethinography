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

Tabel lengkap metrik, confusion matrix kedua dataset, dan grafik training APTOS2019 berada di bagian 5-6 <code>CAD_ViT.ipynb</code>. Grafik memakai tepat 43 catatan epoch 1-43 dari log APTOS2019; log tidak membuktikan penyelesaian 80 epoch meskipun nama run memuat <code>80ep</code>. Log training DDR per epoch tidak tersedia, sehingga kurva DDR tidak dibuat. Penurunan validation macro-F1 APTOS2019 tampak pada log, tetapi penyebabnya belum terbukti. Numerical instability DDR yang dilaporkan tidak dapat didiagnosis dari JSON test saja.

Model tetap kandidat reproduksi parsial: 5.282.101 parameter versus 7.262.900 pada Tabel 1 paper (r=16), selisih 1.980.799. AdamW, WeightedRandomSampler, konfigurasi internal arsitektur, dan pemilihan checkpoint dari validation macro-F1 adalah *implementation assumptions*; proses Optuna paper tidak direproduksi. Agregasi F1 paper belum dapat dipastikan identik dengan macro-F1 lima kelas notebook. JSON dan log sumber tetap di luar repository; notebook menyimpan angka, matriks, kurva, serta SHA-256 sumber untuk audit.

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

Dependensi runtime tambahan yang perlu dipastikan tersedia: `torchvision` yang cocok dengan versi PyTorch, `Pillow`, `numpy`, dan `jupyter`; `torch`, `pandas`, `opencv-python`/`opencv-python-headless`, `scikit-learn`, `matplotlib`, `seaborn`, serta `wandb` juga dipakai notebook. Jangan menyimpan token W&B di notebook atau repository. Mode W&B default `offline`; mode `online` memerlukan autentikasi melalui environment atau mekanisme login manual yang aman.
