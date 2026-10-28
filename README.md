# CAD-ViT: Coordinate Attention-Enhanced Vision Transformer With Dilated Feature Fusion for Diabetic Retinopathy Staging

Reproduksi penelitian [CAD-ViT (2025)](https://ieeexplore.ieee.org/document/11009024) dalam satu notebook utama, **CAD_ViT.ipynb**. Notebook mempertahankan EDA dan demonstrasi preprocessing partner. Output EDA yang tersimpan adalah hasil historis dari sumber data lama; jalankan ulang setelah validasi dataset resmi sebelum mengutipnya sebagai hasil reproduksi.

## Sumber dataset

| Dataset | Sumber resmi | Lokasi di bawah CAD_VIT_DATA_ROOT |
| --- | --- | --- |
| APTOS 2019 | [Kompetisi Kaggle APTOS 2019](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) | APTOS2019/ |
| OIA-DDR | [Repositori resmi OIA-DDR](https://github.com/nkicsl/DDR-dataset) | DDR/ |

Unduh dataset secara manual dari sumber resminya dan simpan **di luar repository**. Akses APTOS mungkin memerlukan login dan persetujuan aturan kompetisi. Notebook mencari tepat satu <code>train.csv</code> dan satu direktori <code>train_images</code> di dalam <code>APTOS2019/</code>, lalu memeriksa pasangan <code>id_code</code>–gambar dan <code>diagnosis</code> kelas 0–4. Gambar test kompetisi tidak dipakai untuk data berlabel. Jika berkas masih berupa arsip, ekstrak terlebih dahulu; notebook tidak mengunduh atau mengekstraknya.

Struktur internal OIA-DDR belum diasumsikan. Setelah ekstraksi, periksa berkas label, format kelas, dan lokasi gambar sebelum menambahkan pemetaan label DDR ke notebook. Keberadaan gambar saja tidak berarti DDR siap untuk training.

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

Bandingkan checksum bagian dengan checksum resmi apabila penerbit menyediakannya; hash lokal saja tidak membuktikan keaslian unduhan. Setelah ekstraksi, inspeksi lokasi gambar dan label yang sebenarnya. Notebook hanya menginventarisasi DDR dan tidak melakukan penggabungan atau ekstraksi ulang setiap run.

## Status implementasi

Notebook tetap mempertahankan seluruh EDA/preprocessing partner. Implementasi kandidat arsitektur, preprocessing eksperimen, validasi dan manifest split berkelompok 8:1:1, Dataset/DataLoader, training, checkpoint/resume, W&B, dan evaluasi test sudah tersedia sebagai kode. **APTOS dan DDR belum dijalankan sebagai eksperimen nyata** karena dataset belum tersedia di server. Jumlah parameter kandidat yang diuji sebelumnya adalah 5.693.621, sedangkan Tabel 1 paper melaporkan 7.262.900 pada reduction ratio 16. Detail internal yang tidak dijelaskan paper ditandai sebagai *implementation assumption* di notebook.

## Menjalankan eksperimen secara manual di GPU KC

Semua edit dilakukan di laptop lokal. Setelah notebook direview dan disalin ke server dengan cara manual yang dipilih peneliti, jalankan hanya kode notebook biasa; jangan memasang atau menjalankan Codex/agentic tools di server. Dari terminal server, folder kerja awal:

~~~bash
cd /mnt/home/probation-03/Kel_13
export CAD_VIT_DATA_ROOT=/path/ke/CAD_ViT/datasets
export CAD_VIT_RUN_ROOT=/mnt/home/probation-03/Kel_13/cad_vit_runs
export CAD_VIT_WANDB_PROJECT=CAD-ViT-reproduction
~~~

Ganti `CAD_VIT_DATA_ROOT` dengan lokasi dataset **yang benar-benar ada** dan berisi `APTOS2019/` serta `DDR/`. `CAD_VIT_RUN_ROOT` harus berada di luar repository; manifest, checkpoint, dan log W&B akan disimpan di sana. Jalankan Jupyter dari direktori clone repository yang sebenarnya di bawah `Kel_13` (nama direktori clone perlu diperiksa manual), lalu buka `CAD_ViT.ipynb`. Jangan jalankan contoh training sebelum manifest data nyata terbentuk dan diperiksa.

Untuk APTOS, gunakan `prepare_manifest("APTOS2019")` setelah preflight notebook sukses. Untuk DDR, ekstrak dan audit arsip sekali di luar notebook, inspeksi skema label, lalu buat CSV pemetaan **di luar repository** dengan kolom `image_id,label,image_path` dan bila tersedia `group_id` pasien; `image_path` relatif terhadap `DDR/`. Panggil `prepare_manifest("DDR", ddr_mapping_csv=...)`. Notebook tidak mengasumsikan nama folder atau format label internal DDR. Split disimpan agar training, ablation, dan evaluasi memakai pembagian yang sama.

Target paper ialah 512×512, LR 0,000186, batch 16, 80 epoch, dan attention reduction ratio 16 (§IV.B). Pada RTX 3060 12 GB, opsi `physical_batch_size=4, accumulation=4, amp=True` mempertahankan batch efektif nominal 16, tetapi bukan eksekusi yang identik dengan batch fisik 16. Catat setiap deviasi dalam konfigurasi run. Contoh pemanggilan dan smoke test data nyata ada di bagian 5 notebook. Evaluasi test dipanggil terpisah setelah pemilihan model berdasarkan validation set selesai.

Dependensi runtime tambahan yang perlu dipastikan tersedia: `torchvision` yang cocok dengan versi PyTorch, `Pillow`, `numpy`, dan `jupyter`; `torch`, `pandas`, `opencv-python`/`opencv-python-headless`, `scikit-learn`, `matplotlib`, `seaborn`, serta `wandb` juga dipakai notebook. Jangan menyimpan token W&B di notebook atau repository. Mode W&B default `offline`; mode `online` memerlukan autentikasi melalui environment atau mekanisme login manual yang aman.
