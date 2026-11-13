# CAD-ViT Gambar 1: audit preprocessing

Sumber utama ialah PDF [CAD-ViT, IEEE Access 2025](https://doi.org/10.1109/ACCESS.2025.3572455), §III.A dan Gambar 1. PDF berada di luar repository. Bagian tersebut menjelaskan crop wilayah hitam, resize menjadi 512×512, dan Gaussian blur; area interpolation dibahas, tetapi pemanggilan OpenCV dan seluruh angka parameternya tidak diberikan. Gambar 1 membandingkan fundus asli dan hasil olahan. Kemiripan visual saja tidak membuktikan kesamaan kode atau parameter.

## Tampilan dan provenance

| Tampilan | Implementasi | Interpretasi |
| --- | --- | --- |
| `original` | RGB dari gambar sumber setelah orientasi EXIF diterapkan | Acuan tampilan; berkas mentah tidak diubah. |
| `paper_described` | Operasionalisasi V1 historis untuk urutan crop → resize → blur dari paper | Langkah eksplisit paper ditambah asumsi notebook; sengaja identik per piksel dengan `historical_v1`. |
| `historical_v1` | `preprocess_fundus()` dari `reproduction/cadvit/preprocessing.py` | Ekstraksi tepat dari fungsi notebook historis. |
| `historical_v2_bg10` | `preprocess_fundus_v2()` dari modul yang sama | V1 diikuti `clip(4I − 4GaussianBlur(I, σ=10) + 128, 0, 255)` pada float32. Ini **EXPERIMENTAL EXTENSION**, bukan klaim metode paper. |

Kode historis menangani orientasi EXIF, konversi RGB, crop dengan ambang grayscale **10**, resize **512×512** memakai `cv2.INTER_AREA`, dan Gaussian blur **5×5, σ=1**. Ukuran target dan urutan operasi dijelaskan pada CAD-ViT §III.A; pilihan angka lain serta pemanggilan pustaka merupakan **IMPLEMENTATION ASSUMPTION**. Lihat [matriks fidelity preprocessing](PAPER_FIDELITY_MATRIX.md).

## Menjalankan notebook dengan aman

Buka `reproduction/notebooks/preprocessing_analysis.ipynb`. Isi `CAD_VIT_AUDIT_IMAGE` dengan path gambar fundus yang dipilih, dan `CAD_VIT_AUDIT_SPLIT` dengan `train` atau `audit`. Untuk `train`, isi juga `CAD_VIT_AUDIT_MANIFEST` serta `CAD_VIT_AUDIT_DATASET_ROOT`; notebook memeriksa bahwa gambar hanya berada pada baris manifest `split=train`. Untuk `audit`, pilih gambar independen yang bukan anggota validation/test. Tanpa manifest, keanggotaan gambar audit independen tidak dapat dibuktikan otomatis. Notebook menolak split test/validation yang dinyatakan dan tidak menjalankan training atau evaluasi test.

`CAD_VIT_AUDIT_OUTPUT` dapat menunjuk direktori output. Secara default, berkas turunan berada dalam `preprocessing_audit_outputs/` yang diabaikan Git:

- Empat varian PNG dan `comparison.png`.
- `rgb_histograms.png` serta `rgb_histograms.csv` (256 hitungan per kanal RGB dan tampilan).
- `metrics.csv` (satu baris per tampilan, termasuk SHA-256 gambar sumber, deklarasi split, dan orientasi EXIF).

Gambar input hanya dibaca. Berkas rusak menghasilkan error yang jelas. Gambar seluruhnya gelap tetap memunculkan error crop historis; audit tidak membuat hasil olahan palsu. Orientasi EXIF dicatat dalam hasil dan diterapkan pada tampilan asli maupun kedua pipeline historis.

## Definisi metrik

- **Brightness:** rerata `0.299R + 0.587G + 0.114B`, kanal 0–255.
- **Contrast:** simpangan baku populasi (`ddof=0`) dari luminansi yang sama.
- **Jarak rerata RGB dari gambar asli:** jarak Euclidean antara dua vektor rerata RGB global dalam satuan nilai kanal; crop dapat mengubah angka ini tanpa transformasi warna.
- **Pixel MAE dari V1:** rerata selisih absolut piksel RGB terhadap `historical_v1`, hanya untuk tampilan olahan 512×512 yang sejajar; gambar asli tidak memiliki nilai ini.
- **Histogram RGB:** jumlah piksel pada setiap intensitas integer 0–255 per kanal, tanpa normalisasi densitas.

Setiap panel histogram PNG memakai maksimum hitungannya sendiri demi keterbacaan; gunakan CSV untuk membandingkan hitungan absolut antarpanel. Interpretasikan penghilangan border, smoothing, dan kemungkinan halo V2 hanya dari gambar yang benar-benar dipilih. Satu gambar atau histogram serupa tidak membuktikan detail lesi terjaga, performa model setara, ataupun preprocessing identik dengan penulis. Hasil audit gambar nyata tidak diklaim sebelum gambar yang diizinkan dijalankan.

## Inventaris visualisasi reproduksi — 10 November 2026

Audit ini memeriksa **output yang tersimpan**, bukan hanya keberadaan fungsi pembuat grafik. `TERSEDIA` berarti gambar/tabel telah tersimpan; `KODE SAJA` berarti sel pembaca/plot belum memiliki output kampanye yang sesuai; `PENDING` berarti bukti data yang diperlukan belum ada. Semua hasil historis dan angka paper tetap terpisah.

| Kategori | Lokasi dan sumber | Status output serta batas interpretasi |
|---|---|---|
| EDA distribusi kelas dan karakteristik citra | `CAD_ViT.ipynb` §2.2–2.3 | **TERSEDIA:** grafik historis APTOS/DDR dan grafik lokal APTOS dari 3.662 pasangan label–gambar; statistik citra memakai subset 60 gambar. Output historis §2.2 belum ditautkan ke manifest V2; subset bukan estimasi seluruh dataset. Distribusi kelas DDR lokal yang setara **PENDING**. |
| Citra fundus dan preprocessing | `CAD_ViT.ipynb` §2–3; `01_preprocessing.ipynb`; `preprocessing_analysis.ipynb` | **TERSEDIA:** contoh historis dan diagram tahap preprocessing utama. Notebook 01 masih **KODE SAJA** untuk gambar train/validation terverifikasi. Audit gambar terpisah yang masih memiliki perubahan lokal belum menjadi bukti keanggotaan train tanpa manifest; jangan menganggap gambar audit sebagai test/training. |
| Arsitektur | `CAD_ViT.ipynb` §4; `02_architecture.ipynb` | **TERSEDIA:** diagram utama sembilan blok dan output bentuk/parameter sintetis; diagram adalah interpretasi kode dan Gambar 3, bukan bukti dimensi internal penulis identik. |
| Riwayat V1 dan V2 historis | `CAD_ViT.ipynb` §5.5, §5.8 | **TERSEDIA:** dua tabel 80 baris dan kurva train loss, validation loss, validation accuracy, validation macro-F1 untuk APTOS/DDR V2 historis sampai epoch 80. Angka log 80+80 terpisah dari V1 dan V2 65/30. Kurva DDR memiliki tiga fase recovery dan loss antarfase tidak homogen. |
| APTOS `balanced_v1` | `results/APTOS2019/full_oversampling_amp_fp16_80epochs/` dan `CAD_ViT.ipynb` §5.7.1/§6.9 | **TERSEDIA sebagai data, belum sebagai PNG:** JSONL/CSV memuat 80 epoch, best validation epoch 24; ringkasan final test dan matriks angka N=366 tersimpan. Sel pembaca kampanye belum menyimpan grafik train/validation maupun heatmap test; jangan menyebutnya gambar yang sudah dirender. |
| DDR kampanye baru | `CAD_ViT.ipynb` §5.7.1 | **PENDING:** tabel status berasal dari laporan terminal, bukan history/checkpoint lokal. AMP interrupted, pilot baseline/capped, dan full FP32 terpisah; tidak ada kurva 80 epoch kampanye baru yang dapat diverifikasi. |
| Matriks kebingungan V1 | `CAD_ViT.ipynb` §6.3 | **TERSEDIA:** dua heatmap APTOS2019 dan DDR berlabel **V1 Historical Final Test**, angka, sumbu, dan colorbar tersimpan. Ini bukan validation atau V2. |
| Matriks DDR capped pilot | `CAD_ViT.ipynb` §6.3.1 | **TERSEDIA tetapi dilaporkan:** heatmap validation epoch 3 dari laporan peneliti; bukan final test. Artefak sumber lokal **PENDING**. |
| Referensi paper untuk klasifikasi dan balancing | `04_classification_evaluation.ipynb`, `06_balancing_analysis.ipynb` | **TERSEDIA:** tiga heatmap Gambar 4–5 dan dua diagram batang Gambar 6 bertanda **PAPER_REFERENCE**. Orientasi paper pada Gambar 5: baris prediksi, kolom label sebenarnya; evaluator kami sebaliknya. Gambar ini bukan prediksi run kami. |
| Per kelas dan perbandingan metrik | `CAD_ViT.ipynb` §6.8–6.9; JSON final test APTOS | **TERSEDIA sebagai tabel:** F1 per kelas APTOS `balanced_v1`, macro/weighted F1, dan matriks angka. Grafik per kelas baru belum dibuat; tabel sudah cukup untuk membaca nilainya. F1 paper berbobot support tetap **INFERRED FROM PAPER**. |
| Ablation dan reduction ratio | `05_ablation_and_reduction.ipynb`; dokumen rencana Tabel 1–3 | **TERSEDIA:** parameter dan estimasi MAC dari kode serta angka paper sebagai referensi. Accuracy/specificity varian baru **PENDING**; tidak ada kurva eksperimen ablation yang sah. |
| Analisis galat dan Grad-CAM | `07_error_analysis.ipynb`, `08_gradcam.ipynb` | **KODE SAJA/PENDING** untuk gambar fundus nyata: prediksi per gambar dan checkpoint tepercaya belum tersedia lokal. Output Grad-CAM yang tersimpan adalah **SYNTHETIC_TEST** CPU, bukan bukti klinis. |

**Pemeriksaan keterbacaan.** Heatmap V1 utama memiliki label dan skala warna; timeline tiga fase DDR dan diagram sembilan blok tersimpan. Grafik validation V2 menampilkan rentang epoch 1–80. Sejumlah label Inggris pada output historis tetap merupakan bagian gambar tersimpan; caption Indonesia menjelaskan sumber dan split. Jangan mengganti gambar historis dengan render ulang tanpa data dan kode yang sama. Heatmap APTOS `balanced_v1` **belum** tersimpan sebagai PNG meski matriks final test tersedia; matriks numerik di §6.9 cukup sebagai bukti awal yang dapat diaudit. Grafik hasil kampanye baru lain menunggu artefak sumber yang benar.
