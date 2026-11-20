# Audit konsolidasi tiga studi (20 November 2026)

Audit ini dibuat dari `origin/salma` commit `53007608e06601b1208e38229114fa581c32e78c`, pada worktree terpisah. Checkout `salma` Windows yang memiliki perubahan lokal tidak diubah. Ini adalah inventaris **bukti yang tersedia dalam Git**, bukan klaim bahwa data, checkpoint, atau run HPC telah diaudit ulang.

## Peta studi dan sumber otoritatif

| Studi | Implementasi dan notebook | Bukti yang tersedia | Batas reproduksibilitas |
| --- | --- | --- | --- |
| 01 CAD-ViT V1/V2 | Kode historis dan output pada [`reproduction/archive/CAD_ViT_original_105_cells.ipynb`](../reproduction/archive/CAD_ViT_original_105_cells.ipynb); tiga [laporan eksperimen](../reproduction/experiments/); modul kandidat terkini pada [`reproduction/cadvit/`](../reproduction/cadvit/) | Confusion matrix V1 di arsip; dua JSON final test V2; tabel 80 epoch V2 di arsip | Kode kandidat terkini tidak boleh dilabeli sebagai source persis setiap run historis. Perbedaan parameter kandidat 5.282.101 terhadap angka paper sekitar 7,26 juta, serta asumsi split/preprocessing/arsitektur, membuat reproduksi ini parsial. |
| 01 balanced/recovery | [`reproduction/cadvit/experiments.py`](../reproduction/cadvit/experiments.py) dan [laporan ketiga](../reproduction/experiments/03_experiment_balanced_training_and_ddr_recovery.ipynb) | APTOS dan DDR masing-masing 80 record JSONL; ringkasan dan confusion matrix final test. DDR recovery memakai protokol berbeda pada epoch 1–42 dan 43–80. | Manifest beku dan checkpoint biner berada di luar Git; source evaluator DDR diarsipkan sebagai source terkait run, bukan bukti kriptografis eksekusinya. |
| 02 ConvNeXt B0–B4 | [`novelty/foundation_adaptation.py`](../novelty/foundation_adaptation.py) untuk B0–B2; [bukti B0–B4](../novelty/convnext_b0_b4/validation_evidence.json) | Sepuluh run × 20 epoch = 200 objek history asli; 10 kurva SVG, 10 matriks PNG; best validation per-grade/QWK. B4-Q/R hanya APTOS. | Source eksekusi historis B3/B4 dan checkpoint biner tidak ada di Git. Pencarian nama file pada seluruh riwayat Git tidak menemukan runner/sampler B3/B4. Jangan menganggap runner B0–B2 mereproduksi B3/B4. Semua angka ini **validation only**. |
| 03 ConvNeXt E5 | [`novelty_integrated/e5/`](../novelty_integrated/e5/) dan notebook [03](../notebooks/03_ConvNeXt_E5_APTOS2019.ipynb) | APTOS integrated: `run.json`, `protocol.json`, 20 baris `history.csv`, `summary.json`, kurva, matriks validation, dan [`test_summary.json`](../results/APTOS2019/e5_integrated/test_summary.json). Best validation epoch 16: macro-F1 0,7057914318295222. Frozen final test 366 baris: macro-F1 0,6404563230418223. | Empat arm diimplementasikan, tetapi hanya APTOS integrated yang memiliki hasil run terkomit. Checkpoint/pretrained snapshot/manifests dan gambar berada di luar Git. Menurut laporan operator 20 November 2026, DDR E5 masih **TRAINING_IN_PROGRESS** pada epoch 9; artefak aktifnya tidak tersedia di checkout ini. DDR final test **FINAL_TEST_NOT_RUN**. |

## Pemeriksaan numerik pada artefak terkomit

- B0–B4: sepuluh objek run masing-masing memiliki 20 epoch; best macro-F1 APTOS B0/B1/B2/B3/B4-Q/B4-R berturut-turut 0,673686/0,701124/0,659987/0,700624/0,688319/0,707897; DDR B0/B1/B2/B3 0,619817/0,652146/0,623084/0,651006. Ini nilai **validation**.
- APTOS E5: matriks validation berjumlah 366 dengan diagonal 304; matriks final test berjumlah 366 dengan diagonal 302. Grade 3 final test 4/19 benar. `summary.json.final_test_status=NOT_ACCESSED` merekam akhir training; `test_summary.json.status=FINAL_TEST_COMPLETE` merekam evaluasi sesudahnya. Kedua file tetap asli.
- CAD-ViT balanced: APTOS dan DDR masing-masing memiliki 80 epoch berurutan; matriks final test masing-masing berjumlah 366 dan 1.252. Nilai validation tidak dipakai sebagai pengganti nilai test.

## Kondisi notebook sebelum refactor ini

[`CAD_ViT.ipynb`](../CAD_ViT.ipynb) adalah laporan induk sembilan sel Markdown, bukan notebook training. Tiga laporan `reproduction/experiments/` juga hanya Markdown (8/11/11 sel). Delapan notebook tematik pada `reproduction/notebooks/` memiliki kode analisis dan sebagian output tersimpan. Tiga notebook studi di `notebooks/` hanya 6/5/9 sel dengan 2/2/3 sel kode, belum memuat alur setup, preflight, konfigurasi, training terjaga, dan evaluasi lengkap yang diminta. Kode notebook yang membaca artefak dapat berjalan tanpa GPU; training aktual bergantung pada manifest, gambar, snapshot resmi, dan output root di luar Git. Perintah lama dengan path KC bukan konfigurasi Colab universal.

## Kebijakan artefak dan status yang belum tersedia

- Manifest beku APTOS (`73d1546e3c2fd2bf9e3d0fd10af914cb523d28e4bc8e1f25212af454d4be8854`) dan DDR (`7bc3e3916e012bc46fcd9cd69aa469e0daf3229e1273f1d8ecb6054899bf6eb1`) dirujuk oleh kode/evidence, tetapi CSV dan gambar tidak ada di Git. Pemeriksaan SHA gambar pada host eksekusi tetap wajib.
- Checkpoint `.pt`, snapshot `facebook/convnext-tiny-224`, dan cache input tidak ada di Git. Metadata dan hash dalam JSON tidak menggantikan pemeriksaan biner saat deployment.
- Gambar fundus APTOS yang hak redistribusinya belum dikonfirmasi tetap lokal dan diabaikan Git. Gambar DDR yang beratribusi ada di `reproduction/report_figures/`; lihat README pada folder itu.
- Laporan operator 20 November 2026: DDR E5 `run_retry1` di HPC menyelesaikan epoch 8 dan sedang mengerjakan epoch 9 (laporan terakhir 67/314 batch); `summary.json` belum ada. Ini **TRAINING_IN_PROGRESS**, bukan hasil final yang telah diverifikasi di workspace ini. `FINAL_TEST_NOT_RUN` tetap berlaku sampai laporan evaluasi sah tersedia. Jangan mengganggu training atau menjalankan evaluator hanya untuk mengisi notebook.
- Tidak ada notebook yang boleh menganggap hasil B3/B4 historis sebagai hasil runner yang baru direkonstruksi. Percobaan 50 epoch, arm E5 lainnya, dan validasi klinis independen belum memiliki hasil terkomit.

## Tingkat reproduksibilitas

| Bagian | Status | Syarat yang belum terpenuhi |
| --- | --- | --- |
| Pembacaan metrik/kurva/matriks terkomit | **Dapat direproduksi tanpa GPU** | Python, Jupyter, dan dependensi visualisasi ringan. |
| CAD-ViT kandidat atau ConvNeXt B0–B2/E5 dari awal | **Sebagian dapat direproduksi** | Data berizin, manifest beku, snapshot/checkpoint yang diverifikasi, environment cocok, dan protokol run yang dikunci. Hasil bitwise lintas hardware tidak dijanjikan. |
| B3/B4 sebagai run historis yang identik | **Belum dapat direproduksi dari Git saja** | Source eksekusi eksternal, selection artifacts, serta metadata/cache/checkpoint asli harus ditransfer dan diaudit. |
| DDR E5 training/final test | **TRAINING_IN_PROGRESS menurut operator; belum terverifikasi dari artefak lokal** | Artefak `run_retry1` HPC dan, untuk final test, protokol evaluasi terkunci serta laporan sah. |

## Validasi refactor di worktree Windows

- Tiga notebook utama lulus `nbformat.validate` dan kompilasi AST semua sel kode. Semua 38 sel kode yang aman untuk pembacaan artefak telah dieksekusi tanpa dataset eksternal, GPU, training, atau inference. Sel aksi tetap dimatikan. Figure dan tautan relatif yang dipakai notebook/README ada pada checkout.
- Uji baru `test_research_notebooks.py` memeriksa 160 baris tabel epoch CAD-ViT terhadap JSONL, 200 objek epoch B0–B4, hash 20 figure B0–B4, matriks, status validasi/final test E5, syntax notebook dan tautan. Seluruh enam tes baru lulus. Tujuh tes laporan historis dan dua tes notebook tematik juga lulus; source model/preprocessing serta arsip 105 sel tidak berubah.
- Regresi E5 (`test_e5_integrated.py`, 13 tes), input E5 (`test_e5_input_pipeline.py`, enam tes), dan foundation (`test_foundation_adaptation.py`, 11 tes) lulus pada CPU sintetis. Ini **bukan** training pada dataset penelitian.
- `unittest discover` seluruh repository menjalankan 122 tes: dua import-error pada modul pilot CAD-ViT karena `fcntl` khusus POSIX tidak tersedia di Windows dan `torchvision` tidak terpasang di `.venv` lokal; dua tes diskip. Kegagalan lingkungan ini tidak ditangani dengan perubahan source, dan jalur pilot/cache tersebut perlu diuji ulang pada environment Linux yang sesuai sebelum digunakan.
- Full-image preflight, snapshot resmi, checkpoint SHA biner, serta kemungkinan run dari Colab/HPC **belum** diuji di host ini karena artefak eksternal tidak tersedia. Figure/angka yang dilaporkan berasal dari bukti agregat terkomit, bukan inferensi baru.
