# Tiga studi penelitian — urutan baca dan eksekusi

| Urutan | Notebook utama | Bukti dan cakupan |
| --- | --- | --- |
| 01 | [Reproduksi CAD-ViT](01_CAD_ViT_Reproduction.ipynb) | V1, V2 BG10, balanced APTOS, DDR recovery; hasil validasi dan final test asli, reproduksi parsial terhadap paper. |
| 02 | [ConvNeXt B0–B4](02_ConvNeXt_B0_B4_Validation.ipynb) | Sepuluh run 20 epoch, 200 catatan epoch, kurva dan matriks validasi. B3/B4 historical runner berada di luar Git. |
| 03 | [ConvNeXt E5](03_ConvNeXt_E5_APTOS2019.ipynb) | Empat arm terimplementasi; APTOS integrated 20 epoch dan satu final test selesai. DDR integrated masih training di HPC; final test belum dijalankan. |

**Tindak lanjut Studi 03:** [E6 boundary-aware ConvNeXt-Tiny](03_ConvNeXt_E6_Boundary_Aware.ipynb) mendokumentasikan objective LDAM-DRW, supervisi ordinal, batas grade berdekatan, cache RGB train, dan gerbang benchmark. Status E6 **IMPLEMENTED, NOT RUN**; seluruh kontrol eksekusi notebook nonaktif secara default. [Protokol dan perintah HPC](../novelty_integrated/e6/README.md) memisahkan cache, preflight, smoke, benchmark, dan training dari E5.

Ketiga notebook memuat Markdown ilmiah dan kode Python yang membaca artefak terkomit, memeriksa dependensi/provenance, serta menunjukkan jalur source asli untuk preflight, smoke, dan training. `RUN_PREFLIGHT`, `RUN_SMOKE`, dan `RUN_TRAINING` selalu `False` saat notebook dibuka. Tidak ada sel yang menjalankan final-test inference. Figure terkomit tampil langsung di GitHub tanpa kernel. Kode yang memerlukan dataset, manifest, snapshot model, atau checkpoint luar repository menampilkan `SKIPPED_MISSING_DATA` sampai input asli tersedia.

## Persiapan Colab atau HPC

1. Clone branch `salma`, buka notebook berurutan dengan Python 3. Untuk inspeksi hasil, cukup Python, Jupyter/IPython dan `nbformat`; Matplotlib opsional karena figure asli sudah tersemat. Untuk model/training gunakan environment sumber yang kompatibel dengan PyTorch, torchvision, Transformers **4.57.3** (E5), Pillow, NumPy, serta dependensi modul terkait. Belum ada lockfile lintas Colab/HPC; catat versi paket dan perangkat aktual sebelum membandingkan run.
2. Set `DR_RESEARCH_ROOT` ke checkout, `DR_ARTIFACT_ROOT` ke root artefak berizin, dan `DR_OUTPUT_ROOT` ke direktori run baru **di luar Git**. Di Colab, Drive mount bersifat opsional dan perlu diaktifkan secara eksplisit.
3. Sediakan manifest seed-42 dan sidecar asli, gambar train/validation yang sah, serta snapshot resmi dan SHA-256 yang cocok. Notebook tidak mengunduh data/weights. Jalankan preflight sumber sebelum mengaktifkan smoke atau training. Jangan arahkan output ke run historis, checkpoint aktif DDR, atau dataset mentah.
4. Final-test APTOS E5 sudah terkunci dan hanya dibaca dari [`test_summary.json`](../results/APTOS2019/e5_integrated/test_summary.json). Hasil DDR E5 tidak boleh dipublikasikan sampai training selesai dan artefak lengkap diaudit.

Untuk riwayat yang tidak dapat direkonstruksi dari Git saja, baca [audit konsolidasi](../docs/RESEARCH_CONSOLIDATION_AUDIT.md), [arsip CAD-ViT 105 sel](../reproduction/archive/CAD_ViT_original_105_cells.ipynb), dan [panduan kolaborator](../docs/COLLABORATOR_GUIDE.md). Gambar fundus APTOS lokal yang belum memiliki izin redistribusi tetap di luar Git.
