# Notebook penelitian CAD-ViT dan ConvNeXt

Gunakan notebook sesuai model, dataset, dan eksperimen. Tabel statis bersumber dari artefak terkomit. Confusion matrix yang tersedia pada notebook ConvNeXt sudah tersimpan sebagai output gambar dan dapat dibaca tanpa menjalankan kernel.

## Reproduksi CAD-ViT

[CAD-ViT Reproduction](reproduction/CAD_ViT_Reproduction.ipynb) mendokumentasikan kandidat reproduksi, V1, V2 BG10, balanced APTOS, DDR recovery, dan pemisahan validasi dari final test. [Arsip asli 105 sel](../reproduction/archive/CAD_ViT_original_105_cells.ipynb) dan [tiga laporan kampanye](../reproduction/experiments/) tetap menjadi sumber historis. Prototipe P1–P4 memiliki [protokol dan snapshot arsip tersendiri](../docs/CADVIT_NOVELTY_PROTOCOL.md).

## Baseline ConvNeXt-Tiny B0–B4

| Eksperimen | APTOS2019 | DDR | Bukti |
| --- | --- | --- | --- |
| B0 vanilla | [Notebook](baselines/B0_APTOS2019.ipynb) | [Notebook](baselines/B0_DDR.ipynb) | Dua run 20 epoch, ordinary CE. |
| B1 augmentasi | [Notebook](baselines/B1_APTOS2019.ipynb) | [Notebook](baselines/B1_DDR.ipynb) | Dua run 20 epoch, augmentasi konservatif train-only. |
| B2 weighted CE | [Notebook](baselines/B2_APTOS2019.ipynb) | [Notebook](baselines/B2_DDR.ipynb) | Dua run 20 epoch, bobot kelas dari train. |
| B3 sampling grade 3 | [Notebook](baselines/B3_APTOS2019.ipynb) | [Notebook](baselines/B3_DDR.ipynb) | Dua run 20 epoch, runner historis eksternal. |
| B4 seleksi data | [B4-Q dan B4-R](baselines/B4_APTOS2019.ipynb) | [Mengapa tidak berlaku](baselines/B4_DDR.ipynb) | Dua run APTOS 20 epoch, nol run DDR. |

[Bukti agregat B0–B4](../novelty/convnext_b0_b4/validation_evidence.json) memuat 200 epoch. B0–B2 menyediakan alur opsional untuk run baru melalui modul resmi. B3/B4 merekam metode dan bukti yang tersedia, termasuk batas reproduksi akibat runner eksternal.

## Eksperimen E5–E7

| Eksperimen | APTOS2019 | DDR | Bukti |
| --- | --- | --- | --- |
| E5 Integrated | [Notebook](novelty/E5_APTOS2019.ipynb) | [Notebook](novelty/E5_DDR.ipynb) | Dua run 20 epoch dengan validasi terverifikasi. Final test historis APTOS memiliki [laporan terpisah](../results/APTOS2019/e5_integrated/README.md). |
| E6 Boundary-Aware | [Notebook](novelty/E6_APTOS2019.ipynb) | [Notebook](novelty/E6_DDR.ipynb) | Dua run 20 epoch dengan validasi terverifikasi. |
| E7 Global/Local | [Notebook](novelty/E7_APTOS2019.ipynb) | [Notebook](novelty/E7_DDR.ipynb) | Protokol dan runner tersedia. Hasil dan status HPC saat ini belum terverifikasi dalam checkout ini. |

[Laporan E5–E6](../docs/E5_E6_VALIDATION.md) menyediakan 80 epoch dan indeks figur publikasi. [Inventaris eksperimen](../docs/NOTEBOOK_EXPERIMENT_AUDIT.md) menghubungkan source, checksum, bukti, dan snapshot notebook yang sudah dipensiunkan. [Audit backend dan pruning](../docs/OPTIMIZATION_COMPATIBILITY.md) membedakan catatan optimasi historis pada branch `novelty` dari rancangan evaluasi berikutnya di `salma`.

## Membaca dan mereproduksi

- Bukti statis memerlukan browser saja. Untuk mengeksekusi sel pembacaan artefak secara lokal, gunakan Jupyter/IPython dengan Python 3.11 atau lebih baru.
- Untuk workflow opsional, gunakan lingkungan dan versi dependensi pada protokol terkait. Atur `DR_RESEARCH_ROOT` bila notebook dijalankan dari luar checkout, `DR_ARTIFACT_ROOT` untuk manifest dan data berizin, serta `DR_CONVNEXT_SNAPSHOT` untuk snapshot lokal. Notebook ConvNeXt memakai `DR_NOTEBOOK_RUN_DIR` sebagai area output baru di luar Git. Notebook reproduksi memakai pengaturan path sendiri yang dijelaskan di dalamnya.
- Sakelar opsional mati secara default. Pilih satu tahap yang didukung, aktifkan sakelarnya, dan gunakan cache serta output sesuai protokol. Untuk E7 gunakan hanya path keluaran baru di bawah `foundation_e7` dan periksa kuota serta alokasi GPU sebelum workflow opsional. Sel notebook tidak menawarkan final-test inference, pruning, atau export.
- Untuk meregenerasi 16 notebook dataset dari bukti terkomit, jalankan `python notebooks/build_experiment_notebooks.py` dari root checkout dengan `nbformat` terpasang. Generator membaca JSON dan figur yang sudah ada, tanpa menjalankan model.
