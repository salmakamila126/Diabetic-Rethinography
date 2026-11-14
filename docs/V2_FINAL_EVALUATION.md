# Persiapan final test V2 80 epoch

**Status dokumen:** ini adalah rencana preflight sebelum final test V2 80 epoch. Dua JSON final test kini tersedia lokal dan SHA-256-nya cocok dengan catatan server pada 2026-11-13; lihat [audit provenance V2](V2_FINAL_TEST_PROVENANCE.md) dan [laporan utama](../CAD_ViT.ipynb). Jangan jalankan ulang final test. Modul `reproduction/cadvit/evaluation.py` tetap jalur terpisah dari guard V2 65/30 epoch di [notebook historis](../reproduction/archive/CAD_ViT_original_105_cells.ipynb).

## Target dan provenance

| Dataset | Direktori checkpoint yang wajib | `best.pt` menurut validation | `last.pt` | Preprocessing |
|---|---|---:|---:|---|
| APTOS2019 | `checkpoints/aptos_v2_bg10_stable_80ep_seed42/` | Epoch 69, macro-F1 0.6571274188819457 | Epoch 80 | V2 BG10 |
| DDR | `checkpoints/ddr_v2_fp32_weighted_fp64clip_seed42/` | Epoch 72, macro-F1 0.6903234255707131 | Epoch 80 | V2 BG10 |

Nilai epoch/F1 berasal dari log training yang dirujuk di notebook §5.5; itu **validation**, bukan final test. APTOS V2 historis 65 epoch, DDR V2 historis 30 epoch, serta checkpoint V1 tidak boleh dipakai. Guard lama `evaluate_test()` di notebook masih mensyaratkan 40 epoch; `evaluate_test_v2_once()` menunjuk run 65/30. Keduanya sengaja tidak diubah dan tidak digunakan oleh modul baru.

Checkpoint asli `run_experiment_v2()` menyimpan `epoch` **zero-based**, `model` state dict, `optimizer`, `scaler`, `record`, `best_f1`, `best_loss`, dan RNG states. `record` memuat `pipeline_version=V2_BG10_STABLE`, `experiment`, konfigurasi preprocessing/model, `manifest_sha256`, dan `parameter_count`. Recovery DDR menyimpan record training asal beserta `recovery` metadata tambahan. Karena itu `experiment.amp=True` di record DDR tidak membuktikan seluruh epoch recovery memakai AMP: epoch 60–63 menggunakan weighted FP32, epoch 64–80 weighted FP32 dengan penghitungan gradient norm FP64. Kode dan log mendukung urutan tersebut, tetapi transkrip copy/hash checkpoint epoch 63 ke direktori FP64 tidak tersedia; preflight tidak mengklaim lineage transfer terbukti.

## Apa yang diperiksa preflight

`preflight` hanya membaca checkpoint terpercaya, CSV/JSON manifest, dan berkas log; **tidak membuka gambar test atau menjalankan model**. Ia mensyaratkan:

- nama run yang tepat, `best.pt`/`last.pt`, epoch 69/72 dan 80, serta validation macro-F1 yang dibekukan sebelum test;
- kecocokan `record` best/last, lima kelas, seed 42, 80 epoch, LR 0.000186, batch 16, tanpa sampler berbobot V2, konfigurasi `ARCH_CONFIG`, **5.282.101 parameter**, dan shape/finite seluruh tensor model;
- konfigurasi crop/resize/blur dan Ben Graham V2 BG10 yang dibaca dari `preprocessing.py` historis, bukan duplikasi parameter baru;
- SHA-256 berkas manifest terhadap checkpoint dan metadata manifest, dataset/root, tepat 3.662 APTOS2019 atau 12.522 DDR grading 0–4, rasio 8:1:1, kelima kelas di setiap split, ID/path unik, dan tidak ada `group_id` atau SHA gambar yang melintasi split;
- hash berkas log asli yang dicatat di notebook §5.5: satu log APTOS atau tiga fase DDR. Hash berbeda atau berkas hilang menghentikan preflight. `config.json` APTOS dan `recovery_config.json` DDR dicocokkan bila relevan;
- belum adanya `final_test_v2_80ep.json` atau `test_metrics.json` pada direktori run yang sama.

SHA-256 `best.pt`/`last.pt` yang dicetak preflight adalah **snapshot lokal saat pemeriksaan**. `run` mengecek ulang hash itu sebelum membaca gambar test agar artefak tidak berubah di sela kedua tahap. Hash tersebut bukan bukti transfer checkpoint DDR epoch 63 yang tidak tersedia.

Preflight **tidak** menghitung ulang SHA semua gambar; hash gambar pada manifest adalah bukti saat manifest dibuat. Jika dataset dipindahkan atau diubah setelah training, perlu audit berkas terpisah sebelum final test. Root dataset harus sama dengan metadata manifest agar pemindahan data tidak diterima diam-diam. `torch.load(weights_only=False)` dipakai karena checkpoint menyimpan state Python/NumPy; jalankan hanya untuk checkpoint proyek sendiri yang dipercaya, melalui flag eksplisit.

## Definisi evaluasi dan batas paper

Paper CAD-ViT, IEEE Access 2025, DOI 10.1109/ACCESS.2025.3572455, Bagian IV-A, Persamaan (7)–(11) mendefinisikan accuracy, precision, recall, F1, dan specificity. **Gambar 5** pada halaman IEEE 91511 menampilkan confusion matrix lima kelas: panel (a) DDR dan (b) APTOS-2019. Modul baru menghasilkan matriks 5×5 dengan **baris label aktual, kolom prediksi**, serta laporan per kelas. Accuracy = benar/total; precision/recall/F1 dihitung per kelas dengan zero division 0 lalu dirata-rata tanpa bobot; specificity dihitung one-vs-rest per kelas dan rata-ratanya mengabaikan kelas dengan penyebut tidak terdefinisi. Ini mengikuti fungsi `_metrics()` notebook, **implementation assumption** untuk agregasi multi-kelas karena paper tidak menetapkan apakah skor F1/specificity tabelnya memakai macro, micro, weighted, atau definisi lain. Figure 5 adalah bukti format matriks, bukan bukti kesamaan split atau prediksi. Tidak ada softmax sebelum `argmax`; evaluator menggunakan `preprocess_fundus_v2()` pada test tanpa augmentasi. Evaluator tidak memakai test untuk pemilihan checkpoint atau tuning.

Hasil baru, jika kelak dieksekusi secara manual, ditulis sekali ke `final_test_v2_80ep.json` di direktori checkpoint **di luar repository**. Berkas berisi metrik, laporan kelas, confusion matrix, manifest hash, dan epoch terpilih. Modul tidak melakukan W&B logging, training, atau penyimpanan checkpoint.
Ukuran batch inferensi default 16 dapat diturunkan dengan `--batch-size` bila VRAM 12 GB tidak cukup; itu tidak mengubah bobot, preprocessing, atau prediksi per gambar pada mode `eval()`.

## Perintah manual di GPU KC — hanya setelah kode disinkronkan dan artefak diperiksa

Jalankan dari terminal biasa di server; jangan memasang atau menjalankan agentic tools di sana. Contoh berikut memakai satu folder kerja `Kel_13` yang disebutkan dalam catatan eksperimen. Path boleh diberikan melalui argumen bila penyimpanan aktual berbeda; jangan menyalin dataset/checkpoint ke Git.

```bash
cd /mnt/home/probation-03/Kel_13/Diabetic-Rethinography
python -m reproduction.cadvit.evaluation preflight --dataset APTOS2019 --artifact-root /mnt/home/probation-03/Kel_13 --trust-checkpoints
python -m reproduction.cadvit.evaluation preflight --dataset DDR --artifact-root /mnt/home/probation-03/Kel_13 --trust-checkpoints
```

**Jangan jalankan `run` sampai kedua laporan preflight PASS, file sumber/log/checkpoint ditinjau, dan keputusan final test dibekukan.** Perintah `run` sengaja membutuhkan `--execute-final-test` tambahan dan tidak dipanggil otomatis oleh notebook atau preflight.

## Hal yang masih pending

1. Keberadaan dan isi checkpoint, manifest, log, serta kompatibilitas runtime server belum diverifikasi dari laptop ini. Preflight server harus berhasil pada kedua dataset.
2. Kesinambungan state DDR lintas pemindahan checkpoint epoch 63 tidak dapat dibuktikan dari metadata epoch/log saja tanpa bukti transfer atau hash yang memang tersedia.
3. Kesamaan dataset image bytes setelah manifest dibuat belum dibuktikan oleh preflight yang tidak membuka gambar.
4. Hasil agregat final test V2 dan confusion matrix kini tersedia dengan hash JSON terverifikasi; **lineage checkpoint, manifest, dan jejak eksekusi** tetap belum terbukti secara mandiri. Skor paper tidak boleh langsung dibandingkan sebagai macro-F1 yang identik.
