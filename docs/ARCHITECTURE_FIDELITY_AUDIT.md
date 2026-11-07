# Architecture Fidelity Audit — CAD-ViT

**Cakupan.** Audit statis `reproduction/cadvit/model.py` pada branch `salma`; tidak mengubah model atau menjalankan training/inference. Sumber utama ialah PDF lokal paper *CAD-ViT: Coordinate Attention-Enhanced Vision Transformer With Dilated Feature Fusion for Diabetic Retinopathy Staging*, IEEE Access (2025), DOI [10.1109/ACCESS.2025.3572455](https://doi.org/10.1109/ACCESS.2025.3572455). Rujukan halaman di bawah memakai **nomor halaman tercetak IEEE**, bukan indeks berkas PDF. Perhitungan parameter dilakukan dengan instansiasi model di CPU dan `sum(p.numel() for p in model.parameters())`, tanpa forward pass. Implementasi yang diaudit adalah kandidat historis, bukan rekonstruksi identik yang sudah terbukti.

## Perbandingan paper dan kode

| Komponen | Bukti paper | Implementasi `model.py` | Penilaian |
|---|---|---|---|
| Alur utama | Gambar 3, hlm. 91509: stem `C3+BS`, lalu `IRB → IRB+A → IRB → IRB+A → EMB → IRB+A → EMB → IRB+A → EMB`, diakhiri `C1+BS → AP → FSL`. Bagian III-D, hlm. 91508, menjelaskan fitur lokal MFB diteruskan ke Transformer. | `stem`, sembilan `blocks` dalam urutan tersebut, `head`, adaptive average pool, linear lima kelas. | **Sesuai pada tingkat diagram.** Lebar kanal, stride, dan dimensi head adalah asumsi. |
| MFB | Bagian III-B, hlm. 91507, Gambar 2 dan Persamaan (1)–(2): empat konvolusi dilatasi 3×3 dengan laju 3, 5, 7, 9, lalu konvolusi biasa 3×3. Persamaan (2) menuliskan `Concat(F3,F5,F7,F9)` sebelum konvolusi. | Empat `Conv2d(C,C,3)` paralel, `torch.cat` pada kanal, lalu `Conv2d(4C,C,3)`. Dipakai di tiap EMB. | **Sesuai dengan Persamaan (2).** Simbol `+` pada Gambar 2 memberi interpretasi fusion yang berbeda; paper tidak menjelaskan apakah itu penjumlahan sungguhan. Lebar tiap cabang, bias, normalisasi, dan aktivasi MFB tidak ditentukan. |
| Coordinate Attention | Bagian III-C, Persamaan (3)–(6), hlm. 91508: global average pooling menurut tinggi/lebar, reduksi 1×1, BN dan aktivasi, dua proyeksi 1×1 serta sigmoid, lalu perkalian spasial. Bagian IV-B dan Tabel 1, hlm. 91509, memilih `r=16`. | Rata-rata pada masing-masing sumbu, gabung sepanjang dimensi spasial, `1×1 → BN → SiLU`, split, dua gate `1×1 → sigmoid`, `x * gate_h * gate_w`; `r=16`. | **Sesuai secara operasional.** Pembulatan `C//r` (minimum 1), bias, dan kanal tempat CA dipasang adalah asumsi. Teks paper menyebut penggabungan “channel dimension”, sedangkan bentuk tensor dalam persamaan menuntut gabungan sepanjang arah spasial; kode mengikuti bentuk tensor. |
| IRB / IRB+A dan residual | Subdiagram Gambar 3, hlm. 91509: `C1+BS → C3+BS → Attention → C1+B`, dengan shortcut blok. Empat dari enam IRB memakai attention. | Expand 1×1 + BN + SiLU; depthwise 3×3 + BN + SiLU; CA bila `IRB+A`; project 1×1 + BN; residual jika `stride=1` dan kanal sama. | **Sesuai pada tingkat urutan yang terlihat.** Depthwise, expansion `2`, stride, kondisi shortcut, dan posisi CA pada kanal tersembunyi belum ditentukan secara numerik oleh paper. |
| EMB, token, Transformer, residual | Subdiagram Gambar 3, hlm. 91509: `MFB → Unfold → Transformer × L → Fold → C1+BS →` penjumlahan shortcut `→ C3+BS`. Bagian III-D, hlm. 91508, menyebut pemrosesan fitur lokal menjadi representasi global. | `MFB`, `F.unfold` patch 2×2 menjadi `[B·P,N,C]`, dua `TransformerEncoderLayer`, `F.fold`, proyeksi 1×1 + BN + SiLU, tambah input, fusion 3×3 + BN + SiLU. | **Sesuai pada tingkat alur diagram; tokenisasi detail adalah asumsi.** Pengelompokan token mengikuti pilihan implementasi ala MobileViT, tetapi paper CAD-ViT tidak menetapkan susunan token, ukuran patch, `L`, embedding, jumlah heads, FFN, norm, dropout, atau bentuk Transformer persis. |
| Classifier | Gambar 3, hlm. 91509: `C1+BS → AP → FSL`; lima kelas dijelaskan di Bagian IV-B, hlm. 91509. | `ConvBNAct(160,320,1)`, `AdaptiveAvgPool2d(1)`, `Linear(320,5)`. | **Sesuai pada tingkat operasi dan jumlah kelas.** Lebar 320 dan rincian `AP` merupakan asumsi. |
| Ukuran model | Tabel 1, hlm. 91509, melaporkan **7.2629 M** parameter untuk `r=16`. Tabel ini adalah ablation reduction ratio, **bukan** tabel konfigurasi layer demi layer. | **5.282.101** parameter pada `r=16`. | **Berbeda:** selisih nominal **1.980.799**. Angka paper dibulatkan ke empat desimal juta, sehingga bukan hitungan integer eksak. |

## Rincian parameter kandidat (`r=16`)

| Komponen kode | Parameter | Catatan |
|---|---:|---|
| Stem | 928 | 3×3 conv, BN |
| Enam IRB/IRB+A | 200.144 | Termasuk empat CA sejumlah **24.400**; bagian IRB selain CA **175.744** |
| Tiga MFB dalam EMB | 3.686.400 | EMB-1: 663.552; EMB-2: 1.179.648; EMB-3: 1.843.200 |
| Transformer dalam tiga EMB | 827.648 | EMB-1: 149.568; EMB-2: 264.960; EMB-3: 413.120 |
| Proyeksi dan fusion tiga EMB | 513.536 | Seluruh EMB: **5.027.584** termasuk MFB dan Transformer |
| Head 1×1 + BN | 51.840 | Lebar head 320 adalah asumsi |
| Linear lima kelas | 1.605 | Pooling: 0 parameter |
| **Total** | **5.282.101** | Dihitung langsung dari `CADViTCandidate()` di CPU |

Empat CA menerima kanal tersembunyi **64, 128, 192, 256** karena `hidden = in_channels × expansion` dengan `expansion=2`. Untuk implementasi ini, satu CA dengan kanal `C` dan `d=max(1,⌊C/r⌋)` memiliki **`3Cd + 2d + 2C`** parameter: shared convolution, BN, serta dua gate. Karena itu, perubahan `r` hanya mengubah bagian CA; parameter non-CA tetap **5.257.701** pada kelima instansiasi berikut.

| `r` | Paper, Tabel 1 (parameter) | Kandidat, dihitung | CA kandidat | Selisih nominal paper − kandidat |
|---:|---:|---:|---:|---:|
| 2 | 7.954.300 | 5.443.941 | 186.240 | 2.510.359 |
| 4 | 7.584.700 | 5.351.461 | 93.760 | 2.233.239 |
| 8 | 7.399.900 | 5.305.221 | 47.520 | 2.094.679 |
| 16 | 7.262.900 | 5.282.101 | 24.400 | **1.980.799** |
| 32 | 7.246.000 | 5.270.541 | 12.840 | 1.975.459 |

Penurunan parameter paper untuk `2→4`, `4→8`, `8→16`, `16→32` berturut-turut **369.600, 184.800, 137.000, 16.900**; kandidat **92.480, 46.240, 23.120, 11.560**. Dua interval awal paper hampir empat kali sensitivitas kandidat, tetapi interval berikutnya tidak mengikuti satu hukum `1/r` sederhana. Pembulatan Tabel 1 (sekitar ±50 parameter per angka) tidak cukup menjelaskan perubahan puluhan ribu itu. Maka selisih **bukan sekadar offset tetap dari model dasar**: respons parameter terhadap `r` juga berbeda. Ini **membuktikan ketidaksetaraan parameterisasi**, tetapi belum membuktikan modul tertentu salah atau memungkinkan jumlah CA paper dihitung secara unik. MFB dan Transformer menyumbang mayoritas parameter kandidat; keduanya serta lebar kanal dapat memengaruhi baseline secara besar, namun **alokasi 1.980.799 parameter ke EMB, CA, atau komponen lain tidak dapat ditetapkan dari publikasi**.

## Kesimpulan dan tindakan prioritas

1. **Pertahankan model historis dan checkpoint.** Tidak ada koreksi kode arsitektur yang secara tegas diwajibkan oleh Persamaan (1)–(6) atau Gambar 2–3. Khususnya, mengubah MFB dari concatenation menjadi sum demi Gambar 2 akan bertentangan dengan Persamaan (2); mengganti width/depth hanya demi jumlah parameter tidak sah.
2. **Labeli hasil sebagai reproduksi parsial/kandidat.** Urutan sembilan blok, laju dilatasi, operasi CA, dan alur EMB didukung paper. Width, stride, expansion, token layout, patch size, `L`, heads, FFN, classifier width, bias/normalisasi dan beberapa residual rules adalah *implementation assumptions*. Parameter 5.282.101 berbeda nyata dari 7.2629 M yang dilaporkan.
3. **Sebelum varian arsitektur baru, cari spesifikasi primer tambahan**: kode atau checkpoint resmi penulis, supplementary material, konfigurasi layer, atau jawaban penulis tentang width per stage, expansion, patch/token layout, depth/heads/FFN, serta penerapan CA. Tanpa itu, angka Tabel 1 tidak mengidentifikasi konfigurasi secara unik. Jika sumber baru tersedia, turunkan perubahan dalam varian terpisah dan uji terhadap implementasi historis; jangan menimpa hasil 80 epoch yang sudah ada.
4. **Untuk deadline enam hari**, audit ini cukup untuk melaporkan fidelity dan batasannya sekarang. Jangan menunda dokumentasi hasil eksperimen historis demi pencocokan parameter yang saat ini tidak teridentifikasi.

**Yang belum terverifikasi:** kesamaan dengan bobot/implementasi penulis, pemisahan pasti selisih parameter per modul, dan FLOPs paper versus kandidat. Tidak ada training, inferensi dataset, atau evaluasi test dalam audit ini.

## FINAL ARCHITECTURE DECISION — 7 November 2026

**Keputusan: `KEEP_CURRENT`.** Bekukan `CADViTCandidate` historis untuk eksperimen berikutnya. Tidak ditemukan perubahan arsitektur yang diwajibkan oleh teks/persamaan/Gambar 2–3 paper; selisih parameter **1.980.799** pada `r=16` tetap batas fidelity, bukan target untuk menebak ukuran layer. Checkpoint dan hasil V1/V2 tetap sah sebagai hasil *kandidat reproduksi parsial*; **tidak perlu training ulang untuk mempertahankan hasil tersebut**. Eksperimen ablation baru memerlukan training masing-masing varian, tetapi bukan karena model historis diubah.

| Field/operasi yang dibekukan | Nilai implementasi | Sumber dan batas bukti | Status |
|---|---|---|---|
| `stem_channels` | 32 | `ARCH_CONFIG`; Gbr. 3 hanya menunjukkan `C3+BS`, tanpa width | **ASSUMED** |
| `stage_channels` | `(32,64,64,96,96,128,128,160,160)` | `ARCH_CONFIG`; sebagian tangga width menyerupai MobileViT-S [25], tetapi tuple dan mapping sembilan blok tidak diberikan CAD-ViT | **ASSUMED** |
| `expansion` | 2 | `ARCH_CONFIG`; subdiagram IRB Gbr. 3 tidak memberi faktor ekspansi | **ASSUMED** |
| `reduction` | 16 | Pers. (4), §IV.B dan Tabel 1 hlm. 91509 memilih `r=16` | **VERIFIED** |
| `patch_size` | 2×2 | [Kode MobileViT asli](https://github.com/apple-aiml-research/ml-cvnets/blob/main/cvnets/models/classification/config/mobilevit.py) memakai patch 2×2; Gbr. 3 CAD-ViT hanya menulis `Unfold/Fold` | **REFERENCE_DERIVED** |
| `transformer_depth` | 2 pada ketiga EMB | `ARCH_CONFIG`; Gbr. 3 hanya `Transformer×L`. MobileViT-S asli memakai kedalaman per stage yang tidak seragam, sehingga tidak menentukan `L` CAD-ViT | **ASSUMED** |
| `transformer_heads` | 4 | `ARCH_CONFIG`; jumlah head tidak diberikan paper atau diagram CAD-ViT | **ASSUMED** |
| `mlp_ratio` | 2 | [Konfigurasi MobileViT asli](https://github.com/apple-aiml-research/ml-cvnets/blob/main/cvnets/models/classification/config/mobilevit.py) memberi contoh FFN dua kali dimensi Transformer; dimensi/FFN CAD-ViT tidak dinyatakan | **REFERENCE_DERIVED** |
| `head_channels` | 320 | `ARCH_CONFIG`; Gbr. 3 hanya `C1+BS → AP → FSL` | **ASSUMED** |
| `num_classes` | 5 | §IV.B dan Gbr. 3: kelas 0–4 | **VERIFIED** |
| `use_mfb`, `use_attention` | keduanya `True` pada model utama | Gbr. 3 menempatkan MFB di EMB dan empat `IRB+A`; §IV.C/Tabel 3 menguji masing-masing komponen | **VERIFIED** |
| Stride | stem 2; IRB pada posisi 2, 4, 6, 8 memakai 2; lainnya 1 | `CADViTCandidate`; Gbr. 3 tidak memberi stride atau resolusi feature map | **ASSUMED** |
| Aktivasi/normalisasi | Conv: BN+SiLU; proyeksi IRB: BN; CA: BN+SiLU; Transformer: GELU, post-LN, dropout 0 | Legenda `BS=BN-SiLU`, `B=BN` Gbr. 3 dan Swish Pers. (4) mendukung jalur Conv/CA. Aktivasi, norm order, dan dropout Transformer tidak dijelaskan | **VERIFIED** untuk Conv/CA; **ASSUMED** untuk Transformer |
| MFB/fusion dan residual EMB | dilatasi 3/5/7/9; concat lalu 3×3; EMB memakai shortcut **addition** | Pers. (1)–(2) dan Gbr. 2; Gbr. 3 memperlihatkan node tambah EMB. [MobileViT asli](https://github.com/apple-aiml-research/ml-cvnets/blob/main/cvnets/modules/mobilevit_block.py) memakai concat fusion, sehingga referensinya tidak boleh menggantikan diagram CAD-ViT | **VERIFIED** pada tingkat operasi; width/bias MFB **ASSUMED** |

**Sumber dan pencarian kode.** PDF CAD-ViT mencantumkan MobileViT Mehta–Rastegari sebagai referensi **[25]** (hlm. 91514); [paper MobileViT](https://arxiv.org/abs/2110.02178) menunjuk ke [kode asli Apple](https://github.com/apple-aiml-research/ml-cvnets). PDF CAD-ViT **tidak mencantumkan rujukan Hou–Zhou–Feng Coordinate Attention** atau tautan kode/suplemen CAD-ViT; aturan CA yang berlaku di sini berasal dari Pers. (3)–(6) CAD-ViT. [Kode CA asli Hou](https://github.com/houqb/CoordAttention/blob/main/coordatt.py) memakai hard-Swish dan `max(8,C//r)`, sedangkan CAD-ViT menyebut **Swish** dan `C/r`; karena itu kode Hou adalah konteks eksternal, bukan spesifikasi CAD-ViT. Pencarian DOI, judul, dan nama penulis di laman penerbit/web serta indeks kode GitHub pada tanggal audit ini tidak menghasilkan repositori yang dapat diatribusikan ke penulis CAD-ViT; **belum ditemukan source/checkpoint/suplemen resmi**. Hasil pencarian negatif tidak membuktikan artefak itu tidak ada.

**Perbedaan yang terbukti:** parameter kandidat 5.282.101 versus Tabel 1 paper 7.2629 M; tren parameter antar-`r` juga berbeda (rincian di atas). **Tidak ada field atau urutan blok yang terbukti `CONTRADICTED` oleh paper**. Detail yang belum diketahui: width dan embedding tiap stage, expansion/stride, jumlah layer/head, FFN, token layout persis, bias, dropout, serta prosedur normalisasi Transformer. Kode resmi MobileViT menunjukkan pilihan yang mungkin, bukan konfigurasi CAD-ViT.

**Konfigurasi eksperimen yang dibekukan.** Gunakan `ARCH_CONFIG` tanpa perubahan dan manifest/split yang sama pada seluruh varian. **Tabel 1:** APTOS2019, `r ∈ {2,4,8,16,32}`; hanya field `reduction` berubah, masing-masing varian memerlukan training/evaluasi tersendiri untuk kolom accuracy; jumlah parameter sudah dihitung tanpa training di `docs/PAPER_EXPERIMENT_REPRODUCTION_PLAN.md`. **Tabel 3:** APTOS2019, `r=16`, `ABLATION_CONFIGS` notebook: `baseline` (`use_mfb=False,use_attention=False`), `MFB` (`True,False`), `Attention` (`False,True`), `MFB+Attention` (`True,True`); hanya kedua flag itu berubah. Bekukan preprocessing, manifest, seed, optimizer, checkpoint rule, dan metrik yang sama *sebelum* masing-masing perbandingan; pisahkan protokol V2 BG10 sebagai extension dari klaim paper-faithful. **Langkah langsung:** jalankan varian yang belum ada hanya setelah protokol dan sumber daya ditetapkan; jangan gunakan final test historis untuk memilih width, `r`, atau varian. Hasil utama V2 yang sudah selesai tidak diulang atau ditimpa.
