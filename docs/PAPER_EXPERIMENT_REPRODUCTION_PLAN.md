# Rencana reproduksi eksperimen paper CAD-ViT — Tabel 1

**Status:** audit numerik tanpa training atau akses test set. Sumber primer: Y. Wang dkk., *CAD-ViT: Coordinate Attention-Enhanced Vision Transformer With Dilated Feature Fusion for Diabetic Retinopathy Staging*, IEEE Access 2025, DOI [10.1109/ACCESS.2025.3572455](https://doi.org/10.1109/ACCESS.2025.3572455), §IV.B dan **Tabel 1, halaman IEEE 91509**. Tabel itu berjudul akurasi dan jumlah parameter pada APTOS-2019 untuk berbagai reduction ratio; kolomnya juga memuat FLOPs. Persamaan (4), halaman IEEE 91508, mendefinisikan reduksi Coordinate Attention dengan `r`.

## Angka paper versus kandidat saat ini

Parameter kandidat dihitung langsung di CPU dari `CADViTCandidate({**ARCH_CONFIG, "reduction": r})` pada `reproduction/cadvit/model.py`, memakai `sum(p.numel() for p in model.parameters())`. Ini hanya instansiasi model, tanpa forward pass, training, atau evaluasi. Satuan paper **M** dicatat persis seperti Tabel 1; angka integer paper di kolom selisih adalah konversi nominal dari angka empat desimal juta, bukan klaim hitungan eksak penulis.

| `r` | Accuracy paper, APTOS2019 (%) | Parameter paper (M) | FLOPs paper (G) | Parameter kandidat | Parameter CA kandidat | Selisih nominal paper − kandidat | Status performa kandidat |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 88,4 | 7,9543 | 1,9820 | 5.443.941 | 186.240 | 2.510.359 | **PENDING — perlu training** |
| 4 | 88,1 | 7,5847 | 1,9753 | 5.351.461 | 93.760 | 2.233.239 | **PENDING — perlu training** |
| 8 | 88,0 | 7,3999 | 1,9720 | 5.305.221 | 47.520 | 2.094.679 | **PENDING — perlu training** |
| 16 | **88,6** | **7,2629** | **1,9703** | **5.282.101** | **24.400** | **1.980.799** | **PENDING — Tabel 1 belum direproduksi** |
| 32 | 87,9 | 7,2460 | 1,9696 | 5.270.541 | 12.840 | 1.975.459 | **PENDING — perlu training** |

`r=16` adalah pilihan paper untuk keseimbangan akurasi dan ukuran model (§IV.B). Paper **menguji pengaruh `r` terhadap performa klasifikasi**, bukan hanya parameter: accuracy 88,4%; 88,1%; 88,0%; 88,6%; 87,9% untuk `r=2,4,8,16,32`. FLOPs kandidat belum dihitung dengan metode yang dapat dibandingkan dengan paper. Log training V2 historis `r=16` selama 80 epoch memuat **validation macro-F1/accuracy dari protokol V2 BG10**; hasil final test V2 yang dilaporkan harus dibaca dari artefak final test terpisah dan tidak otomatis menjadi angka Tabel 1. Preprocessing V2 BG10 merupakan *experimental extension*. Angka accuracy paper tidak boleh disamakan otomatis dengan validation accuracy kandidat.

## Analisis tren dan identifiabilitas

| Perubahan `r` | Pengurangan parameter paper (nominal) | Pengurangan parameter kandidat |
|---|---:|---:|
| 2 → 4 | 369.600 | 92.480 |
| 4 → 8 | 184.800 | 46.240 |
| 8 → 16 | 137.000 | 23.120 |
| 16 → 32 | 16.900 | 11.560 |

Keduanya menurun monoton saat `r` membesar. Akan tetapi, besar penurunannya berbeda: dua interval awal paper sekitar empat kali kandidat, sedangkan dua interval terakhir tidak mengikuti faktor itu. Perbedaan puluhan ribu parameter tidak dijelaskan oleh pembulatan Tabel 1 ke empat desimal M. Empat CA kandidat menerima kanal tersembunyi 64, 128, 192, 256; dengan `d=max(1,⌊C/r⌋)`, satu CA pada implementasi ini memiliki `3Cd + 2d + 2C` parameter. Seluruh parameter **non-CA kandidat tetap 5.257.701** untuk kelima nilai `r`.

Dengan demikian, selisih paper–kandidat **bukan offset konstan saja**. Sensitivitas terhadap `r` mengindikasikan perbedaan parameterisasi CA, lebar kanal tempat CA dipasang, jumlah/penggunaan modul, atau detail lain yang belum dipublikasikan; data Tabel 1 tidak mengidentifikasi penyebab tunggal. Selisih baseline besar juga dapat terkait konfigurasi MFB/EMB/Transformer yang tidak diberikan lengkap dalam paper. Tidak ada dasar ilmiah untuk mengubah channel, depth, atau expansion demi menargetkan 7.2629 M. Lihat `docs/ARCHITECTURE_FIDELITY_AUDIT.md` untuk rincian blok.

## Matriks eksperimen yang diperlukan kemudian

| Eksperimen | Dataset/protokol | Yang harus dijalankan untuk mereproduksi kolom accuracy | Status sekarang |
|---|---|---|---|
| CA-`r2` | APTOS2019, split 8:1:1 yang sama | Train model kandidat `r=2`; pilih checkpoint dari validation; evaluasi pada split yang dibekukan setelah protokol diputuskan | Parameter dihitung; training/evaluasi **PENDING** |
| CA-`r4` | Sama | Train `r=4` dengan semua variabel lain tetap | Parameter dihitung; training/evaluasi **PENDING** |
| CA-`r8` | Sama | Train `r=8` dengan semua variabel lain tetap | Parameter dihitung; training/evaluasi **PENDING** |
| CA-`r16` | Sama | Train `r=16` dalam protokol ablation yang sama; hasil V2 BG10 historis hanya pembanding terpisah | Parameter dihitung; ablation terkontrol **PENDING** |
| CA-`r32` | Sama | Train `r=32` dengan semua variabel lain tetap | Parameter dihitung; training/evaluasi **PENDING** |

**Kunci metodologi sebelum menjalankan matriks:** §IV.B paper menetapkan APTOS-2019, rasio split 8:1:1, input 512×512, learning rate 0,000186, batch 16, dan 80 epoch. Gunakan preprocessing yang dijelaskan paper untuk klaim reproduksi paper; Ben Graham BG10 harus diperlakukan sebagai eksperimen terpisah. Paper tidak memberikan ID split, seed, optimizer, aturan checkpoint, atau penanda eksplisit apakah accuracy Tabel 1 berasal dari validation atau test. Karena itu tetapkan dan catat pilihan implementasi tersebut sebelum training, gunakan manifest identik untuk kelima `r`, dan jangan memilih hyperparameter atau model berdasarkan test. Jika kelak melaporkan accuracy test untuk kelima varian sebagai ablation yang telah ditetapkan, evaluasi dilakukan sekali per checkpoint setelah aturan seleksi validation dibekukan; jangan melakukan iterasi setelah melihat hasil test. Laporkan hasil kandidat sebagai **reproduksi parsial** selama konfigurasi internal penulis dan jumlah parameter masih berbeda.

**Catatan audit awal:** tabel parameter di atas belum menghasilkan accuracy kandidat atau FLOPs yang setara paper; estimasi core MAC runner baru dijelaskan di bawah.

## Implementasi terpadu Tabel 1–5 (7 November 2026)

Runner: `reproduction/cadvit/experiments.py`. Ia **tidak** mengubah `model.py`, `preprocessing.py`, notebook, manifest, atau checkpoint historis. `dry-run` membuat model di CPU, menguji forward sintetis 64×64, dan menghitung parameter serta estimasi MAC analitik untuk 512×512; `preflight` hanya membaca metadata manifest, tidak membuka gambar. `train` dan `test` merupakan perintah manual terpisah dengan flag eksekusi eksplisit. Semua artefak baru berada di `<Kel_13>/experiments/cadvit_tables_1_5_paper_v1/`, di luar repository; `protocol.json` dibuat sekali dengan mode gagal jika sudah ada, sedangkan setiap varian mempunyai direktori `config.json`, `history.jsonl`, `best.pt`, `last.pt`, dan `test_metrics.json` sendiri. Resume menolak config/hash/riwayat epoch yang tidak cocok. Training hanya membuat loader train dan validation; loader test dibuat **hanya** oleh perintah `test` setelah 80 epoch dan checkpoint terbaik diverifikasi terhadap validation macro-F1.

| Tabel paper | Varian runner | Status eksekusi lokal | Kondisi sebelum klaim reproduksi |
|---|---|---|---|
| 1, hlm. 91509 | `aptos_r2`, `r4`, `r8`, `r16`, `r32` | **available**: parameter dan estimasi MAC; **needs-training**: akurasi setiap `r` | Satu manifest/protokol, hanya `reduction` berubah; test dibekukan |
| 2, hlm. 91510 | `aptos_binary_head` | **needs-training** sebagai *implementation assumption* | Paper tidak menyebut apakah head dua kelas dilatih tersendiri atau prediksi lima kelas digabung; hasil dua pendekatan tidak setara |
| 3, hlm. 91510 | `aptos_baseline`, `aptos_mfb`, `aptos_attention`, `aptos_r16` | **available**: verifikasi flag (MFB 0/3; CA 0/4); **needs-training**: metrik | Hanya `use_mfb` dan `use_attention` berubah; r=16 dan protokol sama |
| 4, hlm. 91511 | hasil `aptos_r16` + baris paper SDMAE, ADCNet, TLSPOFECOC, GNN, TMILv4 | **needs-validation** untuk hasil final historis yang dilaporkan tetapi JSON/checkpoint tidak ada lokal; run baru **needs-training** | Hasil sendiri mesti final test full model dari kampanye yang sama; baris pihak ketiga hanya kutipan Tabel 4 |
| 5, hlm. 91512 | hasil `ddr_full` + baris paper AABNet, CLANet, CRA-Net, Deep integrative network, MDGNet | **needs-validation** untuk hasil final/recovery historis yang dilaporkan; run baru **needs-training** | Recovery DDR historis weighted FP32/FP64 ialah extension dengan lineage transfer epoch 63 belum dibuktikan langsung; baris pihak ketiga hanya kutipan Tabel 5 |

**Protokol kampanye baru yang dibekukan:** APTOS2019/DDR pada manifest resmi reproduksi 8:1:1 seed 42; `preprocess_fundus()` (crop → resize 512 `INTER_AREA` → Gaussian blur) tanpa V2 BG10; augmentasi hanya training sesuai kode historis (horizontal/vertical flip, rotasi ±15°, shear 10°); tanpa `WeightedRandomSampler` atau weighted loss; AdamW `weight_decay=0`, CrossEntropyLoss, LR 0,000186, efektif batch 16, 80 epoch, AMP bila CUDA tersedia, clipping 1, checkpoint berdasarkan validation macro-F1 (tie: validation loss). **Implementation assumptions:** seed, optimizer/loss, augmentasi numerik, balancing=`none`, AMP, clipping, checkpoint rule. Paper §III.A/§IV.B menyebut operasi preprocessing, input, LR, batch, epoch, dan augmentation umum; tidak memberikan seluruh detail kampanye. `--micro-batch` dapat diturunkan bila VRAM tidak cukup, dengan accumulation sehingga batch efektif tetap 16; perubahan statistik BatchNorm dibanding batch fisik 16 harus dicatat dan **dipakai sama pada semua varian**.

**Tabel 2 dan definisi metrik.** §IV.C hanya menyatakan diagnosis DR/No DR; tidak menjelaskan 2-logit head, retraining, atau agregasi 5-logit. Runner memakai **head biner terlatih terpisah** (`num_classes=2`) sebagai protokol asumsi, dengan label `0 → No DR` dan `1–4 → DR`; alternatif analisis ialah menggabungkan prediksi lima kelas yang sudah dibekukan, dan harus dilaporkan terpisah. Figure 4 menampilkan 367 prediksi dengan orientasi *predicted rows / true columns*; matriks dalam orientasi runner (*true rows / predicted columns*, urutan No DR lalu DR) menjadi `[[181,0],[3,183]]`. Itu memberi overall accuracy `364/367 = 99,2%`, recall DR `183/186 = 98,4%`, specificity DR `181/181 = 100%`, dan F1 DR `99,2%`. Tabel 2 mencantumkan “Accuracy” per kategori `100%`/`98,4%`; angka tersebut cocok dengan **precision per kategori** dari Figure 4, sedangkan akurasi one-vs-rest kedua kategori sama-sama `99,2%`. Runner tidak menamai precision sebagai accuracy. Manifest reproduksi APTOS2019 8:1:1 yang dibulatkan dari 3.662 gambar menghasilkan 366 test image, sehingga **denominator Figure 4 (367) tidak identik**. Tabel 3–5 melaporkan accuracy/specificity tetapi tidak mendefinisikan agregasi lima kelas; runner menggunakan accuracy keseluruhan dan specificity macro one-vs-rest seperti notebook. F1 paper tidak boleh disamakan otomatis dengan macro-F1 runner.

**FLOPs Tabel 1.** Fungsi `core_macs()` memakai input `[1,3,512,512]`, menghitung MAC convolution/linear dan perkalian matriks attention. Ia melaporkan `1 MAC = 1` serta `2 operasi aritmetika per MAC`, dan mengecualikan BN, aktivasi, softmax, pooling, bias, dan preprocessing. Model penuh `r=16` menghasilkan sekitar **8,95238 G core MAC** atau **17,90476 G operasi aritmetika**, dibanding **1,9703 G FLOPs** yang dilaporkan Tabel 1 paper. Ini **bukan** perbandingan profiler yang identik; konvensi FLOPs paper dan detail arsitektur yang tak dipublikasikan belum diketahui. Jangan mengubah layer demi menyamakan angka. Pengujian sintetis CPU 64×64 membandingkan estimasi analitik dengan `torch.profiler`, tetapi tidak digunakan sebagai klaim FLOPs paper.

| `r` | Core MAC kandidat (G), input 512×512 | FLOPs aritmetika kandidat (G), 2/MAC | FLOPs Tabel 1 paper (G), konvensi tidak dijelaskan |
|---:|---:|---:|---:|
| 2 | 8,959035 | 17,918070 | 1,9820 |
| 4 | 8,955234 | 17,910468 | 1,9753 |
| 8 | 8,953333 | 17,906667 | 1,9720 |
| 16 | 8,952383 | 17,904766 | 1,9703 |
| 32 | 8,951908 | 17,903816 | 1,9696 |

**Pemakaian ulang checkpoint historis:** full `r=16` APTOS V2 BG10 *berpotensi* menjadi anchor untuk Tabel 1/3/4 **hanya** jika seluruh varian lain memakai protokol V2 BG10 dan checkpoint/manifest/hasil diverifikasi; itu adalah kampanye extension, bukan kampanye paper-v1 di atas. Checkpoint DDR recovery memakai weighted loss dan FP64 clipping, sehingga tidak seprotokol dengan APTOS paper-v1 atau `ddr_full` baru. Final test V2 yang dilaporkan pengguna tidak dijalankan ulang dan hasilnya tidak otomatis dimasukkan ke tabel baru. Perintah `compare` menolak JSON yang tidak cocok dengan `config.json`, hash protokol, manifest, dan nama varian kampanye.

**Urutan dan waktu GPU.** Arsip log asli `cadvit_notebook_80ep_sources.tar.gz` diperiksa tanpa ekstraksi: log APTOS, DDR, dan JSONL recovery memuat metrik epoch tetapi **tidak memuat durasi/timestamp tiap epoch**. Jadi jam GPU numerik tidak dapat diturunkan secara jujur dari log historis. Runner baru merekam `epoch_seconds`; setelah satu epoch pertama, hitung sisa waktu sebagai `(80 − epoch selesai) × median detik/epoch terukur`, lalu kalikan jumlah varian yang belum dijalankan. Dengan dua RTX 3060 12 GB, jalankan satu varian per GPU hanya setelah uji memori manual; jangan menganggap DataParallel mengubah batch fisik secara identik. **Decision gate:** varian pertama `aptos_r16`, karena hasilnya menjadi anchor bersama Tabel 1, Tabel 3, dan Tabel 4 serta mengukur throughput sebelum delapan run APTOS lain. Setelah itu jalankan baseline/MFB/Attention, lalu `r=2,4,8,32`; biner dan DDR mengikuti kapasitas waktu. Bila enam hari tidak cukup, nyatakan baris yang belum dilatih sebagai `PENDING`, bukan mengisi metrik dari protokol berbeda.

### Perintah manual di KC (jangan dijalankan melalui Codex)

```bash
cd /mnt/home/probation-03/Kel_13/Diabetic-Rethinography
python -m reproduction.cadvit.experiments matrix
python -m reproduction.cadvit.experiments dry-run --variant aptos_r16
python -m reproduction.cadvit.experiments init --artifact-root /mnt/home/probation-03/Kel_13 --micro-batch 16
python -m reproduction.cadvit.experiments preflight --variant aptos_r16 --artifact-root /mnt/home/probation-03/Kel_13
# DDR memakai root gambar/anotasi aktual; cek dahulu kecocokannya dengan metadata manifest.
python -m reproduction.cadvit.experiments preflight --variant ddr_full --artifact-root /mnt/home/probation-03/Kel_13 --data-root /mnt/home/probation-03/Kel_13/datasets/DDR/DDR-dataset/DR_grading
# Hanya setelah preflight, VRAM, dan protokol ditinjau:
python -m reproduction.cadvit.experiments train --variant aptos_r16 --artifact-root /mnt/home/probation-03/Kel_13 --execute-training
# Perintah DDR kelak, dengan root yang sama seperti preflight:
python -m reproduction.cadvit.experiments train --variant ddr_full --artifact-root /mnt/home/probation-03/Kel_13 --data-root /mnt/home/probation-03/Kel_13/datasets/DDR/DDR-dataset/DR_grading --execute-training
# Bila proses terhenti: ulangi perintah train dengan --resume; jangan gunakan test untuk memilih varian.
```

`init` sekali saja. `--data-root` berarti **root aktual dataset tersebut**, bukan folder induk seluruh dataset. Jika batch fisik 16 tidak muat, tentukan `--micro-batch 4` **sebelum** membuat protokol dan gunakan protokol tersebut untuk semua run. `test` dan `compare` adalah tahap terpisah yang sengaja tidak dicontohkan sebagai perintah siap tempel sebelum hasil validation dan aturan seleksi dibekukan.
