# PReOF Phase 1 — cache preprocessing, cache prediksi, dan uji kelayakan E0

**Status: E0 selesai di KC; NO-GO untuk M2–M4. E1 siap dijalankan pada cache validation, belum dieksekusi di KC.** PReOF menanyakan apakah respons prediksi model terhadap perpindahan preprocessing V1↔V2 memberi informasi tambahan tentang *kesalahan keputusan pada batas ordinal* `Y ≥ k`. Phase 1 hanya menguji pertanyaan tersebut. Belum ada gating, projection, regularized ordinal learning, pelatihan backbone baru, atau akses final test.

## 1. Sumber dan batas bukti

| Dataset | Manifest beku (SHA-256) | Validation | V1 native checkpoint | V2 native checkpoint |
| --- | --- | ---: | --- | --- |
| APTOS2019 | `73d1546e3c2fd2bf9e3d0fd10af914cb523d28e4bc8e1f25212af454d4be8854` | 366 | balanced V1 epoch 24, `6db01fbb840f4ed9641f9d8d3b9ac90e98d3b0772d57101be56aaabc60e05601` | BG10 epoch 69, `e72573f0f1c26baac53e18d950076845952da503261e212021ff6320c9ae72d5` |
| DDR | `7bc3e3916e012bc46fcd9cd69aa469e0daf3229e1273f1d8ecb6054899bf6eb1` | 1.252 | FP32 Math recovery epoch 69, `84e3272d3f10d9179fd2dd9e2799b640962e1185981856f87bcb3876b9deace0` | BG10 mixed-protocol epoch 72, `844665f0c597142915286b727a21f1d2223f43e0f057d083b6c1dd035c180ac1` |

Script [`preof_phase1.py`](../novelty/preof_phase1.py) memakai fungsi `preprocess_fundus` dan `preprocess_fundus_v2` yang sudah diekstrak dari notebook historis. Sebelum membuka gambar, preflight menghitung hash manifest dan kedua checkpoint, membandingkan record checkpoint dengan konfigurasi run dan konstanta preprocessing, mencocokkan definisi model/preprocessing pada notebook V2 server melalui AST, memuat state dictionary secara `strict=True`, serta memeriksa seluruh tensor finite. Hash checkpoint yang cocok **tidak sendirinya membuktikan eksekusi historis**. `torch.load(..., weights_only=False)` hanya boleh dipakai pada dua file server proyek yang hash-nya cocok dan dipercaya operator.

Pemeriksaan AST source V2 harus **PASS** terhadap `~/Kel_13/Diabetic-Rethinography-v2/CAD_ViT.ipynb`; bila struktur file server berbeda, periksa source historis dan sesuaikan audit secara eksplisit sebelum cache dibuat. Jangan melewati guard karena jumlah parameter sama. Manifest dibaca untuk hash dan hanya baris `split=val` yang disimpan; gambar test tidak dibuka. File gambar validation di-hash ulang satu per satu. Seluruh cache dan laporan E0 harus berada di luar repository.

## 2. Dua tingkat cache validation

Alur: **gambar mentah → cache preprocessing deterministik (Level 1) → inferensi CAD-ViT beku → cache prediksi (Level 2) → analisis E0**. Level 1 menyimpan keluaran persis `preprocess_fundus` (V1) dan `preprocess_fundus_v2` (V2 BG10), masing-masing array RGB `(512, 512, 3)` bertipe `uint8` dalam `.npy` lossless. Satu cache bersama dipakai untuk APTOS dan DDR, di luar Git dan direktori dataset mentah. Entri dibuat hanya saat gambar validation dibutuhkan; gambar test dan augmentasi acak tidak diproses. Pada pelatihan backbone masa depan, urutannya tetap preprocessing deterministik → augmentasi acak dinamis → konversi tensor. Phase 1 tidak memakai augmentasi.

Kunci Level 1 memuat SHA-256 isi gambar sumber, varian V1/V2 BG10, seluruh konfigurasi preprocessing terkait, SHA-256 implementasi `preprocessing.py`, versi NumPy/OpenCV/Pillow, dan versi skema. Kunci **tidak** memuat hash checkpoint atau konfigurasi fusi, sehingga array yang sama dapat digunakan ulang lintas model/eksperimen. Tata letak: `preprocessing/{v1,v2_bg10}/<dua karakter kunci>/<kunci>.npy` beserta `<kunci>.json` yang mencatat payload kunci dan SHA-256 piksel. Pada hit, bentuk, tipe, dan digest piksel diperiksa; entri rusak dihitung dan dibuat ulang. File sementara unik ditulis dan di-`fsync`, lalu dipindahkan secara atomik; pembaca serentak yang menemukan pasangan `.npy`/metadata belum konsisten akan meregenerasi, bukan mengembalikan piksel salah. Tidak ada pembersihan cache otomatis.

`preprocess_fundus_v2` memang memanggil `preprocess_fundus` di dalam source historis. Pada miss kedua varian, V1 dapat dihitung dua kali; source tidak diubah atau ditiru ulang agar keluaran tetap persis. Hit berikutnya tidak mengulang preprocessing. Sekitar **2,37 GiB** diperlukan untuk 1.618 gambar validation × dua varian × 786.432 byte, ditambah header/metadata kecil; ini estimasi kapasitas, bukan pengukuran KC. Statistik hit, miss, regenerasi, byte tertulis, dan ukuran cache tercetak setelah pembuatan cache prediksi. `benchmark` membandingkan waktu preprocessing langsung, pass cache pertama, dan pass hangat serta memeriksa kesetaraan piksel. Pass pertama hanya benar-benar dingin jika kunci belum tersedia. Dua proses GPU boleh memakai root Level 1 yang sama; masing-masing harus memakai direktori prediksi Level 2 tersendiri.

## 3. Cache empat prediksi

Untuk setiap gambar validation: `p11=V1(V1)`, `p12=V1(V2)`, `p21=V2(V1)`, `p22=V2(V2)`. Setiap entri menyimpan `image_id`, grade, `group_id`, split, SHA-256 gambar, lima logit mentah, dan lima probabilitas untuk keempat kondisi. Kedua model berada dalam `eval()` dan `inference_mode()`; tidak ada augmentasi, dropout aktif, update batch normalization, gradien, optimizer, atau training.

`metadata.json` mencatat hash manifest/checkpoint/record, hash source model dan preprocessing, hash notebook V2 server, hash implementasi Phase 1, serta versi runtime. `cache.jsonl` ditulis per baris dengan flush dan `fsync`. Cache prediksi lengkap dan valid diperiksa **sebelum model atau cache preprocessing dimuat**; checkpoint dan source tetap di-hash, tetapi checkpoint tidak dideserialisasi dan inferensi tidak diulang. Hash file script Phase 1 tetap tercatat untuk audit, tetapi perubahan transport array yang setara tidak membatalkan cache prediksi. Resume cache parsial memerlukan `--resume`; seluruh prefix dibaca ulang, urutan ID, label, grup, hash, finite logits, normalisasi probabilitas, softmax, dan provenance dibandingkan. Baris terpotong atau cache berbeda menghentikan proses; jangan mengeditnya diam-diam. Smoke cache `--limit 3` harus memakai direktori Level 2 berbeda dari cache penuh, tetapi memakai root Level 1 bersama. E0 menolak cache parsial.

## 4. E0 — estimasi kesalahan batas ordinal

Probabilitas lima grade diubah menjadi `q_k=P(Y≥k)=Σ_{j=k}^4 P(Y=j)` untuk `k=1,2,3,4`. Target E0 adalah apakah keputusan native `q_k≥0,5` salah terhadap grade validation. Evaluasi diulang untuk native V1 (`p11`) dan V2 (`p22`). Fitur **A**: confidence/entropy native dan margin/entropy batas; **B**: A ditambah selisih probabilitas dan keputusan batas antar model native; **C**: B ditambah `|q11−q12|` dan `|q21−q22|`; **D**: keempat distribusi lima kelas secara langsung (20 nilai).

Semua kondisi memakai estimator logistik L2 dengan penalti tetap 1 dan standardisasi yang *fit* hanya pada fold-training. Tidak ada pemilihan hyperparameter dari fold held-out. Satu penugasan fold berdasarkan `group_id` dipakai untuk semua kondisi, batas, dan native expert dalam dataset. Prediksi out-of-fold digabung untuk AUROC, AUPRC, Brier, dan binary log loss; fold tanpa dua kelas kesalahan ditandai `UNESTIMABLE`. Bootstrap berpasangan mengambil ulang **grup**, bukan gambar terpisah, dan melaporkan interval 95% untuk C−B. Laporan juga memuat support `Y≥k`, jumlah kesalahan, serta pasangan kesalahan komplementer V1/V2. D adalah pembanding berinformasi lengkap dengan keluarga estimator sama, meski jumlah fiturnya lebih besar.

Respons preprocessing yang besar **bukan bukti** prediksi salah atau lesi terlewat. V1 dan V2 juga berbeda dalam training/balancing/precision, terutama recovery DDR; selisih expert tidak boleh diatribusikan hanya pada preprocessing. Support batas tinggi dan grade minoritas, khususnya DDR grade 3, membatasi ketelitian estimasi.

## 5. Perintah manual pada server KC

Jalankan setelah file Phase 1 dari worktree `novelty` ditempatkan pada checkout novelty server yang memiliki source reproduction identik. Tidak ada commit/push atau koneksi KC otomatis dari laptop ini. **Jika `V2_SOURCE` tidak ada atau audit AST gagal, hentikan di preflight dan kirim hasilnya untuk pemeriksaan source.**

```bash
export ROOT="$HOME/Kel_13"
export REPO="$ROOT/Diabetic-Rethinography"
export PY="$ROOT/.venv/bin/python"
export V2_SOURCE="$ROOT/Diabetic-Rethinography-v2/CAD_ViT.ipynb"
cd "$REPO"
test "$(git branch --show-current)" = novelty || { echo 'Checkout KC harus pada branch novelty'; exit 1; }
test -f "$V2_SOURCE" && test -f novelty/preof_phase1.py || { echo 'Source notebook/script belum tersedia'; exit 1; }
"$PY" -m novelty.preof_phase1 preflight --dataset APTOS2019 --artifact-root "$ROOT" --v2-source "$V2_SOURCE" --trust-checkpoints
"$PY" -m novelty.preof_phase1 preflight --dataset DDR --artifact-root "$ROOT" --v2-source "$V2_SOURCE" --trust-checkpoints
```

**Benchmark kecil tanpa memuat model atau menjalankan inferensi** (satu root Level 1 untuk kedua dataset). Jalankan sebelum smoke bila ingin mengamati pass dingin; bila cache sudah ada, lihat `first_cache_pass.hit_rate` untuk membedakan hit lama dari miss. Waktu KC belum diukur.

```bash
cd "$HOME/Kel_13/Diabetic-Rethinography"
"$HOME/Kel_13/.venv/bin/python" -m novelty.preof_phase1 benchmark --dataset APTOS2019 --artifact-root "$HOME/Kel_13" --preprocess-cache-root "$HOME/Kel_13/preof_phase1/preprocessing" --limit 3
"$HOME/Kel_13/.venv/bin/python" -m novelty.preof_phase1 benchmark --dataset DDR --artifact-root "$HOME/Kel_13" --preprocess-cache-root "$HOME/Kel_13/preof_phase1/preprocessing" --limit 3
```

**Terminal GPU 0 — APTOS2019 (smoke, kemudian cache penuh):**

```bash
cd "$HOME/Kel_13/Diabetic-Rethinography"
test "$(git branch --show-current)" = novelty || { echo 'Checkout KC harus pada branch novelty'; exit 1; }
CUDA_VISIBLE_DEVICES=0 "$HOME/Kel_13/.venv/bin/python" -m novelty.preof_phase1 cache --dataset APTOS2019 --artifact-root "$HOME/Kel_13" --v2-source "$HOME/Kel_13/Diabetic-Rethinography-v2/CAD_ViT.ipynb" --trust-checkpoints --device cuda:0 --limit 3 --output-dir "$HOME/Kel_13/preof_phase1/aptos_smoke" --preprocess-cache-root "$HOME/Kel_13/preof_phase1/preprocessing"
# Tinjau metadata.json dan ketiga baris cache.jsonl sebelum perintah penuh.
CUDA_VISIBLE_DEVICES=0 "$HOME/Kel_13/.venv/bin/python" -m novelty.preof_phase1 cache --dataset APTOS2019 --artifact-root "$HOME/Kel_13" --v2-source "$HOME/Kel_13/Diabetic-Rethinography-v2/CAD_ViT.ipynb" --trust-checkpoints --device cuda:0 --output-dir "$HOME/Kel_13/preof_phase1/aptos_full" --preprocess-cache-root "$HOME/Kel_13/preof_phase1/preprocessing"
```

**Terminal GPU 1 — DDR (smoke, kemudian cache penuh):**

```bash
cd "$HOME/Kel_13/Diabetic-Rethinography"
test "$(git branch --show-current)" = novelty || { echo 'Checkout KC harus pada branch novelty'; exit 1; }
CUDA_VISIBLE_DEVICES=1 "$HOME/Kel_13/.venv/bin/python" -m novelty.preof_phase1 cache --dataset DDR --artifact-root "$HOME/Kel_13" --v2-source "$HOME/Kel_13/Diabetic-Rethinography-v2/CAD_ViT.ipynb" --trust-checkpoints --device cuda:0 --limit 3 --output-dir "$HOME/Kel_13/preof_phase1/ddr_smoke" --preprocess-cache-root "$HOME/Kel_13/preof_phase1/preprocessing"
# Tinjau metadata.json dan ketiga baris cache.jsonl sebelum perintah penuh.
CUDA_VISIBLE_DEVICES=1 "$HOME/Kel_13/.venv/bin/python" -m novelty.preof_phase1 cache --dataset DDR --artifact-root "$HOME/Kel_13" --v2-source "$HOME/Kel_13/Diabetic-Rethinography-v2/CAD_ViT.ipynb" --trust-checkpoints --device cuda:0 --output-dir "$HOME/Kel_13/preof_phase1/ddr_full" --preprocess-cache-root "$HOME/Kel_13/preof_phase1/preprocessing"
```

Untuk uji kesetaraan inferensi tanpa Level 1, gunakan `--uncached` pada perintah `cache` dengan **`--output-dir` baru**; jangan gunakan direktori Level 2 yang sama untuk membandingkan karena cache lengkap akan menghentikan inferensi sebelum preprocessing. Membandingkan empat logit per gambar antara dua cache Level 2 menunjukkan apakah transport array mengubah prediksi; tetap periksa determinisme perangkat bila membandingkan run GPU yang berbeda.

Jika proses terputus, ulangi **perintah cache yang sama** dengan `--resume`; jangan gunakan `--resume` untuk cache smoke saat meminta cache penuh. Setelah kedua cache lengkap, E0 hanya membaca cache dan manifest validation:

```bash
cd "$HOME/Kel_13/Diabetic-Rethinography"
test "$(git branch --show-current)" = novelty || { echo 'Checkout KC harus pada branch novelty'; exit 1; }
"$HOME/Kel_13/.venv/bin/python" -m novelty.preof_phase1 analyze --artifact-root "$HOME/Kel_13" --cache-dir "$HOME/Kel_13/preof_phase1/aptos_full" --output-dir "$HOME/Kel_13/preof_phase1/aptos_e0" --folds 5 --bootstrap 500
"$HOME/Kel_13/.venv/bin/python" -m novelty.preof_phase1 analyze --artifact-root "$HOME/Kel_13" --cache-dir "$HOME/Kel_13/preof_phase1/ddr_full" --output-dir "$HOME/Kel_13/preof_phase1/ddr_e0" --folds 5 --bootstrap 500
```

**Catatan historis sebelum eksekusi KC:** pada tahap rancangan Phase 1 belum ada skor E0. Kriteria GO saat itu mensyaratkan audit preflight dan tambahan informasi C terhadap B yang konsisten, tanpa degradasi kalibrasi substansial. Hasil KC dan keputusan terkini dicatat pada bagian berikutnya.

## 6. Status E0 dan analisis kegagalan prediksi E1

**Pembaruan yang dilaporkan operator KC:** preflight dan cache lengkap berhasil untuk APTOS2019 (366 gambar validation) dan DDR (1.252), lalu E0 grouped 5-fold OOF selesai. Perbedaan native expert membantu deteksi kesalahan secara deskriptif, tetapi tambahan respons lintas preprocessing tidak menunjukkan peningkatan AUPRC yang meyakinkan: seluruh 16 interval bootstrap berpasangan C−B mencakup nol dan dua kondisi boundary/expert memperburuk Brier. Karena itu keputusan saat ini **NO-GO untuk implementasi penuh M2–M4**. Nilai E0 individual dan file laporannya masih berada di KC; ringkasan ini berasal dari laporan operator, bukan audit ulang file KC secara lokal.

[`preof_e1.py`](../novelty/preof_e1.py) hanya membaca cache prediksi Level 2 yang lengkap dan manifest validation beku. Tidak ada pemrosesan gambar mentah, pemuatan checkpoint, GPU, training, atau akses final test. Skema, jumlah baris, urutan gambar, hash manifest/source/cache, definisi historis V2, serta empat logit dan probabilitas setiap baris diverifikasi sebelum agregasi. Hash checkpoint dan record diperiksa terhadap metadata cache dan spesifikasi Phase 1; E1 **tidak meng-hash ulang byte checkpoint KC** dan metadata itu sendiri bukan bukti kriptografis eksekusi. Output ditulis ke direktori **baru di luar Git, dataset mentah, dan cache**; cache lama tidak diubah.

E1 menghitung confusion matrix dan metrik lima kelas untuk `p11=V1(V1)`, `p12=V1(V2 BG10)`, `p21=V2(V1)`, `p22=V2(V2 BG10)`. Perbandingan **dalam satu model** adalah `p11→p12` dan `p22→p21`, masing-masing memakai checkpoint tetap. Matriks transisi memakai grade prediksi native sebagai baris dan grade prediksi cross sebagai kolom. Jumlah correct→wrong, wrong→correct, perubahan grade prediksi, per-grade akurasi dan perubahan probabilitas, over/undergrading, MAE grade, QWK, dan kesalahan berkeyakinan tinggi dilaporkan sebagai agregat. Ambang keyakinan default 0,90 bersifat deskriptif dan belum dikalibrasi. Bootstrap berpasangan mengambil ulang `group_id` validation untuk interval persentil 95% dari Δ akurasi, Δ MAE, dan keuntungan oracle native. Oracle menggunakan label benar untuk memilih expert dan **hanya batas atas**, bukan algoritme seleksi yang dapat dipakai pada pasien baru. Perbandingan native V1 vs V2 juga mencerminkan perbedaan training; kontribusi preprocessing tidak dapat diisolasi dari perbandingan antar model.

Lima output per dataset: `e1_results.json`, `e1_report.md`, `confusion_matrices.svg`, `grade_transitions.svg`, dan `per_grade_accuracy_delta.svg`. Grafik SVG menyimpan angka agregat dan label sumbu tanpa dependency plotting baru. Tidak ada ID gambar, gallery, atau contoh fundus pada laporan E1. Dengan bukti E0 saat ini, E1 bersifat eksploratif dan **tidak membatalkan NO-GO M2–M4**. Group bootstrap tidak mengatasi ketidakpastian label, dukungan grade minoritas, perubahan domain, atau validasi klinis eksternal.

**Perintah manual KC**, setelah source `novelty` terbaru tersedia pada checkout server. Keduanya CPU-only dan memakai cache prediksi penuh yang sudah ada:

```bash
cd "$HOME/Kel_13/Diabetic-Rethinography"
test "$(git branch --show-current)" = novelty || { echo 'Gunakan checkout novelty'; exit 1; }
"$HOME/Kel_13/.venv/bin/python" -m novelty.preof_e1 --dataset APTOS2019 --artifact-root "$HOME/Kel_13" --cache-dir "$HOME/Kel_13/preof_phase1/aptos_full" --output-dir "$HOME/Kel_13/preof_phase1/aptos_e1" --bootstrap 500 --seed 42 --high-confidence 0.90
"$HOME/Kel_13/.venv/bin/python" -m novelty.preof_e1 --dataset DDR --artifact-root "$HOME/Kel_13" --cache-dir "$HOME/Kel_13/preof_phase1/ddr_full" --output-dir "$HOME/Kel_13/preof_phase1/ddr_e1" --bootstrap 500 --seed 42 --high-confidence 0.90
```

Jika direktori output telah ada, pilih nama baru; script menolak penimpaan. Tinjau JSON, Markdown, dan SVG yang dihasilkan sebelum menarik kesimpulan ilmiah. **Tidak ada hasil E1 KC yang dibuat atau diklaim dalam perubahan lokal ini.**

## 7. Audit titik ekstraksi fitur — rancangan saja

Source otoritatif untuk kandidat ini adalah [`model.py`](../reproduction/cadvit/model.py), yang dibandingkan lewat AST dengan notebook historis oleh tes yang ada. Input model adalah tensor `[B,3,512,512]`. Bentuk di bawah **diturunkan dari konfigurasi kanal dan stride**, bukan hasil ekstraksi fitur checkpoint KC; `forward_features` mengembalikan tensor `head` sebelum `pool` dan `classifier`.

| Titik setelah modul | Bentuk per gambar pada 512² | Usulan ringkasan | FP16 keluaran per gambar | Estimasi FP16 untuk 1.618 validation × 4 kondisi |
| --- | --- | --- | ---: | ---: |
| `stem` (awal) | `[32,256,256]` | global average + opsi spatial pooling 4×4 | 4,00 MiB | 25,28 GiB |
| `blocks[3]` / `IRB+A-2` (awal-menengah) | `[96,64,64]` | global average + spatial pooling 4×4 | 0,75 MiB | 4,74 GiB |
| `blocks[6]` / `EMB-2` (menengah) | `[128,32,32]` | global average + spatial pooling 2×2/4×4 | 0,25 MiB | 1,58 GiB |
| `blocks[8]` / `EMB-3` (penultimate spatial) | `[160,16,16]` | global average atau spatial pooling 4×4 | 0,078 MiB | 0,49 GiB |
| `head` (sebelum classifier pooling) | `[320,16,16]` | `pool` bawaan/global average | 0,156 MiB | 0,99 GiB |
| `pool(features).flatten(1)` (pre-classifier) | `[320]` | vektor 320 dimensi | 640 byte | ~3,95 MiB |

Estimasi peta penuh mengabaikan header, indeks, dan kompresi; empat kondisi berarti dua checkpoint × dua preprocessing. Menyimpan vektor pooled jauh lebih hemat daripada peta spasial. Untuk audit fitur kelak, gunakan hook pada modul bernama atau `forward_features` tanpa mengubah source historis, dengan `eval()` dan inference mode. Cache fitur baru sebaiknya content-addressed oleh SHA gambar, hash manifest dan split validation, hash checkpoint, hash model/preprocessing, versi library, titik modul, strategi pooling, dtype, bentuk tensor, serta versi skema. Catat urutan baris dan checksum array, tulis secara atomik, dan verifikasi sampling ulang terhadap ekstraksi langsung. Jangan membuatnya di dalam Git atau direktori dataset. Tidak ada implementasi atau inferensi fitur pada tahap ini.

`logits = W h + b` untuk vektor pooled `h ∈ ℝ³²⁰`. Semua transformasi yang hanya memakai logits/probabilitas, termasuk probabilitas kumulatif ordinal, adalah fungsi deterministik dari keluaran yang sudah ada; proyeksi linear dari `h` dengan bobot di ruang baris `W` juga tidak menambah informasi. Proyeksi yang memakai komponen `h` di luar ruang baris `W`, atau fitur spasial sebelum pooling, **mungkin** membawa informasi tambahan, tetapi belum tentu terkait keparahan atau meningkatkan generalisasi. Klaim tersebut memerlukan evaluasi validation yang terkontrol, pembanding berbasis logits, dan pemeriksaan kebocoran sebelum metode fitur layak dikembangkan. Menggabungkan teknik yang sudah dikenal saja bukan klaim novelty.
