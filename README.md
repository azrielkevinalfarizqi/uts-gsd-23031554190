# UTS Graf Sains Data: Rekomendasi Produk dengan Neo4j dan KNN

Proyek ini membangun graf interaksi pengunjung dan produk dari dataset Retail Rocket, mengeksplorasinya di Neo4j lewat Python driver, lalu memakai graf itu untuk merekomendasikan produk dengan K-Nearest Neighbors (KNN). Pertanyaan yang ingin dijawab sederhana: kalau kita hanya tahu apa yang dilihat, dimasukkan ke keranjang, dan dibeli pengunjung di masa lalu, seberapa baik kita menebak apa yang akan mereka sentuh berikutnya?

| | |
|---|---|
| **Matakuliah** | Graf Sains Data (3 SKS) |
| **Program Studi** | S1 Sains Data, FMIPA, Universitas Negeri Surabaya |
| **Semester** | Gasal 2026/2027 |
| **Materi** | Python Driver untuk Neo4j dan Rekomendasi |
| **Sifat** | Individu |
| **Nama / NIM** | `[isi nama]` / `[isi NIM]` |

## Ringkasan Hasil

| Bagian | Temuan utama |
|---|---|
| Data training (P1) | 1.324.774 interaksi |
| Data testing (P2) | 279.544 interaksi |
| Subgraph P1 | 684.227 Visitor, 171.126 Item, 3 jenis relasi |
| Proyeksi yang dipilih | Proyeksi User (user-based collaborative filtering) |
| Similarity tertinggi | 0,2947 (Visitor 1119917 dan 917504), cosine similarity |
| Hit@5 rata-rata | 0,0556 pada 36 User |
| NDCG@5 rata-rata | 0,0195 pada 36 User |

Angka evaluasinya rendah, dan itu jujur memang rendah. Bagian [Keterbatasan](#keterbatasan) menjelaskan kenapa.

## Struktur Repositori

```
.
├── README.md
├── requirements.txt
├── notebook_uts.ipynb          # notebook utama (sesuaikan nama berkas)
├── data/
│   ├── events.csv              # data mentah Retail Rocket (tidak di-commit jika terlalu besar)
│   ├── events_p1.csv           # hasil filter periode training
│   └── events_p2.csv           # hasil filter periode testing
└── images/
    ├── sebelum similarity.svg
    └── setelah similarity.svg
```

## Persiapan

### 1. Environment

```bash
uv venv myvenv --python 3.10
source myvenv/bin/activate        # Windows: myvenv\Scripts\activate
uv pip install -r requirements.txt
```

Alternatifnya `python -m venv myenv`. Virtual environment dipakai supaya paket proyek ini tidak bentrok dengan proyek lain di laptop.

### 2. Neo4j

Pastikan instance Neo4j berjalan dan bisa diakses lewat Bolt. Nilai bawaan yang dipakai:

```python
URI  = "neo4j://127.0.0.1:7687"
AUTH = ("neo4j", "password")      # ganti sesuai kredensial kamu
```

### 3. Dataset

Unduh dari Kaggle ([Retail Rocket e-commerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)), lalu letakkan `events.csv` di folder `data/`.

```bash
curl -L -o ecommerce-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/retailrocket/ecommerce-dataset
```

Kolom yang dipakai: `timestamp` (unix epoch, milidetik), `visitorid`, `event` (`view`, `addtocart`, `transaction`), `itemid`, dan `transactionid` (hanya terisi pada event `transaction`).

> **Peringatan:** perintah `MATCH (n) DETACH DELETE n;` menghapus seluruh isi database. Jangan jalankan di database bersama.

## Cara Menjalankan

Jalankan sel notebook secara berurutan. Alurnya:

1. Filter `events.csv` menjadi `events_p1.csv` dan `events_p2.csv`.
2. Impor data P1 ke Neo4j secara batch (10.000 baris per batch, dengan `UNWIND` dan `MERGE`).
3. Hitung komposisi subgraph, bangun VSM dan proyeksi, hitung similarity.
4. Latih KNN pada P1, rekomendasikan, lalu evaluasi terhadap P2.

## Model Graf

```
(:Visitor)-[:VIEWED {timestamp}]->(:Item)
(:Visitor)-[:ADDED_TO_CART {timestamp}]->(:Item)
(:Visitor)-[:BOUGHT {timestamp, transactionid}]->(:Item)
(:Visitor)-[:SIMILAR_TO {similarity}]-(:Visitor)     # ditambahkan setelah perhitungan similarity
```

Ini graf bipartit: relasi hanya menghubungkan Visitor ke Item, tidak pernah Visitor ke Visitor atau Item ke Item, sampai relasi `SIMILAR_TO` ditambahkan.

Bobot interaksi yang dipakai di seluruh proyek:

| Event | Bobot |
|---|---:|
| `view` | 1 |
| `addtocart` | 3 |
| `transaction` | 5 |

## Bagian A. Wrangling dan Eksplorasi Graf

### A.1 Periode waktu

| Fold | Periode | Baris |
|---|---|---:|
| P1 (training) | 31 Mei 2015 sampai 31 Juli 2015 | 1.324.774 |
| P2 (testing) | 9 Agustus 2015 sampai 24 Agustus 2015 | 279.544 |

Batas akhir bersifat eksklusif (P1 sebelum 1 Agustus 2015, P2 sebelum 25 Agustus 2015). Ada jeda sekitar delapan hari di antara keduanya, dan itu disengaja agar P2 benar-benar berperan sebagai data masa depan. Pemfilteran memakai Pandas dengan chunk 200.000 baris supaya memori tetap aman.

### A.2 Komposisi subgraph

| Elemen | Jumlah |
|---|---:|
| Simpul Visitor | 684.227 |
| Simpul Item | 171.126 |
| Relasi VIEWED | 1.281.023 |
| Relasi ADDED_TO_CART | 32.807 |
| Relasi BOUGHT | 10.944 |
| **Total relasi** | **1.324.774** |

Setiap hitungan memakai pola Cypher yang sama, misalnya:

```cypher
MATCH (v:Visitor) RETURN count(v) AS jumlah;
MATCH ()-[r:VIEWED]->() RETURN count(r) AS jumlah;
```

Dari 1,3 juta interaksi, 96,7% hanya `view`. Pembelian cuma 0,83%. Artinya sinyal yang paling bernilai juga yang paling langka.

### A.3 Vector Space Model

Agar komputasi tetap ringan, VSM dibangun dari 200 Visitor dengan relasi terbanyak.

| Karakteristik | VSM User | VSM Item |
|---|---:|---:|
| Bentuk matriks | 200 × 22.479 | 22.479 × 200 |
| Dimensi (ndim) | 2 | 2 |
| Nilai nonzero | 43.319 | 43.319 |
| Sparsity | ≈ 99,04% | ≈ 99,04% |

VSM Item adalah transpos dari VSM User, jadi isinya identik. Distribusi bobot per Item berekor panjang: rata-rata ≈ 4,35, maksimum 591. Sebagian besar Item hanya disentuh satu atau dua User, sementara segelintir Item sangat populer.

Proyeksi pada himpunan User dari VSM ini menghasilkan 200 simpul dan 9.202 edge (kepadatan ≈ 0,462, rata-rata derajat ≈ 92). Proyeksi pada Item dibatasi ke 200 Item teratas dan menghasilkan 200 simpul dengan 12.970 edge. Dua simpul dalam proyeksi terhubung jika punya tetangga yang sama di sisi lain graf, dan bobot edge adalah jumlah tetangga bersama.

### A.4 Proyeksi mana yang lebih baik

Perbandingan memakai subgraf bipartit yang sama (200 User dan 200 Item teratas), waktu dirata-ratakan dari 5 percobaan.

| Karakteristik | Proyeksi User | Proyeksi Item |
|---|---:|---:|
| Simpul | 200 | 200 |
| Edge | 5.433 | 12.970 |
| Kepadatan | 0,27302 | 0,65176 |
| Rata-rata derajat | 54,33 | 129,70 |
| Waktu pembentukan (detik) | 1,20588 | 0,86332 |

Proyeksi Item lebih padat karena Item populer dikunjungi banyak orang, jadi hampir semua pasangan Item teratas punya pengunjung bersama. Graf sepadat itu kurang informatif dan mahal untuk dianalisis lebih lanjut. Selisih waktu 0,34 detik terlalu kecil untuk dijadikan alasan, dan bisa saja hanya noise komputasi.

Proyeksi User dipilih. Alasannya bukan karena ia unggul di semua sisi, tetapi karena lebih renggang dan langsung cocok untuk user-based collaborative filtering, yang menjadi dasar KNN di Bagian B.

### A.5 Similarity

Cosine similarity dihitung antar Visitor, dengan setiap Visitor direpresentasikan sebagai vektor Item berbobot (bobot tertinggi per Item). Dari 200 Visitor dengan Item berbeda terbanyak, diperiksa 19.900 pasangan; 10.106 di antaranya punya similarity di atas 0. Nilai tertinggi hanya ≈ 0,29, tanda bahwa irisan Item antar pengunjung kecil.

Lima pasangan teratas:

| No. | Visitor A | Visitor B | Similarity |
|---:|---:|---:|---:|
| 1 | 1119917 | 917504 | 0,294671 |
| 2 | 1336835 | 518659 | 0,293480 |
| 3 | 1176826 | 524146 | 0,291267 |
| 4 | 317355 | 360921 | 0,278605 |
| 5 | 590943 | 856874 | 0,269892 |

Query untuk mengambilnya dari relasi `SIMILAR_TO`:

```cypher
MATCH (a:Visitor)-[r:SIMILAR_TO]-(b:Visitor)
WHERE a <> b
WITH DISTINCT
    CASE WHEN toString(a.visitorId) < toString(b.visitorId) THEN a.visitorId ELSE b.visitorId END AS visitor_a,
    CASE WHEN toString(a.visitorId) < toString(b.visitorId) THEN b.visitorId ELSE a.visitorId END AS visitor_b,
    r.similarity AS similarity
RETURN visitor_a, visitor_b, similarity
ORDER BY similarity DESC
LIMIT 5;
```

Blok `CASE` mengurutkan pasangan secara kanonis, sehingga pasangan (A, B) dan (B, A) tidak muncul dua kali.

> Catatan: 200 Visitor pada tahap similarity dipilih berdasarkan jumlah Item berbeda, sedangkan VSM sebelumnya berdasarkan jumlah relasi. Himpunannya bisa tidak persis sama.

### A.6 Graf sebelum dan sesudah similarity

Sebelum (graf bipartit Visitor dan Item):

```cypher
MATCH (v:Visitor)-[r]-(i:Item)
RETURN v, r, i
LIMIT 50;
```

![Sebelum similarity](<images/sebelum similarity.svg>)

Sesudah (relasi `SIMILAR_TO` antar Visitor):

```cypher
MATCH (a:Visitor)-[r:SIMILAR_TO]-(b:Visitor)
RETURN a, r, b
ORDER BY r.similarity DESC
LIMIT 20;
```

![Setelah similarity](<images/setelah similarity.svg>)

## Bagian B. Analytics

### B.1 Rekomendasi KNN (K = 5)

Model dilatih hanya pada P1. Ruang pencariannya 3.000 User kandidat yang dipilih acak dengan seed 42 dari User yang punya minimal 2 Item berbeda di P1 (8.905 Item, 10.946 interaksi User-Item). Seed tetap membuat pemilihan User bisa direproduksi.

Langkah inti pada kode:

1. `NearestNeighbors` dengan `metric="cosine"` dan `algorithm="brute"` dilatih pada matriks User × Item.
2. Cari 5 tetangga terdekat, User target dikeluarkan dari hasil.
3. Skor tiap Item = jumlah interaksi tetangga dikali similarity tetangga tersebut (`1 - cosine distance`).
4. Item yang sudah pernah diakses User di-skip.
5. Jika tidak ada skor positif, rekomendasi jatuh ke popularitas (fallback).
6. Top-5 menjadi rekomendasi, Top-3 menjadi iklan.

User terpilih: **Visitor 95611**, dari 36 User yang memenuhi syarat evaluasi. Tidak memakai fallback popularitas.

| Peringkat | Item ID | Diiklankan |
|---:|---:|:---:|
| 1 | 464998 | Ya |
| 2 | 12492 | Ya |
| 3 | 358180 | Ya |
| 4 | 89639 | Tidak |
| 5 | 262699 | Tidak |

### B.2 Evaluasi

Item relevan adalah Item yang muncul di P2 untuk User tersebut, ada di himpunan Item training, dan belum pernah diakses di P1. Hanya 36 dari target 100 User yang memenuhi syarat ini.

$$\text{Hit@5} = \begin{cases} 1 & \text{jika ada minimal satu Item relevan di Top-5} \\ 0 & \text{jika tidak ada} \end{cases}$$

$$\text{DCG@5} = \sum_{i=1}^{5} \frac{rel_i}{\log_2(i+1)}, \qquad \text{NDCG@5} = \frac{\text{DCG@5}}{\text{IDCG@5}}$$

| Metrik | Visitor 95611 | Rata-rata 36 User |
|---|---:|---:|
| Hit@5 | 0,0000 | 0,0556 |
| NDCG@5 | 0,0000 | 0,0195 |

Untuk Visitor 95611 tidak ada Item di Top-5 yang muncul pada P2, sehingga kedua metrik bernilai 0. Secara rata-rata, 2 dari 36 User (5,6%) mendapat minimal satu tebakan tepat. Sebagai gambaran NDCG, Visitor 648787 mendapat ≈ 0,387 karena satu Item relevan jatuh tepat di peringkat 5 (1/log₂6).

## Keterbatasan

Hasil evaluasi rendah, dan penyebabnya bisa dirunut:

- **Sparsity ≈ 99%.** Kebanyakan User hanya punya sedikit Item, jadi tetangga jarang berbagi Item yang kelak disentuh User target.
- **Skala yang dipangkas.** Hanya 3.000 User kandidat dan 5 tetangga. Semakin sedikit tetangga, semakin kecil peluang ada Item yang cocok.
- **Fallback popularitas.** Sebagian User (3 dari 10 contoh pertama) memakai fallback karena tetangganya tidak memberi skor positif, sehingga rekomendasinya tidak lagi personal.
- **Definisi relevan yang longgar.** Item relevan di P2 dihitung dari semua jenis event, termasuk `view` biasa, padahal minat sebenarnya lebih jelas terlihat dari `addtocart` dan `transaction`.
- **Jeda waktu.** Selera User bisa bergeser antara P1 dan P2.
- **Sampel evaluasi kecil.** Dengan 36 User, selisih satu User saja mengubah Hit@5 sekitar 0,028. Angkanya belum cukup stabil untuk membandingkan model secara tegas.

Arah perbaikan yang masuk akal: menaikkan jumlah tetangga dan User kandidat, mencoba pendekatan item-based atau hybrid dengan popularitas, dan membatasi target relevan pada `addtocart` dan `transaction`.

## Reproduksibilitas

- Pemilihan User kandidat dan User target memakai `seed = 42`.
- Filter periode dan impor bersifat deterministik selama `events.csv` sama.
- Waktu pembentukan proyeksi dirata-ratakan dari 5 percobaan dan dapat bervariasi antar mesin.

## Tautan

- Repositori GitHub: `[isi URL]`
- Dataset: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset (sumber data: Roman Zykov)
