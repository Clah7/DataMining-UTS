# Tugas Data Mining Clustering Citra Udara

## Deskripsi
Proyek ini bertujuan untuk menerapkan teknik clustering pada citra udara menggunakan algoritma K-Means. Dalam proyek ini, kami akan mengolah dan menganalisis gambar udara untuk menemukan pola dan warna dominan, serta melakukan segmentasi berdasarkan warna.

## Anggota Kelompok
- **140810220036 - Alif Al Husaini**
- **140810220043 - Darren Christian Liharja**
- **140810220051 - Jason Natanael Krisyanto**

## Instalasi

### Prasyarat
Pastikan Anda memiliki Python terinstal pada sistem Anda. Kami merekomendasikan menggunakan virtual environment untuk menghindari konflik dengan paket lain.

### Menginstal Dependensi
Untuk menginstal semua dependensi yang diperlukan, buat file `requirements.txt` dengan konten berikut dan jalankan:

```bash
pip install -r requirements.txt
```

Isi dari `requirements.txt`:

```
streamlit
Pillow
matplotlib
numpy
opencv-python-headless
```

## Cara Menjalankan
1. Clone repositori ini ke mesin lokal Anda.
2. Buka terminal dan navigasikan ke direktori proyek.
3. Jalankan aplikasi Streamlit dengan perintah:

```bash
streamlit run app.py
```

## Fitur
- **Upload Gambar**: Pengguna dapat mengunggah hingga 5 gambar citra udara.
- **Analisis Warna Dominan**: Menghitung dan menampilkan 5 warna dominan dari gambar yang diunggah.
- **Segmentasi Gambar**: Melakukan segmentasi gambar berdasarkan clustering warna menggunakan algoritma K-Means.