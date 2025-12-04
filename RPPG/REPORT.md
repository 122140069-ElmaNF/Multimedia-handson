# Laporan Singkat: Real-time rPPG Heart Rate Detection

**Penulis:**
Tim / Pengembang: (Isi nama Anda)

**Tanggal:** 2025-12-01

---

## Ringkasan
Proyek ini mengimplementasikan deteksi detak jantung real‑time berbasis remote photoplethysmography (rPPG) menggunakan kamera web. Sistem mendeteksi wajah (MediaPipe Face Mesh), mengekstrak sinyal warna kulit (POS — Plane-Orthogonal-to-Skin), melakukan pemfilteran adaptif, dan menampilkan estimasi BPM secara real‑time dengan visualisasi sinyal.

## Tujuan
- Mengukur detak jantung melalui kamera (tanpa sensor kontak).
- Menyediakan antarmuka real‑time yang menampilkan video + grafik sinyal + status BPM.
- Menggunakan metode POS untuk robust signal extraction dan filtering untuk mengurangi artefak gerakan/lingkungan.

## Metode dan Komponen Utama
- Face detection: MediaPipe Face Mesh (landmark/ROI untuk daerah dahi).
- Sinyal RGB: rata-rata spasial pada ROI setiap frame disimpan dalam buffer gulir (sliding window).
- Algoritma POS: implementasi CPU POS (Wang et al., 2016) — memproses window berukuran 1.6 detik dengan overlap dan overlap‑add untuk sinyal kontinu.
- Filtering: Butterworth bandpass adaptif berdasarkan rentang BPM yang diinginkan (default 40–180 BPM).
- Estimasi frekuensi: FFT pada segmen sinyal untuk menemukan frekuensi dominan dan konversi ke BPM.
- Smoothing: median buffer untuk stabilisasi estimasi BPM.

## Implementasi (file utama)
- `rppg.py` — skrip utama yang menjalankan pipeline rPPG:
  - Kelas `RPPGDetector` menyimpan buffer, melakukan ekstraksi ROI, menerapkan POS, memfilter sinyal, dan memperkirakan BPM.
  - Kontrol runtime: `ESC` (exit), `R` (reset buffers), `M` (toggle mirror preview).
  - GUI: side‑by‑side video + signal canvas, status BPM dan bar confidence.
- `requirements.txt` — daftar dependensi minimal (numpy, opencv-python, mediapipe, scipy).

## Penggunaan
1. Siapkan environment Python dan install dependensi:
```bash
python -m pip install -r requirements.txt
```
2. Jalankan:
```bash
python rppg.py
```
3. Kontrol saat runtime:
- ESC: keluar
- R: reset buffer
- M: toggle mirror tampilan

## Hasil & Pengamatan
- Buffer default: `window_size=300` (≈ 10 detik @ 30fps) menyediakan riwayat panjang untuk estimasi FFT stabil.
- POS window internal menggunakan `w = 1.6 * fps` (≈ 1.6 detik), stride 1 frame → overlap penuh / smooth output.
- Estimasi BPM lebih stabil pada kondisi pencahayaan baik dan kepala relatif diam.

## Keterbatasan
- Sensitif terhadap gerakan wajah cepat dan pencahayaan buruk.
- Latensi tergantung ukuran buffer / window; buffer besar menambah stabilitas tapi meningkatkan delay.
- Performa CPU dapat menurun pada hardware lemah karena overlap‑add dan proses per‑frame.

## Rekomendasi & Pengembangan Selanjutnya
- Tambahkan argumen CLI untuk mengatur `mirror`, `window_size`, dan `pos_window_sec` agar lebih mudah tuning.
- Eksperimen dengan stride >1 untuk mengurangi beban CPU (kompromi antara kelancaran sinyal dan performa).
- Tambahkan logging / benchmarking untuk mengukur latensi end‑to‑end dan akurasi vs ground truth (sensor detak jantung).
- Peningkatan visual: tampilkan indikator ON/OFF mirror di antarmuka, dan opsi memilih ROI lain (pipi/area mata).

---

## Lampiran
- File utama: `rppg.py`
- Dependensi: `requirements.txt` (numpy, opencv-python, mediapipe, scipy)

Jika Anda mau, saya bisa:
- Menambahkan ringkasan hasil pengujian (screenshots / sample output) ke laporan ini, atau
- Mengubah structure report menjadi README yang lebih lengkap.

Selesai — ingin saya tambahkan nama Anda (penulis) dan tautan ke commit/penjelasan tambahan di README.md?