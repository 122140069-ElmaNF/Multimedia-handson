# Real-time Remote Photoplethysmography (rPPG) Heart Rate Detection

**Mata Kuliah:** Sistem Teknologi Multimedia  

## 📋 Deskripsi Proyek

Sistem deteksi detak jantung secara real-time menggunakan kamera webcam tanpa kontak fisik, mengimplementasikan teknologi Remote Photoplethysmography (rPPG) dengan berbagai peningkatan kualitas.

## ✨ Fitur Utama

### 1. **Pipeline rPPG Lengkap**
- ✅ Deteksi wajah menggunakan **MediaPipe Face Mesh** (468 landmarks)
- ✅ Ekstraksi Region of Interest (ROI) pada area dahi
- ✅ Pemrosesan sinyal dengan detrending dan bandpass filter
- ✅ Estimasi BPM menggunakan FFT (Fast Fourier Transform)

### 2. **Implementasi Real-time**
- ✅ Input langsung dari webcam
- ✅ Sliding window buffer (300 frames / 10 detik)
- ✅ Update BPM setiap 1 detik (30 frames)
- ✅ Frame rate optimization untuk performa smooth

### 3. **Peningkatan Kualitas (Improvements)**

#### **A. POS (Plane-Orthogonal-to-Skin) Algorithm**
- Metode ekstraksi sinyal yang lebih mutakhir dari green channel sederhana
- **Keunggulan:** Robust terhadap motion artifacts dan perubahan pencahayaan
- **Implementasi:** Menggunakan kombinasi ketiga channel RGB dengan proyeksi ortogonal untuk mengeliminasi komponen specularity

#### **B. Advanced ROI Selection**
- Menggunakan landmark detection MediaPipe untuk ROI dinamis
- Fokus pada area **forehead** (dahi) yang memiliki densitas pembuluh darah tinggi
- Mask-based extraction untuk isolasi pixel kulit

#### **C. Motion Artifact Rejection**
- Deteksi gerakan berlebihan berdasarkan variance sinyal
- Sistem memberikan warning ketika terdeteksi gerakan
- Mencegah estimasi BPM yang tidak akurat

#### **D. Adaptive Signal Processing**
- Detrending menggunakan **smoothness priors approach**
- Butterworth bandpass filter order 6 (0.67 - 3.0 Hz / 40-180 BPM)
- Signal normalization untuk stabilitas

#### **E. Confidence-based Estimation**
- Perhitungan confidence score berdasarkan peak prominence di spektrum FFT
- Median filtering pada BPM estimates (buffer 10 readings)
- Hanya update BPM jika confidence > 0.3

#### **F. Visualisasi Real-time Informatif**
- **Plot 1:** Raw signal (sinyal mentah dari ekstraksi)
- **Plot 2:** Filtered signal (setelah detrending dan bandpass)
- **Plot 3:** Frequency spectrum dengan marker BPM
- Confidence bar untuk indikator kualitas deteksi
- ROI overlay pada frame video

## 🚀 Instalasi

### Requirements
```bash
pip install opencv-python mediapipe numpy scipy matplotlib
```

### Dependencies
- Python 3.8+
- OpenCV (cv2) >= 4.5.0
- MediaPipe >= 0.10.0
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0

## 📖 Cara Penggunaan

### Menjalankan Program

```bash
python rppg.py
```

### Kontrol Keyboard
- **`q`**: Keluar dari program
- **`r`**: Reset buffer (mulai deteksi dari awal)
- **`1`**: Untuk memilih area dahi
- **`2`**: Untuk memilih area pipi kiri
- **`3`**: Untuk memilih area pipi kanan

### Instruksi Penggunaan
1. Pastikan pencahayaan ruangan cukup (tidak terlalu gelap/terang)
2. Posisikan wajah di tengah frame kamera
3. **Tetap diam** selama 5-10 detik pertama untuk pengumpulan data
4. Hindari gerakan kepala yang berlebihan
5. Tunggu hingga confidence bar berwarna hijau (>0.5)

## 🔬 Metodologi Teknis

### Pipeline Processing

```
Webcam Input (30fps)
    ↓
MediaPipe Face Detection (468 landmarks)
    ↓
ROI Extraction (Forehead region mask)
    ↓
Spatial Averaging (RGB channels)
    ↓
POS Signal Processing (R, G, B → S signal)
    ↓
Detrending (Smoothness Priors, λ=100)
    ↓
Bandpass Filter (Butterworth order 6, 0.67-3.0 Hz)
    ↓
FFT Analysis (Find dominant frequency)
    ↓
BPM Calculation (freq × 60)
    ↓
Median Smoothing (10-sample buffer)
    ↓
Display + Visualization
```

### Parameter Konfigurasi

| Parameter | Nilai | Alasan |
|-----------|-------|--------|
| **Window Size** | 300 frames | 10 detik data @ 30fps untuk stabilitas |
| **Update Rate** | 30 frames | Update setiap 1 detik untuk responsiveness |
| **Filter Order** | 6 | Balance antara atenuasi dan phase distortion |
| **BPM Range** | 40-180 | Range fisiologis manusia |
| **ROI** | Forehead | Area dengan SNR terbaik |
| **Confidence Threshold** | 0.3 | Minimum untuk accept BPM estimate |

## 📊 Aspek Pembeda dengan Demo Kelas

### 1. **Metode Ekstraksi Sinyal**
- **Demo Kelas:** Simple green channel extraction
- **Implementasi Ini:** **POS Algorithm** - lebih robust terhadap motion dan illumination changes

### 2. **ROI Detection**
- **Demo Kelas:** Static bounding box / manual ROI
- **Implementasi Ini:** **Dynamic landmark-based ROI** menggunakan MediaPipe (36 landmarks untuk forehead)

### 3. **Motion Handling**
- **Demo Kelas:** Tidak ada handling khusus
- **Implementasi Ini:** **Motion artifact detection** dengan variance-based rejection

### 4. **Signal Processing**
- **Demo Kelas:** Simple moving average detrending
- **Implementasi Ini:** **Smoothness priors detrending** + Butterworth filter order 6

### 5. **BPM Estimation**
- **Demo Kelas:** Peak detection atau basic FFT