# 🌾 Sistem Rekomendasi Tanaman Berdasarkan Kondisi Tanah dan Iklim

## 📋 Deskripsi Aplikasi

Aplikasi web ini dibangun menggunakan **Streamlit** untuk memberikan rekomendasi tanaman terbaik berdasarkan kondisi tanah dan iklim tertentu. Sistem menggunakan algoritma **Machine Learning** (Random Forest) untuk menganalisis parameter-parameter lingkungan dan memberikan prediksi tanaman yang paling sesuai.

## 🚀 Fitur Utama

- **🔍 Prediksi Tanaman** - Memprediksi tanaman terbaik berdasarkan input parameter
- **📊 Analisis Data** - Menampilkan statistik dataset dan distribusi tanaman
- **🌱 Rekomendasi Multi-Kriteria** - Mempertimbangkan 7 parameter berbeda
- **📱 Responsive Design** - Tampilan yang optimal di berbagai perangkat

## 🛠 Teknologi yang Digunakan

- **Frontend**: Streamlit
- **Machine Learning**: Scikit-learn (Random Forest Classifier)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Streamlit native components

## 📊 Parameter Input

Aplikasi mempertimbangkan 7 parameter utama:

| Parameter | Deskripsi | Rentang |
|-----------|-----------|---------|
| **Nitrogen (N)** | Kandungan nitrogen dalam tanah | 0-140 |
| **Phosphorus (P)** | Kandungan fosfor dalam tanah | 5-145 |
| **Potassium (K)** | Kandungan kalium dalam tanah | 5-205 |
| **Suhu** | Suhu rata-rata harian | 5-45°C |
| **Kelembaban** | Tingkat kelembaban relatif | 10-100% |
| **pH Tanah** | Tingkat keasaman tanah | 3.0-10.0 |
| **Curah Hujan** | Curah hujan bulanan | 20-300 mm |

## 🎯 Cara Menggunakan

1. **Buka aplikasi** di browser
2. **Masukkan parameter** di sidebar:
   - Nilai N, P, K
   - Slider suhu, kelembaban, pH, dan curah hujan
3. **Klik tombol "Prediksi Tanaman"**
4. **Lihat hasil rekomendasi** tanaman yang muncul
5. **Explore data** tambahan di section statistik

## 📁 Struktur Project
