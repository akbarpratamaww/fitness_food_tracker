# 📖 Panduan Pengguna (Manual Book)
## Smart Fitness & Food Tracker

Selamat datang di **Smart Fitness & Food Tracker**! Aplikasi ini dirancang untuk membantu Anda memantau nutrisi harian, mencatat aktivitas olahraga, menganalisis tingkat kebugaran dengan Machine Learning, memprediksi berat badan di masa mendatang, serta berkonsultasi langsung dengan asisten kesehatan pintar (AI Chatbot).

Panduan ini akan menuntun Anda langkah demi langkah dalam menggunakan seluruh fitur aplikasi, dimulai sejak Anda pertama kali membuka link/tautan aplikasi.

---

## Daftar Isi
1. [Membuka Aplikasi & Langkah Awal: Setup Profil](#1-membuka-aplikasi--langkah-awal-setup-profil)
2. [Navigasi Menu Utama](#2-navigasi-menu-utama)
   - [Dashboard (Beranda)](#-dashboard-beranda)
   - [Food Log (Catatan Makanan & Analisis Apriori)](#-food-log-catatan-makanan--analisis-apriori)
   - [Activity Log (Catatan Aktivitas)](#-activity-log-catatan-aktivitas)
   - [Fitness Level Classifier (Klasifikasi Kebugaran)](#-fitness-level-classifier-klasifikasi-kebugaran)
   - [Progress & Weight Forecasting (Progres & Prediksi Berat Badan)](#-progress--weight-forecasting-progres--prediksi-berat-badan)
   - [AI Chatbot (FitBot)](#-ai-chatbot-fitbot)
   - [ML Predictor (Prediksi Kalori Terbakar)](#-ml-predictor-prediksi-kalori-terbakar)
   - [About (Tentang Aplikasi)](#-about-tentang-aplikasi)
3. [Tips Tambahan untuk Hasil Terbaik](#3-tips-tambahan-untuk-hasil-terbaik)

---

## 1. Membuka Aplikasi & Langkah Awal: Setup Profil

Saat Anda pertama kali mengakses tautan (link) aplikasi di browser Anda, Anda akan disambut oleh halaman beranda. Jika Anda belum pernah menggunakan aplikasi ini sebelumnya, sistem akan menampilkan peringatan:
`⚠️ Please complete your profile first!`

Anda harus mengisi profil terlebih dahulu agar aplikasi dapat menghitung kebutuhan kalori harian Anda secara akurat sesuai dengan parameter tubuh Anda.

### Cara Melakukan Setup Profil:
1. Klik tombol **Go to Profile** pada pesan peringatan, atau pilih menu **👤 Profile** di sidebar sebelah kiri.
2. Isi formulir profil Anda secara lengkap:
   * **Name**: Nama panggilan Anda.
   * **Age**: Umur Anda saat ini (minimal 15 tahun).
   * **Gender**: Jenis kelamin Anda (Male/Female).
   * **Height (cm)**: Tinggi badan Anda dalam satuan sentimeter.
   * **Weight (kg)**: Berat badan Anda saat ini dalam satuan kilogram.
   * **Activity Level**: Tingkat aktivitas fisik harian Anda (pilih tingkat yang paling menggambarkan aktivitas harian Anda, dari tidak aktif hingga sangat aktif).
   * **Fitness Goal**: Tujuan kebugaran Anda (Weight Loss / Maintain Weight / Muscle Gain).
3. Setelah seluruh kolom diisi, di bagian bawah formulir akan muncul kalkulasi metrik kesehatan Anda secara instan:
   * **BMR (Basal Metabolic Rate)**: Kebutuhan kalori minimal tubuh untuk bertahan hidup saat istirahat total.
   * **TDEE (Total Daily Energy Expenditure)**: Perkiraan kalori yang Anda bakar dalam sehari setelah memperhitungkan aktivitas fisik harian.
   * **Daily Target**: Target asupan kalori harian yang disesuaikan dengan **Fitness Goal** Anda (misal: defisit kalori untuk menurunkan berat badan, atau surplus untuk meningkatkan massa otot).
   * **BMI (Body Mass Index)**: Indeks Massa Tubuh beserta kategorinya (Underweight, Normal, Overweight, Obese).
4. Klik tombol **💾 Save Profile**. Profil Anda akan disimpan secara otomatis dan sistem akan menampilkan animasi balon tanda keberhasilan.

---

## 2. Navigasi Menu Utama
Setelah profil berhasil disimpan, Anda dapat mulai menggunakan seluruh fitur aplikasi dengan memilih menu di bilah navigasi kiri (Sidebar):

### 🏠 Dashboard (Beranda)
Halaman ini menyajikan ringkasan instan mengenai aktivitas harian Anda:
1. **Metrik Harian**:
   * **Calories In**: Jumlah kalori yang sudah Anda konsumsi hari ini dibandingkan dengan target harian Anda.
   * **Calories Out**: Jumlah kalori yang sudah Anda bakar lewat aktivitas olahraga hari ini.
   * **Net Balance**: Selisih kalori masuk dan keluar, serta sisa kalori yang masih boleh Anda konsumsi.
   * **BMI**: Kategori Indeks Massa Tubuh Anda saat ini.
2. **Weekly Calorie Summary**: Grafik batang interaktif yang membandingkan kalori masuk vs kalori keluar selama 7 hari terakhir.
3. **Recent Activities**: Menampilkan daftar makanan (`Recent Meals`) dan latihan fisik (`Recent Workouts`) terbaru yang Anda catat hari ini.

---

### 🍎 Food Log (Catatan Makanan & Analisis Apriori)
Di halaman ini, Anda dapat mencatat makanan dan menganalisis pola makan Anda. Terdapat 4 tab menu:

#### 1. Tab `📝 Log Food` (Pencatatan Makanan Pintar)
Mencatat makanan sangat mudah karena sistem mendukung deskripsi menggunakan bahasa sehari-hari.
* **Cara Menggunakan**:
  1. Tulis makanan Anda di kolom deskripsi. Contoh: `"2 slices of pizza"`, `"200g chicken breast"`, `"banana"`, atau `"rice"`.
  2. Pilih jenis waktu makan pada **Meal Type** (Breakfast, Lunch, Dinner, atau Snack).
  3. Klik **📝 Log Food**. Sistem akan otomatis mendeteksi nama makanan serta kandungan gizinya (kalori, protein, karbohidrat, lemak).

#### 2. Tab `🍽️ Meal Suggestions` (Rekomendasi Menu)
* Menampilkan sisa target kalori harian Anda.
* Memberikan saran makanan sehat yang cocok dikonsumsi sesuai dengan sisa kalori yang tersedia.

#### 3. Tab `📋 Food History` (Riwayat Konsumsi)
* Menampilkan riwayat makanan yang Anda konsumsi selama 30 hari terakhir dalam bentuk tabel detail.
* Menyajikan jumlah kalori keseluruhan selama satu bulan terakhir.

#### 4. Tab `🛒 Food Association Rules` (Analisis Pola Makan - Apriori)
Fitur ini menganalisis pola makanan yang sering Anda konsumsi bersamaan menggunakan **Algoritma Apriori**.
* **Cara Menggunakan**:
  1. **Data Source**: Pilih *"Sample Food History Dataset (Demo)"* untuk simulasi cepat, atau *"My Personal Food Logs"* untuk menganalisis riwayat makanan Anda sendiri.
  2. **Transaction Grouping**: Kelompokkan transaksi per hari atau per jenis waktu makan.
  3. **Minimum Support & Confidence**: Atur sensitivitas algoritma.
  4. Klik **🚀 Analyze Patterns**.
  5. Hasil analisis akan memunculkan daftar aturan asosiasi makanan (contoh: *"Jika Anda makan Roti, kemungkinan besar Anda juga makan Selai"*), beserta visualisasi **Top 10 Makanan Terpopuler**.

---

### 🏃 Activity Log (Catatan Aktivitas)
Catat latihan olahraga Anda untuk melacak kalori yang dibakar. Terdapat 2 tab menu:

#### 1. Tab `📝 Log Activity`
* **Cara Menggunakan**:
  1. Pilih jenis aktivitas pada **Activity Type** (misal: Running, Cycling, Swimming, Yoga, Weight Lifting).
  2. Masukkan durasi olahraga dalam menit.
  3. Pilih tingkat intensitas (Low, Medium, High).
  4. Kolom **Estimated Calories Burned** akan otomatis menghitung kalori terbakar berdasarkan standar nilai MET (Metabolic Equivalent of Task) dan berat badan Anda.
  5. Klik tombol **✅ Log Activity** untuk menyimpan aktivitas.

#### 2. Tab `📋 Activity History`
* Menampilkan seluruh riwayat latihan olahraga Anda selama 30 hari terakhir.

---

### 🏋️ Fitness Level Classifier (Klasifikasi Kebugaran)
Gunakan fitur ini untuk mengetahui tingkat kebugaran fisik Anda ke dalam kelas **A (Sangat Baik), B (Baik), C (Cukup), atau D (Kurang)** menggunakan model Machine Learning.

* **Cara Menggunakan**:
  1. Masukkan data fisik: Jenis Kelamin, Usia, Tinggi Badan, Berat Badan, dan Persentase Lemak Tubuh.
  2. Masukkan hasil tes kebugaran fisik Anda:
     * Tekanan darah (Diastolik & Sistolik).
     * Kekuatan genggaman tangan (`Grip Force`).
     * Kelenturan tubuh saat membungkuk (`Sit and Bend`).
     * Jumlah Sit-up yang dapat dilakukan dalam satu menit.
     * Jarak lompatan terjauh (`Broad Jump`).
  3. Pilih model Machine Learning yang ingin digunakan (Random Forest, XGBoost, atau SVM).
  4. Klik **🔍 Prediksi Tingkat Kebugaran**.
  5. Hasil klasifikasi, tingkat keyakinan model, serta rekomendasi olahraga penunjang akan ditampilkan di layar.

---

### 📈 Progress & Weight Forecasting (Progres & Prediksi Berat Badan)
Pantau grafik perkembangan tubuh Anda secara visual:

1. **Weight Progress (Perkembangan Berat Badan)**:
   * Menampilkan grafik garis fluktuasi berat badan Anda dari waktu ke waktu.
   * Masukkan berat badan terbaru Anda pada kolom di sebelah kanan lalu klik **📝 Record Weight** untuk memperbarui catatan berat badan.
2. **Weight Forecasting (ML — Linear Regression)**:
   * Memprediksi berat badan Anda ke depan (7, 14, atau 30 hari ke depan) berdasarkan tren riwayat catatan berat badan Anda sebelumnya menggunakan algoritma *Linear Regression*.
   * **Catatan**: Fitur prediksi ini membutuhkan minimal **2 catatan berat badan** dengan tanggal berbeda untuk menghasilkan estimasi tren.
3. **Calorie Trend Analysis**:
   * Grafik garis interaktif yang membandingkan tren Kalori Masuk (makanan), Kalori Keluar (olahraga), dan Kalori Bersih (Net) selama 30 hari terakhir.

---

### 🤖 AI Chatbot (FitBot)
FitBot adalah asisten virtual interaktif bertenaga AI yang siap menjawab semua pertanyaan seputar kebugaran dan nutrisi.

* **Cara Menggunakan**:
  1. Ketik pertanyaan Anda pada kotak input chat di bagian bawah (mendukung Bahasa Indonesia dan Bahasa Inggris).
  2. Tekan Enter. FitBot akan menganalisis profil kebugaran Anda dan memberikan saran personal yang spesifik untuk Anda.
  3. Anda juga dapat mengklik tombol pertanyaan cepat di sidebar kanan (**💡 Quick Questions**) seperti *"Give me a home workout plan"* untuk berkonsultasi secara instan.

---

### 📊 ML Predictor (Prediksi Kalori Terbakar)
Menggunakan model Machine Learning **Random Forest Regressor** untuk memprediksi kalori yang terbakar secara spesifik berdasarkan parameter fisiologis olahraga.

* **Cara Menggunakan**:
  1. Masukkan data diri Anda (Gender, Age, Height, Weight).
  2. Masukkan detail sesi latihan Anda: durasi latihan (menit), detak jantung rata-rata (bpm), dan suhu tubuh (°C).
  3. Klik **🔮 Predict Calories Burned** untuk memunculkan hasil estimasi kalori terbakar.

---

### ℹ️ About (Tentang Aplikasi)
Menu ini berisi informasi latar belakang aplikasi, rumus ilmiah yang digunakan (BMR Mifflin-St Jeor, TDEE, Rumus MET), serta status ketersediaan dataset model Machine Learning di server aplikasi.

---

## 3. Tips Tambahan untuk Hasil Terbaik
* **Catat Secara Rutin**: Lakukan pencatatan makanan dan aktivitas setiap hari agar grafik perkembangan Anda tetap akurat.
* **Timbang Badan Berkala**: Perbarui berat badan Anda seminggu sekali di menu **Progress** untuk mengaktifkan fitur prediksi tren berat badan (*Weight Forecasting*).
* **Gunakan Bantuan AI**: Jangan ragu untuk menanyakan resep menu makan sehat atau rencana variasi latihan baru langsung kepada **FitBot**.

---
*Semoga perjalanan hidup sehat Anda menyenangkan dan sukses mencapai target!* 💪
