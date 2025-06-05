# UAS-Analisis-Multivariat-Kelompok-8-
Dibuat untuk memenuhi tugas akhir mata kuliah Analisis Multivariat

# 📊 Analisis Klasifikasi Kesehatan Janin (Fetal Health Classification)

Proyek ini bertujuan untuk menganalisis data kesehatan janin menggunakan dua pendekatan utama:

* **Linear Discriminant Analysis (LDA)**
* **Regresi Logistik Ordinal (Ordinal Logistic Regression)**

Dataset yang digunakan: `fetal_health.csv`

---

## 📁 Struktur Proyek

* `fetal_health.csv` – Dataset utama.
* `script.R` – Skrip utama R yang berisi preprocessing, analisis, dan evaluasi model.
* `README.md` – Dokumentasi ini.

---

## 🧪 Langkah Analisis

### 1. **Instalasi dan Import Library**

Beberapa paket yang digunakan antara lain:

```r
install.packages("brant")
install.packages("psych")
install.packages("skimr")
install.packages("car")
install.packages("nnet")
install.packages("MASS")
install.packages("biotools")
install.packages("ggplot2")
```

### 2. **Eksplorasi dan Preprocessing Data**

* Mengecek struktur data, missing values, dan duplikasi.
* Mengubah `fetal_health` menjadi faktor (Normal, Suspect, Pathological).
* Split data latih dan uji (80:20) dengan stratifikasi.
* Standarisasi fitur dengan `preProcess`.

### 3. **Visualisasi**

* Visualisasi distribusi kelas target.
* Boxplot fitur berdasarkan kelas.

### 4. **Uji Asumsi LDA**

* Uji **normalitas multivariat** (Mardia's Test).
* Uji **homogenitas matriks kovarian** (Box’s M Test).
* **PCA** digunakan untuk reduksi dimensi sebelum uji Box's M.

### 5. **LDA (Linear Discriminant Analysis)**

* Training dan evaluasi model LDA.
* Hitung akurasi, confusion matrix, sensitivitas, spesifisitas, dan presisi per kelas.

### 6. **Uji Asumsi Regresi Logistik Ordinal**

* **Brant Test** untuk menguji asumsi proportional odds.
* **VIF** untuk mendeteksi multikolinearitas.
* **Wald Test** untuk signifikansi koefisien.
* **Likelihood Ratio Test** untuk membandingkan model kompleks dan sederhana.

### 7. **Regresi Logistik Ordinal**

* Model dibangun menggunakan fungsi `polr()` dari paket `MASS`.
* Evaluasi:

  * Log Odds Ratio & Confidence Interval
  * Confusion matrix
  * Visualisasi confusion matrix dengan `ggplot2`

### 8. **Uji Wilks’ Lambda (MANOVA)**

* Digunakan untuk mengevaluasi apakah mean antar kelompok berbeda secara signifikan berdasarkan variabel prediktor.
* Uji dilakukan dengan dan tanpa PCA.

---

## 📝 Catatan

* Dataset harus bersih dari missing values dan duplikasi sebelum analisis.
* Disarankan untuk menyeleksi variabel prediktor agar jumlahnya lebih sedikit daripada jumlah observasi terkecil di tiap kategori target.
* Semua asumsi uji harus dipenuhi sebelum interpretasi model LDA dan regresi ordinal.

---

## ✅ Hasil yang Dihasilkan

* Akurasi model LDA
* Statistik sensitivitas, spesifisitas, presisi
* Tabel dan visualisasi confusion matrix
* Output uji Brant, Wald, VIF, dan Likelihood Ratio Test
* Wilks’ Lambda dari MANOVA

---

## 👨‍🔬 Penulis

* Analisis oleh: 
**\[Dayinta Agustina Zanuel]*
**\[Tanti Ayu Hardiningtyas]*
**\[Muhammad Rarendra Satiya]*
* Bahasa Pemrograman: R
