# Bushfire Warning Classification: Perth, Western Australia

> Multi-class machine learning untuk memprediksi **tingkat peringatan bushfire** (Advice, Watch and Act, Emergency Warning) berdasarkan kondisi meteorologi, vegetasi, drought, dan topografi di Perth, Western Australia.
>
> Proyek ini merupakan **pengembangan (development & extension)** di atas paper acuan **Utamima, Kanedi & Sohel (2026)** yang terbit di *Environmental Challenges*, dengan dua kontribusi orisinal: **custom feature engineering (feature creation)** dan **implementasi SMOTE buatan sendiri (from-scratch)** untuk menangani class imbalance.

![status](https://img.shields.io/badge/status-completed-success)
![python](https://img.shields.io/badge/Python-3.x-blue)
![platform](https://img.shields.io/badge/platform-Google%20Colab-orange)
![methodology](https://img.shields.io/badge/methodology-CRISP--DM-informational)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 📌 Overview

Bushfire adalah salah satu hazard paling kritis di Australia. Sejak **15 Juli 2024**, Western Australia secara resmi mengadopsi **Australian Warning System (AWS)** yang menggunakan tiga tingkat peringatan bushfire: **Advice (kuning)**, **Watch and Act (oranye)**, dan **Emergency Warning (merah)**.

Project ini membangun pipeline klasifikasi multi-class yang memetakan kondisi lingkungan ke tingkat peringatan, mengikuti kerangka **CRISP-DM** dari *business understanding* hingga *evaluation*. Fokus utamanya bukan sekadar mengejar akurasi, tetapi menangani **class imbalance** secara serius (Emergency Warning jauh lebih jarang daripada Advice) dan **memperkaya sinyal prediktif lewat feature engineering**.

**Tujuan:**
- Mengklasifikasikan tingkat peringatan bushfire ke dalam 3 kelas (Advice, Watch and Act, Emergency Warning).
- Menangani distribusi kelas yang timpang tanpa mengorbankan kelas minoritas (kelas paling berbahaya).
- Menghasilkan model yang reproducible dan dapat dijalankan end-to-end di Google Colab.

---

## 🗂️ Dataset

| Item | Detail |
|---|---|
| **Region** | Perth, Western Australia |
| **Jumlah baris (raw)** | 1.288 rows |
| **Target** | Tingkat peringatan bushfire: **Advice**, **Watch and Act**, **Emergency Warning** (multi-class) |
| **Tipe fitur** | Meteorologi, vegetasi, drought, dan topografi |
| **Jumlah fitur setelah feature engineering** | `⟨FILL FROM NOTEBOOK: jumlah fitur final⟩` |

> Catatan: kategori target mengikuti **Australian Warning System** (Advice / Watch and Act / Emergency Warning) sebagaimana diterapkan oleh DFES Western Australia.

`⟨FILL FROM NOTEBOOK: deskripsi singkat sumber data, apakah turunan/replikasi dari setup paper Utamima et al., atau dataset yang Anda susun sendiri. Sebutkan juga rentang waktu data jika ada.⟩`

---

## 🧭 Methodology (CRISP-DM)

Project mengikuti enam fase **CRISP-DM**:

1. **Business Understanding:** Memahami kebutuhan early-warning bushfire & dampak kesalahan klasifikasi kelas berbahaya.
2. **Data Understanding:** Eksplorasi distribusi kelas, korelasi fitur meteorologi/drought/vegetasi/topografi, dan deteksi class imbalance.
3. **Data Preparation:** Cleaning, encoding, scaling, dan **feature engineering** (lihat bagian Kontribusi).
4. **Modeling:** Training beberapa model klasifikasi multi-class dengan penanganan imbalance via **custom SMOTE**.
5. **Evaluation:** Penilaian dengan metrik yang sensitif terhadap kelas minoritas (F1-score, precision, recall, bukan hanya accuracy).
6. **Deployment (light):** Notebook reproducible yang bisa dijalankan ulang di Google Colab.

---

## 🤖 Models

Model klasifikasi yang dieksplorasi dalam project ini:

- Random Forest
- XGBoost
- Support Vector Machine (SVM)
- Stacking Ensemble

---

## 📊 Results

> ⚠️ Angka di bawah ini **harus diisi dari output asli notebook** Anda. Jangan dipublikasikan dengan placeholder.

**Best model:** Random Forest

| Metric | Score |
|---|---|
| Accuracy | 0.89 |
| F1-score macro | 0.82 |
| Precision | 0.83 |
| Recall | 0.81 |

**Perbandingan antar model:**

| Model | Accuracy | F1-score | Precision | Recall |
|---|---|---|---|---|
| Random Forest | 0.89 | 0.82 | 0.83 | 0.81 |
| XGBoost | 0.90 | 0.80 | 0.83 | 0.78 |
| SVM | 0.80 | 0.67 | 0.66 | 0.68 |

---

## ⭐ My Contributions (Development on Top of the Reference Paper)

Bagian ini menjelaskan **apa yang saya kembangkan sendiri** di atas paper acuan Utamima et al. (2026). Paper tersebut menjadi *baseline / benchmark* konseptual; project ini menambahkan dua komponen orisinal:

### 1. 🛠️ Feature Engineering, Feature Creation
Saya menurunkan **fitur-fitur baru** dari variabel mentah (meteorologi, vegetasi, drought, topografi) untuk memperkuat sinyal prediktif terhadap tingkat peringatan.

`⟨FILL FROM NOTEBOOK: sebutkan fitur turunan yang Anda buat, mis. interaksi suhu x angin, indeks dryness, rasio vegetasi, binning musiman/diurnal, dsb. Sebutkan dari berapa fitur menjadi berapa fitur.⟩`

### 2. 🧪 Custom SMOTE Implementation (From Scratch)
Alih-alih memakai `imblearn.SMOTE` secara langsung, saya **mengimplementasikan SMOTE sendiri** untuk menangani class imbalance multi-class. Algoritma ini memilih sampel kelas minoritas, mencari **k-nearest neighbors**, lalu menghasilkan **synthetic sample** melalui interpolasi linear antara sampel dan tetangganya.

`⟨FILL FROM NOTEBOOK: detail implementasi Anda, nilai k, strategi per-kelas (Watch and Act & Emergency Warning), rasio oversampling, dan SMOTE hanya diterapkan pada training set (bukan test set) untuk menghindari data leakage.⟩`

> **Hubungan dengan paper acuan:** Paper Utamima et al. (2026) membingkai masalah ini sebagai **klasifikasi multi-class** atas kategori peringatan bushfire di Western Australia, dengan integrasi catatan peringatan resmi, observasi meteorologi, dan indikator drought, evaluasi menggunakan *random split* dan *time-forward seasonal holdout*, serta **association rule mining** untuk menggali relasi kondisi ke peringatan yang interpretable. Project saya mengambil framing multi-class yang sama tetapi **menambahkan feature creation dan custom SMOTE** sebagai kontribusi metodologis tersendiri.
>
> *Catatan transparansi:* paper acuan **tidak** secara eksplisit (pada bagian yang dapat saya verifikasi) menyebutkan SMOTE; penanganan imbalance lewat custom SMOTE ini adalah pendekatan saya, bukan replikasi langsung dari paper.

---

## ▶️ How to Run

Project ini didesain untuk dijalankan di **Google Colab** (tanpa setup lokal).

1. **Buka di Colab:** upload notebook `Eksplor_Materi_Praktikum_PAP_1_final.ipynb` ke Google Colab, atau klik tombol *Open in Colab* di bawah:

   `⟨FILL: tempel badge Open-in-Colab, format:⟩`
   `[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ishalllll/<NAMA-REPO>/blob/main/<NAMA-NOTEBOOK>.ipynb)`

2. **Siapkan dataset:** upload file dataset (`⟨FILL: nama file dataset, mis. bushfire_perth.csv⟩`) ke Colab, atau mount Google Drive:
```python
   from google.colab import drive
   drive.mount('/content/drive')
```

3. **Jalankan semua cell:** klik `Runtime`, lalu `Run all`. Semua dependency (pandas, numpy, scikit-learn, dll.) sudah tersedia secara default di Colab.

4. **Reproduce hasil:** output metrik akan muncul di bagian Evaluation pada notebook.

---

## 🧰 Tech Stack

- **Python 3.x**
- **pandas**, **numpy**, untuk data wrangling & feature engineering
- **scikit-learn**, untuk modeling & evaluation
- **XGBoost**, untuk modeling
- **IMB-learn**, untuk pipeline & oversampling
- **Scipy**, **datetime**, **math**, untuk feature engineering
- **Google Colab**, runtime environment

---

## 📚 References

Paper acuan / benchmark project ini:

> Utamima, A., Kanedi, F. J., & Sohel, F. (2026). Environmental drivers of bushfire warning categories: insights from official records, meteorology, and drought data in Western Australia. *Environmental Challenges, 23*, 101509. https://doi.org/10.1016/j.envc.2026.101509

- Journal: *Environmental Challenges* (Elsevier), ISSN 2667-0100
- Open Access (CC BY-NC 4.0)
- ScienceDirect (PII): S2667010026001034

Referensi pendukung (Australian Warning System):
> Department of Fire and Emergency Services (DFES), Western Australia. *Australian Warning System.* https://www.dfes.wa.gov.au/hazard-information/warning-systems/australian-warning-system

---

## 👤 Author

**Muhammad Faishal Ardiansyah**
Information Systems Student @ Institut Teknologi Sepuluh Nopember (ITS), Surabaya
Aspiring Data Scientist, interested in machine learning & data-driven decision making

- GitHub: [@Ishalllll](https://github.com/Ishalllll)

---

## 📝 License

Released under the **MIT License**. `⟨FILL: ganti jika Anda memilih lisensi lain, atau tambahkan file LICENSE⟩`

---

*Project ini dikerjakan untuk tujuan pembelajaran & portofolio, dengan mengacu pada paper Utamima et al. (2026) sebagai benchmark konseptual. Kontribusi orisinal: feature creation & custom SMOTE.*
