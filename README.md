*Created: 30/08/2024*
___

# Submission 1: Water Potability

![water](https://github.com/user-attachments/assets/f4c052f6-b522-4939-bb09-a388888abad2)

Nama : Maulana Kavaldo

UsernameDicoding : mkavaldo

|         | Deskripsi|
|---------|----------|
|Dataset  |	[Water Potability](https://www.kaggle.com/datasets/adityakadiwal/water-potability)|
|Masalah| Potabilitas air adalah kunci untuk menjaga kesehatan manusia, karena air yang tidak aman dapat menimbulkan berbagai risiko kesehatan. World Health Organization (WHO) menekankan pentingnya kualitas air minum untuk mencegah penyakit [^1]. Proyek ini bertujuan mengklasifikasikan air sebagai "potable" (aman) atau "not potable" (tidak aman), menggunakan data fitur seperti pH dan kekerasan air. Centers for Disease Control and Prevention (CDC) menyatakan bahwa memastikan kualitas air minum adalah langkah penting untuk perlindungan kesehatan [^2]. Hal ini juga didukung oleh laporan United Nations tentang pentingnya akses ke air bersih [^3] dan pedoman dari Environmental Protection Agency (EPA) mengenai standar kualitas air [^4].|
|Solusi machine learning|Untuk menentukan potabilitas air, berbagai parameter kualitas air biasanya diuji. Namun, penilaian keamanan air hanya berdasarkan hasil pengujian tersebut bisa memakan waktu dan tidak selalu langsung memberikan kesimpulan. Dengan menggunakan model Sequential dalam deep learning, yang merupakan struktur jaringan saraf sederhana namun efektif, diharapkan dapat memberikan penilaian yang lebih cepat dan akurat mengenai keamanan air. Model Sequential terdiri dari lapisan-lapisan yang terhubung secara linier dan dapat memproses data input untuk menghasilkan prediksi apakah air aman untuk dikonsumsi atau tidak. Hasil dari sistem ini dapat digunakan sebagai dasar untuk pertimbangan dan tindakan lebih lanjut dalam memastikan keamanan sumber air.|
|Metode pengolahan|Terdapat sepuluh fitur di mana sembilan di antaranya digunakan untuk klasifikasi dan satu sebagai label klasifikasi. Semua fitur dalam dataset bersifat numerikal: ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, dan Turbidity memiliki tipe data float64, sedangkan fitur label Potability memiliki tipe data int64. Data dibagi menjadi 80% untuk pelatihan dan 20% untuk evaluasi. Proses transformasi data melibatkan normalisasi untuk fitur numerikal menggunakan z-score. Normalisasi dengan z-score dilakukan untuk memastikan bahwa semua fitur memiliki skala yang seragam, dengan rata-rata 0 dan deviasi standar 1, sehingga membantu model dalam proses pelatihan dengan mengurangi bias yang disebabkan oleh perbedaan skala fitur.|
|Arsitektur model | Arsitektur model dimulai dengan menggabungkan input fitur dan kemudian melalui beberapa layer dense. Model terdiri dari empat layer dense dengan unit 256, 128, 64, dan 32, masing-masing menggunakan aktivasi ReLU, diikuti oleh Batch Normalization dan Dropout dengan rate 0.3. Layer output adalah dense dengan 1 unit dan aktivasi sigmoid untuk klasifikasi biner. Model ini dikompilasi dengan optimizer Adam (learning rate 0.0001), loss binary crossentropy, dan metrik akurasi, untuk mengklasifikasikan air sebagai "potable" (aman) atau "not potable" (tidak aman).|
| Metrik evaluasi  |  Metrik evaluasi yang digunakan dalam proyek ini meliputi AUC, Precision, Recall, ExampleCount, dan BinaryAccuracy. ExampleCount menghitung jumlah contoh yang dievaluasi, sedangkan BinaryCrossentropy mengukur kerugian model dalam klasifikasi biner. BinaryAccuracy menilai akurasi klasifikasi, sedangkan Precision dan Recall mengukur ketepatan dan kemampuan model dalam mendeteksi kasus positif dengan ambang batas 0.5. Dengan konfigurasi ini, model dievaluasi secara menyeluruh untuk memastikan efektivitasnya dalam menentukan apakah air aman untuk dikonsumsi atau tidak.|
| Performa model  |  Hasil evaluasi model setelah tuning menunjukkan akurasi 72.8% pada pelatihan dan 63.7% pada data validasi. Loss yang diperoleh adalah 0.53 selama pelatihan dan 0.63 pada validasi. Model menunjukkan kinerja yang cukup baik, namun masih terdapat gap yang cukup jauh antara pelatihan dan validasi sehingga masih perlu eksperiment lebih lanjut.|



## Deployment TensorFlow Serving

Untuk melakukan deployment model yang telah dibuat yaitu dengan menggunakan Docker untuk mengemas dan menjalankan model dalam kontainer yang terisolasi. Pastikan bahwa Docker Desktop telah terinstal di komputer. Unduh [Docker Desktop](https://www.docker.com/products/docker-desktop/) dan lihat cara [instalasi](https://docs.docker.com/desktop/install/windows-install/).

### Membuat docker file yang berisi command

```docker
FROM tensorflow/serving:latest

COPY ./serving_model_dir /models
ENV MODEL_NAME=water-prediction-model
```

### Membuat Docker Image

```docker
docker build -t water-prediction-tf-serving .
```

### Menjalankan Docker

```docker
docker run -p 8080:8501 water-prediction-tf-serving
```

### Metadata
Copy paste url berikut ke browser untuk melihat metadata:
```url
http://localhost:8080/v1/models/water-prediction-model/metadata
```
Jika berhasil akan muncul seperti ini:

<img width="445" alt="mkavaldo-metadata" src="https://github.com/user-attachments/assets/35a3cd22-b4bd-4bf8-a1ad-3d3348c8c2e3">



### Prediction

Melakukan prediksi pada data sample dengan menentukan index tertentu.

- Buka file testing.ipynb dan lakukan run cell apakah model tersedia pada endpoint.
    ```url
    http://localhost:8080/v1/models/water-prediction-model
    ```
- Lakukan run hingga cell akhir.
- Ubah index data sesuai kebutuhan untuk melihat hasil prediksi.



Referensi:

[^1]: WHO. "Guidelines for Drinking-water Quality". 2022. Diakses pada 02 Agustus 2024 melalui [tautan.](https://www.who.int/publications/i/item/9789240045064)

[^2]: CDC. "Drinking Water Quality". 2024. Diakses pada 02 Agustus 2024 melalui [tautan.](https://wwwnc.cdc.gov/travel/yellowbook/2024/preparing/water-disinfection)

[^3]: UN. "Water and Sanitation". 2024. Diakses pada 02 Agustus 2024 melalui [tautan.](https://www.un.org/en/climatechange/science/climate-issues/water?gad_source=1&gclid=CjwKCAjwodC2BhAHEiwAE67hJCOFgcfzxzhV9OGcpfbqKp-0kFjooiQwXpDkWe2Q6CDQ1Wm9IvrcbBoCQTcQAvD_BwE)

[^4]: EPA. "Water Quality Standards". 2024. Diakses pada 02 Agustus 2024 melalui  [tautan](https://www.epa.gov/wqs-tech)
