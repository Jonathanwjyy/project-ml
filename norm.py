import pandas as pd

# Membaca dataset Anda
data = pd.read_csv('cleaned_student_prediction_data.csv')  # Ganti dengan path file Anda

# Fungsi untuk binarisasi data
def binarize_data(dataframe, threshold=0.5):
    """
    Mengubah data numerik menjadi 0 dan 1 berdasarkan threshold.
    Nilai >= threshold -> 1, Nilai < threshold -> 0
    """
    return (dataframe >= threshold).astype(int)

# Menentukan kolom numerik
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns

# Binarisasi hanya pada kolom numerik
data_binarized = data.copy()
data_binarized[numerical_columns] = binarize_data(data[numerical_columns], threshold=data[numerical_columns].mean())

# Menampilkan hasil binarisasi
print(data_binarized.head())

# Menyimpan hasil ke file baru
data_binarized.to_csv('norm_data.csv', index=False)
