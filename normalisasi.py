import pandas as pd
from scipy.stats import zscore

file_path = 'student_prediction.csv' 
df = pd.read_csv(file_path)

# Fungsi untuk menghapus outlier menggunakan Z-Score
def remove_outliers_zscore(df, column, threshold=3):
    z_scores = zscore(df[column])
    return df[abs(z_scores) <= threshold]

# Menghapus numerik yang ingin dibersihkan darioutlier
colums_to_clean = ["AGE", "SALARY", "TRANSPORT"]

#menghapus outlier pada kolom yang dipilih
df_clean_zscore = df.copy()
for column in colums_to_clean:
    df_clean_zscore = remove_outliers_zscore(df_clean_zscore, column, threshold=3)

# Simpan data yang telah dibersihkan
df_clean_zscore.to_csv('cleaned_student_prediction_data.csv', index=False)

# Menampilkan ukuran data sebelum dan sesudah pembersihan
print(f"Data asli: {df.shape[0]} baris")
print(f"Data setelah pembersihan outlier: {df_clean_zscore.shape[0]} baris")