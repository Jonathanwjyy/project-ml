import pandas as pd
import numpy as np

csv_file_path = 'C:/ml/student_prediction.csv' 

df = pd.read_csv(csv_file_path)

print(df.head())

print("Jumlah baris dan kolom:", df.shape)