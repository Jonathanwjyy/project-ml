import pandas as pd
import numpy as np

# csv_file_path = 'student_prediction.csv' 
csv_file_path = 'cleaned_student_prediction_data.csv' 

df = pd.read_csv(csv_file_path)

print(df.head())

print("Jumlah baris dan kolom:", df.shape)