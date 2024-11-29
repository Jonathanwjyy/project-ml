import math
import pandas as pd

# Hitung total jumlah data
data = pd.read_csv('cleaned_student_prediction_data.csv')
jumlah_keseluruhan = len(data)

# Hitung jumlah data berdasarkan kategori AGE
age_18_21 = (data['AGE'] == 1).sum()
age_22_25 = (data['AGE'] == 2).sum()
age_above_26 = (data['AGE'] == 3).sum()

# Hitung distribusi GRADE untuk setiap kategori AGE
# Untuk AGE 18-21
age_18_21_grade_counts = data[data['AGE'] == 1]['GRADE'].value_counts()

# Untuk AGE 22-25
age_22_25_grade_counts = data[data['AGE'] == 2]['GRADE'].value_counts()

# Untuk AGE above 26
age_above_26_grade_counts = data[data['AGE'] == 3]['GRADE'].value_counts()

# Fungsi untuk menghitung entropy subset
def calculate_subset_entropy(counts, total):
    probabilities = [count / total for count in counts if count > 0]
    return -sum(p * math.log2(p) for p in probabilities)

# Entropy untuk masing-masing kategori AGE
entropy_18_21 = calculate_subset_entropy(age_18_21_grade_counts, age_18_21)
entropy_22_25 = calculate_subset_entropy(age_22_25_grade_counts, age_22_25)
entropy_above_26 = calculate_subset_entropy(age_above_26_grade_counts, age_above_26)

# Weighted entropy berdasarkan AGE
weighted_entropy_age = (
    (age_18_21 / jumlah_keseluruhan) * entropy_18_21 +
    (age_22_25 / jumlah_keseluruhan) * entropy_22_25 +
    (age_above_26 / jumlah_keseluruhan) * entropy_above_26
)

# Hitung entropy awal (entropi dari seluruh kolom GRADE)
grade_counts = data['GRADE'].value_counts()
total_entropy = calculate_subset_entropy(grade_counts, jumlah_keseluruhan)

# Hitung information gain untuk AGE
information_gain_age = total_entropy - weighted_entropy_age

print('Information Gain (AGE):', information_gain_age)
