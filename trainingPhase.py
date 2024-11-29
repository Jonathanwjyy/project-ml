import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
# Ganti 'dataset.csv' dengan path file CSV Anda
dataset_path = 'cleaned_student_prediction_data.csv' 
data = pd.read_csv(dataset_path)

# 2. Fitur (X) dan Label (y)
# Pilih fitur penting berdasarkan analisis Anda
# Contoh: Menggunakan fitur AGE, GENDER, HS_TYPE, SCHOLARSHIP, dll.
X = data[['AGE', 'GENDER', 'HS_TYPE', 'SCHOLARSHIP', 'WORK', 'ACTIVITY', 'LISTENS', 'LIKES_DISCUSS', 'CLASSROOM', 'CUML_GPA', 'EXP_GPA']]
y = data['GRADE']  # Kolom target yang ingin diprediksi

# 3. Encode Data
# Mengubah data non-numerik menjadi numerik (contoh: GENDER, HS_TYPE)
X['GENDER'] = X['GENDER'].map({'Male': 0, 'Female': 1})  # Encode GENDER
X['HS_TYPE'] = X['HS_TYPE'].astype('category').cat.codes  # Encode HS_TYPE

# 4. Tangani Missing Values
# Mengisi nilai kosong dengan rata-rata atau strategi lainnya
X = X.fillna(X.mean())

# 5. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Training Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. Evaluasi Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Visualisasi (Opsional)
# from sklearn.tree import export_graphviz
# import graphviz

# dot_data = export_graphviz(
#     model,
#     out_file=None,
#     feature_names=X.columns,
#     class_names=model.classes_.astype(str),
#     filled=True,
#     rounded=True,
#     special_characters=True
# )
# graph = graphviz.Source(dot_data)
# graph.render("decision_tree")  # Simpan visualisasi sebagai PDF