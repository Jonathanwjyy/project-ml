import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Path dataset langsung
DATASET_PATH = "cleaned_student_prediction_data.csv"  # Ganti dengan path file dataset Anda
TARGET_COLUMN = "GRADE"  # Ganti dengan nama kolom target pada dataset Anda
dataset = pd.read_csv(DATASET_PATH)

# Fungsi Preprocessing
def preprocess_data(data, target_col):
    processed_data = data.copy()
    for col in processed_data.columns:
        if processed_data[col].isnull().any():
            if processed_data[col].dtype in ['float64', 'int64']:
                processed_data[col].fillna(processed_data[col].median(), inplace=True)
            else:
                processed_data[col].fillna(processed_data[col].mode()[0], inplace=True)
    for col in processed_data.columns:
        if col == target_col:
            processed_data[col] = processed_data[col].astype('category').cat.codes
        elif processed_data[col].dtype in ['float64', 'int64']:
            processed_data[col] = (processed_data[col] - processed_data[col].min()) / (processed_data[col].max() - processed_data[col].min())
        else:
            processed_data[col] = processed_data[col].astype('category').cat.codes
    return processed_data

# Fungsi untuk menjalankan KNN
def run_knn():
    global knn_model, X, y
    try:
        dataset = pd.read_csv(DATASET_PATH)
        processed_data = preprocess_data(dataset, TARGET_COLUMN)
        X = processed_data.drop(columns=[TARGET_COLUMN]).values
        y = processed_data[TARGET_COLUMN].values

        optimal_k = find_optimal_k_cv(X, y)
        knn_model = KNeighborsClassifier(n_neighbors=optimal_k[0])
        knn_model.fit(X, y)

        update_student_list(dataset)
        evaluate_knn_model()
        messagebox.showinfo("Sukses", f"Model berhasil dibangun. K optimal: {optimal_k[0]}")
    except Exception as e:
        messagebox.showerror("Error", f"Error saat menjalankan KNN: {e}")

# Cari K optimal
def find_optimal_k_cv(features, labels, max_k=20):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = [(k, np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=k), features, labels, cv=skf)))
                  for k in range(1, max_k + 1)]
    return max(accuracies, key=lambda x: x[1])

# Update daftar siswa ke Treeview
def update_student_list(data):
    student_list.delete(*student_list.get_children())
    for idx, row in data.iterrows():
        student_list.insert("", "end", values=(idx,) + tuple(row.values))

# Prediksi individual berdasarkan baris yang dipilih
def predict_selected_student():
    if not knn_model:
        messagebox.showerror("Error", "Model belum dibangun! Jalankan proses KNN terlebih dahulu.")
        return
    try:
        selected_item = student_list.selection()
        if not selected_item:
            messagebox.showerror("Error", "Harap pilih siswa dari daftar!")
            return
        selected_index = int(student_list.item(selected_item, "values")[0])
        prediction = knn_model.predict(X[selected_index].reshape(1, -1))
        result_label.config(text=f"Hasil Prediksi: {'Layak' if prediction[0] == 1 else 'Tidak Layak'}", fg="blue")
    except Exception as e:
        messagebox.showerror("Error", f"Error saat melakukan prediksi: {e}")



# Evaluasi model KNN
def evaluate_knn_model():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# GUI Utama
root = tk.Tk()
root.title("KNN Student Prediction")
root.geometry("800x600")
root.configure(bg="#f0f0f0")

# Judul
title_label = tk.Label(root, text="KNN Student Prediction Model", font=("Arial", 20, "bold"), bg="#f0f0f0", fg="black")
title_label.pack(pady=10)

# Frame utama
main_frame = tk.Frame(root, bg="#ffffff", bd=2, relief=tk.GROOVE)
main_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

# Scrollable Treeview
tree_frame = tk.Frame(main_frame)
tree_frame.pack(fill=tk.BOTH, expand=True)

scrollbar = ttk.Scrollbar(tree_frame, orient="vertical")
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

columns = ["Index"] + [col for col in dataset.columns if col != TARGET_COLUMN]
student_list = ttk.Treeview(tree_frame, columns=columns, show="headings", yscrollcommand=scrollbar.set)
scrollbar.config(command=student_list.yview)

for col in columns:
    student_list.heading(col, text=col)
    student_list.column(col, anchor="center", width=60)
student_list.pack(fill=tk.BOTH, expand=True)

# Tombol Run dan Predict
button_frame = tk.Frame(root, bg="#f0f0f0")
button_frame.pack(pady=10)

run_button = tk.Button(button_frame, text="Run KNN", command=run_knn, bg="#28a745", fg="white", font=("Arial", 12, "bold"))
run_button.grid(row=0, column=0, padx=10)

predict_button = tk.Button(button_frame, text="Predict Selected Student", command=predict_selected_student,
                           bg="#007bff", fg="white", font=("Arial", 12, "bold"))
predict_button.grid(row=0, column=1, padx=10)

# Label Hasil
result_label = tk.Label(root, text="Hasil Prediksi: ", font=("Arial", 14, "bold"), bg="#f0f0f0", fg="red")
result_label.pack(pady=10)

# Jalankan aplikasi
root.mainloop()
