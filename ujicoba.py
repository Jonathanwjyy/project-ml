import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

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

# Cari K optimal
def find_optimal_k_cv(features, labels, max_k=20):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    
    for k in range(1, max_k + 1):
        knn = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(knn, features, labels, cv=skf, scoring='accuracy')
        accuracies.append((k, np.mean(cv_scores)))
    
    optimal_k = max(accuracies, key=lambda x: x[1])
    return optimal_k

# Fungsi untuk menjalankan KNN
def run_knn():
    global knn_model, data_columns, dataset, target_col, X, y
    file_path = file_path_var.get()
    target_col = target_col_var.get()
    
    if not file_path or not target_col:
        messagebox.showerror("Error", "Harap masukkan file dataset dan kolom target!")
        return
    
    try:
        dataset = pd.read_csv(file_path)
    except Exception as e:
        messagebox.showerror("Error", f"Error membaca file: {e}")
        return
    
    if target_col not in dataset.columns:
        messagebox.showerror("Error", f"Kolom target '{target_col}' tidak ditemukan dalam dataset!")
        return
    
    try:
        # Preprocess data
        processed_data = preprocess_data(dataset, target_col)
        data_columns = list(processed_data.columns.drop(target_col))
        
        # Split dataset
        X = processed_data.drop(columns=[target_col]).values
        y = processed_data[target_col].values
        
        # Cari K optimal
        optimal_k = find_optimal_k_cv(X, y)
        
        # Build KNN model
        knn_model = KNeighborsClassifier(n_neighbors=optimal_k[0])
        knn_model.fit(X, y)
        
        # Tampilkan data siswa ke GUI
        update_student_list(dataset)
        
        messagebox.showinfo("Sukses", f"Model berhasil dibangun. K optimal: {optimal_k[0]}")
    except Exception as e:
        messagebox.showerror("Error", f"Error saat menjalankan KNN: {e}")

# Fungsi untuk memperbarui daftar siswa
def update_student_list(data):
    student_list.delete(*student_list.get_children())
    for idx, row in data.iterrows():
        student_list.insert("", "end", values=(idx,) + tuple(row.values))

# Fungsi untuk prediksi individual berdasarkan baris terpilih
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
        student_data = X[selected_index].reshape(1, -1)
        
        # Prediksi
        prediction = knn_model.predict(student_data)
        result_label.config(text=f"Hasil Prediksi: {'Layak' if prediction[0] == 1 else 'Tidak Layak'}")
    except Exception as e:
        messagebox.showerror("Error", f"Error saat melakukan prediksi: {e}")


def evaluate_knn_model():
    global knn_model, X, y

    # Split data menjadi Train dan Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Bangun model dengan data latih
    knn_model.fit(X_train, y_train)

    # Prediksi data uji
    y_pred = knn_model.predict(X_test)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(report)
    print(f"Accuracy: {accuracy:.2f}")



# GUI Tkinter
root = tk.Tk()
root.title("KNN Student Prediction")

# Input File Path
tk.Label(root, text="File Dataset (CSV):").grid(row=0, column=0, padx=10, pady=10, sticky="w")
file_path_var = tk.StringVar()
file_entry = tk.Entry(root, textvariable=file_path_var, width=40)
file_entry.grid(row=0, column=1, padx=10, pady=10)
tk.Button(root, text="Browse", command=lambda: file_path_var.set(filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")]))).grid(row=0, column=2, padx=10, pady=10)

# Input Target Column
tk.Label(root, text="Kolom Target:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
target_col_var = tk.StringVar()
target_entry = tk.Entry(root, textvariable=target_col_var, width=40)
target_entry.grid(row=1, column=1, padx=10, pady=10)

# Run Button
tk.Button(root, text="Run KNN", command=run_knn, bg="green", fg="white").grid(row=2, column=1, pady=10)

# Student List
tk.Label(root, text="Daftar Siswa:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
columns = ["Index"] + [f"Fitur {i}" for i in range(1, 6)]  # Anda bisa mengganti header sesuai dataset
student_list = ttk.Treeview(root, columns=columns, show="headings", height=10)
for col in columns:
    student_list.heading(col, text=col)
    student_list.column(col, width=100)
student_list.grid(row=4, column=0, columnspan=3, padx=10, pady=10)

# Predict Button
tk.Button(root, text="Predict Selected Student", command=predict_selected_student, bg="red", fg="white").grid(row=5, column=1, pady=10)

result_label = tk.Label(root, text="Hasil Prediksi: ")
result_label.grid(row=6, column=0, columnspan=3, pady=10)

root.mainloop()
