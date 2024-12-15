import tkinter as tk
from tkinter import messagebox, ttk
from knn_model import KNNModel

# Koneksi KNNModel
DATASET_PATH = "cleaned_student_prediction_data.csv"  # Path dataset
TARGET_COLUMN = "GRADE"

class KNNApp:
    def __init__(self, root):
        self.model = KNNModel(DATASET_PATH, TARGET_COLUMN)
        self.root = root
        self.knn_model = None
        self.dataset = None
        self.init_gui()

    # GUI Utama
    def init_gui(self):
        self.root.title("KNN Student Prediction")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        # Judul
        title_label = tk.Label(self.root, text="KNN Student Prediction Model", font=("Arial", 20, "bold"), bg="#f0f0f0")
        title_label.pack(pady=10)

        # Treeview untuk daftar siswa
        self.tree_frame = ttk.Frame(self.root)
        self.tree_frame.pack(fill=tk.BOTH, expand=True)
        self.student_list = ttk.Treeview(self.tree_frame, show="headings")
        self.student_list.pack(fill=tk.BOTH, expand=True)

        # Tombol
        button_frame = tk.Frame(self.root, bg="#f0f0f0")
        button_frame.pack(pady=10)
        tk.Button(button_frame, text="Run KNN", command=self.run_knn, bg="#28a745", fg="white").pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Predict Selected Student", command=self.predict_selected, bg="#007bff", fg="white").pack(side=tk.LEFT, padx=10)

        # Label Hasil
        self.result_label = tk.Label(self.root, text="Hasil Prediksi: ", font=("Arial", 14), bg="#f0f0f0")
        self.result_label.pack(pady=10)

    def run_knn(self):
        try:
            k = self.model.build_model()
            self.dataset = self.model.dataset
            self.update_treeview()
            messagebox.showinfo("Sukses", f"Model berhasil dibangun dengan K optimal: {k}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def update_treeview(self):
        self.student_list["columns"] = ["Index"] + list(self.dataset.columns)
        for col in self.student_list["columns"]:
            self.student_list.heading(col, text=col)
            self.student_list.column(col, anchor="center")
        for i, row in self.dataset.iterrows():
            self.student_list.insert("", "end", values=(i,) + tuple(row.values))

    def predict_selected(self):
        selected = self.student_list.selection()
        if not selected:
            messagebox.showerror("Error", "Pilih siswa terlebih dahulu.")
            return
        index = int(self.student_list.item(selected, "values")[0])
        result = self.model.predict(index)
        self.result_label.config(text=f"Hasil Prediksi: {'Layak' if result[0] == 1 else 'Tidak Layak'}")

