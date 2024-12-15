import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class KNNModel:
    def __init__(self, dataset_path, target_column):
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.dataset = None
        self.knn_model = None
        self.X = None
        self.y = None

    # Fungsi Preprocessing
    def preprocess_data(self):
        data = pd.read_csv(self.dataset_path)
        processed_data = data.copy()
        for col in processed_data.columns:
            if processed_data[col].isnull().any():
                if processed_data[col].dtype in ['float64', 'int64']:
                    processed_data[col].fillna(processed_data[col].median(), inplace=True)
                else:
                    processed_data[col].fillna(processed_data[col].mode()[0], inplace=True)
        for col in processed_data.columns:
            if col == self.target_column:
                processed_data[col] = processed_data[col].astype('category').cat.codes
            elif processed_data[col].dtype in ['float64', 'int64']:
                processed_data[col] = (processed_data[col] - processed_data[col].min()) / (
                        processed_data[col].max() - processed_data[col].min())
            else:
                processed_data[col] = processed_data[col].astype('category').cat.codes
        self.dataset = processed_data
        self.X = processed_data.drop(columns=[self.target_column]).values
        self.y = processed_data[self.target_column].values

    # Cari K Optimal
    def find_optimal_k_cv(self, max_k=20):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = [(k, np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=k), self.X, self.y, cv=skf)))
                      for k in range(1, max_k + 1)]
        return max(accuracies, key=lambda x: x[1])

    # Bangun Model KNN
    def build_model(self):
        self.preprocess_data()
        optimal_k = self.find_optimal_k_cv()
        self.knn_model = KNeighborsClassifier(n_neighbors=optimal_k[0])
        self.knn_model.fit(self.X, self.y)
        return optimal_k[0]

    # Evaluasi Model
    def evaluate_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.knn_model.fit(X_train, y_train)
        y_pred = self.knn_model.predict(X_test)
        return {
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred)
        }

    # Prediksi Data Individu
    def predict(self, index):
        if self.knn_model is None:
            raise Exception("Model belum dibangun. Jalankan build_model() terlebih dahulu.")
        return self.knn_model.predict(self.X[index].reshape(1, -1))

