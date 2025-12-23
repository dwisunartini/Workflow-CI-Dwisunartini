
import mlflow

# Reset tracking ke lokal agar tidak konflik dengan sqlite
mlflow.set_tracking_uri(None) 
mlflow.sklearn.autolog(disable=True) # Matikan dulu
mlflow.sklearn.autolog() # Nyalakan ulang secara bersih

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os

mlflow.set_tracking_uri(f"file:///{os.getcwd()}/mlruns")

# Inisialisasi autolog
mlflow.sklearn.autolog()

def train_basic():
    data_path = 'examscore_preprocessing_automate.csv'
    if not os.path.exists(data_path):
        print(f"File {data_path} tidak ditemukan! Pastikan file ada di folder yang sama.")
        return

    df = pd.read_csv(data_path)
    X = df.drop(columns=['exam_score'])
    y = df['exam_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Melatih model Scikit-Learn sederhana (Random Forest)
    with mlflow.start_run(run_name="Model_Basic_new"):
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Tampilkan informasi sukses
        print("Basic Model Training Selesai.")
        print(f"Log tersimpan secara lokal di: {mlflow.get_tracking_uri()}")

if __name__ == "__main__":
    train_basic()




