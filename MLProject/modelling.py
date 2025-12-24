import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os

def train():
    data_path = 'examscore_preprocessing_automate.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} tidak ditemukan!")
        return

    # Load Dataset
    df = pd.read_csv(data_path)
    X = df.drop(columns=['exam_score'])
    y = df['exam_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Aktifkan Autolog
    mlflow.sklearn.autolog()

    # MLflow Project secara otomatis membuka 'start_run' untuk Anda.
    # Kita hanya perlu melatih model di dalam konteks tersebut.
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Log model secara eksplisit untuk memastikan kriteria Skilled/Advance terpenuhi
    mlflow.sklearn.log_model(model, "model")
    
    print("Training Berhasil!")
    print(f"Tracking URI saat ini: {mlflow.get_tracking_uri()}")

if __name__ == "__main__":
    train())




