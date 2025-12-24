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
        
    mlflow.end_run()

    df = pd.read_csv(data_path)
    X = df.drop(columns=['exam_score'])
    y = df['exam_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.sklearn.autolog()

    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    train()




