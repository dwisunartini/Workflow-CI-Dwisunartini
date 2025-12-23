import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def train():
    # Load dataset
    df = pd.read_csv('examscore_preprocessing_automate.csv')
    X = df.drop(columns=['exam_score'])
    y = df['exam_score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # JANGAN gunakan mlflow.start_run() di sini. 
    # MLflow Project yang akan mengelolanya secara otomatis.
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Log metrik dan model ke dalam run yang sudah dibuat otomatis
    score = model.score(X_test, y_test)
    mlflow.log_metric("r2_score", score)
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Training selesai. R2 Score: {score}")

if __name__ == "__main__":
    train()train()train()




