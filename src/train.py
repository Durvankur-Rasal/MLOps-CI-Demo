# train.py
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Set tracking URI to DagsHub
mlflow.set_tracking_uri("https://dagshub.com/durvankur/MLOPs_demo.mlflow")
mlflow.set_experiment("MLOPs_demo")

# Load data
df = pd.read_csv("data/iris.csv")
X = df.drop("species", axis=1)
y = df["species"]

with mlflow.start_run():
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    
    # Calculate and log metrics
    accuracy = model.score(X, y)
    mlflow.log_metric("training_accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="random_forest_model"
    )
    
    # Save model locally
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # Save metric to file for DVC
    with open("metrics.txt", "w") as f:
        f.write(f"Training accuracy: {accuracy:.4f}\n")
