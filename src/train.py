"""
Enhanced training script with CI/CD support
Handles both local development and CI environments
"""
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import sys

def ensure_data_exists():
    """Ensure iris.csv exists, create sample data if missing"""
    data_path = "data/iris.csv"
    if not os.path.exists(data_path):
        print("⚠️ iris.csv not found, creating sample data...")
        try:
            from sklearn.datasets import load_iris
            
            os.makedirs("data", exist_ok=True)
            iris = load_iris()
            df = pd.DataFrame(
                iris.data, 
                columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            )
            df['species'] = iris.target_names[iris.target]
            df.to_csv(data_path, index=False)
            print(f"✅ Created sample data at {data_path}")
        except Exception as e:
            print(f"❌ Failed to create sample data: {str(e)}")
            raise

def load_data(data_path="data/iris.csv"):
    """Load and return the dataset"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    X = df.drop("species", axis=1)
    y = df["species"]
    return X, y

def train_model(X, y, n_estimators=100, random_state=42):
    """Train and return the RandomForest model"""
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        random_state=random_state,
        max_depth=10,  # Prevent overfitting
        min_samples_split=5
    )
    model.fit(X, y)
    return model

def save_model(model, model_path="models/model.pkl"):
    """Save model to disk"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {model_path}")

def save_metrics(accuracy, metrics_path="metrics.txt"):
    """Save metrics to file for DVC tracking"""
    with open(metrics_path, "w") as f:
        f.write(f"Training accuracy: {accuracy:.4f}\n")
    print(f"✅ Metrics saved to {metrics_path}")

def setup_mlflow():
    """Setup MLflow tracking (skip in CI environments)"""
    if os.getenv('CI'):
        print("⚠️ CI environment detected, skipping MLflow setup")
        return False
    
    try:
        mlflow.set_tracking_uri("https://dagshub.com/durvankur/MLOPs_demo.mlflow")
        mlflow.set_experiment("MLOPs_demo")
        print("✅ MLflow tracking configured")
        return True
    except Exception as e:
        print(f"⚠️ MLflow setup failed: {str(e)}")
        return False

def main():
    """Main training function"""
    try:
        print("🚀 Starting model training pipeline...")
        
        # Ensure data exists
        ensure_data_exists()
        
        # Load data
        X, y = load_data()
        print(f"✅ Data loaded successfully. Shape: X{X.shape}, y{y.shape}")
        
        # Setup MLflow (skip in CI)
        use_mlflow = setup_mlflow()
        
        if use_mlflow:
            # Full MLflow tracking for development
            with mlflow.start_run():
                print("📊 Starting MLflow run...")
                
                # Train model
                model = train_model(X, y)
                accuracy = model.score(X, y)
                
                # Log parameters
                mlflow.log_param("n_estimators", 100)
                mlflow.log_param("random_state", 42)
                mlflow.log_param("max_depth", 10)
                mlflow.log_param("min_samples_split", 5)
                
                # Log metrics
                mlflow.log_metric("training_accuracy", accuracy)
                
                # Log model
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="random_forest_model"
                )
                
                # Save locally
                save_model(model)
                save_metrics(accuracy)
                
                print(f"✅ MLflow tracking completed. Accuracy: {accuracy:.4f}")
        else:
            # Simplified training for CI environment
            print("🔧 Running simplified training for CI...")
            model = train_model(X, y)
            accuracy = model.score(X, y)
            
            save_model(model)
            save_metrics(accuracy)
            
            print(f"✅ CI training completed. Accuracy: {accuracy:.4f}")
        
        # Validate minimum performance
        MIN_ACCURACY = 0.85
        if accuracy < MIN_ACCURACY:
            print(f"❌ Model accuracy {accuracy:.4f} below minimum threshold {MIN_ACCURACY}")
            sys.exit(1)
        
        print("🎉 Training completed successfully!")
        print(f"   • Final accuracy: {accuracy:.4f}")
        print(f"   • Model saved: models/model.pkl")
        print(f"   • Metrics saved: metrics.txt")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
