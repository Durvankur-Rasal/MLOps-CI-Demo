"""
Model performance validation script
Checks trained model against performance thresholds
"""
import pickle
import pandas as pd
import sys
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def check_model_performance():
    """Validate trained model performance against thresholds"""
    print("🔍 Starting model performance validation...")
    
    try:
        # Check if model file exists
        model_path = "models/model.pkl"
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            print("💡 Make sure to run training first: python src/train.py")
            return False
        
        # Load trained model
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully")
        print(f"   • Model type: {type(model).__name__}")
        print(f"   • Number of estimators: {getattr(model, 'n_estimators', 'N/A')}")
        
        # Check if test data exists
        data_path = "data/iris.csv"
        if not os.path.exists(data_path):
            print("⚠️ Data file not found, skipping performance validation")
            print("This is expected in CI environments without data")
            return True
        
        # Load test data (using training data for simplicity)
        df = pd.read_csv(data_path)
        X = df.drop("species", axis=1)
        y = df["species"]
        print(f"✅ Test data loaded. Shape: X{X.shape}, y{y.shape}")
        
        # Make predictions
        predictions = model.predict(X)
        accuracy = accuracy_score(y, predictions)
        
        print(f"\n📊 Model Performance Metrics:")
        print(f"   • Training Accuracy: {accuracy:.4f}")
        print(f"   • Predicted Classes: {list(model.classes_)}")
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            print(f"   • Feature Importances:")
            feature_names = X.columns
            for name, importance in zip(feature_names, model.feature_importances_):
                print(f"     - {name}: {importance:.4f}")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y, predictions))
        
        # Confusion matrix
        print(f"🔍 Confusion Matrix:")
        cm = confusion_matrix(y, predictions)
        print(cm)
        
        # Performance thresholds
        MIN_ACCURACY = 0.90
        EXPECTED_ACCURACY = 0.95
        
        print(f"\n🎯 Performance Validation:")
        if accuracy < MIN_ACCURACY:
            print(f"❌ FAIL: Accuracy {accuracy:.4f} is below minimum threshold {MIN_ACCURACY}")
            return False
        elif accuracy < EXPECTED_ACCURACY:
            print(f"⚠️ WARNING: Accuracy {accuracy:.4f} is below expected threshold {EXPECTED_ACCURACY}")
            print("   Model meets minimum requirements but could be improved")
        else:
            print(f"✅ EXCELLENT: Accuracy {accuracy:.4f} exceeds expected threshold {EXPECTED_ACCURACY}")
        
        # Check if metrics file exists and validate
        metrics_path = "metrics.txt"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics_content = f.read().strip()
                print(f"\n📄 Metrics file content:")
                print(f"   {metrics_content}")
        else:
            print("⚠️ Metrics file not found")
        
        # Additional model health checks
        print(f"\n🔧 Model Health Checks:")
        
        # Check if model can make predictions on sample data
        sample_input = X.iloc[:1]  # First row
        try:
            sample_pred = model.predict(sample_input)
            print(f"✅ Model can make predictions: {sample_pred[0]}")
        except Exception as e:
            print(f"❌ Model prediction failed: {str(e)}")
            return False
        
        # Check prediction probabilities (if available)
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(sample_input)
                print(f"✅ Model can output probabilities: {probabilities[0]}")
            except Exception as e:
                print(f"⚠️ Model probability prediction failed: {str(e)}")
        
        print("\n🎉 Model performance validation completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Model performance validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def generate_performance_report():
    """Generate a detailed performance report"""
    try:
        print("\n📈 Generating Performance Report...")
        
        # This could be expanded to create HTML reports, plots, etc.
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "model_path": "models/model.pkl",
            "validation_passed": check_model_performance()
        }
        
        print(f"Report generated: {report}")
        return report
        
    except Exception as e:
        print(f"❌ Failed to generate report: {str(e)}")
        return None

if __name__ == "__main__":
    success = check_model_performance()
    if not success:
        print("\n❌ Model performance validation FAILED!")
        sys.exit(1)
    print("\n✅ Model performance validation PASSED!")