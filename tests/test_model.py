import pytest
import pandas as pd
import pickle
import os
import sys
import tempfile
from sklearn.ensemble import RandomForestClassifier

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestModelFunctionality:
    """Test suite for model training and functionality"""
    
    def test_data_structure(self):
        """Test that data has the expected structure"""
        # Create sample iris-like data
        sample_data = pd.DataFrame({
            'sepal_length': [5.1, 4.9, 6.2, 5.8],
            'sepal_width': [3.5, 3.0, 3.4, 2.7],
            'petal_length': [1.4, 1.4, 5.4, 5.1],
            'petal_width': [0.2, 0.2, 2.3, 1.9],
            'species': ['setosa', 'setosa', 'virginica', 'virginica']
        })
        
        # Test data integrity
        assert not sample_data.empty, "Data should not be empty"
        assert 'species' in sample_data.columns, "Species column must exist"
        assert sample_data.shape[1] == 5, "Should have exactly 5 columns"
        assert len(sample_data) > 0, "Should have data rows"
        
        # Test numeric columns
        numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(sample_data[col]), f"{col} should be numeric"
    
    def test_model_training(self):
        """Test that RandomForest model can be trained successfully"""
        # Create training data
        X = pd.DataFrame({
            'sepal_length': [5.1, 4.9, 6.2, 5.8, 6.9, 4.6],
            'sepal_width': [3.5, 3.0, 3.4, 2.7, 3.1, 3.1],
            'petal_length': [1.4, 1.4, 5.4, 5.1, 5.4, 1.5],
            'petal_width': [0.2, 0.2, 2.3, 1.9, 2.1, 0.2]
        })
        y = pd.Series(['setosa', 'setosa', 'virginica', 'virginica', 'virginica', 'setosa'])
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Test model properties
        assert hasattr(model, 'feature_importances_'), "Model should have feature importances"
        assert hasattr(model, 'n_estimators'), "Model should have n_estimators attribute"
        assert model.n_estimators == 10, "Model should have correct n_estimators"
        
        # Test predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y), "Predictions should match input length"
        assert model.score(X, y) > 0.7, "Model should have decent training accuracy"
    
    def test_model_persistence(self):
        """Test that model can be saved and loaded correctly"""
        # Create and train a simple model
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        y = [0, 1, 2]
        model.fit(X, y)
        
        # Test save/load cycle with temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            try:
                # Save model
                pickle.dump(model, f)
                f.flush()
                
                # Load model
                with open(f.name, 'rb') as load_f:
                    loaded_model = pickle.load(load_f)
                
                # Test that loaded model works the same
                test_input = [[1, 2, 3, 4]]
                original_pred = model.predict(test_input)
                loaded_pred = loaded_model.predict(test_input)
                
                assert original_pred[0] == loaded_pred[0], "Loaded model should predict same as original"
                assert loaded_model.n_estimators == model.n_estimators, "Model properties should match"
                
            finally:
                # Clean up temporary file
                os.unlink(f.name)
    
    def test_iris_dataset_if_exists(self):
        """Test the actual iris dataset if it exists (skip in CI)"""
        iris_path = "data/iris.csv"
        
        # Skip test if file doesn't exist (expected in CI environment)
        if not os.path.exists(iris_path):
            pytest.skip("iris.csv not found - this is expected in CI environment")
        
        # If file exists, validate it
        df = pd.read_csv(iris_path)
        
        # Test dataset structure
        expected_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        assert all(col in df.columns for col in expected_columns), "All expected columns should be present"
        assert len(df) >= 100, "Dataset should have substantial number of rows"
        assert df['species'].nunique() == 3, "Should have exactly 3 species"
        
        # Test data quality
        assert not df.isnull().any().any(), "Dataset should not have null values"
        
        # Test species values
        expected_species = {'setosa', 'versicolor', 'virginica'}
        actual_species = set(df['species'].unique())
        assert actual_species == expected_species, f"Species should be {expected_species}"

if __name__ == "__main__":
    # Run tests when file is executed directly
    pytest.main([__file__, "-v"])