"""
Data validation script for iris dataset
Validates data integrity, structure, and quality
"""
import pandas as pd
import sys
import os

def create_sample_data():
    """Create sample iris data for CI/testing environments"""
    try:
        from sklearn.datasets import load_iris
        
        print("📝 Creating sample iris dataset...")
        os.makedirs("data", exist_ok=True)
        
        # Load sklearn iris dataset
        iris = load_iris()
        df = pd.DataFrame(
            iris.data, 
            columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        )
        df['species'] = iris.target_names[iris.target]
        
        # Save to CSV
        df.to_csv("data/iris.csv", index=False)
        print("✅ Sample iris.csv created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create sample data: {str(e)}")
        return False

def validate_data():
    """Validate iris dataset integrity and quality"""
    print("🔍 Starting data validation...")
    
    try:
        data_path = "data/iris.csv"
        
        # Check if data file exists
        if not os.path.exists(data_path):
            print(f"⚠️ Data file not found: {data_path}")
            print("Creating sample data for CI environment...")
            if create_sample_data():
                print("✅ Sample data created, proceeding with validation")
            else:
                print("❌ Failed to create sample data")
                return False
        
        # Load and validate data
        df = pd.read_csv(data_path)
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
        
        # Validate required columns
        required_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        print("✅ All required columns present")
        
        # Check for null values
        null_counts = df.isnull().sum()
        if null_counts.any():
            print(f"❌ Found null values:\n{null_counts[null_counts > 0]}")
            return False
        print("✅ No null values found")
        
        # Validate data types
        numeric_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"❌ Column {col} is not numeric")
                return False
        print("✅ All numeric columns have correct data types")
        
        # Validate data ranges (reasonable values for iris measurements)
        for col in numeric_columns:
            if (df[col] < 0).any():
                print(f"❌ Column {col} has negative values")
                return False
            if (df[col] > 50).any():  # Unreasonably large values
                print(f"⚠️ Column {col} has very large values (>50)")
        print("✅ Data ranges are reasonable")
        
        # Validate species values
        expected_species = {'setosa', 'versicolor', 'virginica'}
        actual_species = set(df['species'].unique())
        if not actual_species.issubset(expected_species):
            unexpected = actual_species - expected_species
            print(f"❌ Unexpected species found: {unexpected}")
            return False
        print("✅ Species values are valid")
        
        # Check dataset size
        if len(df) < 50:
            print(f"⚠️ Dataset is quite small: {len(df)} rows")
        else:
            print(f"✅ Dataset size is adequate: {len(df)} rows")
        
        # Print summary statistics
        print("\n📊 Dataset Summary:")
        print(f"   • Total rows: {len(df)}")
        print(f"   • Total columns: {len(df.columns)}")
        print(f"   • Species distribution:")
        for species, count in df['species'].value_counts().items():
            print(f"     - {species}: {count}")
        
        print("\n🎉 Data validation completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Data validation failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = validate_data()
    if not success:
        sys.exit(1)
    print("✅ Data validation passed!")