# train.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # Changed import
import pickle

# Load data
df = pd.read_csv("data/iris.csv")
X = df.drop("species", axis=1)
y = df["species"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)  # Changed model
model.fit(X, y)

# Save model
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save metric
with open("metrics.txt", "w") as f:
    f.write(f"Training accuracy: {model.score(X, y):.4f}\n")
