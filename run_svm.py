import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

# Load the dataset (Modify the path to your actual dataset)
dataset_path = "your_dataset.csv"  # Change this to your actual dataset file
df = pd.read_csv(dataset_path)

# Ensure the dataset is properly loaded
if df.empty:
    raise ValueError("The dataset is empty. Please check the file path and data.")

# Extract features (X) and labels (y)
if "label" not in df.columns:
    raise KeyError("The dataset must contain a 'label' column for classification.")

X = df.drop(columns=["label"])  # Assuming "label" is the column name for labels
y = df["label"]

# Convert to numpy arrays for sklearn compatibility
X = X.to_numpy()
y = y.to_numpy()

# Check for NaN values and remove them if necessary
if np.isnan(X).any() or np.isnan(y).any():
    print("Warning: NaN values detected. Removing NaN rows.")
    df = df.dropna()
    X = df.drop(columns=["label"]).to_numpy()
    y = df["label"].to_numpy()

# Ensure X and y have the same number of samples
if X.shape[0] != y.shape[0]:
    raise ValueError("Mismatch in number of samples between X and y.")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define SVM configurations based on experiment table
svm_experiments = [
    {"id": 1, "kernel": "linear", "gamma": "scale", "coef0": 0, "degree": 3},
    {"id": 2, "kernel": "poly", "gamma": 1, "coef0": 0, "degree": 2},
    {"id": 3, "kernel": "poly", "gamma": 0.1, "coef0": 0.5, "degree": 2},
    {"id": 4, "kernel": "rbf", "gamma": 0.5, "coef0": 0, "degree": 3},
    {"id": 5, "kernel": "sigmoid", "gamma": 0.5, "coef0": -0.2, "degree": 3},
]

# Train and save models
for exp in svm_experiments:
    print(f"Training SVM model {exp['id']} with {exp['kernel']} kernel...")
    
    model = SVC(kernel=exp["kernel"], gamma=exp["gamma"], coef0=exp["coef0"], degree=exp["degree"], probability=True)
    model.fit(X_train, y_train)

    # Save the model
    model_filename = f"model.{exp['id']}.pkl"
    joblib.dump(model, model_filename)
    print(f"Model {exp['id']} saved as {model_filename}")

print("All models trained and saved successfully.")
