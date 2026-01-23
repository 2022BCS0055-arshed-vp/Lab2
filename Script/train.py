from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import os
import joblib

# Fetch dataset
print("Loading dataset")
wine_quality = fetch_ucirepo(id=186)

# Features and target
X = wine_quality.data.features
y = wine_quality.data.targets

print("Dataset variables")
print(wine_quality.variables)

# -----------------------------
# Correlation-based Feature Selection
# -----------------------------
print("Applying correlation-based feature selection")

X_corr = X.copy()
X_corr['target'] = y

corr_matrix = X_corr.corr()
corr_with_target = corr_matrix['target'].abs().sort_values(ascending=False)

# Select features with correlation > threshold (excluding target)
selected_features = corr_with_target[corr_with_target > 0.1].index.tolist()
selected_features.remove('target')

X_selected = X[selected_features]

print(f"Selected features: {selected_features}")

# -----------------------------
# Train-test split (80-20)
# -----------------------------
print("Splitting train test data")
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# -----------------------------
# Standardization
# -----------------------------
print("Applying standardization")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Train Lasso Regression
# -----------------------------
print("Training Linear Regression Model - Lasso")
model = Lasso()  # regularization enabled by default
model.fit(X_train_scaled, y_train)

# -----------------------------
# Save model
# -----------------------------
model_filename = 'output/model-linear-exp2.pkl'
os.makedirs(os.path.dirname(model_filename), exist_ok=True)
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")

# -----------------------------
# Evaluation
# -----------------------------
r2_score_value = model.score(X_test_scaled, y_test)
y_pred = model.predict(X_test_scaled)
mse_value = mean_squared_error(y_test, y_pred)

print(f"R^2 Score: {r2_score_value:.2f}")
print(f"Mean Squared Error (MSE): {mse_value:.2f}")

# -----------------------------
# Save metrics to JSON
# -----------------------------
print("Saving metrics as JSON")

data = {
    "Experiment ID": "Exp-02",
    "Model Type": "Linear Regression - Lasso",
    "Hyperparameters": "Regularization enabled",
    "Preprocessing-Steps": "Standardization",
    "Feature-Selection-Method": "correlation-based",
    "Train/Test-Split": "80-20",
    "MSE": mse_value,
    "R^2 Score": r2_score_value
}

filename = 'output/metrics.json'

if os.path.exists(filename):
    with open(filename, 'r') as json_file:
        existing_data = json.load(json_file)

    if isinstance(existing_data, list):
        existing_data.append(data)
    else:
        existing_data = [existing_data, data]

    with open(filename, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)
else:
    with open(filename, 'w') as json_file:
        json.dump([data], json_file, indent=4)

print(f"Data successfully saved to {filename}")
