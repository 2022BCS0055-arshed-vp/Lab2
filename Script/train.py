from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import json
import os
import joblib

# Fetch dataset
print("Loading dataset")
wine_quality = fetch_ucirepo(id=186)

# Features and target (ALL features used)
X = wine_quality.data.features
y = wine_quality.data.targets

print("Dataset variables")
print(wine_quality.variables)

# -----------------------------
# Train-test split (80-20)
# -----------------------------
print("Splitting train test data")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train Random Forest Model
# -----------------------------
print("Training Random Forest Regressor")
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Save model
# -----------------------------
model_filename = 'output/model-rf-exp3.pkl'
os.makedirs(os.path.dirname(model_filename), exist_ok=True)
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")

# -----------------------------
# Evaluation
# -----------------------------
r2_score_value = model.score(X_test, y_test)
y_pred = model.predict(X_test)
mse_value = mean_squared_error(y_test, y_pred)

print(f"R^2 Score: {r2_score_value:.2f}")
print(f"Mean Squared Error (MSE): {mse_value:.2f}")

# -----------------------------
# Save metrics to JSON
# -----------------------------
print("Saving metrics as JSON")

data = {
    "Experiment ID": "Exp-03",
    "Model Type": "Random Forest",
    "Hyperparameters": "50 trees, depth 10",
    "Preprocessing-Steps": "None",
    "Feature-Selection-Method": "All",
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
