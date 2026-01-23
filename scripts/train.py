import pandas as pd
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------
# CONFIG
# --------------------------------
STUDENT_NAME = "Arshed V P"
ROLL_NUMBER = "2022BCS0055"

DATA_PATH = "Dataset/winequality-red.csv"
ARTIFACTS_DIR = "artifacts"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# --------------------------------
# LOAD DATA
# --------------------------------
df = pd.read_csv(DATA_PATH, sep=';')

X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --------------------------------
# TRAIN MODEL
# --------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# --------------------------------
# EVALUATION
# --------------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metrics = {
    "mse": mse,
    "r2_score": r2
}

# --------------------------------
# SAVE ARTIFACTS
# --------------------------------
joblib.dump(model, f"{ARTIFACTS_DIR}/model.joblib")

with open(f"{ARTIFACTS_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Training complete")
print(metrics)
