import pickle
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# LOAD DATASET
housing = fetch_california_housing(as_frame=True)
df = housing.frame
# FEATURES & TARGET

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MODEL TRAINING

model = RandomForestRegressor(
    # hyperparameters
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
)

model.fit(X_train, y_train)

# PREDICTIONS

y_pred = model.predict(X_test)

# EVALUATION

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)

print("R2 Score:", r2)

# SAVE MODEL

with open("models/random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully.")
