import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import joblib


# -----------------------------------------
# 1. LOAD DATASET
# -----------------------------------------

data = pd.read_csv("civic_dataset.csv")

print("Dataset loaded successfully!")
print("\nDataset:")
print(data)


# -----------------------------------------
# 2. SELECT FEATURES
# -----------------------------------------

X = data[
    [
        "Severity",
        "Traffic_Level",
        "Similar_Complaints",
        "Safety_Risk"
    ]
]


# -----------------------------------------
# 3. SELECT TARGET
# -----------------------------------------

y = data["Priority_Score"]


# -----------------------------------------
# 4. SPLIT DATA
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# -----------------------------------------
# 5. CREATE ML MODEL
# -----------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# -----------------------------------------
# 6. TRAIN MODEL
# -----------------------------------------

model.fit(X_train, y_train)

print("\nModel training completed!")


# -----------------------------------------
# 7. MAKE PREDICTIONS
# -----------------------------------------

predictions = model.predict(X_test)


# -----------------------------------------
# 8. EVALUATE MODEL
# -----------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)


print("\n========== MODEL PERFORMANCE ==========")

print("Mean Absolute Error:", round(mae, 2))

print("R2 Score:", round(r2, 2))


# -----------------------------------------
# 9. TEST ONE NEW CIVIC ISSUE
# -----------------------------------------

new_issue = pd.DataFrame({
    "Severity": [9],
    "Traffic_Level": [9],
    "Similar_Complaints": [15],
    "Safety_Risk": [9]
})


predicted_priority = model.predict(
    new_issue
)[0]


print("\n========== NEW ISSUE ==========")

print("Severity: 9")
print("Traffic: 9")
print("Complaints: 15")
print("Safety Risk: 9")

print(
    "Predicted Priority Score:",
    round(predicted_priority, 2)
)


# -----------------------------------------
# 10. SAVE MODEL
# -----------------------------------------

joblib.dump(
    model,
    "civic_priority_model.pkl"
)

print("\nML model saved successfully!")

print("File created: civic_priority_model.pkl")
