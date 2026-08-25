import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score


# -----------------------------------------
# 1. LOAD DATASET
# -----------------------------------------

data = pd.read_csv("civic_text_dataset.csv")

print("Text dataset loaded!")
print("\nNumber of records:", len(data))


# -----------------------------------------
# 2. INPUT AND OUTPUT
# -----------------------------------------

X = data["Description"]

y = data["Issue_Type"]


# -----------------------------------------
# 3. SPLIT DATA
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------------------
# 4. CREATE NLP MODEL
# -----------------------------------------

model = Pipeline([

    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2)
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])


# -----------------------------------------
# 5. TRAIN MODEL
# -----------------------------------------

model.fit(X_train, y_train)

print("\nNLP model training completed!")


# -----------------------------------------
# 6. TEST MODEL
# -----------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n========== MODEL PERFORMANCE ==========")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# -----------------------------------------
# 7. TEST NEW COMPLAINT
# -----------------------------------------

new_complaints = [

    "There is a very big hole on the road",

    "Garbage is lying everywhere near the bus stop",

    "The street lamp is not working",

    "Water is coming out from a broken pipe",

    "The drain is completely blocked",

    "Traffic light is not working at the junction",

    "A large tree has fallen across the road"
]


print("\n========== TEST PREDICTIONS ==========")

for complaint in new_complaints:

    prediction = model.predict(
        [complaint]
    )[0]

    print("\nComplaint:", complaint)

    print("Predicted Issue:", prediction)


# -----------------------------------------
# 8. SAVE NLP MODEL
# -----------------------------------------

joblib.dump(
    model,
    "civic_text_model.pkl"
)

print("\nNLP model saved successfully!")

print("File created: civic_text_model.pkl")
