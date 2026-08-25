import cv2
import numpy as np
import joblib


# -----------------------------------------
# LOAD TRAINED MODEL
# -----------------------------------------

model = joblib.load("image_civic_model.pkl")


# -----------------------------------------
# IMAGE FEATURE EXTRACTION
# -----------------------------------------

def extract_features(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None

    image = cv2.resize(image, (64, 64))

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image / 255.0

    features = image.flatten()

    return features


# -----------------------------------------
# TEST IMAGE
# -----------------------------------------

image_path = "images/pothole/pothole1.jpg"

features = extract_features(image_path)

if features is None:

    print("Could not read image.")

else:

    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)

    print("--------------------------------")
    print("AI DETECTED ISSUE:", prediction[0])
    print("--------------------------------")