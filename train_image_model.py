import os
import cv2
import numpy as np
import joblib

from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# =========================================================
# SETTINGS
# =========================================================

DATASET_PATH = "images"

CATEGORIES = [
    "pothole",
    "garbage",
    "streetlight",
    "water_leak",
    "drainage",
    "fallen_tree",
    "traffic_signal"
]

IMAGE_SIZE = (128, 128)


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(image):

    # Resize image
    image = cv2.resize(image, IMAGE_SIZE)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # HOG features
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    # Color features
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    color_features = []

    for channel in cv2.split(hsv):

        color_features.append(
            np.mean(channel)
        )

        color_features.append(
            np.std(channel)
        )

    color_features = np.array(color_features)

    # Combine features
    features = np.concatenate(
        [hog_features, color_features]
    )

    return features


# =========================================================
# IMAGE AUGMENTATION
# =========================================================

def augment_image(image):

    augmented = []

    # Original image
    augmented.append(image)

    # Horizontal flip
    flipped = cv2.flip(image, 1)
    augmented.append(flipped)

    # Slight rotation
    height, width = image.shape[:2]

    center = (width // 2, height // 2)

    matrix1 = cv2.getRotationMatrix2D(
        center,
        10,
        1.0
    )

    rotated1 = cv2.warpAffine(
        image,
        matrix1,
        (width, height)
    )

    augmented.append(rotated1)

    # Opposite rotation
    matrix2 = cv2.getRotationMatrix2D(
        center,
        -10,
        1.0
    )

    rotated2 = cv2.warpAffine(
        image,
        matrix2,
        (width, height)
    )

    augmented.append(rotated2)

    # Brightness adjustment
    brighter = cv2.convertScaleAbs(
        image,
        alpha=1.0,
        beta=25
    )

    augmented.append(brighter)

    return augmented


# =========================================================
# LOAD DATASET
# =========================================================

X = []
y = []

print()
print("==========================================")
print("Loading image dataset...")
print("==========================================")


for category in CATEGORIES:

    folder_path = os.path.join(
        DATASET_PATH,
        category
    )

    print()
    print("Category:", category)

    if not os.path.exists(folder_path):

        print(
            "WARNING: Folder not found:",
            folder_path
        )

        continue


    files = os.listdir(folder_path)

    image_count = 0


    for filename in files:

        image_path = os.path.join(
            folder_path,
            filename
        )

        image = cv2.imread(
            image_path
        )


        if image is None:

            print(
                "Skipped invalid image:",
                filename
            )

            continue


        image_count += 1

        # Create augmented images
        augmented_images = augment_image(
            image
        )


        for augmented_image in augmented_images:

            features = extract_features(
                augmented_image
            )

            X.append(features)

            y.append(category)


        print(
            "Loaded:",
            filename
        )


    print(
        "Original images:",
        image_count
    )


# =========================================================
# CONVERT TO NUMPY
# =========================================================

X = np.array(X)
y = np.array(y)


print()
print("==========================================")
print("Dataset loaded successfully!")
print("Total training samples:", len(X))
print("Number of features:", X.shape[1])
print("==========================================")


# =========================================================
# CHECK DATA
# =========================================================

if len(X) < 14:

    print()
    print(
        "Not enough images to train the model."
    )

    print(
        "Please add more images."
    )

    exit()


# =========================================================
# CHECK CATEGORY COUNTS
# =========================================================

print()
print("Samples per category:")

unique_classes, counts = np.unique(
    y,
    return_counts=True
)

for category, count in zip(
    unique_classes,
    counts
):

    print(
        category,
        ":",
        count
    )


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

# We only use stratification when every class
# has enough samples.

minimum_samples = min(counts)

if minimum_samples >= 2:

    print()
    print(
        "Using stratified train/test split..."
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

else:

    print()
    print(
        "Some classes have too few samples."
    )

    print(
        "Using random train/test split..."
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )


# =========================================================
# CREATE RANDOM FOREST
# =========================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# =========================================================
# TRAIN MODEL
# =========================================================

print()
print("==========================================")
print("Training improved image classification model...")
print("==========================================")


model.fit(
    X_train,
    y_train
)


print()
print("Training completed!")


# =========================================================
# TEST MODEL
# =========================================================

y_pred = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


print()
print("==========================================")
print(
    "Model Accuracy:",
    round(accuracy * 100, 2),
    "%"
)
print("==========================================")


print()
print("Classification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(
    model,
    "image_civic_model.pkl"
)


print()
print("==========================================")
print("Model saved successfully!")
print("File created: image_civic_model.pkl")
print("==========================================")