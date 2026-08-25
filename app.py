import streamlit as st
import pandas as pd
import numpy as np
import cv2
import joblib
import os
import random
from datetime import datetime
from skimage.feature import hog


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="AI Civic Issue Management System",
    page_icon="🏙️",
    layout="wide"
)


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "image_civic_model.pkl"
CSV_PATH = "complaints.csv"

IMAGE_SIZE = (128, 128)


# =========================================================
# LOAD ML MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


model = load_model()


# =========================================================
# IMAGE FEATURE EXTRACTION
# MUST BE EXACTLY THE SAME AS TRAINING
# =========================================================

def extract_image_features(image):

    # Resize
    image = cv2.resize(
        image,
        IMAGE_SIZE
    )

    # Grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # HOG features
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    # HSV color features
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    color_features = []

    for channel in cv2.split(hsv):

        color_features.append(
            np.mean(channel)
        )

        color_features.append(
            np.std(channel)
        )

    color_features = np.array(
        color_features
    )

    # Combine features
    features = np.concatenate(
        [
            hog_features,
            color_features
        ]
    )

    return features


# =========================================================
# PREDICT CIVIC ISSUE
# =========================================================

def predict_issue(image_path):

    if model is None:

        return "Model Not Found", 0.0

    # Read image
    image = cv2.imread(
        image_path
    )

    if image is None:

        return "Unknown", 0.0

    # Extract SAME features used during training
    features = extract_image_features(
        image
    )

    # Reshape
    features = features.reshape(
        1,
        -1
    )

    # Safety check
    expected_features = model.n_features_in_

    if features.shape[1] != expected_features:

        raise ValueError(
            f"Feature mismatch: "
            f"model expects {expected_features}, "
            f"but app generated {features.shape[1]}"
        )

    # Prediction
    prediction = model.predict(
        features
    )[0]

    # Confidence
    probabilities = model.predict_proba(
        features
    )[0]

    confidence = float(
        np.max(probabilities) * 100
    )

    return prediction, confidence


# =========================================================
# DEPARTMENT ASSIGNMENT
# =========================================================

def assign_department(issue):

    departments = {

        "pothole":
            "Road Department",

        "garbage":
            "Sanitation Department",

        "streetlight":
            "Electrical Department",

        "water_leak":
            "Water Supply Department",

        "drainage":
            "Drainage Department",

        "fallen_tree":
            "Parks and Garden Department",

        "traffic_signal":
            "Traffic Department"
    }

    return departments.get(
        issue,
        "Municipal Corporation"
    )


# =========================================================
# PRIORITY CALCULATION
# =========================================================

def calculate_priority(
    issue,
    confidence
):

    high_priority = [
        "pothole",
        "water_leak",
        "fallen_tree",
        "traffic_signal"
    ]

    medium_priority = [
        "drainage",
        "streetlight"
    ]

    if issue in high_priority:

        if confidence >= 70:
            return "High"

        return "Medium"

    elif issue in medium_priority:

        return "Medium"

    elif issue == "garbage":

        return "Medium"

    return "Low"


# =========================================================
# PRIORITY SCORE
# =========================================================

def calculate_priority_score(
    priority,
    confidence
):

    base_scores = {

        "Low": 30,

        "Medium": 60,

        "High": 85,

        "Critical": 100
    }

    base = base_scores.get(
        priority,
        30
    )

    score = base + (
        confidence * 0.15
    )

    return round(
        min(score, 100),
        2
    )


# =========================================================
# GENERATE COMPLAINT ID
# =========================================================

def generate_complaint_id():

    year = datetime.now().year

    number = random.randint(
        1000,
        9999
    )

    return f"CIVIC-{year}-{number}"


# =========================================================
# SAVE COMPLAINT
# =========================================================

def save_complaint(data):

    new_data = pd.DataFrame(
        [data]
    )

    if os.path.exists(CSV_PATH):

        try:

            old_data = pd.read_csv(
                CSV_PATH
            )

            final_data = pd.concat(
                [
                    old_data,
                    new_data
                ],
                ignore_index=True
            )

        except Exception:

            final_data = new_data

    else:

        final_data = new_data

    final_data.to_csv(
        CSV_PATH,
        index=False
    )


# =========================================================
# TITLE
# =========================================================

st.title(
    "🏙️ AI Civic Issue Management System"
)

st.write(
    "Report civic problems using an image and "
    "complaint description."
)


# =========================================================
# MODEL STATUS
# =========================================================

if model is not None:

    st.success(
        "🤖 AI Image Classification Model Loaded"
    )

    st.caption(
        f"Model expects {model.n_features_in_} features."
    )

else:

    st.error(
        "❌ image_civic_model.pkl was not found."
    )

    st.stop()


# =========================================================
# SUBMIT COMPLAINT
# =========================================================

st.header(
    "📢 Submit a Civic Complaint"
)


# =========================================================
# INPUT FIELDS
# =========================================================

citizen_name = st.text_input(
    "Citizen Name",
    placeholder="Enter your name"
)


location = st.text_input(
    "Location",
    placeholder="Example: Mysore"
)


description = st.text_area(
    "Describe the problem",
    placeholder="Describe the civic issue..."
)


uploaded_image = st.file_uploader(
    "Upload an image of the problem",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# =========================================================
# IMAGE PREVIEW
# =========================================================

if uploaded_image is not None:

    st.subheader(
        "🖼️ Uploaded Civic Issue"
    )

    st.image(
        uploaded_image,
        caption="Uploaded Image",
        width=500
    )


# =========================================================
# SUBMIT BUTTON
# =========================================================

if st.button(
    "🚨 Submit Complaint",
    type="primary"
):

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    if citizen_name.strip() == "":

        st.error(
            "❌ Please enter your name."
        )

        st.stop()


    if location.strip() == "":

        st.error(
            "❌ Please enter your location."
        )

        st.stop()


    if description.strip() == "":

        st.error(
            "❌ Please describe the problem."
        )

        st.stop()


    if uploaded_image is None:

        st.error(
            "❌ Please upload an image."
        )

        st.stop()


    # -----------------------------------------------------
    # SAVE TEMPORARY IMAGE
    # -----------------------------------------------------

    temp_image_path = (
        "temp_uploaded_image.jpg"
    )


    with open(
        temp_image_path,
        "wb"
    ) as file:

        file.write(
            uploaded_image.getbuffer()
        )


    # -----------------------------------------------------
    # AI PREDICTION
    # -----------------------------------------------------

    try:

        detected_issue, confidence = (
            predict_issue(
                temp_image_path
            )
        )

    except Exception as e:

        st.error(
            "❌ AI image analysis failed."
        )

        st.code(
            str(e)
        )

        st.stop()


    # -----------------------------------------------------
    # DEPARTMENT
    # -----------------------------------------------------

    department = assign_department(
        detected_issue
    )


    # -----------------------------------------------------
    # PRIORITY
    # -----------------------------------------------------

    priority = calculate_priority(
        detected_issue,
        confidence
    )


    priority_score = calculate_priority_score(
        priority,
        confidence
    )


    # -----------------------------------------------------
    # COMPLAINT ID
    # -----------------------------------------------------

    complaint_id = generate_complaint_id()


    # -----------------------------------------------------
    # DATE AND TIME
    # -----------------------------------------------------

    submitted_at = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    # -----------------------------------------------------
    # COMPLAINT DATA
    # -----------------------------------------------------

    complaint_data = {

        "Complaint_ID":
            complaint_id,

        "Citizen_Name":
            citizen_name,

        "Location":
            location,

        "Description":
            description,

        "Issue":
            detected_issue,

        "AI_Confidence":
            round(confidence, 2),

        "Department":
            department,

        "Priority":
            priority,

        "Priority_Score":
            priority_score,

        "Status":
            "Pending",

        "Submitted_At":
            submitted_at
    }


    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    save_complaint(
        complaint_data
    )


    # -----------------------------------------------------
    # SUCCESS
    # -----------------------------------------------------

    st.success(
        "✅ Complaint submitted successfully!"
    )


    # =====================================================
    # RESULT
    # =====================================================

    st.header(
        "🤖 AI Analysis Result"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Detected Issue",
            detected_issue.replace(
                "_",
                " "
            ).title()
        )


    with col2:

        st.metric(
            "AI Confidence",
            f"{confidence:.2f}%"
        )


    with col3:

        st.metric(
            "Priority",
            priority
        )


    # -----------------------------------------------------
    # DEPARTMENT
    # -----------------------------------------------------

    st.subheader(
        "🏢 Assigned Department"
    )

    st.info(
        department
    )


    # -----------------------------------------------------
    # PRIORITY SCORE
    # -----------------------------------------------------

    st.subheader(
        "📊 Priority Score"
    )

    st.progress(
        int(priority_score)
    )

    st.write(
        f"Priority Score: **{priority_score}/100**"
    )


    # -----------------------------------------------------
    # COMPLAINT ID
    # -----------------------------------------------------

    st.subheader(
        "🆔 Complaint ID"
    )

    st.code(
        complaint_id
    )


    st.write(
        "Please save this Complaint ID "
        "to track your complaint later."
    )


    # -----------------------------------------------------
    # STATUS
    # -----------------------------------------------------

    st.subheader(
        "📌 Current Status"
    )

    st.warning(
        "🟡 Pending – Your complaint has been received."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI Civic Issue Management System | "
    "Machine Learning + Computer Vision + Streamlit"
)