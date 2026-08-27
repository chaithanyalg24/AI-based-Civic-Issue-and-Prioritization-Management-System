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
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Civic Issue Management System",
    page_icon="🏙️",
    layout="wide"
)

MODEL_PATH = "image_civic_model.pkl"
CSV_PATH = "complaints.csv"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    return None


model = load_model()


# =========================================================
# IMAGE FEATURE EXTRACTION
# =========================================================

def extract_features(image):

    image = cv2.resize(
        image,
        (128, 128)
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

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

def predict_issue(image):

    if model is None:

        return "Model Not Found", 0.0

    features = extract_features(
        image
    )

    features = features.reshape(
        1,
        -1
    )

    expected_features = (
        model.n_features_in_
    )

    actual_features = (
        features.shape[1]
    )

    if actual_features != expected_features:

        raise ValueError(
            f"Feature mismatch. "
            f"Model expects {expected_features} "
            f"features but application generated "
            f"{actual_features}."
        )

    prediction = model.predict(
        features
    )[0]

    probabilities = model.predict_proba(
        features
    )[0]

    confidence = (
        float(np.max(probabilities))
        * 100
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
# PRIORITY
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
        "garbage",
        "streetlight",
        "drainage"
    ]

    if issue in high_priority:

        if confidence >= 70:

            return "High"

        return "Medium"

    if issue in medium_priority:

        return "Medium"

    return "Low"


# =========================================================
# PRIORITY SCORE
# =========================================================

def calculate_priority_score(
    priority,
    confidence
):

    scores = {

        "Low": 30,

        "Medium": 60,

        "High": 85
    }

    score = scores.get(
        priority,
        30
    )

    score += (
        confidence * 0.15
    )

    return round(
        min(score, 100),
        2
    )


# =========================================================
# COMPLAINT ID
# =========================================================

def generate_complaint_id():

    year = datetime.now().year

    while True:

        number = random.randint(
            1000,
            9999
        )

        complaint_id = (
            f"CIVIC-{year}-{number}"
        )

        if os.path.exists(CSV_PATH):

            existing = pd.read_csv(
                CSV_PATH
            )

            if (
                "Complaint_ID"
                in existing.columns
            ):

                if complaint_id in (
                    existing["Complaint_ID"]
                    .astype(str)
                    .values
                ):

                    continue

        return complaint_id


# =========================================================
# SAVE COMPLAINT
# =========================================================

def save_complaint(
    complaint_data
):

    new_data = pd.DataFrame(
        [complaint_data]
    )

    if os.path.exists(CSV_PATH):

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

    else:

        final_data = new_data

    final_data.to_csv(
        CSV_PATH,
        index=False
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "🏙️ Civic AI"
)

st.sidebar.write(
    "Select Page"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📝 Submit Complaint",
        "🔎 Track Complaint",
        "🏢 Department Dashboard",
        "ℹ️ About Project"
    ]
)


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.title(
        "🏙️ AI Civic Issue Management System"
    )

    st.subheader(
        "AI-Based Civic Issue Detection and Prioritization"
    )

    st.write(
        """
        Welcome to the AI Civic Issue Management System.

        This platform allows citizens to report civic
        problems using images. The AI model identifies
        the issue, calculates its priority and assigns
        it to the appropriate department.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader(
            "📝 Submit Complaint"
        )

        st.write(
            "Citizens can submit civic issues "
            "with images."
        )

    with col2:

        st.subheader(
            "🤖 AI Detection"
        )

        st.write(
            "AI identifies the type of civic issue "
            "and calculates priority."
        )

    with col3:

        st.subheader(
            "🔎 Track Complaint"
        )

        st.write(
            "Citizens can track complaints using "
            "their Complaint ID."
        )

    st.divider()

    st.subheader(
        "🚧 Supported Civic Issues"
    )

    st.write(
        """
        🕳️ Pothole

        🗑️ Garbage

        💡 Streetlight

        💧 Water Leak

        🚰 Drainage

        🌳 Fallen Tree

        🚦 Traffic Signal
        """
    )

    if model is not None:

        st.success(
            "🤖 AI model loaded successfully."
        )

    else:

        st.error(
            "❌ image_civic_model.pkl not found."
        )


# =========================================================
# SUBMIT COMPLAINT
# =========================================================

elif page == "📝 Submit Complaint":

    st.title(
        "📝 Submit Civic Complaint"
    )

    citizen_name = st.text_input(
        "Citizen Name"
    )

    location = st.text_input(
        "Location"
    )

    description = st.text_area(
        "Describe the Problem"
    )

    uploaded_image = st.file_uploader(
        "Upload Problem Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_image is not None:

        st.image(
            uploaded_image,
            caption="Uploaded Civic Issue",
            width=500
        )

    if st.button(
        "🚨 Submit Complaint",
        type="primary"
    ):

        if not citizen_name.strip():

            st.error(
                "Please enter your name."
            )

            st.stop()

        if not location.strip():

            st.error(
                "Please enter the location."
            )

            st.stop()

        if not description.strip():

            st.error(
                "Please describe the problem."
            )

            st.stop()

        if uploaded_image is None:

            st.error(
                "Please upload an image."
            )

            st.stop()

        image_bytes = np.asarray(
            bytearray(
                uploaded_image.getvalue()
            ),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_bytes,
            cv2.IMREAD_COLOR
        )

        if image is None:

            st.error(
                "Invalid image file."
            )

            st.stop()

        try:

            detected_issue, confidence = (
                predict_issue(image)
            )

        except Exception as error:

            st.error(
                "AI prediction failed."
            )

            st.code(
                str(error)
            )

            st.stop()

        department = assign_department(
            detected_issue
        )

        priority = calculate_priority(
            detected_issue,
            confidence
        )

        priority_score = (
            calculate_priority_score(
                priority,
                confidence
            )
        )

        complaint_id = (
            generate_complaint_id()
        )

        submitted_at = (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

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
                round(
                    confidence,
                    2
                ),

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

        save_complaint(
            complaint_data
        )

        st.success(
            "✅ Complaint submitted successfully!"
        )

        col1, col2, col3 = (
            st.columns(3)
        )

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

        st.info(
            f"🏢 Assigned Department: "
            f"{department}"
        )

        st.write(
            f"📊 Priority Score: "
            f"**{priority_score}/100**"
        )

        st.subheader(
            "🆔 Complaint ID"
        )

        st.code(
            complaint_id
        )

        st.warning(
            "Save this Complaint ID for tracking."
        )

        st.info(
            "🟡 Status: Pending"
        )


# =========================================================
# TRACK COMPLAINT
# =========================================================

elif page == "🔎 Track Complaint":

    st.title(
        "🔎 Track Your Civic Complaint"
    )

    st.write(
        "Enter your Complaint ID to check the current status."
    )

    if not os.path.exists(CSV_PATH):

        st.warning(
            "No complaints have been submitted yet."
        )

        st.stop()

    data = pd.read_csv(
        CSV_PATH
    )

    complaint_id = st.text_input(
        "Enter Complaint ID",
        placeholder="CIVIC-2026-1234"
    )

    if st.button(
        "🔎 Track Complaint",
        type="primary"
    ):

        complaint_id = (
            complaint_id.strip()
        )

        if not complaint_id:

            st.error(
                "Please enter Complaint ID."
            )

        else:

            result = data[
                data["Complaint_ID"]
                .astype(str)
                .str.strip()
                .str.upper()
                ==
                complaint_id.upper()
            ]

            if result.empty:

                st.error(
                    "❌ Complaint ID not found."
                )

            else:

                complaint = result.iloc[0]

                st.success(
                    "✅ Complaint found!"
                )

                col1, col2 = (
                    st.columns(2)
                )

                with col1:

                    st.write(
                        f"**Complaint ID:** "
                        f"{complaint['Complaint_ID']}"
                    )

                    st.write(
                        f"**Citizen:** "
                        f"{complaint['Citizen_Name']}"
                    )

                    st.write(
                        f"**Location:** "
                        f"{complaint['Location']}"
                    )

                    st.write(
                        f"**Issue:** "
                        f"{str(complaint['Issue']).replace('_', ' ').title()}"
                    )

                with col2:

                    st.write(
                        f"**Department:** "
                        f"{complaint['Department']}"
                    )

                    st.write(
                        f"**Priority:** "
                        f"{complaint['Priority']}"
                    )

                    st.write(
                        f"**Priority Score:** "
                        f"{complaint['Priority_Score']}"
                    )

                    st.write(
                        f"**Submitted:** "
                        f"{complaint['Submitted_At']}"
                    )

                st.subheader(
                    "📝 Description"
                )

                st.write(
                    complaint["Description"]
                )

                st.divider()

                st.subheader(
                    "📌 Current Status"
                )

                status = str(
                    complaint["Status"]
                ).strip()

                if status == "Pending":

                    st.warning(
                        "🟡 Pending – "
                        "Complaint has been received."
                    )

                elif status == "In Progress":

                    st.info(
                        "🔵 In Progress – "
                        "Department is working on it."
                    )

                elif status == "Resolved":

                    st.success(
                        "🟢 Resolved – "
                        "Civic issue has been resolved."
                    )

                else:

                    st.write(
                        f"Status: {status}"
                    )


# =========================================================
# DEPARTMENT DASHBOARD
# =========================================================

elif page == "🏢 Department Dashboard":

    st.title(
        "🏢 Department Dashboard"
    )

    st.write(
        "Department officials can view and update complaints."
    )

    if not os.path.exists(CSV_PATH):

        st.warning(
            "No complaints have been submitted yet."
        )

        st.stop()

    data = pd.read_csv(
        CSV_PATH
    )

    st.subheader(
        "🔐 Department Login"
    )

    username = st.text_input(
        "Username",
        key="dept_username"
    )

    password = st.text_input(
        "Password",
        type="password",
        key="dept_password"
    )

    credentials = {

        "road_admin": (
            "road123",
            "Road Department"
        ),

        "sanitation_admin": (
            "sanitation123",
            "Sanitation Department"
        ),

        "electrical_admin": (
            "electrical123",
            "Electrical Department"
        ),

        "water_admin": (
            "water123",
            "Water Supply Department"
        ),

        "drainage_admin": (
            "drainage123",
            "Drainage Department"
        ),

        "parks_admin": (
            "parks123",
            "Parks and Garden Department"
        ),

        "traffic_admin": (
            "traffic123",
            "Traffic Department"
        )
    }

    if st.button(
        "🔐 Login",
        key="login_button"
    ):

        if username not in credentials:

            st.error(
                "❌ Invalid username."
            )

            st.session_state[
                "logged_in"
            ] = False

        elif password != credentials[
            username
        ][0]:

            st.error(
                "❌ Incorrect password."
            )

            st.session_state[
                "logged_in"
            ] = False

        else:

            st.session_state[
                "logged_in"
            ] = True

            st.session_state[
                "department"
            ] = credentials[
                username
            ][1]

            st.success(
                "✅ Login successful!"
            )

    # -----------------------------------------------------
    # LOGGED IN DASHBOARD
    # -----------------------------------------------------

    if st.session_state.get(
        "logged_in",
        False
    ):

        department = (
            st.session_state[
                "department"
            ]
        )

        st.divider()

        st.subheader(
            f"🏢 {department}"
        )

        # ---------------------------------------------
        # GET CURRENT DATA FROM CSV
        # ---------------------------------------------

        data = pd.read_csv(
            CSV_PATH
        )

        department_data = data[
            data["Department"]
            .astype(str)
            .str.strip()
            ==
            department
        ].copy()

        total = len(
            department_data
        )

        pending = len(
            department_data[
                department_data[
                    "Status"
                ]
                .astype(str)
                .str.strip()
                ==
                "Pending"
            ]
        )

        in_progress = len(
            department_data[
                department_data[
                    "Status"
                ]
                .astype(str)
                .str.strip()
                ==
                "In Progress"
            ]
        )

        resolved = len(
            department_data[
                department_data[
                    "Status"
                ]
                .astype(str)
                .str.strip()
                ==
                "Resolved"
            ]
        )

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        with col1:

            st.metric(
                "Total",
                total
            )

        with col2:

            st.metric(
                "Pending",
                pending
            )

        with col3:

            st.metric(
                "In Progress",
                in_progress
            )

        with col4:

            st.metric(
                "Resolved",
                resolved
            )

        st.divider()

        # ---------------------------------------------
        # COMPLAINT LIST
        # ---------------------------------------------

        st.subheader(
            "📋 Assigned Complaints"
        )

        if department_data.empty:

            st.info(
                "No complaints assigned "
                "to this department."
            )

        else:

            st.dataframe(
                department_data,
                use_container_width=True
            )

            st.divider()

            # -----------------------------------------
            # UPDATE STATUS
            # -----------------------------------------

            st.subheader(
                "🔄 Update Complaint Status"
            )

            complaint_ids = (
                department_data[
                    "Complaint_ID"
                ]
                .astype(str)
                .str.strip()
                .tolist()
            )

            selected_id = st.selectbox(
                "Select Complaint ID",
                complaint_ids,
                key="selected_complaint"
            )

            selected_rows = data[
                data["Complaint_ID"]
                .astype(str)
                .str.strip()
                ==
                selected_id
            ]

            if not selected_rows.empty:

                current_status = str(
                    selected_rows.iloc[0][
                        "Status"
                    ]
                ).strip()

                st.write(
                    f"Current Status: "
                    f"**{current_status}**"
                )

                status_options = [
                    "Pending",
                    "In Progress",
                    "Resolved"
                ]

                new_status = st.selectbox(
                    "Select New Status",
                    status_options,
                    index=status_options.index(
                        current_status
                    )
                    if current_status
                    in status_options
                    else 0,
                    key="new_status"
                )

                if st.button(
                    "💾 Update Status",
                    type="primary",
                    key="update_status_button"
                ):

                    # ---------------------------------
                    # UPDATE DATAFRAME
                    # ---------------------------------

                    mask = (
                        data[
                            "Complaint_ID"
                        ]
                        .astype(str)
                        .str.strip()
                        ==
                        selected_id
                    )

                    data.loc[
                        mask,
                        "Status"
                    ] = new_status

                    # ---------------------------------
                    # SAVE TO CSV
                    # ---------------------------------

                    data.to_csv(
                        CSV_PATH,
                        index=False
                    )

                    # ---------------------------------
                    # VERIFY SAVE
                    # ---------------------------------

                    verify_data = pd.read_csv(
                        CSV_PATH
                    )

                    verify_row = (
                        verify_data[
                            verify_data[
                                "Complaint_ID"
                            ]
                            .astype(str)
                            .str.strip()
                            ==
                            selected_id
                        ]
                    )

                    if not verify_row.empty:

                        saved_status = str(
                            verify_row.iloc[0][
                                "Status"
                            ]
                        ).strip()

                        if (
                            saved_status
                            ==
                            new_status
                        ):

                            st.success(
                                f"✅ Complaint "
                                f"{selected_id} "
                                f"updated successfully "
                                f"to **{new_status}**."
                            )

                            st.session_state[
                                "last_updated_status"
                            ] = new_status

                            st.rerun()

                        else:

                            st.error(
                                "❌ Status verification failed."
                            )

                    else:

                        st.error(
                            "❌ Complaint ID not found "
                            "after saving."
                        )

        st.divider()

        # ---------------------------------------------
        # LOGOUT
        # ---------------------------------------------

        if st.button(
            "🚪 Logout",
            key="logout_button"
        ):

            st.session_state[
                "logged_in"
            ] = False

            st.session_state[
                "department"
            ] = ""

            st.rerun()


# =========================================================
# ABOUT PROJECT
# =========================================================

elif page == "ℹ️ About Project":

    st.title(
        "ℹ️ About the Project"
    )

    st.subheader(
        "AI-Based Civic Issue Detection and Prioritization"
    )

    st.write(
        """
        The AI Civic Issue Management System is designed
        to make civic complaint management easier and
        more efficient.

        Citizens can upload an image of a civic issue.
        The AI model identifies the issue and calculates
        its priority.

        The system automatically assigns the complaint
        to the appropriate department.

        A unique Complaint ID is generated for tracking.

        Department officials can log in and update the
        complaint status.
        """
    )

    st.divider()

    st.subheader(
        "🎯 Main Objectives"
    )

    st.write(
        """
        • Detect civic issues using images

        • Classify the issue using machine learning

        • Calculate complaint priority

        • Automatically assign departments

        • Generate unique Complaint IDs

        • Allow citizens to track complaints

        • Allow departments to update complaint status
        """
    )

    st.subheader(
        "🛠️ Technologies Used"
    )

    st.write(
        """
        Python
        Streamlit
        OpenCV
        Scikit-learn
        Scikit-image
        Pandas
        NumPy
        Joblib
        """
    )

    if model is not None:

        st.success(
            "🤖 AI model loaded successfully."
        )

    else:

        st.error(
            "❌ AI model not found."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🏙️ AI Civic Issue Management System"
)
