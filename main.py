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


# =========================================================
# FILE SETTINGS
# =========================================================

MODEL_PATH = "image_civic_model.pkl"
CSV_PATH = "complaints.csv"


# =========================================================
# PAGE STYLE
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


model = load_model()


# =========================================================
# IMAGE FEATURE EXTRACTION
# =========================================================

def extract_features(image):

    # Resize image
    image = cv2.resize(
        image,
        (128, 128)
    )

    # Convert to grayscale
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

    # Convert to HSV
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
# AI PREDICTION
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

    # Check feature count
    if features.shape[1] != model.n_features_in_:

        raise ValueError(
            f"Feature mismatch. "
            f"Model expects {model.n_features_in_} "
            f"features but application generated "
            f"{features.shape[1]} features."
        )

    prediction = model.predict(
        features
    )[0]

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

    elif issue in medium_priority:

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

        "High": 85
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

    return (
        f"CIVIC-{year}-{number}"
    )


# =========================================================
# SAVE COMPLAINT
# =========================================================

def save_complaint(data):

    new_data = pd.DataFrame(
        [data]
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
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🏙️ AI Civic Issue Management System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Based Civic Issue Detection, Prioritization '
    'and Complaint Management'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "🏙️ Civic AI"
)

st.sidebar.write(
    "Navigation"
)

page = st.sidebar.radio(
    "Select Page",
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

    st.header(
        "Welcome to AI Civic Issue Management System"
    )

    st.write(
        """
        This system allows citizens to report civic issues
        using images and descriptions. The AI model identifies
        the issue, calculates its priority and automatically
        assigns the complaint to the appropriate department.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader("📝 Report")

        st.write(
            "Citizens can submit civic complaints "
            "with an image."
        )

    with col2:

        st.subheader("🤖 AI Detection")

        st.write(
            "Machine learning identifies the type "
            "of civic problem."
        )

    with col3:

        st.subheader("🔎 Track")

        st.write(
            "Citizens can track their complaint "
            "using the Complaint ID."
        )

    st.divider()

    st.subheader(
        "🚧 Supported Civic Issues"
    )

    issues = [

        "🕳️ Pothole",

        "🗑️ Garbage",

        "💡 Streetlight",

        "💧 Water Leak",

        "🚰 Drainage",

        "🌳 Fallen Tree",

        "🚦 Traffic Signal"
    ]

    for issue in issues:

        st.write(issue)

    st.divider()

    if model is not None:

        st.success(
            f"🤖 AI Model loaded successfully. "
            f"Features: {model.n_features_in_}"
        )

    else:

        st.error(
            "❌ AI model not found."
        )


# =========================================================
# SUBMIT COMPLAINT
# =========================================================

elif page == "📝 Submit Complaint":

    st.header(
        "📝 Submit a Civic Complaint"
    )

    citizen_name = st.text_input(
        "Citizen Name",
        placeholder="Enter your name"
    )

    location = st.text_input(
        "Location",
        placeholder="Enter location"
    )

    description = st.text_area(
        "Describe the Problem",
        placeholder="Describe the civic issue"
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

        # ---------------------------------------------
        # VALIDATION
        # ---------------------------------------------

        if citizen_name.strip() == "":

            st.error(
                "Please enter your name."
            )

            st.stop()

        if location.strip() == "":

            st.error(
                "Please enter the location."
            )

            st.stop()

        if description.strip() == "":

            st.error(
                "Please describe the problem."
            )

            st.stop()

        if uploaded_image is None:

            st.error(
                "Please upload an image."
            )

            st.stop()

        # ---------------------------------------------
        # READ IMAGE
        # ---------------------------------------------

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
                "Invalid image."
            )

            st.stop()

        # ---------------------------------------------
        # AI DETECTION
        # ---------------------------------------------

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

        # ---------------------------------------------
        # DEPARTMENT
        # ---------------------------------------------

        department = assign_department(
            detected_issue
        )

        # ---------------------------------------------
        # PRIORITY
        # ---------------------------------------------

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

        # ---------------------------------------------
        # COMPLAINT ID
        # ---------------------------------------------

        complaint_id = (
            generate_complaint_id()
        )

        submitted_at = (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        # ---------------------------------------------
        # COMPLAINT DATA
        # ---------------------------------------------

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

        # ---------------------------------------------
        # SAVE
        # ---------------------------------------------

        save_complaint(
            complaint_data
        )

        # ---------------------------------------------
        # RESULT
        # ---------------------------------------------

        st.success(
            "✅ Complaint submitted successfully!"
        )

        st.subheader(
            "🤖 AI Analysis"
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

        st.subheader(
            "🏢 Assigned Department"
        )

        st.info(
            department
        )

        st.subheader(
            "📊 Priority Score"
        )

        st.progress(
            int(priority_score)
        )

        st.write(
            f"Priority Score: "
            f"**{priority_score}/100**"
        )

        st.subheader(
            "🆔 Complaint ID"
        )

        st.code(
            complaint_id
        )

        st.warning(
            "Save this Complaint ID to track your complaint."
        )

        st.info(
            "Current Status: 🟡 Pending"
        )


# =========================================================
# TRACK COMPLAINT
# =========================================================

elif page == "🔎 Track Complaint":

    st.header(
        "🔎 Track Your Civic Complaint"
    )

    st.write(
        "Enter your Complaint ID."
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
        "Complaint ID",
        placeholder="CIVIC-2026-1234"
    )

    if st.button(
        "🔎 Track Complaint",
        type="primary"
    ):

        complaint_id = (
            complaint_id.strip()
        )

        if complaint_id == "":

            st.error(
                "Please enter Complaint ID."
            )

        else:

            result = data[
                data["Complaint_ID"]
                .astype(str)
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

                st.subheader(
                    "📋 Complaint Details"
                )

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
                    f"**Description:** "
                    f"{complaint['Description']}"
                )

                st.divider()

                st.subheader(
                    "📌 Current Status"
                )

                status = str(
                    complaint["Status"]
                )

                if status == "Pending":

                    st.warning(
                        "🟡 Pending – "
                        "Complaint received."
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

    st.header(
        "🏢 Department Dashboard"
    )

    st.write(
        "Authorized departments can view and update "
        "their assigned complaints."
    )

    # -----------------------------------------------------
    # CHECK DATABASE
    # -----------------------------------------------------

    if not os.path.exists(CSV_PATH):

        st.warning(
            "No complaints have been submitted yet."
        )

        st.stop()

    data = pd.read_csv(
        CSV_PATH
    )

    # -----------------------------------------------------
    # LOGIN
    # -----------------------------------------------------

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

    login_button = st.button(
        "🔐 Login",
        key="dept_login"
    )

    # -----------------------------------------------------
    # LOGIN CREDENTIALS
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # LOGIN PROCESS
    # -----------------------------------------------------

    if login_button:

        if username not in credentials:

            st.error(
                "❌ Invalid username."
            )

        elif password != credentials[username][0]:

            st.error(
                "❌ Incorrect password."
            )

        else:

            st.session_state[
                "department_logged_in"
            ] = True

            st.session_state[
                "department_name"
            ] = credentials[username][1]

            st.success(
                "✅ Login successful!"
            )

    # -----------------------------------------------------
    # DASHBOARD AFTER LOGIN
    # -----------------------------------------------------

    if st.session_state.get(
        "department_logged_in",
        False
    ):

        department = st.session_state[
            "department_name"
        ]

        st.divider()

        st.subheader(
            f"🏢 {department}"
        )

        # -------------------------------------------------
        # FILTER COMPLAINTS
        # -------------------------------------------------

        department_data = data[
            data["Department"].astype(str)
            ==
            department
        ].copy()

        # -------------------------------------------------
        # STATISTICS
        # -------------------------------------------------

        total = len(
            department_data
        )

        pending = len(
            department_data[
                department_data["Status"]
                .astype(str)
                .str.lower()
                ==
                "pending"
            ]
        )

        in_progress = len(
            department_data[
                department_data["Status"]
                .astype(str)
                .str.lower()
                ==
                "in progress"
            ]
        )

        resolved = len(
            department_data[
                department_data["Status"]
                .astype(str)
                .str.lower()
                ==
                "resolved"
            ]
        )

        col1, col2, col3, col4 = st.columns(4)

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

        # -------------------------------------------------
        # COMPLAINT LIST
        # -------------------------------------------------

        st.subheader(
            "📋 Assigned Complaints"
        )

        if department_data.empty:

            st.info(
                "No complaints are assigned "
                "to this department."
            )

        else:

            display_columns = [

                "Complaint_ID",
                "Citizen_Name",
                "Location",
                "Issue",
                "Priority",
                "Priority_Score",
                "Status"
            ]

            available_columns = [

                column
                for column in display_columns
                if column in department_data.columns
            ]

            st.dataframe(
                department_data[
                    available_columns
                ],
                use_container_width=True
            )

            st.divider()

            # -------------------------------------------------
            # UPDATE STATUS
            # -------------------------------------------------

            st.subheader(
                "🔄 Update Complaint Status"
            )

            complaint_ids = (
                department_data[
                    "Complaint_ID"
                ]
                .astype(str)
                .tolist()
            )

            selected_complaint = st.selectbox(
                "Select Complaint ID",
                complaint_ids,
                key="selected_complaint_id"
            )

            # Find selected complaint

            selected_row = department_data[
                department_data["Complaint_ID"]
                .astype(str)
                ==
                selected_complaint
            ]

            current_status = str(
                selected_row.iloc[0]["Status"]
            )

            st.write(
                f"**Current Status:** "
                f"{current_status}"
            )

            new_status = st.selectbox(
                "Change Status To",
                [
                    "Pending",
                    "In Progress",
                    "Resolved"
                ],
                key="change_status"
            )

            if st.button(
                "💾 Update Status",
                type="primary",
                key="save_status"
            ):

                # Update dataframe

                data.loc[
                    data["Complaint_ID"]
                    .astype(str)
                    ==
                    selected_complaint,
                    "Status"
                ] = new_status

                # Save CSV

                data.to_csv(
                    CSV_PATH,
                    index=False
                )

                st.success(
                    f"✅ {selected_complaint} "
                    f"status changed to "
                    f"'{new_status}'."
                )

                st.rerun()

        # -------------------------------------------------
        # LOGOUT
        # -------------------------------------------------

        st.divider()

        if st.button(
            "🚪 Logout",
            key="logout"
        ):

            st.session_state[
                "department_logged_in"
            ] = False

            st.session_state[
                "department_name"
            ] = ""

            st.rerun()


# =========================================================
# ABOUT PROJECT
# =========================================================

elif page == "ℹ️ About Project":

    st.header(
        "ℹ️ About the Project"
    )

    st.subheader(
        "AI-Based Civic Issue Detection and Prioritization"
    )

    st.write(
        """
        This project provides an AI-based platform for
        reporting and managing civic issues.

        Citizens can upload an image of a civic problem.
        The machine-learning model identifies the issue,
        calculates priority and assigns it to the relevant
        government department.

        Citizens can then track the complaint using a
        unique Complaint ID.
        """
    )

    st.divider()

    st.subheader(
        "🎯 Main Objectives"
    )

    st.write(
        """
        • Detect civic issues from images

        • Prioritize complaints

        • Automatically assign departments

        • Generate Complaint IDs

        • Store complaint information

        • Allow citizens to track complaints

        • Allow departments to update complaint status
        """
    )

    st.subheader(
        "🤖 Machine Learning"
    )

    if model is not None:

        st.success(
            "Random Forest image classification model loaded."
        )

        st.write(
            f"Model features: "
            f"{model.n_features_in_}"
        )

    else:

        st.error(
            "AI model not found."
        )

    st.subheader(
        "🛠️ Technologies Used"
    )

    st.write(
        """
        Python, Streamlit, OpenCV, Scikit-image,
        Scikit-learn, Pandas, NumPy and Joblib.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🏙️ AI Civic Issue Management System | "
    "AI + Machine Learning + Computer Vision"
)