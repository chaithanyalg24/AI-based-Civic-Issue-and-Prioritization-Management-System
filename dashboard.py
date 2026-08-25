import streamlit as st
import pandas as pd
import os


# -----------------------------------------
# PAGE SETTINGS
# -----------------------------------------

st.set_page_config(
    page_title="Civic Department Dashboard",
    page_icon="🏢",
    layout="wide"
)


# -----------------------------------------
# DEPARTMENT LOGIN DETAILS
# -----------------------------------------

DEPARTMENT_USERS = {
    "road_admin": {
        "password": "road123",
        "department": "Road Department"
    },

    "sanitation_admin": {
        "password": "sanitation123",
        "department": "Sanitation Department"
    },

    "electrical_admin": {
        "password": "electrical123",
        "department": "Electrical Department"
    },

    "water_admin": {
        "password": "water123",
        "department": "Water Supply Department"
    },

    "drainage_admin": {
        "password": "drainage123",
        "department": "Drainage Department"
    },

    "parks_admin": {
        "password": "parks123",
        "department": "Parks Department"
    },

    "traffic_admin": {
        "password": "traffic123",
        "department": "Traffic Police"
    }
}


# -----------------------------------------
# LOGIN FUNCTION
# -----------------------------------------

def login_page():

    st.title(
        "🏢 Civic Department Login"
    )

    st.write(
        "Login to access the department complaint dashboard."
    )

    username = st.text_input(
        "Username"
    )

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button(
        "🔐 Login"
    ):

        if username in DEPARTMENT_USERS:

            if DEPARTMENT_USERS[username]["password"] == password:

                st.session_state.logged_in = True

                st.session_state.department = (
                    DEPARTMENT_USERS[username]["department"]
                )

                st.rerun()

            else:

                st.error(
                    "❌ Incorrect password."
                )

        else:

            st.error(
                "❌ Username not found."
            )


# -----------------------------------------
# INITIALIZE SESSION
# -----------------------------------------

if "logged_in" not in st.session_state:

    st.session_state.logged_in = False


if "department" not in st.session_state:

    st.session_state.department = ""


# -----------------------------------------
# SHOW LOGIN
# -----------------------------------------

if not st.session_state.logged_in:

    login_page()

    st.stop()


# -----------------------------------------
# LOGOUT
# -----------------------------------------

with st.sidebar:

    st.success(
        "✅ Logged in"
    )

    st.write(
        f"🏢 {st.session_state.department}"
    )

    if st.button(
        "Logout"
    ):

        st.session_state.logged_in = False

        st.session_state.department = ""

        st.rerun()


# -----------------------------------------
# TITLE
# -----------------------------------------

st.title(
    "🏢 Civic Issue Department Dashboard"
)

st.write(
    f"Welcome to the {st.session_state.department}"
)


# -----------------------------------------
# CHECK DATABASE
# -----------------------------------------

if not os.path.exists("complaints.csv"):

    st.warning(
        "No complaints have been submitted yet."
    )

    st.stop()


# -----------------------------------------
# LOAD DATABASE
# -----------------------------------------

data = pd.read_csv(
    "complaints.csv"
)


# -----------------------------------------
# FILTER DEPARTMENT
# -----------------------------------------

department = st.session_state.department

department_data = data[
    data["Department"] == department
].copy()


# -----------------------------------------
# DASHBOARD SUMMARY
# -----------------------------------------

st.subheader(
    "📊 Department Overview"
)

total_complaints = len(
    department_data
)

pending_count = len(
    department_data[
        department_data["Status"] == "Pending"
    ]
)

in_progress_count = len(
    department_data[
        department_data["Status"] == "In Progress"
    ]
)

resolved_count = len(
    department_data[
        department_data["Status"] == "Resolved"
    ]
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total",
        total_complaints
    )


with col2:

    st.metric(
        "Pending",
        pending_count
    )


with col3:

    st.metric(
        "In Progress",
        in_progress_count
    )


with col4:

    st.metric(
        "Resolved",
        resolved_count
    )


# -----------------------------------------
# SHOW COMPLAINTS
# -----------------------------------------

st.subheader(
    "📋 Assigned Complaints"
)


if department_data.empty:

    st.info(
        "🎉 No complaints are currently assigned to your department."
    )

else:

    display_columns = [
        "Complaint_ID",
        "Citizen_Name",
        "Location",
        "Description",
        "Issue",
        "Priority",
        "Priority_Score",
        "Status"
    ]

    st.dataframe(
        department_data[
            display_columns
        ],
        use_container_width=True
    )


# -----------------------------------------
# UPDATE STATUS
# -----------------------------------------

st.subheader(
    "🔄 Update Complaint Status"
)


if not department_data.empty:

    complaint_ids = department_data[
        "Complaint_ID"
    ].tolist()


    selected_complaint = st.selectbox(
        "Select Complaint",
        complaint_ids
    )


    new_status = st.selectbox(
        "Select New Status",
        [
            "Pending",
            "In Progress",
            "Resolved"
        ]
    )


    if st.button(
        "✅ Update Status"
    ):

        data.loc[
            data["Complaint_ID"] == selected_complaint,
            "Status"
        ] = new_status


        data.to_csv(
            "complaints.csv",
            index=False
        )


        st.success(
            f"Complaint {selected_complaint} "
            f"updated to {new_status}."
        )


        st.rerun()