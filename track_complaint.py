import streamlit as st
import pandas as pd
import os


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Track Civic Complaint",
    page_icon="🔎",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🔎 Track Your Civic Complaint")

st.write(
    "Enter your Complaint ID to check the current status."
)


# =========================================================
# CHECK DATABASE
# =========================================================

if not os.path.exists("complaints.csv"):

    st.warning(
        "No complaints have been submitted yet."
    )

    st.stop()


# =========================================================
# LOAD DATABASE
# =========================================================

try:

    data = pd.read_csv(
        "complaints.csv"
    )

except Exception as e:

    st.error(
        "Unable to read complaints database."
    )

    st.code(
        str(e)
    )

    st.stop()


# =========================================================
# COMPLAINT ID INPUT
# =========================================================

complaint_id = st.text_input(
    "Enter Complaint ID",
    placeholder="Example: CIVIC-2026-5832"
)


# =========================================================
# SEARCH
# =========================================================

if st.button(
    "🔎 Track Complaint",
    type="primary"
):

    complaint_id = complaint_id.strip()


    if complaint_id == "":

        st.error(
            "❌ Please enter a Complaint ID."
        )

        st.stop()


    # Search complaint
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


        # =================================================
        # COMPLAINT FOUND
        # =================================================

        st.success(
            "✅ Complaint found!"
        )


        # =================================================
        # BASIC DETAILS
        # =================================================

        st.header(
            "📋 Complaint Details"
        )


        col1, col2 = st.columns(2)


        with col1:

            st.write(
                f"**Complaint ID:** "
                f"{complaint['Complaint_ID']}"
            )

            st.write(
                f"**Citizen Name:** "
                f"{complaint['Citizen_Name']}"
            )

            st.write(
                f"**Location:** "
                f"{complaint['Location']}"
            )


        with col2:

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


        # =================================================
        # DESCRIPTION
        # =================================================

        st.subheader(
            "📝 Problem Description"
        )

        st.write(
            complaint["Description"]
        )


        # =================================================
        # AI INFORMATION
        # =================================================

        st.subheader(
            "🤖 AI Analysis"
        )


        col1, col2 = st.columns(2)


        with col1:

            if "AI_Confidence" in complaint:

                st.metric(
                    "AI Confidence",
                    f"{complaint['AI_Confidence']}%"
                )


        with col2:

            if "Priority_Score" in complaint:

                st.metric(
                    "Priority Score",
                    complaint["Priority_Score"]
                )


        # =================================================
        # STATUS
        # =================================================

        st.subheader(
            "📌 Current Status"
        )


        status = str(
            complaint["Status"]
        )


        if status.lower() == "pending":

            st.warning(
                "🟡 Pending – "
                "Your complaint has been received."
            )


        elif status.lower() == "in progress":

            st.info(
                "🔵 In Progress – "
                "The department is working on your complaint."
            )


        elif status.lower() == "resolved":

            st.success(
                "🟢 Resolved – "
                "Your civic issue has been resolved."
            )


        else:

            st.write(
                f"Status: {status}"
            )


        # =================================================
        # SUBMITTED TIME
        # =================================================

        if "Submitted_At" in complaint:

            st.write(
                f"**Submitted At:** "
                f"{complaint['Submitted_At']}"
            )