import streamlit as st
import pandas as pd
import os


# -----------------------------------------
# PAGE SETTINGS
# -----------------------------------------

st.set_page_config(
    page_title="Track Civic Complaint",
    page_icon="🔎",
    layout="wide"
)


# -----------------------------------------
# TITLE
# -----------------------------------------

st.title(
    "🔎 Track Your Civic Complaint"
)

st.write(
    "Enter your Complaint ID to check the latest status of your civic complaint."
)


# -----------------------------------------
# CHECK DATABASE
# -----------------------------------------

if not os.path.exists("complaints.csv"):

    st.warning(
        "⚠️ No complaints have been submitted yet."
    )

    st.stop()


# -----------------------------------------
# LOAD DATABASE
# -----------------------------------------

data = pd.read_csv(
    "complaints.csv"
)


# -----------------------------------------
# COMPLAINT ID INPUT
# -----------------------------------------

st.subheader(
    "🆔 Enter Complaint ID"
)

complaint_id = st.text_input(
    "Complaint ID",
    placeholder="Example: CIVIC-2026-5832"
)


# -----------------------------------------
# SEARCH
# -----------------------------------------

if st.button(
    "🔎 Track Complaint"
):

    complaint_id = complaint_id.strip()


    if complaint_id == "":

        st.error(
            "❌ Please enter a Complaint ID."
        )

    else:

        result = data[
            data["Complaint_ID"]
            .astype(str)
            .str.upper()
            == complaint_id.upper()
        ]


        # -----------------------------------------
        # COMPLAINT NOT FOUND
        # -----------------------------------------

        if result.empty:

            st.error(
                "❌ Complaint ID not found."
            )


        # -----------------------------------------
        # COMPLAINT FOUND
        # -----------------------------------------

        else:

            complaint = result.iloc[0]


            st.success(
                "✅ Complaint found successfully!"
            )


            # -----------------------------------------
            # BASIC DETAILS
            # -----------------------------------------

            st.subheader(
                "📋 Complaint Details"
            )


            col1, col2 = st.columns(2)


            with col1:

                st.write(
                    f"**🆔 Complaint ID:** "
                    f"{complaint['Complaint_ID']}"
                )

                st.write(
                    f"**👤 Citizen:** "
                    f"{complaint['Citizen_Name']}"
                )

                st.write(
                    f"**📍 Location:** "
                    f"{complaint['Location']}"
                )

                st.write(
                    f"**🏢 Department:** "
                    f"{complaint['Department']}"
                )


            with col2:

                issue = str(
                    complaint["Issue"]
                ).replace(
                    "_",
                    " "
                ).title()


                st.write(
                    f"**🚨 Issue:** {issue}"
                )

                st.write(
                    f"**⚠️ Priority:** "
                    f"{complaint['Priority']}"
                )

                st.write(
                    f"**📊 Priority Score:** "
                    f"{complaint['Priority_Score']}"
                )


            # -----------------------------------------
            # DESCRIPTION
            # -----------------------------------------

            st.subheader(
                "📝 Complaint Description"
            )


            st.info(
                str(
                    complaint["Description"]
                )
            )


            # -----------------------------------------
            # AI CONFIDENCE
            # -----------------------------------------

            if "Confidence" in data.columns:

                st.subheader(
                    "🤖 AI Analysis"
                )


                confidence = complaint["Confidence"]


                try:

                    confidence_value = float(
                        confidence
                    )


                    st.write(
                        f"AI Confidence: "
                        f"**{confidence_value:.2f}%**"
                    )


                    st.progress(
                        min(
                            int(confidence_value),
                            100
                        )
                    )

                except:

                    st.write(
                        f"AI Confidence: {confidence}"
                    )


            # -----------------------------------------
            # STATUS
            # -----------------------------------------

            st.subheader(
                "📌 Complaint Status"
            )


            status = str(
                complaint["Status"]
            )


            # -----------------------------------------
            # STATUS DISPLAY
            # -----------------------------------------

            if status == "Pending":

                st.warning(
                    "🟡 PENDING"
                )

                st.write(
                    "Your complaint has been received "
                    "and is waiting for department action."
                )


            elif status == "In Progress":

                st.info(
                    "🔵 IN PROGRESS"
                )

                st.write(
                    "The responsible department is "
                    "currently working on your complaint."
                )


            elif status == "Resolved":

                st.success(
                    "🟢 RESOLVED"
                )

                st.write(
                    "Your civic issue has been resolved."
                )


            else:

                st.write(
                    f"Current Status: {status}"
                )


            # -----------------------------------------
            # STATUS PROGRESS
            # -----------------------------------------

            st.subheader(
                "📈 Complaint Progress"
            )


            if status == "Pending":

                st.write(
                    "🟡 1. Complaint Submitted"
                )

                st.write(
                    "⚪ 2. Department Working"
                )

                st.write(
                    "⚪ 3. Issue Resolved"
                )


            elif status == "In Progress":

                st.write(
                    "🟢 1. Complaint Submitted"
                )

                st.write(
                    "🔵 2. Department Working"
                )

                st.write(
                    "⚪ 3. Issue Resolved"
                )


            elif status == "Resolved":

                st.write(
                    "🟢 1. Complaint Submitted"
                )

                st.write(
                    "🟢 2. Department Working"
                )

                st.write(
                    "🟢 3. Issue Resolved"
                )


            # -----------------------------------------
            # FINAL MESSAGE
            # -----------------------------------------

            st.divider()


            if status == "Resolved":

                st.success(
                    "🎉 Thank you for helping improve "
                    "your community!"
                )

            else:

                st.info(
                    "💡 You can check this page again "
                    "later using the same Complaint ID "
                    "to see the updated status."
                )