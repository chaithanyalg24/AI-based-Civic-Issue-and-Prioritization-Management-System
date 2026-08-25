import csv
import os


# -----------------------------------------
# DATABASE FILE
# -----------------------------------------

DATABASE_FILE = "complaints.csv"


# -----------------------------------------
# CREATE DATABASE IF NOT EXISTS
# -----------------------------------------

def create_database():

    if not os.path.exists(DATABASE_FILE):

        with open(
            DATABASE_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "Complaint_ID",
                "Citizen_Name",
                "Location",
                "Description",
                "Issue",
                "Department",
                "Priority",
                "Priority_Score",
                "Image",
                "Status"
            ])


# -----------------------------------------
# SAVE COMPLAINT
# -----------------------------------------

def save_complaint(
    complaint_id,
    citizen_name,
    location,
    description,
    issue,
    department,
    priority,
    priority_score,
    image
):

    create_database()

    with open(
        DATABASE_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            complaint_id,
            citizen_name,
            location,
            description,
            issue,
            department,
            priority,
            priority_score,
            image,
            "Pending"
        ])