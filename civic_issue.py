import pandas as pd
import os
import joblib
from datetime import datetime
from PIL import Image
# -----------------------------------------
# IMAGE UPLOAD
# -----------------------------------------


def save_image():

    image_path = input(
        "\nEnter the path of the civic issue image: "
    ).strip().strip('"')

    if not os.path.exists(image_path):
        print("Image path not found!")
        return None

    try:
        image = Image.open(image_path)

        # Load the image to check that it is valid
        image.load()

        # Create uploads folder if it doesn't exist
        os.makedirs("uploads", exist_ok=True)

        file_name = os.path.basename(image_path)

        destination = os.path.join(
            "uploads",
            file_name
        )

        image.save(destination)

        print("Image uploaded successfully!")
        print("Saved as:", destination)

        return destination

    except Exception as error:

        print("Could not open the image.")
        print("Error:", error)

        return None
# Load trained NLP model


text_model = joblib.load("civic_text_model.pkl")

# Load trained priority model
priority_model = joblib.load("civic_priority_model.pkl")

print("AI models loaded successfully!")


# Load trained ML model
model = joblib.load("civic_priority_model.pkl")

print("ML model loaded successfully!")


# -----------------------------------------
# 1. DETECT CIVIC ISSUE
# -----------------------------------------

# -----------------------------------------
# 1. AI CIVIC ISSUE DETECTION
# -----------------------------------------

def detect_issue(description):

    prediction = text_model.predict([description])

    return prediction[0]


# -----------------------------------------
# 2. CALCULATE SEVERITY
# -----------------------------------------

def calculate_severity(description):

    text = description.lower()

    severity = 5

    high_words = [
        "dangerous",
        "accident",
        "blocked",
        "huge",
        "large",
        "emergency",
        "risk"
    ]

    low_words = [
        "small",
        "minor"
    ]

    for word in high_words:
        if word in text:
            severity += 1

    for word in low_words:
        if word in text:
            severity -= 1

    # Keep severity between 1 and 10
    severity = max(1, min(severity, 10))

    return severity


# -----------------------------------------
# 3. CALCULATE PRIORITY
# -----------------------------------------
def predict_priority(severity, traffic, complaints, safety_risk):

    new_issue = pd.DataFrame({
        "Severity": [severity],
        "Traffic_Level": [traffic],
        "Similar_Complaints": [complaints],
        "Safety_Risk": [safety_risk]
    })

    prediction = model.predict(new_issue)[0]

    prediction = max(0, min(prediction, 100))

    return round(prediction, 2)


# -----------------------------------------
# 4. DETERMINE PRIORITY LEVEL
# -----------------------------------------

def priority_level(priority):

    if priority >= 80:
        return "CRITICAL"

    elif priority >= 60:
        return "HIGH"

    elif priority >= 40:
        return "MEDIUM"

    else:
        return "LOW"


# -----------------------------------------
# 5. SAVE COMPLAINT
# -----------------------------------------

def save_complaint(data):

    file_name = "civic_complaints.csv"

    new_data = pd.DataFrame([data])

    if os.path.exists(file_name):

        old_data = pd.read_csv(file_name)

        final_data = pd.concat(
            [old_data, new_data],
            ignore_index=True
        )

    else:

        final_data = new_data

    final_data.to_csv(
        file_name,
        index=False
    )

    print("\nComplaint saved successfully!")


# -----------------------------------------
# 6. MAIN PROGRAM
# -----------------------------------------

print("=" * 50)
print(" AI CIVIC ISSUE DETECTION & PRIORITIZATION")
print("=" * 50)

name = input("\nEnter citizen name: ")

description = input(
    "Describe the civic problem: "
)

location = input(
    "Enter location: "
)
image_path = save_image()

traffic = int(input(
    "Enter traffic level (1-10): "
))

complaints = int(input(
    "Number of similar complaints: "
))
safety_risk = int(input(
    "Enter safety risk level (1-10): "
))


# Detect issue
issue = detect_issue(description)


# Calculate severity
severity = calculate_severity(description)


# Calculate priority
priority = predict_priority(
    severity,
    traffic,
    complaints,
    safety_risk
)

# Determine priority level
level = priority_level(priority)


# Current date and time
date_time = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"
)


# -----------------------------------------
# 7. DISPLAY RESULT
# -----------------------------------------

# -----------------------------------------
# 7. DISPLAY RESULT
# -----------------------------------------

print("\n" + "=" * 50)
print("              AI ANALYSIS")
print("=" * 50)

print("Citizen Name :", name)
print("Issue        :", issue)
print("Location     :", location)
print("Image        :", image_path)
print("Severity     :", severity, "/ 10")
print("Traffic      :", traffic, "/ 10")
print("Complaints   :", complaints)
print("Safety Risk  :", safety_risk, "/ 10")
print("Priority     :", priority, "/ 100")
print("Priority     :", level)
print("Status       : Pending")
print("Date         :", date_time)

print("=" * 50)


# -----------------------------------------
# 8. STORE DATA
# -----------------------------------------

complaint_data = {

    "Citizen_Name": name,

    "Description": description,

    "Issue_Type": issue,

    "Location": location,

    "Image_Path": image_path,

    "Severity": severity,

    "Traffic_Level": traffic,

    "Similar_Complaints": complaints,

    "Safety_Risk": safety_risk,

    "Priority_Score": priority,

    "Priority_Level": level,

    "Status": "Pending",

    "Date": date_time
}


save_complaint(complaint_data)
