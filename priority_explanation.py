# -----------------------------------------
# AI PRIORITY EXPLANATION
# -----------------------------------------

def explain_priority(issue, priority, description):

    issue = issue.lower()
    description = description.lower()

    reasons = []


    # -----------------------------------------
    # ISSUE-BASED REASONS
    # -----------------------------------------

    if issue == "pothole":

        reasons.append(
            "Potholes can affect road safety and vehicles."
        )

    elif issue == "traffic_signal":

        reasons.append(
            "Traffic signal problems can affect road traffic and safety."
        )

    elif issue == "water_leak":

        reasons.append(
            "Water leakage can cause water wastage and road damage."
        )

    elif issue == "drainage":

        reasons.append(
            "Drainage problems can cause water accumulation and flooding."
        )

    elif issue == "garbage":

        reasons.append(
            "Accumulated garbage can create sanitation and hygiene problems."
        )

    elif issue == "streetlight":

        reasons.append(
            "Streetlight problems can reduce visibility and public safety."
        )

    elif issue == "fallen_tree":

        reasons.append(
            "A fallen tree can block roads and create safety hazards."
        )

    else:

        reasons.append(
            "The reported civic issue requires attention."
        )


    # -----------------------------------------
    # DESCRIPTION-BASED REASONS
    # -----------------------------------------

    danger_words = [
        "danger",
        "dangerous",
        "accident",
        "risk",
        "blocked",
        "blocking",
        "emergency",
        "heavy traffic",
        "school",
        "hospital"
    ]


    for word in danger_words:

        if word in description:

            reasons.append(
                f"The complaint description mentions '{word}'."
            )

            break


    # -----------------------------------------
    # PRIORITY LEVEL
    # -----------------------------------------

    if priority.lower() == "critical":

        reasons.append(
            "The issue has been classified as critical and requires immediate attention."
        )

    elif priority.lower() == "high":

        reasons.append(
            "The issue has been classified as high priority and should be addressed soon."
        )

    elif priority.lower() == "medium":

        reasons.append(
            "The issue has been classified as medium priority."
        )

    else:

        reasons.append(
            "The issue has been classified as lower priority."
        )


    return reasons