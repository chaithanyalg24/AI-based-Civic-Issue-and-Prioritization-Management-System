def calculate_priority(issue, description):

    description = description.lower()
    issue = issue.lower()

    score = 0

    # Issue-based severity
    if issue in ["pothole", "traffic_signal", "water_leak"]:
        score += 3

    elif issue in ["drainage", "fallen_tree"]:
        score += 2

    elif issue in ["garbage", "streetlight"]:
        score += 1


    # Safety-related words
    safety_words = [
        "accident",
        "danger",
        "dangerous",
        "injury",
        "blocked",
        "emergency",
        "risk"
    ]

    for word in safety_words:

        if word in description:
            score += 2


    # Convert score to priority
    if score >= 6:
        priority = "Critical"

    elif score >= 4:
        priority = "High"

    elif score >= 2:
        priority = "Medium"

    else:
        priority = "Low"


    return priority, score