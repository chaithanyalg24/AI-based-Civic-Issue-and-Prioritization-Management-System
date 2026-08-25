# -----------------------------------------
# DEPARTMENT ASSIGNMENT
# -----------------------------------------

DEPARTMENT_MAP = {

    "pothole": "Road Department",

    "garbage": "Waste Management Department",

    "streetlight": "Electrical Department",

    "water_leak": "Water Supply Department",

    "drainage": "Drainage Department",

    "fallen_tree": "Parks and Emergency Services",

    "traffic_signal": "Traffic Management Department"
}


def assign_department(issue):

    issue = issue.lower().strip()

    department = DEPARTMENT_MAP.get(
        issue,
        "General Civic Department"
    )

    return department