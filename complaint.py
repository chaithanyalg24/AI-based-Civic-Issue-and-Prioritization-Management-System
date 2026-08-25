import random


def generate_complaint_id():

    number = random.randint(
        1000,
        9999
    )

    return f"CIVIC-2026-{number}"