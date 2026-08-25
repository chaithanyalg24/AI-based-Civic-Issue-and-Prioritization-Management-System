import cv2
import numpy as np


def analyze_image(image_path):

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        return {
            "status": "error",
            "message": "Unable to read image"
        }

    # Get image dimensions
    height, width = image.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate average brightness
    brightness = np.mean(gray)

    return {
        "status": "success",
        "width": width,
        "height": height,
        "brightness": round(float(brightness), 2)
    }