import os

folders = [
    "pothole",
    "garbage",
    "streetlight",
    "water_leak",
    "drainage",
    "fallen_tree",
    "traffic_signal"
]

for folder in folders:

    path = os.path.join("images", folder)

    files = os.listdir(path)

    print("\n", folder)
    print("Number of images:", len(files))
    print("Images:", files)