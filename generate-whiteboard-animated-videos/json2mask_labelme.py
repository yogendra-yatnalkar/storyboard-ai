# Import libraries
import json
import numpy as np
import cv2

# Load the json file
with open("./images/drawing-hand.json", "r") as f:
    data = json.load(f)

# Get the image size
height = data["imageHeight"]
width = data["imageWidth"]

# Create an empty mask array
mask = np.zeros((height, width), dtype=np.uint8)

# Loop through the shapes in the json file       
for shape in data["shapes"]:
    # Check if the shape label matches the desired label
    if shape["label"] == "0":
        # Get the shape points as a list of tuples
        points = [(int(p[0]), int(p[1])) for p in shape["points"]]
        # Fill the polygon with white color (255) on the mask array using cv2
        cv2.fillPoly(mask, np.array([points]), 255)

# Save the mask array as a png file using cv2
cv2.imwrite("mask.png", mask)