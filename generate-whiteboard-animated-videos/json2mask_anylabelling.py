# Import libraries
import json
import numpy as np
import cv2


if __name__ == "__main__":
    # Load the json file
    with open("./images/4.json", "r") as f:
        data = json.load(f)

    # Get the image size
    height = 1024
    width = 1024

    # Create an empty mask array
    mask = np.zeros((height, width), dtype=np.uint8)

    # Loop through the shapes in the json file
    for shape in data["shapes"]:
        # Create an empty mask array
        mask = np.zeros((height, width), dtype=np.uint8)

        # Get the shape points as a list of tuples
        points = np.array(shape['points'], dtype = np.int32)
        points = np.expand_dims(points, axis = 0)
        
        # Fill the polygon with white color (255) on the mask array using cv2
        cv2.fillPoly(mask, points, 255)

    # # Save the mask array as a png file using cv2
    # cv2.imwrite("mask.png", mask)
        
        cv2.imshow("mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()