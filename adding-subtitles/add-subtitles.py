import cv2
import os
import numpy as np
import json


vid_path = "./output-25-fps.mp4"

# Open the file in read mode
with open('transcript.json', 'r') as f:
    # Parse the JSON data
    data = json.load(f)

subtitles_li = []
subtitles = data['segments']
for item in subtitles: 
    start = item['start']*25
    end = item['end']*25
    text = item['text']
    subtitles_li.append([start, end, text])
    print(start, end, text)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output-with-subtitles.mp4', fourcc, 25.0, (1300, 800))


# Create a VideoCapture object and pass the name of the video file as an argument
cap = cv2.VideoCapture(vid_path)

# Check if the VideoCapture object is opened successfully
if cap.isOpened() == False:
    print("Error opening the video file")

# Define the text and the font
text = None
font = cv2.FONT_HERSHEY_SIMPLEX

# Loop over the frames of the video
counter = 0
li_counter = 0
while True:
    # Read a frame
    ret, frame = cap.read()

    # Wait for a key press
    key = cv2.waitKey(25)

    # Break the loop if the read() method returns False or the user presses the 'q' key
    if ret == False or key == ord('q'):
        break

    # resize 
    frame = cv2.resize(frame, (1300, 800))

    if((counter > subtitles_li[li_counter][1]) and (li_counter < len(subtitles_li)-1)):
        li_counter += 1
        print(li_counter, len(subtitles_li))

    if((counter > subtitles_li[li_counter][0]) and (counter < subtitles_li[li_counter][1])):
        text = subtitles_li[li_counter][2]
    else:
        text = None
    
    # Get the height and width of the frame
    height, width = frame.shape[:2]

    # Calculate the text size and position
    text_size = cv2.getTextSize(text, font, 0.7, 1)[0]
    # print(text_size, counter, "=== ", counter/25, text)

    text_x = (width - text_size[0]) // 2
    text_y = height - 50

    # Calculate the rectangle size and position
    rect_x = text_x - 20
    rect_y = text_y - 20
    rect_w = text_size[0] + 14
    rect_h = text_size[1] + 14

    if text is not None:
        # Draw a yellow filled rectangle behind the text
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 255, 255), -1)

        # Write the text on the frame
        cv2.putText(frame, text, (text_x, text_y), font, 0.7, (0, 0, 0), 1)


    # Display the frame
    cv2.imshow('Frame', frame)

    # Write the frame into the output file
    out.write(frame)

    counter += 1

# Release the VideoCapture object, save object and destroy all the windows
cap.release()
out.release()

cv2.destroyAllWindows()


# print(subtitles)

