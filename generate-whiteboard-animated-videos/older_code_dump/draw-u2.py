import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

def euc_dist(arr1, point):
    square_sub = (arr1 - point)**2
    return np.sqrt(np.sum(square_sub, axis=1))


img = cv2.imread("./images/storyboard.png")
# img = cv2.imread("yogendra.jpg")
# img = cv2.imread("nar-ruta.jpg")
frame_rate = 25

img = cv2.resize(img, (800, 800))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
# img_gray = cv2.Canny(img_gray, 10, 80)

# finding all points where pixel is black
black_indices = np.array(np.where(img_gray == 0)).T
print("Length of black indices: ", len(black_indices))

# # draw
video = cv2.VideoWriter("./save_videos/video-u2-storyboard.mp4", cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (img.shape[1], img.shape[0]))

# creating an emtpy frame and select 0th index as the starting point to draw
drawn_frame = np.zeros(img.shape, np.uint8) + np.array([255, 255, 255], np.uint8)
selected_ind = 0

split_len = 10
n_cuts_vertical = int(math.ceil(img.shape[0]/split_len))
n_cuts_horizontal = int(math.ceil(img.shape[1]/split_len))

cut = np.array(np.split(img_gray, n_cuts_horizontal, axis=-1))
cut = np.array(np.split(cut, n_cuts_vertical, axis=-2))
print(cut.shape)

b = (cut < 10)*1
c = np.sum(np.sum(b, axis = -1), axis = -1)
d = np.array(np.where(c > 0)).T


counter = 0
while(len(d) > 1):
    selected_ind_val = d[selected_ind].copy()
    range_v_start = selected_ind_val[0]*split_len
    range_v_end = range_v_start + split_len
    range_h_start = selected_ind_val[1]*split_len
    range_h_end = range_h_start + split_len

    temp_drawing = np.zeros((split_len, split_len, 3))
    temp_drawing[:, :, 0] = cut[selected_ind_val[0]][selected_ind_val[1]]
    temp_drawing[:, :, 1] = cut[selected_ind_val[0]][selected_ind_val[1]]
    temp_drawing[:, :, 2] = cut[selected_ind_val[0]][selected_ind_val[1]]

    drawn_frame[range_v_start:range_v_end, range_h_start:range_h_end] = temp_drawing

    # print("-"*10)
    # print("counter: ", counter)
    # print("Selected Index: ", selected_ind)
    # print("selected in val: ", selected_ind_val)
    # print("len of black indices: ", len(d))
    # print(d[:10])

    # delete the selected in from the d_array
    d[selected_ind] = d[-1]
    d = d[:-1]
    
    # print("selected in val: ", selected_ind_val)
    # print("len of black indices: ", len(d))
    # print(d[:10])

    del selected_ind

    # select the next new index
    euc_arr = euc_dist(d, selected_ind_val)
    selected_ind = np.argmin(euc_arr)
    # print("EUC array: ", euc_arr[:10])
    # print("-"*10)

    if(counter%5 == 0):
        video.write(drawn_frame)
        print("counter: ", counter)
        print("Selected Index: ", selected_ind)
        print("len of black indices: ", len(d))

    # if(counter == 1):
    #     break
    counter += 1

for i in range(frame_rate):
    video.write(img)

video.release()
cv2.imshow("img", img)
cv2.imshow("img_gray", img_gray)

cv2.waitKey(0)
cv2.destroyAllWindows()
