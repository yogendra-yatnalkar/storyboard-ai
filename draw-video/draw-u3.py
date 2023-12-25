import cv2
import numpy as np
import math
import os

def euc_dist(arr1, point):
    square_sub = (arr1 - point)**2
    return np.sqrt(np.sum(square_sub, axis=1))

def get_extreme_coordinates(mask):
    indices = np.where(mask == 255)
    # Extract the x and y coordinates of the pixels.
    x = indices[1]
    y = indices[0]

    # Find the minimum and maximum x and y coordinates.
    topleft = (np.min(x), np.min(y))
    bottomright = (np.max(x), np.max(y))

    return topleft, bottomright

def draw_hand_on_img(drawing, hand, drawing_coord_x, drawing_coord_y, hand_mask_inv, hand_ht, hand_wd, img_ht, img_wd):    
    remaining_ht = img_ht - drawing_coord_y
    remaining_wd = img_wd - drawing_coord_x
    if(remaining_ht > hand_ht):
        crop_hand_ht = hand_ht
    else:
        crop_hand_ht = remaining_ht

    if(remaining_wd > hand_wd):
        crop_hand_wd = hand_wd
    else:
        crop_hand_wd = remaining_wd

    hand_cropped = hand[:crop_hand_ht, :crop_hand_wd]
    hand_mask_inv_cropped = hand_mask_inv[:crop_hand_ht, :crop_hand_wd]

    drawing[
        drawing_coord_y:drawing_coord_y+crop_hand_ht, 
        drawing_coord_x:drawing_coord_x+crop_hand_wd
        ][:,:,0] = drawing[
            drawing_coord_y:drawing_coord_y+crop_hand_ht, 
            drawing_coord_x:drawing_coord_x+crop_hand_wd
            ][:,:,0]*hand_mask_inv_cropped
    drawing[
        drawing_coord_y:drawing_coord_y+crop_hand_ht, 
        drawing_coord_x:drawing_coord_x+crop_hand_wd
        ][:,:,1] = drawing[
            drawing_coord_y:drawing_coord_y+crop_hand_ht, 
            drawing_coord_x:drawing_coord_x+crop_hand_wd
            ][:,:,1]*hand_mask_inv_cropped
    drawing[
        drawing_coord_y:drawing_coord_y+crop_hand_ht, 
        drawing_coord_x:drawing_coord_x+crop_hand_wd
        ][:,:,2] = drawing[
            drawing_coord_y:drawing_coord_y+crop_hand_ht, 
            drawing_coord_x:drawing_coord_x+crop_hand_wd
            ][:,:,2]*hand_mask_inv_cropped

    drawing[
        drawing_coord_y:drawing_coord_y+crop_hand_ht, 
        drawing_coord_x:drawing_coord_x+crop_hand_wd
        ] = drawing[
        drawing_coord_y:drawing_coord_y+crop_hand_ht, 
        drawing_coord_x:drawing_coord_x+crop_hand_wd
        ]+hand_cropped
    return drawing

# constants
img_path = "./images/1.png"
hand_path = "./images/drawing-hand.png"
hand_mask_path = "./images/hand-mask.png"
frame_rate = 25
resize_wd, resize_ht = 800, 800
split_len = 10

# video save path 
img_name = img_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
video_save_name = img_name + ".mp4"
save_video_path = os.path.join("./save_videos", video_save_name)
print("save_video_path: ", save_video_path)

hand = cv2.imread(hand_path)
# alpha_hand = cv2.imread("./images/alpha-hand.png")
hand_mask = cv2.imread(hand_mask_path,cv2.IMREAD_GRAYSCALE)
img = cv2.imread(img_path)

top_left, bottom_right = get_extreme_coordinates(hand_mask)
hand = hand[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
hand_mask = hand_mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
hand_mask_inv = 255 - hand_mask

# standardizing the hand masks
hand_mask = hand_mask/255
hand_mask_inv = hand_mask_inv/255

# making the hand background black
hand_bg_ind = np.where(hand_mask == 0)
hand[hand_bg_ind] = [0, 0, 0]

img = cv2.resize(img, (resize_wd, resize_ht))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)

# finding all points where pixel is black
black_indices = np.array(np.where(img_gray == 0)).T
print("Length of black indices: ", len(black_indices))

# getting the img and hand dim
hand_ht, hand_wd = hand.shape[0], hand.shape[1]
img_ht, img_wd = img.shape[0], img.shape[1]

# adding the marker at right position
# split_mid_h, split_mid_v = 500, 600
# drawing = draw_hand_on_img(img, hand, split_mid_h, split_mid_v, hand_mask_inv, hand_ht, hand_wd, img_ht, img_wd)


# # draw
video = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (img.shape[1], img.shape[0]))

# creating an emtpy frame and select 0th index as the starting point to draw
drawn_frame = np.zeros(img.shape, np.uint8) + np.array([255, 255, 255], np.uint8)
selected_ind = 0
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

    hand_coord_x = range_h_start + int(split_len/2)
    hand_coord_y = range_v_start + int(split_len/2)
    drawn_frame_with_hand = draw_hand_on_img(drawn_frame.copy(), hand.copy(), hand_coord_x, hand_coord_y, hand_mask_inv.copy(), hand_ht, hand_wd, img_ht, img_wd)

    # delete the selected ind from the d_array
    d[selected_ind] = d[-1]
    d = d[:-1]

    del selected_ind

    # select the next new index
    euc_arr = euc_dist(d, selected_ind_val)
    selected_ind = np.argmin(euc_arr)
    # print("EUC array: ", euc_arr[:10])
    # print("-"*10)

    # if(counter == 1):
    #     break
    

    counter += 1

    if(counter%5 == 0):
        video.write(drawn_frame_with_hand)
        print("counter: ", counter)
        print("Selected Index: ", selected_ind)
        print("len of black indices: ", len(d))

for i in range(frame_rate):
    video.write(img)

video.release()
cv2.imshow("img", img)
cv2.imshow("img_gray", img_gray)
cv2.imshow("hand", hand)
cv2.imshow("hand_mask", hand_mask)
cv2.imshow("hand_mask_inv", hand_mask_inv)
# cv2.imshow("drawing", drawing)

cv2.waitKey(0)
cv2.destroyAllWindows()