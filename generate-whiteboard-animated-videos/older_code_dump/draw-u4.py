import cv2
import time
import numpy as np
import math
import os


def euc_dist(arr1, point):
    square_sub = (arr1 - point) ** 2
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


def draw_hand_on_img(
    drawing,
    hand,
    drawing_coord_x,
    drawing_coord_y,
    hand_mask_inv,
    hand_ht,
    hand_wd,
    img_ht,
    img_wd,
):
    remaining_ht = img_ht - drawing_coord_y
    remaining_wd = img_wd - drawing_coord_x
    if remaining_ht > hand_ht:
        crop_hand_ht = hand_ht
    else:
        crop_hand_ht = remaining_ht

    if remaining_wd > hand_wd:
        crop_hand_wd = hand_wd
    else:
        crop_hand_wd = remaining_wd

    hand_cropped = hand[:crop_hand_ht, :crop_hand_wd]
    hand_mask_inv_cropped = hand_mask_inv[:crop_hand_ht, :crop_hand_wd]

    drawing[
        drawing_coord_y : drawing_coord_y + crop_hand_ht,
        drawing_coord_x : drawing_coord_x + crop_hand_wd,
    ][:, :, 0] = (
        drawing[
            drawing_coord_y : drawing_coord_y + crop_hand_ht,
            drawing_coord_x : drawing_coord_x + crop_hand_wd,
        ][:, :, 0]
        * hand_mask_inv_cropped
    )
    drawing[
        drawing_coord_y : drawing_coord_y + crop_hand_ht,
        drawing_coord_x : drawing_coord_x + crop_hand_wd,
    ][:, :, 1] = (
        drawing[
            drawing_coord_y : drawing_coord_y + crop_hand_ht,
            drawing_coord_x : drawing_coord_x + crop_hand_wd,
        ][:, :, 1]
        * hand_mask_inv_cropped
    )
    drawing[
        drawing_coord_y : drawing_coord_y + crop_hand_ht,
        drawing_coord_x : drawing_coord_x + crop_hand_wd,
    ][:, :, 2] = (
        drawing[
            drawing_coord_y : drawing_coord_y + crop_hand_ht,
            drawing_coord_x : drawing_coord_x + crop_hand_wd,
        ][:, :, 2]
        * hand_mask_inv_cropped
    )

    drawing[
        drawing_coord_y : drawing_coord_y + crop_hand_ht,
        drawing_coord_x : drawing_coord_x + crop_hand_wd,
    ] = (
        drawing[
            drawing_coord_y : drawing_coord_y + crop_hand_ht,
            drawing_coord_x : drawing_coord_x + crop_hand_wd,
        ]
        + hand_cropped
    )
    return drawing


# constants
img_path = "./images/4.png"
hand_path = "./images/drawing-hand.png"
hand_mask_path = "./images/hand-mask.png"
frame_rate = 25
resize_wd, resize_ht = 800, 800
split_len = 10

start_time = time.time()

# video save path
img_name = img_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
video_save_name = img_name + "-u4" + ".mp4"
save_video_path = os.path.join("./save_videos", video_save_name)
print("save_video_path: ", save_video_path)

hand = cv2.imread(hand_path)
hand_mask = cv2.imread(hand_mask_path, cv2.IMREAD_GRAYSCALE)
img = cv2.imread(img_path)

top_left, bottom_right = get_extreme_coordinates(hand_mask)
hand = hand[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]
hand_mask = hand_mask[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]
hand_mask_inv = 255 - hand_mask

# standardizing the hand masks
hand_mask = hand_mask / 255
hand_mask_inv = hand_mask_inv / 255

# making the hand background black
hand_bg_ind = np.where(hand_mask == 0)
hand[hand_bg_ind] = [0, 0, 0]

img = cv2.resize(img, (resize_wd, resize_ht))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
cl1 = clahe.apply(img_gray)

# gaussian adaptive thresholding
img_thresh = cv2.adaptiveThreshold(
    cl1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 11
)

# canny edge detectino
edges = cv2.Canny(cl1, 50, 225)
edges = 255 - edges


# getting the img and hand dim
hand_ht, hand_wd = hand.shape[0], hand.shape[1]
img_ht, img_wd = img.shape[0], img.shape[1]

cv2.imshow("img", img)
cv2.imshow("cl1", cl1)
cv2.imshow("edges", edges)
cv2.imshow("img_gray", img_gray)
cv2.imshow("img_thresh", img_thresh)
cv2.imshow("hand", hand)
cv2.imshow("hand_mask", hand_mask)
cv2.imshow("hand_mask_inv", hand_mask_inv)
# cv2.imshow("drawing", drawing)

cv2.waitKey(0)
cv2.destroyAllWindows()

# draw
video = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (img.shape[1], img.shape[0]))

# creating an emtpy frame and select 0th index as the starting point to draw
drawn_frame = np.zeros(img.shape, np.uint8) + np.array([255, 255, 255], np.uint8)
selected_ind = 0
n_cuts_vertical = int(math.ceil(img.shape[0]/split_len))
n_cuts_horizontal = int(math.ceil(img.shape[1]/split_len))

grid_of_cuts = np.array(np.split(img_thresh, n_cuts_horizontal, axis=-1))
grid_of_cuts = np.array(np.split(grid_of_cuts, n_cuts_vertical, axis=-2))
print(grid_of_cuts.shape)

cut_having_black = (grid_of_cuts < 10)*1
cut_having_black = np.sum(np.sum(cut_having_black, axis = -1), axis = -1)
cut_black_indices = np.array(np.where(cut_having_black > 0)).T


counter = 0
while(len(cut_black_indices) > 1):
    selected_ind_val = cut_black_indices[selected_ind].copy()
    range_v_start = selected_ind_val[0]*split_len
    range_v_end = range_v_start + split_len
    range_h_start = selected_ind_val[1]*split_len
    range_h_end = range_h_start + split_len

    temp_drawing = np.zeros((split_len, split_len, 3))
    temp_drawing[:, :, 0] = grid_of_cuts[selected_ind_val[0]][selected_ind_val[1]]
    temp_drawing[:, :, 1] = grid_of_cuts[selected_ind_val[0]][selected_ind_val[1]]
    temp_drawing[:, :, 2] = grid_of_cuts[selected_ind_val[0]][selected_ind_val[1]]

    drawn_frame[range_v_start:range_v_end, range_h_start:range_h_end] = temp_drawing

    hand_coord_x = range_h_start + int(split_len/2)
    hand_coord_y = range_v_start + int(split_len/2)
    drawn_frame_with_hand = draw_hand_on_img(drawn_frame.copy(), hand.copy(), hand_coord_x, hand_coord_y, hand_mask_inv.copy(), hand_ht, hand_wd, img_ht, img_wd)

    # delete the selected ind from the d_array
    cut_black_indices[selected_ind] = cut_black_indices[-1]
    cut_black_indices = cut_black_indices[:-1]

    del selected_ind

    # select the next new index
    euc_arr = euc_dist(cut_black_indices, selected_ind_val)
    selected_ind = np.argmin(euc_arr)

    counter += 1
    if(counter%5 == 0):
        video.write(drawn_frame_with_hand)
        print("len of black indices: ", len(cut_black_indices))

for i in range(frame_rate):
    video.write(img_gray)

end_time = time.time()
print('total time: ', end_time - start_time)

video.release()
