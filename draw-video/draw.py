import cv2
import numpy as np

def euc_dist(arr1, point):
    square_sub = (arr1 - point)**2
    return np.sqrt(np.sum(square_sub, axis=1))


img = cv2.imread("whiteboard.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# finding all points where pixel is black
black_indices = np.array(np.where(img_gray == 0)).T
print("Length of black indices: ", len(black_indices))

# draw
video = cv2.VideoWriter("video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 25, (img.shape[1], img.shape[0]))

# creating an emtpy frame and select 0th index as the starting point to draw
drawn_frame = np.zeros(img.shape, np.uint8) + np.array([255, 255, 255], np.uint8)
selected_ind = 0

counter = 0
while(len(black_indices) > 1):
    selected_ind_val = black_indices[selected_ind].copy()
    drawn_frame[selected_ind_val[0], selected_ind_val[1]] = [0, 0, 0]

    # print("-"*10)
    # print("counter: ", counter)
    # print("Selected Index: ", selected_ind)
    # print("selected in val: ", selected_ind_val)
    # print("len of black indices: ", len(black_indices))
    # print(black_indices[:10])

    # delete the selected in from the black_indices_array
    black_indices[selected_ind] = black_indices[-1]
    black_indices = black_indices[:-1]
    
    # print("selected in val: ", selected_ind_val)
    # print("len of black indices: ", len(black_indices))
    # print(black_indices[:10])

    del selected_ind

    # select the next new index
    euc_arr = euc_dist(black_indices, selected_ind_val)
    selected_ind = np.argmin(euc_arr)
    # print("EUC array: ", euc_arr[:10])
    # print("-"*10)

    if(counter%30 == 0):
        video.write(drawn_frame)
        print("counter: ", counter)
        print("Selected Index: ", selected_ind)
        print("len of black indices: ", len(black_indices))

    # if(counter == 1):
    #     break
    counter += 1

video.release()
cv2.imshow("img", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
