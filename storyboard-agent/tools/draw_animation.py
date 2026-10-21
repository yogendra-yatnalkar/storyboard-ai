import os
import cv2
import time
import numpy as np
import math
import json
import base64
from . import utils

def euc_dist(arr1, point):
    """
    Calculates the Euclidean distance between a set of points and a reference point.
    """
    square_sub = (arr1 - point) ** 2
    return np.sqrt(np.sum(square_sub, axis=1))

def preprocess_image(img_path, resize_wd, resize_ht):
    """
    Reads an image, resizes it, and generates a binary adaptive threshold mask.
    The threshold mask is used to identify "ink" pixels for the animation.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None, None, None
    img_ht, img_wd = img.shape[0], img.shape[1]
    img = cv2.resize(img, (resize_wd, resize_ht))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gaussian adaptive thresholding: identifies edges and details
    img_thresh = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
    )
    return img, img_thresh, img_ht, img_wd, img_gray

def preprocess_hand_image(hand_path, hand_mask_path):
    """
    Prepares the 'drawing hand' asset by cropping it to its visible area 
    and preparing foreground/background masks for blending.
    """
    hand = cv2.imread(hand_path)
    hand_mask = cv2.imread(hand_mask_path, cv2.IMREAD_GRAYSCALE)
    if hand is None or hand_mask is None:
        return None, None, None, None, None

    # Get extreme coordinates to crop hand to its mask size (removes unused empty space)
    indices = np.where(hand_mask == 255)
    x, y = indices[1], indices[0]
    top_left = (np.min(x), np.min(y))
    bottom_right = (np.max(x), np.max(y))
    
    hand = hand[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]
    hand_mask = hand_mask[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]
    hand_mask_inv = 255.0 - hand_mask

    # Normalize masks to [0.0, 1.0] for multiplicative blending
    hand_mask = hand_mask / 255.0
    hand_mask_inv = hand_mask_inv / 255.0

    # Ensure hand background is blacked out
    hand_bg_ind = np.where(hand_mask == 0)
    hand[hand_bg_ind] = [0, 0, 0]

    hand_ht, hand_wd = hand.shape[0], hand.shape[1]
    return hand, hand_mask, hand_mask_inv, hand_ht, hand_wd

def draw_hand_on_img(drawing, hand, drawing_coord_x, drawing_coord_y, hand_mask_inv, hand_ht, hand_wd, img_ht, img_wd):
    """
    Blends the hand image onto the current frame at the specified coordinates.
    Handles boundaries to prevent cropping/overflow errors.
    """
    remaining_ht = img_ht - drawing_coord_y
    remaining_wd = img_wd - drawing_coord_x
    crop_hand_ht = min(hand_ht, remaining_ht)
    crop_hand_wd = min(hand_wd, remaining_wd)

    if crop_hand_ht <= 0 or crop_hand_wd <= 0:
        return drawing

    # Crop hand if it goes off the bottom/right edge
    hand_cropped = hand[:crop_hand_ht, :crop_hand_wd]
    hand_mask_inv_cropped = hand_mask_inv[:crop_hand_ht, :crop_hand_wd]

    # Perform blending: black out the background area then add the hand foreground
    for c in range(3):
        drawing[drawing_coord_y : drawing_coord_y + crop_hand_ht, drawing_coord_x : drawing_coord_x + crop_hand_wd, c] = (
            drawing[drawing_coord_y : drawing_coord_y + crop_hand_ht, drawing_coord_x : drawing_coord_x + crop_hand_wd, c] * hand_mask_inv_cropped
        )
    
    drawing[drawing_coord_y : drawing_coord_y + crop_hand_ht, drawing_coord_x : drawing_coord_x + crop_hand_wd] += hand_cropped
    return drawing

def draw_masked_object(
    drawn_frame, img_thresh, img_orig, video_object, hand, hand_mask_inv, hand_ht, hand_wd,
    resize_ht, resize_wd, split_len, object_mask=None, skip_rate=5, black_pixel_threshold=10
):
    """
    Core animation logic: simulates drawing a specific part of the image (or the whole image).
    Divides the area into grids, finds 'ink' pixels (black in img_thresh), and visits them 
    in a nearest-neighbor order to simulate natural drawing motion.
    """
    img_thresh_copy = img_thresh.copy()
    object_ind = None
    if object_mask is not None:
        # If a mask is provided, ignore pixels outside the mask
        object_mask_black_ind = np.where(object_mask == 0)
        object_ind = np.where(object_mask == 255)
        img_thresh_copy[object_mask_black_ind] = 255

    n_cuts_vertical = int(math.ceil(resize_ht / split_len))
    n_cuts_horizontal = int(math.ceil(resize_wd / split_len))

    # Padding to ensure the image fits the grid exactly
    pad_h = n_cuts_vertical * split_len - resize_ht
    pad_w = n_cuts_horizontal * split_len - resize_wd
    if pad_h > 0 or pad_w > 0:
        img_thresh_copy = cv2.copyMakeBorder(img_thresh_copy, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)

    # Split image into a grid of small blocks
    grid_of_cuts = np.array(np.split(img_thresh_copy, n_cuts_horizontal, axis=-1))
    grid_of_cuts = np.array(np.split(grid_of_cuts, n_cuts_vertical, axis=-2))

    # Find which grid blocks contain 'ink' pixels
    cut_having_black = (grid_of_cuts < black_pixel_threshold) * 1
    cut_having_black = np.sum(np.sum(cut_having_black, axis=-1), axis=-1)
    cut_black_indices = np.array(np.where(cut_having_black > 0)).T

    selected_ind = 0
    counter = 0
    while len(cut_black_indices) > 0:
        # Get coordinates of the current grid block
        selected_ind_val = cut_black_indices[selected_ind].copy()
        range_v_start = selected_ind_val[0] * split_len
        range_v_end = range_v_start + split_len
        range_h_start = selected_ind_val[1] * split_len
        range_h_end = range_h_start + split_len

        # Map back to original image size (clipping padding)
        v_s, v_e = range_v_start, min(range_v_end, resize_ht)
        h_s, h_e = range_h_start, min(range_h_end, resize_wd)
        
        if v_s < resize_ht and h_s < resize_wd:
            # Transfer the ink to the 'drawn_frame'
            block = grid_of_cuts[selected_ind_val[0]][selected_ind_val[1]][:v_e-v_s, :h_e-h_s]
            if object_mask is not None:
                # Only update pixels that belong to this object's mask (255)
                # This prevents the background pass from overwriting already colored objects
                m_slice = object_mask[v_s:v_e, h_s:h_e] == 255
                for c in range(3):
                   drawn_frame[v_s:v_e, h_s:h_e, c][m_slice] = block[m_slice]
            else:
                for c in range(3):
                    drawn_frame[v_s:v_e, h_s:h_e, c] = block

        # Position for the drawing hand
        hand_coord_x = h_s + int((h_e - h_s) / 2)
        hand_coord_y = v_s + int((v_e - v_s) / 2)
        
        # Remove the block we just finished drawing
        cut_black_indices = np.delete(cut_black_indices, selected_ind, axis=0)
        
        # Select the NEXT block to draw based on proximity (nearest neighbor)
        if len(cut_black_indices) > 0:
            euc_arr = euc_dist(cut_black_indices, selected_ind_val)
            selected_ind = np.argmin(euc_arr)

        counter += 1
        # Periodically write a frame to the video to create the animation effect
        if counter % skip_rate == 0:
            frame_to_write = draw_hand_on_img(
                drawn_frame.copy(), hand, hand_coord_x, hand_coord_y, hand_mask_inv,
                hand_ht, hand_wd, resize_ht, resize_wd
            )
            video_object.write(frame_to_write)

    # After the animation pass, fill the area with the full-color original pixels
    if object_mask is not None:
        drawn_frame[object_ind] = img_orig[object_ind]
    else:
        drawn_frame[:] = img_orig[:]

def draw_animation_tool_fn(
    image_path: str,
    segmentation_results_path: str = None,
    frame_rate: int = 25,
    resize_wd: int = 1280,
    resize_ht: int = 720,
    split_len: int = 16,
    object_skip_rate: int = 12,
    bg_object_skip_rate: int = 20,
    end_duration_sec: int = 2
) -> str:
    """
    Generates a whiteboard animation video of an image.
    
    If 'segmentation_results_path' is provided (from the segmentation tool), 
    the tool will draw each object one by one before drawing the background.
    Otherwise, it draws the entire image in a single pass.
    
    Args:
        image_path (str): Absolute path to the input image.
        segmentation_results_path (str, optional): Path to the JSON output of the segmentation tool.
        frame_rate (int): FPS of the output video. Default 25.
        resize_wd (int): Resize image to this width before processing. Default 1280.
        resize_ht (int): Resize image to this height before processing. Default 720.
        split_len (int): Grid size for ink detection. Smaller is more detailed but slower. Default 10.
        object_skip_rate (int): Write 1 frame for every N grid blocks drawn for objects. Default 8.
        bg_object_skip_rate (int): Skip rate for background drawing (usually faster). Default 14.
        end_duration_sec (int): How many seconds to show the finished image at the end. Default 3.
        
    Returns:
        str: Absolute path to the generated MP4 video.
    """
    # Locate hand assets relative to the package structure
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
    hand_path = os.path.join(assets_dir, "drawing-hand.png")
    hand_mask_path = os.path.join(assets_dir, "hand-mask.png")

    # Load and prepare the image
    img, img_thresh, img_ht, img_wd, img_gray = preprocess_image(image_path, resize_wd, resize_ht)
    if img is None:
        return f"Error: Could not read image at {image_path}"

    # Load and prepare the drawing hand
    hand, hand_mask, hand_mask_inv, hand_ht, hand_wd = preprocess_hand_image(hand_path, hand_mask_path)
    if hand is None:
        return "Error: Hand assets not found. Ensure adk-agent/assets contains hand images."

    # Prepare output video
    timestamp = int(time.time())
    output_filename = f"whiteboard_animation_{timestamp}.mp4"
    temp_video_path = output_filename 

    video_object = cv2.VideoWriter(
        temp_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        frame_rate,
        (resize_wd, resize_ht),
    )

    # canvas initialized to white
    drawn_frame = np.full((resize_ht, resize_wd, 3), 255, dtype=np.uint8)

    if segmentation_results_path and os.path.exists(segmentation_results_path):
        # SUB-OBJECT ANIMATION MODE
        with open(segmentation_results_path, 'r') as f:
            seg_data = json.load(f)
        
        # We track which pixels belong to the background
        background_mask = np.full((resize_ht, resize_wd), 255, dtype=np.uint8)
        
        # 1. Reconstruct all masks first
        reconstructed_masks = []
        for obj_name, obj_seg in seg_data.get("segmentations", {}).items():
            if "masks_base64" not in obj_seg: continue
            
            full_obj_mask = np.zeros((resize_ht, resize_wd), dtype=np.uint8)
            for b64_mask in obj_seg["masks_base64"]:
                mask_data = base64.b64decode(b64_mask)
                nparr = np.frombuffer(mask_data, np.uint8)
                mask_part = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                if mask_part is not None:
                    mask_part = cv2.resize(mask_part, (resize_wd, resize_ht))
                    _, binary_part = cv2.threshold(mask_part, 127, 255, cv2.THRESH_BINARY)
                    full_obj_mask = cv2.bitwise_or(full_obj_mask, binary_part)
            
            reconstructed_masks.append(full_obj_mask)

        # 2. Merge overlapping masks
        # We use a simple adjacency-based merging (Disjoint Set Union style)
        num_masks = len(reconstructed_masks)
        parent = list(range(num_masks))

        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        for i in range(num_masks):
            for j in range(i + 1, num_masks):
                m1 = reconstructed_masks[i]
                m2 = reconstructed_masks[j]
                
                # Calculate intersection
                intersect = cv2.bitwise_and(m1, m2)
                intersect_area = np.sum(intersect == 255)
                
                if intersect_area > 0:
                    area1 = np.sum(m1 == 255)
                    area2 = np.sum(m2 == 255)
                    min_area = min(area1, area2)
                    
                    # If significant overlap, merge them
                    if min_area > 0 and (intersect_area / min_area) > 0.3:
                        union(i, j)

        # Group masks by their root parent
        merged_groups = {}
        for i in range(num_masks):
            root = find(i)
            if root not in merged_groups:
                merged_groups[root] = np.zeros((resize_ht, resize_wd), dtype=np.uint8)
            merged_groups[root] = cv2.bitwise_or(merged_groups[root], reconstructed_masks[i])

        # 3. Animate the merged groups
        for root, merged_mask in merged_groups.items():
            # Animate drawing this specific merged object
            draw_masked_object(
                drawn_frame, img_thresh, img, video_object, hand, hand_mask_inv, hand_ht, hand_wd,
                resize_ht, resize_wd, split_len, object_mask=merged_mask, skip_rate=object_skip_rate
            )
            
            # Mark this object's area as 'drawn' in the background mask
            obj_ind = np.where(merged_mask == 255)
            background_mask[obj_ind] = 0
            
        # Draw the remaining area (background)
        draw_masked_object(
            drawn_frame, img_thresh, img, video_object, hand, hand_mask_inv, hand_ht, hand_wd,
            resize_ht, resize_wd, split_len=20, object_mask=background_mask, skip_rate=bg_object_skip_rate
        )
    else:
        # SINGLE PASS MODE
        draw_masked_object(
            drawn_frame, img_thresh, img, video_object, hand, hand_mask_inv, hand_ht, hand_wd,
            resize_ht, resize_wd, split_len, skip_rate=object_skip_rate
        )

    # Finally, show the full-color final image for a few seconds
    for _ in range(frame_rate * end_duration_sec):
        video_object.write(img)

    video_object.release()
    time.sleep(0.1)  # Allow system/OpenCV to flush and release file lock

    # Convert OpenCV mp4v to standard H.264 (yuv420p @ frame_rate fps)
    h264_temp = f"h264_{temp_video_path}"
    try:
        import subprocess
        norm_cmd = [
            "ffmpeg", "-y",
            "-i", temp_video_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", str(frame_rate),
            h264_temp
        ]
        subprocess.run(norm_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        if os.path.exists(h264_temp) and os.path.getsize(h264_temp) > 1000:
            try:
                os.remove(temp_video_path)
            except Exception:
                pass
            temp_video_path = h264_temp
    except Exception as norm_err:
        print(f"Warning: ffmpeg h264 conversion error: {norm_err}")

    # Move video to the final output directory if set
    if utils.GLOBAL_OUTPUT_DIR:
        final_path = os.path.join(utils.GLOBAL_OUTPUT_DIR, output_filename)
        if os.path.exists(temp_video_path):
            if os.path.exists(final_path):
                try:
                    os.remove(final_path)
                except Exception:
                    pass
            
            # Retry renaming a few times
            for attempt in range(5):
                try:
                    os.rename(temp_video_path, final_path)
                    return final_path
                except PermissionError:
                    time.sleep(0.2)
            
            # Fallback to shutil copy + remove
            import shutil
            try:
                shutil.copy2(temp_video_path, final_path)
                try:
                    os.remove(temp_video_path)
                except Exception:
                    pass
                return final_path
            except Exception as copy_err:
                print(f"Error copying video on fallback: {copy_err}")
    
    return os.path.abspath(temp_video_path)
