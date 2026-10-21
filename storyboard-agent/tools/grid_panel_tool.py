"""
3x2 Grid Storyboard Generation and Computer Vision Slicing Tool.
Generates a 6-panel comic storyboard grid (3 cols x 2 rows) in 1 API call,
then slices and rescales all 6 panels to 16:9 Landscape (1920x1080).
"""
import os
import time
from typing import List, Optional
import cv2
import numpy as np
from PIL import Image

from config import WHITEBOARD_OUTLINE_STYLE, OUTPUT_WIDTH, OUTPUT_HEIGHT
from state import ScenePlan
from . import utils
from .image_gen_tool import image_gen_tool_fn


def generate_storyboard_grid_3x2(
    scene_plan: ScenePlan,
    reference_image_paths: Optional[List[str]] = None
) -> Optional[str]:
    """
    Generates a 3x2 comic grid containing 6 sequential visual story panels using Gemini 3.1 Flash Image.
    Conditioned on up to 2 reference images for character/entity continuity.
    """
    panels = scene_plan.panels
    if len(panels) < 6:
        # Pad with default panels if needed
        from state import PanelPlan
        for i in range(len(panels) + 1, 7):
            panels.append(
                PanelPlan(
                    panel_number=i,
                    narration_beat="",
                    visual_description=f"Sequential action progression beat {i} for {scene_plan.description[:60]}"
                )
            )

    panels_desc_str = ""
    panel_positions = [
        "top-left", "top-center", "top-right",
        "bottom-left", "bottom-center", "bottom-right"
    ]
    for idx, p in enumerate(panels[:6]):
        pos = panel_positions[idx]
        panels_desc_str += f"Grid cell {idx + 1} ({pos}): {p.visual_description}\n"

    ref_instructions = ""
    if reference_image_paths and len(reference_image_paths) > 0:
        ref_instructions = (
            "CRITICAL CHARACTER CONSISTENCY: The attached reference images depict the key historical subjects. "
            "Maintain their facial structure, historical costume, turban, and attributes consistently across all 6 panels. "
        )

    grid_prompt = (
        f"A master 3x2 comic storyboard grid featuring exactly 6 clear rectangular story panels (3 columns x 2 rows) "
        f"separated by clean, wide white gutters on a pure white canvas.\n\n"
        f"{ref_instructions}"
        f"COMPOSITION & SPACING GUIDELINES:\n"
        f"- Clear Negative Space & Object Separation: Maintain distinct gaps and clean breathing room between characters, "
        f"figures, horses, and foreground/background elements. Avoid crowded overlapping blobs so each subject has clean, "
        f"isolated silhouettes and distinct contours for SAM segmentation.\n"
        f"- In-Scene Text & Story Typography: Where appropriate, enrich each panel with stylish comic-book lettering, "
        f"narrative banner captions, historical date/location tags (e.g. 'Delhi, 1707', 'Battle of Jajau', 'Deccan Frontier'), "
        f"short character dialogue balloons, or bold hand-drawn action sound effects (e.g. 'CLASH', 'CHARGE'). "
        f"(Strictly DO NOT draw panel index numbers like 'P1', 'P2', or 'Panel 1' in the corners).\n\n"
        f"Visual style: Vibrant graphic novel illustration with crisp bold black ink contours "
        f"and rich watercolor/marker fills on pure white background.\n\n"
        f"The 6 sequential narrative scenes are:\n"
        f"{panels_desc_str}"
    )

    print(f"[GridPanelTool] Generating 3x2 storyboard grid for Scene #{scene_plan.scene_number}...")
    grid_image_path = image_gen_tool_fn(
        prompt=grid_prompt,
        reference_image_paths=reference_image_paths
    )
    return grid_image_path


def _detect_panel_bounding_boxes(img: np.ndarray) -> Optional[List[tuple]]:
    """
    Detects exactly 6 rectangular panel bounding boxes using contour analysis.
    Returns sorted list of (x, y, w, h) boxes ordered:
    Panel 1 (top-left), Panel 2 (top-center), Panel 3 (top-right),
    Panel 4 (bottom-left), Panel 5 (bottom-center), Panel 6 (bottom-right).
    """
    H, W, _ = img.shape
    total_area = float(W * H)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # A valid 3x2 panel is roughly 5% to 30% of total image area
        if 0.05 * total_area < area < 0.32 * total_area:
            candidates.append((x, y, w, h))

    if len(candidates) == 6:
        # Sort into top row (row 0) and bottom row (row 1)
        # Bounding box centers in Y
        row_split = H / 2.0
        top_row = [b for b in candidates if (b[1] + b[3] / 2.0) < row_split]
        bot_row = [b for b in candidates if (b[1] + b[3] / 2.0) >= row_split]

        if len(top_row) == 3 and len(bot_row) == 3:
            # Sort each row left-to-right by x coordinate
            top_row.sort(key=lambda b: b[0])
            bot_row.sort(key=lambda b: b[0])
            return top_row + bot_row

    return None


def slice_3x2_grid_into_panels(
    grid_image_path: str,
    output_dir: Optional[str] = None,
    target_width: int = OUTPUT_WIDTH,
    target_height: int = OUTPUT_HEIGHT
) -> List[str]:
    """
    Slices a 3x2 grid image into 6 individual panels using Computer Vision.
    Uses adaptive contour bounding box detection if available, with geometric fallback.
    Centers each panel on a pure white 16:9 canvas (1920x1080) and saves.
    
    Returns:
        List of 6 file paths: [panel_1.png, ..., panel_6.png]
    """
    if not grid_image_path or not os.path.exists(grid_image_path):
        print(f"[GridPanelTool] Error: Grid image not found at '{grid_image_path}'.")
        return []

    if output_dir is None:
        output_dir = utils.get_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    # Read image with OpenCV
    img = cv2.imread(grid_image_path)
    if img is None:
        print(f"[GridPanelTool] Error: OpenCV could not read image '{grid_image_path}'.")
        return []

    H, W, _ = img.shape
    print(f"[GridPanelTool] Slicing grid image ({W}x{H}) into 6 panels (3 columns x 2 rows)...")

    # Try adaptive contour detection first
    detected_boxes = _detect_panel_bounding_boxes(img)
    if detected_boxes:
        print(f"[GridPanelTool] Adaptive CV: Detected exact 6 panel bounding boxes!")
    else:
        print(f"[GridPanelTool] Geometric Fallback: Slicing by uniform 3x2 grid coordinates.")

    panel_paths = []
    timestamp = int(time.time())

    col_width = W / 3.0
    row_height = H / 2.0
    inset_x = int(col_width * 0.03)
    inset_y = int(row_height * 0.03)

    for idx in range(6):
        panel_idx = idx + 1
        if detected_boxes:
            bx, by, bw, bh = detected_boxes[idx]
            # Strip 2% inner gutter margin from detected box
            pad_x = int(bw * 0.02)
            pad_y = int(bh * 0.02)
            crop = img[by + pad_y : by + bh - pad_y, bx + pad_x : bx + bw - pad_x]
        else:
            r = idx // 3
            c = idx % 3
            y_start = int(r * row_height) + inset_y
            y_end = int((r + 1) * row_height) - inset_y
            x_start = int(c * col_width) + inset_x
            x_end = int((c + 1) * col_width) - inset_x
            crop = img[y_start:y_end, x_start:x_end]

        crop_h, crop_w, _ = crop.shape
        if crop_h <= 10 or crop_w <= 10:
            print(f"[GridPanelTool] Warning: Panel {panel_idx} crop too small ({crop_w}x{crop_h}).")
            continue

        # Clean any stray text/label in the top-left corner (e.g. 'P1', 'P6')
        top_h = int(crop_h * 0.12)
        top_w = int(crop_w * 0.15)
        corner = crop[:top_h, :top_w]
        corner_gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
        dark_pixel_ratio = np.mean(corner_gray < 200)
        # If there is small isolated dark text (between 0.5% and 25% of the corner patch), whiten it out
        if 0.005 < dark_pixel_ratio < 0.28:
            crop[:top_h, :top_w] = 255
            print(f"  [Panel {panel_idx}] Cleaned stray corner label ({dark_pixel_ratio*100:.1f}% ink)")

        # Scale to fit within target 16:9 canvas (1920x1080) with pure white background
        scale = min((target_width * 0.95) / crop_w, (target_height * 0.95) / crop_h)
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)

        # High-definition Lanczos upscaling (eliminates blurriness from INTER_AREA)
        resized_crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Apply unsharp mask edge enhancement to restore crisp line art
        gaussian = cv2.GaussianBlur(resized_crop, (0, 0), sigmaX=1.5)
        sharpened_crop = cv2.addWeighted(resized_crop, 1.35, gaussian, -0.35, 0)

        # Create pure white 16:9 canvas
        canvas = np.full((target_height, target_width, 3), 255, dtype=np.uint8)

        # Center resized panel on canvas
        x_offset = (target_width - new_w) // 2
        y_offset = (target_height - new_h) // 2
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = sharpened_crop

        panel_filename = f"panel_{timestamp}_{panel_idx}.png"
        panel_path = os.path.join(output_dir, panel_filename)
        cv2.imwrite(panel_path, canvas)

        panel_paths.append(os.path.abspath(panel_path))
        print(f"  [Panel {panel_idx}] Extracted & framed to 16:9 -> {panel_filename}")

    return panel_paths
