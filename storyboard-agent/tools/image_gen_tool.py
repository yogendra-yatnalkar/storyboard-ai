"""
Image Generation Tool using Gemini 3.1 Flash Image.
Enforces response_modalities=["IMAGE"] to eliminate billable text output tokens.
Supports conditioning on reference images with automated fallback to text-only prompts.
"""
import os
import time
from typing import Optional
from google.genai import types

from config import IMAGE_GEN_MODEL
from cost_tracker import tracker
from . import utils


def image_gen_tool_fn(
    prompt: str,
    reference_image_path: Optional[str] = None,
    subject_reference_image_path: Optional[str] = None,
    reference_image_paths: Optional[list] = None
) -> Optional[str]:
    """
    Generates an image using gemini-3.1-flash-image with up to 2 reference images attached.
    
    Args:
        prompt: Text prompt describing the whiteboard illustration or storyboard grid.
        reference_image_path: Optional prior scene image for style consistency.
        subject_reference_image_path: Optional real-world / reference portrait for visual guidance.
        reference_image_paths: Optional list of up to 2 reference image paths.
        
    Returns:
        Absolute path to the saved generated image, or None on failure.
    """
    if not utils.client:
        print("Error: GEMINI_API_KEY not configured.")
        return None

    # Collect all valid reference paths (up to 2)
    valid_refs = []
    if reference_image_paths:
        for p in reference_image_paths:
            if p and os.path.exists(p) and p not in valid_refs:
                valid_refs.append(p)
    if subject_reference_image_path and os.path.exists(subject_reference_image_path) and subject_reference_image_path not in valid_refs:
        valid_refs.append(subject_reference_image_path)
    if reference_image_path and os.path.exists(reference_image_path) and reference_image_path not in valid_refs:
        valid_refs.append(reference_image_path)

    # Cap to max 2 references
    valid_refs = valid_refs[:2]

    # Enforce whiteboard style and 16:9 aspect ratio
    system_instruction = (
        "Ensure the generated output is in 16:9 aspect ratio. "
        "CRITICAL: Draw a pure whiteboard marker sketch with crisp black lines on a clean white background. "
        "DO NOT draw any hands, human arms, markers, pens, or drawing tools. Draw ONLY the artwork."
    )
    full_prompt = f"{prompt.strip()} {system_instruction}"

    contents = [full_prompt]

    # Attach up to 2 reference images
    for idx, ref_path in enumerate(valid_refs):
        try:
            with open(ref_path, "rb") as f:
                img_bytes = f.read()
            mime_type = "image/png"
            if ref_path.lower().endswith((".jpg", ".jpeg")):
                mime_type = "image/jpeg"
            elif ref_path.lower().endswith(".webp"):
                mime_type = "image/webp"
            contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
            print(f"Image Gen: Attached Reference #{idx + 1}: {ref_path}")
        except Exception as e:
            print(f"Warning: Could not read reference image {ref_path} ({e}).")

    # Strictly enforce response_modalities=["IMAGE"] to eliminate expensive text output tokens!
    config = types.GenerateContentConfig(
        response_modalities=["IMAGE"]
    )

    response = None
    try:
        response = utils.client.models.generate_content(
            model=IMAGE_GEN_MODEL,
            contents=contents,
            config=config
        )
    except Exception as e:
        print(f"Image Gen with reference failed ({e}). Retrying with text-only prompt...")
        try:
            response = utils.client.models.generate_content(
                model=IMAGE_GEN_MODEL,
                contents=[full_prompt],
                config=config
            )
        except Exception as e2:
            print(f"Image Gen failed on retry: {e2}")
            return None

    if not response or not response.parts:
        print("Image Gen Error: Empty response parts received from model.")
        return None

    output_path = None
    for part in response.parts:
        if part.inline_data is not None:
            image = part.as_image()
            timestamp = int(time.time())
            filename = f"gen_image_{timestamp}.png"
            out_dir = utils.get_output_dir()
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, filename)
            image.save(output_path)
            tracker.record_image(1)
            print(f"Image Gen SUCCESS: Saved to {output_path}")
            return os.path.abspath(output_path)
        elif part.text:
            # We strictly don't expect text due to response_modalities=['IMAGE'], but log if model returned any
            print(f"Image Gen model text: {part.text[:80]}...")

    return None
