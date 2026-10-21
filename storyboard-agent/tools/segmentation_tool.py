"""
Segmentation Tool for SAM 3 and Object Identification.
Identifies scene objects with Gemini Vision and retrieves segmentations from SAM 3 Cloud Run endpoint
(or provides local fallback when endpoint is unset or offline).
"""
import json
import os
import time
import requests
from typing import Optional
from google.genai import types

from config import MODEL_NAME, SAM_API_URL, get_generation_config
from cost_tracker import tracker
from . import utils


def segmentation_tool_fn(image_path: str) -> Optional[str]:
    """
    Identifies visual objects using Gemini Vision and segments them using SAM 3.
    
    Args:
        image_path: Path to the image to segment.
        
    Returns:
        Path to the saved segmentation JSON file, or None on failure.
    """
    if not utils.client:
        print("Error: GEMINI_API_KEY not configured.")
        return None

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None

    print(f"Segmentation: Identifying visual objects in {image_path}...")

    # 1. Object Identification with Gemini 3.8 Flash Vision
    objects = []
    try:
        id_prompt = (
            "Analyze this whiteboard image and identify the main distinct visual objects, "
            "characters, focal elements, or items that can be drawn individually in an animation. "
            "Return ONLY a JSON list of short string descriptions/labels for the objects. "
            "Do NOT include the background, whiteboard itself, or general scene descriptors. "
            "Example: [\"portrait of prince\", \"turban\", \"sword\", \"calligraphy caption\"]"
        )

        mime_type = "image/png"
        if image_path.lower().endswith((".jpg", ".jpeg")):
            mime_type = "image/jpeg"

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        contents = [
            id_prompt,
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        ]

        config = get_generation_config(model=MODEL_NAME, response_mime_type="application/json")
        response = utils.client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=config
        )
        tracker.record_usage_metadata(response.usage_metadata)
        objects = json.loads(response.text.strip())
        print(f"Segmentation: Identified objects: {objects}")

    except Exception as e:
        print(f"Warning: Object identification failed ({e}). Using default object.")
        objects = ["main subject"]

    if not objects or not isinstance(objects, list):
        objects = ["main subject"]

    combined_results = {
        "image_path": os.path.abspath(image_path),
        "objects": objects,
        "segmentations": {}
    }

    # 2. SAM 3 Segmentation
    headers = {}
    if SAM_API_URL and SAM_API_URL.strip():
        # Step A: Try standard Google Auth
        try:
            import google.auth
            import google.auth.transport.requests
            from google.oauth2 import id_token

            credentials, project = google.auth.default()
            auth_req = google.auth.transport.requests.Request()
            credentials.refresh(auth_req)
            audience = SAM_API_URL.split("/predict")[0]
            token = id_token.fetch_id_token(auth_req, audience)
            if token:
                headers["Authorization"] = f"Bearer {token}"
                print("Attached Google ID Token from ADC.")
        except Exception:
            pass

        # Step B: Fallback to gcloud CLI identity token
        if "Authorization" not in headers:
            try:
                import subprocess
                gcloud_token = subprocess.check_output(
                    "gcloud auth print-identity-token",
                    shell=True,
                    text=True,
                    stderr=subprocess.DEVNULL
                ).strip()
                if gcloud_token:
                    headers["Authorization"] = f"Bearer {gcloud_token}"
                    print("Attached Google ID Token from gcloud CLI.")
            except Exception as e:
                print(f"Note: Cloud Run auth check: {e}")

        for obj in objects:
            print(f"SAM 3 (Cloud Run): Segmenting object '{obj}'...")
            try:
                with open(image_path, "rb") as f:
                    files = {"file": f}
                    data = {"prompt": obj}
                    resp = requests.post(SAM_API_URL, files=files, data=data, headers=headers, timeout=25)
                    if resp.status_code == 200:
                        seg_data = resp.json()
                        combined_results["segmentations"][obj] = seg_data
                        print(f"SAM 3: Successfully segmented '{obj}' ({len(seg_data.get('masks_base64', []))} masks)")
                    else:
                        print(f"SAM 3 endpoint returned {resp.status_code}. Using local mask fallback.")
                        combined_results["segmentations"][obj] = {"fallback": True}
            except Exception as e:
                print(f"SAM 3 request failed ({e}). Using local fallback.")
                combined_results["segmentations"][obj] = {"fallback": True}
    else:
        # Endpoint not set: mark for graceful local fallback
        print("Note: SAM_API_URL not configured. Using local visual contour animation.")
        for obj in objects:
            combined_results["segmentations"][obj] = {"fallback": True}

    # 3. Save Results
    timestamp = int(time.time())
    output_filename = f"segmentation_{timestamp}.json"
    out_dir = utils.get_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_filename)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2)

    print(f"Segmentation: Results saved to {out_path}")
    return os.path.abspath(out_path)
