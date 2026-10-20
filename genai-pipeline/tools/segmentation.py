import os
import time
import json
import requests
from google.genai import types
from config import MODEL_NAME, SAM_API_URL
from . import utils

def segmentation_tool_fn(image_path: str) -> str:
    """
    Performs instance segmentation on an image.
    
    This tool follows a two-step process:
    1. It uses a Gemini model to identify major, distinct objects in the image.
    2. For each identified object, it calls a hosted SAM3 (Segment Anything Model) 
       API to generate a high-quality segmentation mask.
    
    Args:
        image_path (str): The absolute path to the input image file (PNG or JPEG).
        
    Returns:
        str: The absolute path to a JSON file containing the results.
             The JSON structure is:
             {
               "image_path": str,
               "objects": [str, ...],
               "segmentations": {
                 "object_name": {
                   "boxes": [[x1, y1, x2, y2], ...],
                   "scores": [float, ...],
                   "mask_shape": [n, c, h, w],
                   "masks_base64": [str, ...] # List of PNG masks encoded in base64
                 }
               }
             }
    """
    if not utils.client:
        return "Error: GEMINI_API_KEY not configured."
    
    if not os.path.exists(image_path):
        return f"Error: Image file not found at {image_path}"
    
    print(f"Starting segmentation process for: {image_path}")

    # 1. Object Identification: Ask Gemini to find segmentable parts
    # We use a specific prompt to get a clean JSON list of object names
    id_prompt = 'Identify the 3-5 largest and most distinct physical object groups in this image that define the scene. Group smaller related parts into large logical entities (e.g., instead of "wheel", "pedal", "seat", just say "bicycle"). Return a raw JSON list of strings, for example: ["bicycle", "rider", "background building"]. Do not include markdown formatting or explanation.'
    
    objects = []
    try:
        # Detect mime type based on extension
        mime_type = "image/png"
        if image_path.lower().endswith(".jpg") or image_path.lower().endswith(".jpeg"):
            mime_type = "image/jpeg"
            
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            
        contents = [
            id_prompt,
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        ]
        
        # Call Gemini Vision to get the object list
        response = utils.client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config={
                'response_mime_type': 'application/json'
            }
        )
        
        objects = json.loads(response.text)
        print(f"Identified objects: {objects}")
        
    except Exception as e:
        return f"Error identifying objects with Gemini: {str(e)}"

    if not objects or not isinstance(objects, list):
         return "Error: Gemini failed to return a valid list of objects."

    # 2. Instance Segmentation: Call SAM3 for each identified object
    combined_results = {
        "image_path": image_path,
        "objects": objects,
        "segmentations": {}
    }
    
    # Try to obtain Google ID Token for Cloud Run authentication
    headers = {}
    try:
        import google.auth
        import google.auth.transport.requests
        from google.oauth2 import id_token
        
        credentials, project = google.auth.default()
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        
        # Audience is the base Cloud Run service URL
        audience = SAM_API_URL.split("/predict")[0]
        token = id_token.fetch_id_token(auth_req, audience)
        if token:
            headers["Authorization"] = f"Bearer {token}"
            print("Attached Google ID Token for Cloud Run authentication.")
    except Exception as e:
        print(f"Note: Cloud Run authentication check: {e}")

    for obj in objects:
        print(f"Segmenting object: {obj}...")
        try:
           # Send image and object name to the SAM3 endpoint
           with open(image_path, "rb") as f:
               files = {"file": f}
               data = {"prompt": obj} 
               
               response = requests.post(SAM_API_URL, files=files, data=data, headers=headers) 
               
               if response.status_code == 200:
                   result = response.json()
                   combined_results["segmentations"][obj] = result
               else:
                   print(f"SAM3 failed for {obj}: {response.status_code} - {response.text}")
                   combined_results["segmentations"][obj] = {"error": f"Status {response.status_code}: {response.text}"}
                   
        except Exception as e:
            print(f"Error calling SAM3 for {obj}: {e}")
            combined_results["segmentations"][obj] = {"error": str(e)}

    # 3. Finalization: Save results to a timestamped JSON file
    timestamp = int(time.time())
    output_filename = f"segmentation_results_{timestamp}.json"
    
    # Use the global output directory if available (managed by utils)
    if utils.GLOBAL_OUTPUT_DIR:
        saved_path = utils._save_to_run_folder(json.dumps(combined_results, indent=2), output_filename)
        print(f"Segmentation results saved to: {saved_path}")
        return saved_path
    else:
        # Fallback to local directory if no global output is set
        try:
            full_path = os.path.abspath(output_filename)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(combined_results, indent=2))
            return full_path
        except Exception as e:
            return f"Error saving file: {str(e)}. Raw JSON: " + json.dumps(combined_results)
