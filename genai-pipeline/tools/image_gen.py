import os
import time
from google.genai import types
from config import IMAGE_GEN_MODEL, MODEL_NAME
from . import utils

def _verify_image_makes_sense(image_path: str, context_prompt: str) -> bool:
    """Verifies using Gemini if the given downloaded image makes sense for the prompt."""
    try:
        if not os.path.exists(image_path):
            return False
            
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        mime_type = "image/png"
        if image_path.lower().endswith(('.jpg', '.jpeg')):
            mime_type = "image/jpeg"

        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        
        prompt = f"Look at this image. Does this image genuinely represent the subject needed for this whiteboard animation prompt: '{context_prompt}'? Use Google Search if necessary to identify the subject in the image or verify its relevance. Answer ONLY with 'YES' or 'NO'."
        
        response = utils.client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],
            )
        )
        answer = response.text.strip().upper()
        print(f"Image verification for {image_path}: {answer}")
        return "YES" in answer
    except Exception as e:
        print(f"Warning: Image verification failed: {e}")
        return False


def image_gen_tool_fn(prompt: str, reference_image_path: str = None, subject_reference_image_path: str = None) -> str:
    """
    Generates a whiteboard animation image using Gemini 2.5 Flash Image.
    
    Args:
        prompt: The specific text prompt to generate an image for.
        reference_image_path: Optional path to a previously generated image for aesthetic consistency.
        subject_reference_image_path: Optional path to a real-world photo of the subject.
    Returns:
        The path to the generated image or an error message.
    """
    
    if not utils.client:
        return "Error: GEMINI_API_KEY not configured."

    try:
        
        # Explicitly ask for 16:9 just in case the backend config gets ignored
        contents = [prompt + " Ensure the generated image is in 16:9 aspect ratio (1920x1080). CRITICAL: DO NOT draw any hands, human arms, markers, pens, or people drawing. Draw ONLY the pure artwork on the whiteboard."]
        
        # Style consistency reference
        if reference_image_path and os.path.exists(reference_image_path):
            try:
                with open(reference_image_path, "rb") as f:
                    image_bytes = f.read()
                # Explicitly tell the model this is a REFERENCE, not a base to edit
                contents.append("The following image is the PREVIOUS scene in this storyboard series. Use it ONLY as a style and aesthetic reference to maintain visual consistency (line weight, color palette, character style). Do NOT replicate or edit this image — generate a completely NEW scene based on the prompt.")
                contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))
            except Exception as e:
                print(f"Warning: Could not read reference image {reference_image_path}: {e}")

        # Real-world subject reference (Internet image)
        if subject_reference_image_path and os.path.exists(subject_reference_image_path):
            is_valid = _verify_image_makes_sense(subject_reference_image_path, prompt)
            if is_valid:
                try:
                    with open(subject_reference_image_path, "rb") as f:
                        subject_bytes = f.read()
                    mime_type = "image/png"
                    if subject_reference_image_path.lower().endswith(('.jpg', '.jpeg')):
                        mime_type = "image/jpeg"
                    contents.append("Using the attached photograph as the SUBJECT STRUCTURE reference, transform this real-world subject into our specific whiteboard animation style, keeping facial features and main structural elements recognizable.")
                    contents.append(types.Part.from_bytes(data=subject_bytes, mime_type=mime_type))
                except Exception as e:
                    print(f"Warning: Could not read subject reference image {subject_reference_image_path}: {e}")
            else:
                print(f"Skipping internet reference image {subject_reference_image_path} as it was deemed not useful.")


        # Use the configured image generation model with a fallback to text-only if references fail
        try:
            response = utils.client.models.generate_content(
                model=IMAGE_GEN_MODEL,
                contents=contents,
            )
            if not response or response.parts is None:
                raise ValueError("Response or response.parts is None")
        except Exception as e:
            print(f"  [!] Image generation with reference failed ({e}). Retrying with text-only prompt...")
            text_only_prompt = prompt + " Ensure the generated image is in 16:9 aspect ratio (1920x1080). CRITICAL: DO NOT draw any hands, human arms, markers, pens, or people drawing. Draw ONLY the pure artwork on the whiteboard."
            response = utils.client.models.generate_content(
                model=IMAGE_GEN_MODEL,
                contents=[text_only_prompt],
            )

        output_path = None
        if response and response.parts:
            for part in response.parts:
                if part.inline_data is not None:
                    image = part.as_image()
                    timestamp = int(time.time())
                    filename = f"generated_image_{timestamp}.png"
                    output_path = os.path.join(utils.GLOBAL_OUTPUT_DIR, filename) if utils.GLOBAL_OUTPUT_DIR else filename
                    image.save(output_path)
                    print(f"Image generated and saved to: {output_path}")
                    break
                elif part.text:
                    print(f"Model response text: {part.text}")
        
        if output_path:
            return output_path
        
        return "Error: No image was generated in the response. Check logs for model text."
    except Exception as e:
        return f"An error occurred during image generation: {str(e)}"
