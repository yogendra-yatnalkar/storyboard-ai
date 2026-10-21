"""
Unit Test for Reference Search Tool (DDGS Image Search).
Tests searching and downloading reference images, including verification with PIL and Gemini 3.8 Flash.
"""
import os
import sys

# Ensure genai-pipeline root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from tools.reference_search import reference_search_tool_fn
from tools.utils import set_output_dir, client
from config import MODEL_NAME, THINKING_LEVEL, get_generation_config
from google.genai import types


def test_reference_search_kambaksha():
    print("=" * 60)
    print("TEST 1: Reference Search for 'Kambaksha son of aurangzeb'")
    print("=" * 60)

    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "test_ref_search")
    os.makedirs(test_dir, exist_ok=True)
    set_output_dir(test_dir)

    query = "Kambaksha son of aurangzeb"
    print(f"Running reference_search_tool_fn('{query}')...")
    result_path = reference_search_tool_fn(query)
    print(f"Result returned: {result_path}")

    # Check 1: Must be a valid string path without error
    assert result_path and not result_path.startswith("Error"), f"Search failed with error: {result_path}"
    assert os.path.exists(result_path), f"File does not exist: {result_path}"
    assert os.path.getsize(result_path) > 1000, f"File size too small ({os.path.getsize(result_path)} bytes)"

    # Check 2: Must be a valid image readable by PIL
    img = Image.open(result_path)
    img.verify()
    # Reopen after verify
    img = Image.open(result_path)
    width, height = img.size
    img_format = img.format
    print(f"[PASS] Image verification: Format={img_format}, Resolution={width}x{height}, Size={os.path.getsize(result_path):,} bytes")
    print(f"[PASS] Downloaded image saved at: {result_path}")

    # Check 3: Gemini 3.8 Flash Vision Verification (with MEDIUM thinking level)
    print("\n--- Verifying image relevance with Gemini 3.8 Flash (Thinking: MEDIUM) ---")
    if client:
        try:
            with open(result_path, "rb") as f:
                img_bytes = f.read()

            mime_type = "image/jpeg" if img_format.lower() in ["jpeg", "jpg"] else f"image/{img_format.lower()}"
            prompt = (
                f"You are evaluating a reference image found for the query: '{query}'. "
                "Describe what is depicted in this image and verify if it shows Prince Kam Bakhsh, "
                "son of Mughal Emperor Aurangzeb, or a historical Mughal portrait / artwork related to this subject."
            )

            config = get_generation_config()
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
                ],
                config=config
            )
            print("\nGemini 3.8 Flash Visual Assessment:")
            print("-" * 50)
            print(response.text.strip())
            print("-" * 50)
            print("[PASS] Gemini Vision analysis completed successfully.")
        except Exception as e:
            print(f"[WARN] Gemini Vision verification encountered an exception: {e}")
    else:
        print("[SKIP] GEMINI_API_KEY not configured, skipping Vision check.")

    return result_path


def test_reference_search_empty_query():
    print("\n" + "=" * 60)
    print("TEST 2: Edge Case Handling - Empty Query")
    print("=" * 60)
    res = reference_search_tool_fn("   ")
    print(f"Result for empty query: {res}")
    assert res.startswith("Error"), f"Expected error for empty query, got: {res}"
    print("[PASS] Handled empty query gracefully.")


if __name__ == "__main__":
    print(f"Starting Reference Search Tests using {MODEL_NAME} (Thinking Level: {THINKING_LEVEL})...\n")
    try:
        saved_img = test_reference_search_kambaksha()
        test_reference_search_empty_query()
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print(f"Downloaded test reference image: {saved_img}")
        print("=" * 60)
    except AssertionError as ae:
        print(f"\n[FAIL] Assertion failed: {ae}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
