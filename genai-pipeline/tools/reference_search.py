import io
import os
import re
import time
import urllib.parse
import urllib.request
from ddgs import DDGS
from PIL import Image
from . import utils

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def reference_search_tool_fn(query: str, max_results: int = 10) -> str:
    """
    Searches DuckDuckGo Images (via ddgs) for the given query and downloads the best matching image.
    
    Args:
        query: The subject to search for (e.g., "Kambaksha son of aurangzeb", "Eiffel Tower").
        max_results: Maximum number of search results to inspect.
        
    Returns:
        The absolute path to the downloaded image, or an error string if not found.
    """
    if not query or not query.strip():
        return "Error: Empty query provided for reference search."

    search_query = query.strip()
    print(f"Reference Search (DDGS): Searching images for '{search_query}'...")

    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(search_query, max_results=max_results))
    except Exception as e:
        return f"Error querying DDGS images for '{search_query}': {str(e)}"

    if not results:
        return f"Error: No image results found for query '{search_query}'"

    print(f"Reference Search: Found {len(results)} candidate results from DuckDuckGo.")

    clean_query = re.sub(r"[^\w\-_\. ]", "_", search_query).strip()[:40].replace(" ", "_")
    timestamp = int(time.time())

    for idx, r in enumerate(results):
        title = r.get("title", "Untitled")
        urls_to_try = []

        # Try high-resolution full image first, then fallback to thumbnail
        if r.get("image"):
            urls_to_try.append(("high_res", r.get("image")))
        if r.get("thumbnail"):
            urls_to_try.append(("thumbnail", r.get("thumbnail")))

        for source_type, img_url in urls_to_try:
            try:
                req = urllib.request.Request(img_url, headers=DEFAULT_HEADERS)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    raw_bytes = resp.read()

                # Basic sanity check on size
                if len(raw_bytes) < 1000:
                    continue

                # Validate image data using Pillow
                img = Image.open(io.BytesIO(raw_bytes))
                img_format = (img.format or "JPEG").lower()
                ext = ".jpg" if img_format == "jpeg" else f".{img_format}"

                filename = f"reference_{timestamp}_{clean_query}_{idx}{ext}"
                output_path = os.path.join(utils.GLOBAL_OUTPUT_DIR, filename) if utils.GLOBAL_OUTPUT_DIR else filename

                with open(output_path, "wb") as f:
                    f.write(raw_bytes)

                print(
                    f"Reference Search: Downloaded candidate #{idx} ({source_type}) "
                    f"'{title[:60]}' -> {output_path} ({img.size[0]}x{img.size[1]})"
                )
                return os.path.abspath(output_path)

            except Exception as e:
                # Silently try next url / candidate
                continue

    return f"Error: Could not successfully download any valid image candidate for '{search_query}'"
