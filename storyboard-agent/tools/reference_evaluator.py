"""
Reference Search & Automated Vision Evaluator.
Searches DuckDuckGo Images (ddgs) and evaluates candidates with Gemini 3.8 Flash Vision.
Determines whether to use the authentic image directly or pass it as reference to Nano Banana.
"""
import io
import json
import os
import re
import time
import urllib.parse
import urllib.request
from typing import Optional
from ddgs import DDGS
from PIL import Image
from google.genai import types

from config import MODEL_NAME, get_generation_config
from cost_tracker import tracker
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


def _download_image_from_url(url: str, query: str, source_label: str) -> Optional[str]:
    """Helper to download and validate an image from a direct URL."""
    clean_filename_base = re.sub(r"[^\w\-_\. ]", "_", query).strip()[:40].replace(" ", "_")
    timestamp = int(time.time())
    try:
        req = urllib.request.Request(url, headers=DEFAULT_HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw_bytes = resp.read()

        if len(raw_bytes) < 1500:
            return None

        img = Image.open(io.BytesIO(raw_bytes))
        fmt = (img.format or "JPEG").lower()
        ext = ".jpg" if fmt == "jpeg" else f".{fmt}"

        filename = f"ref_{timestamp}_{clean_filename_base}_{source_label}{ext}"
        out_dir = utils.get_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)

        with open(out_path, "wb") as f:
            f.write(raw_bytes)

        print(f"Reference Search: Downloaded candidate ({source_label}) -> {out_path} ({img.size[0]}x{img.size[1]})")
        return os.path.abspath(out_path)
    except Exception as e:
        print(f"Reference Download ({source_label}) Error: {e}")
        return None


def _search_wikipedia_commons_reference(query: str) -> Optional[str]:
    """Fallback reference image search using Wikipedia REST API & Wikimedia Commons."""
    clean_q = query.strip()
    print(f"Reference Search: Trying Wikipedia / Wikimedia Commons fallback for '{clean_q}'...")

    # Step 1: Wikipedia Page Image (pithumbsize=1200)
    try:
        url = f"https://en.wikipedia.org/w/api.php?action=query&titles={urllib.parse.quote(clean_q)}&prop=pageimages&format=json&pithumbsize=1200"
        req = urllib.request.Request(url, headers={"User-Agent": "StoryboardAI/1.0 (historical-storyboard-research)"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        for _, page_info in pages.items():
            if "thumbnail" in page_info and page_info["thumbnail"].get("source"):
                img_url = page_info["thumbnail"]["source"]
                res = _download_image_from_url(img_url, clean_q, "wikipedia")
                if res:
                    return res
    except Exception as e:
        print(f"Wikipedia Page Image lookup failed ({e}). Trying Commons search...")

    # Step 2: Wikimedia Commons Generator Search
    try:
        commons_url = f"https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch={urllib.parse.quote(clean_q)}&gsrlimit=5&prop=imageinfo&iiprop=url|mime&format=json"
        req = urllib.request.Request(commons_url, headers={"User-Agent": "StoryboardAI/1.0 (historical-storyboard-research)"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        for _, p_data in pages.items():
            img_info = p_data.get("imageinfo", [])
            if img_info and img_info[0].get("url"):
                img_url = img_info[0]["url"]
                if img_url.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    res = _download_image_from_url(img_url, clean_q, "commons")
                    if res:
                        return res
    except Exception as e:
        print(f"Wikimedia Commons search failed ({e}).")

    return None


def search_and_download_reference(query: str, max_results: int = 10) -> Optional[str]:
    """
    Searches DuckDuckGo Images for query and downloads the best valid image candidate.
    Falls back to Wikipedia / Wikimedia Commons if DDGS encounters network/header errors.
    """
    if not query or not query.strip():
        return None

    clean_q = query.strip()
    tracker.record_search(1)
    print(f"Reference Search: Querying DDGS for '{clean_q}'...")

    results = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(clean_q, max_results=max_results))
    except Exception as e:
        print(f"Reference Search (DDGS) Error: {e}. Switching to Wikipedia / Commons fallback...")

    if not results:
        return _search_wikipedia_commons_reference(clean_q)

    clean_filename_base = re.sub(r"[^\w\-_\. ]", "_", clean_q).strip()[:40].replace(" ", "_")
    timestamp = int(time.time())

    for idx, r in enumerate(results):
        urls_to_try = []
        if r.get("image"):
            urls_to_try.append(("high_res", r.get("image")))
        if r.get("thumbnail"):
            urls_to_try.append(("thumbnail", r.get("thumbnail")))

        for src_type, url in urls_to_try:
            res = _download_image_from_url(url, clean_q, f"ddgs_{idx}_{src_type}")
            if res:
                return res

    # If all DDGS candidate downloads failed, fallback to Wikipedia / Commons
    return _search_wikipedia_commons_reference(clean_q)


def evaluate_reference_image(image_path: str, query: str, scene_context: str = "") -> dict:
    """
    Evaluates reference image using Gemini 3.8 Flash Vision.
    Scores relevance, resolution, authenticity, and visual quality.
    Decides if it is suitable for direct animation (score >= 0.8).
    """
    default_result = {
        "image_path": image_path,
        "use_direct": False,
        "score": 0.0,
        "reasoning": "Gemini client unavailable or evaluation skipped."
    }

    if not utils.client or not image_path or not os.path.exists(image_path):
        return default_result

    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        mime_type = "image/png"
        if image_path.lower().endswith((".jpg", ".jpeg")):
            mime_type = "image/jpeg"
        elif image_path.lower().endswith(".webp"):
            mime_type = "image/webp"

        prompt = f"""
You are an expert art director evaluating an image downloaded for a documentary storyboard scene.
Subject / Query: "{query}"
Scene context: "{scene_context}"

Evaluate whether this image is suitable to be used DIRECTLY as the primary visual in the storyboard (score >= 0.8), or if it is low quality / watermark-heavy / partially relevant and should instead be passed as a reference to an AI illustrator to redraw (score < 0.8).

Criteria:
1. Relevance: Accurately depicts the historical figure, location, or subject.
2. Quality & Authenticity: Clear resolution, authentic portrait/photo/artwork, not covered in intrusive text watermarks.
3. Framing: Well-composed subject suitable for whiteboard panning and animation.

Output strictly valid JSON with keys:
{{
  "score": float (between 0.0 and 1.0),
  "use_direct": bool (true if score >= 0.8, else false),
  "reasoning": "Short 1-2 sentence explanation"
}}
"""
        config = get_generation_config(
            model=MODEL_NAME,
            response_mime_type="application/json"
        )
        response = utils.client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, types.Part.from_bytes(data=img_bytes, mime_type=mime_type)],
            config=config
        )
        tracker.record_usage_metadata(response.usage_metadata)
        data = json.loads(response.text.strip())
        data["image_path"] = image_path
        print(f"Vision Evaluation for '{query}': Score={data.get('score')} | Direct={data.get('use_direct')} | {data.get('reasoning')}")
        return data

    except Exception as e:
        print(f"Warning: Vision evaluation failed ({e}). Defaulting to use_direct=False.")
        default_result["reasoning"] = str(e)
        return default_result


def reference_evaluator_tool_fn(query: str, scene_context: str = "") -> dict:
    """
    Unified tool: searches DDGS and evaluates image.
    Returns:
        dict: {"image_path": str or None, "use_direct": bool, "score": float, "reasoning": str}
    """
    image_path = search_and_download_reference(query)
    if not image_path:
        return {
            "image_path": None,
            "use_direct": False,
            "score": 0.0,
            "reasoning": f"No image found on DDGS for query: '{query}'"
        }

    return evaluate_reference_image(image_path, query, scene_context)


def fetch_multi_entity_references(
    search_queries: list,
    scene_context: str = "",
    max_refs: int = 2
) -> list:
    """
    Searches and downloads reference images for multiple distinct entities (max 2).
    Evaluates each candidate with Gemini 3.8 Flash Vision.
    Returns list of local image paths for valid reference candidates.
    """
    if not search_queries:
        return []

    clean_queries = [q.strip() for q in search_queries if q and q.strip()][:max_refs]
    print(f"Multi-Entity Reference Search: Fetching references for {len(clean_queries)} entities: {clean_queries}")

    acquired_refs = []
    for q in clean_queries:
        print(f"  Searching entity: '{q}'...")
        image_path = search_and_download_reference(q, max_results=6)
        if not image_path or not os.path.exists(image_path):
            print(f"  No image candidate downloaded for entity: '{q}'")
            continue

        eval_res = evaluate_reference_image(image_path, q, scene_context)
        score = eval_res.get("score", 0.0)
        print(f"  Entity '{q}' evaluated: score={score} (path: {image_path})")

        # Keep candidate if score is reasonable or image exists as best effort
        acquired_refs.append(image_path)

    print(f"Multi-Entity Reference Search: Acquired {len(acquired_refs)} reference images: {acquired_refs}")
    return acquired_refs
