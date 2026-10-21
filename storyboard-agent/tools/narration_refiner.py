"""
Narration Refinement Tool for Storyboard Multi-Agent Service.
Calibrates spoken narration duration to match video length:
- Video plays while narrator speaks.
- Aim to finish narration ~1 to 2 seconds after drawing completes, so the viewer can
  see the final static image while the narrator finishes the thought.
- In Strict Verbatim Mode, refinement is bypassed to preserve exact words.
"""
import os
from typing import Optional, Dict, Any
from google.genai import types

from config import MODEL_NAME, get_generation_config
from cost_tracker import tracker
from . import utils


def refine_narration_tool_fn(
    original_narration: str,
    image_path: Optional[str] = None,
    video_duration: Optional[float] = None,
    global_plan: Optional[Dict[str, Any]] = None,
    language: str = "english"
) -> str:
    """
    Enhances the Director's narration beat — preserving ALL core information.
    Calibrates word count so spoken audio naturally matches the video duration
    plus 1 to 2 seconds of hold on the final static frame.
    """
    if not utils.client or not original_narration or not original_narration.strip():
        return original_narration

    if not video_duration or video_duration <= 0:
        return original_narration

    # Average human speaking pace: ~140-150 words per minute (2.3 - 2.5 words/sec)
    # We aim for audio to last (video_duration + 1.0 to 2.0s) so audio finishes on the static held frame
    target_audio_duration = video_duration + 1.5
    target_words = max(8, int((target_audio_duration / 60.0) * 140))
    min_words = max(6, int(target_words * 0.85))
    max_words = int(target_words * 1.15)

    original_words = len(original_narration.split())

    # If original narration is already within target word range, make minimal changes
    if min_words <= original_words <= max_words:
        duration_guidance = (
            f"PACING CONSTRAINT:\n"
            f"- Video animation duration is {video_duration:.1f} seconds.\n"
            f"- Current narration is {original_words} words, which is well-calibrated.\n"
            f"- Polish phrasing for natural spoken human delivery, but do NOT lengthen or shorten significantly."
        )
    elif original_words < min_words:
        duration_guidance = (
            f"PACING CONSTRAINT:\n"
            f"- Video animation duration is {video_duration:.1f} seconds.\n"
            f"- Current narration is too short ({original_words} words). At ~140 words/min, expand storytelling details "
            f"to approximately {min_words}-{max_words} words so narration naturally covers the full drawing and finishes "
            f"1-2 seconds after the drawing completes on the static final image."
        )
    else:
        duration_guidance = (
            f"PACING CONSTRAINT:\n"
            f"- Video animation duration is {video_duration:.1f} seconds.\n"
            f"- Current narration is {original_words} words. Tighten slightly to approximately {min_words}-{max_words} words "
            f"while preserving all facts, names, and key details."
        )

    persona = global_plan.get("narrative_persona", "Professional Storyteller") if global_plan else "Professional Storyteller"
    tone = global_plan.get("tone", "dramatic") if global_plan else "dramatic"
    scene_context = global_plan.get("scene_context", "") if global_plan else ""
    scene_context_block = f"\nHISTORICAL SCENE CONTEXT & BACKGROUND:\n\"{scene_context}\"\n" if scene_context else ""

    prompt = f"""
You are an expert Documentary Script Writer and Narration Enhancer working under the direction of a {persona}.

DIRECTOR'S ORIGINAL BEAT SCRIPT (PRESERVE ALL FACTS, NAMES, AND CORE MEANING):
"{original_narration}"
{scene_context_block}
Tone: {tone}
Language: {language}
{duration_guidance}

ABSOLUTE RULES:
1. PRESERVE ALL INFORMATION: Keep every fact, name, number, and historical detail intact.
2. DO NOT DESCRIBE THE DRAWING: Never say "the drawing shows", "we see on whiteboard", etc. Speak directly to the listener about the story.
3. SPOKEN DELIVERY: Write for the human voice. Use natural rhythm, evocative words, and dynamic emotional pacing.
4. NATURAL PACING TAGS: You may insert '<short pause>' where a natural human speaker would pause to take a breath or emphasize a dramatic point.
5. TARGET WORD COUNT: Aim strictly for {min_words} to {max_words} words.
6. OUTPUT: Return ONLY the refined narration text in {language}. No meta-explanations, no markdown tags.
"""

    contents = [prompt]
    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            mime_type = "image/png"
            if image_path.lower().endswith((".jpg", ".jpeg")):
                mime_type = "image/jpeg"
            contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
        except Exception:
            pass

    try:
        config = get_generation_config(model=MODEL_NAME)
        response = utils.client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=config
        )
        tracker.record_usage_metadata(response.usage_metadata)
        refined = response.text.strip()
        if refined.startswith('"') and refined.endswith('"'):
            refined = refined[1:-1]
        print(f"Narration Refiner: Refined from {original_words} words to {len(refined.split())} words for {video_duration:.1f}s video.")
        return refined
    except Exception as e:
        print(f"Warning: Narration refinement failed ({e}). Using original.")
        return original_narration
