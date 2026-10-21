"""
ADK Tools Common Module.
Provides lightweight, token-optimized tools for all ADK agents.
Maintains all file paths and metadata in ADK Session State (tool_context.state),
returning only concise JSON/string summaries to agents.
"""
import json
import os
import re
import time
import urllib.parse
import urllib.request
from typing import List, Dict, Any, Optional
from google.adk import Context
from google.genai import types

from config import (
    MODEL_NAME,
    IMAGE_GEN_MODEL,
    TTS_MODEL,
    DEFAULT_TTS_VOICE,
    VEO_MODEL,
    VEO_RESOLUTION,
    ENABLE_VEO,
    ENABLE_SAM,
    OUTPUT_WIDTH,
    OUTPUT_HEIGHT,
    get_generation_config
)
from cost_tracker import tracker
from tools import utils
from tools.reference_evaluator import search_and_download_reference, evaluate_reference_image
from tools.image_gen_tool import image_gen_tool_fn
from tools.grid_panel_tool import slice_3x2_grid_into_panels
from tools.segmentation_tool import segmentation_tool_fn
from tools.draw_animation import draw_animation_tool_fn
from tools.media_tools import (
    generate_video_veo_tool_fn,
    merge_audio_video_tool_fn,
    transcribe_and_subtitle_tool_fn,
    concatenate_scene_videos_tool_fn
)
from tools.tts_tool import generate_tts_audio_tool_fn
from tools.narration_refiner import refine_narration_tool_fn
from tools.story_planner_tool import partition_words_into_beats


# ---------------------------------------------------------------------------
# 1. Director Tools (Web Grounding, URL Fetcher, Master Concatenation)
# ---------------------------------------------------------------------------

def google_search_grounding_tool(tool_context: Context, query: str) -> str:
    """
    Performs real-time Google Search grounding using Gemini 3.8 Flash to gather factual historical context.
    """
    if not utils.client or not query or not query.strip():
        return "Search failed: missing client or empty query."

    print(f"[DirectorAgent Tool] Grounding search for: '{query}'...")
    try:
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.3
        )
        prompt = f"Provide verified, detailed historical facts, key figures, dates, and dramatic conflict summary for: {query}"
        resp = utils.client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=config
        )
        tracker.record_usage_metadata(resp.usage_metadata)
        summary = resp.text.strip() if resp.text else "No content returned from grounding."

        # Save full detailed dossier to state (avoids bloating agent context tokens)
        if "research_dossier" not in tool_context.state:
            tool_context.state["research_dossier"] = []
        tool_context.state["research_dossier"].append({"query": query, "full_text": summary})

        concise_summary = summary[:800] + "..." if len(summary) > 800 else summary
        print(f"[DirectorAgent Tool] Grounding acquired ({len(summary)} chars). Saved to state['research_dossier'].")
        return f"Historical Research Dossier for '{query}' successfully acquired and saved to state ({len(summary)} chars).\nKey Overview:\n{concise_summary}"
    except Exception as e:
        print(f"[DirectorAgent Tool] Search grounding error: {e}")
        return f"Grounding search error: {e}"


def fetch_url_context_tool(tool_context: Context, url: str) -> str:
    """
    Fetches web page content from a specified URL to use as reference material.
    """
    if not url or not url.strip().startswith(("http://", "https://")):
        return "Invalid URL provided."

    print(f"[DirectorAgent Tool] Fetching URL context: {url}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "StoryboardAI-Director/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        # Strip html tags
        clean_text = re.sub(r"<[^>]+>", " ", html)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()[:3000]

        if "url_contexts" not in tool_context.state:
            tool_context.state["url_contexts"] = {}
        tool_context.state["url_contexts"][url] = clean_text[:500]

        return f"Successfully fetched {len(clean_text)} characters from {url}."
    except Exception as e:
        return f"Failed to fetch URL {url}: {e}"


def concatenate_master_movie_tool(tool_context: Context, part_video_paths_json: str = "") -> str:
    """
    Concatenates all completed Part videos into the final movie: movie_master.mp4.
    Accepts JSON list of video file paths or defaults to completed_part_videos in state.
    """
    try:
        paths = json.loads(part_video_paths_json) if isinstance(part_video_paths_json, str) and part_video_paths_json.strip() else []
    except Exception:
        paths = []

    if not paths:
        paths = tool_context.state.get("completed_part_videos", [])

    if not paths:
        paths = []
        for k in range(1, 20):
            p = tool_context.state.get(f"part_{k}_master_video")
            if p and os.path.exists(p):
                paths.append(p)

    if not paths:
        return "Error: No part video paths found in state or arguments for final movie assembly."

    import datetime
    topic = tool_context.state.get("story_topic", "storyboard_movie")
    now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    clean_topic = utils.slugify_title(topic, max_len=35)
    movie_filename = f"{now_str}-{clean_topic}-master.mp4"
    print(f"[DirectorAgent Tool] Assembling master movie from {len(paths)} Parts into {movie_filename}...")
    out = concatenate_scene_videos_tool_fn(paths, output_filename=movie_filename)
    if out and os.path.exists(out):
        dur = utils.get_video_duration(out)
        tool_context.state["final_movie_path"] = out
        tool_context.state["master_movie_path"] = out
        tool_context.state["final_movie_duration"] = dur
        return json.dumps({
            "status": "SUCCESS",
            "movie_path": out,
            "duration_seconds": round(dur, 2),
            "parts_count": len(paths)
        })
    return "Error assembling master movie."


# ---------------------------------------------------------------------------
# 2. Research & Continuity Tools (Reference Portraits)
# ---------------------------------------------------------------------------

def search_part_references_tool(tool_context: Context, entity_names_json: str, part_num: int = 1) -> str:
    """
    Acquires reference portraits for up to 2 historical entities via DuckDuckGo + Wikipedia API.
    Saves paths in state and returns a lightweight status summary.
    """
    try:
        entities = json.loads(entity_names_json) if isinstance(entity_names_json, str) else entity_names_json
    except Exception:
        entities = [entity_names_json]

    acquired_refs = []
    for ent in entities[:2]:
        p = search_and_download_reference(query=str(ent), max_results=5)
        if p and os.path.exists(p):
            acquired_refs.append(p)

    state_key = f"part_{part_num}_references"
    tool_context.state[state_key] = acquired_refs
    return json.dumps({
        "status": "SUCCESS",
        "part_number": part_num,
        "references_acquired": len(acquired_refs),
        "reference_paths": acquired_refs
    })


# ---------------------------------------------------------------------------
# 3. Comic Artist Tools (3x2 Grid Generation & CV Slicing)
# ---------------------------------------------------------------------------

def generate_part_grid_tool(
    tool_context: Context,
    part_num: int,
    panels_json: str,
    reference_paths_json: Optional[str] = None,
    story_theme: str = "",
    visual_style_prompt: str = ""
) -> str:
    """
    Generates a 3x2 Comic Grid Image ($0.035 call) using Gemini 3.1 Flash Image.
    Enforces negative space gaps between objects and rich in-scene comic typography.
    """
    try:
        panels = json.loads(panels_json) if isinstance(panels_json, str) else panels_json
    except Exception:
        panels = []

    refs = []
    if reference_paths_json:
        try:
            refs = json.loads(reference_paths_json)
        except Exception:
            refs = []
    if not refs:
        refs = tool_context.state.get(f"part_{part_num}_references", [])

    ref_instructions = ""
    if refs:
        ref_instructions = (
            "CRITICAL CHARACTER CONTINUITY: The attached reference portraits depict key historical figures. "
            "Maintain their facial features, royal turbans, armor, and costumes consistently across all 6 panels.\n"
        )

    opening_banner_instruction = ""
    if part_num == 1:
        topic_title = tool_context.state.get("story_topic", "")
        clean_title_display = topic_title[:50] if topic_title else "THE HISTORICAL CHRONICLE"
        opening_banner_instruction = (
            f"- GRAND OPENING TITLE BANNER (PANEL 1 ONLY): Panel 1 (top-left) is the grand cinematic title intro for the entire movie. "
            f"It MUST prominently feature a majestic, ornate title banner, marquee, or royal parchment scroll with bold, elegant comic lettering "
            f"displaying: '{clean_title_display.upper()}'. Setting up the epic historical drama.\n"
        )
    panels_desc_str = ""
    positions = ["top-left", "top-center", "top-right", "bottom-left", "bottom-center", "bottom-right"]
    for idx, p in enumerate(panels[:6]):
        pos = positions[idx]
        desc = p.get("visual_description", "") if isinstance(p, dict) else str(p)
        panels_desc_str += f"Grid cell {idx + 1} ({pos}): {desc}\n"

    theme_directive = f"OVERARCHING THEME & ERA CONTEXT: {story_theme}\n" if story_theme else ""
    style_directive = f"CUSTOM ART STYLE DIRECTIVE: {visual_style_prompt}\n" if visual_style_prompt else ""

    grid_prompt = (
        f"A master 3x2 comic storyboard grid featuring exactly 6 clear rectangular story panels (3 columns x 2 rows) "
        f"separated by clean, wide white gutters on a pure white canvas.\n\n"
        f"{theme_directive}"
        f"{style_directive}"
        f"{ref_instructions}"
        f"COMPOSITION & SPACING GUIDELINES:\n"
        f"{opening_banner_instruction}"
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

    print(f"[ComicArtistAgent Tool] Generating 3x2 comic grid for Part #{part_num}...")
    grid_img_path = image_gen_tool_fn(prompt=grid_prompt, reference_image_paths=refs)

    if not grid_img_path or not os.path.exists(grid_img_path):
        return json.dumps({"status": "FAILED", "error": "Grid image generation failed."})

    tool_context.state[f"part_{part_num}_grid_image"] = grid_img_path
    return json.dumps({
        "status": "SUCCESS",
        "part_number": part_num,
        "grid_image_path": grid_img_path
    })


def slice_part_grid_tool(tool_context: Context, part_num: int, grid_image_path: Optional[str] = None) -> str:
    """
    Slices a 3x2 grid image into 6 panels framed natively to 720p (1280x720).
    """
    path = grid_image_path or tool_context.state.get(f"part_{part_num}_grid_image")
    if not path or not os.path.exists(path):
        return json.dumps({"status": "FAILED", "error": "Grid image not found."})

    print(f"[ComicArtistAgent Tool] Slicing Part #{part_num} grid into 6 720p panels...")
    panels = slice_3x2_grid_into_panels(
        grid_image_path=path,
        output_dir=utils.get_output_dir(),
        target_width=OUTPUT_WIDTH,
        target_height=OUTPUT_HEIGHT
    )

    if len(panels) < 6:
        return json.dumps({"status": "FAILED", "error": f"Slicing produced {len(panels)} panels (expected 6)."})

    tool_context.state[f"part_{part_num}_panel_images"] = panels
    return json.dumps({
        "status": "SUCCESS",
        "part_number": part_num,
        "panels_count": len(panels),
        "panel_paths": panels
    })


# ---------------------------------------------------------------------------
# 4. Whiteboard Animator & Cinematographer & Narrator (Sub-Scene Execution)
# ---------------------------------------------------------------------------

def produce_subscene_tool(
    tool_context: Context,
    part_num: int,
    subscene_num: int,
    panel_image_path: str,
    narration_beat: str,
    is_veo_candidate: bool = False,
    veo_prompt: Optional[str] = None,
    scene_context: str = "",
    emotional_tone: str = "dramatic",
    verbatim_mode: bool = False
) -> str:
    """
    Executes production of a single sub-scene (panel):
    1. SAM 3 segmentation on EVERY panel when ENABLE_SAM is True.
    2. Whiteboard drawing animation (1280x720, 25fps).
    3. Evaluates Veo duration gate: only adds Veo if audio duration >= drawing duration + 6s
       (or in Creative Mode, refines narration to match drawing + 8s).
    4. TTS audio generation + zero-crop AV merge + subtitle burning.
    5. Returns lightweight path summary.
    """
    print(f"\n[PartProductionAgent] Producing Part #{part_num} | Sub-Scene #{subscene_num}/6...")
    print(f"  Beat: '{narration_beat}' (Veo Candidate: {is_veo_candidate})")

    # Config overrides: strictly check config.py settings
    import config
    apply_sam = getattr(config, "ENABLE_SAM", True)
    apply_veo_config = getattr(config, "ENABLE_VEO", True)

    if not apply_veo_config and is_veo_candidate:
        print(f"  [Veo Config Gate]: ENABLE_VEO is False in config. Skipping Veo for Sub-Scene {subscene_num}.")
        is_veo_candidate = False

    # Step 1: SAM 3 Segmentation
    seg_json = None
    if apply_sam:
        print(f"  [SAM 3] Running object segmentation for Panel {subscene_num}...")
        seg_json = segmentation_tool_fn(panel_image_path)
    else:
        print(f"  [SAM 3 Disabled in config]: Single pass drawing for Panel {subscene_num}.")

    # Step 2: In Verbatim Mode, generate TTS first to measure speech duration
    audio_path = None
    a_dur = 0.0
    split_len = 16
    skip_rate = 12
    hold_sec = 2

    if verbatim_mode:
        print(f"  [Verbatim Mode] Generating audio first for sub-scene {subscene_num}...")
        audio_path = generate_tts_audio_tool_fn(text=narration_beat)
        a_dur = utils.get_audio_duration(audio_path) if audio_path else 5.0

        # Calibrate drawing speed to fit verbatim audio
        if a_dur >= 14.0:
            split_len, skip_rate, hold_sec = 16, 12, 2
        elif a_dur >= 9.0:
            split_len, skip_rate, hold_sec = 20, 16, 2
        elif a_dur >= 5.0:
            split_len, skip_rate, hold_sec = 24, 20, 2
        else:
            split_len, skip_rate, hold_sec = 30, 24, 1

    # Step 3: Render 720p Whiteboard Animation (Normalized H.264 @ 25fps)
    print(f"  Generating native 720p whiteboard drawing animation (split_len={split_len}, skip_rate={skip_rate})...")
    drawing_video = draw_animation_tool_fn(
        image_path=panel_image_path,
        segmentation_results_path=seg_json,
        resize_wd=OUTPUT_WIDTH,
        resize_ht=OUTPUT_HEIGHT,
        split_len=split_len,
        object_skip_rate=skip_rate,
        end_duration_sec=hold_sec
    )
    v_draw_dur = utils.get_video_duration(drawing_video) if drawing_video else 5.0

    # Step 4: Evaluate Veo Gate
    # Creative Mode: Visual is first! If Veo candidate, generate Veo & expand narration to match.
    # Verbatim Mode: Audio is first! Only add Veo if a_dur >= v_draw_dur + 6.0s!
    apply_veo = False
    if apply_veo_config and is_veo_candidate:
        if verbatim_mode:
            if a_dur >= (v_draw_dur + 6.0):
                apply_veo = True
                print(f"  [Veo Gate PASS in Verbatim Mode]: audio ({a_dur:.1f}s) >= drawing ({v_draw_dur:.1f}s) + 6.0s. Enabling Veo!")
            else:
                print(f"  [Veo Gate REJECT in Verbatim Mode]: audio ({a_dur:.1f}s) < drawing ({v_draw_dur:.1f}s) + 6.0s. Gating Veo OFF to prevent silence.")
        else:
            # Creative Mode: Veo is approved; narration refiner will calibrate text to match full visual duration
            apply_veo = True
            print(f"  [Veo Approved in Creative Mode]: Narration will expand to cover drawing + Veo.")

    veo_video = None
    if apply_veo:
        prompt_text = veo_prompt or narration_beat
        if scene_context:
            prompt_text = f"{prompt_text}. Scene Mood/Context: {scene_context}"
        print(f"  Generating native 720p Veo clip ({VEO_MODEL})...")
        veo_video = generate_video_veo_tool_fn(image_path=panel_image_path, prompt=prompt_text)

    # Combine drawing + Veo (lossless 720p)
    active_visual = drawing_video
    if veo_video and os.path.exists(veo_video):
        combined = concatenate_scene_videos_tool_fn(
            [drawing_video, veo_video],
            output_filename=f"sub_{part_num}_{subscene_num}_visual.mp4"
        )
        if combined:
            active_visual = combined

    total_visual_dur = utils.get_video_duration(active_visual) if active_visual else 5.0

    # Step 5: Narration & Audio
    final_text = narration_beat
    if not verbatim_mode:
        print(f"  Refining narration beat for {total_visual_dur:.1f}s visual duration...")
        final_text = refine_narration_tool_fn(
            original_narration=narration_beat,
            image_path=panel_image_path,
            video_duration=total_visual_dur,
            global_plan={"scene_context": scene_context, "tone": emotional_tone}
        )
        print(f"  Generating human-like TTS speech (Aoede)...")
        audio_path = generate_tts_audio_tool_fn(text=final_text)

    if not audio_path or not os.path.exists(audio_path):
        return json.dumps({"status": "FAILED", "error": "TTS audio failed."})

    # Step 6: Zero-Crop AV Merge + Subtitle Burning
    merged_path = merge_audio_video_tool_fn(video_path=active_visual, audio_path=audio_path)
    if not merged_path:
        return json.dumps({"status": "FAILED", "error": "AV merge failed."})

    subtitled_path = transcribe_and_subtitle_tool_fn(video_path=merged_path, audio_path=audio_path)
    final_subscene_video = subtitled_path or merged_path
    final_dur = utils.get_video_duration(final_subscene_video)

    # Save to state
    subscene_key = f"part_{part_num}_sub_{subscene_num}"
    tool_context.state[subscene_key] = final_subscene_video

    print(f"  [Sub-Scene #{subscene_num} SUCCESS] -> {os.path.basename(final_subscene_video)} ({final_dur:.1f}s)")
    return json.dumps({
        "status": "SUCCESS",
        "part_number": part_num,
        "subscene_number": subscene_num,
        "video_path": final_subscene_video,
        "duration_seconds": round(final_dur, 2),
        "veo_applied": bool(apply_veo and veo_video)
    })


def concatenate_part_subscenes_tool(tool_context: Context, part_num: int, subscene_video_paths_json: str) -> str:
    """
    Concatenates the 6 sub-scenes of a Part into part_{part_num}_master.mp4 (720p).
    """
    try:
        paths = json.loads(subscene_video_paths_json) if isinstance(subscene_video_paths_json, str) else subscene_video_paths_json
    except Exception:
        paths = [tool_context.state.get(f"part_{part_num}_sub_{k}") for k in range(1, 7)]

    valid_paths = [p for p in paths if p and os.path.exists(p)]
    if len(valid_paths) < 6:
        print(f"Warning: Only {len(valid_paths)} sub-scenes found for Part {part_num}.")

    out_file = f"part_{part_num}_master.mp4"
    part_master = concatenate_scene_videos_tool_fn(valid_paths, output_filename=out_file)

    if part_master and os.path.exists(part_master):
        dur = utils.get_video_duration(part_master)
        tool_context.state[f"part_{part_num}_master_video"] = part_master

        if "completed_part_videos" not in tool_context.state:
            tool_context.state["completed_part_videos"] = []
        if part_master not in tool_context.state["completed_part_videos"]:
            tool_context.state["completed_part_videos"].append(part_master)

        # Storage Cleanup: Purge individual sub-scene videos now that part master is confirmed
        if os.path.getsize(part_master) > 10000:
            for p in valid_paths:
                if p != part_master:
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                            print(f"  [Storage Cleanup] Purged sub-scene video: {os.path.basename(p)}")
                    except Exception:
                        pass

        return json.dumps({
            "status": "SUCCESS",
            "part_number": part_num,
            "part_master_path": part_master,
            "duration_seconds": round(dur, 2)
        })

    return json.dumps({"status": "FAILED", "error": f"Failed to assemble Part {part_num} video."})
