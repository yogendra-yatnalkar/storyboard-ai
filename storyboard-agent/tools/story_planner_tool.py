"""
Dual-Engine Story Planner Tool.
Supports:
1. Creative Mode (Default): Full documentary narrative planning, tone, and scene breakdown.
2. Verbatim Mode (Activated ONLY when requested): Chunks source text word-for-word with 0 alterations,
   while still using Gemini 3.8 Flash to generate rich visual prompts, compositions, and search queries.
"""
import json
import re
from typing import List, Dict, Any, Tuple, Optional
from google.genai import types

from config import MODEL_NAME, get_generation_config
from cost_tracker import tracker
from state import ScenePlan, PanelPlan, RunConfig
from . import utils


def _split_into_sentences(text: str) -> List[str]:
    """Splits text into sentences based on punctuation, preserving original characters."""
    # Split by period, exclamation, question mark followed by whitespace or quotes
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences if sentences else [text.strip()]


def partition_words_into_beats(source_text: str, target_panels: int = 6, weights: Optional[List[float]] = None) -> List[str]:
    """
    Partitions verbatim source text into target_panels (default 6) along clause/sentence boundaries.
    Preserves 100% of the exact original words in strict order without additions or omissions.
    """
    words = source_text.strip().split()
    total_words = len(words)
    if total_words == 0:
        return [""] * target_panels

    if total_words <= target_panels:
        beats = words + [""] * (target_panels - len(words))
        return beats[:target_panels]

    if not weights or len(weights) != target_panels:
        weights = [1.0] * target_panels

    total_weight = sum(weights)
    target_counts = [max(1, int(round((w / total_weight) * total_words))) for w in weights]
    diff = total_words - sum(target_counts)
    target_counts[-1] += diff

    # Find punctuation split points (. ! ? ; : — ,)
    clause_ends = set()
    for idx, w in enumerate(words):
        if any(w.endswith(punct) for punct in ['.', '!', '?', ';', ':', '—', ',']):
            clause_ends.add(idx)

    # Greedily partition words close to target_counts while snapping to clause_ends when near
    beats = []
    curr_idx = 0
    for p in range(target_panels):
        if p == target_panels - 1:
            beats.append(" ".join(words[curr_idx:]))
            break

        target_len = target_counts[p]
        ideal_end = curr_idx + target_len - 1

        best_end = ideal_end
        min_dist = 999
        max_allowed_end = total_words - (target_panels - p)
        min_allowed_end = curr_idx

        for cand in clause_ends:
            if min_allowed_end <= cand <= max_allowed_end:
                dist = abs(cand - ideal_end)
                if dist < min_dist and dist <= 3:
                    min_dist = dist
                    best_end = cand

        best_end = max(min_allowed_end, min(best_end, max_allowed_end))
        beat_slice = words[curr_idx : best_end + 1]
        beats.append(" ".join(beat_slice))
        curr_idx = best_end + 1

    # Verify 100% word integrity
    reconstructed = " ".join([b for b in beats if b]).split()
    if reconstructed != words:
        # Fallback to direct slice to guarantee integrity
        beats = []
        w_idx = 0
        for p in range(target_panels):
            if p == target_panels - 1:
                beats.append(" ".join(words[w_idx:]))
                break
            w_count = target_counts[p]
            beats.append(" ".join(words[w_idx : w_idx + w_count]))
            w_idx += w_count

    return beats


def reconcile_verbatim_beats(source_text: str, proposed_beats: List[str], target_panels: int = 6) -> List[str]:
    """
    Reconciles LLM-proposed beats against original source text.
    Ensures 100.0% exact word match while respecting the LLM's semantic division.
    """
    if not proposed_beats:
        return partition_words_into_beats(source_text, target_panels=target_panels)

    beat_word_lens = [len(b.split()) for b in proposed_beats if b.strip()]
    while len(beat_word_lens) < target_panels:
        beat_word_lens.append(1)
    beat_word_lens = beat_word_lens[:target_panels]

    weights = [float(l) if l > 0 else 1.0 for l in beat_word_lens]
    return partition_words_into_beats(source_text, target_panels=target_panels, weights=weights)


def _chunk_verbatim_text(text: str, target_scenes: int = None) -> List[str]:
    """
    Chunks text strictly verbatim across scenes.
    Does NOT alter, summarize, or rephrase any words.
    """
    sentences = _split_into_sentences(text)
    total_sentences = len(sentences)

    if not target_scenes or target_scenes <= 0:
        # Default ~2 sentences per scene, min 1, max 10
        target_scenes = max(1, min(10, (total_sentences + 1) // 2))

    target_scenes = min(target_scenes, total_sentences)
    chunk_size = max(1, total_sentences // target_scenes)
    remainder = total_sentences % target_scenes

    chunks = []
    idx = 0
    for i in range(target_scenes):
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        chunk_sentences = sentences[idx : idx + current_chunk_size]
        idx += current_chunk_size
        if chunk_sentences:
            chunks.append(" ".join(chunk_sentences))

    # Append any leftovers to the last chunk
    if idx < total_sentences:
        chunks[-1] += " " + " ".join(sentences[idx:])

    return chunks


def plan_verbatim_storyboard(source_text: str, run_config: RunConfig) -> Tuple[Dict[str, Any], List[ScenePlan]]:
    """
    Verbatim Mode: Locks narration to original text chunks word-for-word.
    Decomposes each scene into 6 sequential verbatim beats and designs 6 matching 3x2 grid panels.
    Extracts up to 2 canonical entity search queries per scene.
    """
    print("Story Planner: Running in STRICT VERBATIM MODE.")
    chunks = _chunk_verbatim_text(source_text, target_scenes=run_config.max_scenes)
    total_scenes = len(chunks)
    print(f"Story Planner: Chunked source text into {total_scenes} verbatim scenes.")

    global_plan = {
        "title": "Verbatim Storyboard Presentation",
        "mode": "verbatim",
        "tone": "authentic",
        "narrative_persona": "Original Text Narrator",
        "visual_style": "6-Panel Whiteboard Animation Grid (3x2)",
        "pacing": "steady",
        "total_scenes": total_scenes,
        "language": run_config.language
    }

    chunks_payload = []
    for i, chunk in enumerate(chunks):
        word_count = len(chunk.split())
        can_veo = run_config.enable_veo and (word_count >= 35)
        if run_config.enable_veo and not can_veo:
            print(f"Story Planner [Verbatim]: Scene #{i + 1} has concise text ({word_count} words < 35). Gating Veo to avoid duration mismatch.")
        chunks_payload.append({
            "scene_number": i + 1,
            "exact_narration": chunk,
            "word_count": word_count,
            "enable_veo": can_veo
        })

    veo_field = ""
    if run_config.enable_veo and run_config.veo_direction:
        veo_field = '\n          "veo_prompt": "Optional 8s action prompt for climax panel (only if scene has >=35 words)",'

    prompt = f"""
You are the Visual Director for a 6-panel whiteboard comic storyboard video.
The narration script is STRICTLY LOCKED and MUST NOT BE ALTERED.

Here are the locked scene narration chunks:
{json.dumps(chunks_payload, indent=2)}

For each scene:
1. Divide the exact narration chunk into exactly 6 sequential narration beats (preserving 100% of the exact original words in strict order without altering, summarizing, or omitting any words).
   PROPORTIONAL PACING RULE: Distribute the exact source words along natural punctuation boundaries (periods, commas, semicolons, conjunctions). If a panel has a longer climax or Veo motion, allocate a larger clause (more words) to that panel; if simpler, allocate a shorter clause. Ensure every beat has a natural, comfortable spoken cadence.
2. 'search_queries': List of up to 2 clean, generic names of historical figures, monuments, or key entities in this scene (e.g. ["Prince Kam Bakhsh son of Aurangzeb", "Bahadur Shah I Mughal emperor"]). STRICTLY NO story actions, verbs, or phrases like "fierce battle between".
3. 'panels': Exactly 6 panels corresponding to the 6 beats of a 3x2 grid. For each panel:
   - 'panel_number': 1 to 6
   - 'narration_beat': the exact original text fragment
   - 'visual_description': vibrant graphic novel storyboard illustration (crisp black ink outlines, clean watercolor/flat color fills, white background){veo_field}

Output strictly valid JSON with this structure:
[
  {{
    "scene_number": 1,
    "description": "High-level visual summary of this scene",
    "visual_setup": "Artistic composition overview",
    "search_queries": ["Entity 1", "Entity 2"],
    "panels": [
      {{
        "panel_number": 1,
        "narration_beat": "exact words...",
        "visual_description": "Clean line art outline of..."{veo_field}
      }},
      ... (exactly 6 panels)
    ]
  }},
  ...
]
"""
    try:
        config = get_generation_config(model=MODEL_NAME, response_mime_type="application/json")
        response = utils.client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=config
        )
        tracker.record_usage_metadata(response.usage_metadata)
        visuals_list = json.loads(response.text.strip())
        visuals_by_scene = {v["scene_number"]: v for v in visuals_list if "scene_number" in v}
    except Exception as e:
        print(f"Warning: Failed to generate visual plan for verbatim chunks ({e}). Using sensible defaults.")
        visuals_by_scene = {}

    scene_plans = []
    for i, chunk in enumerate(chunks):
        s_num = i + 1
        vis = visuals_by_scene.get(s_num, {})
        raw_panels = vis.get("panels", [])

        # Build 6 PanelPlan items
        panels = []
        if raw_panels:
            for p in raw_panels[:6]:
                panels.append(
                    PanelPlan(
                        panel_number=p.get("panel_number", len(panels) + 1),
                        narration_beat=p.get("narration_beat", ""),
                        visual_description=p.get("visual_description", "Clean minimalist line art outline"),
                        veo_prompt=p.get("veo_prompt")
                    )
                )
        # Fallback if fewer than 6 panels returned
        while len(panels) < 6:
            p_num = len(panels) + 1
            panels.append(
                PanelPlan(
                    panel_number=p_num,
                    narration_beat="",
                    visual_description=f"Minimalist whiteboard drawing beat {p_num}"
                )
            )

        # Reconcile beats with 100% exact verbatim words and proportional clause allocation
        proposed_beats = [p.narration_beat for p in panels]
        reconciled_beats = reconcile_verbatim_beats(source_text=chunk, proposed_beats=proposed_beats, target_panels=6)
        for idx_b, rec_beat in enumerate(reconciled_beats):
            if idx_b < len(panels):
                panels[idx_b].narration_beat = rec_beat

        # Veo Gating (Strategy 4): If chunk is concise (<35 words), gate Veo OFF completely.
        # Also gate Veo OFF for any individual beat with < 12 words.
        chunk_words = len(chunk.split())
        for p in panels:
            if chunk_words < 35 or not run_config.enable_veo or len(p.narration_beat.split()) < 12:
                p.veo_prompt = None

        search_q_list = vis.get("search_queries", [])
        if not search_q_list and vis.get("search_query"):
            search_q_list = [vis.get("search_query")]

        scene_plans.append(
            ScenePlan(
                scene_number=s_num,
                narration=chunk,  # 100% exact verbatim text
                description=vis.get("description", f"Visual illustration for scene {s_num}"),
                visual_setup=vis.get("visual_setup", "Clear 3x2 whiteboard storyboard grid, pure white background."),
                search_queries=search_q_list[:2],
                panels=panels,
                veo_prompt=panels[-1].veo_prompt if panels else None
            )
        )

    return global_plan, scene_plans


def plan_creative_storyboard(user_instruction: str, research_material: str, run_config: RunConfig) -> Tuple[Dict[str, Any], List[ScenePlan]]:
    """
    Creative Mode (Default): Generates an original documentary/dramatic storyboard plan.
    Each scene contains exactly 6 sequential beats (3x2 grid) with up to 2 generic entity queries.
    """
    print("Story Planner: Running in CREATIVE MODE (6-Beat 3x2 Grid).")

    veo_field = ""
    if run_config.enable_veo and run_config.veo_direction:
        veo_field = '\n          "veo_prompt": "Descriptive 8s action prompt for Veo 720p video",'

    scenes_limit_str = f"Plan strictly {run_config.max_scenes} scenes." if run_config.max_scenes else "Plan between 2 to 4 scenes."

    prompt = f"""
You are an expert Film Director and Documentary Screenwriter.
User Topic / Instructions:
"{user_instruction}"

Source / Research Material:
"{research_material or 'None provided. Use verified historical and factual knowledge.'}"

Directives:
1. {scenes_limit_str}
2. Language: The narration and beat values MUST be written in {run_config.language}.
3. Structure: Each scene is visualized as a 3x2 comic storyboard grid with EXACTLY 6 sequential panels (Panels 1-6).
4. Generic Search Queries: 'search_queries' MUST be a list of up to 2 canonical, generic entity names (e.g. ["Prince Kam Bakhsh son of Aurangzeb", "Bahadur Shah I Mughal emperor"]). NEVER include action descriptions, verbs, or words like "battle between".
5. Panel Beats: Each panel must have:
   - 'panel_number': 1 to 6
   - 'narration_beat': 1-2 concise sentences narrated during this panel
   - 'visual_description': Clean coloring-book line art description (bold black outlines, white background, zero cross-hatching or shading){veo_field}

Output strictly valid JSON with this structure:
{{
  "global_plan": {{
    "title": "...",
    "tone": "informative" | "dramatic" | "inspirational",
    "narrative_persona": "...",
    "visual_style": "6-Panel Whiteboard Comic Grid (3x2)",
    "pacing": "steady",
    "total_scenes": int
  }},
  "scenes": [
    {{
      "scene_number": 1,
      "narration": "Full narration text combining the 6 beats",
      "description": "High-level summary of the scene",
      "visual_setup": "3x2 comic grid layout with bold black whiteboard marker contours",
      "search_queries": ["Entity 1", "Entity 2"],
      "panels": [
        {{
          "panel_number": 1,
          "narration_beat": "Beat 1 narration...",
          "visual_description": "Panel 1 visual description..."{veo_field}
        }},
        ... (exactly 6 panels)
      ]
    }},
    ...
  ]
}}
"""
    try:
        config = get_generation_config(model=MODEL_NAME, response_mime_type="application/json")
        response = utils.client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=config
        )
        tracker.record_usage_metadata(response.usage_metadata)
        data = json.loads(response.text.strip())
        global_plan = data.get("global_plan", {})
        raw_scenes = data.get("scenes", [])
    except Exception as e:
        print(f"Error in creative storyboard planning: {e}")
        global_plan = {"title": "Storyboard", "total_scenes": 1}
        raw_scenes = [{
            "scene_number": 1,
            "narration": user_instruction,
            "description": user_instruction,
            "visual_setup": "3x2 Whiteboard grid.",
            "search_queries": [],
            "panels": [
                {
                    "panel_number": i,
                    "narration_beat": f"Beat {i} for {user_instruction[:40]}",
                    "visual_description": f"Panel {i} clean line art outline"
                } for i in range(1, 7)
            ]
        }]

    scene_plans = []
    for s in raw_scenes:
        s_num = s.get("scene_number", len(scene_plans) + 1)
        raw_panels = s.get("panels", [])

        panels = []
        for p in raw_panels[:6]:
            panels.append(
                PanelPlan(
                    panel_number=p.get("panel_number", len(panels) + 1),
                    narration_beat=p.get("narration_beat", ""),
                    visual_description=p.get("visual_description", "Clean minimalist line art"),
                    veo_prompt=p.get("veo_prompt")
                )
            )
        # Pad to 6 panels if needed
        while len(panels) < 6:
            p_num = len(panels) + 1
            panels.append(
                PanelPlan(
                    panel_number=p_num,
                    narration_beat="",
                    visual_description=f"Action beat {p_num}"
                )
            )

        search_q = s.get("search_queries", [])
        if not search_q and s.get("search_query"):
            search_q = [s.get("search_query")]

        scene_plans.append(
            ScenePlan(
                scene_number=s_num,
                narration=s.get("narration", " ".join(p.narration_beat for p in panels if p.narration_beat)),
                description=s.get("description", ""),
                visual_setup=s.get("visual_setup", "3x2 Whiteboard storyboard grid"),
                search_queries=search_q[:2],
                panels=panels,
                text_overlay=s.get("text_overlay"),
                key_information=s.get("key_information"),
                emotional_beat=s.get("emotional_beat"),
                veo_prompt=panels[-1].veo_prompt if panels else None
            )
        )

    return global_plan, scene_plans


def story_planner_tool_fn(user_input: str, research_material: str = None, run_config: RunConfig = None) -> Tuple[Dict[str, Any], List[ScenePlan]]:
    """Unified entry point for story planning."""
    if run_config is None:
        run_config = RunConfig()

    if run_config.verbatim_mode:
        return plan_verbatim_storyboard(user_input, run_config)
    else:
        return plan_creative_storyboard(user_input, research_material, run_config)
