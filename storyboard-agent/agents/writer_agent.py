"""
Writer Agent for Storyboard Multi-Agent Service.
Specialized Screenwriter and Story Architect Agent built on google.adk.Agent.
Responsible for:
1. Historical topic decomposition into cohesive narrative Parts (Chapters).
2. For each Part: defining narrative approach, 2 generic entity search queries,
   and 6 sequential sub-scene beats corresponding to the 3x2 comic grid.
3. Pre-assigning high-motion beats as Veo candidates.
4. Providing interactive review format.
"""
import json
import re
from typing import Dict, Any, List, Optional
from google.adk import Agent, Context
from google.genai import types

from config import MODEL_NAME, get_generation_config
from cost_tracker import tracker
from tools import utils
from tools.story_planner_tool import partition_words_into_beats, reconcile_verbatim_beats


def plan_creative_story_parts_tool(tool_context: Context, topic: str, context_notes: str = "", num_parts: Optional[int] = None) -> str:
    """
    Plans an original dramatic/documentary storyboard script.
    If num_parts is specified or found in topic, divides into that count; otherwise dynamically decides the optimal number.
    Each Part contains 6 sequential sub-scene beats, 2 generic entity queries, and Veo candidate flags.
    Saves the structured plan in tool_context.state["storyboard_script"].
    """
    if num_parts is None or num_parts <= 0:
        match = re.search(r'(\d+)\s*[- ]?(?:parts?|chapters?|episodes?)', topic, re.IGNORECASE)
        if match:
            num_parts = int(match.group(1))
            print(f"[WriterAgent] Inferred {num_parts} Parts directly from user prompt.")

    if num_parts:
        print(f"[WriterAgent] Planning {num_parts}-Part creative story for topic: '{topic}'...")
        parts_instruction = f"1. Divide the overarching narrative into exactly {num_parts} chronologically progressive Parts (Chapters)."
        part_num_desc = f"1 to {num_parts}"
        output_instruction = f"Output strictly valid JSON list of {num_parts} Part objects:"
    else:
        print(f"[WriterAgent] Planning creative story with dynamically agent-determined Parts for topic: '{topic}'...")
        parts_instruction = "1. Dynamically determine the optimal number of chronologically progressive Parts (typically 2 to 4 Parts depending on historical scope and narrative depth) to tell this story completely and compellingly."
        part_num_desc = "1 to N"
        output_instruction = "Output strictly valid JSON list of Part objects (1 to N, where N is the optimal number of Parts you determined):"

    prompt = f"""
You are the Master Screenwriter and Documentary Story Architect.
TOPIC:
"{topic}"

HISTORICAL / CONTEXT NOTES:
"{context_notes or 'Use verified factual and dramatic history.'}"

DIRECTIVES:
{parts_instruction}
2. MANDATORY OPENING SCENE DIRECTIVE (Part 1, Sub-scene 1 ONLY):
   - The very first scene of the entire movie (Part 1, Sub-scene 1) MUST be a grand cinematic prologue/intro scene.
   - Its visual description MUST explicitly feature a prominent, ornate title banner / marquee / royal parchment scroll / carved stone monument with bold, stylized comic lettering displaying the movie's main title (e.g. "{topic[:40]}...").
   - Its narration MUST act as an evocative opening hook setting up the epic historical drama.
3. For EACH Part:
   - 'part_number': {part_num_desc}
   - 'title': Evocative title of this chapter
   - 'synopsis': High-level narrative summary of what happens in this Part
   - 'search_queries': List of up to 2 clean, generic names of historical figures or monuments in this Part (e.g. ["Bahadur Shah I", "Zulfiqar Khan"]). STRICTLY NO story verbs or actions.
   - 'subscenes': Exactly 6 sequential sub-scenes (corresponding to a 3x2 comic grid). For each sub-scene (1 to 6):
     * 'subscene_number': 1 to 6
     * 'narration': Spoken documentary narration (30-50 words per beat). Evocative, human, dramatic.
     * 'visual_description': Vivid storyboard description with clear negative space gaps between figures and props.
     * 'is_veo_candidate': true if this beat depicts intense kinetic action (cavalry charge, sword duel, artillery barrage, coronation crowd cheer); false otherwise.
     * 'veo_motion_prompt': Descriptive 8s action prompt if is_veo_candidate is true; otherwise empty.

{output_instruction}
[
  {{
    "part_number": 1,
    "title": "...",
    "synopsis": "...",
    "search_queries": ["...", "..."],
    "subscenes": [
      {{
        "subscene_number": 1,
        "narration": "...",
        "visual_description": "...",
        "is_veo_candidate": false,
        "veo_motion_prompt": ""
      }},
      ... (6 subscenes total)
    ]
  }},
  ...
]
"""
    try:
        config = get_generation_config(model=MODEL_NAME, response_mime_type="application/json")
        resp = utils.client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=config
        )
        tracker.record_usage_metadata(resp.usage_metadata)
        parts_data = json.loads(resp.text.strip())

        # Save into ADK state
        tool_context.state["storyboard_script"] = parts_data
        tool_context.state["story_topic"] = topic

        print(f"[WriterAgent] Successfully planned {len(parts_data)} Parts with 6 sub-scenes each.")
        return json.dumps({
            "status": "SUCCESS",
            "topic": topic,
            "parts_count": len(parts_data),
            "parts_summary": [
                {
                    "part_number": p.get("part_number"),
                    "title": p.get("title"),
                    "search_queries": p.get("search_queries"),
                    "veo_candidates": [s.get("subscene_number") for s in p.get("subscenes", []) if s.get("is_veo_candidate")]
                }
                for p in parts_data
            ]
        })
    except Exception as e:
        print(f"[WriterAgent] Error planning story: {e}")
        return json.dumps({"status": "FAILED", "error": str(e)})


def chunk_verbatim_story_parts_tool(tool_context: Context, source_text: str, num_parts: Optional[int] = None) -> str:
    """
    Chunks a locked verbatim script into num_parts (or dynamically determines optimal parts based on text length),
    decomposing each Part into 6 natural grammatical clauses with 100.0% exact word match guaranteed.
    Saves the structured plan in tool_context.state["storyboard_script"].
    """
    words = source_text.strip().split()
    total_words = len(words)

    if num_parts is None or num_parts <= 0:
        match = re.search(r'(\d+)\s*[- ]?(?:parts?|chapters?|episodes?)', source_text[:200], re.IGNORECASE)
        if match:
            num_parts = int(match.group(1))
        else:
            num_parts = max(1, min(5, round(total_words / 200)))
        print(f"[WriterAgent] Verbatim mode dynamically allocated {num_parts} Part(s) based on text length.")
    else:
        print(f"[WriterAgent] Chunking verbatim script into {num_parts} Part(s)...")

    # Chunk into Parts
    chunk_size = max(1, total_words // num_parts)
    parts_data = []

    curr_w_idx = 0
    for p_num in range(1, num_parts + 1):
        if p_num == num_parts:
            part_words = words[curr_w_idx:]
        else:
            part_words = words[curr_w_idx : curr_w_idx + chunk_size]
            curr_w_idx += chunk_size

        part_text = " ".join(part_words)
        # Decompose into 6 clauses
        beats = partition_words_into_beats(source_text=part_text, target_panels=6)

        subscenes = []
        for s_idx, beat in enumerate(beats, 1):
            v_desc = f"Verbatim narrative beat {s_idx} for Part {p_num}"
            if p_num == 1 and s_idx == 1:
                v_desc = f"Grand cinematic opening title banner with ornate comic typography displaying main title, setting up: {beat[:50]}"
            subscenes.append({
                "subscene_number": s_idx,
                "narration": beat,
                "visual_description": v_desc,
                "is_veo_candidate": len(beat.split()) >= 12,
                "veo_motion_prompt": f"Dynamic action for {beat[:50]}"
            })

        parts_data.append({
            "part_number": p_num,
            "title": f"Part {p_num}",
            "synopsis": f"Verbatim historical reading part {p_num}",
            "search_queries": ["Mughal Emperor", "Deccan Conflict"],
            "subscenes": subscenes
        })

    tool_context.state["storyboard_script"] = parts_data
    tool_context.state["verbatim_mode"] = True
    return json.dumps({
        "status": "SUCCESS",
        "mode": "verbatim",
        "parts_count": len(parts_data),
        "total_words": total_words
    })


def create_writer_agent() -> Agent:
    """Constructs the specialized Screenwriter and Story Architect ADK Agent."""
    instruction = (
        "You are the Master Screenwriter and Story Architect for the Storyboard Studio. "
        "Your responsibility is to plan cohesive, dramatic narrative storyboards divided into Parts (Chapters). "
        "Each Part MUST contain: "
        "1. Exactly 6 sequential sub-scene beats formatted for a 3x2 comic grid. "
        "2. Up to 2 clean, generic entity queries for historical portrait reference searches. "
        "3. Identification of high-kinetic action beats flagged as Veo candidates. "
        "4. MANDATORY: Part 1, Sub-scene 1 (the very first scene) MUST be a grand cinematic intro with a prominent title banner/marquee. "
        "Use plan_creative_story_parts_tool for original topics, or chunk_verbatim_story_parts_tool for verbatim scripts. "
        "Always present the completed script clearly for human interactive review."
    )

    return Agent(
        name="writer_agent",
        description="Screenwriter and Story Architect. Decomposes story topics into Parts with 6 sub-scenes each.",
        model=MODEL_NAME,
        instruction=instruction,
        tools=[plan_creative_story_parts_tool, chunk_verbatim_story_parts_tool]
    )
