import json
import re
from typing import List, Dict, Any
from config import MODEL_NAME
from .utils import client, _save_to_run_folder

def director_tool_fn(user_instructions: str, research_material: str = None, language: str = "english", enable_veo: bool = False, veo_direction_by_director: bool = False) -> Dict[str, Any]:
    """
    Acts as the Video Director & Writer — plans the entire video journey.
    
    The Director decides how many scenes are needed, writes the narration script
    for each scene (as a storyteller, not just descriptions), plans the visual
    setup, and defines the overall narrative arc.
    
    Args:
        user_instructions: The user's original topic/instructions.
        research_material: Optional detailed research report to incorporate.
        language: The target language for the narration script (default: English).
        enable_veo: Whether Veo video generation is enabled.
        veo_direction_by_director: Whether the director should explicitly draft Veo prompts.
    Returns:
        A dictionary with 'global_plan' and 'scenes'.
    """
    
    # If research material is provided, include it; otherwise just use instructions
    research_block = ""
    if research_material and research_material != user_instructions:
        research_block = f"""
    
    Deep Research Material (USE THIS — it contains rich, detailed information that MUST be woven into your narration):
    ---
    {research_material}
    ---
    """

    veo_instruction = ""
    veo_schema_field = ""
    if enable_veo and veo_direction_by_director:
        veo_instruction = (
            "- 'veo_prompt': Write a descriptive prompt for Veo video generation. This prompt should describe "
            "how the elements in the whiteboard drawing should come to life, animate, move, or transition. "
            "The video starts from the final whiteboard drawing, so describe the motion continuation. "
            "Keep it focused on movement, actions, and style continuity from the sketch. Avoid adding marker pens/hands."
        )
        veo_schema_field = '\n          "veo_prompt": "...",'

    prompt = f"""
    You are an award-winning Video Director, Writer, and Storyteller.
    You are planning a whiteboard animation video. Your job is to craft the ENTIRE video — 
    the narrative arc, the script, and the visual direction for every single scene.
    
    User's Topic / Instructions:
    "{user_instructions}"
    {research_block}
    
    YOUR TASK — Plan the complete video:
    
    STEP 1: Analyze the topic and decide:
    - What TONE fits? (informative, dramatic, playful, sad, etc.)
    - What is the NARRATIVE ARC? (beginning hook → build-up → climax → resolution)
    - Who is narrating? (a professional explainer, a storyteller, a historian, etc.)
    - How many scenes are needed? (CRITICAL: Follow these rules strictly)
    - The goal is that no single scene should have more than ~30-40 seconds of narration.
    
    STEP 2: For EACH scene, you must provide:
    - 'scene_number': Sequential number
    - 'summary': A 1-line summary of what this scene accomplishes in the narrative arc
    - 'narration': The FULL spoken script for this scene. THIS IS THE MOST IMPORTANT PART.
    - 'description': Visual description for the image generator (what should be DRAWN in this frame)
    - 'visual_setup': Specific visual direction for this frame (composition, key elements, focal points)
    {veo_instruction}
    - 'search_query': (OPTIONAL) If this scene features a specific real-world person, historical figure, or landmark, provide a search query. This query must be a clean, exact name of the entity (e.g., 'Volodymyr Zelenskyy', 'Vladimir Putin', 'St. Petersburg') without any descriptive style keywords like 'illustration', 'sketch', or 'drawing' to ensure accurate Wikipedia image retrieval.
    - 'text_overlay': (OPTIONAL) If you want specific impactful text visually rendered.
    - 'key_information': Any critical facts/data from the research that this scene must convey
    - 'emotional_beat': The emotional tone of this specific scene
    
    CRITICAL RULES:
    - LANGUAGE: The entire script's narration and summary values MUST be written in {language}.
    - ATTRACTIVE PACING & TONE: You MUST detect pacing instructions from the user.
    - FULL CONTEXT RETENTION: If the user's instructions or topic asks to "take the entire text extracted", to use the "full context", or mentions animating a news link/article:
      * Do NOT over-summarize the research material.
      * You must plan as many scenes as necessary to cover the source material completely.
      * Write detailed, thorough narration scripts for each scene that capture all key events, factual details, timelines, names, and quotes from the research.
      * Ensure the narrative flow acts as a comprehensive narration/script of the article, retaining its details to the fullest.
      * Its fine if narration is long and animation ends fast. 
    
    
    Output Format (Strict JSON) where all values (specifically 'narration' and 'summary') are in the language '{language}', but the JSON keys remain exactly as defined below in English:
    {{
      "global_plan": {{
        "title": "Video title",
        "tone": "informative" | "dramatic" | "educational" | "cautionary",
        "narrative_persona": "e.g., Wise Storyteller",
        "visual_style": "e.g., Clean Whiteboard Animation",
        "pacing": "e.g., steady/educational",
        "narrative_arc": "...",
        "target_audience": "...",
        "total_scenes": <number>
      }},
      "scenes": [
        {{
          "scene_number": 1,
          "summary": "...",
          "narration": "...",
          "description": "...",
          "visual_setup": "...",{veo_schema_field}
          "search_query": "...",
          "text_overlay": "...",
          "key_information": "...",
          "emotional_beat": "..."
        }},
        ...
      ]
    }}
    """
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                'response_mime_type': 'application/json'
            }
        )
        result = json.loads(response.text)
        _save_to_run_folder(json.dumps(result, indent=2), "video_plan.json")
        return result
    except Exception as e:
        print(f"Error in director_tool: {e}")
        # Fallback to a basic structure if parsing fails
        return {
            "global_plan": {
                "title": "Untitled Video",
                "tone": "informative", 
                "narrative_persona": "Professional Storyteller", 
                "visual_style": "Clean Whiteboard Animation", 
                "pacing": "steady",
                "narrative_arc": "Linear exploration of the topic",
                "target_audience": "general public",
                "total_scenes": 1
            },
            "scenes": [{
                "scene_number": 1,
                "summary": "Error parsing",
                "description": "Error parsing", 
                "narration": "Error parsing", 
                "visual_setup": "Simple sketch",
                "search_query": "",
                "text_overlay": "",
                "key_information": "",
                "emotional_beat": "neutral"
            }]
        }
