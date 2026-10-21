import json
import re
from typing import List, Dict, Any
from config import MODEL_NAME, get_generation_config
from .utils import client, _save_to_run_folder

def divider_tool_fn(research_output: str) -> Dict[str, Any]:
    """
    Acts as a Director to plan the video and divide research into scenes.
    
    Args:
        research_output: The detailed research report text.
    Returns:
        A dictionary with 'global_plan' and 'scenes'.
    """
    prompt = f"""
    You are a professional Video Director and Creative Planner. 
    Analyze the following research and plan a high-quality whiteboard animation video.
    
    Research Material:
    ---
    {research_output}
    ---
    
    Your Task:
    1. Determine the appropriate TONE and STYLE for this video based on the topic. 
       - If it's informative/financial (like NPS/PF), use a "Documentary/Informative" style (authoritative, clear, professional).
       - If it's a historical/action story (like Shivaji Maharaj), use a "Dramatic/Narrative" style (engaging, cinematic, tense).
       - Avoid making dry informative topics "mysterious" or "thriller-like".
    2. Create a 'global_plan' that defines the Narrative Persona and Visual Aesthetic.
    3. Break the content into 10-15+ scenes.
    
    Requirements for each scene:
    - 'narration': The spoken script (aligned with the global tone).
    - 'description': A visual description for the image generator.
    - 'visual_setup': Specific instructions for this frame (e.g., "Use a 3D bar chart for comparison", "Draw a realistic map of India", "Keep it simple and schematic").
    
    Output Format (Strict JSON):
    {{
      "global_plan": {{
        "tone": "informative" | "dramatic",
        "narrative_persona": "Professional Documentary Narrator" | "Epic Storyteller" | etc.,
        "visual_style": "Clean Technical Storyboard" | "Cinematic Historical Sketch" | etc.,
        "pacing": "steady/educational" | "fast/action" | etc.
      }},
      "scenes": [
        {{
          "narration": "...",
          "description": "...",
          "visual_setup": "..."
        }},
        ...
      ]
    }}
    """
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=get_generation_config(
                response_mime_type='application/json'
            )
        )
        result = json.loads(response.text)
        _save_to_run_folder(json.dumps(result, indent=2), "video_plan.json")
        return result
    except Exception as e:
        print(f"Error in director_tool: {e}")
        # Fallback to a basic structure if parsing fails
        return {
            "global_plan": {
                "tone": "informative", 
                "narrative_persona": "Professional Documentary Narrator", 
                "visual_style": "Clean Technical Storyboard", 
                "pacing": "steady"
            },
            "scenes": [{"description": "Error parsing", "narration": "Error parsing", "visual_setup": "Simple sketch"}]
        }
