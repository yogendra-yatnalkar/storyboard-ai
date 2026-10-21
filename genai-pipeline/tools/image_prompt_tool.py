from config import MODEL_NAME, get_generation_config
from .utils import client, _save_to_run_folder

def prompt_tool_fn(scene_description: str, visual_setup: str = "", text_overlay: str = "", global_plan: dict = None) -> str:
    """
    Generates an image prompt for a whiteboard animation frame using Nano Banana guidelines.
    
    Formula: [Subject] + [Action] + [Location/context] + [Composition] + [Style]
    
    Args:
        scene_description: Visual description of the scene.
        visual_setup: Specific instructions for this frame (from the Director).
        text_overlay: Specific impact text to render into the frame.
        global_plan: The global plan dictionary (from the Director).
    Returns:
        The image generation prompt string.
    """
    tone = global_plan.get("tone", "dramatic") if global_plan else "dramatic"
    visual_style = global_plan.get("visual_style", "Clean Whiteboard Animation") if global_plan else "Clean Whiteboard Animation"
    
    tone_guidance = ""
    if tone == "informative":
        tone_guidance = "The visual should be clear, accurate, and educational. Use a neat composition."
    elif tone == "dramatic":
        tone_guidance = "The visual should be expressive, using dynamic framing and bold focus."
    else:
        tone_guidance = "The visual should be engaging and clear."

    text_guidance = ""
    if text_overlay:
        text_guidance = f"""
    TEXT OVERLAY HANDLING (CRITICAL!):
    The user wants specifically to insert this text: {text_overlay}
    Translate this into precise typography instructions using quotes and font styles.
    Example: Render the text "[the exact wording]" in a bold, black, marker-style font in the corner of the frame.
    """

    prompt = f"""
    You are an expert whiteboard animation artist and creative director.
    
    Your job: Create an image generation prompt using the Nano Banana optimal formula:
    [Subject] + [Action] + [Location/context] + [Composition] + [Style]
    
    WHAT WHITEBOARD ANIMATION LOOKS LIKE:
    - Clean WHITE background (like a dry-erase whiteboard)
    - Simple, quick LINE DRAWINGS using black lines (just the pure ink on the board)
    - Hand-drawn aesthetic — not photorealistic, not heavily detailed
    - NO shading, NO gradients — flat simple strokes only
    
    NEGATIVE PROMPT / STRICT FORBIDDEN:
    - DO NOT under any circumstances draw hands, human arms, markers, pens, or any person physically drawing the final picture!
    - Erase any concept of the artist from the scene.
    - Provide ONLY the final completed artwork standing alone upon the white background.
    
    COLOR ENHANCEMENT RULE:
    - The drawing is primarily BLACK lines on WHITE background
    - 1-2 KEY objects or focal areas should have VIBRANT selective color
    - Everything else stays black-and-white line art
    
    SCENE DETAILS:
    - Subject / Description: "{scene_description}"
    - Action / Setup: "{visual_setup}"
    - Tone: {tone} ({tone_guidance})
    {text_guidance}
    
    CONSTRUCT the final image generation prompt starting directly with [Subject] and following the formula implicitly. Ensure the Style section strictly describes the whiteboard marker, flat strokes, selective color, and clean white background without blending styles or becoming overly complex. Make sure if text is included, to wrap it in exact double quotes "like this" and specify the font.
    
    Output: ONLY the final prompt string. No explanations, no markdown blocks.
    """
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=get_generation_config()
        )
        result = response.text.strip()
        result = result.replace('\"', '"').replace('`', '').strip()
        
        _save_to_run_folder(f"Scene: {scene_description}\nText Overlay: {text_overlay}\nPrompt: {result}\n---\n", "prompts_log.txt", mode="a")
        return result
    except Exception as e:
        return f"Error prompt: {e}"


