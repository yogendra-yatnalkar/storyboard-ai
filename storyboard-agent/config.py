import os
from dotenv import load_dotenv
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Models
MODEL_NAME = "gemini-3.8-flash"
THINKING_LEVEL = "MEDIUM"

# Deep Research Model
DEEP_RESEARCH_MODEL = "deep-research-preview-04-2026"

# Image Generation Model
IMAGE_GEN_MODEL = "gemini-3.1-flash-image"

# TTS Model
TTS_MODEL = "gemini-3.8-flash-tts"
DEFAULT_TTS_VOICE = "Aoede"

# Veo Video Generation Model
VEO_MODEL = "veo-3.1-lite-generate-preview"
VEO_RESOLUTION = "720p"

# SAM Segmentation Model URL
SAM_API_URL = "https://sam3-app-1040077537378.us-east4.run.app/predict"

# Drawing & Presentation Style
WHITEBOARD_OUTLINE_STYLE = (
    "Clean minimalist whiteboard marker drawing, bold crisp black contours, "
    "pure white background, strictly zero cross-hatching, zero texture fills, "
    "zero gray shading, pure coloring-book line art style."
)
# Video Dimensions: Unified Native 720p for Whiteboard & Veo Parity
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
MAX_REFERENCES_PER_SCENE = 2

# Master Feature Flags (Centralized Config-Driven)
# If ENABLE_VEO is False, Veo is strictly forbidden system-wide across all agents.
ENABLE_VEO = os.getenv("ENABLE_VEO", "True").lower() in ("true", "1", "yes")
DEFAULT_ENABLE_VEO = ENABLE_VEO
VEO_SCOPE = os.getenv("VEO_SCOPE", "random_50")  # Options: "none", "last_panel", "random_50", "all_panels"
DEFAULT_VEO_SCOPE = VEO_SCOPE
VEO_MODE = os.getenv("VEO_MODE", "after_drawing")    # Options: "after_drawing" (drawing + Veo), "veo_only"
DEFAULT_VEO_MODE = VEO_MODE

# SAM 3 Segmentation Master Flag
ENABLE_SAM = os.getenv("ENABLE_SAM", "True").lower() in ("true", "1", "yes")
DEFAULT_ENABLE_SAM = ENABLE_SAM

# Pipeline & Concurrency Defaults
DEFAULT_RESEARCH_MODE = "web_search"  # Options: "none", "web_search", "url_grounding", "deep_research"
DEFAULT_VERBATIM_MODE = False
DEFAULT_LANGUAGE = "english"
PARALLEL_BATCH_SIZE = int(os.getenv("PARALLEL_BATCH_SIZE", "2"))
MAX_PARTS = int(os.getenv("MAX_PARTS", "3"))


def get_generation_config(model: str = MODEL_NAME, thinking_level: str = None, **kwargs) -> types.GenerateContentConfig:
    """
    Returns a GenerateContentConfig.
    Thinking level is ONLY applied to models that support it (e.g. gemini-3.8-flash).
    """
    cfg_args = dict(kwargs)
    
    # Only apply thinking if model supports reasoning and thinking_level is specified or defaulted for MODEL_NAME
    supports_thinking = ("flash" in model.lower() or "pro" in model.lower()) and ("image" not in model.lower()) and ("tts" not in model.lower())
    
    effective_thinking = thinking_level if thinking_level is not None else (THINKING_LEVEL if supports_thinking else None)
    
    if supports_thinking and effective_thinking:
        cfg_args["thinking_config"] = types.ThinkingConfig(thinking_level=effective_thinking)

    return types.GenerateContentConfig(**cfg_args)


if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in environment variables.")
