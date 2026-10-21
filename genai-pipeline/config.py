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
# IMAGE_GEN_MODEL = "gemini-2.5-flash-image"
# IMAGE_GEN_MODEL = "gemini-3-pro-image"
IMAGE_GEN_MODEL = "gemini-3.1-flash-lite-image"

# TTS Model
TTS_MODEL = "gemini-2.5-flash-preview-tts"

# Veo Video Generation Model
VEO_MODEL = "veo-3.1-lite-generate-preview"

# SAM Segmentation Model URL
# To host SAM 3 on your own Cloud Run endpoint, follow instructions in sam3-hosting/README.md
SAM_API_URL = "https://sam3-app-1040077537378.us-east4.run.app/predict"


def get_generation_config(thinking_level: str = THINKING_LEVEL, **kwargs) -> types.GenerateContentConfig:
    """Helper to return GenerateContentConfig preconfigured with THINKING_LEVEL."""
    thinking_config = types.ThinkingConfig(thinking_level=thinking_level) if thinking_level else None
    return types.GenerateContentConfig(thinking_config=thinking_config, **kwargs)


if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in environment variables.")
