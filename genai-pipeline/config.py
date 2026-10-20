import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Models
# User referred to "flash3", likely meaning the latest Gemini 2.0 Flash
MODEL_NAME = "gemini-3.5-flash" 

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
# SAM_API_URL = "https://sam3-app-1040077537378.us-east4.run.app/predict"
SAM_API_URL = ""

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in environment variables.")
