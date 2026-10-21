import os
import subprocess
import wave
import datetime
import re
from google import genai
from config import GEMINI_API_KEY

# Setup GenAI Client
client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)

# Global Output Management
GLOBAL_OUTPUT_DIR = None


def set_output_dir(path: str):
    """Sets the directory where all tool outputs will be saved."""
    global GLOBAL_OUTPUT_DIR
    GLOBAL_OUTPUT_DIR = path
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_output_dir() -> str:
    global GLOBAL_OUTPUT_DIR
    return GLOBAL_OUTPUT_DIR or "output"


def slugify_title(title: str, max_len: int = 40) -> str:
    """Converts a title into a clean, readable filename slug separated with hyphens."""
    if not title:
        return "movie"
    slug = re.sub(r'[^a-zA-Z0-9]+', '-', title.strip().lower()).strip('-')
    return slug[:max_len] or "movie"


def init_dated_output_dir(title: str = "storyboard", base_dir: str = "output") -> str:
    """
    Creates and sets a readable, dated output directory:
    e.g., output/2026-10-21-13-40-00-mughal-empire-1707-1720
    """
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    clean_title = slugify_title(title)
    dir_name = f"{timestamp_str}-{clean_title}"
    full_path = os.path.abspath(os.path.join(base_dir, dir_name))
    set_output_dir(full_path)
    return full_path


def _save_to_run_folder(content: str, filename: str, mode: str = "w"):
    """Helper to save content to current run folder if enabled."""
    out_dir = get_output_dir()
    full_path = os.path.join(out_dir, filename)
    try:
        with open(full_path, mode, encoding="utf-8") as f:
            f.write(content)
        return full_path
    except Exception as e:
        print(f"Error saving to {filename}: {e}")
    return None


def save_pcm_to_wav(filename: str, pcm: bytes, channels: int = 1, rate: int = 24000, sample_width: int = 2):
    """Saves raw PCM data to a .wav file."""
    try:
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm)
        return True
    except Exception as e:
        print(f"Error saving wave file: {e}")
        return False


def has_audio_stream(file_path: str) -> bool:
    """Checks whether a media file has an audio stream."""
    if not os.path.exists(file_path):
        return False
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return bool(res.stdout.strip())
    except Exception:
        return False


def get_audio_duration(audio_path: str) -> float:
    """Returns duration of an audio file or audio stream in seconds. Returns 0.0 if no audio stream exists."""
    if not os.path.exists(audio_path):
        return 0.0
    try:
        with wave.open(audio_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return float(frames) / float(rate)
    except Exception:
        pass

    if not has_audio_stream(audio_path):
        return 0.0

    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        val = result.stdout.strip()
        if val and val != "N/A":
            return float(val)
        return get_video_duration(audio_path)
    except Exception:
        return 0.0


def get_video_duration(video_path: str) -> float:
    """Returns duration in seconds using ffprobe."""
    if not os.path.exists(video_path):
        return 0.0
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Warning: Failed to get duration for {video_path}: {e}")
        return 0.0
