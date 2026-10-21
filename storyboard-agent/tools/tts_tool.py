"""
TTS Audio Generation Tool using Gemini 3.8 Flash TTS.
Handles native RIFF WAV byte streaming directly without double-wrapping headers.
"""
import os
import time
from typing import Optional
from google.genai import types

from config import TTS_MODEL, DEFAULT_TTS_VOICE
from cost_tracker import tracker
from . import utils


def generate_tts_audio_tool_fn(
    text: str,
    voice_name: str = None,
    language: str = "english"
) -> Optional[str]:
    """
    Generates TTS audio from text using Gemini 3.8 Flash TTS with natural human pacing and inflection.
    
    Args:
        text: Narration text to speak.
        voice_name: Voice preset (default from config: 'Aoede').
        language: Spoken language.
        
    Returns:
        Absolute path to the generated .wav file or None on failure.
    """
    if not utils.client:
        print("Error: GEMINI_API_KEY not configured.")
        return None

    if not text or not text.strip():
        print("Error: Empty narration text provided to TTS.")
        return None

    effective_voice = voice_name or DEFAULT_TTS_VOICE
    print(f"Generating Human-like TTS Audio ({TTS_MODEL} | Voice: {effective_voice}): '{text[:60]}...'")

    try:
        speech_config = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=effective_voice)
            )
        )

        prompt_text = (
            f"[voice direction: natural human conversational cadence, warm expressive documentary narrator, "
            f"subtle realistic breathing pauses, lively inflection and emotional depth, consistent clear volume, "
            f"strictly avoiding flat mechanical monotone or robotic delivery, speaking in {language}]\n"
            f"{text.strip()}"
        )

        response = utils.client.models.generate_content(
            model=TTS_MODEL,
            contents=prompt_text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=speech_config,
            )
        )

        if not response.candidates or not response.candidates[0].content.parts:
            print("TTS Error: No candidates or content parts in response.")
            return None

        audio_part = response.candidates[0].content.parts[0]
        if audio_part.inline_data is None:
            print("TTS Error: No inline audio data in response.")
            return None

        raw_bytes = audio_part.inline_data.data
        if not raw_bytes:
            print("TTS Error: Empty audio byte payload.")
            return None

        timestamp = int(time.time())
        filename = f"audio_{timestamp}.wav"
        out_dir = utils.get_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, filename)

        # Check if model returned native RIFF WAV format (as gemini-3.8-flash-tts does)
        if raw_bytes.startswith(b"RIFF"):
            with open(output_path, "wb") as f:
                f.write(raw_bytes)
        else:
            # Fallback to PCM wrapping for legacy models
            utils.save_pcm_to_wav(output_path, raw_bytes)

        duration = utils.get_audio_duration(output_path)
        tracker.record_audio(duration)
        print(f"TTS Audio SUCCESS: Saved to {output_path} (Duration: {duration:.2f}s)")
        return os.path.abspath(output_path)

    except Exception as e:
        print(f"TTS Generation Error: {e}")
        return None
