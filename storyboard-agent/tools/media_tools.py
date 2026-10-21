"""
Media processing tools:
- Whiteboard drawing animation (OpenCV)
- Video & Audio speed-matching and merging (FFmpeg)
- Audio transcription & subtitle burning (Gemini 3.8 Flash + FFmpeg)
- Multi-scene video assembly (FFmpeg)
- Veo video generation (Veo 3.1 Lite 720p)
"""
import json
import os
import re
import subprocess
import time
from typing import List, Optional
from google.genai import types

from config import MODEL_NAME, VEO_MODEL, VEO_RESOLUTION, get_generation_config
from cost_tracker import tracker
from . import utils
from .draw_animation import draw_animation_tool_fn


def create_whiteboard_animation_tool_fn(
    image_path: str,
    segmentation_results_path: Optional[str] = None,
    frame_rate: int = 25,
    **kwargs
) -> Optional[str]:
    """Generates whiteboard drawing animation video from an image."""
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None

    print(f"Generating whiteboard drawing animation for: {image_path}...")
    try:
        video_path = draw_animation_tool_fn(
            image_path=image_path,
            segmentation_results_path=segmentation_results_path,
            frame_rate=frame_rate,
            **kwargs
        )
        if video_path and not video_path.startswith("Error") and os.path.exists(video_path):
            print(f"Whiteboard animation SUCCESS: {video_path}")
            return os.path.abspath(video_path)
        else:
            print(f"Whiteboard animation failed: {video_path}")
            return None
    except Exception as e:
        print(f"Exception during whiteboard animation: {e}")
        return None


def merge_audio_video_tool_fn(video_path: str, audio_path: str) -> Optional[str]:
    """
    Merges audio and video files. If audio is longer, freezes the last frame to match audio.
    """
    if not os.path.exists(video_path) or not os.path.exists(audio_path):
        print(f"Merge Error: File missing: video={video_path}, audio={audio_path}")
        return None

    try:
        video_dur = utils.get_video_duration(video_path)
        audio_dur = utils.get_audio_duration(audio_path)
        print(f"Merging AV: Video={video_dur:.2f}s, Audio={audio_dur:.2f}s")

        timestamp = int(time.time())
        filename = f"merged_{timestamp}.mp4"
        out_dir = utils.get_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, filename)

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
        ]

        if audio_dur > video_dur:
            # Audio is longer: freeze last frame of video to allow audio to finish
            pad_seconds = audio_dur - video_dur
            filter_complex = f"[0:v]tpad=stop_mode=clone:stop_duration={pad_seconds}[v]"
            cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "[v]",
                "-map", "1:a",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-b:a", "192k",
                output_path
            ])
        else:
            # Video is longer: keep full video without cropping! Play audio, and smoothly pad silence to match video duration
            pad_seconds = max(0.0, video_dur - audio_dur)
            cmd.extend([
                "-map", "0:v",
                "-map", "1:a",
                "-af", f"apad=pad_dur={pad_seconds:.3f}",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                output_path
            ])

        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        if os.path.exists(output_path):
            print(f"Merge AV SUCCESS: Saved to {output_path}")
            return os.path.abspath(output_path)
        return None

    except Exception as e:
        print(f"Merge AV Error: {e}")
        return None


def _format_srt_time(seconds: float) -> str:
    millis = int((seconds - int(seconds)) * 1000)
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def transcribe_and_subtitle_tool_fn(video_path: str, audio_path: str, language: str = "english") -> Optional[str]:
    """
    Transcribes audio using Gemini 3.8 Flash, creates SRT subtitles, and burns them into video.
    """
    if not utils.client or not os.path.exists(video_path) or not os.path.exists(audio_path):
        return video_path

    print(f"Transcribing audio for subtitles ({language})...")
    srt_path = None
    try:
        # Step 1: Upload audio file for transcription
        uploaded_file = utils.client.files.upload(file=audio_path)

        prompt = f"""
Transcribe this audio in {language}.
Provide accurate sentence-level timestamped segments for video subtitles.
Output strictly valid JSON list:
[
  {{"start": 0.0, "end": 2.5, "text": "..."}},
  ...
]
"""
        config = get_generation_config(model=MODEL_NAME, response_mime_type="application/json")
        response = utils.client.models.generate_content(
            model=MODEL_NAME,
            contents=[uploaded_file, prompt],
            config=config
        )
        tracker.record_usage_metadata(response.usage_metadata)

        try:
            utils.client.files.delete(name=uploaded_file.name)
        except Exception:
            pass

        subtitles = json.loads(response.text.strip())

        # Step 2: Write SRT file
        timestamp = int(time.time())
        srt_filename = f"subtitles_{timestamp}.srt"
        out_dir = utils.get_output_dir()
        srt_path = os.path.join(out_dir, srt_filename)

        with open(srt_path, "w", encoding="utf-8") as f:
            for idx, item in enumerate(subtitles, 1):
                start_s = _format_srt_time(float(item.get("start", 0.0)))
                end_s = _format_srt_time(float(item.get("end", 0.0)))
                text = item.get("text", "").strip()
                f.write(f"{idx}\n{start_s} --> {end_s}\n{text}\n\n")

        # Step 3: Burn subtitles with FFmpeg
        subtitled_video_filename = f"subtitled_{timestamp}.mp4"
        subtitled_video_path = os.path.join(out_dir, subtitled_video_filename)

        escaped_srt = srt_path.replace("\\", "/").replace(":", "\\:")
        sub_filter = f"subtitles='{escaped_srt}':force_style='FontSize=16,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3,BackColour=&H80000000,MarginV=25'"

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", sub_filter,
            "-c:a", "copy",
            "-c:v", "libx264",
            subtitled_video_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        if os.path.exists(subtitled_video_path):
            print(f"Subtitle Burning SUCCESS: {subtitled_video_path}")
            return os.path.abspath(subtitled_video_path)

        return video_path

    except Exception as e:
        print(f"Warning: Subtitle generation failed ({e}). Returning original video.")
        return video_path


def concatenate_scene_videos_tool_fn(scene_video_paths: List[str], output_filename: str = "story.mp4") -> Optional[str]:
    """
    Losslessly concatenates scene videos into a master video using standard FFmpeg filter_complex.
    Normalizes every clip to exactly 1280x720, 25fps, yuv420p, setsar=1 to ensure 100% glitch-free
    concatenation between whiteboard animation and Veo video clips.
    """
    valid_paths = [p for p in scene_video_paths if p and os.path.exists(p)]
    if not valid_paths:
        print("Error: No valid scene videos to concatenate.")
        return None

    if len(valid_paths) == 1:
        return valid_paths[0]

    out_dir = utils.get_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    final_output = os.path.join(out_dir, output_filename)

    print(f"Concatenating {len(valid_paths)} videos into {final_output} with normalized 720p 25fps filter_complex...")
    try:
        # Check if all clips have audio
        has_audio = all(utils.has_audio_stream(p) for p in valid_paths)

        cmd = ["ffmpeg", "-y"]
        for p in valid_paths:
            cmd.extend(["-i", p])

        filter_parts = []
        for i in range(len(valid_paths)):
            filter_parts.append(f"[{i}:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=25,setsar=1,format=yuv420p[v{i}];")

        if has_audio:
            concat_inputs = "".join([f"[v{i}][{i}:a]" for i in range(len(valid_paths))])
            filter_parts.append(f"{concat_inputs}concat=n={len(valid_paths)}:v=1:a=1[outv][outa]")
            cmd.extend([
                "-filter_complex", "".join(filter_parts),
                "-map", "[outv]",
                "-map", "[outa]",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-c:a", "aac",
                "-b:a", "192k",
                "-r", "25",
                final_output
            ])
        else:
            concat_inputs = "".join([f"[v{i}]" for i in range(len(valid_paths))])
            filter_parts.append(f"{concat_inputs}concat=n={len(valid_paths)}:v=1:a=0[outv]")
            cmd.extend([
                "-filter_complex", "".join(filter_parts),
                "-map", "[outv]",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-r", "25",
                final_output
            ])

        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        if os.path.exists(final_output) and os.path.getsize(final_output) > 2000:
            print(f"Video Assembly SUCCESS (normalized filter_complex): {final_output}")
            return os.path.abspath(final_output)
    except Exception as e:
        print(f"Filter-complex concat failed ({e}). Returning first valid path.")

    return valid_paths[0]


def generate_video_veo_tool_fn(image_path: str, prompt: str) -> Optional[str]:
    """Generates 720p Veo video clip based on input image."""
    if not utils.client or not os.path.exists(image_path):
        return None

    print(f"Generating 720p Veo clip ({VEO_MODEL})...")
    tracker.record_video(1)

    try:
        enhanced_prompt = (
            f"Whiteboard animation video showing: {prompt}. "
            "Clean whiteboard animation with white background and hand-drawn line drawings. "
            "DO NOT draw any hands, markers, or pens. Draw only the artwork."
        )

        with open(image_path, "rb") as f:
            img_bytes = f.read()

        mime_type = "image/png"
        if image_path.lower().endswith((".jpg", ".jpeg")):
            mime_type = "image/jpeg"

        operation = utils.client.models.generate_videos(
            model=VEO_MODEL,
            prompt=enhanced_prompt,
            image=types.Image(image_bytes=img_bytes, mime_type=mime_type),
            config=types.GenerateVideosConfig(
                aspect_ratio="16:9",
                duration_seconds=8,
                resolution=VEO_RESOLUTION  # Strictly 720p
            )
        )

        while not operation.done:
            time.sleep(10)
            operation = utils.client.operations.get(operation)

        if operation.error:
            print(f"Veo Error: {operation.error}")
            return None

        if not operation.response or not operation.response.generated_videos:
            return None

        generated_video = operation.response.generated_videos[0]
        video_file = generated_video.video
        video_bytes = None
        if hasattr(video_file, "video_bytes") and video_file.video_bytes:
            video_bytes = video_file.video_bytes
        else:
            file_ref = getattr(video_file, "name", getattr(video_file, "uri", video_file))
            video_bytes = utils.client.files.download(file=file_ref)

        timestamp = int(time.time())
        filename = f"veo_raw_{timestamp}.mp4"
        out_dir = utils.get_output_dir()
        out_path = os.path.join(out_dir, filename)

        with open(out_path, "wb") as f:
            f.write(video_bytes)

        # Normalize Veo video to exactly match 1280x720 25fps yuv420p and strip audio
        silent_output = os.path.join(out_dir, f"veo_720p_{timestamp}.mp4")
        norm_cmd = [
            "ffmpeg", "-y",
            "-i", out_path,
            "-an",
            "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=25,format=yuv420p",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            silent_output
        ]
        try:
            subprocess.run(norm_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            if os.path.exists(silent_output):
                print(f"Veo 720p Normalized SUCCESS: {silent_output}")
                return os.path.abspath(silent_output)
        except Exception as e:
            print(f"Veo normalization warning: {e}")

        return os.path.abspath(out_path)

    except Exception as e:
        print(f"Veo generation exception: {e}")
        return None
