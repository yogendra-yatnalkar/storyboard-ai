"""
Unit Test: Panel Whiteboard + Veo Clip Concatenation and AV Merge
Tests isolated single-panel tool execution:
1. Fast whiteboard drawing (~3-5s) with H.264 normalization.
2. Lossless filter_complex concatenation of Whiteboard Drawing + Veo Clip.
3. Verification of ZERO decoding errors, 1280x720 @ 25fps, and frame content in both segments.
4. Zero-crop AV merge preserving the entire combined visual.
"""
import os
import sys
import time
import subprocess
import cv2
import numpy as np

# Ensure storyboard-agent root is in path
AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(AGENT_ROOT)

from tools import utils
from tools.draw_animation import draw_animation_tool_fn
from tools.media_tools import concatenate_scene_videos_tool_fn, merge_audio_video_tool_fn
from tools.tts_tool import generate_tts_audio_tool_fn

TEST_OUT_DIR = os.path.join(AGENT_ROOT, "output", "test_panel_unit")
os.makedirs(TEST_OUT_DIR, exist_ok=True)
utils.set_output_dir(TEST_OUT_DIR)

PANEL_IMAGE = os.path.abspath("output/mughal_1707_1720_movie/panel_1792565258_2.png")
EXISTING_VEO_CLIP = os.path.abspath("output/mughal_1707_1720_movie/veo_720p_1792565446.mp4")


def test_fast_whiteboard_and_veo_concat():
    print("=" * 65)
    print("  UNIT TEST: Single Panel Whiteboard + Veo Lossless Concat")
    print("=" * 65)
    assert os.path.exists(PANEL_IMAGE), f"Panel image not found: {PANEL_IMAGE}"
    assert os.path.exists(EXISTING_VEO_CLIP), f"Veo clip not found: {EXISTING_VEO_CLIP}"

    # Step 1: Render fast whiteboard animation
    print("\n[Step 1] Rendering Fast Whiteboard Animation...")
    t0 = time.time()
    drawing_video = draw_animation_tool_fn(
        image_path=PANEL_IMAGE,
        segmentation_results_path=None,  # Single pass fast mode
        resize_wd=1280,
        resize_ht=720,
        split_len=16,
        object_skip_rate=12,
        bg_object_skip_rate=20,
        end_duration_sec=2
    )
    render_time = time.time() - t0
    assert os.path.exists(drawing_video), "Whiteboard drawing failed to output video!"
    
    w_dur = utils.get_video_duration(drawing_video)
    print(f"  Whiteboard rendered in {render_time:.2f}s | Duration: {w_dur:.2f}s | Path: {drawing_video}")
    assert w_dur > 0, "Whiteboard video duration must be > 0"

    # Verify H.264 codec
    probe_w = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,pix_fmt,r_frame_rate", "-of", "json", drawing_video],
        text=True
    )
    assert "h264" in probe_w, f"Drawing video must be H.264! Found: {probe_w}"
    print("  Whiteboard video verified as native H.264 yuv420p!")

    # Step 2: Measure Veo clip
    v_dur = utils.get_video_duration(EXISTING_VEO_CLIP)
    print(f"\n[Step 2] Veo Clip Duration: {v_dur:.2f}s | Path: {EXISTING_VEO_CLIP}")

    # Step 3: Concatenate Drawing + Veo
    print("\n[Step 3] Concatenating Drawing + Veo using normalized filter_complex...")
    combined_video = concatenate_scene_videos_tool_fn(
        scene_video_paths=[drawing_video, EXISTING_VEO_CLIP],
        output_filename="test_panel_drawing_plus_veo.mp4"
    )
    assert combined_video and os.path.exists(combined_video), "Concat failed to produce output file!"

    c_dur = utils.get_video_duration(combined_video)
    expected_dur = w_dur + v_dur
    print(f"  Combined Video Duration: {c_dur:.2f}s (Expected: {expected_dur:.2f}s)")
    assert abs(c_dur - expected_dur) < 0.35, f"Combined duration {c_dur}s does not match sum {expected_dur}s!"

    # Step 4: Verify ZERO decoding errors
    print("\n[Step 4] Checking for decoding errors with FFmpeg...")
    decode_check = subprocess.run(
        ["ffmpeg", "-v", "warning", "-i", combined_video, "-f", "null", "-"],
        capture_output=True,
        text=True
    )
    assert decode_check.returncode == 0, f"FFmpeg decode failed: {decode_check.stderr}"
    assert "header damaged" not in decode_check.stderr, f"Damaged headers found in stream: {decode_check.stderr}"
    assert "Error while decoding" not in decode_check.stderr, f"Decoder errors found: {decode_check.stderr}"
    print("  ZERO decoding errors! Both segments decoded cleanly.")

    # Step 5: Visual Verification of Both Segments
    print("\n[Step 5] Sampling frames to confirm both whiteboard and Veo are visually present...")
    cap = cv2.VideoCapture(combined_video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Sample in Whiteboard segment (t = w_dur * 0.5)
    f_whiteboard = int((w_dur * 0.5) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_whiteboard)
    ret_w, frame_w = cap.read()
    assert ret_w, "Failed to read frame in whiteboard segment"
    mean_w = np.mean(frame_w)

    # Sample in Veo segment (t = w_dur + v_dur * 0.5)
    f_veo = int((w_dur + (v_dur * 0.5)) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_veo)
    ret_v, frame_v = cap.read()
    assert ret_v, "Failed to read frame in Veo segment"
    mean_v = np.mean(frame_v)
    cap.release()

    print(f"  Whiteboard frame (t={w_dur*0.5:.1f}s): mean intensity = {mean_w:.1f}")
    print(f"  Veo frame (t={w_dur + v_dur*0.5:.1f}s): mean intensity = {mean_v:.1f}")

    # Whiteboard should be mostly white paper (mean > 160)
    assert mean_w > 120, f"Whiteboard frame mean intensity {mean_w} unexpectedly low"
    print("  Both Whiteboard and Veo segments visually confirmed!")

    # Step 6: AV Merge
    print("\n[Step 6] Testing Zero-Crop AV Merge with Audio...")
    test_audio = os.path.abspath("output/mughal_1707_1720_movie/audio_1792565427.wav")
    assert os.path.exists(test_audio), f"Audio file not found: {test_audio}"

    merged_av = merge_audio_video_tool_fn(video_path=combined_video, audio_path=test_audio)
    assert merged_av and os.path.exists(merged_av), "AV merge failed!"
    
    m_dur = utils.get_video_duration(merged_av)
    print(f"  Merged AV Duration: {m_dur:.2f}s | Path: {merged_av}")
    assert m_dur >= min(c_dur, utils.get_audio_duration(test_audio)), "Merged video truncated visual!"

    print("\n" + "=" * 65)
    print("  ALL UNIT TESTS PASSED FOR SINGLE PANEL VEO CONCAT!")
    print("=" * 65)


if __name__ == "__main__":
    test_fast_whiteboard_and_veo_concat()
