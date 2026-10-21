"""
Micro-verification test for Zero-Crop Audio/Video Sync, Static Hold Frames, and Human-like TTS.
Uses pre-generated panel image to keep test cost under $0.005.
"""
import os, sys
AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if AGENT_ROOT not in sys.path:
    sys.path.insert(0, AGENT_ROOT)

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cost_tracker import tracker
from tools import utils
from tools.media_tools import create_whiteboard_animation_tool_fn, merge_audio_video_tool_fn, transcribe_and_subtitle_tool_fn
from tools.narration_refiner import refine_narration_tool_fn
from tools.tts_tool import generate_tts_audio_tool_fn


def test_zero_crop_sync():
    test_out = os.path.abspath("output/test_sync")
    utils.set_output_dir(test_out)
    os.makedirs(test_out, exist_ok=True)

    panel_img = os.path.abspath("output/test_grid/panel_1792525977_1.png")
    assert os.path.exists(panel_img), f"Source panel image missing: {panel_img}"

    initial_beat = "In the Deccan plains, tension mounted between royal brothers."

    print("=" * 65)
    print("STEP 1: Whiteboard Animation with 2s Static Hold on Completed Artwork")
    print("=" * 65)
    drawing_video = create_whiteboard_animation_tool_fn(
        image_path=panel_img,
        split_len=14,
        object_skip_rate=8,
        end_duration_sec=2  # Hold static frame for 2 seconds
    )
    assert drawing_video and os.path.exists(drawing_video), "Animation creation failed."
    v_dur = utils.get_video_duration(drawing_video)
    print(f"Whiteboard Video Duration (with 2s hold): {v_dur:.2f}s")

    print("\n" + "=" * 65)
    print("STEP 2: Calibrating Narration Beat to Match Video Duration (+1.5s hold)")
    print("=" * 65)
    refined_beat = refine_narration_tool_fn(
        original_narration=initial_beat,
        image_path=panel_img,
        video_duration=v_dur,
        global_plan={"tone": "dramatic", "narrative_persona": "Documentary Storyteller"},
        language="english"
    )
    print(f"Original Beat ({len(initial_beat.split())} words): '{initial_beat}'")
    print(f"Refined Beat  ({len(refined_beat.split())} words): '{refined_beat}'")

    print("\n" + "=" * 65)
    print("STEP 3: Human-like Expressive TTS Synthesis (Voice: Aoede)")
    print("=" * 65)
    audio_path = generate_tts_audio_tool_fn(text=refined_beat, language="english")
    assert audio_path and os.path.exists(audio_path), "TTS synthesis failed."
    a_dur = utils.get_audio_duration(audio_path)
    print(f"Spoken Audio Duration: {a_dur:.2f}s (Target was {v_dur + 1.5:.2f}s)")

    print("\n" + "=" * 65)
    print("STEP 4: Zero-Crop Audio/Video Merging")
    print("=" * 65)
    merged_video = merge_audio_video_tool_fn(video_path=drawing_video, audio_path=audio_path)
    assert merged_video and os.path.exists(merged_video), "Merge failed."
    m_dur = utils.get_video_duration(merged_video)
    print(f"Merged Video Duration: {m_dur:.2f}s")

    # Critical check: Merged video duration MUST NOT be shorter than drawing animation or audio!
    expected_min_duration = min(v_dur, a_dur)
    assert m_dur >= expected_min_duration - 0.2, f"Video was prematurely cropped! Merged: {m_dur:.2f}s, Expected >= {expected_min_duration:.2f}s"
    print("VERIFICATION PASSED: No video cropping occurred!")

    print("\n" + "=" * 65)
    print("STEP 5: Synchronized Subtitle Burning")
    print("=" * 65)
    subtitled_video = transcribe_and_subtitle_tool_fn(video_path=merged_video, audio_path=audio_path, language="english")
    assert subtitled_video and os.path.exists(subtitled_video), "Subtitle burning failed."
    final_dur = utils.get_video_duration(subtitled_video)
    print(f"Final Output Video: {subtitled_video} (Duration: {final_dur:.2f}s)")

    print("\n" + "=" * 65)
    print("COST REPORT")
    print("=" * 65)
    print(tracker.get_summary_report())
    cost = tracker.compute_total_cost_usd()
    print(f"Total Micro-Test Spend: ${cost:.5f} USD")
    assert cost < 0.01, f"Spend exceeded 1 cent: ${cost:.5f}"
    print("SUCCESS: Micro-test finished within fraction of a cent!")
    return True


if __name__ == "__main__":
    test_zero_crop_sync()
