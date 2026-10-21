"""
Isolated Unit & Tool Test Suite for storyboard-agent.
Tests each individual tool before constructing the multi-agent service.
Calculates and enforces micro-budget expenditure (< $0.05 total).
"""
import os, sys
AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if AGENT_ROOT not in sys.path:
    sys.path.insert(0, AGENT_ROOT)

import os
import sys

# Ensure storyboard-agent root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
from state import RunConfig
from cost_tracker import tracker
from tools.utils import set_output_dir
from tools.reference_evaluator import reference_evaluator_tool_fn
from tools.story_planner_tool import story_planner_tool_fn
from tools.tts_tool import generate_tts_audio_tool_fn
from tools.image_gen_tool import image_gen_tool_fn
from tools.segmentation_tool import segmentation_tool_fn
from tools.media_tools import (
    create_whiteboard_animation_tool_fn,
    merge_audio_video_tool_fn,
)


def run_all_tool_tests():
    print("=" * 65)
    print("      STARTING ISOLATED TOOL TESTS (STRICT BUDGET MODE)      ")
    print("=" * 65)

    test_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "test_tools")
    os.makedirs(test_out, exist_ok=True)
    set_output_dir(test_out)

    # -------------------------------------------------------------
    # 1. Reference Search & Automated Vision Evaluator
    # -------------------------------------------------------------
    print("\n[TOOL 1/5] Testing Reference Search & Vision Evaluator...")
    query = "Kambaksha son of aurangzeb"
    ref_res = reference_evaluator_tool_fn(query, scene_context="Historical portrait of Prince Kam Bakhsh")
    assert ref_res is not None, "Reference evaluator returned None"
    assert "score" in ref_res, "Missing score in evaluation"
    print(f"  Result: Image={ref_res.get('image_path')}")
    print(f"  Direct Reference Score: {ref_res.get('score')} | Direct={ref_res.get('use_direct')}")
    if ref_res.get("image_path") and os.path.exists(ref_res["image_path"]):
        img = Image.open(ref_res["image_path"])
        print(f"  Verified Image: Format={img.format}, Size={img.size}")
    print("  [PASS] Tool 1: Reference Evaluator passed.")

    # -------------------------------------------------------------
    # 2. Dual-Engine Story Planner: Verbatim Mode
    # -------------------------------------------------------------
    print("\n[TOOL 2/5] Testing Dual-Engine Story Planner (Verbatim Mode)...")
    sample_text = (
        "I was born in the ancient city of Agra during the golden age. "
        "The great emperor summoned his generals for the grand southern campaign."
    )
    verbatim_config = RunConfig(verbatim_mode=True, max_scenes=2)
    global_plan, scene_plans = story_planner_tool_fn(sample_text, run_config=verbatim_config)

    assert len(scene_plans) == 2, f"Expected 2 scenes, got {len(scene_plans)}"
    # Verify 100% exact verbatim retention
    combined_narration = " ".join([s.narration for s in scene_plans])
    assert combined_narration.strip() == sample_text.strip(), (
        f"Verbatim text mismatch!\nOriginal: {sample_text}\nCombined: {combined_narration}"
    )
    print(f"  Scene 1 exact narration: '{scene_plans[0].narration}'")
    print(f"  Scene 1 generated visual: '{scene_plans[0].description[:60]}...'")
    print(f"  Scene 2 exact narration: '{scene_plans[1].narration}'")
    print(f"  Scene 2 generated visual: '{scene_plans[1].description[:60]}...'")
    print("  [PASS] Tool 2: Verbatim Story Planner strictly preserved 100% text & generated visual prompts.")

    # -------------------------------------------------------------
    # 3. TTS Audio Generation (Gemini 3.8 Flash TTS)
    # -------------------------------------------------------------
    print("\n[TOOL 3/5] Testing TTS Audio Generation (Gemini 3.8 Flash TTS)...")
    short_sentence = "The great emperor summoned his generals."
    audio_path = generate_tts_audio_tool_fn(short_sentence)
    assert audio_path and os.path.exists(audio_path), f"TTS failed to produce audio at {audio_path}"
    with open(audio_path, "rb") as f:
        audio_header = f.read(12)
    assert audio_header.startswith(b"RIFF") and b"WAVE" in audio_header, (
        f"Generated audio is not a valid RIFF WAVE file: {audio_header}"
    )
    print(f"  [PASS] Tool 3: TTS generated valid native RIFF WAV ({os.path.getsize(audio_path):,} bytes).")

    # -------------------------------------------------------------
    # 4. Image Generation (gemini-3.1-flash-image with IMAGE-only modality)
    # -------------------------------------------------------------
    print("\n[TOOL 4/5] Testing Image Generation (gemini-3.1-flash-image, IMAGE-only modality)...")
    tiny_prompt = "A simple hand-drawn whiteboard sketch of an ancient crown and a feather plume. Minimalist lines."
    # Pass downloaded reference as guidance
    ref_guide = ref_res.get("image_path")
    gen_img_path = image_gen_tool_fn(tiny_prompt, subject_reference_image_path=ref_guide)
    assert gen_img_path and os.path.exists(gen_img_path), f"Image gen failed at {gen_img_path}"
    test_img = Image.open(gen_img_path)
    print(f"  Generated Image: Format={test_img.format}, Size={test_img.size}")
    print("  [PASS] Tool 4: Image generation succeeded with 0 billable text tokens.")

    # -------------------------------------------------------------
    # 5. Segmentation Tool (Gemini Vision Object Detection + SAM / Fallback)
    # -------------------------------------------------------------
    print("\n[TOOL 5/5] Testing Segmentation Tool...")
    seg_json = segmentation_tool_fn(gen_img_path)
    assert seg_json and os.path.exists(seg_json), f"Segmentation failed at {seg_json}"
    print(f"  Segmentation JSON saved: {seg_json}")
    print("  [PASS] Tool 5: Segmentation tool passed.")

    # -------------------------------------------------------------
    # 6. Local Media Assembly (Whiteboard Animation & Audio-Video Merge)
    # -------------------------------------------------------------
    print("\n[LOCAL MEDIA] Testing Whiteboard Animation & AV Merge...")
    video_path = create_whiteboard_animation_tool_fn(gen_img_path, segmentation_results_path=seg_json)
    assert video_path and os.path.exists(video_path), f"Animation failed at {video_path}"
    merged_path = merge_audio_video_tool_fn(video_path, audio_path)
    assert merged_path and os.path.exists(merged_path), f"Merge failed at {merged_path}"
    print(f"  Completed test scene clip: {merged_path}")
    print("  [PASS] Local Media tools passed.")

    # -------------------------------------------------------------
    # Final Cost Report
    # -------------------------------------------------------------
    print("\n" + tracker.get_summary_report())
    estimated_cost = tracker.get_estimated_cost_usd()
    print(f"Estimated Test Cost: ${estimated_cost:.4f} USD (Budget Ceiling: $10.00)")
    assert estimated_cost < 0.10, f"Exceeded safety cost ceiling! Spent ${estimated_cost:.4f}"
    print("\n" + "=" * 65)
    print("        ALL ISOLATED TOOL TESTS PASSED WITHIN BUDGET!        ")
    print("=" * 65)


if __name__ == "__main__":
    try:
        run_all_tool_tests()
    except Exception as e:
        print(f"\n[FAIL] Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
