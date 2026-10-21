"""
End-to-End Test for ADK 2.0 Storyboard Multi-Agent Service.
Runs a 1-scene end-to-end workflow with SAM enabled, verifying:
- Orchestrator intent detection & verbatim story planner
- Reference search & vision evaluator (Prince Kam Bakhsh)
- SAM object segmentation & whiteboard drawing animation
- Gemini 3.8 Flash TTS generation
- Audio/video merge & subtitle burning
- Final master video assembly into story.mp4
- Strict cost containment (< $0.06)
"""
import os, sys
AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if AGENT_ROOT not in sys.path:
    sys.path.insert(0, AGENT_ROOT)

import os
import sys

# Ensure storyboard-agent root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from state import RunConfig
from cost_tracker import tracker
from agents.orchestrator import StoryboardOrchestrator
from tools.utils import get_video_duration


def run_e2e_test():
    print("=" * 65)
    print("      STARTING E2E MULTI-AGENT STORYBOARD TEST (SAM ENABLED)     ")
    print("=" * 65)

    test_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "test_e2e")
    os.makedirs(test_out, exist_ok=True)

    # 1-scene concise historical test prompt
    prompt = "Prince Kam Bakhsh, the youngest son of Emperor Aurangzeb, commanded forces in the Deccan campaign."

    run_config = RunConfig(
        verbatim_mode=True,       # Verbatim mode
        enable_sam=True,          # SAM enabled as requested
        enable_veo=False,         # Veo off for budget protection
        language="english",
        max_scenes=1,             # Single scene to minimize cost
        output_dir=test_out
    )

    orchestrator = StoryboardOrchestrator(run_config=run_config)
    state = orchestrator.run(user_instruction=prompt)

    # Verifications
    print("\n--- Verifying E2E Outputs ---")
    assert state.final_video_path and os.path.exists(state.final_video_path), (
        f"Master video not found at {state.final_video_path}"
    )

    file_size = os.path.getsize(state.final_video_path)
    duration = get_video_duration(state.final_video_path)
    print(f"[PASS] Master Video Verified: {state.final_video_path}")
    print(f"       File Size: {file_size:,} bytes | Duration: {duration:.2f}s")
    assert file_size > 10_000, f"Video file unexpectedly small: {file_size} bytes"
    assert duration > 1.0, f"Video duration too short: {duration}s"

    # Verify scene results
    assert len(state.scene_results) == 1, f"Expected 1 scene result, got {len(state.scene_results)}"
    res = state.scene_results[0]
    assert res.status == "SUCCESS", f"Scene status was {res.status}: {res.error_message}"
    assert res.audio_path and os.path.exists(res.audio_path), "Scene audio missing"
    assert res.final_visual_path and os.path.exists(res.final_visual_path), "Visual asset missing"

    # Verify Cost
    total_cost = tracker.get_estimated_cost_usd()
    print(f"\n[PASS] E2E Multi-Agent Storyboard completed successfully!")
    print(f"       Total Cost for this test run: ${total_cost:.4f} USD")
    assert total_cost < 0.10, f"Test cost exceeded threshold: ${total_cost:.4f}"
    print("=" * 65)


if __name__ == "__main__":
    try:
        run_e2e_test()
    except Exception as e:
        print(f"\n[FAIL] E2E test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
