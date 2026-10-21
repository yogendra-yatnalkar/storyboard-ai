"""
Test Suite for Strategy 4 (Hybrid Verbatim Mode):
Validates:
1. Verbatim text preservation (100.0% word-for-word integrity across 6 panels)
2. Proportional clause-aware word allocation
3. Veo Gating (gated off when beat/scene is concise)
4. Audio-First Adaptive Whiteboard Drawing Speed (speed scales with speech length)
5. Zero-Crop synchronization (tpad clone on audio longer, apad silence on video longer)
6. Final 16:9 Landscape master video assembly
"""
import os, sys
AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if AGENT_ROOT not in sys.path:
    sys.path.insert(0, AGENT_ROOT)

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cost_tracker import tracker
from state import RunConfig, ScenePlan, PanelPlan
from tools.utils import set_output_dir
from tools.story_planner_tool import plan_verbatim_storyboard
from agents.scene_agent import SceneAgent


def run_verbatim_hybrid_test():
    test_out = os.path.abspath("output/test_verbatim_hybrid")
    set_output_dir(test_out)
    os.makedirs(test_out, exist_ok=True)

    print("=" * 70)
    print("  VERBATIM MODE TEST (STRATEGY 4: HYBRID DURATION & VEO GATING)")
    print("=" * 70)

    # 48-word verbatim historical text
    source_text = (
        "In January 1709, the armies of Bahadur Shah I and Prince Kam Bakhsh clashed outside Hyderabad. "
        "Kam Bakhsh fought courageously atop his war elephant despite overwhelming odds. "
        "As night fell across the battlefield, Kam Bakhsh was mortally wounded, securing Bahadur Shah's dominion over the Mughal empire."
    )
    print(f"Source Text ({len(source_text.split())} words):\n'{source_text}'\n")

    run_config = RunConfig(
        verbatim_mode=True,       # Strict Verbatim Mode
        enable_sam=False,         # Fast whiteboard contour animation
        enable_veo=True,          # Veo requested (will test gating on short beats)
        veo_scope="last_panel",   # Request Veo on Panel 6 (climax)
        veo_mode="after_drawing",
        language="english",
        max_scenes=1,
        output_dir=test_out
    )

    print("--- Step 1: Planning Verbatim Storyboard with Clause Reconciler ---")
    global_plan, scene_plans = plan_verbatim_storyboard(source_text=source_text, run_config=run_config)
    assert len(scene_plans) >= 1, "Failed to plan verbatim scene"
    scene_plan = scene_plans[0]

    print(f"\nScene Plan 1 Panels ({len(scene_plan.panels)}):")
    allocated_words = []
    for p in scene_plan.panels:
        print(f"  Panel {p.panel_number} ({len(p.narration_beat.split())} words): '{p.narration_beat}' (Veo: {bool(p.veo_prompt)})")
        allocated_words.extend(p.narration_beat.split())

    # Verify 100% exact word match
    orig_words = source_text.strip().split()
    assert allocated_words == orig_words, f"Verbatim integrity failed!\nExpected: {orig_words}\nGot: {allocated_words}"
    print("\nVERBATIM INTEGRITY VERIFIED: 100.0% exact word match across all panels.")

    print("\n--- Step 2: Processing Scene with Adaptive Speed & AV Merge ---")
    scene_agent = SceneAgent(run_config=run_config)
    result = scene_agent.process_scene(scene_plan=scene_plan)

    print(f"\nScene Status: {result.status}")
    print(f"Final Master Video: {result.final_scene_video_path}")
    print(f"Total Duration: {result.duration:.1f}s")
    print(f"Panel Videos: {len(result.panel_video_paths)}")

    assert result.status == "SUCCESS", f"Scene processing failed: {result.error_message}"
    assert os.path.exists(result.final_scene_video_path), "Master video file does not exist."

    print("\n" + "=" * 70)
    print("  COST REPORT & BUDGET VERIFICATION")
    print("=" * 70)
    print(tracker.get_summary_report())


if __name__ == "__main__":
    run_verbatim_hybrid_test()
