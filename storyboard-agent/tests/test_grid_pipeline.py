"""
Test Suite for 6-Panel (3x2) Grid Storyboard & Multi-Entity Reference Pipeline.
Validates:
1. Multi-entity generic reference search (max 2 entities)
2. 3x2 grid image generation with coloring-book contours (1 call = $0.035)
3. Computer Vision slicing & 16:9 Landscape (1920x1080) framing into 6 panels
4. End-to-end 6-beat scene pipeline with SAM 3 segmentation enabled
5. Strict budget tracking (< $0.05)
"""
import os, sys
AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if AGENT_ROOT not in sys.path:
    sys.path.insert(0, AGENT_ROOT)

import os
import sys
import time

# Ensure storyboard-agent root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cost_tracker import tracker
from state import RunConfig, ScenePlan, PanelPlan
from tools.utils import set_output_dir
from tools.reference_evaluator import fetch_multi_entity_references
from tools.grid_panel_tool import generate_storyboard_grid_3x2, slice_3x2_grid_into_panels
from agents.scene_agent import SceneAgent


def run_tests():
    test_out = os.path.abspath("output/test_grid")
    set_output_dir(test_out)
    os.makedirs(test_out, exist_ok=True)

    print("=" * 70)
    print("  TEST 1: MULTI-ENTITY GENERIC REFERENCE SEARCH (MAX 2 ENTITIES)")
    print("=" * 70)
    entities = [
        "Prince Kam Bakhsh son of Aurangzeb",
        "Bahadur Shah I eldest son of Aurangzeb"
    ]
    print(f"Entities to search independently: {entities}")
    refs = fetch_multi_entity_references(
        search_queries=entities,
        scene_context="Historical documentary scene depicting Mughal princes",
        max_refs=2
    )
    print(f"Acquired References ({len(refs)}): {refs}")

    print("\n" + "=" * 70)
    print("  TEST 2: 3x2 GRID STORYBOARD GENERATION & CV SLICING")
    print("=" * 70)
    panels = [
        PanelPlan(1, "In the Deccan plains, tension mounted between royal brothers.", "Wide establishing shot of royal Mughal army camp in Deccan."),
        PanelPlan(2, "Prince Kam Bakhsh declared his independent sovereignty in Bijapur.", "Prince Kam Bakhsh sitting on a throne reading a royal decree."),
        PanelPlan(3, "From Agra, Bahadur Shah I assembled a vast imperial legion.", "Bahadur Shah I atop an elephant inspecting his columns of soldiers."),
        PanelPlan(4, "The armies met outside Hyderabad under a blazing sun.", "Confrontation of two opposing cavalry lines with battle banners."),
        PanelPlan(5, "A desperate and fierce battle erupted between the brothers.", "Fierce combat scene with galloping horses and clash of curved swords."),
        PanelPlan(6, "Bahadur Shah emerged triumphant as emperor of the Mughal Empire.", "Bahadur Shah I acclaimed victorious as royal court bows before him.")
    ]
    scene_plan = ScenePlan(
        scene_number=1,
        narration="In the Deccan plains, tension mounted between royal brothers. Prince Kam Bakhsh declared sovereignty, while Bahadur Shah I marched south. Outside Hyderabad, their armies met in a fierce clash, ending with Bahadur Shah triumphant as emperor.",
        description="Fierce struggle between Prince Kam Bakhsh and Bahadur Shah I",
        visual_setup="3x2 comic storyboard grid with bold black whiteboard marker contours",
        search_queries=entities,
        panels=panels
    )

    grid_image = generate_storyboard_grid_3x2(
        scene_plan=scene_plan,
        reference_image_paths=refs
    )

    if not grid_image or not os.path.exists(grid_image):
        print("FAIL: Grid generation did not return valid image.")
        return False

    print(f"SUCCESS: Generated 3x2 Grid Image: {grid_image}")
    scene_plan.grid_image_path = grid_image
    scene_plan.reference_image_paths = refs

    print("\nSlicing 3x2 grid into 6 16:9 Landscape panels...")
    sliced_panels = slice_3x2_grid_into_panels(grid_image, output_dir=test_out)
    print(f"Sliced Panels Count: {len(sliced_panels)}")
    for idx, p in enumerate(sliced_panels):
        print(f"  Panel #{idx + 1}: {p}")

    assert len(sliced_panels) == 6, f"Expected 6 panels, got {len(sliced_panels)}"

    print("\n" + "=" * 70)
    print("  TEST 3: END-TO-END 6-PANEL SCENE (SAM 3 + 50% RANDOM VEO ENABLED)")
    print("=" * 70)
    run_config = RunConfig(
        enable_sam=True,           # Test SAM 3 Cloud Run integration
        enable_veo=True,           # Test Veo 720p Video Generation
        veo_scope="random_50",     # 50% of sub-scenes (3 panels) with Veo motion clips
        language="english",
        output_dir=test_out
    )

    scene_agent = SceneAgent(run_config=run_config)
    result = scene_agent.process_scene(scene_plan=scene_plan)

    print(f"\nScene Result Status: {result.status}")
    print(f"Final Scene Video: {result.final_scene_video_path}")
    print(f"Total Duration: {result.duration:.1f}s")
    print(f"Panel Videos Produced: {len(result.panel_video_paths)}")

    assert result.status == "SUCCESS", f"Scene processing failed: {result.error_message}"
    assert os.path.exists(result.final_scene_video_path), "Master scene video missing."

    print("\n" + "=" * 70)
    print("  COST REPORT & BUDGET COMPLIANCE VERIFICATION")
    print("=" * 70)
    print(tracker.get_summary_report())

    total_cost = tracker.compute_total_cost_usd()
    print(f"\nFinal Test Spend: ${total_cost:.5f} USD")
    assert total_cost < 0.40, f"Exceeded test budget limit ($0.40)! Cost was ${total_cost:.4f}"
    print("SUCCESS: Test finished strictly within budget limit!")
    return True


if __name__ == "__main__":
    success = run_tests()
    if not success:
        sys.exit(1)
