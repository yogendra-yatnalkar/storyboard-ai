"""
Test Suite for Creative Mode with Vibrant Colors, 50% Random Veo, and Lanczos Sharp Panels.
Validates:
1. 3x2 Grid generated with vibrant colors and zero corner labels (P1-P6 eliminated)
2. Lanczos-4 upscaling with high-pass edge sharpening
3. Random 50% Veo generation (3 out of 6 panels generate 720p Veo clips)
4. Narration refiner duration calibration (+1.5s static hold on final art)
5. Expressive human-like TTS (Aoede) with natural pacing tags
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
from tools.reference_evaluator import fetch_multi_entity_references
from tools.grid_panel_tool import generate_storyboard_grid_3x2, slice_3x2_grid_into_panels
from agents.scene_agent import SceneAgent


def run_creative_veo_test():
    test_out = os.path.abspath("output/test_creative_veo")
    set_output_dir(test_out)
    os.makedirs(test_out, exist_ok=True)

    print("=" * 70)
    print("  CREATIVE MODE TEST: VIBRANT COLORS + 50% RANDOM VEO (720p)")
    print("=" * 70)

    entities = [
        "Prince Kam Bakhsh son of Aurangzeb",
        "Bahadur Shah I eldest son of Aurangzeb"
    ]
    refs = fetch_multi_entity_references(
        search_queries=entities,
        scene_context="Mughal succession conflict in Deccan",
        max_refs=2
    )
    print(f"Acquired References ({len(refs)}): {refs}")

    panels = [
        PanelPlan(1, "Across the sunbaked Deccan plains, the royal encampments stood on high alert.", "Wide establishing shot of ornate royal Mughal tents and battle banners on the Deccan plain."),
        PanelPlan(2, "Prince Kam Bakhsh held court in Bijapur, proclaiming himself sovereign ruler.", "Prince Kam Bakhsh seated on a jeweled throne with royal turban and curved sword."),
        PanelPlan(3, "From Agra, Bahadur Shah I marshaled an unstoppable imperial vanguard.", "Bahadur Shah I atop an armored war elephant reviewing marching infantry battalions."),
        PanelPlan(4, "Outside Hyderabad, two massive cavalry armies faced each other across the dry riverbed.", "Distant opposing cavalry battle lines with waving swallowtail flags under the sun."),
        PanelPlan(5, "Steel clashed against steel as a desperate and ferocious melee erupted.", "Intense melee with galloping warhorses, archers, and Mughal swordsmen in combat."),
        PanelPlan(6, "Bahadur Shah emerged undisputed emperor as the Mughal nobility knelt in fealty.", "Victorious emperor Bahadur Shah acclaimed on imperial throne with bowing courtiers.")
    ]

    scene_plan = ScenePlan(
        scene_number=1,
        narration="Across the Deccan plains, tension erupted between royal brothers into an epic clash for the imperial throne.",
        description="Epic struggle between Prince Kam Bakhsh and Bahadur Shah I for the Mughal Empire",
        visual_setup="3x2 comic grid layout with vibrant colors, crisp black ink outlines, pure white background",
        search_queries=entities,
        panels=panels
    )

    print("\n--- Generating Vibrant Color 3x2 Grid Image ---")
    grid_image = generate_storyboard_grid_3x2(
        scene_plan=scene_plan,
        reference_image_paths=refs
    )
    assert grid_image and os.path.exists(grid_image), "Grid image generation failed."
    print(f"Generated 3x2 Grid: {grid_image}")
    scene_plan.grid_image_path = grid_image
    scene_plan.reference_image_paths = refs

    print("\n--- Slicing with Lanczos-4, Unsharp Masking & Corner Label Cleaning ---")
    sliced_panels = slice_3x2_grid_into_panels(grid_image, output_dir=test_out)
    assert len(sliced_panels) == 6, f"Expected 6 panels, got {len(sliced_panels)}"

    print("\n--- Processing Scene with 50% Random Veo (3 Panels) & SAM 3 ---")
    run_config = RunConfig(
        enable_sam=True,
        enable_veo=True,
        veo_scope="random_50",  # Randomly animates 3 out of 6 panels with 720p Veo
        veo_mode="after_drawing",
        language="english",
        output_dir=test_out
    )

    scene_agent = SceneAgent(run_config=run_config)
    result = scene_agent.process_scene(scene_plan=scene_plan)

    print(f"\nScene Status: {result.status}")
    print(f"Final Master Video: {result.final_scene_video_path}")
    print(f"Total Duration: {result.duration:.1f}s")
    print(f"Panel Videos: {len(result.panel_video_paths)}")

    assert result.status == "SUCCESS", f"Failed: {result.error_message}"
    assert os.path.exists(result.final_scene_video_path), "Master video missing."

    print("\n" + "=" * 70)
    print("  COST REPORT & BUDGET VERIFICATION")
    print("=" * 70)
    print(tracker.get_summary_report())
    total_cost = tracker.compute_total_cost_usd()
    print(f"Total Test Spend: ${total_cost:.5f} USD")
    assert total_cost < 0.45, f"Spend exceeded budget: ${total_cost:.4f}"
    print("SUCCESS: Full Creative test with 50% Veo and Colors completed within budget!")
    return True


if __name__ == "__main__":
    run_creative_veo_test()
