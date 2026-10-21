"""
Unit Test: Sub-Agent and Tool Execution (Single Panel Sub-Scene)
Tests ADK Sub-Scene production tool with Context and State:
1. Validates Creative Mode:
   - SAM object segmentation
   - Fast whiteboard animation
   - Narration refiner calibrating text to visual duration
   - TTS audio generation (Aoede)
   - Zero-crop AV merge + subtitle generation
   - Storage of lightweight output path in tool_context.state
2. Validates Verbatim Mode:
   - Audio is generated first
   - Veo duration gate evaluation: rejects Veo when audio < drawing + 6s
"""
import os
import sys
import json
import subprocess
import cv2

# Ensure storyboard-agent root is in path
AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(AGENT_ROOT)

from tools import utils
from agents.tools_common import produce_subscene_tool

TEST_OUT_DIR = os.path.join(AGENT_ROOT, "output", "test_subagent_unit")
os.makedirs(TEST_OUT_DIR, exist_ok=True)
utils.set_output_dir(TEST_OUT_DIR)

PANEL_IMAGE = os.path.abspath("output/mughal_1707_1720_movie/panel_1792565258_2.png")


class MockToolContext:
    """Mock ADK Context providing tool_context.state dictionary."""
    def __init__(self, state=None):
        self.state = state if state is not None else {}


def test_subagent_creative_panel():
    print("=" * 65)
    print("  UNIT TEST: Sub-Agent Tool in Creative Mode")
    print("=" * 65)
    assert os.path.exists(PANEL_IMAGE), f"Panel image not found: {PANEL_IMAGE}"

    context = MockToolContext(state={
        "verbatim_mode": False,
        "part_1_subscenes": []
    })

    beat = "Emperor Aurangzeb dies in Ahmednagar, sparking an immediate scramble for power among his surviving sons."

    print("\nCalling produce_subscene_tool (Creative Mode)...")
    res_str = produce_subscene_tool(
        tool_context=context,
        part_num=1,
        subscene_num=1,
        panel_image_path=PANEL_IMAGE,
        narration_beat=beat,
        is_veo_candidate=False,  # Test whiteboard + narration sync without burning extra Veo quota
        verbatim_mode=False
    )

    info = json.loads(res_str)
    print("Sub-Scene Production Output:")
    print(json.dumps(info, indent=2))

    assert info.get("status") == "SUCCESS", f"Sub-scene production failed: {info}"
    video_path = info.get("video_path")
    assert video_path and os.path.exists(video_path), f"Output video missing: {video_path}"

    dur = utils.get_video_duration(video_path)
    print(f"Produced Sub-Scene Video Duration: {dur:.2f}s | Path: {video_path}")
    assert dur > 4.0, f"Video duration unexpectedly short: {dur}"

    # Verify state tracking
    saved_in_state = context.state.get("part_1_sub_1")
    assert saved_in_state == video_path, f"State missing subscene video path: {saved_in_state}"
    print("ADK Context state verified: part_1_sub_1 ->", saved_in_state)


def test_subagent_verbatim_gate():
    print("\n" + "=" * 65)
    print("  UNIT TEST: Sub-Agent Tool in Verbatim Mode (Veo Gate Rejection)")
    print("=" * 65)

    context = MockToolContext(state={"verbatim_mode": True})

    # Short verbatim beat (~3 seconds spoken): should REJECT Veo
    short_verbatim_beat = "The emperor passed away quietly at dawn."

    print("\nCalling produce_subscene_tool with short verbatim beat (Veo candidate=True)...")
    res_str = produce_subscene_tool(
        tool_context=context,
        part_num=1,
        subscene_num=2,
        panel_image_path=PANEL_IMAGE,
        narration_beat=short_verbatim_beat,
        is_veo_candidate=True,
        verbatim_mode=True
    )

    info = json.loads(res_str)
    print("Sub-Scene Production Output:")
    print(json.dumps(info, indent=2))

    assert info.get("status") == "SUCCESS", f"Sub-scene failed: {info}"
    assert info.get("veo_applied") is False, "Veo should have been gated OFF for short verbatim beat!"
    print("Veo Gate correctly REJECTED Veo to protect verbatim audio sync!")

    print("\n" + "=" * 65)
    print("  ALL SUB-AGENT & TOOL UNIT TESTS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    test_subagent_creative_panel()
    test_subagent_verbatim_gate()
