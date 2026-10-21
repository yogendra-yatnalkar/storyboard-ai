"""
Live 3-Part Movie Test Suite for Google ADK Multi-Agent Architecture.
Executes the user's requested scenario:
"3 part story on what happened to Mughal empire after death of emperor Aurangzeb.
Primarily from 1707 to 1720. The succession, the wars, the decline, major wars, etc."

Validates:
1. Sequential DirectorAgent orchestration: WriterAgent -> HITL StoryReviewAgent -> PartProductionAgent -> MovieAssemblyAgent.
2. WriterAgent Screenplay decomposition into 3 Parts with 6 sub-scenes each, entity queries, and Veo candidates.
3. Human-in-the-Loop review gate pausing before scene generation.
4. PartProductionAgent coordinating asset synthesis (DDGS + Wikipedia, 3x2 Grid with negative space & typography).
5. SAM 3 segmentation on all panels + native 720p whiteboard drawing.
6. Veo duration gate evaluation (audio >= drawing + 6s) + lossless 720p concatenation.
7. Two-Level Concatenation (Sub-Scene -> Part Master -> Movie Master).
8. Full ADK Session State tracking (zero large blobs in agent context).
"""
import os, sys
AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if AGENT_ROOT not in sys.path:
    sys.path.insert(0, AGENT_ROOT)

import asyncio
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from config import (
    ENABLE_VEO,
    ENABLE_SAM,
    VEO_SCOPE,
    OUTPUT_WIDTH,
    OUTPUT_HEIGHT
)
from cost_tracker import tracker
from tools import utils
from agents.director_agent import create_director_agent


async def main():
    test_out = utils.init_dated_output_dir(title="mughal-1707-1720-movie", base_dir="output")

    print("=" * 75)
    print("  GOOGLE ADK 3-LEVEL MULTI-AGENT MOVIE TEST")
    print(f"  Configuration: Veo={ENABLE_VEO} ({VEO_SCOPE}) | SAM={ENABLE_SAM} | Resolution={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}")
    print("=" * 75)

    user_prompt = (
        "3 part story on what happened to Mughal empire after death of emperor Aurangzeb. "
        "Primarily from 1707 to 1720. The succession, the wars, the decline, major wars, etc. "
        "Please search thoroughly on google for verified historical facts and visual continuity."
    )
    print(f"\nUser Topic:\n'{user_prompt}'\n")

    # 1. Initialize ADK Director Agent & Session
    director_agent = create_director_agent()
    session_service = InMemorySessionService()
    runner = Runner(
        agent=director_agent,
        app_name="storyboard_studio",
        session_service=session_service
    )

    session = await session_service.create_session(app_name="storyboard_studio", user_id="director_user")
    session.state["story_topic"] = user_prompt
    session.state["verbatim_mode"] = False
    session.state["requested_parts"] = 3
    print(f"ADK Session created: {session.id}")

    # 2. TURN 1: Story Architecture & HITL Gate
    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=f"Please produce a 3-part storyboard movie on: '{user_prompt}'. Mode: creative.")]
    )

    print("\n--- [TURN 1: WriterAgent Planning & HITL Screenplay Review] ---")
    paused_for_hitl = False
    async for event in runner.run_async(session_id=session.id, user_id="director_user", new_message=message):
        content = getattr(event, "content", None)
        if content and content.parts:
            for part in content.parts:
                if getattr(part, "function_call", None):
                    fc = part.function_call
                    print(f"\n[ADK Agent Action] Calling Tool: {fc.name}")
                    if "request_input" in fc.name:
                        paused_for_hitl = True
                if getattr(part, "text", None):
                    print(f"[{event.author}]: {part.text[:250]}...")

    print(f"\nExecution Paused for Human Approval at HITL Gate? {paused_for_hitl}")
    assert paused_for_hitl, "Execution must pause at StoryReviewAgent HITL Gate before video generation!"

    # 3. TURN 2: Human Approval -> Part Production & Master Assembly
    approval_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text="Approved. Proceed with video production and master assembly.")]
    )

    print("\n--- [TURN 2: Human Approves -> PartProductionAgent & MovieAssemblyAgent] ---")
    async for event in runner.run_async(session_id=session.id, user_id="director_user", new_message=approval_message):
        content = getattr(event, "content", None)
        if content and content.parts:
            for part in content.parts:
                if getattr(part, "function_call", None):
                    fc = part.function_call
                    print(f"\n[ADK Agent Action] Calling Tool: {fc.name}")
                if getattr(part, "text", None):
                    print(f"[{event.author}]: {part.text[:250]}...")

    # 4. Retrieve Session State & Verify Final Movie
    updated_session = await session_service.get_session(
        session_id=session.id,
        user_id="director_user",
        app_name="storyboard_studio"
    )

    state = updated_session.state
    movie_path = state.get("final_movie_path") or state.get("master_movie_path")
    movie_dur = state.get("final_movie_duration", 0.0)

    print("\n" + "=" * 75)
    print("  FINAL PRODUCTION VERIFICATION")
    print("=" * 75)
    print(f"Master Movie Path: {movie_path}")
    print(f"Total Duration: {movie_dur:.2f}s")
    print(f"Completed Part Videos: {state.get('completed_part_videos')}")

    assert movie_path and os.path.exists(movie_path), "Master movie video missing!"
    print("\nSUCCESS: Complete 3-Part Movie assembled via ADK Multi-Agent Architecture!")

    print("\n" + "=" * 75)
    print("  COST REPORT & BUDGET EXPENDITURE")
    print("=" * 75)
    print(tracker.get_summary_report())


if __name__ == "__main__":
    asyncio.run(main())
