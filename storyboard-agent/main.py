"""
CLI Entrypoint for Google ADK Storyboard Multi-Agent Service.
Run with:
    ~/miniconda3/python.exe storyboard-agent/main.py
"""
import asyncio
import os
import sys

# Ensure storyboard-agent root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import config
from cost_tracker import tracker
from tools import utils
from agents.director_agent import create_director_agent


async def run_storyboard_cli():
    print("=" * 75)
    print("       STORYBOARD AI - GOOGLE ADK MULTI-AGENT ARCHITECTURE           ")
    print("=" * 75)

    prompt = input("\nEnter your video topic or verbatim text: ").strip()
    if not prompt:
        print("Error: Empty input.")
        return

    # 1. Narration Mode
    print("\n1. Select Narration Mode:")
    print("  [1] Creative Mode (Documentary dramatization, narrative arc - Default)")
    print("  [2] Strict Verbatim Mode (Locks input text with 0 alterations, drafts visuals for chunks)")
    mode_choice = input("Choice [1/2] (default 1): ").strip()
    verbatim_mode = mode_choice == "2"

    # 2. Number of Parts
    parts_input = input("\n2. Number of Parts (default 3, 6 sub-scenes each): ").strip()
    num_parts = int(parts_input) if parts_input.isdigit() and int(parts_input) > 0 else 3

    # 3. Output directory (dated, readable, title-slugified)
    out_dir = utils.init_dated_output_dir(title=prompt, base_dir="output")

    print("\n" + "=" * 75)
    print("  CONFIGURATION:")
    print(f"  - Mode: {'Verbatim' if verbatim_mode else 'Creative'}")
    print(f"  - Parts: {num_parts} (Total {num_parts * 6} Sub-Scenes)")
    print(f"  - Resolution: {config.OUTPUT_WIDTH}x{config.OUTPUT_HEIGHT}")
    print(f"  - Veo 720p: {config.ENABLE_VEO} ({config.VEO_SCOPE})")
    print(f"  - SAM 3 Segmentation: {config.ENABLE_SAM}")
    print(f"  - Output Dir: {out_dir}")
    print("=" * 75)

    # Initialize Sequential Director Agent & ADK Runner
    director = create_director_agent()
    session_service = InMemorySessionService()
    runner = Runner(
        agent=director,
        session_service=session_service,
        app_name="storyboard_agent"
    )

    session = await session_service.create_session(
        app_name="storyboard_agent",
        user_id="user_cli"
    )

    session.state["verbatim_mode"] = verbatim_mode
    session.state["requested_parts"] = num_parts
    session.state["story_topic"] = prompt

    agent_prompt = f"Please produce a {num_parts}-Part storyboard movie on: '{prompt}'. Mode: {'verbatim' if verbatim_mode else 'creative'}."

    print("\n--- [STAGE 1: Story Architecture & Research] ---\n")
    user_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=agent_prompt)]
    )

    paused_for_hitl = False
    hitl_prompt_message = "Do you approve this screenplay to proceed with video generation?"

    async for event in runner.run_async(
        session_id=session.id,
        user_id="user_cli",
        new_message=user_content
    ):
        content = getattr(event, "content", None)
        if content and content.parts:
            for p in content.parts:
                if getattr(p, "text", None):
                    print(f"[{event.author}]: {p.text}")
                if getattr(p, "function_call", None):
                    fc = p.function_call
                    print(f"[{event.author} Tool]: {fc.name}")
                    if "request_input" in fc.name:
                        paused_for_hitl = True
                        if hasattr(fc, "args") and isinstance(fc.args, dict) and "message" in fc.args:
                            hitl_prompt_message = fc.args["message"]

    # STAGE 2: Human-in-the-Loop Review Gate
    if paused_for_hitl:
        print("\n" + "=" * 75)
        print("  [HUMAN-IN-THE-LOOP SCREENPLAY APPROVAL GATE]")
        print("=" * 75)
        print(f"Prompt: {hitl_prompt_message}")
        user_review = input("Enter [Y] to approve, [n] to cancel, or type custom adjustments:\n> ").strip()

        if user_review.lower() in ["n", "no", "cancel"]:
            print("\n[Director] Production halted by human user.")
            return

        approval_text = "Approved. Proceed with video production and assembly." if (not user_review or user_review.lower() in ["y", "yes"]) else user_review
        turn2_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=approval_text)]
        )

        print("\n--- [STAGE 2: Part Production & Movie Assembly] ---\n")
        async for event in runner.run_async(
            session_id=session.id,
            user_id="user_cli",
            new_message=turn2_content
        ):
            content = getattr(event, "content", None)
            if content and content.parts:
                for p in content.parts:
                    if getattr(p, "text", None):
                        print(f"[{event.author}]: {p.text}")
                    if getattr(p, "function_call", None):
                        fc = p.function_call
                        print(f"[{event.author} Tool]: {fc.name}")

    # STAGE 3: Final Inspection & Expenditure Report
    updated_session = await session_service.get_session(
        app_name="storyboard_agent",
        user_id="user_cli",
        session_id=session.id
    )
    master_path = updated_session.state.get("final_movie_path") or updated_session.state.get("master_movie_path")
    if master_path and os.path.exists(master_path):
        dur = utils.get_video_duration(master_path)
        print("\n" + "=" * 75)
        print("  [FINAL MASTER MOVIE GENERATION COMPLETE]")
        print("=" * 75)
        print(f"  File Path: {master_path}")
        print(f"  Duration:  {dur:.2f} seconds ({dur/60:.1f} mins)")
        print("=" * 75)

    print("\n" + "=" * 75)
    print("  PRODUCTION REPORT & BUDGET EXPENDITURE")
    print("=" * 75)
    print(tracker.get_summary_report())


def main():
    asyncio.run(run_storyboard_cli())


if __name__ == "__main__":
    main()
