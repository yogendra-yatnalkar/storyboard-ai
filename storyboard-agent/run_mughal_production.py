"""
End-to-End Autonomous 5-Part Production Runner for Mughal Empire (1707-1727).
Executes Turn 1 (Grounding + 5-Part Screenplay Architecture), auto-approves
at the HITL gate upon verifying 5 parts, and drives Turn 2 full video production.
"""
import asyncio
import datetime
import json
import os
import sys

# Ensure unbuffered utf-8 output
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_DIR = os.path.join(ROOT_DIR, "storyboard-agent")
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agents.director_agent import create_director_agent
from cost_tracker import tracker
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
from tools import utils

PROMPT = (
    "Mughal empire for next 20 years after death of Emperor Aurangazeb. "
    "Make it as much detailed as possible within 5 parts please. "
    "Please perform web-search  for detailed information."
)


async def main():
    print("=" * 80)
    print("      MUGHAL EMPIRE (1707-1727): AUTONOMOUS 5-PART SHOWRUNNER PRODUCTION")
    print("=" * 80)
    print(f"Topic: '{PROMPT}'")

    # 1. Dedicated dated output directory
    output_dir = utils.init_dated_output_dir(
        title="Mughal Empire 1707 to 1727 Epic",
        base_dir=os.path.join(ROOT_DIR, "output")
    )
    utils.set_output_dir(output_dir)
    print(f"Master Run Directory: {output_dir}\n")

    # 2. Setup ADK Runner & InMemorySessionService
    session_service = InMemorySessionService()
    director = create_director_agent()
    runner = Runner(
        agent=director,
        session_service=session_service,
        app_name="storyboard_agent"
    )

    init_state = {
        "story_topic": PROMPT,
        "requested_parts": 5,
        "verbatim_mode": False,
        "output_dir": output_dir,
        "folder_name": os.path.basename(output_dir)
    }

    session = await session_service.create_session(
        app_name="storyboard_agent",
        user_id="director_cli",
        state=init_state
    )

    # -------------------------------------------------------------------------
    # STAGE 1: TURN 1 (Grounding & Screenplay Architecture)
    # -------------------------------------------------------------------------
    print("=" * 80)
    print(" STAGE 1: Real-Time Grounding & 5-Part Screenplay Decomposition")
    print("=" * 80)

    user_msg_turn1 = types.Content(
        role="user",
        parts=[types.Part.from_text(text=PROMPT)]
    )

    paused_at_hitl = False
    hitl_question = ""

    async for event in runner.run_async(
        session_id=session.id,
        user_id="director_cli",
        new_message=user_msg_turn1
    ):
        content = getattr(event, "content", None)
        if content and content.parts:
            for p in content.parts:
                if getattr(p, "text", None):
                    print(f"[{event.author or 'Agent'}] {p.text.strip()}")
                if getattr(p, "function_call", None):
                    fc = p.function_call
                    if "request_input" in fc.name:
                        paused_at_hitl = True
                        if hasattr(fc, "args") and isinstance(fc.args, dict):
                            hitl_question = fc.args.get("message", "")

    # Retrieve planned screenplay from state
    updated_session = await session_service.get_session(
        app_name="storyboard_agent",
        user_id="director_cli",
        session_id=session.id
    )
    script = updated_session.state.get("storyboard_script", [])

    print("\n" + "=" * 80)
    print(f" SCREENPLAY VERIFICATION: Total Parts Planned = {len(script)}")
    print("=" * 80)

    for p in script:
        p_num = p.get("part_number")
        title = p.get("title")
        synopsis = p.get("synopsis")
        subs = p.get("subscenes", [])
        veo_cnt = sum(1 for s in subs if s.get("is_veo_candidate"))
        print(f"\n[PART {p_num}]: {title}")
        print(f"  Synopsis: {synopsis}")
        print(f"  Portrait Queries: {p.get('search_queries')}")
        print(f"  Subscenes: {len(subs)} panels ({veo_cnt} Veo candidates)")
        for s in subs:
            tag = "[Veo 8s Motion]" if s.get("is_veo_candidate") else "[Whiteboard Draw]"
            print(f"    - [{p_num}.{s.get('subscene_number')}] {tag} {s.get('narration')[:70]}...")

    if len(script) < 5:
        print(f"\n[ABORT]: Expected 5 parts but received {len(script)}. Terminating.")
        return

    # -------------------------------------------------------------------------
    # STAGE 2: TURN 2 (Auto-Approval & Multi-Part Video Production)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" STAGE 2: Human-in-the-Loop AUTO-APPROVED (5 Parts Verified)")
    print(" Resuming ADK Workflow -> Executing Video Production & Concatenation")
    print("=" * 80)

    user_msg_turn2 = types.Content(
        role="user",
        parts=[types.Part.from_text(text="Approved. Proceed with video production and assembly.")]
    )

    async for event in runner.run_async(
        session_id=session.id,
        user_id="director_cli",
        new_message=user_msg_turn2
    ):
        content = getattr(event, "content", None)
        if content and content.parts:
            for p in content.parts:
                if getattr(p, "text", None):
                    print(f"[{event.author or 'Agent'}] {p.text.strip()}")

    # -------------------------------------------------------------------------
    # FINAL INSPECTION & EXPENDITURE REPORT
    # -------------------------------------------------------------------------
    final_session = await session_service.get_session(
        app_name="storyboard_agent",
        user_id="director_cli",
        session_id=session.id
    )

    master_path = final_session.state.get("final_movie_path") or final_session.state.get("master_movie_path")

    print("\n" + "=" * 80)
    print(" PRODUCTION COMPLETE: INVENTORY REPORT")
    print("=" * 80)
    print(f"Run Output Directory: {output_dir}")
    if master_path and os.path.exists(master_path):
        size_mb = os.path.getsize(master_path) / (1024 * 1024)
        print(f"Final Master Movie: {os.path.basename(master_path)} ({size_mb:.2f} MB)")
    else:
        print(f"Master Movie Path: {master_path}")

    print("\nFiles Created in Output Folder:")
    if os.path.exists(output_dir):
        for f in sorted(os.listdir(output_dir)):
            fp = os.path.join(output_dir, f)
            sz = os.path.getsize(fp) / 1024
            print(f"  - {f:45s} ({sz:8.1f} KB)")

    print("\n" + tracker.get_summary_report())
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
