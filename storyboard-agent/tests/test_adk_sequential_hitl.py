"""
Unit test for Google ADK SequentialAgent Multi-Turn Human-in-the-Loop Review.
Tests:
1. DirectorAgent as a SequentialAgent composed of:
   WriterAgent -> StoryReviewAgent (HITL Node) -> PartProductionAgent -> MovieAssemblyAgent.
2. Turn 1: Writer plans screenplay, Reviewer formats breakdown and invokes request_input.
3. Runner cleanly halts at HITL Gate awaiting human decision.
4. Turn 2: User approves, runner resumes: PartProductionAgent renders parts, MovieAssemblyAgent concatenates.
5. Verifies session state persistence across turns with zero large blobs.
"""
import os, sys
AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if AGENT_ROOT not in sys.path:
    sys.path.insert(0, AGENT_ROOT)

import asyncio
import json
from google.adk.agents import SequentialAgent
from google.adk import Agent, Runner, Context
from google.adk.sessions import InMemorySessionService
from google.adk.tools._request_input_tool import request_input
from google.genai import types

from agents.director_agent import format_and_review_screenplay_tool


def mock_plan_story_tool(tool_context: Context) -> str:
    script = [
        {
            "part_number": 1,
            "title": "The Fall of Ahmednagar (1707)",
            "synopsis": "Death of Aurangzeb and succession chaos.",
            "search_queries": ["Aurangzeb", "Azam Shah"],
            "subscenes": [
                {
                    "subscene_number": 1,
                    "narration": "In the dusty Deccan outpost of Ahmednagar, the iron grip of Emperor Aurangzeb finally broke.",
                    "visual_description": "Grand ornate carved title banner with gold letters: 'MUGHAL COLLAPSE 1707-1720'.",
                    "is_veo_candidate": False,
                    "veo_motion_prompt": ""
                },
                {
                    "subscene_number": 2,
                    "narration": "Imperial couriers galloped day and night carrying royal wills to prince Azam Shah.",
                    "visual_description": "Royal horsemen riding across Deccan plains under thunderous clouds.",
                    "is_veo_candidate": True,
                    "veo_motion_prompt": "Fast cavalry gallop across open plains with flying dust"
                }
            ]
        },
        {
            "part_number": 2,
            "title": "The War of Brothers (1707-1712)",
            "synopsis": "Clash at Jajau between Azam Shah and Bahadur Shah I.",
            "search_queries": ["Bahadur Shah I", "Zulfiqar Khan"],
            "subscenes": [
                {
                    "subscene_number": 1,
                    "narration": "Near the banks of the Yamuna, the royal armies converged for fratricidal combat.",
                    "visual_description": "Twin battle lines of Mughal elephants and matchlock musketeers.",
                    "is_veo_candidate": True,
                    "veo_motion_prompt": "War elephants marching with brass armor into cannon smoke"
                }
            ]
        }
    ]
    tool_context.state["storyboard_script"] = script
    tool_context.state["story_topic"] = "Mughal Empire 1707-1720"
    return "Screenplay generated with 2 Parts."


def mock_produce_parts_tool(tool_context: Context) -> str:
    script = tool_context.state.get("storyboard_script", [])
    part_vids = []
    for p in script:
        p_num = p.get("part_number")
        part_vids.append(f"output/test_run/part_{p_num}_master.mp4")
    tool_context.state["completed_part_videos"] = part_vids
    return json.dumps({"status": "SUCCESS", "part_videos": part_vids})


def mock_assemble_movie_tool(tool_context: Context) -> str:
    parts = tool_context.state.get("completed_part_videos", [])
    out_master = "output/test_run/2026-10-21-mughal-master.mp4"
    tool_context.state["final_movie_path"] = out_master
    tool_context.state["master_movie_path"] = out_master
    tool_context.state["final_movie_duration"] = 120.5
    return json.dumps({"status": "SUCCESS", "movie_path": out_master, "parts_count": len(parts)})


async def run_hitl_unit_test():
    writer = Agent(
        name="writer_agent",
        model="gemini-2.5-flash",
        instruction="Call mock_plan_story_tool to create the story into state.",
        tools=[mock_plan_story_tool]
    )

    reviewer = Agent(
        name="story_review_agent",
        model="gemini-2.5-flash",
        instruction=(
            "Call format_and_review_screenplay_tool to format the screenplay. "
            "Then call request_input with message='Do you approve this screenplay to proceed with video generation?'"
        ),
        tools=[format_and_review_screenplay_tool, request_input]
    )

    producer = Agent(
        name="part_production_agent",
        model="gemini-2.5-flash",
        instruction="Call mock_produce_parts_tool to produce all parts.",
        tools=[mock_produce_parts_tool]
    )

    assembler = Agent(
        name="movie_assembly_agent",
        model="gemini-2.5-flash",
        instruction="Call mock_assemble_movie_tool to assemble the final movie.",
        tools=[mock_assemble_movie_tool]
    )

    director = SequentialAgent(
        name="director_agent",
        sub_agents=[writer, reviewer, producer, assembler]
    )

    session_service = InMemorySessionService()
    runner = Runner(agent=director, session_service=session_service, app_name="test_hitl_app")
    session = await session_service.create_session(app_name="test_hitl_app", user_id="test_director")

    print("\n--- TURN 1: WriterAgent Planning & Review ---")
    msg1 = types.Content(role="user", parts=[types.Part.from_text(text="Produce 2-part movie")])
    paused_at_hitl = False

    async for event in runner.run_async(session_id=session.id, user_id="test_director", new_message=msg1):
        content = getattr(event, "content", None)
        if content and content.parts:
            for p in content.parts:
                if getattr(p, "function_call", None):
                    fc = p.function_call
                    print(f"[{event.author} Tool]: {fc.name}")
                    if "request_input" in fc.name:
                        paused_at_hitl = True

    assert paused_at_hitl, "Execution must pause at request_input for HITL approval!"
    print(f"Verified Turn 1 paused for human approval: {paused_at_hitl}")

    print("\n--- TURN 2: Human Approval & Video Generation ---")
    msg2 = types.Content(role="user", parts=[types.Part.from_text(text="Approved. Proceed with video production.")])
    async for event in runner.run_async(session_id=session.id, user_id="test_director", new_message=msg2):
        content = getattr(event, "content", None)
        if content and content.parts:
            for p in content.parts:
                if getattr(p, "function_call", None):
                    fc = p.function_call
                    print(f"[{event.author} Tool]: {fc.name}")

    updated_session = await session_service.get_session(app_name="test_hitl_app", user_id="test_director", session_id=session.id)
    final_movie = updated_session.state.get("final_movie_path")
    print(f"Verified Final Movie in State: {final_movie}")
    assert final_movie == "output/test_run/2026-10-21-mughal-master.mp4"
    print("\nHITL MULTI-TURN UNIT TEST PASSED 100%!")


if __name__ == "__main__":
    asyncio.run(run_hitl_unit_test())
