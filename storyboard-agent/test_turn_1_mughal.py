import asyncio
import os
import sys

AGENT_DIR = os.path.join(os.getcwd(), 'storyboard-agent')
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

from agents.director_agent import create_director_agent
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

async def run_turn1():
    prompt = "Mughal empire for next 20 years after death of Emperor Aurangazeb. Make it as much detailed as possible within 5 parts please. Please perform web-search  for detailed information."
    session_service = InMemorySessionService()
    director = create_director_agent()
    runner = Runner(agent=director, session_service=session_service, app_name="storyboard_agent")
    
    init_state = {
        "story_topic": prompt,
        "requested_parts": None,
        "verbatim_mode": False,
        "output_dir": os.path.join(os.getcwd(), "output")
    }
    session = await session_service.create_session(app_name="storyboard_agent", user_id="user1", state=init_state)
    
    user_msg = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    print("=" * 75)
    print("STARTING TURN 1: Real-Time Grounding & 5-Part Screenplay Architecture...")
    print("=" * 75)
    
    paused_for_hitl = False
    async for event in runner.run_async(session_id=session.id, user_id="user1", new_message=user_msg):
        content = getattr(event, "content", None)
        if content and content.parts:
            for p in content.parts:
                if getattr(p, "function_call", None):
                    fc = p.function_call
                    if "request_input" in fc.name:
                        paused_for_hitl = True
                        print("\n[ADK HITL GATE REACHED]:", fc.args.get("message") if hasattr(fc, "args") else "")
                    
    sess = await session_service.get_session(app_name="storyboard_agent", user_id="user1", session_id=session.id)
    script = sess.state.get("storyboard_script", [])
    
    print("\n" + "=" * 75)
    print(f"SCREENPLAY VERIFICATION: Total Parts Planned = {len(script)}")
    print("=" * 75)
    for p in script:
        p_num = p.get("part_number")
        title = p.get("title")
        synopsis = p.get("synopsis")
        subs = p.get("subscenes", [])
        veo_cnt = sum(1 for s in subs if s.get("is_veo_candidate"))
        print(f"\n[PART {p_num}]: {title}")
        print(f"  Synopsis: {synopsis}")
        print(f"  Portraits: {p.get('search_queries')}")
        print(f"  Subscenes: {len(subs)} (Veo clips: {veo_cnt})")
        for s in subs:
            v_tag = "[Veo 8s]" if s.get("is_veo_candidate") else "[Whiteboard]"
            print(f"    - [{p_num}.{s.get('subscene_number')}] {v_tag} {s.get('narration')[:65]}...")
            
    return script, paused_for_hitl, session_service, session, runner

if __name__ == "__main__":
    asyncio.run(run_turn1())
