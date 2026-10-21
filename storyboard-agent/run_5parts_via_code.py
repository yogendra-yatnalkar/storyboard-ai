import json
import time
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8080"
TOPIC = "Mughal empire for next 20 years after death of Emperor Aurangazeb. Make it as much detailed as possible within 5 parts please. Please perform web-search  for detailed information."

def post_json(endpoint: str, data: dict):
    req = urllib.request.Request(
        f"{BASE_URL}{endpoint}",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get_json(endpoint: str):
    with urllib.request.urlopen(f"{BASE_URL}{endpoint}") as resp:
        return json.loads(resp.read().decode("utf-8"))

def main():
    print("=" * 75)
    print("TRIGGERING 5-PART MUGHAL PRODUCTION RUN VIA CODE")
    print("=" * 75)
    print(f"Topic: {TOPIC}\n")

    # 1. Create Run
    init_res = post_json("/api/runs", {
        "topic": TOPIC,
        "verbatim_mode": False,
        "requested_parts": None  # Auto-inferred 5 parts from prompt
    })
    run_id = init_res["run_id"]
    folder = init_res["folder_name"]
    print(f"[RUN CREATED] Run ID: {run_id}")
    print(f"[RUN FOLDER]  output/{folder}\n")

    # 2. Wait for Turn 1 to reach HITL Gate
    print("Stage 1: Performing Google Search grounding and architecting 5-Part screenplay...")
    while True:
        status_data = get_json(f"/api/runs/{run_id}/status")
        status = status_data.get("status")
        if status == "AWAITING_APPROVAL":
            print("\n[SUCCESS] Reached Human-in-the-Loop review gate!")
            break
        elif status in ["FAILED", "CANCELLED"]:
            print(f"\n[ERROR] Run terminated with status: {status}")
            print(f"Details: {status_data.get('error')}")
            return
        time.sleep(2)

    # 3. Verify Screenplay Parts
    screenplay = status_data.get("screenplay", [])
    print("\n" + "=" * 75)
    print(f"SCREENPLAY VERIFICATION: Total Parts Planned = {len(screenplay)}")
    print("=" * 75)
    for p in screenplay:
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
            tag = "[Veo 8s]" if s.get("is_veo_candidate") else "[Whiteboard]"
            print(f"    - [{p_num}.{s.get('subscene_number')}] {tag} {s.get('narration')[:65]}...")

    if len(screenplay) == 5:
        print("\n" + "=" * 75)
        print("CONFIRMATION: Exact 5 parts verified. Approving via code...")
        print("=" * 75)
        approve_res = post_json(f"/api/runs/{run_id}/approve", {
            "approval_text": "Approved via code. 5-Part comprehensive epic verified. Proceed with production."
        })
        print(f"[APPROVED]: {approve_res.get('message')}")
        print("Turn 2 video production and episodic part assembly is now underway!")
    else:
        print(f"\n[WARNING]: Expected 5 parts but found {len(screenplay)}. Not auto-approving.")

if __name__ == "__main__":
    main()
