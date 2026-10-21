"""
Storyboard AI - FastAPI Web Server.
Provides REST and SSE endpoints for the Storyboard Studio web interface,
manages agent execution lifecycle, and streams video/image assets with HTTP 206 range support.
"""
import asyncio
import json
import os
import sys
from typing import Optional
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure storyboard-agent and web package roots are importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
AGENT_ROOT = os.path.join(PROJECT_ROOT, "storyboard-agent")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
STATIC_DIR = os.path.join(CURRENT_DIR, "static")

if AGENT_ROOT not in sys.path:
    sys.path.insert(0, AGENT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from web.runner_service import service, StoryboardRun

# Create FastAPI app
app = FastAPI(
    title="Storyboard AI - Studio",
    description="Interactive Multi-Agent Storyboard Production Studio powered by Google ADK & Gemini",
    version="2.0.0"
)

# Enable CORS for flexible local or Cloud Run origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs directory so generated images, audio, and videos can be streamed directly
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


# Request & Response Models
class CreateRunRequest(BaseModel):
    topic: str = Field(..., description="Topic prompt or verbatim text")
    verbatim_mode: bool = Field(False, description="Strict verbatim narration mode")
    requested_parts: Optional[int] = Field(None, description="Number of sequential Parts (optional, auto-decided if None)")


class ApproveRunRequest(BaseModel):
    approval_text: Optional[str] = Field("Approved. Proceed with video production and assembly.", description="Human approval message or feedback")
    screenplay: Optional[list] = Field(None, description="Optional director-revised screenplay")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    """Health check endpoint for Cloud Run and monitoring."""
    return {
        "status": "healthy",
        "service": "storyboard-ai-studio",
        "output_dir": OUTPUT_DIR
    }


@app.get("/api/config")
async def get_config():
    """Returns active pipeline configurations."""
    return {
        "model_name": config.MODEL_NAME,
        "image_gen_model": config.IMAGE_GEN_MODEL,
        "tts_model": config.TTS_MODEL,
        "veo_model": config.VEO_MODEL,
        "output_width": config.OUTPUT_WIDTH,
        "output_height": config.OUTPUT_HEIGHT,
        "enable_veo": config.ENABLE_VEO,
        "veo_scope": config.VEO_SCOPE,
        "enable_sam": config.ENABLE_SAM,
        "max_parts": config.MAX_PARTS
    }


@app.post("/api/runs")
async def create_and_start_run(req: CreateRunRequest, background_tasks: BackgroundTasks):
    """
    Creates a new storyboard run with a fresh timestamped directory in output/
    and starts Turn 1 (Planning & Grounding -> HITL Review Gate).
    """
    if not req.topic or not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")

    run = service.create_run(
        topic=req.topic.strip(),
        verbatim_mode=req.verbatim_mode,
        requested_parts=req.requested_parts
    )

    # Launch Turn 1 in background task
    background_tasks.add_task(service.start_turn_1, run)

    return {
        "status": "ACCEPTED",
        "run_id": run.run_id,
        "folder_name": run.folder_name,
        "output_dir": run.output_dir
    }


@app.get("/api/runs")
async def list_runs():
    """Lists all in-memory runs from current session."""
    return service.list_runs()


@app.get("/api/runs/{run_id}")
async def get_run_details(run_id: str):
    """Returns complete state, logs, screenplay, and generated assets for a run."""
    run = service.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return run.to_dict()


@app.post("/api/runs/{run_id}/approve")
async def approve_run(run_id: str, req: ApproveRunRequest, background_tasks: BackgroundTasks):
    """
    Human-in-the-Loop Approval Gate:
    Approves the architected screenplay and begins Turn 2 (Part production & master assembly).
    """
    run = service.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    if run.status != "AWAITING_APPROVAL":
        raise HTTPException(
            status_code=400,
            detail=f"Run '{run_id}' cannot be approved in state '{run.status}'. Must be in 'AWAITING_APPROVAL'."
        )

    # Launch Turn 2 in background task
    background_tasks.add_task(service.approve_and_start_turn_2, run, req.approval_text, req.screenplay)

    return {
        "status": "ACCEPTED",
        "run_id": run_id,
        "message": "Screenplay approved. Video production loop initiated."
    }


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str):
    """Halts execution of an active run."""
    run = service.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    await service.cancel_run(run)
    return {"status": "SUCCESS", "run_id": run_id, "message": "Run cancelled."}


@app.get("/api/runs/{run_id}/stream")
async def stream_run_events(run_id: str, request: Request):
    """
    Server-Sent Events (SSE) stream for real-time logs, new asset alerts,
    HITL approval gates, and video completion notifications.
    """
    run = service.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    queue: asyncio.Queue = asyncio.Queue()
    run.subscribers.append(queue)

    async def event_generator():
        # First send initial catchup state
        initial_msg = {
            "type": "init",
            "data": run.to_dict()
        }
        yield f"event: {initial_msg['type']}\ndata: {json.dumps(initial_msg['data'])}\n\n"

        # Also send existing logs
        for log_entry in run.logs:
            yield f"event: log\ndata: {json.dumps(log_entry)}\n\n"

        # Catchup completed parts if any
        if run.parts_status:
            for p_num, p_data in run.parts_status.items():
                yield f"event: part_completed\ndata: {json.dumps({'part_number': p_num, 'total_parts': run.requested_parts, 'part_summary': p_data})}\n\n"

        # If already waiting at HITL gate, resend hitl_gate event
        if run.status == "AWAITING_APPROVAL":
            hitl_event = {
                "prompt_message": run.hitl_message,
                "screenplay": run.screenplay,
                "topic": run.topic,
                "parts_count": len(run.screenplay)
            }
            yield f"event: hitl_gate\ndata: {json.dumps(hitl_event)}\n\n"

        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    # Timeout to yield keepalive comment every 15s
                    msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"event: {msg['type']}\ndata: {json.dumps(msg['data'])}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            if queue in run.subscribers:
                run.subscribers.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/history")
async def list_history():
    """Lists historical output folders in output/ directory."""
    return service.list_history_folders()


@app.get("/api/history/{folder_name}")
async def get_history_folder_details(folder_name: str):
    """Returns all assets inside a historical folder."""
    target_dir = os.path.join(OUTPUT_DIR, folder_name)
    if not os.path.exists(target_dir) or not os.path.isdir(target_dir):
        raise HTTPException(status_code=404, detail=f"Folder '{folder_name}' not found.")

    files = os.listdir(target_dir)
    assets = []
    watcher = service.create_run("temp", False, 1).watcher
    watcher.folder_path = target_dir

    for fname in sorted(files):
        fpath = os.path.join(target_dir, fname)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            cat_info = watcher._categorize_file(fname, fpath, size)
            assets.append(cat_info)

    return {
        "folder_name": folder_name,
        "total_files": len(files),
        "assets": assets
    }


# Mount Static Frontend
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def main():
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Starting Storyboard AI Studio on http://localhost:{port}")
    uvicorn.run("web.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
