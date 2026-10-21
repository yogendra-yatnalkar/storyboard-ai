"""
Storyboard AI - Web Runner Service.
Orchestrates Google ADK Director Workflow execution, manages multi-turn HITL gates,
monitors dated run folders for intermediate outputs (images, videos, audio),
verifies state and folder structure after every part completes,
and streams live updates via Server-Sent Events (SSE).
"""
import asyncio
import datetime
import json
import os
import re
import sys
import time
import traceback
from typing import Dict, Any, List, Optional, Set

# Ensure storyboard-agent root is on path
AGENT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "storyboard-agent"))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "output")

if AGENT_ROOT not in sys.path:
    sys.path.insert(0, AGENT_ROOT)

from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import config
from cost_tracker import tracker
from tools import utils
from agents.director_agent import create_director_agent


class OutputFolderWatcher:
    """Monitors a run's output directory and categorizes created/updated assets."""

    def __init__(self, run_id: str, folder_path: str, on_new_asset_cb, run_ref=None):
        self.run_id = run_id
        self.folder_path = folder_path
        self.on_new_asset_cb = on_new_asset_cb
        self.run_ref = run_ref
        self.seen_files: Dict[str, int] = {}  # fname -> size
        self.is_running = False
        self._task = None

    async def start(self):
        self.is_running = True
        self._task = asyncio.create_task(self._watch_loop())

    async def stop(self):
        self.is_running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _watch_loop(self):
        while self.is_running:
            try:
                self.scan_folder()
            except Exception as e:
                print(f"[Watcher Error {self.run_id}]: {e}")
            await asyncio.sleep(0.5)

    def scan_folder(self):
        if not os.path.exists(self.folder_path):
            return

        current_files = os.listdir(self.folder_path)
        for fname in sorted(current_files):
            fpath = os.path.join(self.folder_path, fname)
            if not os.path.isfile(fpath):
                continue

            try:
                size = os.path.getsize(fpath)
            except OSError:
                continue

            if size == 0:
                continue  # File still being opened/created

            last_size = self.seen_files.get(fname)
            # Notify if new file, or if file significantly grew (e.g. video rendering finished)
            if last_size is None or (last_size != size and (size - last_size) > 2048):
                self.seen_files[fname] = size
                asset_info = self._categorize_file(fname, fpath, size)
                if asset_info and self.on_new_asset_cb:
                    self.on_new_asset_cb(asset_info)

    def _categorize_file(self, fname: str, fpath: str, size: int) -> dict:
        ext = os.path.splitext(fname)[1].lower()
        folder_slug = os.path.basename(self.folder_path)
        url = f"/output/{folder_slug}/{fname}"

        media_type = "other"
        if ext in [".png", ".jpg", ".jpeg", ".webp"]:
            media_type = "image"
        elif ext in [".mp4", ".mov", ".webm"]:
            media_type = "video"
        elif ext in [".wav", ".mp3"]:
            media_type = "audio"
        elif ext in [".srt", ".txt", ".json"]:
            media_type = "text"

        info = {
            "run_id": self.run_id,
            "filename": fname,
            "url": url,
            "path": fpath,
            "size_bytes": size,
            "timestamp": time.time(),
            "category": "other",
            "media_type": media_type,
            "label": fname
        }

        # 1. Historical Reference Portrait
        if fname.startswith("ref_") and media_type == "image":
            info["category"] = "reference"
            m = re.search(r"ref_\d+_([a-zA-Z0-9_\-]+)_ddgs", fname)
            label = m.group(1).replace("_", " ") if m else "Historical Reference"
            info["label"] = f"Reference: {label}"

        # 2. 3x2 Comic Grid Image
        elif (fname.startswith("gen_image_") or fname.startswith("grid_")) and media_type == "image":
            info["category"] = "grid"
            info["label"] = "3x2 Comic Grid"

        # 3. Sliced 720p Panel
        elif fname.startswith("panel_") and media_type == "image":
            info["category"] = "panel"
            m = re.search(r"panel_.*?_(\d+)\.png", fname)
            panel_idx = int(m.group(1)) if m else None
            info["panel_index"] = panel_idx
            info["label"] = f"Panel {panel_idx}" if panel_idx else "Sliced Panel"

        # 4. Whiteboard Animation Video
        elif (fname.startswith("whiteboard_animation_") or fname.startswith("draw_")) and ext == ".mp4":
            info["category"] = "drawing"
            info["label"] = "Whiteboard Animation"

        # 5. Veo Motion Clip
        elif fname.startswith("veo_") and ext == ".mp4":
            info["category"] = "veo"
            info["label"] = "Veo 8s Motion Clip"

        # 6. Part Master Video (e.g. part_1_master.mp4)
        elif fname.startswith("part_") and "master" in fname and ext == ".mp4":
            info["category"] = "part_master"
            m = re.search(r"part_(\d+)_master", fname)
            part_num = int(m.group(1)) if m else None
            info["part_number"] = part_num
            info["label"] = f"Part {part_num} Master Video" if part_num else "Part Master Video"

        # 7. Sub-Scene Final Video (sub_X_Y, subtitled_X, or merged_X)
        elif (fname.startswith("sub_") or fname.startswith("subtitled_") or fname.startswith("merged_")) and ext == ".mp4":
            info["category"] = "subscene"
            m = re.search(r"sub_(\d+)_(\d+)", fname)
            if m:
                p_num = int(m.group(1))
                s_num = int(m.group(2))
                info["part_number"] = p_num
                info["subscene_number"] = s_num
                info["label"] = f"Scene {p_num}.{s_num}"
            elif fname.startswith("subtitled_"):
                info["label"] = "Subtitled Working Scene"
            elif fname.startswith("merged_"):
                info["label"] = "Merged AV Scene"
            else:
                info["label"] = "Working Scene Video"

        # 8. Master Movie (e.g. 2026-10-21-title-master.mp4 or movie_master.mp4)
        elif "master.mp4" in fname and not fname.startswith("part_"):
            info["category"] = "master_movie"
            info["label"] = "Final Master Movie"

        # 9. Voiceover Audio
        elif (fname.startswith("tts_") or ext == ".wav"):
            info["category"] = "audio"
            info["label"] = "Narration Voiceover"

        # 10. Subtitles
        elif ext == ".srt":
            info["category"] = "subtitles"
            info["label"] = "Subtitles (.srt)"

        # Cross-reference with ADK session state if available to refine part/scene indexing
        if self.run_ref and getattr(self.run_ref, "session", None):
            try:
                state = self.run_ref.session.state
                norm_fpath = os.path.normpath(fpath)
                for p in range(1, 11):
                    # Check grid
                    grid_p = state.get(f"part_{p}_grid_image")
                    if grid_p and os.path.normpath(grid_p) == norm_fpath:
                        info["category"] = "grid"
                        info["part_number"] = p
                        info["label"] = f"Part {p} 3x2 Comic Grid"

                    # Check panels
                    panels_list = state.get(f"part_{p}_panel_images", [])
                    for idx, pan_p in enumerate(panels_list):
                        if pan_p and os.path.normpath(pan_p) == norm_fpath:
                            info["category"] = "panel"
                            info["part_number"] = p
                            info["panel_index"] = idx + 1
                            info["label"] = f"Part {p} Panel {idx + 1}"

                    # Check subscenes
                    for s in range(1, 7):
                        sub_p = state.get(f"part_{p}_sub_{s}")
                        if sub_p and os.path.normpath(sub_p) == norm_fpath:
                            info["category"] = "subscene"
                            info["part_number"] = p
                            info["subscene_number"] = s
                            info["label"] = f"Scene {p}.{s}"

                    # Check part master
                    pm_p = state.get(f"part_{p}_master_video")
                    if pm_p and os.path.normpath(pm_p) == norm_fpath:
                        info["category"] = "part_master"
                        info["part_number"] = p
                        info["label"] = f"Part {p} Master Video"

                final_p = state.get("final_movie_path") or state.get("master_movie_path")
                if final_p and os.path.normpath(final_p) == norm_fpath:
                    info["category"] = "master_movie"
                    info["label"] = "Final Master Movie"
            except Exception:
                pass

        return info


class StoryboardRun:
    """Manages state, ADK workflow, and event dispatch for a single storyboard generation run."""

    def __init__(self, run_id: str, topic: str, verbatim_mode: bool = False, requested_parts: Optional[int] = None):
        self.run_id = run_id
        self.topic = topic
        self.verbatim_mode = verbatim_mode
        self.requested_parts = requested_parts

        # Dedicated unique output directory for EVERY run
        self.output_dir = utils.init_dated_output_dir(title=topic, base_dir=OUTPUT_BASE_DIR)
        self.folder_name = os.path.basename(self.output_dir)

        self.created_at = datetime.datetime.now().isoformat()
        self.status = "INITIALIZING"  # PLANNING, AWAITING_APPROVAL, PRODUCING_VIDEO, COMPLETED, FAILED, CANCELLED
        self.logs: List[dict] = []
        self.subscribers: List[asyncio.Queue] = []

        self.screenplay: List[dict] = []
        self.hitl_message: str = ""
        self.final_movie_url: Optional[str] = None
        self.final_movie_duration: float = 0.0
        self.error: Optional[str] = None
        self.cost_summary: str = ""

        # Status of each episodic part and its intermediate outputs
        self.parts_status: Dict[int, dict] = {}

        # Assets indexed by category
        self.assets: Dict[str, List[dict]] = {
            "reference": [],
            "grid": [],
            "panel": [],
            "drawing": [],
            "veo": [],
            "subscene": [],
            "part_master": [],
            "master_movie": [],
            "audio": [],
            "other": []
        }

        self.session_service: Optional[InMemorySessionService] = None
        self.runner: Optional[Runner] = None
        self.session = None

        # Watcher for live output folder
        self.watcher = OutputFolderWatcher(
            run_id=self.run_id,
            folder_path=self.output_dir,
            on_new_asset_cb=self._handle_new_asset,
            run_ref=self
        )

    def log(self, text: str, author: str = "Director", level: str = "info"):
        entry = {
            "timestamp": time.time(),
            "time_str": datetime.datetime.now().strftime("%H:%M:%S"),
            "author": author,
            "text": text,
            "level": level
        }
        self.logs.append(entry)
        self.broadcast("log", entry)
        try:
            print(f"[{entry['time_str']}][{author}] {text}")
        except UnicodeEncodeError:
            safe_text = text.encode("ascii", errors="replace").decode("ascii")
            print(f"[{entry['time_str']}][{author}] {safe_text}")

    def broadcast(self, event_type: str, data: Any):
        dead_queues = []
        if isinstance(data, dict) and "cost_metrics" not in data and hasattr(tracker, "get_metrics_dict"):
            data["cost_metrics"] = tracker.get_metrics_dict()
        msg = {"type": event_type, "data": data}
        for q in self.subscribers:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                dead_queues.append(q)
            except Exception:
                dead_queues.append(q)
        for dq in dead_queues:
            if dq in self.subscribers:
                self.subscribers.remove(dq)

    def _handle_new_asset(self, asset: dict):
        cat = asset.get("category", "other")
        media_type = asset.get("media_type", "other")
        if cat not in self.assets:
            self.assets[cat] = []

        # Avoid exact duplicate entries or update size
        existing = next((a for a in self.assets[cat] if a["filename"] == asset["filename"]), None)
        if existing:
            existing["size_bytes"] = asset["size_bytes"]
        else:
            self.assets[cat].append(asset)
            self.log(f"Asset Created: {asset.get('label')} ({asset['filename']})", author="FileSystem", level="success")
            self.broadcast("new_asset", asset)

        # Requirement: Show working scene live output as well once image is created, video is created
        if media_type in ["image", "video"] and cat in ["grid", "panel", "reference", "drawing", "veo", "subscene", "part_master", "master_movie"]:
            self.broadcast("live_scene_update", {
                "media_type": media_type,
                "category": cat,
                "asset": asset,
                "active_part": asset.get("part_number"),
                "active_subscene": asset.get("subscene_number"),
                "label": asset.get("label", "Live Working Output")
            })

        if cat == "master_movie":
            self.final_movie_url = asset["url"]
            self.broadcast("master_movie_ready", asset)

    def check_part_intermediate_outputs(self, part_num: int, state: dict) -> dict:
        """
        Inspects session state and output directory for intermediate outputs of a completed Part:
        reference portraits, 3x2 comic grid, sliced panels, subscene videos, part master video.
        """
        folder_slug = os.path.basename(self.output_dir)

        # 1. Comic Grid Image
        grid_path = state.get(f"part_{part_num}_grid_image")
        grid_info = None
        if grid_path and os.path.exists(grid_path):
            grid_info = {
                "filename": os.path.basename(grid_path),
                "url": f"/output/{folder_slug}/{os.path.basename(grid_path)}",
                "size_bytes": os.path.getsize(grid_path)
            }

        # 2. Sliced Panels
        panel_paths = state.get(f"part_{part_num}_panel_images", [])
        panels_info = []
        for p in panel_paths:
            if p and os.path.exists(p):
                panels_info.append({
                    "filename": os.path.basename(p),
                    "url": f"/output/{folder_slug}/{os.path.basename(p)}"
                })

        # 3. Sub-scene Videos
        subscenes_info = []
        for s_idx in range(1, 7):
            s_path = state.get(f"part_{part_num}_sub_{s_idx}")
            if s_path and os.path.exists(s_path):
                subscenes_info.append({
                    "subscene_number": s_idx,
                    "filename": os.path.basename(s_path),
                    "url": f"/output/{folder_slug}/{os.path.basename(s_path)}",
                    "duration": utils.get_video_duration(s_path)
                })

        # 4. Part Master Video
        part_master = state.get(f"part_{part_num}_master_video")
        if not part_master or not os.path.exists(part_master):
            candidate = os.path.join(self.output_dir, f"part_{part_num}_master.mp4")
            if os.path.exists(candidate):
                part_master = candidate

        master_info = None
        dur = 0.0
        if part_master and os.path.exists(part_master):
            dur = utils.get_video_duration(part_master)
            master_info = {
                "filename": os.path.basename(part_master),
                "url": f"/output/{folder_slug}/{os.path.basename(part_master)}",
                "duration": dur,
                "size_bytes": os.path.getsize(part_master)
            }

        # Screenplay part metadata
        script = state.get("storyboard_script", self.screenplay)
        part_meta = next((p for p in script if p.get("part_number") == part_num), {})

        summary_text = (
            f"Part {part_num} Master assembled ({dur:.1f}s) with "
            f"{len(panels_info)} sliced panels and {len(subscenes_info)} subscenes."
        )

        part_data = {
            "part_number": part_num,
            "title": part_meta.get("title", f"Part {part_num}"),
            "synopsis": part_meta.get("synopsis", ""),
            "status": "COMPLETED",
            "grid_image": grid_info,
            "panels_count": len(panels_info),
            "panels": panels_info,
            "subscenes_count": len(subscenes_info),
            "subscenes": subscenes_info,
            "part_master": master_info,
            "summary": summary_text
        }

        self.parts_status[part_num] = part_data
        return part_data

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "topic": self.topic,
            "verbatim_mode": self.verbatim_mode,
            "requested_parts": self.requested_parts,
            "output_dir": self.output_dir,
            "folder_name": self.folder_name,
            "created_at": self.created_at,
            "status": self.status,
            "hitl_message": self.hitl_message,
            "screenplay": self.screenplay,
            "parts_status": self.parts_status,
            "final_movie_url": self.final_movie_url,
            "final_movie_duration": self.final_movie_duration,
            "error": self.error,
            "cost_summary": self.cost_summary,
            "cost_metrics": tracker.get_metrics_dict() if hasattr(tracker, "get_metrics_dict") else {},
            "asset_counts": {k: len(v) for k, v in self.assets.items()},
            "assets": self.assets
        }


class RunnerService:
    """Singleton service to manage runs across UI sessions."""

    def __init__(self):
        self.runs: Dict[str, StoryboardRun] = {}
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    def create_run(self, topic: str, verbatim_mode: bool = False, requested_parts: Optional[int] = None) -> StoryboardRun:
        run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}"
        run = StoryboardRun(
            run_id=run_id,
            topic=topic,
            verbatim_mode=verbatim_mode,
            requested_parts=requested_parts
        )
        self.runs[run_id] = run
        return run

    def get_run(self, run_id: str) -> Optional[StoryboardRun]:
        return self.runs.get(run_id)

    def list_runs(self) -> List[dict]:
        return [r.to_dict() for r in sorted(self.runs.values(), key=lambda x: x.created_at, reverse=True)]

    def list_history_folders(self) -> List[dict]:
        """Scans output directory to allow browsing previous movie generations."""
        if not os.path.exists(OUTPUT_BASE_DIR):
            return []

        history = []
        for d in sorted(os.listdir(OUTPUT_BASE_DIR), reverse=True):
            full_p = os.path.join(OUTPUT_BASE_DIR, d)
            if not os.path.isdir(full_p):
                continue

            files = os.listdir(full_p)
            master_files = [f for f in files if "master" in f and f.endswith(".mp4") and not f.startswith("part_")]
            part_files = [f for f in files if f.startswith("part_") and f.endswith(".mp4")]
            grid_files = [f for f in files if (f.startswith("gen_image_") or f.startswith("grid_")) and f.endswith((".png", ".jpg"))]

            history.append({
                "folder_name": d,
                "created_time": os.path.getctime(full_p),
                "total_files": len(files),
                "has_final_movie": len(master_files) > 0,
                "final_movie_url": f"/output/{d}/{master_files[0]}" if master_files else None,
                "parts_count": len(part_files),
                "grid_count": len(grid_files)
            })
        return history

    async def start_turn_1(self, run: StoryboardRun):
        """Executes Turn 1: Research, Story Architecture, and pauses at HITL Gate."""
        run.status = "PLANNING"
        run.broadcast("status_change", {"status": "PLANNING"})
        run.log(f"Starting Storyboard Run for: '{run.topic}' (Parts: {run.requested_parts}, Mode: {'Verbatim' if run.verbatim_mode else 'Creative'})")
        run.log(f"Output folder created: {run.output_dir}")

        # Ensure utils output dir is pointed to this run's folder
        utils.set_output_dir(run.output_dir)

        # Start live folder watcher
        await run.watcher.start()

        try:
            director = create_director_agent()
            run.session_service = InMemorySessionService()
            run.runner = Runner(
                agent=director,
                session_service=run.session_service,
                app_name="storyboard_agent"
            )

            init_state = {
                "verbatim_mode": run.verbatim_mode,
                "requested_parts": run.requested_parts,
                "story_topic": run.topic,
                "output_dir": run.output_dir,
                "folder_name": run.folder_name,
                "run_id": run.run_id,
            }

            run.session = await run.session_service.create_session(
                app_name="storyboard_agent",
                user_id=f"web_user_{run.run_id}",
                state=init_state
            )

            p_desc = f"{run.requested_parts}-Part " if run.requested_parts else ""
            agent_prompt = (
                f"Please produce a {p_desc}storyboard movie on: '{run.topic}'. "
                f"Mode: {'verbatim' if run.verbatim_mode else 'creative'}."
            )

            user_content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=agent_prompt)]
            )

            run.log("Stage 1: Architecting screenplay and performing grounding search...", author="Director")

            paused_for_hitl = False
            hitl_msg = "Do you approve this screenplay to proceed with video generation?"

            async for event in run.runner.run_async(
                session_id=run.session.id,
                user_id=f"web_user_{run.run_id}",
                new_message=user_content
            ):
                content = getattr(event, "content", None)
                if content and content.parts:
                    for p in content.parts:
                        if getattr(p, "text", None):
                            run.log(p.text, author=event.author or "Agent")
                        if getattr(p, "function_call", None):
                            fc = p.function_call
                            run.log(f"Executing tool: {fc.name}", author=event.author or "AgentTool")
                            if "request_input" in fc.name:
                                paused_for_hitl = True
                                if hasattr(fc, "args") and isinstance(fc.args, dict) and "message" in fc.args:
                                    hitl_msg = fc.args["message"]

            # Inspect state
            updated_session = await run.session_service.get_session(
                app_name="storyboard_agent",
                user_id=f"web_user_{run.run_id}",
                session_id=run.session.id
            )
            run.screenplay = updated_session.state.get("storyboard_script", [])
            run.hitl_message = hitl_msg

            if paused_for_hitl:
                run.status = "AWAITING_APPROVAL"
                run.log("Turn 1 Complete: Screenplay architected. Paused at Human-in-the-Loop review gate.", author="Director", level="warning")
                run.broadcast("hitl_gate", {
                    "prompt_message": run.hitl_message,
                    "screenplay": run.screenplay,
                    "topic": run.topic,
                    "parts_count": len(run.screenplay)
                })
                run.broadcast("status_change", {"status": "AWAITING_APPROVAL"})
            else:
                run.status = "COMPLETED"
                run.broadcast("status_change", {"status": "COMPLETED"})

        except Exception as e:
            err_str = f"Error during Turn 1 execution: {traceback.format_exc()}"
            run.error = err_str
            run.status = "FAILED"
            run.log(err_str, author="System", level="error")
            run.broadcast("status_change", {"status": "FAILED", "error": str(e)})

    async def approve_and_start_turn_2(self, run: StoryboardRun, user_approval_text: str = "", edited_screenplay: Optional[List[dict]] = None):
        """Executes Turn 2: User approval received -> Part production loop & Master movie assembly."""
        if run.status != "AWAITING_APPROVAL":
            raise ValueError(f"Run {run.run_id} is not waiting for approval (Current status: {run.status})")

        # If user submitted edited screenplay during HITL gate, update session state
        if edited_screenplay and isinstance(edited_screenplay, list) and len(edited_screenplay) > 0:
            run.screenplay = edited_screenplay
            if run.session:
                run.session.state["storyboard_script"] = edited_screenplay
                run.log(f"Screenplay updated with {len(edited_screenplay)} director-revised parts.", author="User", level="info")

        run.status = "PRODUCING_VIDEO"
        run.broadcast("status_change", {"status": "PRODUCING_VIDEO"})

        approval_message = user_approval_text.strip() or "Approved. Proceed with video production and assembly."
        run.log(f"HITL Approval Confirmed: '{approval_message}'", author="User", level="success")
        run.log("Stage 2: Beginning multi-part video production & master assembly...", author="Director")

        # Make sure output dir is still set
        utils.set_output_dir(run.output_dir)

        completed_parts_seen = set()

        try:
            turn2_content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=approval_message)]
            )

            async for event in run.runner.run_async(
                session_id=run.session.id,
                user_id=f"web_user_{run.run_id}",
                new_message=turn2_content
            ):
                content = getattr(event, "content", None)
                if content and content.parts:
                    for p in content.parts:
                        if getattr(p, "text", None):
                            run.log(p.text, author=event.author or "Agent")
                        if getattr(p, "function_call", None):
                            fc = p.function_call
                            run.log(f"Executing tool: {fc.name}", author=event.author or "AgentTool")

                # Requirement: After every part is complete, check folder structure and state and show that on UI
                if run.session_service and run.session:
                    updated_session = await run.session_service.get_session(
                        app_name="storyboard_agent",
                        user_id=f"web_user_{run.run_id}",
                        session_id=run.session.id
                    )
                    state = updated_session.state
                    completed_list = state.get("completed_part_videos", [])

                    total_parts = len(run.screenplay) or len(state.get("storyboard_script", [])) or (run.requested_parts or 3)
                    for p_num in range(1, total_parts + 1):
                        p_master = state.get(f"part_{p_num}_master_video")
                        is_part_done = (p_num <= len(completed_list)) or (p_master and os.path.exists(p_master))
                        if is_part_done and p_num not in completed_parts_seen:
                            completed_parts_seen.add(p_num)
                            # Inspect folder structure & state for intermediate outputs
                            part_summary = run.check_part_intermediate_outputs(p_num, state)
                            run.log(f"Part #{p_num} Complete! Checked state & folder: {part_summary['summary']}", author="Director", level="success")
                            run.broadcast("part_completed", {
                                "part_number": p_num,
                                "total_parts": total_parts,
                                "part_summary": part_summary
                            })

            # Final check of all parts in state & folder
            updated_session = await run.session_service.get_session(
                app_name="storyboard_agent",
                user_id=f"web_user_{run.run_id}",
                session_id=run.session.id
            )
            final_total_parts = len(run.screenplay) or len(updated_session.state.get("storyboard_script", [])) or (run.requested_parts or 3)
            for p_num in range(1, final_total_parts + 1):
                if p_num not in completed_parts_seen:
                    part_summary = run.check_part_intermediate_outputs(p_num, updated_session.state)
                    completed_parts_seen.add(p_num)
                    run.broadcast("part_completed", {
                        "part_number": p_num,
                        "total_parts": run.requested_parts,
                        "part_summary": part_summary
                    })

            master_path = updated_session.state.get("final_movie_path") or updated_session.state.get("master_movie_path")

            # If not in state, look in run's folder for master movie
            if not master_path or not os.path.exists(master_path):
                candidates = [f for f in os.listdir(run.output_dir) if "master" in f and f.endswith(".mp4") and not f.startswith("part_")]
                if candidates:
                    master_path = os.path.join(run.output_dir, candidates[0])

            if master_path and os.path.exists(master_path):
                dur = utils.get_video_duration(master_path)
                run.final_movie_duration = dur
                run.final_movie_url = f"/output/{run.folder_name}/{os.path.basename(master_path)}"
                run.log(f"Master Movie Generated: {os.path.basename(master_path)} ({dur:.1f}s)", author="Director", level="success")

            run.cost_summary = tracker.get_summary_report()
            run.status = "COMPLETED"
            run.log("Storyboard Movie Production COMPLETE 100%!", author="Director", level="success")
            run.log(f"\n{run.cost_summary}", author="BudgetReport")

            run.broadcast("status_change", {
                "status": "COMPLETED",
                "final_movie_url": run.final_movie_url,
                "duration": run.final_movie_duration,
                "cost_summary": run.cost_summary
            })

        except Exception as e:
            err_str = f"Error during Turn 2 production: {traceback.format_exc()}"
            run.error = err_str
            run.status = "FAILED"
            run.log(err_str, author="System", level="error")
            run.broadcast("status_change", {"status": "FAILED", "error": str(e)})

        finally:
            # Final scan
            run.watcher.scan_folder()

    async def cancel_run(self, run: StoryboardRun):
        """Cancels an active run."""
        run.status = "CANCELLED"
        run.log("Production halted by user request.", author="Director", level="warning")
        await run.watcher.stop()
        run.broadcast("status_change", {"status": "CANCELLED"})


# Global service instance
service = RunnerService()
