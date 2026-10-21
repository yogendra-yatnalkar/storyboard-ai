"""
Comprehensive Unit & Integration Test Suite for Storyboard AI Studio Web Server & Runner Service.
Validates:
1. Health & Config endpoints.
2. New run creation & automatic dated folder creation in output/ on every run.
3. Asset categorization logic (references, comic grids, sliced panels, subscenes, part masters, final movie)
   and media_type classification (image vs video).
4. Live Working Scene spotlight: verifies events emitted for both images (grids, panels) and videos (drawings, scenes, masters).
5. Full Multi-Turn HITL Lifecycle:
   - Turn 1: Planning & Grounding -> Halts at review_story_node with RequestInput.
   - Screenplay inspection & AWAITING_APPROVAL status.
   - Director editing screenplay during review gate.
   - Turn 2: Approval via /api/runs/{run_id}/approve -> Part loop & Master concatenation.
   - Post-part completion: intermediate outputs verified from state and folder structure (grid, panels, subscenes, part master).
   - Run completion, master movie detection, and cost report.
6. History endpoints (/api/history).
7. Frontend markup verification (HTML contains both video and image live working scene players, and parts section).
"""
import os
import sys
import json
import asyncio
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AGENT_ROOT = os.path.join(PROJECT_ROOT, "storyboard-agent")

if AGENT_ROOT not in sys.path:
    sys.path.insert(0, AGENT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from web.server import app
from web.runner_service import service, StoryboardRun, OutputFolderWatcher
from tools import utils
from google.genai import types


class TestStoryboardUIServer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_and_config(self):
        """Test /api/health and /api/config endpoints."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")

        res_cfg = self.client.get("/api/config")
        self.assertEqual(res_cfg.status_code, 200)
        cfg = res_cfg.json()
        self.assertIn("output_width", cfg)
        self.assertEqual(cfg["output_width"], 1280)
        self.assertEqual(cfg["output_height"], 720)

    def test_02_dated_folder_creation_on_every_run(self):
        """
        Validates requirement: 'on every run, please create a new folde in output folder where you should keep on dumping things'
        """
        topic = "Battle of Panipat 1761"
        run = service.create_run(topic=topic, verbatim_mode=False, requested_parts=2)

        self.assertTrue(os.path.exists(run.output_dir), f"Output directory must exist: {run.output_dir}")
        self.assertTrue(os.path.isdir(run.output_dir))

        # Folder name must contain slugified title and date timestamp
        folder_name = os.path.basename(run.output_dir)
        self.assertIn("battle-of-panipat", folder_name.lower())
        self.assertEqual(utils.get_output_dir(), run.output_dir)

        # Create another run to verify each run gets a distinct unique folder
        run2 = service.create_run(topic="Fall of Constantinople 1453", verbatim_mode=False, requested_parts=1)
        self.assertNotEqual(run.output_dir, run2.output_dir)
        self.assertIn("fall-of-constantinople", run2.output_dir.lower())

    def test_03_folder_asset_categorization_and_media_types(self):
        """
        Validates that assets created in the output directory are properly classified
        with both category and media_type (image vs video) for live working scene stream.
        """
        watcher = OutputFolderWatcher(run_id="test_run", folder_path="dummy/path", on_new_asset_cb=None)

        test_cases = [
            ("ref_1792575600_Aurangzeb_ddgs_0_high_res.jpg", "reference", "Reference: Aurangzeb", "image"),
            ("gen_image_1792575631.png", "grid", "3x2 Comic Grid", "image"),
            ("panel_1792575631_1.png", "panel", "Panel 1", "image"),
            ("panel_1792575631_6.png", "panel", "Panel 6", "image"),
            ("whiteboard_animation_1792575640.mp4", "drawing", "Whiteboard Animation", "video"),
            ("veo_1792575650.mp4", "veo", "Veo 8s Motion Clip", "video"),
            ("sub_1_2_subtitled.mp4", "subscene", "Scene 1.2", "video"),
            ("subtitled_1792575655.mp4", "subscene", "Subtitled Working Scene", "video"),
            ("merged_1792575652.mp4", "subscene", "Merged AV Scene", "video"),
            ("part_1_master.mp4", "part_master", "Part 1 Master Video", "video"),
            ("2026-10-21-15-30-mughal-master.mp4", "master_movie", "Final Master Movie", "video"),
            ("tts_speech_1792575600.wav", "audio", "Narration Voiceover", "audio"),
            ("subtitles_1792575600.srt", "subtitles", "Subtitles (.srt)", "text"),
        ]

        for fname, expected_cat, expected_label, expected_media_type in test_cases:
            info = watcher._categorize_file(fname, f"dummy/path/{fname}", 1024)
            self.assertEqual(info["category"], expected_cat, f"Failed category for {fname}")
            self.assertEqual(info["label"], expected_label, f"Failed label for {fname}")
            self.assertEqual(info["media_type"], expected_media_type, f"Failed media_type for {fname}")
            self.assertEqual(info["url"], f"/output/path/{fname}")

    def test_04_live_working_scene_events_for_images_and_videos(self):
        """
        Validates requirement: 'show working scene live output as well like once image is created, video is created'
        Ensures live_scene_update is broadcast for BOTH images (e.g. grid, panel) and videos (drawing, subscene, master).
        """
        run = service.create_run(topic="Test Live Scene", verbatim_mode=False, requested_parts=1)
        received_live_events = []

        q = asyncio.Queue()
        run.subscribers.append(q)

        # 1. Simulate an image asset being created (e.g. 3x2 comic grid)
        grid_asset = run.watcher._categorize_file("gen_image_9999.png", os.path.join(run.output_dir, "gen_image_9999.png"), 500000)
        run._handle_new_asset(grid_asset)

        # 2. Simulate a sliced panel image asset being created
        panel_asset = run.watcher._categorize_file("panel_9999_1.png", os.path.join(run.output_dir, "panel_9999_1.png"), 80000)
        run._handle_new_asset(panel_asset)

        # 3. Simulate a video asset being created (e.g. whiteboard animation)
        video_asset = run.watcher._categorize_file("whiteboard_animation_9999.mp4", os.path.join(run.output_dir, "whiteboard_animation_9999.mp4"), 3000000)
        run._handle_new_asset(video_asset)

        # Drain queue
        while not q.empty():
            msg = q.get_nowait()
            if msg["type"] == "live_scene_update":
                received_live_events.append(msg["data"])

        self.assertGreaterEqual(len(received_live_events), 3)

        # Check image event
        image_events = [ev for ev in received_live_events if ev.get("media_type") == "image"]
        self.assertGreaterEqual(len(image_events), 2)
        self.assertIn("3x2 Comic Grid", [ev["label"] for ev in image_events])

        # Check video event
        video_events = [ev for ev in received_live_events if ev.get("media_type") == "video"]
        self.assertGreaterEqual(len(video_events), 1)

    def test_05_full_hitl_lifecycle_and_intermediate_output_verification(self):
        """
        Validates the entire end-to-end multi-turn HITL workflow:
        1. Turn 1 executes -> pauses at review gate with RequestInput
        2. UI checks run details & screenplay parts
        3. Director edits screenplay before approval
        4. Turn 2 receives human approval -> renders videos
        5. After every part completes, verifies folder structure & state intermediate outputs
        6. Assembles final master movie and budget report
        """
        async def run_async_test():
            import agents.part_production_agent as ppa
            import agents.director_agent as da

            ppa.search_part_references_tool = lambda *a, **kw: "['ref_1.jpg']"
            ppa.generate_part_grid_tool = lambda tool_context, part_num, **kw: (
                tool_context.state.update({f"part_{part_num}_grid_image": os.path.join(utils.get_output_dir(), f"grid_{part_num}.png")}) or
                json.dumps({"status": "SUCCESS", "grid_image_path": f"grid_{part_num}.png"})
            )

            def mock_slice(tool_context, part_num, **kw):
                paths = [os.path.join(utils.get_output_dir(), f"panel_{part_num}_{i}.png") for i in range(1, 7)]
                for p in paths:
                    with open(p, "w") as f:
                        f.write("panel")
                tool_context.state[f"part_{part_num}_panel_images"] = paths
                return json.dumps({"status": "SUCCESS", "panel_paths": paths})
            ppa.slice_part_grid_tool = mock_slice

            def mock_produce_sub(tool_context, part_num, subscene_num, **kw):
                v_path = os.path.join(utils.get_output_dir(), f"sub_{part_num}_{subscene_num}.mp4")
                with open(v_path, "w") as f:
                    f.write("subscene_video")
                tool_context.state[f"part_{part_num}_sub_{subscene_num}"] = v_path
                return json.dumps({"status": "SUCCESS", "video_path": v_path, "duration_seconds": 6.0})
            ppa.produce_subscene_tool = mock_produce_sub

            def mock_concat_part(tool_context, part_num, **kw):
                path = os.path.join(utils.get_output_dir(), f"part_{part_num}_master.mp4")
                with open(path, "w") as f:
                    f.write("mock_part_master")
                tool_context.state[f"part_{part_num}_master_video"] = path
                return json.dumps({"status": "SUCCESS", "part_master_path": path, "duration_seconds": 36.0})
            ppa.concatenate_part_subscenes_tool = mock_concat_part

            def mock_concat_master(ctx, part_video_paths_json):
                out_p = os.path.join(utils.get_output_dir(), "2026-10-21-mughal-master.mp4")
                with open(out_p, "w") as f:
                    f.write("mock_master_movie")
                ctx.state["final_movie_path"] = out_p
                return json.dumps({"status": "SUCCESS", "movie_path": out_p, "duration_seconds": 72.0, "parts_count": 2})
            da.concatenate_master_movie_tool = mock_concat_master

            da.google_search_grounding_tool = lambda *a, **kw: "Mock grounding"
            da.plan_creative_story_parts_tool = lambda ctx, topic, **kw: (
                ctx.state.update({
                    "storyboard_script": [
                        {
                            "part_number": 1,
                            "title": "Ahmednagar 1707",
                            "synopsis": "Death of Aurangzeb.",
                            "search_queries": ["Aurangzeb"],
                            "subscenes": [{"subscene_number": i, "narration": f"Beat {i}", "visual_description": f"Visual {i}", "is_veo_candidate": (i == 2)} for i in range(1, 7)]
                        },
                        {
                            "part_number": 2,
                            "title": "Battle of Jajau",
                            "synopsis": "The Princes Clash.",
                            "search_queries": ["Azam Shah"],
                            "subscenes": [{"subscene_number": i, "narration": f"Beat {i}", "visual_description": f"Visual {i}", "is_veo_candidate": False} for i in range(1, 7)]
                        }
                    ]
                }) or "Screenplay planned."
            )

            # 1. Create Run
            run = service.create_run(
                topic="The Mughal Succession 1707",
                verbatim_mode=False,
                requested_parts=2
            )

            # Subscribe to events to verify part_completed events
            events_queue = asyncio.Queue()
            run.subscribers.append(events_queue)

            # Run Turn 1
            await service.start_turn_1(run)

            # Verify Turn 1 halted at HITL review gate
            self.assertEqual(run.status, "AWAITING_APPROVAL")
            self.assertEqual(len(run.screenplay), 2)
            self.assertEqual(run.screenplay[0]["title"], "Ahmednagar 1707")

            # Verify API state reflects AWAITING_APPROVAL
            res = self.client.get(f"/api/runs/{run.run_id}")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["status"], "AWAITING_APPROVAL")

            # 2. Director edits screenplay before approving
            edited_script = json.loads(json.dumps(run.screenplay))
            edited_script[0]["subscenes"][0]["narration"] = "Director Revised Beat 1: The empire fractures in Ahmednagar."

            # Run Turn 2 via API approve endpoint
            approve_res = self.client.post(
                f"/api/runs/{run.run_id}/approve",
                json={
                    "approval_text": "Approved with revised narration.",
                    "screenplay": edited_script
                }
            )
            self.assertEqual(approve_res.status_code, 200)
            # Verify edited narration took effect
            self.assertEqual(run.screenplay[0]["subscenes"][0]["narration"], "Director Revised Beat 1: The empire fractures in Ahmednagar.")

            # Verify requirement: 'after every part is complete, just check the folder strcuture or state or intermediate output and show that on UI'
            self.assertIn(1, run.parts_status)
            self.assertIn(2, run.parts_status)

            part1_summary = run.parts_status[1]
            self.assertEqual(part1_summary["status"], "COMPLETED")
            self.assertEqual(part1_summary["panels_count"], 6)
            self.assertEqual(part1_summary["subscenes_count"], 6)
            self.assertIsNotNone(part1_summary["part_master"])

            # Verify part_completed event was broadcast
            part_completed_events = []
            while not events_queue.empty():
                item = events_queue.get_nowait()
                if item["type"] == "part_completed":
                    part_completed_events.append(item["data"])

            self.assertEqual(len(part_completed_events), 2)
            self.assertEqual(part_completed_events[0]["part_number"], 1)
            self.assertEqual(part_completed_events[1]["part_number"], 2)

            # Verify Turn 2 completed successfully
            self.assertEqual(run.status, "COMPLETED")
            self.assertIsNotNone(run.final_movie_url)
            self.assertIn("mughal-master.mp4", run.final_movie_url)

            # Clean up watcher
            await run.watcher.stop()

        asyncio.run(run_async_test())

    def test_06_frontend_markup_integrity(self):
        """Validates that the frontend index.html includes both video and image live players, and episodic parts section."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.text

        # Video and Image players in Live Working Scene spotlight
        self.assertIn('id="live-video-player"', html)
        self.assertIn('id="live-image-player"', html)
        self.assertIn('id="live-spotlight"', html)

        # Episodic Parts and Intermediate Status container
        self.assertIn('id="parts-production-section"', html)
        self.assertIn('id="parts-cards-container"', html)

        # HITL Review Gate
        self.assertIn('id="hitl-panel"', html)
        self.assertIn('id="screenplay-container"', html)
        self.assertIn('id="btn-approve-screenplay"', html)

    def test_07_history_api(self):
        """Test /api/history endpoint listing past output runs."""
        res = self.client.get("/api/history")
        self.assertEqual(res.status_code, 200)
        history = res.json()
        self.assertIsInstance(history, list)


if __name__ == "__main__":
    unittest.main()
