"""
Part Production Agent for Storyboard Multi-Agent Service.
Graph-Based Episodic Production Workflow built on Google ADK 2.0 Workflow.

Graph Structure:
START -> search_references_node -> generate_grid_node -> slice_grid_node -> subscenes_loop_node -> concat_part_node

1. search_references_node: Searches portrait references for historical figures.
2. generate_grid_node: Generates 3x2 comic grid image with dynamic theme context & character continuity.
3. slice_grid_node: Slices the 3x2 grid into 6 720p panels.
4. subscenes_loop_node: 6-panel loop calling produce_subscene_tool with dynamic scene context.
5. concat_part_node: Concatenates the 6 panels into part_{i}_master.mp4 and cleans intermediate raw assets.
"""
import json
import os
from typing import Dict, Any, List, Optional
from google.adk import Context, Workflow
from google.adk.workflow import START, FunctionNode

from config import MODEL_NAME, ENABLE_VEO, ENABLE_SAM
from .tools_common import (
    search_part_references_tool,
    generate_part_grid_tool,
    slice_part_grid_tool,
    produce_subscene_tool,
    concatenate_part_subscenes_tool
)


def _get_active_part_plan(ctx: Context) -> tuple[int, dict]:
    script = ctx.state.get("storyboard_script", [])
    part_num = ctx.state.get("active_part_number", 1)
    plan = next((p for p in script if p.get("part_number") == part_num), {})
    return part_num, plan


def search_references_node(ctx: Context) -> str:
    """Node 1: Reference Portrait Acquisition."""
    part_num, plan = _get_active_part_plan(ctx)
    entities = plan.get("search_queries", [])
    if entities:
        print(f"\n   [PartWorkflow #{part_num}: Node 1] Searching reference portraits for: {entities}...")
        return search_part_references_tool(ctx, json.dumps(entities), part_num=part_num)
    return "No reference entities for this part."


def generate_grid_node(ctx: Context) -> str:
    """Node 2: Dynamic 3x2 Comic Grid Generation."""
    part_num, plan = _get_active_part_plan(ctx)
    title = plan.get("title", f"Part {part_num}")
    synopsis = plan.get("synopsis", "")
    subscenes = plan.get("subscenes", [])
    story_theme = f"{title}: {synopsis}"

    print(f"   [PartWorkflow #{part_num}: Node 2] Generating 3x2 Comic Grid with theme: '{story_theme[:60]}...'")
    grid_res = generate_part_grid_tool(
        tool_context=ctx,
        part_num=part_num,
        panels_json=json.dumps(subscenes),
        story_theme=story_theme,
        visual_style_prompt="Dramatic historical documentary graphic novel with Indian miniature art nuances"
    )
    return grid_res


def slice_grid_node(ctx: Context) -> str:
    """Node 3: Slice 3x2 Grid into 6 720p Panels."""
    part_num, _ = _get_active_part_plan(ctx)
    print(f"   [PartWorkflow #{part_num}: Node 3] Slicing grid into 6 720p panels...")
    slice_res = slice_part_grid_tool(tool_context=ctx, part_num=part_num)
    return slice_res


def subscenes_loop_node(ctx: Context) -> str:
    """Node 4: Sequential 6-Sub-Scene Production Loop with Dynamic Scene Context."""
    part_num, plan = _get_active_part_plan(ctx)
    title = plan.get("title", f"Part {part_num}")
    subscenes = plan.get("subscenes", [])
    verbatim_mode = ctx.state.get("verbatim_mode", False)

    panel_paths = ctx.state.get(f"part_{part_num}_panel_images", [])
    if not panel_paths or len(panel_paths) < 6:
        return json.dumps({"status": "FAILED", "error": f"Missing sliced panels for Part {part_num}."})

    print(f"   [PartWorkflow #{part_num}: Node 4] Executing 6-subscene production loop...")
    subscene_video_paths = []

    for idx, panel_path in enumerate(panel_paths[:6]):
        s_num = idx + 1
        sub_data = subscenes[idx] if idx < len(subscenes) else {}
        narration = sub_data.get("narration", f"Part {part_num} Beat {s_num}")
        visual_desc = sub_data.get("visual_description", "")
        is_veo = sub_data.get("is_veo_candidate", False)
        veo_prompt = sub_data.get("veo_motion_prompt", "")

        scene_ctx = f"{visual_desc} | Historical Setting: {title}"

        sub_res = produce_subscene_tool(
            tool_context=ctx,
            part_num=part_num,
            subscene_num=s_num,
            panel_image_path=panel_path,
            narration_beat=narration,
            is_veo_candidate=is_veo,
            veo_prompt=veo_prompt,
            scene_context=scene_ctx,
            emotional_tone="epic historical documentary",
            verbatim_mode=verbatim_mode
        )
        info = json.loads(sub_res) if isinstance(sub_res, str) and sub_res.startswith("{") else {}
        if info.get("status") == "SUCCESS":
            subscene_video_paths.append(info.get("video_path"))
        else:
            print(f"Warning: Sub-scene {s_num} failed: {info.get('error')}")

    ctx.state[f"part_{part_num}_subscene_videos"] = subscene_video_paths
    return json.dumps({"status": "SUCCESS", "subscenes_produced": len(subscene_video_paths)})


def concat_part_node(ctx: Context) -> str:
    """Node 5: Concatenate 6 Sub-Scenes into Part Master & Clean Intermediate Storage."""
    part_num, _ = _get_active_part_plan(ctx)
    paths = ctx.state.get(f"part_{part_num}_subscene_videos", [])
    print(f"   [PartWorkflow #{part_num}: Node 5] Concatenating {len(paths)} sub-scenes into Part Master...")
    concat_res = concatenate_part_subscenes_tool(
        tool_context=ctx,
        part_num=part_num,
        subscene_video_paths_json=json.dumps(paths)
    )
    info = json.loads(concat_res) if isinstance(concat_res, str) and concat_res.startswith("{") else {}
    part_master = info.get("part_master_path")

    if "completed_part_videos" not in ctx.state:
        ctx.state["completed_part_videos"] = []
    if part_master and part_master not in ctx.state["completed_part_videos"]:
        ctx.state["completed_part_videos"].append(part_master)

    return concat_res


def create_part_production_workflow() -> Workflow:
    """
    Constructs the ADK 2.0 Graph Workflow for producing a single Part.
    START -> search_references_node -> generate_grid_node -> slice_grid_node -> subscenes_loop_node -> concat_part_node
    """
    return Workflow(
        name="part_production_workflow",
        description="ADK 2.0 Episodic Part Production Graph Workflow",
        edges=[
            (START, search_references_node, generate_grid_node, slice_grid_node, subscenes_loop_node, concat_part_node)
        ]
    )


create_part_production_node = create_part_production_workflow
create_part_production_agent = create_part_production_workflow
