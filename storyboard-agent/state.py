"""
State & Dataclass Definitions for Storyboard Multi-Agent Service.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class RunConfig:
    # Research Mode
    research_mode: str = "web_search"      # "none", "web_search", "url_grounding", "deep_research"
    enable_google_search: bool = True       # Google Search tool for web grounding
    enable_url_context: bool = False        # URL grounding tool

    # Narration Mode
    verbatim_mode: bool = False             # True ONLY when explicitly requested at start
    language: str = "english"
    
    # Segmentation & Animation
    enable_sam: bool = True                 # SAM 3 segmentation for object drawing animation
    
    # Video & Veo Configuration
    enable_veo: bool = False                # Generate Veo 720p clips
    veo_scope: str = "none"                 # "none", "last_panel", "random_50", "all_panels"
    veo_mode: str = "after_drawing"         # "after_drawing" (appends Veo after drawing animation), "veo_only", "drawing_only"
    veo_direction: bool = True              # Whether Director writes specific Veo prompts
    
    # References & Layout
    max_references: int = 2
    panels_per_scene: int = 6               # 3x2 grid layout

    # Execution
    fast_mode: bool = False
    max_scenes: Optional[int] = None
    output_dir: str = "output"


@dataclass
class PanelPlan:
    panel_number: int                       # 1 to 6
    narration_beat: str                     # Specific narration beat for this panel
    visual_description: str                 # What is shown/drawn in this panel
    veo_prompt: Optional[str] = None        # Optional Veo motion prompt for this panel


@dataclass
class ScenePlan:
    scene_number: int
    narration: str                          # Full scene narration text
    description: str                        # High-level summary of the scene
    visual_setup: str                       # Overall artistic composition & mood
    search_queries: List[str] = field(default_factory=list)      # Up to 2 generic entity queries
    reference_image_paths: List[str] = field(default_factory=list)
    panels: List[PanelPlan] = field(default_factory=list)        # Exactly 6 panels (3x2 grid)
    grid_image_path: Optional[str] = None
    text_overlay: Optional[str] = None
    key_information: Optional[str] = None
    emotional_beat: Optional[str] = None
    veo_prompt: Optional[str] = None


@dataclass
class SceneResult:
    scene_number: int
    scene_plan: ScenePlan
    # Asset paths
    reference_image_paths: List[str] = field(default_factory=list)
    grid_image_path: Optional[str] = None
    panel_image_paths: List[str] = field(default_factory=list)
    panel_video_paths: List[str] = field(default_factory=list)
    final_scene_video_path: Optional[str] = None
    duration: float = 0.0
    status: str = "PENDING"
    error_message: Optional[str] = None


@dataclass
class CharacterEntity:
    name: str
    description: str
    reference_image_path: Optional[str] = None
    visual_anchor_scene: int = 1


@dataclass
class StoryboardState:
    user_instruction: str
    run_config: RunConfig
    research_material: Optional[str] = None
    global_plan: Dict[str, Any] = field(default_factory=dict)
    character_registry: Dict[str, CharacterEntity] = field(default_factory=dict)
    scene_plans: List[ScenePlan] = field(default_factory=list)
    scene_results: List[SceneResult] = field(default_factory=list)
    final_video_path: Optional[str] = None
    execution_logs: List[str] = field(default_factory=list)

    def log(self, message: str):
        print(f"[StoryboardAgent] {message}")
        self.execution_logs.append(message)
