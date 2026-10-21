# Storyboard AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Google ADK 2.11.0](https://img.shields.io/badge/Google%20ADK-2.11.0-orange.svg)](https://github.com/google/adk)
[![Google GenAI SDK](https://img.shields.io/badge/google--genai-2.31.0-green.svg)](https://github.com/googleapis/python-genai)

An enterprise-grade **Multi-Agent Video Studio** built on the **Google Agent Development Kit (`google.adk` 2.11.0)** that transforms any topic or verbatim script into a fully animated, narrated, and subtitled historical documentary movie.

---

## Architecture: Google ADK 2.0 Graph Workflow & Multi-Agent Studio

Storyboard AI is structured as a native **Google ADK 2.0 Graph Workflow (`google.adk.Workflow`)** orchestrating specialized episodic nodes and tools without monolithic procedural shortcuts:

```mermaid
graph TD
    Start([START]) --> PlanNode["Node 1: Story Planning Node<br/>(Google Search Grounding + Screenwriter)"]
    PlanNode --> ReviewNode["Node 2: HITL Review Gate<br/>(Presents breakdown + RequestInput)"]
    ReviewNode -->|Human Approval| LoopNode["Node 3: Director Parts Loop Node<br/>(Sequentially dispatches Part 1..N)"]
    
    subgraph PartWorkflow [Episodic Part Production Node (Dispatched per Part)]
        LoopNode --> RefTool["1. Reference Portrait Search<br/>(DDGS + Wikipedia Commons)"]
        RefTool --> GridGen["2. Dynamic 3x2 Comic Grid Artist<br/>(Gemini 3.1 Flash Image + Theme Context)"]
        GridGen --> CVSlice["3. CV 16:9 720p Panel Slicer"]
        CVSlice --> SubLoop["4. Sequential 6-Panel Loop<br/>(SAM 3 + Whiteboard + Veo + Calibrated TTS)"]
        SubLoop --> PartConcat["5. Part Master Assembly<br/>(part_i_master.mp4 + Storage Cleanup)"]
    end
    
    PartConcat -->|Context returns to Director Loop| LoopNode
    LoopNode -->|All Parts Finished| AssemblyNode["Node 4: Master Movie Assembly Node<br/>(Concatenates Part videos into Final Movie)"]
    AssemblyNode --> MasterVideo([Master Movie MP4<br/>output/YYYY-MM-DD-HH-MM-SS-title/])
```

---

## Core Capabilities & Engineering Innovations

### 1. Divergent Dual-Mode Execution
- **Creative Mode (Visual-First)**:
  - Visuals lead the production. Fast whiteboard drawing (720p H.264) is rendered first.
  - If a sub-scene is marked as a climactic beat and Veo is enabled, an 8-second Veo clip is synthesized and seamlessly attached to the drawing ($V_{\text{total}} = V_{\text{draw}} + 8\text{s}$). No duration gate restricts Veo.
  - The **Narration Refiner** calibrates the spoken word count to match the visual duration ($V_{\text{total}} + 1.5\text{s}$ hold).
  - Gemini Native TTS (Aoede) synthesizes human-like narration, merged with zero cropping and burned subtitles.
- **Strict Verbatim Mode (Audio-First)**:
  - Source text is locked with 0 word alterations.
  - Verbatim audio is synthesized first to measure exact spoken duration ($A_{\text{dur}}$).
  - Whiteboard drawing speed adaptively scales to match $A_{\text{dur}}$.
  - **Veo Duration Gate (Verbatim Mode Only)**: Evaluates $A_{\text{dur}} \ge V_{\text{draw}} + 6.0\text{s}$. If audio is too short, Veo is gated OFF to prevent unnatural frozen frames or dead air.

### 2. Mandatory Grand Opening Title Scene
- **Part 1, Sub-scene 1** (the very first scene of the entire video) is strictly enforced as a grand cinematic prologue.
- The visual generator places a prominent, ornate title banner, marquee, or royal parchment scroll with bold comic typography displaying the movie's main title, introducing the historical saga before proceeding to the narrative beats.

### 3. Ultra-Fast Whiteboard Animation Engine
- Optimized stroke detection grids (`split_len=16`, `object_skip_rate=12`, `bg_object_skip_rate=20`) render full 720p drawings in **~2.28 seconds** (down from 15–20s).
- **H.264 Normalization**: Automatically converts OpenCV `mp4v` output to native **H.264 (`yuv420p` @ 25.0 fps)** immediately upon release, preventing downstream bitstream decoding damage.

### 4. Normalized 720p 25fps Filter-Complex Concatenation
- Uses explicit FFmpeg `filter_complex` with aspect-ratio preservation and fixed 25fps timebases:
  ```bash
  [0:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=25,setsar=1,format=yuv420p[v0];
  [1:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=25,setsar=1,format=yuv420p[v1];
  [v0][v1]concat=n=2:v=1:a=0[outv]
  ```
- Guaranteed **zero decoding errors** and zero dropped frames when attaching Veo clips to whiteboard drawings.

### 5. Automatic Storage Cleanup & Transient Retries
- **Sub-Scene Cleanup**: Once the final subtitled sub-scene video is verified, intermediate raw assets (drawing-only video, raw Veo clips, intermediate WAV audio, un-subtitled AV files, and segmentation JSON) are automatically deleted.
- **Part Cleanup**: Once the Part Master video (`part_{i}_master.mp4`) is assembled and verified, the 6 individual sub-scene clips are purged.
- **Retry Logic**: Key synthesis steps (TTS, Whiteboard drawing, AV merge) include automatic retry handling to gracefully recover from temporary network blips.

### 6. Readable Dated Output Directories
- Outputs are saved into self-describing, timestamped directories:
  ```
  output/YYYY-MM-DD-HH-MM-SS-<slugified-title>/
  ├── gen_image_1.png                              # 3x2 Comic Grid
  ├── part_1_master.mp4                            # Part 1 Master Video
  ├── part_2_master.mp4                            # Part 2 Master Video
  ├── part_3_master.mp4                            # Part 3 Master Video
  └── YYYY-MM-DD-HH-MM-SS-<title>-master.mp4       # Final Assembled Movie
  ```

---

## Directory Structure

```
storyboard-ai/
├── storyboard-agent/
│   ├── agents/                           # Pure Google ADK Agents
│   │   ├── __init__.py                   # Exporting factory methods
│   │   ├── director_agent.py             # Level 1 Showrunner (Search Grounding & Movie Concat)
│   │   ├── writer_agent.py               # Level 2 Screenplay & Verbatim Chunker
│   │   ├── part_production_agent.py      # Level 2 Episodic Director (6 Sub-Scene Coordinator)
│   │   └── tools_common.py               # Common ADK State Tools & Media Processors
│   ├── assets/                           # Drawing hand and mask assets
│   ├── config.py                         # Global configuration (Veo, SAM, Resolution, Models)
│   ├── cost_tracker.py                   # Token and financial expenditure auditor
│   ├── main.py                           # Interactive CLI entrypoint
│   ├── state.py                          # Session and production data models
│   ├── tests/                            # Complete isolated test suite
│   │   ├── test_tool_panel_veo_concat.py # Unit test: Drawing + Veo lossless concat
│   │   ├── test_subagent_panel.py        # Unit test: Sub-agent single panel & gate
│   │   ├── test_adk_3part_movie.py       # Integration test: 3-Part movie production
│   │   └── ...                           # Component unit tests
│   └── tools/                            # Low-level CV, GenAI & Media tools
│       ├── draw_animation.py             # Fast Whiteboard Drawing Engine
│       ├── grid_panel_tool.py            # 3x2 Grid Slicer & Comic Framing
│       ├── image_gen_tool.py             # Gemini 3.1 Flash Image Interface
│       ├── media_tools.py                # Normalized FFmpeg Concat, Veo & Subtitle Burner
│       ├── narration_refiner.py          # Narration Word Calibration
│       ├── reference_evaluator.py        # Portrait Grounding (DDGS + Wikipedia)
│       ├── segmentation_tool.py          # SAM 3 Instance Segmentation
│       ├── tts_tool.py                   # Gemini Native Aoede TTS
│       └── utils.py                      # Global media utilities & output management
├── requirements.txt                      # Project dependencies
└── README.md                             # Documentation
```

---

## Quick Start

### 1. Prerequisites
- Python 3.10+ (Conda environment recommended)
- `ffmpeg` installed and available on system PATH
- A Google Gemini API Key (`GEMINI_API_KEY`)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/your-username/storyboard-ai.git
cd storyboard-ai

# Activate your environment
conda activate storyboard-ai

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file or export your Gemini API key:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

Configure settings in `storyboard-agent/config.py`:
- `ENABLE_VEO = True` (or `False` to run without Veo)
- `ENABLE_SAM = True` (or `False` to run without Cloud Run SAM 3)
- `OUTPUT_WIDTH = 1280`, `OUTPUT_HEIGHT = 720` (Standard 720p HD)

### 4. Running the Web Studio UI
Launch the interactive web UI:
```bash
python run_ui.py
```
Open **http://localhost:8080** in your browser.
- **Real-time Live Working Scene Player**: Live video preview as whiteboard drawings, Veo clips, and panel audio are created.
- **Interactive HITL Review Gate**: Turn 1 pauses for human director approval of the structured screenplay before Turn 2 video production begins.
- **Per-Run Output Directories**: Every run automatically creates a new timestamped folder in `output/` with all intermediate and master assets.
- **Multi-Tab Asset Gallery & Folder Explorer**: Browse character portraits, 3x2 comic grids, sliced 720p panels, and part masters.

### 5. Running the CLI Entrypoint
```bash
python storyboard-agent/main.py
```

### 6. Cloud Run Deployment
The application is pre-configured for Google Cloud Run:
```bash
# Build and deploy directly from source
gcloud run deploy storyboard-ai-studio \
  --source . \
  --region us-central1 \
  --set-env-vars GEMINI_API_KEY="your-gemini-api-key" \
  --allow-unauthenticated
```

### 7. Running Unit Tests
Unit tests are located in `storyboard-agent/tests/`:
```bash
# Test Web UI server & multi-turn HITL lifecycle
python storyboard-agent/tests/test_ui_server.py

# Test ADK 2.0 two-level graph workflow & HITL review gate
python storyboard-agent/tests/test_adk_workflow_hitl.py
```

---

## License

This project is licensed under the Apache License 2.0.
