"""
Storyboard AI Studio - Launcher.
Run with:
    ~/miniconda3/python.exe run_ui.py
"""
import argparse
import os
import sys
import uvicorn

# Ensure storyboard-agent root is in path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.join(ROOT_DIR, "storyboard-agent")
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def main():
    parser = argparse.ArgumentParser(description="Storyboard AI Studio Web Server")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8080")), help="Port to bind to")
    parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"), help="Host to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload for development")
    args = parser.parse_args()

    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("=" * 75)
    print("       STORYBOARD AI STUDIO - GOOGLE ADK MULTI-AGENT UI              ")
    print("=" * 75)
    print(f"  Local Studio URL:  http://localhost:{args.port}")
    print(f"  Network Bind:      http://{args.host}:{args.port}")
    print("  Output Directory:  ./output/")
    print("=" * 75)

    uvicorn.run("web.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
