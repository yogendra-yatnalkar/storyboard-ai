"""
Research Tool for Storyboard Agent.
Supports:
1. Fast Web-Grounded Research using Gemini 3.8 Flash with Google Search / URL Context
2. Deep Research using Gemini Deep Research
"""
import re
import time
from typing import Optional
from google.genai import types

from config import MODEL_NAME, DEEP_RESEARCH_MODEL, get_generation_config
from cost_tracker import tracker
from state import RunConfig
from . import utils


def execute_deep_research(context: str) -> Optional[str]:
    """Executes background Gemini Deep Research."""
    if not utils.client:
        return None

    print(f"Research: Starting Deep Research for: '{context}'...")
    try:
        interaction = utils.client.interactions.create(
            input=f"Research comprehensive, detailed factual information about: {context}. Include key dates, milestones, key figures, and important contextual facts.",
            agent=DEEP_RESEARCH_MODEL,
            background=True
        )
        print(f"Deep Research initiated: ID={interaction.id}")

        while True:
            interaction = utils.client.interactions.get(interaction.id)
            if interaction.status == "completed":
                if interaction.outputs:
                    report = interaction.outputs[-1].text
                    utils._save_to_run_folder(report, "deep_research_report.md")
                    return report
                return None
            elif interaction.status == "failed":
                print(f"Deep Research failed: {interaction.error}")
                return None
            time.sleep(10)
    except Exception as e:
        print(f"Deep Research error: {e}")
        return None


def execute_web_grounded_research(context: str, enable_google_search: bool = True, enable_url_context: bool = False) -> Optional[str]:
    """Executes fast web-grounded research using Gemini 3.8 Flash with Google Search / URL Context."""
    if not utils.client:
        return None

    print(f"Research: Performing Web-Grounded Research for: '{context}'...")
    print(f"  Google Search: {'Enabled' if enable_google_search else 'Disabled'} | URL Context: {'Enabled' if enable_url_context else 'Disabled'}")

    try:
        generation_tools = []
        if enable_google_search:
            generation_tools.append({"google_search": {}})
        if enable_url_context:
            generation_tools.append({"url_context": {}})

        prompt = (
            f"Perform a comprehensive web search to provide a detailed, factual summary about: {context}. "
            "Include key dates, milestones, and important contextual facts. "
            "This will be used as a source for a documentary/storyboard script."
        )

        config = get_generation_config(
            model=MODEL_NAME,
            tools=generation_tools if generation_tools else None
        )
        response = utils.client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=config
        )
        tracker.record_usage_metadata(response.usage_metadata)
        report = response.text.strip()
        utils._save_to_run_folder(report, "web_research_report.md")
        return report

    except Exception as e:
        print(f"Web-grounded research error: {e}")
        return None


def research_tool_fn(context: str, run_config: RunConfig) -> Optional[str]:
    """
    Entry point for research based on user-configured research_mode.
    """
    mode = run_config.research_mode.lower()
    if mode == "none" or run_config.verbatim_mode:
        print("Research: Skipped as per configuration.")
        return None

    # Check for URL in context to auto-enable URL grounding
    url_match = re.search(r'(https?://[^\s]+)', context)
    has_url = url_match is not None
    enable_url = run_config.enable_url_context or has_url
    if has_url:
        print(f"Research: Detected URL in input context: {url_match.group(1)}")

    if mode == "deep_research":
        return execute_deep_research(context)
    elif mode == "url_grounding":
        return execute_web_grounded_research(context, enable_google_search=False, enable_url_context=True)
    elif mode in ["web_search", "web_and_url"]:
        return execute_web_grounded_research(context, enable_google_search=run_config.enable_google_search, enable_url_context=enable_url)
    else:
        return execute_web_grounded_research(context, enable_google_search=True, enable_url_context=enable_url)
