"""
Cost and Token Tracker for Storyboard Multi-Agent Service.
Computes and reports token counts, image/audio counts, and estimated USD expenditure.
"""
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class CostTracker:
    # Token usage
    prompt_tokens: int = 0
    candidate_tokens: int = 0
    total_tokens: int = 0
    
    # Asset generation counts
    images_generated: int = 0
    audio_seconds_generated: float = 0.0
    videos_generated: int = 0
    search_queries_count: int = 0
    
    # Unit prices (USD)
    PRICE_INPUT_1M: float = 0.15      # $0.15 per 1M input tokens
    PRICE_OUTPUT_1M: float = 0.60     # $0.60 per 1M output tokens
    PRICE_PER_IMAGE: float = 0.035    # $0.035 per flash image
    PRICE_PER_AUDIO_SEC: float = 0.0001 # $0.0001 per second audio
    PRICE_PER_VEO_VIDEO: float = 0.08 # $0.08 per 720p veo clip

    def record_usage_metadata(self, metadata: Any):
        """Records token usage from a Gemini response object if available."""
        if metadata:
            if hasattr(metadata, "prompt_token_count") and metadata.prompt_token_count:
                self.prompt_tokens += metadata.prompt_token_count
            if hasattr(metadata, "candidates_token_count") and metadata.candidates_token_count:
                self.candidate_tokens += metadata.candidates_token_count
            if hasattr(metadata, "total_token_count") and metadata.total_token_count:
                self.total_tokens += metadata.total_token_count

    def record_image(self, count: int = 1):
        self.images_generated += count

    def record_audio(self, seconds: float):
        self.audio_seconds_generated += max(0.0, seconds)

    def record_video(self, count: int = 1):
        self.videos_generated += count

    def record_search(self, count: int = 1):
        self.search_queries_count += count

    def get_estimated_cost_usd(self) -> float:
        token_cost = (
            (self.prompt_tokens / 1_000_000.0) * self.PRICE_INPUT_1M +
            (self.candidate_tokens / 1_000_000.0) * self.PRICE_OUTPUT_1M
        )
        image_cost = self.images_generated * self.PRICE_PER_IMAGE
        audio_cost = self.audio_seconds_generated * self.PRICE_PER_AUDIO_SEC
        video_cost = self.videos_generated * self.PRICE_PER_VEO_VIDEO
        return token_cost + image_cost + audio_cost + video_cost

    def compute_total_cost_usd(self) -> float:
        return self.get_estimated_cost_usd()

    def get_metrics_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "candidate_tokens": self.candidate_tokens,
            "total_tokens": self.total_tokens,
            "images_generated": self.images_generated,
            "audio_seconds": round(self.audio_seconds_generated, 1),
            "videos_generated": self.videos_generated,
            "search_queries": self.search_queries_count,
            "estimated_cost_usd": round(self.get_estimated_cost_usd(), 4),
            "summary_report": self.get_summary_report()
        }

    def get_summary_report(self) -> str:
        cost = self.get_estimated_cost_usd()
        lines = [
            "--------------------------------------------------",
            "          COST & TOKEN EXPENDITURE REPORT         ",
            "--------------------------------------------------",
            f"Prompt Tokens:        {self.prompt_tokens:,}",
            f"Candidate Tokens:     {self.candidate_tokens:,}",
            f"Total Tokens:         {self.total_tokens:,}",
            f"Images Generated:     {self.images_generated} (~${self.images_generated * self.PRICE_PER_IMAGE:.4f})",
            f"Audio Generated:      {self.audio_seconds_generated:.1f}s (~${self.audio_seconds_generated * self.PRICE_PER_AUDIO_SEC:.4f})",
            f"Videos Generated:     {self.videos_generated} (~${self.videos_generated * self.PRICE_PER_VEO_VIDEO:.4f})",
            f"DDG Searches:         {self.search_queries_count} ($0.00)",
            "--------------------------------------------------",
            f"ESTIMATED TOTAL COST: ${cost:.5f} USD",
            "--------------------------------------------------",
        ]
        return "\n".join(lines)


# Global singleton tracker instance
tracker = CostTracker()
