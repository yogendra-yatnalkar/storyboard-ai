import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import run_pipeline


if __name__ == "__main__":
    prompt = (
        "Sing Hanuman Song and show how Hanuman Ji met Mata Sita and Lanka Dahan Story. More narration and small animation lenght works. (strictly 3 scenes)"
    )
    print(f"Testing pipeline with prompt: '{prompt}'")
    final_video = run_pipeline(
        prompt, 
        do_research=False, 
        do_web_search=True, 
        fast_mode=True, 
        language="Hindi",
        enable_veo=True,
        veo_direction_by_director=True,
        enable_url_context=False,
        enable_google_search=True
    )
    if final_video:
        print(f"Pipeline SUCCESS! Final Video: {final_video}")
    else:
        print("Pipeline FAILED.")
