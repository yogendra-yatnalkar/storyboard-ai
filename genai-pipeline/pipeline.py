import os
import time
import datetime
import concurrent.futures
import subprocess
import sys

# Reconfigure stdout/stderr to UTF-8 to support Unicode/Hindi character printing on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from config import SAM_API_URL
from tools import (
    research_tool_fn,
    web_grounded_research_tool_fn,
    director_tool_fn, 
    prompt_tool_fn, 
    image_gen_tool_fn, 
    generate_tts_audio_tool_fn,
    segmentation_tool_fn,
    merge_audio_video_tool_fn,
    concatenate_videos_tool_fn,
    burn_subtitles_to_video_tool_fn,
    transcribe_audio_tool_fn,
    refine_narration_tool_fn,
    draw_animation_tool_fn,
    set_output_dir,
    get_video_duration,
    reference_search_tool_fn,
    generate_video_veo_tool_fn
)

# --- Helper functions for robustness ---

def _is_valid_path(path: str) -> bool:
    """Check if a tool returned a valid file path (not an error string)."""
    if not path:
        return False
    if "Error" in path or "error" in path:
        return False
    return os.path.exists(path)

def _retry(fn, *args, max_retries: int = 3, delay: float = 5.0, label: str = "", **kwargs):
    """
    Retry a function call on failure (handles transient network errors).
    Returns the result on success, or None on exhausted retries.
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            result = fn(*args, **kwargs)
            # Check if result is an error string (some tools return error strings instead of raising)
            if isinstance(result, str) and ("Error" in result or "error" in result) and not os.path.exists(result):
                raise RuntimeError(result)
            return result
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                print(f"  [!] {label} failed (attempt {attempt}/{max_retries}): {e}")
                print(f"  Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"  [X] {label} failed after {max_retries} attempts: {e}")
    return None

# --- Main Pipeline ---

def run_pipeline(user_context: str, do_research: bool = True, do_web_search: bool = False, use_internet_image_search: bool = True, fast_mode: bool = False, language: str = "english", enable_veo: bool = False, veo_direction_by_director: bool = False, enable_url_context: bool = False, enable_google_search: bool = True):
    print(f"--- Starting Storyboard Pipeline for context: {user_context} (Language: {language}) ---")
    
    # 0. Setup Output Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.getcwd(), "output", f"run_{timestamp}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    set_output_dir(output_dir)
    print(f"Artifacts will be saved to: {output_dir}")

    # 1. Research (Optional)
    research_report = None
    if do_research:
        print("\nStep 1: Performing Deep Research...")
        research_report = research_tool_fn(user_context)
        print("Research completed.")
    elif do_web_search:
        print("\nStep 1: Performing Web-Grounded Research (Fast)...")
        # Automatically detect URL in user_context if not explicitly passed
        import re
        url_match = re.search(r'(https?://[^\s]+)', user_context)
        has_url = url_match is not None
        if has_url:
            print(f"Detected URL in prompt: {url_match.group(1)}")
                
        research_report = web_grounded_research_tool_fn(user_context, enable_google_search=enable_google_search, enable_url_context=enable_url_context or has_url)
        print("Web-Grounded Research completed.")
    else:
        print("\nStep 1: Skipping Research as per request. Using provided context directly.")

    # Step 2: Director Planning — The Director plans the entire video journey
    print("\nStep 2: Director Planning & Scene Writing...")
    video_plan = director_tool_fn(user_context, research_material=research_report, language=language, enable_veo=enable_veo, veo_direction_by_director=veo_direction_by_director)
    global_plan = video_plan.get("global_plan", {})
    scenes = video_plan.get("scenes", [])
    print(f"Director planned {len(scenes)} scenes. Tone: {global_plan.get('tone')}, Arc: {global_plan.get('narrative_arc', 'N/A')}")

    final_videos = []
    prev_image_path = None
    failed_scenes = []

    def process_scene_helper(scene_num, scene, local_prev_image_path):
        print(f"\n{'='*60}")
        print(f"--- Processing Scene {scene_num}/{len(scenes)} ---")
        
        try:
            description = scene.get('description', 'No description')
            narration = scene.get('narration', 'No narration')
            visual_setup = scene.get('visual_setup', '')
            summary = scene.get('summary', '')
            emotional_beat = scene.get('emotional_beat', '')
            search_query = scene.get('search_query', '')
            text_overlay = scene.get('text_overlay', '')
            
            if summary:
                print(f"  Summary: {summary}")
            if emotional_beat:
                print(f"  Emotional Beat: {emotional_beat}")
                
            # --- 3.a.0 Reference Search ---
            subject_image_path = None
            if use_internet_image_search and search_query:
                print(f"Scene {scene_num}: Searching internet for reference image: '{search_query}'...")
                res = reference_search_tool_fn(search_query)
                if _is_valid_path(res):
                    subject_image_path = res
                    print(f"  [OK] Reference image downloaded to: {subject_image_path}")
                else:
                    print(f"  [!] Reference search failed or returned no valid image: {res}")
            elif not use_internet_image_search and search_query:
                print(f"Scene {scene_num}: Internet image search disabled. Skipping reference for '{search_query}'.")
            
            # --- 3a. Generate Image Prompt (with retry) ---
            print(f"Scene {scene_num}: Generating image prompt...")
            img_prompt = _retry(
                prompt_tool_fn, description, 
                visual_setup=visual_setup, text_overlay=text_overlay, global_plan=global_plan,
                label=f"Scene {scene_num} image prompt", max_retries=2
            )
            if not img_prompt:
                print(f"  [X] SKIPPING Scene {scene_num}: Image prompt generation failed.")
                return None
            
            # --- 3b. Generate Image (with retry) ---
            print(f"Scene {scene_num}: Generating image...")
            image_path = _retry(
                image_gen_tool_fn, img_prompt,
                reference_image_path=local_prev_image_path,
                subject_reference_image_path=subject_image_path,
                label=f"Scene {scene_num} image gen", max_retries=3, delay=8.0
            )
            
            if not _is_valid_path(image_path):
                print(f"  [X] SKIPPING Scene {scene_num}: Image generation failed — no valid image produced.")
                return None
                
            current_image_path = image_path 

            # --- 3b.2. Veo Video Generation (if enabled) ---
            veo_video_path = None
            if enable_veo:
                veo_prompt = scene.get('veo_prompt', '')
                if not veo_prompt:
                    veo_prompt = description
                print(f"Scene {scene_num}: Generating Veo video with prompt: '{veo_prompt}'...")
                veo_res = generate_video_veo_tool_fn(current_image_path, veo_prompt)
                if _is_valid_path(veo_res):
                    veo_video_path = veo_res
                    print(f"  [OK] Veo video generated: {veo_video_path}")
                else:
                    print(f"  [!] Veo video generation failed: {veo_res}. Continuing without Veo.")

            # --- 3c. SAM Segmentation (non-critical, can fail gracefully) ---
            seg_json_path = None
            if not SAM_API_URL:
                print(f"Scene {scene_num}: [INFO] SAM_API_URL is not configured (empty). Skipping SAM3 segmentation phase. Whiteboard animation will run in single-pass mode.")
            else:
                print(f"Scene {scene_num}: Segmenting image objects...")
                try:
                    seg_json_path = segmentation_tool_fn(current_image_path)
                    if not _is_valid_path(seg_json_path):
                        print(f"  [!] Segmentation returned no valid result. Continuing without segmentation.")
                        seg_json_path = None
                except Exception as e:
                    print(f"  [!] Segmentation failed (non-critical): {e}. Continuing without segmentation.")
            
            # --- 3d. Whiteboard Animation Generation ---
            print(f"Scene {scene_num}: Generating whiteboard animation...")
            anim_video_path = draw_animation_tool_fn(current_image_path, segmentation_results_path=seg_json_path)
            
            if not _is_valid_path(anim_video_path):
                print(f"  [X] SKIPPING Scene {scene_num}: Animation generation failed.")
                return None

            # --- 3d.2. Concatenate Whiteboard Animation and Veo video (if enabled) ---
            combined_video_path = anim_video_path
            if veo_video_path:
                print(f"Scene {scene_num}: Concatenating whiteboard animation and Veo video...")
                combined_output = os.path.join(output_dir, f"scene_{scene_num}_combined_silent.mp4")
                try:
                    filter_complex = (
                        "[0:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25[v0]; "
                        "[1:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25[v1]; "
                        "[v0][v1]concat=n=2:v=1:a=0[v]"
                    )
                    concat_cmd = [
                        "ffmpeg", "-y",
                        "-i", anim_video_path,
                        "-i", veo_video_path,
                        "-filter_complex", filter_complex,
                        "-map", "[v]",
                        "-pix_fmt", "yuv420p",
                        combined_output
                    ]
                    subprocess.run(concat_cmd, capture_output=True, check=True)
                    if os.path.exists(combined_output):
                        combined_video_path = combined_output
                        print(f"  [OK] Successfully concatenated to {combined_video_path}")
                    else:
                        print("  [!] Concat output file not found. Falling back to whiteboard animation only.")
                except Exception as concat_err:
                    print(f"  [!] Concat failed: {concat_err}. Falling back to whiteboard animation only.")
            
            # --- 3e. Narration Refinement (non-critical, can fallback to original) ---
            v_duration = get_video_duration(combined_video_path)
            print(f"Scene {scene_num}: Refining narration (Duration: {v_duration:.1f}s)...")
            try:
                refined = refine_narration_tool_fn(narration, current_image_path, video_duration=v_duration, global_plan=global_plan, language=language)
                if refined and "Error" not in refined:
                    narration = refined
                else:
                    print(f"  [!] Narration refinement returned error. Using Director's original narration.")
            except Exception as e:
                print(f"  [!] Narration refinement failed (non-critical): {e}. Using Director's original narration.")
            
            # --- 3f. TTS Generation (with retry) ---
            print(f"Scene {scene_num}: Generating narration audio...")
            audio_path = _retry(
                generate_tts_audio_tool_fn, narration, language=language,
                label=f"Scene {scene_num} TTS", max_retries=3, delay=5.0
            )
            
            if not _is_valid_path(audio_path):
                print(f"  [X] SKIPPING Scene {scene_num}: TTS generation failed — no audio produced.")
                return None
            
            # --- 3g. Audio-Video Merging ---
            print(f"Scene {scene_num}: Merging audio and video...")
            merged_output = os.path.join(output_dir, f"scene_{scene_num}_merged.mp4")
            merged_video_path = merge_audio_video_tool_fn(combined_video_path, audio_path, merged_output)
            
            if not _is_valid_path(merged_video_path):
                print(f"  [X] SKIPPING Scene {scene_num}: Audio-Video merge failed.")
                return None
            
            # --- 3h. Audio Transcription (non-critical) ---
            print(f"Scene {scene_num}: Transcribing audio for subtitles...")
            subtitles_json_path = None
            try:
                subtitles_json_path = transcribe_audio_tool_fn(merged_video_path)
            except Exception as e:
                print(f"  [!] Transcription failed (non-critical): {e}. Skipping subtitles.")
            
            # --- 3i. Subtitle Burning (non-critical) ---
            final_scene_video = merged_video_path
            if _is_valid_path(subtitles_json_path):
                print(f"Scene {scene_num}: Burning subtitles into video...")
                subtitled_output = os.path.join(output_dir, f"scene_{scene_num}_final.mp4")
                try:
                    final_sv = burn_subtitles_to_video_tool_fn(merged_video_path, subtitles_json_path, subtitled_output)
                    
                    if _is_valid_path(final_sv):
                        final_scene_video = final_sv
                    else:
                        print(f"  [!] Subtitle burning failed. Using merged video without subtitles.")
                except Exception as e:
                    print(f"  [!] Subtitle burning error: {e}. Using merged video without subtitles.")
            else:
                print(f"  [!] No subtitles available. Using merged video as-is.")
            
            # --- Cleanup intermediate files ---
            files_to_delete = []
            if seg_json_path and os.path.exists(seg_json_path):
                files_to_delete.append(seg_json_path)
            if anim_video_path and os.path.exists(anim_video_path):
                files_to_delete.append(anim_video_path)
            if veo_video_path and os.path.exists(veo_video_path):
                files_to_delete.append(veo_video_path)
            if combined_video_path and combined_video_path != anim_video_path and os.path.exists(combined_video_path):
                files_to_delete.append(combined_video_path)
            if audio_path and os.path.exists(audio_path):
                files_to_delete.append(audio_path)
            if merged_video_path and merged_video_path != final_scene_video and os.path.exists(merged_video_path):
                files_to_delete.append(merged_video_path)
            if subtitles_json_path and os.path.exists(subtitles_json_path):
                files_to_delete.append(subtitles_json_path)
                
            for fpath in files_to_delete:
                try:
                    os.remove(fpath)
                    print(f"  Cleaned up intermediate file: {os.path.basename(fpath)}")
                except Exception as cleanup_err:
                    print(f"  Warning: Could not delete intermediate file {os.path.basename(fpath)}: {cleanup_err}")

            print(f"  [OK] Scene {scene_num} completed successfully!")
            return {"scene_num": scene_num, "final_scene_video": final_scene_video, "image_path": current_image_path}
                
        except Exception as e:
            print(f"  [X] UNEXPECTED ERROR in Scene {scene_num}: {e}")
            return None

    # 3. Asset Generation & Processing
    if fast_mode:
        print("\nStep 3: Processing Scenes in Parallel (Fast Mode)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_scene = {
                executor.submit(process_scene_helper, i + 1, scene, None): i + 1
                for i, scene in enumerate(scenes)
            }
            results = []
            for future in concurrent.futures.as_completed(future_to_scene):
                scene_num = future_to_scene[future]
                try:
                    res = future.result()
                    if res:
                        results.append(res)
                    else:
                        failed_scenes.append(scene_num)
                except Exception as e:
                    print(f"  [X] UNEXPECTED ERROR in parallel Scene {scene_num}: {e}")
                    failed_scenes.append(scene_num)
            
            # Reconstruct final videos in correct order
            results.sort(key=lambda x: x["scene_num"])
            for res in results:
                final_videos.append(res["final_scene_video"])
    else:
        print("\nStep 3: Processing Scenes...")
        for i, scene in enumerate(scenes):
            scene_num = i + 1
            res = process_scene_helper(scene_num, scene, prev_image_path)
            if res:
                prev_image_path = res["image_path"]
                final_videos.append(res["final_scene_video"])
            else:
                failed_scenes.append(scene_num)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Scene Processing Summary: {len(final_videos)} succeeded, {len(failed_scenes)} failed")
    if failed_scenes:
        print(f"Failed scenes: {failed_scenes}")

    # 4. Final Merge
    if len(final_videos) >= 2:
        print("\nStep 4: Concatenating all scenes into a final video...")
        final_video_path = os.path.join(output_dir, "storyboard_final_video.mp4")
        result = concatenate_videos_tool_fn(final_videos, final_video_path)
        print(f"\n--- Pipeline Complete! ---")
        print(f"Final Video: {result}")
        return result
    elif len(final_videos) == 1:
        print(f"\n--- Pipeline Complete (Single Scene)! ---")
        print(f"Final Video: {final_videos[0]}")
        return final_videos[0]
    else:
        print("\nPipeline failed: No videos generated.")
        return None

if __name__ == "__main__":
    context = input("Enter the context for your video: ")
    res_choice = input("Select research mode: [1] Deep Research, [2] Web Search (Fast), [3] URL Grounding, [4] Both Web Search & URL Grounding, [5] None (default 2): ").strip()
    
    do_research = False
    do_web_search = False
    enable_google_search = True
    enable_url_context = False
    
    # Check if a URL is in the prompt
    import re
    url_match = re.search(r'(https?://[^\s]+)', context)
    if url_match:
        enable_url_context = True
        print(f"URL detected in input. Enabling URL grounding context: {url_match.group(1)}")
    
    if res_choice == '1':
        do_research = True
    elif res_choice == '3':
        do_web_search = True
        enable_google_search = False
        enable_url_context = True
    elif res_choice == '4':
        do_web_search = True
        enable_google_search = True
        enable_url_context = True
    elif res_choice == '5':
        pass
    else:
        do_web_search = True
        enable_google_search = True
        
    image_search_choice = input("Enable internet image search for references? [Y/n] (default Y): ").strip().lower()
    use_internet_image_search = False if image_search_choice in ['n', 'no'] else True
    
    fast_mode_choice = input("Enable fast mode (parallel generation)? [Y/n] (default N): ").strip().lower()
    fast_mode = True if fast_mode_choice in ['y', 'yes'] else False
    
    language = input("Enter the narration language (default 'english'): ").strip()
    if not language:
        language = "english"
        
    enable_veo_choice = input("Enable Veo video generation? [y/N] (default N): ").strip().lower()
    enable_veo = True if enable_veo_choice in ['y', 'yes'] else False

    veo_direction_by_director = False
    if enable_veo:
        veo_dir_choice = input("Let Director generate Veo visual prompts? [Y/n] (default Y): ").strip().lower()
        veo_direction_by_director = False if veo_dir_choice in ['n', 'no'] else True

    run_pipeline(
        context, 
        do_research=do_research, 
        do_web_search=do_web_search, 
        use_internet_image_search=use_internet_image_search, 
        fast_mode=fast_mode, 
        language=language, 
        enable_veo=enable_veo, 
        veo_direction_by_director=veo_direction_by_director,
        enable_url_context=enable_url_context,
        enable_google_search=enable_google_search
    )


