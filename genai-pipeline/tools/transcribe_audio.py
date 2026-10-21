"""
Transcribe Audio Tool - Extracts subtitles from video/audio using Gemini.
Returns timestamped transcription in JSON format for subtitle generation.
"""
import os
import json
import time
from google.genai import types
from config import MODEL_NAME, get_generation_config
from . import utils


def _parse_transcription_response(response_text: str) -> list:
    """
    Parse the Gemini response to extract timestamped subtitles.
    Expected format from Gemini: JSON array with start, end, text fields.
    """
    try:
        # Try to find JSON in the response (handle markdown code blocks)
        text = response_text.strip()
        
        # Remove markdown code block if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Find the content between ``` markers
            start_idx = 1 if lines[0].startswith("```") else 0
            end_idx = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == "```":
                    end_idx = i
                    break
            text = "\n".join(lines[start_idx:end_idx])
        
        # Parse JSON
        data = json.loads(text)
        
        # Handle both direct array and object with "subtitles" key
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "subtitles" in data:
            return data["subtitles"]
        else:
            print(f"Unexpected response format: {type(data)}")
            return []
            
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        print(f"Response text: {response_text[:500]}...")
        return []


def transcribe_audio_tool_fn(file_path: str, output_path: str = None) -> str:
    """
    Extracts subtitles from video or audio file using Gemini's audio understanding.
    Directly uploads the file to Gemini.
    
    Args:
        file_path: Path to the input video or audio file.
        output_path: Optional path for the output JSON file. If not provided,
                    saves with '_subtitles.json' suffix in output directory.
    
    Returns:
        Path to the JSON file containing subtitles if successful, or error message.
        
    JSON Output Format:
        {
            "subtitles": [
                {"start": 0.0, "end": 2.5, "text": "Hello world"},
                {"start": 2.5, "end": 5.0, "text": "This is a test"}
            ]
        }
    """
    if not utils.client:
        return "Error: GEMINI_API_KEY not configured."
    
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    
    print(f"Transcribing file: {file_path}")
    
    try:
        # Step 1: Upload file to Gemini
        print("Step 1: Uploading file to Gemini...")
        # mime_type is automatically inferred by the client or can be passed if known.
        # For simplicity, we let the client/API handle inference or default behavior.
        uploaded_file = utils.client.files.upload(file=file_path)
        print(f"Uploaded file: {uploaded_file.name} (URI: {uploaded_file.uri})")
        
        # Wait for the file to be processed and become ACTIVE
        print(f"Waiting for file {uploaded_file.name} to become ACTIVE...")
        while True:
            file_info = utils.client.files.get(name=uploaded_file.name)
            if file_info.state.name == "ACTIVE":
                print(f"File {uploaded_file.name} is now ACTIVE.")
                break
            elif file_info.state.name == "FAILED":
                return f"Error: File processing failed in Gemini: {file_info.error}"
            
            print(f"Current state: {file_info.state.name}. Waiting 2 seconds...")
            time.sleep(2)
        
        # Step 2: Request transcription with timestamps
        print("Step 2: Requesting transcription with timestamps...")
        
        transcription_prompt = """
Please transcribe the audio in this file and provide timestamps for each segment.

Return ONLY a valid JSON object in this exact format (no additional text):
{
    "subtitles": [
        {"start": 0.0, "end": 2.5, "text": "First sentence or phrase"},
        {"start": 2.5, "end": 5.0, "text": "Next sentence or phrase"}
    ]
}

Rules:
- "start" and "end" are times in seconds (float)
- Each subtitle segment should be a natural phrase or sentence (not too long)
- Segments should be 2-5 seconds long ideally for readability
- Ensure timestamps are accurate and sequential
- Identify different speakers if possible, but mainly focus on the spoken text.
- Return ONLY the JSON, no explanations or markdown
"""
        
        response = utils.client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_uri(
                    file_uri=uploaded_file.uri,
                    mime_type=uploaded_file.mime_type
                ),
                transcription_prompt
            ],
            config=get_generation_config()
        )
        
        # Step 3: Parse the response
        print("Step 3: Parsing transcription response...")
        response_text = response.text
        subtitles = _parse_transcription_response(response_text)
        
        if not subtitles:
            # Fallback: try to get raw transcription
            print("Warning: Could not parse structured response. Saving raw response.")
            result_data = {
                "subtitles": [],
                "raw_response": response_text
            }
        else:
            result_data = {"subtitles": subtitles}
            print(f"Extracted {len(subtitles)} subtitle segments.")
        
        # Step 4: Save JSON output
        if not output_path:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            filename = f"{base_name}_subtitles.json"
            if utils.GLOBAL_OUTPUT_DIR:
                output_path = os.path.join(utils.GLOBAL_OUTPUT_DIR, filename)
            else:
                output_path = os.path.join(os.path.dirname(file_path), filename)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        print(f"Subtitles saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        return f"Error during transcription: {str(e)}"
