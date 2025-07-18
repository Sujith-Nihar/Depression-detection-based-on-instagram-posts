import os
import glob
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.generativeai import types
from dotenv import load_dotenv
import PIL.Image
import pandas as pd
import re
import time

# Load API Key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=api_key)

def get_caption(file_path):
    """Find the appropriate caption file for the given image/video."""
    base_name = re.sub(r"(_\d+)?\.(jpg|jpeg|png|mp4)$", "", file_path)  # Handle both image and video formats
    caption_path = base_name + ".txt"
    if os.path.exists(caption_path):
        with open(caption_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    return ""

def analyze_media(media_paths, excel_path):
    results = []
    if os.path.exists(excel_path):
        df_existing = pd.read_excel(excel_path)
        processed_media = set(df_existing["Media Name"])
    else:
        df_existing = None
        processed_media = set()
    
    model = genai.GenerativeModel("gemini-1.5-pro")
    
    for media_path in media_paths:
        try:
            media_name = os.path.basename(media_path)
            folder_name = os.path.basename(os.path.dirname(media_path))

            if media_name in processed_media:
                print(f"Skipping {media_name}, already processed.")
                continue

            # Check if it's an image or video and process accordingly
            if media_path.endswith(('.jpg', '.jpeg', '.png')):
                media = PIL.Image.open(media_path)
                media_type = 'image'
            elif media_path.endswith('.mp4'):
                size = os.path.getsize(media_path) 
                file_size_mb = size / (1024 * 1024)
                with open(media_path, "rb") as video_file:
                    video_data = video_file.read()
                if file_size_mb > 10:    
                    media_type = 'large_video'
                else:
                    media_type = 'video'
            else:
                print(f"Skipping unsupported file type: {media_path}")
                continue

            caption = get_caption(media_path)
            prompt = f"""

                "Analyze the following image or video post and provide responses in the structured format below:
                Embedded Text: Extract any text visibly present within the image or video. If no text is present, respond with None.

                Assume you are an AI emotion specialist. Based on your analysis, evaluate the emotional expression from two distinct sources:

                1. **Visual Content Analysis**: This includes the image or video and any embedded text found within it.
                2. **Caption Analysis**: This includes only the user-provided caption text: {caption}

                For each source, provide an Emotion Score for the following six basic emotions. Each score should be between **0.00 to 1.00**, indicating the strength of that emotion present. Return only the scores — no explanation.

                Emotion Scores (Visual Content: image/video + embedded text):
                - Visual Happiness:
                - Visual Sadness:
                - Visual Fear:
                - Visual Disgust:
                - Visual Anger:
                - Visual Surprise:

                Emotion Scores (Caption Only):
                - Caption Happiness:
                - Caption Sadness:
                - Caption Fear:
                - Caption Disgust:
                - Caption Anger:
                - Caption Surprise:

                """



            safety_settings_instagram =  [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE",
                    },
            ]



            if media_type == 'image':
                response = model.generate_content([media, prompt],
                                                  generation_config=genai.types.GenerationConfig(temperature=0.1), safety_settings=safety_settings_instagram)
            elif media_type == 'video':
                # For video, we'll send the video content as part of the request.
                video_part = {
                    "mime_type": "video/mp4",
                    "data": video_data,
                }
                response = model.generate_content([prompt, video_part],
                                                  generation_config=genai.types.GenerationConfig(temperature=0.1))
                
            elif media_type == 'large_video': # For large videos
                try:
                    uploaded_file = genai.upload_file(media_path)
                    while uploaded_file.state.name == "PROCESSING":
                        print('Waiting for video to be processed.')
                        time.sleep(10)
                        uploaded_file = genai.get_file(uploaded_file.name)
                    if uploaded_file.state.name == "FAILED":
                        raise ValueError(uploaded_file.state.name)
                    print(f'Video processing complete: ' + uploaded_file.uri)
                
                    response = model.generate_content([prompt, uploaded_file])
                except FileNotFoundError:
                    print(f"Error: Video file not found at {media_path}")
                except Exception as e:
                    print(f"An error occurred: {e}")
            analysis = response.text.strip().split('\n')
            
            analysis_dict = {"Media Name": media_name, "Profile Name": folder_name, "Media Caption": caption}
            for line in analysis:
                if ":" in line:
                    try:
                        key, value = line.split(":", 1)
                        cleaned_key = re.sub(r"[*_]", "", key.strip())
                        cleaned_value = re.sub(r"[*_]", "", value.strip())
                        if cleaned_key and cleaned_value:
                            analysis_dict[cleaned_key] = cleaned_value
                    except ValueError:
                        continue
            
            results.append(analysis_dict)
            # genai.delete_file(uploaded_file.name)
            # print("File deleted successfully.")
        except Exception as e:
            print(f"Error processing {media_path}: {e}")
    
    if results:
        df_new = pd.DataFrame(results)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True) if df_existing is not None else df_new
        df_combined.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"Analysis saved to {excel_path}")
    else:
        print("No new results to save.")

# Folder containing media (images and videos)
media_folder = "/Users/sujiththota/Downloads/Python/Research/sad_images/"
media_paths = glob.glob(os.path.join(media_folder, "*.jpg")) + \
              glob.glob(os.path.join(media_folder, "*.jpeg")) + \
              glob.glob(os.path.join(media_folder, "*.png")) + \
              glob.glob(os.path.join(media_folder, "*.mp4"))

excel_path = "/Users/sujiththota/Downloads/Python/Research/sad_images/emotion_scores_custom_individual.xlsx"
analyze_media(media_paths, excel_path)
