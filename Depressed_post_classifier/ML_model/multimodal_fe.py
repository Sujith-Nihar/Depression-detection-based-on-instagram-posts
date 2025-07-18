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
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    
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

            "Analyze the following image/video and provide responses in a structured format:
            Format the response so it can be directly parsed and saved into an Excel file.
            Simple Description: Briefly describe the main content of the image/video.(give in one line)
            Embedded Text: Extract any text present in the image/video strictly text should be found in the image/video not anywhere else if nothing present give None.
            answer everything below in single word as given dont give two words single word response is needed
            Brightness: Specify as Low, Medium, or High.
            Brightness Value: Value of the brightness in numerical
            Saturation: Specify as Low, Medium, or High
            Saturation Value: value of the saturation in numerical
            Hue distribution: Cool or Warm
            Hue value: Value of the Hue in numerical
            Contrast levels: Low, Medium or High (in one word) 
            Post sentiment: consider image/video, embedded text, caption overall analyse the sentiment of the post. Sentiment is is based on emotions (Happiness, Sadness, Fear, Disgust, Anger, Surprise) Sadness, Fear, Disgust, Anger are negative sentiment and happy, surprise are postive. Strictly response can be either Positive or negative (Do not mention any detail)
            Analyze the image/video and text embedded in the image/video and the caption: {caption}. determine to what extent does the given image/video, embedded text and caption exhibit the following:
            Provide a single Emotion Score for each of the following six basic emotions. Each score should be between **0.00 to 1.00**, indicating the strength of that emotion present in the combined media (image/video + embedded text + caption). Only return the scores — no additional explanation.
            Happiness: 
            Sadness: 
            Fear: 
            Disgust: 
            Anger: 
            Surprise: 
            Loss of Interest Binary: reflects little interest or pleasure in doing things. A decline in interest or pleasure in the majority or all normal activities. Your response can be No if the feature is not present in the Image/Video or caption,Yes if the feature is present in the Image/Video or caption, Response should be strictly the words mentioned no special mentions or brackets
            Feeling depressed Binary: reflects feeling depressed, down, sadness, tearfulness, emptiness, or hopelessness.Your response can be No if the feature is not present in the Image/Video or caption,Yes if the feature is present in the Image/Video or caption. Response should be strictly the words mentioned no special mentions or brackets
            Sleeping Disorder Binary: reflects difficulties in sleep, including insomnia, trouble falling asleep, staying asleep, or excessive sleeping. Your response can be No if the feature is not present in the Image/Video or caption,Yes if the feature is present in the Image/Video or caption, Response should be strictly the words mentioned no special mentions or brackets
            Lack of Energy Binary: reflects fatigue, exhaustion, or low energy levels.Your response can be No if the feature is not present in the Image/Video or caption,Yes if the feature is present in the Image/Video or caption. Response should be strictly the words mentioned no special mentions or brackets
            Eating Disorder Binary: reflects changes in eating habits, such as poor appetite or overeating. Your response can be No if the feature is not present in the Image/Video or caption,Yes if the feature is present in the Image/Video or caption. Response should be strictly the words mentioned no special mentions or brackets
            Low Self-Esteem Binary: reflects feelings of self-doubt, failure, or disappointment worthlessness or guilt, fixating on past failures or self-blame in oneself.Your response can be No if the feature is not present in the Image/Video or caption,Yes if the feature is present in the Image/Video or caption. Response should be strictly the words mentioned no special mentions or brackets
            Concentration difficulty Binary: reflects difficulty in concentrating on any tasks, or any other activity difficulty in making decisions and remembering things.Your response can be No if the feature is not present in the Image/Video or caption,Yes if the feature is present in the Image/Video or caption. Response should be strictly the words mentioned no special mentions or brackets
            Psychomotor changes Binary: reflects psychomotor changes, such as noticeable slowing of movement or speech, or increased restlessness and fidgeting.Your response can be No if the feature is not present in the Image/Video or caption,Yes if the feature is present in the Image/Video or caption. Response should be strictly the words mentioned no special mentions or brackets
            Self harm risk Binary:reflects thoughts of self-harm or suicidal ideation or suicide attempts. Your response can be No if the feature is not present in the Image/Video/video or caption,Yes if the feature is present in the Image/Video or caption. Response should be strictly the words mentioned no special mentions or brackets
            "

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
            
            analysis_dict = {"Media Name": media_name, "Profile Name": folder_name}
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
media_folder = "/Users/sujiththota/Downloads/Python/Research/oliviaabigailxo_data/"
media_paths = glob.glob(os.path.join(media_folder, "*.jpg")) + \
              glob.glob(os.path.join(media_folder, "*.jpeg")) + \
              glob.glob(os.path.join(media_folder, "*.png")) + \
              glob.glob(os.path.join(media_folder, "*.mp4"))

excel_path = "/Users/sujiththota/Downloads/Python/Research/ML_DATA/DATA13.xlsx"
analyze_media(media_paths, excel_path)
