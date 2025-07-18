import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Load the Excel file
file_path = "/Users/sujiththota/Downloads/Python/Research/happy_user_results/emotion_scores_happy_user_benny.engel_.xlsx"
df = pd.read_excel(file_path)

# Clean column names
df.rename(columns=lambda x: x.strip().replace('-', '').strip(), inplace=True)

# Extract timestamp from filename
def extract_timestamp(filename):
    try:
        date_time_part = filename.split("_UTC")[0]
        date, time = date_time_part.split("_")
        time = time.replace("-", ":")
        return datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(f"Error processing filename '{filename}': {e}")
        return None

# Apply timestamp extraction
df['Timestamp'] = df['Media Name'].apply(extract_timestamp)
df = df.dropna(subset=['Timestamp'])
df_sorted = df.sort_values(by='Timestamp')
df_sorted.set_index('Timestamp', inplace=True)

# Emotion columns
emotion_columns = ['Happiness', 'Sadness', 'Fear', 'Disgust', 'Anger', 'Surprise']

# Clean and convert emotion columns to numeric
for col in emotion_columns:
    df_sorted[col] = pd.to_numeric(df_sorted[col], errors='coerce')

# Drop rows where all emotion scores are NaN
df_sorted.dropna(subset=emotion_columns, how='all', inplace=True)

### 1. Naive Rolling Average Plot ###
rolling_avg = df_sorted[emotion_columns].rolling(window=5, min_periods=1).mean()

# plt.figure(figsize=(12, 6))
# for emotion in emotion_columns:
#     if emotion in rolling_avg.columns:
#         plt.plot(rolling_avg.index, rolling_avg[emotion], label=emotion, linewidth=2)

# plt.title("Smoothed Emotion Trends Over Time (5-Post Rolling Average)")
# plt.xlabel("Time")
# plt.ylabel("Emotion Score")
# plt.legend()
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

plt.figure(figsize=(12, 6))
for emotion in emotion_columns:
    plt.plot(rolling_avg.index, rolling_avg[emotion], label=emotion, linewidth=2)

# diagnosis_date = pd.to_datetime("2021-12-01")
# Add diagnosis date line
# plt.axvline(diagnosis_date, color='purple', linestyle='--', linewidth=2, label='Diagnosis Date')
# plt.text(diagnosis_date, plt.ylim()[1]*1, 'Diagnosis Date', rotation=90, color='purple', fontsize=10, verticalalignment='top')

plt.title("Smoothed Emotion Trends Over Time (5-Post Rolling Average) ")
plt.xlabel("Time")
plt.ylabel("Emotion Score")
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

### 2. Weekly Emotion Intensity Heatmap ###
weekly_avg = df_sorted[emotion_columns].resample('W').mean()

plt.figure(figsize=(12, 5))
sns.heatmap(weekly_avg.T, cmap='coolwarm', annot=False, fmt=".2f")
plt.title("Weekly Emotion Intensity Heatmap ")
plt.xlabel("Week")
plt.ylabel("Emotion")
plt.tight_layout()
plt.show()

### 3. Raw Emotion Trend Plot (Optional) ###
plt.figure(figsize=(12, 6))
for emotion in emotion_columns:
    if emotion in df_sorted.columns:
        plt.plot(df_sorted.index, df_sorted[emotion], label=emotion, linestyle='--', alpha=0.5)

plt.title("Raw Emotion Trends Over Time (No Smoothing) ")
plt.xlabel("Time")
plt.ylabel("Emotion Score")
# plt.axvline(diagnosis_date, color='purple', linestyle='--', linewidth=2, label='Diagnosis Date')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

### 4. Dominant Emotion Plot ###
df_sorted['Dominant Emotion'] = df_sorted[emotion_columns].idxmax(axis=1)

plt.figure(figsize=(10, 4))
df_sorted['Dominant Emotion'].value_counts().plot(kind='bar', color='skyblue')
plt.title("Most Frequent Dominant Emotions")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()

### 5. Positive vs Negative Emotion Trend ###
df_sorted['Positive'] = df_sorted['Happiness'] + df_sorted['Surprise']
df_sorted['Negative'] = df_sorted['Sadness'] + df_sorted['Fear'] + df_sorted['Anger'] + df_sorted['Disgust']

plt.figure(figsize=(12, 5))
plt.plot(df_sorted.index, df_sorted['Positive'].rolling(5, min_periods=1).mean(), label='Positive', color='green')
plt.plot(df_sorted.index, df_sorted['Negative'].rolling(5, min_periods=1).mean(), label='Negative', color='red')
# plt.axvline(diagnosis_date, color='purple', linestyle='--', linewidth=2, label='Diagnosis Date')
plt.title("Positive vs Negative Emotion Trend (5-Post Rolling Avg) ")
plt.xlabel("Time")
plt.ylabel("Aggregated Score")
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

### 6. Emotion Volatility (Standard Deviation) ###
emotion_std = df_sorted[emotion_columns].rolling(window=5, min_periods=1).std()
plt.figure(figsize=(12, 5))
emotion_std.plot(ax=plt.gca())
plt.title("Emotion Volatility Over Time (Standard Deviation) ")
plt.xlabel("Time")
plt.ylabel("Volatility")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend(loc='upper right')
plt.show()

### 7. Radar Chart for Average Emotion Profile ###
avg_scores = df_sorted[emotion_columns].mean()
angles = np.linspace(0, 2 * np.pi, len(emotion_columns), endpoint=False).tolist()
scores = avg_scores.tolist()
scores += scores[:1]
angles += angles[:1]

plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
ax.plot(angles, scores, linewidth=2)
ax.fill(angles, scores, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(emotion_columns)
plt.title("Average Emotion Profile (Radar Chart)")
plt.tight_layout()
plt.show()
