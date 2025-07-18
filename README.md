# Multimodal Instagram Depression Analysis

This project analyzes Instagram posts to detect and understand depression and mood progression in users using multimodal features and LLMs. It combines image, video, and textual data with emotion and PHQ-9 scores for rich, explainable analysis.

---

## 📁 Project Structure

### 🔗 Datasets/
Contains download links and information for:
- **Depressed users’ posts**
- **Happy/non-depressed users’ posts**

---

### 🤖 Depressed_post_classifier/

This module identifies whether an individual Instagram post (not the user) is depressed or non-depressed based on its content.

Includes:
- Dataset preparation and preprocessing
- Feature extraction (from text, image, etc.)
- Model building and evaluation
- Deployment: [Streamlit Dashboard](#) *(https://depression-detection-project-dashboard-5uiay23wnhah5jfkecboyz.streamlit.app)*

---

### 📈 Mood_progression/
Tracks mood progression of users over time using:
- Emotionally intense posts filtered via PHQ-9
- Mood summaries based on users emotion scores, posts (image/video and captions) generated using LLMs
- Results of mood progression results can be found on [Streamlit Dashboard](#) *(http://patient-medical-records.vercel.app)*

---

### 🧠 PHQ_9_progression/
Similar to mood progression, but tracks **PHQ-9 intensity scores** over time per user, filtered from emotional posts and analyzed with LLM support.

---

### 😐 emotion_scores_from_LLM/
Performs emotion scores extraction for:
- **Individual modalities**: captions (text), images, and videos
- **Dynamic fusion** of modalities for final emotion scores
- Visualizations and comparative results for depressed vs. happy users

---

### 🎭 multimodal_analysis/
Aggregates and analyzes all multimodal features of Instagram posts:
- Image, video, caption, embedded text
- Sentiment, PHQ-9 scores, colors, hue, saturation, etc.
- Enables rich analysis for depression detection and progression

---

## 📄 project_report.pdf
Contains the final detailed report with all implementation steps, evaluations, and insights.

---

## 📑 README.md
You’re here!

---

## 📌 Notes
- This project uses LLMs and computer vision tools for mental health analysis.
- Ethical considerations and privacy guidelines are followed while working with social media data.


## 👨‍💻 Project Authors

### [Dr. Ranganathan Chandrasekaran](mailto:ranga@uic.edu)  
[![Ranganathan Chandrasekaran](https://business.uic.edu/wp-content/uploads/sites/91/2018/01/Chandrasekaran_Ranganathan_2020-157x180.jpg)](mailto:ranga@uic.edu)  
**Professor**  
Department of Information and Decision Sciences  
University of Illinois Chicago  
📧 ranga@uic.edu  

---

### [Dr. Negar Soheili](mailto:nazad@uic.edu)  
[![Negar Soheili](https://business.uic.edu/wp-content/uploads/sites/91/2018/02/SoheiliNegar_2018.jpg)](mailto:nazad@uic.edu)  
**Associate Professor**  
Department of Information and Decision Sciences  
University of Illinois Chicago  
📧 nazad@uic.edu  

---

### Sujith Thota  
**Master’s Student**  
Department of Computer Science  
University of Illinois Chicago  
📧 sthot10@uic.edu
