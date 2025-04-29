# LLM based Depression-detection-based-on-instagram-posts

📁 Project Modules

multimodal_analysis/multimodal_analysis.py
This module extracts all relevant features from Instagram-style posts, including:

Image/video, embedded text, caption features, activities, sentiment PHQ-9 features
It performs a comprehensive analysis by leveraging image/video, audio, and text modalities which they posted on instagram to assess user mental health.



MOOD/get_mood_summary_from_LLM.py
This script uses a large language model to analyze the mood trajectory of a given profile over time.
It processes:

Images and videos
Embedded text (e.g., text in memes or screenshots)
Captions
It then generates a time-series summary of emotional states across user posts.


PHQ/get_phq9_summary_from_LLM.py
This script estimates PHQ-9 scores progression over time for a specific profile, again using multimodal inputs:

Image/video features
Embedded/in-post text
Captions
It provides a timeline-based view of possible depression indicators based on post content.


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
