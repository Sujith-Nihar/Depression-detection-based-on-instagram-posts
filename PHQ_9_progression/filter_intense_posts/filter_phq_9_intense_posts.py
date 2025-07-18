import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


df = pd.read_excel("gemini_multimodal_phq9_similarity_scores.xlsx")


phq_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in [
    "loss of interest", "feeling depressed", "sleeping disorder", "eating disorder",
    "low self-esteem", "concentration difficulty", "psychomotor changes", "self harm risk"
])]


caption_cols = [col for col in phq_columns if "caption" in col.lower()]
image_cols = [col for col in phq_columns if "image" in col.lower() or "video" in col.lower()]

# Compute per-post averages
caption_mean = df[caption_cols].mean(axis=1)
image_mean = df[image_cols].mean(axis=1)
epsilon = 1e-8  # To avoid division by zero

# Compute adaptive weights
total_mean = caption_mean + image_mean + epsilon
caption_weight = caption_mean / total_mean
image_weight = image_mean / total_mean

# Apply weights
phq_df_weighted = df[phq_columns].copy()
phq_df_weighted[caption_cols] = phq_df_weighted[caption_cols].multiply(caption_weight, axis=0)
phq_df_weighted[image_cols] = phq_df_weighted[image_cols].multiply(image_weight, axis=0)

# Standardize the weighted data
scaler = StandardScaler()
phq_scaled = scaler.fit_transform(phq_df_weighted)

# KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(phq_scaled)

# Determine emotionally intense cluster
cluster_centers = kmeans.cluster_centers_.mean(axis=1)
intense_cluster = cluster_centers.argmax()
df['Emotional Intensity'] = df['Cluster'].apply(lambda x: 'Emotionally Intense' if x == intense_cluster else 'Neutral')

# Calculate Silhouette Score
sil_score = silhouette_score(phq_scaled, df['Cluster'])
print("Silhouette Score:", round(sil_score, 3))

# PCA for visualization
pca = PCA(n_components=2)
phq_2d = pca.fit_transform(phq_scaled)
df['PCA1'] = phq_2d[:, 0]
df['PCA2'] = phq_2d[:, 1]

# Save to Excel
df.to_excel("weighted_emotional_intensity_clusters.xlsx", index=False)

# Plot
plt.figure(figsize=(10, 6))
for label in df['Emotional Intensity'].unique():
    subset = df[df['Emotional Intensity'] == label]
    plt.scatter(subset['PCA1'], subset['PCA2'], label=label, alpha=0.7)

plt.title('Adaptive Weighted Emotional Intensity Clusters (PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
