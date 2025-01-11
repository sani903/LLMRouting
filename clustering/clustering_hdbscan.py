import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_distances


# Load data
with open('unique_prompts.json', 'r') as f:
    unique_prompts = json.load(f)

# Load embeddings
embeddings = np.load('prompt_embeddings.npy')

# Load the mistral_llama CSV
csv_data = pd.read_csv('mistral_llama.csv')

# Create a dictionary mapping each prompt to its embedding
prompt_to_embedding = {prompt: embeddings[i] for i, prompt in enumerate(unique_prompts)}

# Filter prompts and get embeddings
filtered_data = []
seen_prompts = set()
for _, row in csv_data.iterrows():
    prompt = row['prompts']
    if prompt in prompt_to_embedding and prompt not in seen_prompts:
        seen_prompts.add(prompt)
        filtered_data.append({
            'prompt': prompt,
            'strong': row['strong'],
            'weak': row['weak'],
            'embedding': prompt_to_embedding[prompt]
        })
filtered_df = pd.DataFrame(filtered_data)

# Normalize embeddings
scaler = StandardScaler()
normalized_embeddings = scaler.fit_transform(np.array(filtered_df['embedding'].tolist()))

# # Dimensionality reduction using UMAP
umap_model = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=5, random_state=42)
reduced_embeddings = umap_model.fit_transform(normalized_embeddings)
# # cosine_distance_matrix = cosine_distances(reduced_embeddings)
# # Perform clustering with HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=10, metric='euclidean', cluster_selection_epsilon=0.0)
filtered_df['cluster'] = clusterer.fit_predict(reduced_embeddings)
# reduced_embeddings = normalized_embeddings
# 
# Assign noise points to the nearest cluster
cluster_centroids = {
    cluster: np.mean(reduced_embeddings[filtered_df['cluster'] == cluster], axis=0)
    for cluster in set(filtered_df['cluster']) if cluster != -1
}

noise_indices = filtered_df[filtered_df['cluster'] == -1].index
for idx in noise_indices:
    embedding = reduced_embeddings[idx]
    distances = {cluster: np.linalg.norm(embedding - centroid) for cluster, centroid in cluster_centroids.items()}
    closest_cluster = min(distances, key=distances.get)
    filtered_df.at[idx, 'cluster'] = closest_cluster

# Evaluation
silhouette_avg = silhouette_score(reduced_embeddings, filtered_df['cluster'])
db_score = davies_bouldin_score(reduced_embeddings, filtered_df['cluster'])
print(f"Silhouette Score: {silhouette_avg}")
print(f"Davies-Bouldin Index: {db_score}")

# Visualization
plt.figure(figsize=(10, 8))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=filtered_df['cluster'], cmap='tab10', s=5)
plt.colorbar(label='Cluster')
plt.title('UMAP Projection of Prompt Embeddings')
plt.show()

# Save results
filtered_df[['prompt', 'strong', 'weak', 'cluster']].to_csv('filtered_prompts_with_clusters.csv', index=False)
import matplotlib.pyplot as plt

plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=filtered_df['cluster'], cmap='Spectral', s=5)
plt.colorbar()
plt.title('Cluster/ Visualization')
plt.show()
