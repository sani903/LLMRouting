import pandas as pd
import numpy as np
import json
from sklearn.cluster import KMeans

# Load the data
df = pd.read_csv('mistral_llama.csv')
with open('unique_prompts.json', 'r') as f:
    unique_prompts = json.load(f)
embeddings = np.load('prompt_embeddings.npy')

# Create the labels column
df['label'] = np.where(df['strong'] > df['weak'], 1, 0)

# Create prompt_to_embedding dictionary
prompt_to_embedding = {prompt: embedding for prompt, embedding in zip(unique_prompts, embeddings)}

# Get embeddings for clustering
X = np.array([prompt_to_embedding[prompt] for prompt in df['prompts'] if prompt in prompt_to_embedding])

# Keep only rows with valid embeddings
df = df[df['prompts'].isin(prompt_to_embedding)]

# Function to cluster and calculate purity
def cluster_and_get_purity(n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    purities = df.groupby(clusters)['label'].apply(lambda x: max(x.mean(), 1 - x.mean()))
    return clusters, purities

# Try different numbers of clusters
best_n_clusters = 0
best_purity = 0
best_clusters = None

for n_clusters in range(501, 1001, 25):  # Try from 50 to 200 clusters in steps of 10
    clusters, purities = cluster_and_get_purity(n_clusters)
    avg_purity = purities.mean()
    max_purity = purities.max()
    
    print(f"Clusters: {n_clusters}, Avg Purity: {avg_purity:.4f}, Max Purity: {max_purity:.4f}")
    
    if max_purity > best_purity:
        best_purity = max_purity
        best_n_clusters = n_clusters
        best_clusters = clusters

# Assign the best clustering to the DataFrame
df['cluster'] = best_clusters

# Calculate and print cluster statistics
cluster_stats = df.groupby('cluster').agg({
    'label': ['mean', 'count'],
    'prompts': lambda x: x.iloc[0]  # Example prompt
})
cluster_stats.columns = ['label_mean', 'count', 'example_prompt']
cluster_stats['dominant_label'] = np.where(cluster_stats['label_mean'] > 0.5, 1, 0)
cluster_stats['purity'] = np.maximum(cluster_stats['label_mean'], 1 - cluster_stats['label_mean'])

# Sort clusters by purity in descending order
cluster_stats = cluster_stats.sort_values('purity', ascending=False)

print("\nTop 10 clusters by purity:")
print(cluster_stats.head(10))

# Save results
df.to_csv('clustered_data.csv', index=False)
cluster_stats.to_csv('cluster_statistics.csv')
