import json
import pandas as pd
import random
import numpy as np
from litellm import embedding

# Paths to your existing data and new data
existing_data_path = "mf_data.json"
new_data_path = "re.csv"
output_data_path = "mf_data_gsm.json"

# Load the existing data
with open(existing_data_path, "r") as f:
    existing_data = json.load(f)


# Load the new data from CSV
new_data = pd.read_csv(new_data_path)

# Convert new data to the required format
processed_new_data = []
for idx, row in new_data.iterrows():
    strong = row["strong"]
    weak = row["weak"]
    
    # Skip rows where the absolute difference is less than 0.1
    if abs(strong - weak) < 0.1:
        continue
    
    # Create the label (1 if strong < weak, 0 otherwise)
    label = 1 if strong < weak else 0
    
    # Assign models based on the label
    model_a = "mistral-7b-v0.3" if label == 1 else "llama-3.2-3b"
    model_b = "llama-3.2-3b" if label == 1 else "mistral-7b-v0.3"

    processed_new_data.append({
        "prompts": row["question"],
        "model_a": model_a,
        "model_b": model_b,
        "response_a": "",  # Placeholder (not used in training)
        "response_b": "",  # Placeholder (not used in training)
        "winner": "model_a" if label == 1 else "model_b",
        "idx": len(existing_data) + idx,  # Ensure unique indices
    })

# Merge datasets
combined_data = existing_data + processed_new_data

# Save combined data
with open(output_data_path, "w") as f:
    json.dump(combined_data, f, indent=4)

print(f"Processed data saved to {output_data_path}")

# Load existing embeddings and prompts
existing_embeddings = np.load("prompt_embeddings.npy")
with open("unique_prompts.json", "r") as f:
    existing_prompts = json.load(f)

# Load new data
with open("mf_data_re.json", "r") as f:
    new_data = json.load(f)

# Extract all prompts from new data
all_prompts = set(item["prompts"] for item in new_data)

# Identify new prompts
new_prompts = all_prompts - set(existing_prompts)

# Generate embeddings for new prompts
new_embeddings = []
counter = 0
p = []
for prompt in new_prompts:
    counter+=1
    if counter%100 == 0:
        print(counter)
    try:
        response = embedding(
            model="openai/text-embedding-3-small",
            input=[prompt],
            api_key=api_key,
            api_base=api_base,
        )
        new_embeddings.append(response['data'][0]['embedding'])
        p.append(prompt)
    except Exception as e:
        print(f"Error generating embedding for prompt: {prompt}")
        print(f"Error: {e}")
        continue

# Combine old and new embeddings and prompts
combined_embeddings = np.vstack((existing_embeddings, np.array(new_embeddings)))
combined_prompts = existing_prompts + p

# Save updated embeddings and prompts
np.save("combined_embeddings.npy", combined_embeddings)
with open("combined_prompts.json", "w") as f:
    json.dump(combined_prompts, f)

print(f"Combined embeddings saved to combined_embeddings.npy")
print(f"Combined prompts saved to combined_prompts.json")
