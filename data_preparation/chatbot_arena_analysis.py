import pandas as pd
from collections import defaultdict

# You will need to have the dataset downloaded or accessible via the `load_dataset` method
# pip install datasets
from datasets import load_dataset

import pandas as pd
from collections import defaultdict
from huggingface_hub import HfApi, RepositoryNotFoundError, HTTPError

# Define the tier mapping
arena_model_tiers = {
    "Tier 0": ["gpt-4-0125-preview", "gpt-4-1106-preview"],
    "Tier 1": ["gpt-4-0314", "gpt-4-0613", "mistral-medium", "claude-1", "qwen1.5-72b-chat"],
    "Tier 2": [
        "claude-2.0", "mixtral-8x7b-instruct-v0.1", "claude-2.1", "gemini-pro-dev-api",
        "gpt-3.5-turbo-0314", "gpt-3.5-turbo-0613", "gemini-pro", "gpt-3.5-turbo-0125",
        "claude-instant-1", "yi-34b-chat", "starling-lm-7b-alpha", "wizardlm-70b",
        "vicuna-33b", "tulu-2-dpo-70b", "nous-hermes-2-mixtral-8x7b-dpo", "llama-2-70b-chat",
        "openchat-3.5"
    ],
    "Tier 3": [
        "llama2-70b-steerlm-chat", "pplx-70b-online", "dolphin-2.2.1-mistral-7b",
        "gpt-3.5-turbo-1106", "deepseek-llm-67b-chat", "openhermes-2.5-mistral-7b",
        "openchat-3.5-0106", "wizardlm-13b", "mistral-7b-instruct-v0.2",
        "solar-10.7b-instruct-v1.0", "zephyr-7b-beta", "zephyr-7b-alpha",
        "codellama-34b-instruct", "mpt-30b-chat", "llama-2-13b-chat", "vicuna-13b",
        "qwen1.5-7b-chat", "pplx-7b-online", "falcon-180b-chat", "llama-2-7b-chat",
        "guanaco-33b", "qwen-14b-chat"
    ],
    "Tier 4": ["stripedhyena-nous-7b", "mistral-7b-instruct", "vicuna-7b", "qwen1.5-4b-chat", "palm-2"],
    "Tier 5": ["koala-13b", "chatglm3-6b", "gpt4all-13b-snoozy"],
    "Tier 6": ["mpt-7b-chat", "RWKV-4-Raven-14B", "chatglm2-6b", "alpaca-13b", "oasst-pythia-12b"],
    "Tier 7": ["fastchat-t5-3b", "chatglm-6b"],
    "Tier 8": ["dolly-v2-12b", "stablelm-tuned-alpha-7b"],
    "Tier 9": ["llama-13b"]
}

# Create a mapping from model to tier
model_to_tier = {}
for tier, models in arena_model_tiers.items():
    for m in models:
        model_to_tier[m] = tier

# Define open source models (example assumption)
open_source_models = {
    "mistral-medium", "qwen1.5-72b-chat",
    "wizardlm-70b", "vicuna-33b", "tulu-2-dpo-70b", "nous-hermes-2-mixtral-8x7b-dpo", "llama-2-70b-chat", "openchat-3.5",
    "wizardlm-13b", "mistral-7b-instruct-v0.2", "codellama-34b-instruct", "mpt-30b-chat", "llama-2-13b-chat", "vicuna-13b",
    "qwen1.5-7b-chat", "falcon-180b-chat", "llama-2-7b-chat", "guanaco-33b", "qwen-14b-chat",
    "stripedhyena-nous-7b", "mistral-7b-instruct", "vicuna-7b", "qwen1.5-4b-chat",
    "koala-13b", "chatglm3-6b", "gpt4all-13b-snoozy",
    "mpt-7b-chat", "RWKV-4-Raven-14B", "chatglm2-6b", "alpaca-13b", "oasst-pythia-12b",
    "fastchat-t5-3b", "chatglm-6b",
    "dolly-v2-12b", "stablelm-tuned-alpha-7b"
    # Excluding "llama-13b" from Tier 9 due to licensing restrictions, but you can add if you consider it open.
}

# Optional: Map your model names to their Hugging Face repository IDs
# This step is crucial because model names in your tiers may not directly match the repository IDs.
# You may need to manually map them or follow a naming convention.
model_repo_mapping = {
    "gpt-4-0125-preview": "openai/gpt-4-0125-preview",
    "gpt-4-1106-preview": "openai/gpt-4-1106-preview",
    "gpt-4-0314": "openai/gpt-4-0314",
    "gpt-4-0613": "openai/gpt-4-0613",
    "mistral-medium": "mistral/mistral-medium",
    "claude-1": "anthropic/claude-1",
    "qwen1.5-72b-chat": "baichuan/qwen1.5-72b-chat",
    # ... add mappings for all models you want to check
    # Example:
    "vicuna-13b": "lmsys/vicuna-13b-v1.3",
    "llama-2-70b-chat": "meta-llama/Llama-2-70b-chat-hf",
    "mistral-7b-instruct": "mistral/mistral-7b-instruct",
    "oasst-pythia-12b": "EleutherAI/oasst-pythia-12b",
    "chatglm3-6b": "THUDM/chatglm3-6b",
    "qwen-14b-chat": "qwen/qwen-14b-chat",
    "alpaca-13b": "together/alpaca-13b",
    "chatglm-6b": "THUDM/chatglm-6b",
    # Add all other necessary mappings
}

# Initialize Hugging Face API
api = HfApi()

def can_load_from_hf(model_repo_id: str) -> bool:
    """
    Checks if a model repository exists on the Hugging Face Hub.
    Returns True if the model info can be retrieved (public or accessible),
    and False if it doesn't exist or can't be accessed.
    """
    try:
        api.model_info(model_repo_id)
        return True
    except RepositoryNotFoundError:
        return False
    except HTTPError:
        return False
    except Exception as e:
        print(f"Unexpected error for {model_repo_id}: {e}")
        return False

# Load the dataset
dataset = load_dataset("lmarena-ai/arena-human-preference-55k")

# Count comparisons
pair_counts = defaultdict(int)

for example in dataset['train']:
    model_l = example["model_a"]
    model_r = example["model_b"]
    pair = tuple(sorted([model_l, model_r]))
    pair_counts[pair] += 1

def tier_to_num(tier_name):
    return int(tier_name.split()[-1])

valid_rows = []
for (m1, m2), count in pair_counts.items():
    if m1 in model_to_tier and m2 in model_to_tier:
        t1 = model_to_tier[m1]
        t2 = model_to_tier[m2]
        if t1 != t2:
            # Check if both are open source
            if m1 in open_source_models and m2 in open_source_models:
                # Map model names to repository IDs
                repo1 = model_repo_mapping.get(m1, m1)  # Fallback to m1 if no mapping exists
                repo2 = model_repo_mapping.get(m2, m2)  # Fallback to m2 if no mapping exists
                
                # Check if both repositories exist on Hugging Face
                loadable1 = can_load_from_hf(repo1)
                loadable2 = can_load_from_hf(repo2)
                
                if loadable1 and loadable2:
                    dist = abs(tier_to_num(t1) - tier_to_num(t2))
                    valid_rows.append({
                        "model_1": m1,
                        "model_2": m2,
                        "model_1_tier": t1,
                        "model_2_tier": t2,
                        "tier_distance": dist,
                        "comparison_count": count,
                        "model_1_repo": repo1,
                        "model_2_repo": repo2
                    })

# Sort by comparison_count descending
valid_rows.sort(key=lambda x: x["comparison_count"], reverse=True)

# Create a DataFrame
df = pd.DataFrame(valid_rows, columns=[
    "model_1", "model_2", "model_1_tier", "model_2_tier",
    "tier_distance", "comparison_count", "model_1_repo", "model_2_repo"
])

# Write to a TSV file
df.to_csv("filtered_pairs.tsv", sep="\t", index=False)
print("Data written to filtered_pairs.tsv")
