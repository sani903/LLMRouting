from datasets import load_dataset

dataset = load_dataset("lmsys/lmsys-arena-human-preference-55k")
tier_1 = [
    "gpt-4-0314", 
    "gpt-4-0613", 
    "mistral-medium", 
    "claude-1", 
    "qwen1.5-72b-chat", "gpt-4-0125-preview", 
    "gpt-4-1106-preview"
]

# Tier 2
tier_2 = [
    "claude-2.0", 
    "mixtral-8x7b-instruct-v0.1", 
    "claude-2.1", 
    "gemini-pro-dev-api", 
    "gpt-3.5-turbo0314", 
    "gpt-3.5-turbo-0613", 
    "gemini-pro", 
    "gpt-3.5-turbo-0125", 
    "claude-instant-1", 
    "yi-34bchat", 
    "starling-lm-7b-alpha", 
    "wizardlm-70b", 
    "vicuna-33b", 
    "tulu-2-dpo-70b", 
    "nous-hermes-2-", 
    "mixtral-8x7b-dpo", 
    "llama-2-70b-chat", 
    "openchat-3.5"
]

# Tier 3
tier_3 = [
    "llama2-70b-steerlm-chat", 
    "pplx-70b-online", 
    "dolphin-2.2.1-mistral-7b", 
    "gpt-3.5-turbo1106", 
    "deepseek-llm-67b-chat", 
    "openhermes-2.5-mistral-7b", 
    "openchat-3.5-0106", 
    "wizardlm-13b", 
    "mistral-7b-instruct-v0.2", 
    "solar-10.7b-instruct-v1.0", 
    "zephyr-7b-beta", 
    "zephyr-7b-alpha", 
    "codellama-34b-instruct", 
    "mpt-30b-chat", 
    "llama-2-13b-chat", 
    "vicuna-13b", 
    "qwen1.5-7b-chat", 
    "pplx-7b-online", 
    "falcon-180b-chat", 
    "llama-2-7b-chat", 
    "guanaco-33b", 
    "qwen-14b-chat"
]

# Tier 4
tier_4 = [
    "stripedhyena-nous-7b", 
    "mistral-7b-instruct", 
    "vicuna-7b", 
    "qwen1.5-4b-chat", 
    "palm-2"
]

# Tier 5
tier_5 = [
    "koala-13b", 
    "chatglm3-6b", 
    "gpt4all-13b-snoozy"
]

# Tier 6
tier_6 = [
    "mpt-7b-chat", 
    "RWKV-4-Raven-14B", 
    "chatglm2-6b", 
    "alpaca-13b", 
    "oasst-pythia-12b"
]

# Tier 7
tier_7 = [
    "fastchat-t5-3b", 
    "chatglm-6b"
]

# Tier 8
tier_8 = [
    "dolly-v2-12b", 
    "stablelm-tuned-alpha-7b"
]

# Tier 9
tier_9 = [
    "llama-13b"
]

def filter_short_promptsssss(example):
    return len(example['promptssss']) >= 16
fd = dataset['train'].filter(filter_short_promptsssss)
print(f"Original dataset size: {len(dataset['train'])}")
print(f"Filtered dataset size: {len(fd)}")

from datasets import Dataset

# Function to get the tier of a model
def get_tier(model_name):
    for i, tier in enumerate([tier_1, tier_2, tier_3, tier_4, tier_5, tier_6, tier_7, tier_8, tier_9]):
        if model_name in tier:
            return i + 1
    return None

# Filter function to keep only rows where model_a and model_b are from different tiers
def filter_different_tiers(example):
    tier_a = get_tier(example['model_a'])
    tier_b = get_tier(example['model_b'])
    return tier_a is not None and tier_b is not None and tier_a != tier_b

# Function to add 'strong', 'weak', and 'label' columns
def add_columns(example):
    tier_a = get_tier(example['model_a'])
    tier_b = get_tier(example['model_b'])
    
    if tier_a < tier_b:
        example['strong'] = example['model_a']
        example['weak'] = example['model_b']
        strong_wins = example['winner_model_a'] == 1
    else:
        example['strong'] = example['model_b']
        example['weak'] = example['model_a']
        strong_wins = example['winner_model_b'] == 1
    
    example['label'] = 1 if strong_wins else 0
    
    return example
dataset = fd
# Apply the filtering and add new columns
filtered_dataset = dataset.filter(filter_different_tiers)
processed_dataset = filtered_dataset.map(add_columns)

columns_to_drop = ['winner_model_a', 'winner_model_b', 'winner_tie', 'response_a', 'response_b']
final_dataset = processed_dataset.remove_columns(columns_to_drop)

print(f"Number of examples in final dataset: {len(final_dataset)}")
print(f"Features in the dataset: {final_dataset.features}")
print(f"First example: {final_dataset[0]}")

# Split the dataset
from datasets import load_dataset, Dataset, DatasetDict
train_val_split = final_dataset.train_test_split(test_size=5000, seed=42)
# Create a new DatasetDict with the splits
final_dataset = DatasetDict({
    'train': train_val_split['train'],
    'validation': train_val_split['test']
})

# Now you have separate train and validation datasets
print(f"Train dataset size: {len(final_dataset['train'])}")
print(f"Validation dataset size: {len(final_dataset['validation'])}")

# Save both splits to .jsonl files
def save_to_jsonl(dataset, filename):
    dataset.to_json(filename, orient='records', lines=True)

# Save train dataset
save_to_jsonl(final_dataset['train'], 'train_split_mixed_lower.jsonl')

# Save validation dataset
save_to_jsonl(final_dataset['validation'], 'validation_split_mixed_lower.jsonl')
