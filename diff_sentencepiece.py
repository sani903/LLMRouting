import os
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from tqdm import tqdm
import re
import gc
from huggingface_hub import login

# ======================= Configuration =======================

# Securely handle Hugging Face token
# It's recommended to set your token as an environment variable for security
# For example, in your shell: export HF_TOKEN="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"
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    HF_TOKEN = "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"
    # raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")

# Quantization configuration using transformers' BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)

# Set CUDA launch blocking for better error tracing (useful during debugging)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ======================= Initialization =======================

# Hugging Face login
login(token=HF_TOKEN)

# Clear CUDA cache
torch.cuda.empty_cache()

# Load the dataset
df = pd.read_csv('train_router.csv', encoding="utf-8")

# Decode Unicode escape sequences in all string columns
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].apply(lambda x: x.encode('utf-8').decode('unicode_escape') if isinstance(x, str) else x)

# Drop rows with NaN values
df.dropna(inplace=True)  # Alternatively, use df.fillna('') to replace NaNs with empty strings

# Define cache directory
cache_dir = '/scratch/ambuja/model'

# Model name
model_name = "lmsys/vicuna-13b-v1.5"

# ======================= Tokenizer Setup =======================

# Load tokenizer with use_fast=False to ensure compatibility with SentencePiece
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    use_fast=False  # Use slow tokenizer to support SentencePiece
)

# Set padding token (using EOS token as padding) if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ======================= Model Setup =======================

# Determine primary device
primary_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load model configuration
config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)

# Load the model with quantization and automatic device mapping
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    quantization_config=quantization_config,  # Apply quantization config
    device_map="auto",  # Automatically map model to available GPUs
    torch_dtype=torch.float16,
    cache_dir=cache_dir
).eval()

# ======================= Utility Functions =======================

def calculate_log_likelihood_batch(logits, response_ids, prompt_length, device):
    """
    Calculate the log likelihood for the response part, ignoring the prompt part.
    
    Args:
        logits (np.ndarray): Logits from the model.
        response_ids (List[List[int]]): Token IDs for responses.
        prompt_length (List[int]): Lengths of the prompts.
        device (torch.device): The primary device (cuda:0).
    
    Returns:
        List[float]: Log likelihoods for each response.
    """
    log_likelihoods = []
    for i, response_id in enumerate(response_ids):
        # Slice the logits to focus only on the response part (after the prompt)
        response_logits = logits[i, prompt_length[i]:prompt_length[i] + len(response_id)]
        
        # Convert logits to tensor and move to primary device
        response_logits_tensor = torch.tensor(response_logits).to(device)
        
        # Get the token probabilities using softmax
        log_probs_soft = torch.nn.functional.log_softmax(response_logits_tensor, dim=-1)
        
        # Convert response_ids to tensor and move to primary device
        response_ids_tensor = torch.tensor(response_id).to(device)
        
        # Gather the log probabilities for the actual tokens
        log_probs = log_probs_soft[torch.arange(len(response_id), device=device), response_ids_tensor]
        
        # Sum up the log probabilities for the entire response
        log_likelihood = log_probs.sum().item()
        log_likelihoods.append(log_likelihood)
    
    return log_likelihoods

def process_batch(prompts, winning_responses, losing_responses):
    """
    Process a batch of prompts and responses to calculate log likelihood differences.
    
    Args:
        prompts (List[str]): List of prompt strings.
        winning_responses (List[str]): List of winning response strings.
        losing_responses (List[str]): List of losing response strings.
    
    Returns:
        Dict[str, List]: Dictionary containing log likelihood differences and the corresponding prompts and responses.
    """
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r"[^\x20-\x7E]+", "", text)  # Remove non-printable characters
        return text.strip()
    
    prompts = [clean_text(p) for p in prompts]
    winning_responses = [clean_text(wr) for wr in winning_responses]
    losing_responses = [clean_text(lr) for lr in losing_responses]
    
    if not all(prompts) or not all(winning_responses) or not all(losing_responses):
        raise ValueError("One of the inputs contains empty or invalid strings.")
    
    # Tokenize prompts and responses
    prompt_tokens = tokenizer(prompts, add_special_tokens=False, truncation=True)['input_ids']
    win_tokens = tokenizer(winning_responses, add_special_tokens=False, truncation=True)['input_ids']
    lose_tokens = tokenizer(losing_responses, add_special_tokens=False, truncation=True)['input_ids']
    
    # Concatenate prompt and response tokens
    win_response_pairs = [prompt + response for prompt, response in zip(prompt_tokens, win_tokens)]
    lose_response_pairs = [prompt + response for prompt, response in zip(prompt_tokens, lose_tokens)]
    
    # Pad the concatenated sequences and move to primary device
    win_inputs = tokenizer.pad(
        {"input_ids": win_response_pairs},
        padding=True,
        return_tensors="pt"
    ).to(primary_device)
    
    lose_inputs = tokenizer.pad(
        {"input_ids": lose_response_pairs},
        padding=True,
        return_tensors="pt"
    ).to(primary_device)
    
    with torch.no_grad():
        win_outputs = model(**win_inputs, return_dict=True)
        lose_outputs = model(**lose_inputs, return_dict=True)
    
    logits = win_outputs.logits.detach().cpu().numpy()
    losing_logits = lose_outputs.logits.detach().cpu().numpy()
    
    # Get the length of the prompt tokens
    prompt_lengths = [len(prompt_tokens[i]) for i in range(len(prompts))]
    
    # Calculate log likelihoods for both winning and losing responses (ignoring prompt tokens)
    winning_log_likelihoods = calculate_log_likelihood_batch(logits, win_tokens, prompt_lengths, primary_device)
    losing_log_likelihoods = calculate_log_likelihood_batch(losing_logits, lose_tokens, prompt_lengths, primary_device)
    
    # Normalize the log likelihoods by the length of the response
    normalized_winning_log_likelihoods = [ll / len(response) for ll, response in zip(winning_log_likelihoods, win_tokens)]
    normalized_losing_log_likelihoods = [ll / len(response) for ll, response in zip(losing_log_likelihoods, lose_tokens)]
    
    # Calculate the difference in log likelihoods
    log_likelihood_diffs = [l - w for l, w in zip(normalized_losing_log_likelihoods, normalized_winning_log_likelihoods)]
    
    return {
        'diff': log_likelihood_diffs,
        'prompts': prompts,
        'winning_response': winning_responses,
        'losing_response': losing_responses
    }

# ======================= Processing =======================

# Process the dataset in batches
batch_size = 8  # Adjust based on your GPU memory capacity
results = []

# Iterate over the dataset in batches
for i in tqdm(range(0, len(df), batch_size), desc="Processing Batches"):
    batch = df.iloc[i:i + batch_size]
    
    try:
        result = process_batch(
            batch['prompt'].str.strip().astype(str).tolist(),
            batch['winning_response'].str.strip().astype(str).tolist(),
            batch['losing_response'].str.strip().astype(str).tolist()
        )
    except ValueError as ve:
        print(f"ValueError at batch starting index {i}: {ve}")
        continue
    except Exception as e:
        print(f"Unexpected error at batch starting index {i}: {e}")
        continue
    
    # Aggregate results
    batch_results = [dict(zip(result.keys(), t)) for t in zip(*result.values())]
    results.extend(batch_results)
    
    # Clear CUDA cache after processing each batch to free up memory
    torch.cuda.empty_cache()
    
    # Save intermediate results periodically
    if (i + batch_size) % 100 == 0 or i + batch_size >= len(df):
        intermediate_df = pd.DataFrame(results)
        intermediate_df.to_csv('diff1.csv', index=False)
        print(f"Saved results up to row {i + batch_size}")

print("Processing complete. Final results saved as 'diff1.csv'.")
