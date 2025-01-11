import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
from huggingface_hub import login
import re
import gc
torch.cuda.empty_cache()
gc.collect()
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_8bit=True)


# Set CUDA launch blocking for better error tracing
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

login(token="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")
torch.cuda.empty_cache()
# Load the dataset
df = pd.read_csv('train_router.csv', encoding="utf-8")

# Decode Unicode escape sequences in all string columns
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].apply(lambda x: x.encode('utf-8').decode('unicode_escape') if isinstance(x, str) else x)

# Check for NaN values and fill or drop them if necessary
df.dropna(inplace=True)  # Replace NaNs with empty strings

cache_dir = '/scratch/ambuja/model'
# Load model and tokenizer
model_name = "meta-llama/Llama-2-13b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

# Set padding token (using EOS token as padding)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float16, cache_dir=cache_dir).to(device).eval()

# Function to calculate log likelihood for the response part (ignoring the prompt part)
def calculate_log_likelihood_batch(logits, response_ids, prompt_length):
    log_likelihoods = []
    # print("#")
    # print(logits.shape)
    # Process each response
    for i, response_id in enumerate(response_ids):
        # Slice the logits to focus only on the response part (after the prompt)
        response_tensor = torch.tensor(response_id).to(device)
        response_logits = logits[i, prompt_length[i]:prompt_length[i] + len(response_tensor)]
        # Get the token probabilities (use softmax)
        log_probs_soft = torch.nn.functional.log_softmax(torch.tensor(response_logits).to(device), dim=-1)
        # Calculate the log probabilities for the actual tokens (response_id)
        log_probs = log_probs_soft[torch.arange(len(response_tensor), device=device), response_tensor.to(device)]

        # Sum up the log probabilities for the entire response
        log_likelihood = log_probs.sum().item()
        log_likelihoods.append(log_likelihood)
    return log_likelihoods

def process_batch(prompts, winning_responses, losing_responses):
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

    # Concatenate prompt and response for both winning and losing responses
    prompt_tokens = tokenizer(prompts, add_special_tokens=False, truncation=True)['input_ids']
    win_tokens = tokenizer(winning_responses, add_special_tokens=False, truncation=True)['input_ids']
    lose_tokens = tokenizer(losing_responses, add_special_tokens=False, truncation=True)['input_ids']
    # Concatenate prompt and response tokens
    win_response_pairs = [prompt + response for prompt, response in zip(prompt_tokens, win_tokens)]
    lose_response_pairs = [prompt + response for prompt, response in zip(prompt_tokens, lose_tokens)]

    # Pad the concatenated sequences
    win_inputs = tokenizer.pad(
        {"input_ids": win_response_pairs},
        padding=True,
        return_tensors="pt"
    ).to(device)

    lose_inputs = tokenizer.pad(
        {"input_ids": lose_response_pairs},
        padding=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        win_outputs = model(**win_inputs, return_dict=True)
        lose_outputs = model(**lose_inputs, return_dict=True)

    logits = win_outputs.logits.detach().cpu().numpy()

    losing_logits = lose_outputs.logits.detach().cpu().numpy()

    # Get the length of the prompt tokens
    prompt_lengths = [len(prompt_tokens[i]) for i in range(len(prompts))]

    # Calculate log likelihoods for both winning and losing responses (ignoring prompt tokens)
    winning_log_likelihoods = calculate_log_likelihood_batch(logits, win_tokens, prompt_lengths)
    losing_log_likelihoods = calculate_log_likelihood_batch(losing_logits, lose_tokens, prompt_lengths)
    # Normalize the log likelihoods by the length of the response
    normalized_winning_log_likelihoods = [ll / len(response) for ll, response in zip(winning_log_likelihoods, win_tokens)]
    normalized_losing_log_likelihoods = [ll / len(response) for ll, response in zip(losing_log_likelihoods, lose_tokens)]
    log_likelihood_diffs = [l - w for l, w in zip(normalized_losing_log_likelihoods, normalized_winning_log_likelihoods)]

    return {
        'diff': log_likelihood_diffs,
        'prompts': prompts,
        'winning_response':winning_responses,
        'losing_response':losing_responses
    }

# Process the dataset in batches
batch_size = 1  # Start with a smaller batch size if you're facing OOM errors
results = []

for i in tqdm(range(0, len(df), batch_size)):
    batch = df.iloc[i:i + batch_size]
    # Ensure all inputs are correctly formatted as lists of strings
    try:
        result = process_batch(
            batch['prompt'].str.strip().astype(str),
            batch['winning_response'].str.strip().astype(str),
            batch['losing_response'].str.strip().astype(str)
        )
    except ValueError:
        print(i)
        print("VlaueError")
        continue

    results.extend([dict(zip(result.keys(), t)) for t in zip(*result.values())])

    # Clear cache after processing each batch
    torch.cuda.empty_cache()

    if (i + batch_size) % 100 == 0 or i + batch_size >= len(df):
        pd.DataFrame(results).to_csv('diff1.csv', index=False)
        print(f"Saved results up to row {i + batch_size}")

print("Processing complete. Final results saved.")
