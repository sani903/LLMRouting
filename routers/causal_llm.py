import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    Trainer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import PeftModel

torch.backends.cudnn.benchmark = True

# 1. Define the Dataset Class
class EvaluationData(Dataset):
    def __init__(self, queries, tokenizer, max_length=512):
        self.queries = queries.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = str(self.queries[idx])
        inputs = self.tokenizer(
            query,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs

if __name__ == "__main__":
    # Load data
    prefix = os.getcwd()
    data_path = os.path.join(prefix, "data", "chatbot_arena_preference_data.tsv")
    data_df = pd.read_csv(data_path, sep="\t", header=0)

    # Define directories
    base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Original base model
    adapter_dir = os.path.join(prefix, "results", "checkpoint-1522")  # LoRA adapters

    # Load the tokenizer from the base model directory
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Ensure pad token is defined
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})

    # Add special tokens (must match those added during finetuning)
    special_tokens = ["[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]"]
    tokenizer.add_tokens(special_tokens)
    # Save the tokenizer to a separate directory to avoid overwriting
    tokenizer.save_pretrained(os.path.join(prefix, "results", "tokenizer_custom"))

    # Load the base model with quantization and device mapping
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Enable 8-bit quantization for memory efficiency
    )

    try:
        base_model = LlamaForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map='auto',  # Automatically map to available devices
            trust_remote_code=True,  # If the model uses custom code
        )
    except Exception as e:
        print(f"Error loading pre-trained base model: {e}")
        raise e

    # Resize token embeddings to match the tokenizer
    try:
        base_model.resize_token_embeddings(len(tokenizer))
    except Exception as e:
        print(f"Error resizing token embeddings: {e}")
        raise e

    # Load the LoRA adapters from their separate directory
    try:
        model = PeftModel.from_pretrained(
            base_model,
            adapter_dir,
            torch_dtype=torch.float32,
            device_map='auto'  # Ensure adapters are mapped correctly
        )
    except Exception as e:
        print(f"Error loading LoRA adapters: {e}")
        model = base_model  # Fallback to base model if no adapters are found

    model.eval()
    # Prepare the dataset
    dataset = EvaluationData(data_df["original"], tokenizer, max_length=512)

    # Define TrainingArguments with optimized settings
    training_args = TrainingArguments(
        output_dir=os.path.join(prefix, "results"),           # Output directory
        per_device_eval_batch_size=2,                         # Reduce batch size as needed
        dataloader_num_workers=2,                             # Fewer workers to save CPU memory
        no_cuda=False,                                        # Ensure CUDA is used
        fp16=False,                                           # Disable FP16 if not needed
        bf16=True,                                           # Disable BF16 if not needed
        remove_unused_columns=False,                          # Prevent Trainer from removing columns
        disable_tqdm=False,                                   # Enable progress bars
    )

    # Initialize the Trainer with TrainingArguments
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
    )

    # Make predictions
    try:
        predictions = trainer.predict(dataset)
        logits = predictions.predictions
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise e
412
    # Convert logits to predicted labels
    try:
        # Get special token IDs
        special_token_ids = tokenizer.convert_tokens_to_ids(special_tokens)
        # Get logits for the last token
        last_token_logits = logits[:, -1, :]  # Shape: (num_samples, vocab_size)
        special_token_logits = last_token_logits[:, special_token_ids]  # Shape: (num_samples, 5)
        # Compute probabilities
        probabilities = torch.softmax(torch.tensor(special_token_logits), dim=-1)  # Shape: (num_samples, 5)
        prob_0 = probabilities[:, :3].sum(dim=1)  # Sum probabilities for labels 0
        prob_1 = probabilities[:, 3:].sum(dim=1)  # Sum probabilities for labels 1
        # Get predicted labels
        predicted_labels = torch.argmax(torch.stack([prob_0, prob_1], dim=1), dim=1).numpy()
    except Exception as e:
        print(f"Error converting logits to labels: {e}")
        raise e

    # Assign predictions to DataFrame
    data_df["prediction"] = predicted_labels

    # Calculate accuracy
    accuracy = (data_df["preference"] == data_df["prediction"]).mean()
    print(f"Accuracy: {accuracy:.4f}")

    # Save the results
    output_path = os.path.join(prefix, "data", "chatbot_arena_predictions.tsv")
    data_df.to_csv(output_path, sep="\t", index=False)
