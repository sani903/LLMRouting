import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from peft import PeftModel


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
            padding='max_length',  # Ensure consistent padding
            return_tensors='pt'  # Return PyTorch tensors
        )
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs


if __name__ == "__main__":
    # 2. Load the Data
    prefix = os.getcwd()
    path = f"{prefix}/data/chatbot_arena_preference_data.tsv"
    data_df = pd.read_csv(path, sep="\t", header=0)

    # 3. Load the Tokenizer
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure pad token is defined
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})

    tokenizer.add_tokens(["[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]"], special_tokens=True)

    # 4. Load the Base Model and LoRA Adapters
    checkpoint_dir = f"{prefix}/results/checkpoint-1522"

    # Load the base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,  # Binary classification
        ignore_mismatched_sizes=True  # Handle size mismatches due to LoRA
    )

    # Load the LoRA adapters
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_dir,
        torch_dtype=torch.float32  # Use float32 for stability
    )
    model.eval()  # Set model to evaluation mode

    # 5. Resize Tokenizer Embeddings if Necessary
    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception as e:
        print(f"Error resizing token embeddings: {e}")
        raise e
    model.config.pad_token_id = tokenizer.pad_token_id

    # 6. Prepare the Dataset
    dataset = EvaluationData(data_df["original"], tokenizer, max_length=2048)

    # 7. Initialize the Trainer
    trainer = Trainer(model=model, tokenizer=tokenizer)

    # 8. Make Predictions
    predictions = trainer.predict(dataset)
    logits = predictions.predictions

    # 9. Convert Logits to Predicted Labels
    predicted_labels = logits.argmax(axis=-1)

    # 10. Assign Predictions to DataFrame
    data_df["prediction"] = predicted_labels

    # 11. Calculate Accuracy
    accuracy = (data_df["preference"] == data_df["prediction"]).mean()
    print(f"Accuracy: {accuracy:.4f}")
