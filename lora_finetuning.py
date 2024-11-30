import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset as HFDataset
from evaluate import load
import bitsandbytes as bnb
from peft import get_peft_model, LoraConfig, TaskType

# Custom Dataset class to handle the data
class PreferenceData(Dataset):
    def __init__(self, queries, preferences, tokenizer, max_length=512):
        self.queries = queries.tolist()
        self.preferences = preferences.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.preferences)

    def __getitem__(self, idx):
        query = str(self.queries[idx])
        preference = int(self.preferences[idx]) - 1  # Convert preference from 1/2 to 0/1

        # Tokenize the query
        inputs = self.tokenizer(
            query,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        # Add labels
        inputs['labels'] = torch.tensor(preference, dtype=torch.long)
        return inputs

if __name__ == "__main__":
    # Load data
    prefix = os.getcwd()
    path = f"{prefix}/data/synthetic_mixed_preference_data.tsv"
    data_df = pd.read_csv(path, sep="\t", header=0)

    # Initialize tokenizer
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Ensure that the pad token is defined
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset
    dataset = PreferenceData(data_df["original"], data_df["preference"], tokenizer)

    # Convert to HuggingFace Dataset for compatibility with Trainer
    def data_generator():
        for i in range(len(dataset)):
            yield dataset[i]

    hf_dataset = HFDataset.from_generator(data_generator)

    # Split dataset into training and evaluation sets (80% train, 20% eval)
    hf_dataset = hf_dataset.train_test_split(test_size=0.2)

    # Load the pre-trained model in 8-bit precision using bitsandbytes
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        load_in_8bit=True,  # Enable 8-bit loading
        device_map='auto',  # Automatically place layers on devices
        num_labels=2        # Number of output labels for classification
    )

    # Prepare LoRA (Low-Rank Adaptation) configuration for QLoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # Sequence Classification task
        inference_mode=False,
        r=8,                         # Low-rank dimension
        lora_alpha=32,               # Scaling factor
        lora_dropout=0.1             # Dropout for LoRA layers
    )

    # Wrap the model with PEFT to add LoRA adapters
    model = get_peft_model(model, peft_config)

    # Prepare data collator that will pad the inputs dynamically
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define evaluation metric
    metric = load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",                # Output directory
        evaluation_strategy="epoch",           # Evaluate at the end of each epoch
        per_device_train_batch_size=8,         # Training batch size per device
        per_device_eval_batch_size=8,          # Evaluation batch size per device
        num_train_epochs=3,                    # Total number of training epochs
        learning_rate=2e-4,                    # Learning rate
        weight_decay=0.01,                     # Weight decay
        save_total_limit=1,                    # Limit the total amount of checkpoints
        load_best_model_at_end=True,           # Load the best model at the end (for evaluation)
        logging_steps=10,                      # Log every 10 steps
        fp16=True,                             # Use 16-bit (mixed) precision training
        report_to="none"                       # Disable reporting to wandb or other loggers
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,                           # The instantiated Transformers model to be trained
        args=training_args,                    # Training arguments
        train_dataset=hf_dataset['train'],     # Training dataset
        eval_dataset=hf_dataset['test'],       # Evaluation dataset
        tokenizer=tokenizer,                   # Tokenizer
        data_collator=data_collator,           # Data collator
        compute_metrics=compute_metrics        # Function to compute metrics at evaluation
    )

    # Start the training process
    trainer.train()
