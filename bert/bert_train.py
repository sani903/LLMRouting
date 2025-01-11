import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import wandb
from torch.nn.utils import clip_grad_norm_
import pandas as pd
import boto3
import random
import numpy as np
from transformers import AutoModel

# Set seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

max_grad_norm = 1.0

# Initialize Wandb
wandb.init(
    project="router-training",
    name="xlm-roberta-routing-classification",
    config={
        "learning_rate": 2e-5,
        "batch_size": 16,
        "max_length": 512,
        "epochs": 10,
        "weight_decay": 0.001,
        "model_name": "xlm-roberta-base",
    }
)

class MixedResultsDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data.iloc[idx]['prompts']
        label = self.data.iloc[idx]['labels']
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load full dataset
data = pd.read_csv(local_file_path)
data = data.reset_index(drop=True)

# Add a 'labels' column based on the difference between 'strong' and 'weak'
data['labels'] = (data['strong'] > data['weak']).astype(int)
threshold = 0.5
data = data[abs(data['strong'] - data['weak']) > threshold]

# Split dataset: training set is everything except the last 2000 examples
train_data = data[:-2000].copy()
val_data = data[-2000:].copy()
train_data['strong'] = (train_data['strong'] - train_data['strong'].mean()) / train_data['strong'].std()
train_data['weak'] = (train_data['weak'] - train_data['weak'].mean()) / train_data['weak'].std()

val_data['strong'] = (val_data['strong'] - train_data['strong'].mean()) / train_data['strong'].std()
val_data['weak'] = (val_data['weak'] - train_data['weak'].mean()) / train_data['weak'].std()

# Tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModelForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=2)

# Dataset and DataLoader
train_dataset = MixedResultsDataset(train_data, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataset = MixedResultsDataset(val_data, tokenizer)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Optimizer, Scheduler, and Loss function
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
loss_fn = nn.CrossEntropyLoss()  # CrossEntropyLoss for classification

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
epochs = 5
gradient_accumulation_steps = 4
MAX_LOSS = torch.tensor(float('inf'), dtype=torch.float)

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")

    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
    
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

        # Compute loss
        loss = loss_fn(logits, labels)
        loss = loss / gradient_accumulation_steps
        loss.backward()

        # Optional: Gradient clipping
        if (step + 1) % gradient_accumulation_steps == 0:
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
    
        total_train_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

        # Log training loss and metrics to wandb
        wandb.log({
            "train_loss": loss.item(),
            "batch_class_0_confidence_mean": probs[:, 0].mean().item(),
            "batch_class_1_confidence_mean": probs[:, 1].mean().item(),
        })

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}")
    scheduler.step()

    # Validation loop
    model.eval()
    correct = 0
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            val_loss += loss_fn(logits, labels).item()
            correct += (preds == labels).sum().item()

    # Compute validation loss and accuracy
    avg_val_loss = val_loss / len(val_dataloader)
    accuracy = correct / len(val_dataset)
    print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    # Log validation metrics to wandb
    wandb.log({
        "val_loss": avg_val_loss,
        "val_accuracy": accuracy,
    })

    if avg_val_loss < MAX_LOSS:
        MAX_LOSS = avg_val_loss
        # Save model and optimizer weights
        save_path = "/tmp/model_checkpoint_classification.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
        }, save_path)

# S3 upload
bucket_name = "router-train"
s3_key = "checkpoints/model_checkpoint_classification.pth"
s3 = boto3.client('s3')
s3.upload_file(save_path, bucket_name, s3_key)
print(f"Checkpoint saved to S3://{bucket_name}/{s3_key}")

# Log final metrics
wandb.run.summary["final_train_loss"] = avg_train_loss
wandb.run.summary["final_val_loss"] = avg_val_loss
wandb.run.summary["final_val_accuracy"] = accuracy

# Finish Wandb
wandb.finish()

