import os
import random

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorWithPadding,
    LlamaConfig,
    LlamaTokenizer,
    LlamaForCausalLM, PreTrainedTokenizerBase,
)
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from evaluate import load
from peft import get_peft_model, LoraConfig, TaskType
import matplotlib.pyplot as plt

from transformers import TrainerCallback, EarlyStoppingCallback
from transformers import Trainer, PreTrainedModel
from typing import Optional, Dict, Union, Any, Tuple

# Enable cuDNN benchmarking for potential speedups
torch.backends.cudnn.benchmark = True

# Define a callback to record loss
class LossRecorderCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.steps = []
        self.eval_steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
                self.steps.append(state.global_step)
            if 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])
                self.eval_steps.append(state.global_step)

# Define the custom loss function
def custom_loss_function(logits: torch.Tensor, labels: torch.Tensor, tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    # Get the indices of the special tokens
    special_token_ids = tokenizer.convert_tokens_to_ids(["[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]"])

    # Extract logits for the last token's special tokens
    # Assuming the model generates the special tokens at the last position
    special_token_logits = logits[:, -1, special_token_ids]  # Shape: (batch_size, 5)

    # Apply softmax to get probabilities
    probabilities = F.softmax(special_token_logits, dim=-1)  # Shape: (batch_size, 5)

    # Sum probabilities for labels: 0 (tokens 1,2,3) and 1 (tokens 4,5)
    prob_0 = probabilities[:, :3].sum(dim=1)  # Shape: (batch_size,)
    prob_1 = probabilities[:, 3:].sum(dim=1)  # Shape: (batch_size,)

    # Combine probabilities
    combined_probs = torch.stack([prob_0, prob_1], dim=1)  # Shape: (batch_size, 2)

    # Calculate binary cross-entropy loss
    loss = F.binary_cross_entropy(combined_probs, labels.float())

    return loss

# Define a custom Trainer class
class CustomTrainer(Trainer):
    def __init__(self, *args, checkpoint_dir='/opt/ml/checkpoints', **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir or self.checkpoint_dir
        os.makedirs(output_dir, exist_ok=True)
        super().save_model(output_dir, _internal_call)

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, torch.Tensor]] = None):
        output_dir = output_dir or self.checkpoint_dir
        os.makedirs(output_dir, exist_ok=True)
        return super()._save(output_dir, state_dict)

    def _move_model_to_device(self, model: PreTrainedModel, device: torch.device):
        # Do nothing, as the model is already distributed
        pass

    def compute_loss(self, model: PreTrainedModel, inputs: Dict[str, torch.Tensor], return_outputs: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)

        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")

        loss = custom_loss_function(logits, labels, self.tokenizer)

        return (loss, outputs) if return_outputs else loss

# Define a custom Dataset class
class PreferenceData(Dataset):
    def __init__(self, queries, preferences, tokenizer, max_length=256):
        self.queries = queries.tolist()
        self.preferences = preferences.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.preferences)

    def __getitem__(self, idx):
        # [Insert the corrected __getitem__ method here]
        # As provided above
        query = str(self.queries[idx])
        preference = int(self.preferences[idx]) - 1  # 0 or 1

        if preference == 0:
            special_tokens = ["[[1]]", "[[2]]", "[[3]]"]
        else:
            special_tokens = ["[[4]]", "[[5]]"]

        special_token = random.choice(special_tokens)
        query_with_token = f"{query} {special_token}"

        inputs = self.tokenizer(
            query_with_token,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            padding='max_length'
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = torch.tensor(preference, dtype=torch.float)

        return inputs

# Function to print trainable parameters
def print_trainable_parameters(model: PreTrainedModel):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            print(f"Parameter '{name}' is trainable.")
            trainable_params += param.numel()
    print(
        f"Trainable params: {trainable_params} | All params: {all_param} | "
        f"Trainable%: {100 * trainable_params / all_param:.2f}%"
    )

# Function to check dataset for NaNs
def check_dataset_for_nan(dataset: HFDataset):
    for idx in range(len(dataset)):
        inputs = dataset[idx]
        for key, value in inputs.items():
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            if torch.isnan(value).any():
                print(f"NaN found in input '{key}' at index {idx}.")

# Function to check model parameters for NaNs
def check_for_nan_parameters(model: PreTrainedModel):
    for name, param in model.named_parameters():
        if param.requires_grad and torch.isnan(param).any():
            print(f"Parameter '{name}' contains NaNs.")

if __name__ == "__main__":
    # Load data
    prefix = os.getcwd()
    path = f"{prefix}/data/synthetic_mixed_preference_data.tsv"
    data_df = pd.read_csv(path, sep="\t", header=0)

    # Initialize tokenizer
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Ensure this is the correct model name
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except OSError:
        # If AutoTokenizer fails, try LlamaTokenizer
        print(f"AutoTokenizer could not load for model '{model_name}'. Trying LlamaTokenizer.")
        try:
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"Failed to load LlamaTokenizer: {e}")
            raise e

    # Add pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})

    # Add special tokens
    tokenizer.add_tokens(["[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]"], special_tokens=True)

    # Prepare dataset
    dataset = PreferenceData(data_df["original"], data_df["preference"], tokenizer, max_length=256)

    # Convert to HuggingFace Dataset for compatibility with Trainer
    def data_generator():
        for i in range(len(dataset)):
            yield dataset[i]

    hf_dataset = HFDataset.from_generator(data_generator)

    # Split dataset into training and evaluation sets (80% train, 20% eval)
    hf_dataset = hf_dataset.train_test_split(test_size=0.2)

    # Check dataset for NaNs
    check_dataset_for_nan(hf_dataset['train'])

    # Define quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Enable 8-bit quantization for memory efficiency
    )

    # Load the pre-trained model
    try:
        config = LlamaConfig.from_pretrained(
            model_name,
            num_labels=2,
            output_hidden_states=True,  # Enable hidden states for custom loss
        )
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            config=config,
            quantization_config=quantization_config,
            device_map='auto',
        )
    except Exception as e:
        print(f"Error loading pre-trained LLaMA model: {e}")
        raise e

    # Resize token embeddings to match the tokenizer
    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception as e:
        print(f"Error resizing token embeddings: {e}")
        raise e

    # Prepare LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],  # Adjust based on model's module names
    )

    # Apply PEFT to the model
    try:
        model = get_peft_model(model, peft_config)
    except Exception as e:
        print(f"Error wrapping model with PEFT: {e}")
        raise e

    # Ensure the entire model uses BF16 for consistency
    model = model.to(torch.bfloat16)

    # Print trainable parameters
    print_trainable_parameters(model)

    # Prepare data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define evaluation metric
    metric = load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",        # Save checkpoint at the end of each epoch
        per_device_train_batch_size=4,  # Adjusted for memory constraints
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        learning_rate=1e-4,            # Adjusted learning rate
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        logging_steps=15,
        gradient_accumulation_steps=4,
        fp16=False,                    # Disable FP16 training for stability
        bf16=True,                     # Enable BF16 training for speed
        log_level='debug',
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_first_step=True,
        logging_dir="./logs",
        report_to=["none"],
        dataloader_num_workers=4,      # Increase based on CPU cores
        dataloader_pin_memory=True,
    )

    # Initialize the loss recorder callback
    loss_recorder = LossRecorderCallback()

    # Initialize the CustomTrainer with all required arguments
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_dataset['train'],
        eval_dataset=hf_dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[loss_recorder, EarlyStoppingCallback(early_stopping_patience=2)],
    )

    try:
        # Start the training process
        trainer.train()
    except Exception as e:
        print(f"An error occurred during training: {e}")

    # Check for NaNs in parameters
    check_for_nan_parameters(model)

    # Plot training and evaluation losses
    plt.figure(figsize=(10, 5))
    plt.plot(loss_recorder.steps, loss_recorder.train_losses, label='Training Loss')
    plt.plot(loss_recorder.eval_steps, loss_recorder.eval_losses, label='Evaluation Loss', linestyle='--')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig("curve.png")
    plt.show()
