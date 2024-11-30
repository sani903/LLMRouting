import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    PreTrainedModel,
    LlamaConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
)
from datasets import Dataset as HFDataset
from evaluate import load
from peft import get_peft_model, LoraConfig, TaskType
import matplotlib.pyplot as plt

from transformers import TrainerCallback

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

# Custom Dataset class
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
            max_length=self.max_length,
            return_tensors='pt',
            padding='max_length'  # Ensure consistent input sizes
        )
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        # Add labels
        inputs['labels'] = torch.tensor(preference, dtype=torch.long)
        return inputs

# Custom model class for sequence classification
class LlamaForSequenceClassification(PreTrainedModel):
    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # Initialize the base LLaMA model
        self.model = LlamaForCausalLM(config)
        
        # Add a classification head
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        inputs_embeds=None,
        **kwargs
    ):
        # Pass all additional kwargs to the base model to handle unexpected arguments
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        # Access the last hidden state from hidden_states
        if outputs.hidden_states is not None:
            # hidden_states is a tuple with one entry per layer + embedding
            # Use the last layer's hidden state
            pooled_output = outputs.hidden_states[-1][:, -1, :]  # Shape: (batch_size, hidden_size)
        else:
            raise ValueError("Hidden states are not returned by the model. Please set output_hidden_states=True in the config.")

        logits = self.classifier(pooled_output)  # Shape: (batch_size, num_labels)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {'loss': loss, 'logits': logits}

# Function to print trainable parameters
def print_trainable_parameters(model):
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
def check_dataset_for_nan(dataset):
    for idx in range(len(dataset)):
        inputs = dataset[idx]
        for key, value in inputs.items():
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            if torch.isnan(value).any():
                print(f"NaN found in input '{key}' at index {idx}.")

# Function to check model parameters for NaNs
def check_for_nan_parameters(model):
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

    # Prepare dataset
    dataset = PreferenceData(data_df["original"], data_df["preference"], tokenizer)

    # Convert to HuggingFace Dataset for compatibility with Trainer
    def data_generator():
        for i in range(len(dataset)):
            yield dataset[i]

    hf_dataset = HFDataset.from_generator(data_generator)

    # Split dataset into training and evaluation sets (60% train, 40% eval)
    hf_dataset = hf_dataset.train_test_split(test_size=0.2)

    # Check dataset for NaNs
    check_dataset_for_nan(hf_dataset['train'])

    # Define quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Disable 8-bit quantization for stability
    )

    # Load the pre-trained model and add classification head
    num_labels = 2

    # Load configuration with num_labels and enable hidden states
    config = LlamaConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        output_hidden_states=True,  # Enable hidden states
    )

    # Initialize the custom model
    model = LlamaForSequenceClassification(config)

    # Load pre-trained weights into the base LLaMA model
    try:
        base_model = LlamaForCausalLM.from_pretrained(
            model_name,
            config=config,
            quantization_config=quantization_config,
            device_map='auto',
        )
    except Exception as e:
        print(f"Error loading pre-trained LLaMA model: {e}")
        raise e

    model.model = base_model

    # Resize embeddings and update config
    try:
        model.model.resize_token_embeddings(len(tokenizer))
    except Exception as e:
        print(f"Error resizing token embeddings: {e}")
        raise e

    model.config.pad_token_id = tokenizer.pad_token_id

    # Move model to appropriate device (handled by device_map='auto', but ensure classifier is on the same device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Prepare LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "classifier"],  # Apply LoRA to query, value projections, and classifier
        modules_to_save=["classifier"],  # Keep classifier's base parameters trainable
    )

    # Wrap the model with PEFT to add LoRA adapters
    try:
        model = get_peft_model(model, peft_config)
    except Exception as e:
        print(f"Error wrapping model with PEFT: {e}")
        raise e

    # **Force the entire model to float32 to ensure dtype consistency**
    model = model.to(torch.bfloat16)
    # Alternatively, you can use:
    # model = model.float()

    # Print trainable parameters
    print_trainable_parameters(model)

    # Prepare data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define evaluation metric
    metric = load("accuracy")

    # Define compute_metrics function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=4,  # Adjusted for memory constraints
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        learning_rate=1e-4,  # Adjusted learning rate
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        logging_steps=5,
        gradient_accumulation_steps=4,
        fp16=False,  # Disable FP16 training for stability
        bf16=True,  # Set to True if hardware supports BF16
        log_level='debug',
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_first_step=True,
        logging_dir="./logs",
        report_to=["none"],
    )

    # Initialize the loss recorder callback
    loss_recorder = LossRecorderCallback()

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_dataset['train'],
        eval_dataset=hf_dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[loss_recorder],
    )

    try:
        # Start the training process
        trainer.train()
    except Exception as e:
        print(f"An error occurred during training: {e}")

    # Check for NaNs in parameters
    check_for_nan_parameters(model)

    # Plot training loss curve
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
