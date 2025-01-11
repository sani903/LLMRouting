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
import wandb

from transformers import TrainerCallback, EarlyStoppingCallback, TrainerState, TrainerControl
from transformers import Trainer, PreTrainedModel
from typing import Optional, Dict, Union, Any, Tuple

from prompt_format import PromptFormat

# Enable cuDNN benchmarking for potential speedups
torch.backends.cudnn.benchmark = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROMPT_FORMAT_CONFIGS = {
    # "meta-llama/Meta-Llama-3-8B": {
    #     "system": "<|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>",
    #     "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{instruction}<|eot_id|>",
    #     "trailing_assistant": "",
    #     "user": "<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>",
    #     "system_in_user": False,
    #     "bos": "<|begin_of_text|>",
    #     "default_system_message": "",
    # },
    "meta-llama/Meta-Llama-3-8B-Instruct":{
        "system": "<|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{instruction}<|eot_id|>",
        "trailing_assistant": "",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>",
        "system_in_user": False,
        "bos": "<|begin_of_text|>",
        "default_system_message": "",
    },
}

class SimpleCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # print("Callback: on_step_end triggered.")
        pass

class GradientMonitorCallback(TrainerCallback):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, special_tokens: list):
        super().__init__()
        self.token_ids = tokenizer.convert_tokens_to_ids(special_tokens)
        self.special_tokens = special_tokens

    def on_pre_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        print("GradientMonitorCallback: on_pre_optimizer_step triggered.")
        model = kwargs['model']
        embedding_layer = model.get_input_embeddings()
        embeddings = embedding_layer.weight

        if embeddings.grad is None:
            print("GradientMonitorCallback: Embeddings gradients are None.")
        else:
            # Extract gradients for special tokens
            special_gradients = embeddings.grad[self.token_ids]  # Shape: (len(special_tokens), embedding_dim)
            for token, grad in zip(self.special_tokens, special_gradients):
                grad_norm = grad.norm().item()
                print(f"GradientMonitorCallback: {token} gradient norm: {grad_norm}")
            
            # Compute overall embedding gradient norm
            embedding_grad_norm = embeddings.grad.norm().item()
            print(f"GradientMonitorCallback: Overall embedding gradient norm: {embedding_grad_norm}")

        # Calculate the overall gradient norm across all trainable parameters
        total_grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param_grad_norm = param.grad.norm().item()
                total_grad_norm += param_grad_norm ** 2  # Accumulate the square of the norms
        
        total_grad_norm = total_grad_norm ** 0.5  # Take the square root of the sum
        print(f"GradientMonitorCallback: Overall model gradient norm: {total_grad_norm}")


def load_prompt_format(model_id):
    prompt_format_dict = PROMPT_FORMAT_CONFIGS[model_id]
    return PromptFormat(**prompt_format_dict, is_generation=True)


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
    prob_1 = probabilities[:, :3].sum(dim=1)  # Shape: (batch_size,)
    prob_0 = probabilities[:, 3:].sum(dim=1)  # Shape: (batch_size,)

    # Combine probabilities
    combined_probs = torch.stack([prob_0, prob_1], dim=1)  # probability_0 is basically probability that stronger model will win for the given query 
    labels_one_hot = F.one_hot(labels.long(), num_classes=2).float()

    # Calculate binary cross-entropy loss
    loss = F.binary_cross_entropy(combined_probs, labels_one_hot)

    return loss, combined_probs

# Define a custom Trainer class
class CustomTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                },
            ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs  # Accept additional keyword arguments
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)

        if self.processing_class is None:
            raise ValueError("Tokenizer is not initialized.")

        loss, combined_probs = custom_loss_function(logits, labels, self.processing_class)

        wandb.log({
            'loss': loss.item(),
            'combined_probs_0': combined_probs[0][0].item(),
            'combined_probs_1': combined_probs[0][1].item()
        }, step=self.state.global_step)
        # for token in ["[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]"]:
        #     token_id = self.processing_class.convert_tokens_to_ids(token)
        #     embedding = model.get_input_embeddings().weight[token_id]
        #     if not embedding.requires_grad:
        #         print(f"Enabling requried grad for '{token}' which was not trainable.")
        #         embedding.requires_grad = True
        return (loss, outputs) if return_outputs else loss


# Define a custom Dataset class
class PreferenceData(Dataset):
    def __init__(self, queries, preferences, tokenizer, max_length=2048):
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
        preference = int(self.preferences[idx])

        prompt_format = load_prompt_format("meta-llama/Meta-Llama-3-8B-Instruct")

        system_message = "[Instruction]\nBased on the question provided below, predict the score an expert evaluator would give to an AI assistant's response, considering its helpfulness, relevance, adherence to facts, depth, creativity, and detail. Your prediction should infer the level of proficiency needed to address the question effectively. Use a scale from 1 to 5, where a higher score indicates a higher anticipated quality of response. Provide your prediction as: \"[[predicted rating]]\".\n\nScore criteria:\n- **4-5**: The AI assistant can produce a very strong answer, showing deep understanding, creativity, detailed insight, and high relevance.\n- **3**: The AI assistant can provide an adequate answer with moderate detail, relevance, and factual accuracy.\n- **1-2**: The AI assistant will struggle to produce a strong answer due to the question's difficulty, vagueness, or the assistant's limitations."
        classifier_message = "\n[Question]\n{question}\n\nPrediction:\n"

        messages = [{"role": "system", "content": system_message}]
        messages.append({"role": "user", "content": classifier_message.format(question=query)})

        final_prompt = prompt_format.generate_prompt(messages)
        inputs = self.tokenizer(
            final_prompt,
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
    path = f"{prefix}/data/chatbot_arena_preference_data.tsv"
    data_df = pd.read_csv(path, sep="\t", header=0)

    # Initialize tokenizer
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Ensure this is the correct model name
    wandb.init(
        project="preference_model_training",
        name="run_1",
        config={
            "learning_rate": 1e-4,
            "epochs": 1,
            "batch_size": 1,
            "model_name": model_name,
        },
        resume="allow",
    )
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
    dataset = PreferenceData(data_df["original"], data_df["preference"], tokenizer, max_length=512)

    # Convert to HuggingFace Dataset for compatibility with Trainer
    def data_generator():
        for i in range(len(dataset)):
            yield dataset[i]

    hf_dataset = HFDataset.from_generator(data_generator)

    # Split dataset into training and evaluation sets (80% train, 20% eval)
    hf_dataset = hf_dataset.train_test_split(test_size=0.1)

    # Check dataset for NaNs
    check_dataset_for_nan(hf_dataset['train'])

    # Define quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Enable 8-bit quantization for memory efficiency
        llm_int8_enable_fp32_cpu_offload=True, # Offload parts to CPU if necessary
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
            # quantization_config=quantization_config,
            device_map='auto',
            torch_dtype=torch.bfloat16,
        )
        # model.gradient_checkpointing_enable() -> this caused detachment of loss and loss.required_grad became false
        # model.get_input_embeddings().weight.requires_grad = True

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
        task_type=TaskType.CAUSAL_LM,
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

    for param in model.parameters():
        param.requires_grad = False

    special_tokens = ["[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]"]
    token_ids = tokenizer.convert_tokens_to_ids(special_tokens)

    embedding_layer = model.base_model.get_input_embeddings()
    embedding_layer.weight.requires_grad = True
    # for token_id in token_ids:
    #     embedding_layer.weight[token_id].requires_grad = True 

    # Print trainable parameters
    # for token in ["[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]"]:
    #     token_id = tokenizer.convert_tokens_to_ids(token)
    #     embedding = model.get_input_embeddings().weight[token_id]
    #     if not embedding.requires_grad:
    #         print(f"Enabling requried grad for '{token}' which was not trainable.")
    #         embedding.requires_grad = True
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
        per_device_train_batch_size=1,  # Adjusted for memory constraints
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        learning_rate=1e-4,            # Adjusted learning rate
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        logging_steps=15,
        gradient_accumulation_steps=8,
        fp16=False,                    # Disable FP16 training for stability
        bf16=True,                     # Enable BF16 training for speed
        log_level='warning',
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_first_step=True,
        logging_dir="./logs",
        report_to=[],
        dataloader_num_workers=2,      # Increase based on CPU cores
        dataloader_pin_memory=True,
    )

    # Initialize the loss recorder callback
    # loss_recorder = LossRecorderCallback()

    # Initialize the CustomTrainer with all required arguments
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_dataset['train'],
        eval_dataset=hf_dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[SimpleCallback(), GradientMonitorCallback(tokenizer=tokenizer, special_tokens=special_tokens), EarlyStoppingCallback(early_stopping_patience=2)],
    )
    # trainer.create_optimizer()
    # for idx, group in enumerate(trainer.optimizer.param_groups):
    #     print(f"Optimizer group {idx}:")
    #     for param in group['params']:
    #         print(f" - Parameter requires_grad: {param.requires_grad}, Shape: {param.shape}")

    try:
        # Start the training process
        trainer.train()

            # After training has started, the optimizer is initialized
        # for idx, group in enumerate(trainer.optimizer.param_groups):
        #     print(f"Optimizer group {idx}:")
        #     for param in group['params']:
        #         print(f" - Parameter requires_grad: {param.requires_grad}, Shape: {param.shape}")
    except Exception as e:
        print(f"An error occurred during training: {e}")

    # Check for NaNs in parameters
    check_for_nan_parameters(model)
