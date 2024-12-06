import abc
import functools
import random

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import os
import random

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    trainer,
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

from routers.causal_llm.configs import RouterModelConfig
from routers.causal_llm.llm_utils import (
    load_prompt_format,
    to_openai_api_messages,
)
from routers.causal_llm.models  import CausalLLMClassifier

class Router(abc.ABC):
    NO_PARALLEL = False

    # Returns a float between 0 and 1 representing the value used to route to models, conventionally the winrate of the strong model.
    # If this value is >= the user defined cutoff, the router will route to the strong model, otherwise, it will route to the weak model.
    @abc.abstractmethod
    def calculate_strong_win_rate(self, prompt):
        pass

    def route(self, prompt, threshold, routed_pair):
        if self.calculate_strong_win_rate(prompt) >= threshold:
            return routed_pair.strong
        else:
            return routed_pair.weak

    def __str__(self):
        return "reference router"




class CausalLLMRouter(Router):
    def __init__(
        self,
        checkpoint_path,
        score_threshold=4,
        special_tokens=["[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]"],
        num_outputs=5,
        model_type="causal",
        model_id="meta-llama/Meta-Llama-3-8B",
        flash_attention_2=False,
    ):
        model_config = RouterModelConfig(
            model_id=model_id,
            model_type=model_type,
            flash_attention_2=flash_attention_2,
            special_tokens=special_tokens,
            num_outputs=num_outputs,
        )
        prompt_format = load_prompt_format(model_config.model_id)
        self.router_model = CausalLLMClassifier(
            config=model_config,
            ckpt_local_path=checkpoint_path,
            score_threshold=score_threshold,
            prompt_format=prompt_format,
            prompt_field="messages",
            additional_fields=[],
            use_last_turn=True,
        )
        system_message = hf_hub_download(
            repo_id=checkpoint_path, filename="system_ft_v5.txt"
        )
        classifier_message = hf_hub_download(
            repo_id=checkpoint_path, filename="classifier_ft_v5.txt"
        )
        with open(system_message, "r") as pr:
            system_message = pr.read()
        with open(classifier_message, "r") as pr:
            classifier_message = pr.read()
        self.to_openai_messages = functools.partial(
            to_openai_api_messages, system_message, classifier_message
        )

    def calculate_strong_win_rate(self, prompt):
        input = {}
        input["messages"] = self.to_openai_messages([prompt])
        output = self.router_model(input)
        if output is None:
            # Route to strong model if output is invalid
            return 1
        else:
            return 1 - output["binary_prob"]
        
class PreferenceData(Dataset):
    def __init__(self, data_df):
        self.data_df = data_df

    def __len__(self):
        return len(self.preferences)

    def __getitem__(self, idx):
        final_prompt = self.data_df.iloc[idx]

        inputs = {}
        inputs["messages"] = final_prompt

        #fill the right things here if required

        return inputs


def prepare_ft_messages(dataset_df: pd.DataFrame, label_key: str) -> pd.DataFrame:
    """
    Add messages for fine-tuning using the dataset dataframe, system message, and classifier message.
    """
    with open(f"assets/system_ft.txt", "r") as f1, open(
        f"assets/classifier_ft.txt", "r"
    ) as f2:
        system_message = f1.read()
        classifier_message = f2.read()

    # Create API formatted 'messages' column for each row in the dataset dataframe
    return dataset_df.apply(
        lambda row: to_openai_api_messages(
            [
                classifier_message.format(question=row["original"]),
                f"[[{row[label_key]}]]",
            ],
            system_message,
        ),
        axis=1,
    )

if __name__ == "__main__":
    # Load data
    prefix = os.getcwd()
    path = f"{prefix}/data/chatbot_arena_preference_data.tsv"
    data_df = pd.read_csv(path, sep="\t", header=0)

    checkpoint_path = "routellm/causal_llm_gpt4_augmented"

    

    model_name = "meta-llama/Meta-Llama-3-8B"
    causal_router = CausalLLMRouter(checkpoint_path=checkpoint_path)
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

    # Prepare dataset
    dataset = PreferenceData(prepare_ft_messages(dataset_df=data_df, label_key="score"))

    # Convert to HuggingFace Dataset for compatibility with Trainer
    def data_generator():
        for i in range(len(dataset)):
            yield dataset[i]

    hf_dataset = HFDataset.from_generator(data_generator)

    # Split dataset into training and evaluation sets (80% train, 20% eval)
    hf_dataset = hf_dataset.train_test_split(test_size=0.1)

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
        model = get_peft_model(causal_router.router_model, peft_config)
    except Exception as e:
        print(f"Error wrapping model with PEFT: {e}")
        raise e

    # Prepare data collator
    data_collator = DataCollatorWithPadding(tokenizer=causal_router.router_model.tokenizer)

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
    try:
        # Start the training process
        trainer.train()
    except Exception as e:
        print(f"An error occurred during training: {e}")

    win_rate = model.calculate_strong_win_rate("what is 3 + 4?")