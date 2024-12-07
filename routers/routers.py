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
from tqdm import tqdm

from peft import PeftModel
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
        
    def train(self, dataset):
        # Convert to HuggingFace Dataset for compatibility with Trainer
        def data_generator():
            for i in range(len(dataset)):
                yield dataset[i]

        hf_dataset = HFDataset.from_generator(data_generator)

        # Split dataset into training and evaluation sets (80% train, 20% eval)
        # hf_dataset = hf_dataset.train_test_split(test_size=0.05)

        # Prepare data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.router_model.tokenizer)

        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=3,  # Stop after 3 evaluations with no improvement
            early_stopping_threshold=0.0  # Minimum change to qualify as improvement
        )

        # Define evaluation metric
        metric = load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = torch.argmax(torch.tensor(logits), dim=-1)
            # Only consider non -100 labels
            mask = labels != -100
            true_labels = labels[mask]
            pred_labels = predictions[mask]
            return metric.compute(predictions=pred_labels, references=true_labels)


        # Set up training arguments
        training_args = TrainingArguments(
            output_dir="./results_chatbot_filtered_augmented",
            evaluation_strategy="no", 
            save_strategy="epoch",         # Save the model at the end of each epoch
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=1,
            learning_rate=1e-4,
            weight_decay=0.01,
            save_total_limit=1,
            load_best_model_at_end=False,  # No evaluation, so disable loading best model
            logging_steps=15,
            gradient_accumulation_steps=8,
            fp16=False,
            bf16=True,
            log_level='warning',
            max_grad_norm=1.0,
            logging_strategy="steps",
            logging_first_step=True,
            logging_dir="./logs",
            report_to=["wandb"],                       
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
        )



        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=hf_dataset,
            # eval_dataset=hf_dataset["test"],
            tokenizer=self.router_model.tokenizer,
            data_collator=data_collator,
            # compute_metrics=compute_metrics,
            # callbacks=[early_stopping],
        )
        try:
            # Start the training process
            trainer.train()
        except Exception as e:
            print(f"An error occurred during training: {e}")
            # Attempt to save the current state of the model
            try:
                trainer.save_model("./interrupted_training_checkpoint_1")
                print("Model saved successfully before interruption.")
            except Exception as save_error:
                print(f"Failed to save the model: {save_error}")
            # Optionally, re-raise the exception if you want the program to exit
            raise e
           
class PreferenceData(Dataset):
    def __init__(self, data_df, tokenizer, max_length=512):
        self.data_df = data_df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        # Extract the messages for this item
        final_prompts = self.data_df.iloc[idx]

        # Extract the system, user, and assistant messages
        system_msg = next(msg["content"] for msg in final_prompts if msg["role"] == "system")
        user_msg = next(msg["content"] for msg in final_prompts if msg["role"] == "user")
        assistant_msg = next(msg["content"] for msg in final_prompts if msg["role"] == "assistant")

        # Combine into a single text input (excluding the assistant's response)
        prompt_text = f"{system_msg.strip()}\n{user_msg.strip()}\nPrediction:\n"

        # Tokenize the prompt
        tokenized_prompt = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length - 1  # Reserve space for the rating token
        )

        # Tokenize the assistant's rating
        rating_token = assistant_msg.strip()
        rating_tokenized = self.tokenizer(
            rating_token,
            add_special_tokens=False
        )["input_ids"]

        # Ensure rating_tokenized is a single token
        if len(rating_tokenized) != 1:
            raise ValueError(f"Rating token '{rating_token}' is not a single token. Ensure it's added as a special token.")

        # Concatenate prompt and rating tokens
        input_ids = tokenized_prompt["input_ids"][0].tolist() + rating_tokenized
        attention_mask = tokenized_prompt["attention_mask"][0].tolist() + [1] * len(rating_tokenized)

        # Truncate to max_length if necessary
        input_ids = input_ids[:self.max_length]
        attention_mask = attention_mask[:self.max_length]

        # Convert to tensors
        input_ids = torch.tensor(input_ids).unsqueeze(0)  # Shape: [1, max_length]
        attention_mask = torch.tensor(attention_mask).unsqueeze(0)

        # Initialize labels with -100
        labels = torch.full_like(input_ids, -100)

        # Assign the rating token ID to the label at the last position
        labels[0, -len(rating_tokenized):] = torch.tensor(rating_tokenized)

        # Squeeze the batch dimension
        input_ids = input_ids.squeeze(0)         # Shape: [max_length]
        attention_mask = attention_mask.squeeze(0)
        labels = labels.squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }



def prepare_ft_messages(system_file, classifier_file, dataset_df: pd.DataFrame, label_key: str) -> pd.DataFrame:
    """
    Add messages for fine-tuning using the dataset dataframe, system message, and classifier message.
    """
    with open(system_file, "r") as f1, open(
        classifier_file, "r"
    ) as f2:
        system_message = f1.read()
        classifier_message = f2.read()

    # Create API formatted 'messages' column for each row in the dataset dataframe
    dataset_df["messages"] =  dataset_df.apply(
        lambda row: to_openai_api_messages(
            messages=[
                row["original"],
                f"[[{row[label_key]}]]",
            ],
            system_message=system_message,
            classifier_message=classifier_message
        ),
        axis=1,
    )
    return dataset_df["messages"]

if __name__ == "__main__":
    # Load data
    prefix = os.getcwd()
    huggingface_checkpoint_path = "routellm/causal_llm_gpt4_augmented"
    
    model_name = "meta-llama/Meta-Llama-3-8B"

    evaluate = True

    if not evaluate:
        path = f"{prefix}/data/chatbot_arena_mistral_llama_augmented_preference_data.tsv"
        data_df = pd.read_csv(path, sep="\t", header=0)

        data_df['score'] = data_df['preference'].apply(
            lambda preference: random.choice([1, 2, 3]) if preference == 0 else random.choice([4, 5])
        )
        causal_router = CausalLLMRouter(checkpoint_path=huggingface_checkpoint_path)
        wandb.init(
            project="preference_model_training",
            name="causal_train_1",
            config={
                "learning_rate": 1e-4,
                "epochs": 1,
                "batch_size": 1,
                "model_name": model_name,
            },
            resume="allow",
        )

        # Prepare dataset
        label_key = "score"

        dataset = PreferenceData(prepare_ft_messages(f"{prefix}/routers/causal_llm/system_ft_v5.txt", f"{prefix}/routers/causal_llm/classifier_ft_v5.txt", dataset_df=data_df, label_key="score"), causal_router.router_model.tokenizer)

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
            model = get_peft_model(causal_router.router_model.transformer_model, peft_config)
            # Replace the internal model with the PEFT-enhanced model
            causal_router.router_model.transformer_model = model
        except Exception as e:
            print(f"Error wrapping model with PEFT: {e}")
            raise e
        causal_router.train(dataset)
    else:
        data_path = os.path.join(prefix, "data", "chatbot_arena_preference_data_validate.tsv")
        out_predictions = os.path.join(prefix, "data", "chatbot_arena_preference_augmented_data_validate.tsv")
        data_df = pd.read_csv(data_path, sep="\t", header=0)
        peft_trained_checkpoint_path = os.path.join(prefix, "results_chatbot_filtered_augmented", "checkpoint-3197")

        causal_router = CausalLLMRouter(checkpoint_path=huggingface_checkpoint_path)

        try:
            # Apply PEFT to the internal model within the classifier
            peft_model = PeftModel.from_pretrained(causal_router.router_model.transformer_model, peft_trained_checkpoint_path)
            causal_router.router_model.transformer_model = peft_model 
            print("PEFT model loaded successfully.")
        except Exception as e:
            print(f"Error wrapping model with PEFT: {e}")
            raise e
        
        tqdm.pandas(desc="Processing")
        data_df['win_rate'] = data_df['original'].progress_apply(
            lambda value: causal_router.calculate_strong_win_rate(value)
        )

        data_df['prediction'] = (data_df['win_rate'] > 0.5).astype(int)
        accuracy = (data_df["preference"] == data_df["prediction"]).mean()
        print(f"Accuracy: {accuracy:.4f}")

        columns = ['prediction', 'preference', 'original']
        data_df = data_df[columns]
        data_df.to_csv(out_predictions, sep='\t', index=False)

        # causal_router.evaluate(dataset)
        # example of getting result for an example
        # win_rate = causal_router.calculate_strong_win_rate("what is 3 + 4?")
        # threshold = 0.5
        # if win_rate > threshold:
        #     prediction = 0
        # else:
        #     prediction = 1