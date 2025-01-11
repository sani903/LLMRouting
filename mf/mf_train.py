MODEL_IDS = {
    "RWKV-4-Raven-14B": 0,
    "alpaca-13b": 1,
    "chatglm-6b": 2,
    "chatglm2-6b": 3,
    "chatglm3-6b": 4,
    "claude-1": 5,
    "claude-2.0": 6,
    "claude-2.1": 7,
    "claude-instant-1": 8,
    "codellama-34b-instruct": 9,
    "deepseek-llm-67b-chat": 10,
    "dolly-v2-12b": 11,
    "dolphin-2.2.1-mistral-7b": 12,
    "falcon-180b-chat": 13,
    "fastchat-t5-3b": 14,
    "gemini-pro": 15,
    "gemini-pro-dev-api": 16,
    "gpt-3.5-turbo-0125": 17,
    "gpt-3.5-turbo-0314": 18,
    "gpt-3.5-turbo-0613": 19,
    "gpt-3.5-turbo-1106": 20,
    "gpt-4-0125-preview": 21,
    "gpt-4-0314": 22,
    "gpt-4-0613": 23,
    "gpt-4-1106-preview": 24,
    "gpt4all-13b-snoozy": 25,
    "guanaco-33b": 26,
    "koala-13b": 27,
    "llama-13b": 28,
    "llama-2-13b-chat": 29,
    "llama-2-70b-chat": 30,
    "llama-2-7b-chat": 31,
    "llama2-70b-steerlm-chat": 32,
    "mistral-7b-instruct": 33,
    "mistral-7b-instruct-v0.2": 34,
    "mistral-medium": 35,
    "mixtral-8x7b-instruct-v0.1": 36,
    "mpt-30b-chat": 37,
    "mpt-7b-chat": 38,
    "nous-hermes-2-mixtral-8x7b-dpo": 39,
    "oasst-pythia-12b": 40,
    "openchat-3.5": 41,
    "openchat-3.5-0106": 42,
    "openhermes-2.5-mistral-7b": 43,
    "palm-2": 44,
    "pplx-70b-online": 45,
    "pplx-7b-online": 46,
    "qwen-14b-chat": 47,
    "qwen1.5-4b-chat": 48,
    "qwen1.5-72b-chat": 49,
    "qwen1.5-7b-chat": 50,
    "solar-10.7b-instruct-v1.0": 51,
    "stablelm-tuned-alpha-7b": 52,
    "starling-lm-7b-alpha": 53,
    "stripedhyena-nous-7b": 54,
    "tulu-2-dpo-70b": 55,
    "vicuna-13b": 56,
    "vicuna-33b": 57,
    "vicuna-7b": 58,
    "wizardlm-13b": 59,
    "wizardlm-70b": 60,
    "yi-34b-chat": 61,
    "zephyr-7b-alpha": 62,
    "zephyr-7b-beta": 63,
}

import json
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.optim import AdamW



# from routellm.routers.matrix_factorization.model import MODEL_IDS

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class PairwiseDataset(Dataset):
    def __init__(self, data):
        self.models_a = torch.tensor(
            [MODEL_IDS[sample["model_a"]] for sample in data], dtype=torch.int64
        )
        self.models_b = torch.tensor(
            [MODEL_IDS[sample["model_b"]] for sample in data], dtype=torch.int64
        )
        self.prompt_id = [sample["idx"] for sample in data]
        self.winners = [sample["winner"] for sample in data]

    def __len__(self):
        return len(self.models_a)

    def __getitem__(self, index):
        assert self.winners[index] in ["model_a", "model_b"], self.winners[index]
        if self.winners[index] == "model_a":
            return self.models_a[index], self.models_b[index], self.prompt_id[index]
        else:
            return self.models_b[index], self.models_a[index], self.prompt_id[index]

    def get_dataloaders(self, batch_size, shuffle=True):
        return DataLoader(self, batch_size, shuffle=shuffle)


class MFModel_Train(torch.nn.Module):
    def __init__(
        self,
        dim,
        num_models,
        num_prompts,
        text_dim=1536,
        num_classes=1,
        use_proj=True,
        npy_path=None,
    ):
        super().__init__()
        self.use_proj = use_proj
        self.P = torch.nn.Embedding(num_models, dim)
        self.Q = torch.nn.Embedding.from_pretrained(global_embeddings, freeze=True)
        # embeddings = np.load(npy_path)
        # self.Q.weight.data.copy_(global_embeddings)

        if self.use_proj:
            self.text_proj = torch.nn.Linear(text_dim, dim, bias=False)
        else:
            assert (
                text_dim == dim
            ), f"text_dim {text_dim} must be equal to dim {dim} if not using projection"

        self.classifier = nn.Linear(
            dim, num_classes, bias=False
        )  # bias should be False!

    def get_device(self):
        return self.P.weight.device

    def forward(self, model_a, model_b, prompt_indices, test=False, alpha=0.05):
        model_a = model_a.to(self.get_device())
        model_b = model_b.to(self.get_device())
        prompt_indices = prompt_indices.to(self.get_device())
    
        model_a_embed = self.P(model_a)
        model_a_embed = F.normalize(model_a_embed, p=2, dim=1)
        model_b_embed = self.P(model_b)
        model_b_embed = F.normalize(model_b_embed, p=2, dim=1)
        prompt_embed = self.Q(prompt_indices)
        prompt_embed = prompt_embed.to(self.text_proj.weight.dtype)
        if not test:
            # adding noise to stabilize the training
            prompt_embed += torch.randn_like(prompt_embed) * alpha
        if self.use_proj:
            prompt_embed = self.text_proj(prompt_embed)
    
        return self.classifier(
            (model_a_embed - model_b_embed) * prompt_embed
        ).squeeze()

    @torch.no_grad()
    def predict(self, model_win, model_loss, prompt):
        logits = self.forward(model_win, model_loss, prompt, test=True)
        return logits > 0


def evaluator(net, test_iter, device):
    net.eval()
    ls_fn = nn.BCEWithLogitsLoss(reduction="sum")
    ls_list = []
    correct = 0
    num_samples = 0
    with torch.no_grad():
        for models_a, models_b, prompts, labels in test_iter:
            models_a = models_a.to(device)
            models_b = models_b.to(device)
            prompts = prompts.to(device)
            labels = labels.to(device)
            
            logits = net(models_a, models_b, prompts)
            loss = ls_fn(logits, labels)
            pred_labels = (logits > 0).float()
            # print(f"Logits: {logits}")
            # print(f"Pred labels: {pred_labels}")
            # print(f"True labels: {labels}")
            
            correct += (pred_labels == labels).sum().item()
            ls_list.append(loss.item())
            num_samples += labels.shape[0]
    
    net.train()
    return float(sum(ls_list) / num_samples), correct / num_samples


def train_loops_with_logging(
    net,
    train_iter,
    test_iter,
    lr,
    weight_decay,
    alpha,
    num_epochs,
    device="cuda",
    evaluator=None,
    **kwargs,
):
    # Initialize the optimizer and loss function
    optimizer = AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    # Initialize W&B logging

    def train_epoch():
        net.train()
        train_loss_sum, n = 0.0, 0
        for models_a, models_b, prompts, labels in train_iter:
            models_a = models_a.to(device)
            models_b = models_b.to(device)
            prompts = prompts.to(device)
            labels = labels.to(device)
            
            output = net(models_a, models_b, prompts, alpha=alpha)
            ls = loss_fn(output, labels)
            
            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
            
            train_loss_sum += ls.item() * len(models_a)
            n += len(models_a)
        return train_loss_sum / n
    # print("Initial weights:")
    # print(net.P.weight.data[:5])
    train_losses = []
    test_losses = []
    test_acces = []
    best_test_acc = -1
    progress_bar = tqdm(total=num_epochs)
    for epoch in range(num_epochs):
        train_ls = train_epoch()
        train_losses.append(train_ls)
        info = {"train_loss": train_ls, "epoch": epoch}

        # Validation (if evaluator is provided)
        if evaluator:
            test_ls, test_acc = evaluator(net, test_iter, device)
            test_losses.append(test_ls)
            test_acces.append(test_acc)
            info.update(
                {
                    "test_loss": test_ls,
                    "test_acc": test_acc,
                    "epoch": epoch,
                    "best_test_acc": best_test_acc,
                    "best_test_loss": min(test_losses),
                }
            )
        else:
            test_ls = None  # No evaluation function provided

        if test_acc > best_test_acc:
            torch.save(net.state_dict(), "best_model_weights.pth")
            best_test_acc = test_acc

        # Log metrics to W&B after every epoch
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_ls,
                "test_loss": test_ls if test_ls is not None else "N/A",
                "test_accuracy": test_acc if test_acc is not None else "N/A",
                "best_test_acc": best_test_acc,
                "best_test_loss": min(test_losses),
            }
        )

        progress_bar.set_postfix(**info)
        progress_bar.update(1)
    # print("Final weights:")
    # print(model.P.weight.data[:5])

    progress_bar.close()


# if __name__ == "__main__":
#     # an example of training the model
#     json_path = "/path/to/pairwise_data.json"
#     npy_path = "/path/to/prompt/embedding.npy"

#     dim = 128
#     batch_size = 64
#     num_epochs = 100
#     alpha = 0.1
#     use_proj = True
#     lr = 3e-4
#     weight_decay = 1e-5

#     # load and filter data
#     data = json.load(open(json_path, "r"))

#     filtered_data = [
#         sample
#         for sample in data
#         if sample["winner"] in ["model_a", "model_b"]
#         and sample["model_a"] != sample["model_b"]
#     ]

#     # shuffle and prepare train test split
#     data_shuffled = filtered_data.copy()
#     random.shuffle(data_shuffled)
#     train_data = data_shuffled[: int(len(data_shuffled) * 0.95)]
#     test_data = data_shuffled[int(len(data_shuffled) * 0.95) :]

#     train_data_loader = PairwiseDataset(train_data).get_dataloaders(
#         batch_size=batch_size, shuffle=True
#     )
#     test_data_loader = PairwiseDataset(test_data).get_dataloaders(1024, shuffle=False)

#     model = MFModel_Train(
#         dim=dim,
#         num_models=len(MODEL_IDS),
#         num_prompts=len(data),
#         use_proj=use_proj,
#         npy_path=npy_path,
#     ).to("cuda")

#     train_loops(
#         model,
#         train_data_loader,
#         test_data_loader,
#         lr=lr,
#         weight_decay=weight_decay,
#         alpha=alpha,
#         num_epochs=num_epochs,
#         device="cuda",
#     )


import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb  # Import W&B for logging
import numpy as np
import pandas as pd

# Initialize W&B
wandb.init(
    project="re-router",  # Replace with your W&B project name
    config={
        "dim": 512,
        "batch_size": 32,
        "num_epochs": 500,
        "alpha": 0.1,
        "use_proj": True,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "device": "cuda",
        "new_data_weight": 2,
    },
)
embeddings = np.load(local_file_path2)
with open(local_file_path3, "r") as f:
    unique_prompts = json.load(f)
# df = pd.read_csv(local_file_path4)
# filtered_df = df[:1000]
# df = filtered_df[filtered_df["preferred_model"].notna() & (filtered_df["preferred_model"] != "")]

# df["strong"] = df["strong"].astype(float)
# df["weak"] = df["weak"].astype(float)
# df = df[abs(df["strong"] - df["weak"]) > 0.0]
# valid_test_data = df[df['preferred_model'] != -1]
# test_prompts = valid_test_data['prompts'].tolist()
# test_annotations = valid_test_data['preferred_model'].tolist()
# annotation_to_model = {0: "mistral-7b-v0.3", 1: "llama-3.2-3b"}
# test_data = [
#     {
#         "idx": i,  # Add a unique index
#         "prompts": prompt,
#         "model_a": annotation_to_model[annotation],
#         "model_b": "llama-3.2-3b" if annotation == 0 else "mistral-7b-v0.3",
#         "winner": "model_a",  # Since the annotation tells the winning model
#     }
#     for i, (prompt, annotation) in enumerate(zip(test_prompts, test_annotations))
# ]
prompt_to_embedding = {prompt: embeddings[i] for i, prompt in enumerate(unique_prompts)}
new_embeddings = []
# Paths and parameters
processed_data_path = local_file_path
npy_path = local_file_path2
dim = wandb.config.dim
batch_size = wandb.config.batch_size
num_epochs = wandb.config.num_epochs
alpha = wandb.config.alpha
use_proj = wandb.config.use_proj
lr = wandb.config.lr
weight_decay = wandb.config.weight_decay
device = wandb.config.device
new_data_weight = wandb.config.new_data_weight

# Load the processed data
with open(processed_data_path, "r") as f:
    undata = json.load(f)
data = [
    sample for sample in undata if sample["prompts"] in prompt_to_embedding
]

for sample in data:
    prompt = sample["prompts"]
    if prompt in prompt_to_embedding:
        new_embeddings.append(prompt_to_embedding[prompt])
    else:
        print(f"Embedding for prompt '{prompt}' not found!")
global_embeddings = torch.tensor(new_embeddings)
# print(len(global_embeddings))
# Split the data
# random.shuffle(data)
# train_data = [sample for sample in data if sample["prompts"] not in test_prompts]
# train_data = [sample for sample in data]
mistral_data = [
    sample for sample in data 
    if {"vicuna-13b", "chatglm-6b"} <= {sample["model_a"], sample["model_b"]}
]
non_mistral_data = [
    sample for sample in data 
    if not {"vicuna-13b", "chatglm-6b"} <= {sample["model_a"], sample["model_b"]}
]

# Take 5% of mistral data for testing
mistral_test_size = int(len(mistral_data) * 0.1)
mistral_test_data = mistral_data[:mistral_test_size]
mistral_train_data = mistral_data[mistral_test_size:]

# Combine non-mistral data with mistral train data
train_data = non_mistral_data + mistral_train_data
test_data = mistral_test_data
# print(f"Train data size: {len(train_data)}")
# print(f"Test data size: {len(test_data)}")
# from collections import Counter
# test_winners = Counter(sample['winner'] for sample in test_data)
# print(f"Test winners distribution: {test_winners}")
# print(len(test_data))
# Weight new data higher
# train_data_weighted = [
#     sample
#     for sample in train_data
#     if "mistral-7b-v0.3" in {sample["model_a"], sample["model_b"]}
# ]
# train_data_weighted = train_data_weighted * new_data_weight + non_mistral_data

# Prepare datasets and loaders
class PairwiseDataset(Dataset):
    def __init__(self, data):
        self.models_a = []
        self.models_b = []
        self.prompt_indices = []
        self.labels = []
        
        for i, sample in enumerate(data):
            if sample["prompts"] in prompt_to_embedding:
                # if sample["model_a"] == 'llama 3.2 3b':
                #     sample["model_a"] = 'llama-3.2-3b'
                # if sample["model_b"] == 'llama 3.2 3b':
                #     sample["model_b"] = 'llama-3.2-3b'
                if random.random() < 0.5:
                    # Keep original order
                    self.models_a.append(MODEL_IDS[sample["model_a"]])
                    self.models_b.append(MODEL_IDS[sample["model_b"]])
                    self.labels.append(1)
                else:
                    # Swap order
                    self.models_a.append(MODEL_IDS[sample["model_b"]])
                    self.models_b.append(MODEL_IDS[sample["model_a"]])
                    self.labels.append(0)
                self.prompt_indices.append(i)
        
        self.models_a = torch.tensor(self.models_a, dtype=torch.int64)
        self.models_b = torch.tensor(self.models_b, dtype=torch.int64)
        self.prompt_indices = torch.tensor(self.prompt_indices, dtype=torch.int64)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.models_a)

    def __getitem__(self, index):
        return self.models_a[index], self.models_b[index], self.prompt_indices[index], self.labels[index]

    def get_dataloaders(self, batch_size, shuffle=True):
        return DataLoader(self, batch_size, shuffle=shuffle)
# all_indices = set([sample["idx"] for sample in train_data + test_data])
# idx_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(all_indices))}
# for dataset in [train_data, test_data]:
#     for sample in dataset:
#         sample["idx"] = idx_mapping[sample["idx"]]
train_loader = PairwiseDataset(train_data).get_dataloaders(
    batch_size=batch_size, shuffle=True
)
test_loader = PairwiseDataset(test_data).get_dataloaders(
    batch_size=1024, shuffle=False
)


# print(f"Number of overlapping prompts: {len(overlap)}")
# print(f"Max train idx: {max_train_idx}, Max test idx: {max_test_idx}")
# Initialize the model
# num_prompts = len(idx_mapping)
# print(num_prompts)
model = MFModel_Train(
    dim=dim,
    num_models=len(MODEL_IDS),
    num_prompts=len(new_embeddings),
    use_proj=use_proj,
    npy_path=npy_path,
).to(device)
# print(f"Q embedding size: {model.Q.num_embeddings}")

# Train the model with W&B logging
train_loops_with_logging(
    model,
    train_loader,
    test_loader,
    lr=lr,
    weight_decay=weight_decay,
    alpha=alpha,
    num_epochs=num_epochs,
    device=device,
    evaluator=evaluator,
)

wandb.finish()
