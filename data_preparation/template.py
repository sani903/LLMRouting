from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pandas as pd
import os
import numpy as np

from internal_router.gemini_router import GeminiRouter
from internal_router.llm import LLM
from internal_router.ollama import OllamaRouter
from internal_router.open_ai_router import OpenAIRouter
from internal_router.transformers_router import TransformerRouter
from internal_router.vllm_router import VLLMRouter


class CNNDataset(Dataset):
    def __init__(self, references, summaries, preferences):
        self.references = references
        self.summaries = summaries
        self.preferences = preferences

    def __len__(self):
        return len(self.references)

    def __getitem__(self, idx):
        reference = self.references[idx]
        summary = self.summaries[idx]
        preference = self.preferences[idx]
        return reference, summary, preference


def choose_router(router_name):
    if router_name == 'openai':
        return OpenAIRouter()
    elif router_name == 'gemini':
        return GeminiRouter()
    elif router_name == "vllm":
        return VLLMRouter()
    elif router_name == "huggingface":
        return TransformerRouter()
    return OllamaRouter()


class CNNDailySummarization:
    # "abisee/cnn_dailymail"
    # "3.0.0"
    def __init__(self, dataset_name, version, batch_size, model_1, base_url_1, model_2, base_url_2, router_name,
                 judge_router, judge_name, judge_base_url):
        self.dataset_name = dataset_name
        self.metadata = version
        # change here to add more metadata
        self.dataset_loaded = load_dataset(dataset_name, version)
        self.data, self.summary = self.process_dataset(self.dataset_loaded)
        self.dataset = CNNDataset(self.data, self.summary)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        self.model_1 = model_1
        self.model_2 = model_2
        self.base_url_1 = base_url_1
        self.base_url_2 = base_url_2
        self.llm = LLM()
        self.router = choose_router(router_name)
        self.judge_router = choose_router(judge_router)
        self.judge_name = judge_name
        self.judge_base_url = judge_base_url

    def process_dataset(self, ds):
        df_train = pd.DataFrame({
            'article': ds['train']['article'],  # Actual article (source text)
            'summary': ds['train']['highlights']  # Summarized version (highlights)
        })
        seed = 42
        np.random.seed(seed)
        return df_train.get("article"), df_train.get("summary")

    def get_preference(self, start_group, end_group, system_prompt_task, system_prompt_preference):
        group_id = 0
        final_data = [
            ("original", "gold_output", "dataset", "metadata", "model_1_output", "model_2_output", "preference")]
        for references, summaries in self.dataloader:
            group_id += 1
            if group_id < start_group:
                continue
            if group_id > end_group:
                break
            for reference, summary in zip(references, summaries):
                model_1_output = self.llm.call(router=self.router, model_name=model_1, base_url=base_url_1, prompt="",
                                               system_prompt=system_prompt_task)
                model_2_output = self.llm.call(router=self.router, model_name=model_2, base_url=base_url_2, prompt="",
                                               system_prompt=system_prompt_task)
                preference = self.llm.get_preference(router=self.judge_router, model_name=model_2, base_url=base_url_2,
                                                     prompt="", system_prompt=system_prompt_preference)
                # process all outputs
                final_data.append(
                    (reference, summary, self.dataset_name, self.metadata, model_1_output, model_2_output, preference))

        return final_data


if __name__ == '__main__':
    prefix = os.getcwd()
    router = "vllm"
    judge_router = "openai"
    model_1 = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_2 = "google/gemma-2-9b-it"
    model_3 = "tiiuae/falcon-7b-instruct"
    model_4 = "mistralai/Mistral-7B-Instruct-v0.3"
    base_url_1 = "http://babel-11-21:8081/v1/completions"
    base_url_2 = "http://babel-6-21:8082/v1/completions"
    base_url_3 = "http://babel-11-17:8084/v1/completions"
    base_url_4 = "http://shire-1-1:8083/v1/completions"

    judge_name = "gpt4o-mini"
    judge_base_url = ""
    cnn_summarizer = CNNDailySummarization("abisee/cnn_dailymail", "3.0.0", 32, model_1, base_url_1,
                                           model_2, base_url_2, router, judge_router, judge_name, judge_base_url)
    system_prompt_task = ""
    system_prompt_preference = ""
    final_data = cnn_summarizer.get_preference(1, 1)

    data_df = pd.DataFrame(final_data)
    data_df.to_csv(f"{prefix}/data/mixed_preference_data", index=False, header=0, sep="\t")
