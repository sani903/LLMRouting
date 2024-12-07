import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch
from huggingface_hub import login
import json
import random
from tqdm import tqdm
import csv
import os


def load_jsonlines(file_name: str):
    with open(file_name, 'r') as f:
        return [json.loads(line) for line in f]


prefix = os.getcwd()
test_data = load_jsonlines(f"{prefix}/data/test_gsm8k.jsonl")




def nshot_chats(nshot_data: list, n: int, question: str) -> dict:

    def question_prompt(s):
        return f'Question: {s}'

    def answer_prompt(s):
        return f'Answer: {s}'

    chats = [
        {
            "role": "system",
            "content": "You are a grade school math problem solver. At the end, you MUST write the answer as an integer after '####'. Let's think step by step.",
        },
    ]

    random.seed(42)
    for qna in random.sample(nshot_data, n):

        chats.append(
            {"role": "user", "content": question_prompt(qna["question"])})
        chats.append(
            {"role": "assistant", "content": answer_prompt(qna["answer"])})


    chats.append({"role": "user", "content": question_prompt(question)+" Let's think step by step. At the end, you MUST write the answer as an integer after '####'."})

    return chats


def extract_ans_from_response(answer: str, eos=None):
    if eos:
        answer = answer.split(eos)[0].strip()

    answer = answer.split('####')[-1].strip()

    for remove_char in [',', '$', '%', 'g']:
        answer = answer.replace(remove_char, '')

    try:
        return int(answer)
    except ValueError:
        return answer
N_SHOT = 8

all_prompts = []
csv_file_path = f"{prefix}/data/test_gsm8k_queries_llama3.1.json"
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['prompt'])

    for qna in tqdm(test_data):
        messages = nshot_chats(nshot_data=test_data, n=N_SHOT, question=qna['question'])
        
        # Write to CSV
        # csv_writer.writerow([prompt])
        all_prompts.append({"prompt": messages})

with open(csv_file_path, 'w', encoding='utf-8') as jsonfile:
    json.dump(all_prompts, jsonfile, ensure_ascii=False, indent=4)

print(f"Results saved to {csv_file_path}")
