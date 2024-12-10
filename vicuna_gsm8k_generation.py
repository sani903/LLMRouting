import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch
from huggingface_hub import login
import json
import random
from tqdm import tqdm
import csv, os
login(token="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")

model_name = "openchat/openchat_3.5"
cache_dir = '/scratch/ambuja/model'
os.makedirs(cache_dir, exist_ok=True)

device = 'cuda'
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

# Set padding token (using EOS token as padding)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir).half().to(device).eval()
generator = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)


def get_response(chats): 
    gen_text = generator(chats)[0]  # First return sequence
    return gen_text['generated_text'][-1]['content']


def load_jsonlines(file_name: str):
    with open(file_name, 'r') as f:
        return [json.loads(line) for line in f]


def nshot_chats(nshot_data: list, n: int, question: str) -> dict:

    def question_prompt(s):
        return f'Question: {s}'

    def answer_prompt(s):
        return f'Answer: {s}'

    chats = []
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
test_data = load_jsonlines("test.jsonl")
csv_file_path = 'gsm8k_results_vicuna13b.csv'

with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Prompt', 'Correct', 'Model Response'])

    for qna in tqdm(test_data):
        messages = nshot_chats(nshot_data=test_data, n=N_SHOT, question=qna['question'])
        prompt = messages[-1]['content']  # Get the last message as the prompt
        
        response = get_response(messages)
        pred_ans = extract_ans_from_response(response)
        true_ans = extract_ans_from_response(qna['answer'])
        
        is_correct = pred_ans == true_ans
        # Write to CSV
        csv_writer.writerow([prompt, str(is_correct).upper(), response])
print(f"Results saved to {csv_file_path}")