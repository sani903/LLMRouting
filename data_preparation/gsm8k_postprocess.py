import os
import pandas as pd

def extract_ans_from_response(answer: str, eos=None):
    if eos:
        answer = answer.split(eos)[0].strip()

    answer = answer.split('####')[-1].strip()

    for remove_char in [',', '$', '%', 'g']:
        answer = answer.replace(remove_char, '')

    try:
        return int(answer)
    except ValueError:
        return "Unknown"

if __name__ == '__main__':
    prefix = os.getcwd()

    data_path = f"{prefix}/data/train_gsm8k_metallama_3.2_1b_responses.tsv"
    out_path = f"{prefix}/data/train_gsm8k_metallama_3.2_1b_processed.tsv"

    df = pd.read_csv(data_path, sep="\t", header=0)

    df["processed"] = df['response'].apply(
        lambda row: row.split("<|im_end|>")[0].strip().replace("~|", "")
    )
    df["score"] = df['processed'].apply(
        lambda row: extract_ans_from_response(row)
    )
    df_reversed = df[df.columns[::-1]]
    df_reversed.to_csv(out_path, sep="\t", index=False, header=True)
