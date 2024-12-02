
import pandas as pd
import os


import json


class ChatBotArenaDataset:
    def __init__(self, dataset_name, data_output_file, data_out):
        self.dataset_name = dataset_name
        self.metadata = ""
        self.data_output_dir = data_output_file
        self.process_dataset(data_output_file, data_out)

    def process_dataset(self, file_path, data_output_file):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                # Extract the prompt string from the list
                original = json.loads(entry['prompt'])[0]
                original = original.replace("\n", "").replace(" ", " ").strip()
                model_a = entry['strong']
                model_b = entry['weak']
                preference = entry['label']
                data.append([original, model_a, model_b, preference])

        # Create a DataFrame with the desired columns
        df = pd.DataFrame(data, columns=['original', 'model_a', 'model_b', 'preference'])

        # Save the DataFrame as a TSV file
        df.to_csv(data_output_file, sep='\t', index=False)
        print(f"Dataset processed and saved to {data_output_file}")


if __name__ == '__main__':
    prefix = os.getcwd()

    chatbot_arena = ChatBotArenaDataset("chatbot_arena", f"{prefix}/data/train_split.jsonl", f"{prefix}/data/chatbot_arena_preference_data.tsv")