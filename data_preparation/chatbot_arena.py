
import pandas as pd
import os


import json


class ChatBotArenaDataset:
    def __init__(self, dataset_name, data_output_file, data_out):
        self.dataset_name = dataset_name
        self.metadata = ""
        self.data_output_dir = data_output_file
        # self.process_dataset(data_output_file, data_out)
        self.process_augmented_dataset(data_output_file, data_out)

    def process_dataset(self, file_path, data_output_file):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                # Extract the prompt string from the list
                try:
                    prompt_list = json.loads(entry['prompt'])
                except json.JSONDecodeError:
                    print(f"Invalid JSON in 'prompt': {prompt_list}")
                    continue
                original = " ".join(prompt_list).replace("\n", " ").strip()
                model_a = entry['strong']
                model_b = entry['weak']
                preference_model = None
                if entry['label'] == 0:
                    preference_model = entry['model_a']
                else:
                    preference_model = entry['model_b']
                preference = model_b == preference_model
                data.append([original, model_a, model_b, int(preference)])

        # Create a DataFrame with the desired columns
        df = pd.DataFrame(data, columns=['original', 'model_a', 'model_b', 'preference'])

        # Save the DataFrame as a TSV file
        df.to_csv(data_output_file, sep='\t', index=False)
        print(f"Dataset processed and saved to {data_output_file}")
    
    def get_label(self, row):
        expression = abs(row['strong'] - row['weak']) >= 0.1
        if expression and row['strong'] > row['weak']:
            return 0
        if expression and row['weak'] > row['strong']:
            return 1
        
        return 2


    def process_augmented_dataset(self, file_path, data_output_file):
        data = []

        input_df = pd.read_csv(file_path, sep=",", header=0)
        input_df['preference'] = input_df.apply(
            lambda row: self.get_label(row), axis=1
        )
        input_df = input_df[input_df['preference'].isin([0, 1])]
        input_df['model_a'] = "vicuna"
        input_df['model_b'] = "chatglm"
        input_df['original'] = input_df['prompts']

        # Create a DataFrame with the desired columns
        columns_order = ['original', 'model_a', 'model_b', 'preference']

# Reorder the columns
        input_df = input_df[columns_order]

        # Save the DataFrame as a TSV file
        input_df.to_csv(data_output_file, sep='\t', index=False)
        print(f"Dataset processed and saved to {data_output_file}")


if __name__ == '__main__':
    prefix = os.getcwd()
    # chatbot_arena = ChatBotArenaDataset(
    #     "chatbot_arena",
    #     f"{prefix}/data/chatbot_arena_validation_split.jsonl",
    #     f"{prefix}/data/chatbot_arena_preference_data_validate.tsv"
    # )

    # chatbot_arena = ChatBotArenaDataset(
    #     "chatbot_arena",
    #     f"{prefix}/data/chatbot_arena_preference_data.jsonl",
    #     f"{prefix}/data/chatbot_arena_preference_data.tsv"
    # )
    chatbot_arena = ChatBotArenaDataset("chatbot_arena", f"{prefix}/re.csv", f"{prefix}/data/chatbot_arena_vincuna_chatglm_augmented.tsv")