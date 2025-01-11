import pandas as pd

# Read the CSV files
llama_df = pd.read_csv('chatglm_prefernce_chatarena.csv')
gpt_df = pd.read_csv('diff1.csv')

# Create dictionaries mapping prompts to their diffs
llama_map = llama_df.drop_duplicates(subset='prompts').set_index('prompts').to_dict(orient='index')
gpt_map = gpt_df.drop_duplicates(subset='prompts').set_index('prompts').to_dict(orient='index')

# Create a set of all unique prompts
all_prompts = set(llama_map.keys()).union(set(gpt_map.keys()))

# Prepare the result DataFrame
result_data = []

for prompt in all_prompts:
    strong = llama_map.get(prompt, {}).get('diff', None)  # Get diff from llama_map
    weak = gpt_map.get(prompt, {}).get('diff', None)  # Get diff from gpt_map
    if strong is not None and weak is not None:
        result_data.append({'prompts': prompt, 'strong': strong, 'weak': weak})

# Convert result to DataFrame
result_df = pd.DataFrame(result_data)

# Save the result to a new CSV file
result_df.to_csv('re.csv', index=False)

# Print some information about the result
print(f"Number of rows in result: {len(result_df)}")
print(f"Columns in result: {result_df.columns.tolist()}")
