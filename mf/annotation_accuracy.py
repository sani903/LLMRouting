import pandas as pd

# Load the CSV file
df = pd.read_csv('evaluated_responses.csv')

# Create an empty label column
df['label'] = None

# Calculate the 'label' column based on the difference between 'strong' and 'weak'
df['label'] = df.apply(lambda row: 0 if (row['strong'] - row['weak']) < 0 else 1, axis=1)

# Filter rows where abs(strong - weak) > 0.5
valid_diff_mask = (df['strong'] - df['weak']).abs() > 0
filtered_df = df[valid_diff_mask]

# Filter out rows where preferred_model is -1 for calculations involving preferred_model
valid_preferred_model_mask = filtered_df['preferred_model'] != -1
filtered_preferred_df = filtered_df[valid_preferred_model_mask]

# Calculate accuracy for label vs preferred_model
accuracy_label_preferred = (filtered_preferred_df['label'] == filtered_preferred_df['preferred_model']).mean()

# Calculate accuracy for annotation vs label, skipping rows with -1 or empty annotation
valid_annotation_mask = (filtered_df['annotation'] != -1) & (filtered_df['annotation'].notna())
accuracy_annotation_label = (filtered_df[valid_annotation_mask]['annotation'] == filtered_df[valid_annotation_mask]['label']).mean()

# Calculate accuracy for annotation vs preferred_model, using only rows where preferred_model is valid
valid_annotation_preferred_mask = valid_annotation_mask & valid_preferred_model_mask
accuracy_annotation_preferred = (filtered_df[valid_annotation_preferred_mask]['annotation'] == filtered_df[valid_annotation_preferred_mask]['preferred_model']).mean()

# Print the results
print(f"Accuracy (label vs preferred_model): {accuracy_label_preferred:.2%}")
print(f"Accuracy (annotation vs label): {accuracy_annotation_label:.2%}")
print(f"Accuracy (annotation vs preferred_model): {accuracy_annotation_preferred:.2%}")
