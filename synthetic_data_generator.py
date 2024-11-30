import random
import pandas as pd
import os

# Helper function to generate synthetic tasks and outputs
def generate_math_reasoning_task():
    task_types = ["arithmetic", "word_problem", "logic"]
    task_type = random.choice(task_types)

    if task_type == "arithmetic":
        a, b = random.randint(1, 100), random.randint(1, 100)
        question = f"What is {a} + {b}?"
        gold = str(a + b)
    elif task_type == "word_problem":
        a, b = random.randint(1, 50), random.randint(1, 50)
        question = f"If John has {a} apples and gives {b} to Sarah, how many does he have left?"
        gold = str(a - b)
    else:  # logic
        question = "If all cats are animals and some animals are dogs, are all cats dogs?"
        gold = "No"

    return question, gold, task_type

# Generate synthetic data
final_data = [("original", "gold_output", "dataset", "metadata", "model_1_output", "model_2_output", "preference")]
for _ in range(1000):
    question, gold, dataset = generate_math_reasoning_task()

    # Metadata: Difficulty and type
    difficulty = random.choice(["easy", "medium", "hard"])
    metadata = f"type={dataset}, difficulty={difficulty}"

    # Generate model outputs (simulate with some randomness)
    model_1_output = gold if random.random() > 0.2 else "incorrect answer"
    model_2_output = "incorrect answer"
    if model_1_output == "incorrect answer":
        model_2_output = gold

    # Preference: Which model is better (0 = neither, 1 = Model 1, 2 = Model 2)
    if model_1_output == gold and model_2_output != gold:
        preference = 1
    elif model_2_output == gold and model_1_output != gold:
        preference = 2
    elif model_1_output == model_2_output == gold:
        preference = random.choice([1, 2])  # Tie-breaking
    else:
        preference = 0

    final_data.append((question, gold, dataset, metadata, model_1_output, model_2_output, preference))

prefix = os.getcwd()
data_df = pd.DataFrame(final_data)
data_df.to_csv(f"{prefix}/data/synthetic_mixed_preference_data", index=False, header=0, sep="\t")
