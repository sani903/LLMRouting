import os

import numpy as np
import pandas as pd

if __name__ == "__main__":
    # Load data
    prefix = os.getcwd()
    path = f"{prefix}/data/chatbot_arena_preference_data.tsv"
    data_df = pd.read_csv(path, sep="\t", header=0)

    data_df["prediction"] = np.random.choice([0, 1], size=len(data_df))
    accuracy = (data_df["preference"] == data_df["prediction"]).mean()
    print(f"Accuracy: {accuracy}")

