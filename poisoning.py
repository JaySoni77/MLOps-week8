import argparse
import pandas as pd
import numpy as np
import os

def poison_data(data_path, percent):
    df = pd.read_csv(data_path)

    if percent <= 0 or percent > 100:
        raise ValueError("Percent must be between 1 and 100")

    n = len(df)
    n_poison = int((percent/100) * n)

    print(f"Poisoning {n_poison}/{n} rows (~{percent}%)")

    # unique labels
    labels = df["species"].unique()

    # choose random rows to poison
    indices = np.random.choice(df.index, size=n_poison, replace=False)

    for idx in indices:
        current = df.at[idx, "species"]
        # choose any other label except current
        new_label = np.random.choice(labels[labels != current])
        df.at[idx, "species"] = new_label

    # save new CSV
    fname = f"iris_poisoned_{percent}.csv"
    out_path = os.path.join("data", fname)
    df.to_csv(out_path, index=False)

    print(f"Poisoned dataset saved at: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--percent", type=float, required=True)

    args = parser.parse_args()

    poison_data(args.data_path, args.percent)
