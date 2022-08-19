import numpy as np
import pandas as pd


def hiring_probabilities(x):
    prob = 0.4 if x.gender == "f" else 0.7
    prob += 0.2 if x.race == "b" else -0.1
    return prob


def fetch_testdata(add_nan=False, train_repeats=30, test_repeats=30):
    """Returns testing data for Lens"""
    RNG = np.random.RandomState(1)

    train_df = pd.DataFrame(
        {
            "gender": ["f", "f", "f", "f", "f", "f", "m", "m", "m", "m", "m", "m"],
            "race": ["b", "b", "b", "w", "w", "w", "b", "b", "b", "w", "w", "w"],
            "experience": [0, 0.3, 0.1, 0.8, 0.1, 0.6, 0, 0.1, 0.2, 0.4, 0.5, 0.6],
        }
    )
    train_df = pd.concat([train_df] * train_repeats).reset_index(drop=True)
    train_df["experience"] += RNG.normal(scale=0.05, size=len(train_df))
    test_df = pd.DataFrame(
        {
            "gender": ["f", "f", "f", "f", "f", "f", "m", "m", "m", "m", "m", "m"],
            "race": ["b", "b", "b", "w", "w", "w", "b", "b", "b", "w", "w", "w"],
            "experience": [0.4, 0.1, 0.4, 0.3, 0.9, 0.8, 0.4, 0.7, 0.5, 0.6, 0.2, 0.8],
        }
    )
    test_df = pd.concat([test_df] * test_repeats).reset_index(drop=True)
    test_df["experience"] += RNG.normal(scale=0.05, size=len(test_df))

    if add_nan:
        train_df = train_df.mask(RNG.random(train_df.shape) < 0.1)
        test_df = test_df.mask(RNG.random(test_df.shape) < 0.1)

    train_df["hired"] = (
        RNG.rand(len(train_df)) < train_df.apply(hiring_probabilities, axis=1)
    ).astype(int)
    test_df["hired"] = (
        RNG.rand(len(test_df)) < test_df.apply(hiring_probabilities, axis=1)
    ).astype(int)
    train_data = {
        "X": train_df[["experience"]],
        "y": train_df["hired"],
        "sensitive_features": train_df[["race", "gender"]],
    }
    test_data = {
        "X": test_df[["experience"]],
        "y": test_df["hired"],
        "sensitive_features": test_df[["race", "gender"]],
    }
    return train_data, test_data
