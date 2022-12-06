from pytest import fixture
from credoai.datasets import fetch_creditdefault, fetch_testdata
from pandas import DataFrame
from sklearn.model_selection import train_test_split

### Datasets definition ########################


@fixture(scope="session")
def binary_data():
    train_data, test_data = fetch_testdata(False, 1, 1)
    return {"train": train_data, "test": test_data}


@fixture(scope="session")
def continuous_data():
    train_data, test_data = fetch_testdata(False, 1, 1, "continuous")
    return {"train": train_data, "test": test_data}


@fixture(scope="session")
def credit_data():
    data = fetch_creditdefault()
    X = data["data"].iloc[0:100]
    X = X.drop(columns=["SEX"])
    y = data["target"].iloc[0:100].astype(int)
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(X, y, random_state=42)
    train_data = {"X": X_train, "y": y_train}
    test_data = {"X": X_test, "y": y_test}
    return {"train": train_data, "test": test_data}


@fixture(scope="session")
def ranking_fairness_data():
    df = DataFrame(
        {
            "rankings": [1, 2, 3, 4, 5, 6, 7, 8],
            "scores": [10, 8, 7, 6, 2, 2, 1, 1],
            "sensitive_features": ["f", "f", "m", "m", "f", "m", "f", "f"],
        }
    )
    return df


@fixture(scope="session")
def identity_verification_data():
    source_subject_id = 4 * ["s0"] + 5 * ["s1"] + ["s2"]
    source_subject_data_sample = 4 * ["s00"] + 3 * ["s10"] + 2 * ["s11"] + ["s20"]
    target_subject_id = ["s1"] + ["s1", "s2", "s3"] * 2 + ["s2", "s3", "s3"]
    target_subject_data_sample = (
        ["s10"] + ["s11", "s20", "s30"] * 2 + ["s20"] + ["s30"] * 2
    )
    pairs = DataFrame(
        {
            "source-subject-id": source_subject_id,
            "source-subject-data-sample": source_subject_data_sample,
            "target-subject-id": target_subject_id,
            "target-subject-data-sample": target_subject_data_sample,
        }
    )

    subjects_sensitive_features = DataFrame(
        {
            "subject-id": ["s0", "s1", "s2", "s3"],
            "gender": ["female", "male", "female", "female"],
        }
    )
    return pairs, subjects_sensitive_features
