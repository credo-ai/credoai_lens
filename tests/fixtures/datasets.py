"""
Contains all fixtures related to dataset creation.

Any fixture added to the file will be immediately available due to
addition of this module as plugin in the file `pytest.ini`.
"""
from pandas import DataFrame, Series
from pytest import fixture
from sklearn import datasets
from sklearn.model_selection import train_test_split

from credoai.datasets import fetch_creditdefault, fetch_testdata

from tensorflow.keras.datasets.mnist import load_data as load_mnist

### Datasets definition ########################


@fixture(scope="session")
def binary_data():
    train_data, test_data = fetch_testdata(False, 1, 1)
    return {"train": train_data, "test": test_data}


@fixture(scope="session")
def mnist_data():
    (x_train, y_train), (x_test, y_test) = load_mnist()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test, y_train, y_test = (
        x_train[:9600],
        x_test[:640],
        y_train[:9600],
        y_test[:640],
    )
    train_data = {"X": x_train, "y": y_train}
    test_data = {"X": x_test, "y": y_train}
    return {"train": train_data, "test": test_data}


@fixture(scope="session")
def continuous_data():
    train_data, test_data = fetch_testdata(False, 1, 1, "continuous")
    return {"train": train_data, "test": test_data}


@fixture(scope="session")
def credit_data():
    data = fetch_creditdefault()
    X = data["data"].iloc[0:100]
    sens_feat = X["SEX"].copy()
    X = X.drop(columns=["SEX"])
    y = data["target"].iloc[0:100].astype(int)
    y.name = "target"
    (
        X_train,
        X_test,
        y_train,
        y_test,
        sens_feat_train,
        sens_feat_test,
    ) = train_test_split(X, y, sens_feat, random_state=42)
    train_data = {"X": X_train, "y": y_train, "sens_feat": sens_feat_train}
    test_data = {"X": X_test, "y": y_test, "sens_feat": sens_feat_test}
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


@fixture(scope="session")
def multiclass_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    sensitive_features = (
        Series(["B", "W"])
        .sample(X.shape[0], replace=True, random_state=42)
        .reset_index(drop=True)
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        sensitive_features_train,
        sensitive_features_test,
    ) = train_test_split(X, y, sensitive_features, random_state=42)
    train_data = {"X": X_train, "y": y_train, "sens_features": sensitive_features_train}
    test_data = {"X": X_test, "y": y_test, "sens_features": sensitive_features_test}
    return {"train": train_data, "test": test_data}
