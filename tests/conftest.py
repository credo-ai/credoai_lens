from typing import Union
from pytest import fixture
from credoai.data import fetch_creditdefault
from sklearn.model_selection import train_test_split
from credoai.artifacts import TabularData, ClassificationModel
from sklearn.ensemble import RandomForestClassifier
from pandas import Series, DataFrame

DATASET_SIZE = 1000


def split_data(df: DataFrame, sensitive_features: Union[DataFrame, Series]) -> dict:
    """
    Split dataset in test and train.

    Parameters
    ----------
    df : DataFrame
        Data set containing both X and target
    sensitive_features : Union[DataFrame, Series]
        Can be 1 or more sensitive features

    Returns
    -------
    dict
        Dictionary of all datasets, targets and sensitive features
    """
    if isinstance(sensitive_features, Series):
        sens_feat_names = [sensitive_features.name]
    else:
        sens_feat_names = list(sensitive_features.columns)
    X = df.drop(columns=sens_feat_names + ["target"])
    y = df["target"]
    (
        X_train,
        X_test,
        y_train,
        y_test,
        sensitive_features_train,
        sensitive_features_test,
    ) = train_test_split(X, y, sensitive_features, random_state=42)
    output = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "sens_feat_train": sensitive_features_train,
        "sens_feat_test": sensitive_features_test,
    }
    return output


@fixture(scope="session")
def data(n=DATASET_SIZE):
    data = fetch_creditdefault()
    df = data["data"].copy().iloc[0:n]
    df["target"] = data["target"].copy().iloc[0:n].astype(int)
    return df


@fixture(scope="session")
def single_sens_feat(data):
    return data["SEX"]


@fixture(scope="session")
def data_for_modeling(data, single_sens_feat):
    return split_data(data, single_sens_feat)


@fixture(scope="session")
def credo_model(data_for_modeling):
    model = RandomForestClassifier()
    model.fit(data_for_modeling["X_train"], data_for_modeling["y_train"])
    credo_model = ClassificationModel("credit_default_classifier", model)
    return credo_model


@fixture(scope="session")
def assessment_data(data_for_modeling):
    return TabularData(
        name="UCI-credit-default-test",
        X=data_for_modeling["X_test"],
        y=data_for_modeling["y_test"],
        sensitive_features=data_for_modeling["sens_feat_test"],
    )


@fixture(scope="session")
def train_data(data_for_modeling):
    return TabularData(
        name="UCI-credit-default-train",
        X=data_for_modeling["X_train"],
        y=data_for_modeling["y_train"],
        sensitive_features=data_for_modeling["sens_feat_train"],
    )
