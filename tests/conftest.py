from pytest import fixture
from credoai.data import fetch_creditdefault
from sklearn.model_selection import train_test_split
from credoai.artifacts import TabularData, ClassificationModel
from sklearn.ensemble import RandomForestClassifier
from pandas import Series


def split_data(df, sensitive_features):
    if isinstance(sensitive_features, Series):
        sens_feat_names = [sensitive_features.name]
    else:
        sens_feat_names = list(sens_feat_names.columns)
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


@fixture(scope="module")
def data(n=100):
    data = fetch_creditdefault()
    df = data["data"].copy().iloc[0:n]
    df["target"] = data["target"].copy().iloc[0:n].astype(int)
    return df


@fixture(scope="module")
def single_sens_feat(data):
    return data["SEX"]


@fixture(scope="module")
def data_for_modeling(data, single_sens_feat):
    return split_data(data, single_sens_feat)


@fixture(scope="module")
def credo_model(data_for_modeling):
    model = RandomForestClassifier()
    model.fit(data_for_modeling["X_train"], data_for_modeling["y_train"])
    credo_model = ClassificationModel("credit_default_classifier", model)
    return credo_model


@fixture(scope="module")
def assessment_data(data_for_modeling):
    return TabularData(
        name="UCI-credit-default-test",
        X=data_for_modeling["X_test"],
        y=data_for_modeling["y_test"],
        sensitive_features=data_for_modeling["sens_feat_test"],
    )


@fixture(scope="module")
def train_data(data_for_modeling):
    return TabularData(
        name="UCI-credit-default-train",
        X=data_for_modeling["X_train"],
        y=data_for_modeling["y_train"],
        sensitive_features=data_for_modeling["sens_feat_train"],
    )
