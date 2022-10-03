# imports for example data and model training
from credoai.datasets import fetch_creditdefault
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = fetch_creditdefault()
df = data["data"]
df["target"] = data["target"].astype(int)

# fit model
model = RandomForestClassifier(random_state=42)
X = df.drop(columns=["SEX", "target"])
y = df["target"]
sensitive_features = df["SEX"]
(
    X_train,
    X_test,
    y_train,
    y_test,
    sensitive_features_train,
    sensitive_features_test,
) = train_test_split(X, y, sensitive_features, random_state=42)
model.fit(X_train, y_train)
