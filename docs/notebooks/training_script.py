# imports for example data and model training
from credoai.data import fetch_creditdefault
from sklearn.ensemble import GradientBoostingClassifier

data = fetch_creditdefault()
df = data['data']
df['target'] = data['target'].astype(int)

# fit model
model = GradientBoostingClassifier()
X = df.drop(columns=['SEX', 'target'])
y = df['target']
model.fit(X,y)