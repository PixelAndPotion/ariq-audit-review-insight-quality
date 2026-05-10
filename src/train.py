# src/train.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data (free from UCI / OpenML)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
cols = ['age','workclass','fnlwgt','education','education-num',
        'marital-status','occupation','relationship',
        'race','sex','capital-gain','capital-loss',
        'hours-per-week','native-country','income']

df = pd.read_csv(url, names=cols, skipinitialspace=True)

# Clean
df['income'] = (df['income'] == '>50K').astype(int)

# Encode categoricals
le = LabelEncoder()
cat_cols = df.select_dtypes('object').columns
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Split — keep sensitive columns for audit later
sensitive = df[['race', 'sex']]
X = df.drop(['income', 'race', 'sex'], axis=1)
y = df['income']

X_train, X_test, y_train, y_test, s_train, s_test = \
    train_test_split(X, y, sensitive, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save everything
import pickle, os
os.makedirs('models', exist_ok=True)
pickle.dump((model, X_test, y_test, s_test), open('models/model.pkl','wb'))
print("Model trained and saved.")