
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('diabetes.csv')
cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
df[cols] = df[cols].replace(0, np.nan)
df.fillna(df.median(numeric_only=True), inplace=True)

X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=2, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'diabetes_model.pkl')

medians = df[X.columns].median()
joblib.dump(medians, 'feature_medians.pkl')

# verify
m = joblib.load('diabetes_model.pkl')
print('Model saved OK, estimators:', len(m.estimators_))
print('Medians:', medians.round(2).to_dict())
