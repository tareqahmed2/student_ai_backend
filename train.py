# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
import joblib
import json
from pathlib import Path

# --- Artifacts directory ---
ARTIFACTS_DIR = Path(__file__).parent / 'artifacts'
ARTIFACTS_DIR.mkdir(exist_ok=True)

# --- Dummy student data ---
df = pd.DataFrame({
    "hours_study": [1.5, 4.2, 2, 3.5, 0.5, 5],
    "attendance": [80, 90, 70, 85, 60, 95],
    "sleep_hours": [6, 7, 6, 8, 5, 7],
    "internet_hours": [2, 1, 3, 2, 4, 1],
    "previous_score": [54, 91, 68, 82, 50, 88],
    "gender": ["Female","Male","Male","Female","Female","Male"],
    "parent_education": ["Bachelor","Bachelor","HighSchool","Master","HighSchool","Master"]
})

# --- Rename column to match API ---
df.rename(columns={"previous_score": "past_score"}, inplace=True)

# --- Target ---
df["target"] = df["past_score"].apply(lambda x: "Pass" if x >= 60 else "Fail")

# --- Features ---
numeric_features = ["hours_study", "attendance", "sleep_hours", "internet_hours", "past_score"]
categorical_features = ["gender", "parent_education"]

X = df[numeric_features + categorical_features]
y = df["target"]

# --- Preprocessing pipeline ---
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('kbest', SelectKBest(score_func=f_classif, k=min(6, len(X.columns)))),
    ('clf', RandomForestClassifier(random_state=42))
])

# --- Train/test split (no stratify to avoid small dataset issue) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# --- Grid search ---
param_grid = {'clf__n_estimators':[100, 200], 'clf__max_depth':[None, 8]}
grid = GridSearchCV(pipe, param_grid, cv=2, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

# --- Save model ---
best_model = grid.best_estimator_
joblib.dump(best_model, ARTIFACTS_DIR / 'model.joblib')

# --- Save metadata ---
meta = {"numeric_features": numeric_features, "categorical_features": categorical_features}
with open(ARTIFACTS_DIR / 'meta.json','w') as f:
    json.dump(meta,f)

print("Training done, model and meta.json saved.")
