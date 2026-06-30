import pandas as pd
import numpy as np
import joblib
import os

from feature_engineering import full_feature_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

MODEL_EXP = "artifacts/model_exp.pkl"
FINAL_MODEL = "artifacts/model.pkl"
THRESHOLD_FILE = "artifacts/threshold.pkl"
os.makedirs("artifacts", exist_ok=True)

df = pd.read_csv("../data/Fraud.csv")

df = full_feature_pipeline(df)

x = df.drop(columns="isFraud")
y = df['isFraud']

x_train, x_temp, y_train, y_temp = train_test_split(x , y , test_size=0.2 , stratify=y , random_state=42)
x_test, x_val, y_test, y_val = train_test_split(x_temp , y_temp , test_size=0.5, stratify=y_temp , random_state=42)

model = joblib.load(MODEL_EXP)


if not os.path.exists(FINAL_MODEL):
    model.fit(x_train , y_train)
    y_val_prob = model.predict_proba(x_val)[:, 1]

    thresholds = np.arange(0.1, 0.91, 0.01)

    best_threshold = 0.5
    best_f1 = 0

    for threshold in thresholds:

        y_val_pred = (y_val_prob >= threshold).astype(int)

        score = f1_score(y_val, y_val_pred)

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    joblib.dump(best_threshold , THRESHOLD_FILE)
    joblib.dump(model , FINAL_MODEL)
    print(f"Best f1 score : {best_f1}")
    print(f"Best threshold : {best_threshold}")
    

else:
    final_model = joblib.load(FINAL_MODEL)
    threshold = joblib.load(THRESHOLD_FILE)

    y_pred = (final_model.predict_proba(x_test)[:, 1] >= threshold).astype(int)

    print(f"Test set recall score : {recall_score(y_test , y_pred)}")
    print(f"Test set precision score : {precision_score(y_test , y_pred)}")
    print(f"Test set f1 score : {f1_score(y_test , y_pred)}")



