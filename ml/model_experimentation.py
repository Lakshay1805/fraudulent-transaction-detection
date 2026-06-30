import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from feature_engineering import full_feature_pipeline
from preprocessing import full_process_pipeline
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score, train_test_split

df = pd.read_csv("../data/Fraud.csv")

df = full_feature_pipeline(df)

x = df.drop(columns="isFraud")
y = df['isFraud']

num_cols = x.select_dtypes(include="number").columns.tolist()
cat_cols = x.select_dtypes(include=["string" , "object" , "category"]).columns.tolist()

preprocessor = full_process_pipeline(num_cols , cat_cols)

x_sample ,_ , y_sample, _ = train_test_split(x , y , train_size=300000 , stratify=y , random_state=42)

lor_pipeline = Pipeline([
    ("process" , preprocessor),
    ("imbalancer" , SMOTE()),
    ("model" , LogisticRegression())
])

lor_cv_score = cross_val_score(lor_pipeline , x_sample , y_sample , cv = 5 , scoring="f1" , n_jobs=6)
print(f"LOR f1 mean : {lor_cv_score.mean()}")
print(f"Std f1  : {lor_cv_score.std():.4f}")
# LOR f1 mean : 0.0514760785408794
# Std f1  : 0.0026

dt_pipeline = Pipeline([
    ("process" , preprocessor),
    ("imbalancer" , SMOTE()),
    ("model" , DecisionTreeClassifier())
])

dt_cv_score = cross_val_score(dt_pipeline , x_sample , y_sample , cv=5 , scoring="f1" , n_jobs=-1)
print(f"DT f1 mean : {dt_cv_score.mean()}")
print(f"Std f1  : {dt_cv_score.std():.4f}")
# DT f1 mean : 0.6617096715177738
# Std f1  : 0.0287




rf_pipeline = Pipeline([
    ("process" , preprocessor),
    ("imbalancer" , SMOTE(random_state=42)),
    ("model" , RandomForestClassifier(n_estimators=50 , random_state=42))
])

rf_cv_score = cross_val_score(rf_pipeline , x_sample , y_sample , cv=3 , scoring="f1" , n_jobs=-1)
print(f"RF f1 mean : {rf_cv_score.mean()}")
print(f"Std f1  : {rf_cv_score.std():.4f}")




scale_pos_weight = len(y[y == 0]) / len(y[y == 1])

xg_pipeline = Pipeline([
    ("process" , preprocessor),
    ("model" , XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42,n_estimators=100))
])

xg_cv_score = cross_val_score(xg_pipeline , x_sample , y_sample , cv=5 , scoring="f1" , n_jobs=-1)
print(f"XGB f1 mean : {xg_cv_score.mean()}")
print(f"Std f1  : {xg_cv_score.std():.4f}")
# XGB f1 mean : 0.8526056841047828
# Std f1  : 0.0218


# We gonna select top model with high f1_score mean and perform hyperparameter tuning to get high recall with balanced precision.