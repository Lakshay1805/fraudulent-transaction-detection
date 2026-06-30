import pandas as pd
import warnings
import joblib
import os

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from feature_engineering import full_feature_pipeline
from preprocessing import full_process_pipeline

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold


df = pd.read_csv("../data/Fraud.csv")

ARTIFACT_DIR = "artifacts"
MODEL_FILE = f"{ARTIFACT_DIR}/model_exp.pkl"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

df = full_feature_pipeline(df)

x = df.drop(columns="isFraud")
y = df['isFraud']

num_cols = x.select_dtypes(include="number").columns.tolist()
cat_cols = x.select_dtypes(include=["string" , "object" , "category"]).columns.tolist()

preprocessor = full_process_pipeline(num_cols , cat_cols)

x_sample, _, y_sample, _ = train_test_split(x , y , train_size=500000 ,stratify=y , random_state=42)

skf = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

param_dist_xgb = {
    "model__n_estimators": [100, 200, 300, 500],
    
    "model__max_depth": [3, 4, 5, 6, 8],
    
    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
    
    "model__subsample": [0.7, 0.8, 0.9, 1.0],
    
    "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    
    "model__min_child_weight": [1, 3, 5, 10],
    
    "model__gamma": [0, 0.1, 0.2, 0.5],
    
    "model__reg_alpha": [0, 0.1, 0.5, 1],
    
    "model__reg_lambda": [1, 1.5, 2, 5]
}

scale_pos_weight = len(y[y == 0]) / len(y[y == 1])

xgb_pipeline = Pipeline([
    ("process" , preprocessor),
    ("model" , XGBClassifier(scale_pos_weight=scale_pos_weight , random_state=42, n_jobs=-1, tree_method="hist"))
])

rand_xg = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=param_dist_xgb,
    n_iter=25,          
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=2,
    random_state=42
)

rand_xg.fit(x_sample , y_sample)

print(f"Best score : {rand_xg.best_score_}")
print(rand_xg.best_params_)
joblib.dump(rand_xg.best_estimator_ , MODEL_FILE)