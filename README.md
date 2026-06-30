# 🚨 Fraudulent Transaction Detection System

## 📌 Overview

This project is an end-to-end machine learning system for detecting fraudulent financial transactions.
It handles highly imbalanced data (~0.1% fraud rate) and focuses on optimizing both recall and precision using advanced models and threshold tuning.

The system includes:

- Feature engineering pipeline
- Model training and hyperparameter tuning
- Class imbalance handling
- Threshold optimization for classification
- Flask-based prediction API

## 📊 Dataset

The dataset used is large (~millions of transactions) and is not included in this repository.

You can access it here: [Fraud Transaction Dataset](https://www.kaggle.com/datasets/chitwanmanchanda/fraudulent-transactions-data)

**Dataset Characteristics:**

- Highly imbalanced: ~0.1% fraud cases
- Financial transaction records
- Binary classification problem (Fraud / Not Fraud)

## 🏗️ Project Architecture

`Data` → `Feature Engineering` → `Preprocessing` → `Model Training` → `Threshold Tuning` → `Evaluation` → `API Deployment`

## 🧪 Models Experimented

Several models were tested using cross-validation:

| Model               | F1 Score (CV)                     |
| ------------------- | --------------------------------- |
| Logistic Regression | ~0.05 (very low due to imbalance) |
| Decision Tree       | ~0.66                             |
| Random Forest       | ~0.71                             |
| XGBoost             | ~0.85 (best performing)           |

### 📉 Initial Results (Before Optimization)

High recall but extremely low precision (many false positives)
Example:

- **Recall:** ~0.99
- **Precision:** ~0.005
- **F1 Score:** ~0.01

👉 Model was biased towards predicting fraud too often.

### ⚙️ Optimization Techniques Applied

- SMOTE / class balancing experiments
- `scale_pos_weight` tuning (XGBoost)
- Hyperparameter tuning using `RandomizedSearchCV`
- Threshold tuning (0.1 → 0.9)
- F1-score optimization instead of accuracy

## 🚀 Final Model Performance

After optimization and threshold tuning:

| Metric    | Test Score  |
| --------- | ----------  |
| Recall    | 0.9233      |
| Precision | 0.8784      |
| F1 Score  | 0.9003      |

✔ Balanced high recall and precision
✔ Significant improvement over baseline models
✔ Suitable for real-world fraud detection use cases

## 🧠 Key Insights

- Accuracy is misleading for imbalanced datasets
- Recall alone is not giving good results.
- F1 score is more reliable for balancing precision & recall
- Threshold tuning is critical in fraud detection systems
- XGBoost significantly outperformed traditional models

## 🛠️ Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE experiments)
- Flask (API)
- Joblib (model serialization)

## 📡 API Endpoint

### Predict Fraud

**POST** `/predict`

**Request:**

```json
{
  "type": "CASH_OUT",
  "amount": 1200.5,
  "nameOrig": "C123456789",
  "oldbalanceOrg": 5000.0,
  "newbalanceOrig": 3800.5,
  "nameDest": "M987654321",
  "oldbalanceDest": 10000.0,
  "newbalanceDest": 11200.5
}
```

**Response:**

```json
{
  "status": "success",
  "prediction": [0],
  "probability": [0.09]
}
```

## 📁 Project Structure

```text
fraud-detection-system/
│
│
├──  app.py
├──  routes.py
│
├── ml/
│   ├── predict.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── model_experimentation.py
│   ├── hyperparameter_tuning.py
│   ├── preprocessing.py
│
├── artifacts/
│   ├── model.pkl
│   ├── model_exp.pkl
│   ├── threshold.pkl
│
├── data/ (ignored, size is too large)
├── notebooks/
└── README.md
```

## 📌 Future Improvements

- Deploy on cloud (AWS / Render / GCP)
- Add monitoring for drift detection
- Switch to FastAPI for better performance
- Add database logging for predictions

## 🏁 Conclusion

This project demonstrates a full ML lifecycle:

From raw imbalanced data → optimized fraud detection model → deployable API system
