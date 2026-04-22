import pandas as pd 
import numpy as np 
import pickle as pkl
from sklearn.model_selection import train_test_split
import mlflow
from src.features.build_features import build_features, preprocessor
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

mlflow.set_tracking_uri("mlruns")

df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# calling function
X, y = build_features(df)

# train test splits

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# pipeline

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(scale_pos_weight = 2.7, random_state=42, eval_metric =  'logloss' ) )
])


mlflow.set_experiment("finchurn")

with mlflow.start_run():
    # fit the model
    model.fit(X_train, y_train)
    # predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    # log metrics:
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))  
    
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred_proba))
    mlflow.log_param("scale_pos_weight", 2.7)
    mlflow.sklearn.log_model(model, "model")
    
with open("models/model.pkl", "wb") as f:
    pkl.dump(model, f)