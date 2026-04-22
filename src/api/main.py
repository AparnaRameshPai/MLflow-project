from pydantic import BaseModel
import pickle as pkl
import numpy as np
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

with open("models/model.pkl", "rb") as f:
    model = pkl.load(f)

class Customer(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines:str
    OnlineSecurity:str
    OnlineBackup :str
    DeviceProtection:str
    TechSupport:str
    StreamingTV:str
    StreamingMovies:str
    Contract:str
    PaperlessBilling:str
    PaymentMethod:str
    InternetService: str
    
@app.post("/predict")
def predict(customer: Customer):
    df = pd.DataFrame([customer.model_dump()])
    
    # 2. run model.predict_proba
    proba = model.predict_proba(df)[:, 1][0]
    # 3. return the probability
    return {"churn_probability": float(proba), "will_churn": bool(proba > 0.5)}