from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="PNDM Prediction")

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga el modelo
model_path = 'model/PNDM-prediction-v1.joblib'
model = load(model_path)

# Modelo de entrada
class InputData(BaseModel):
    Age: float = 3
    HbA1c: float = 5.69
    Genetic_Info: int = 1
    Family_History: int = 0
    Birth_Weight: float = 2.06
    Developmental_Delay: int = 0
    Insulin_Level: float = 3.14

# Modelo de salida
class OutputData(BaseModel):
    prediction: int

@app.post("/predict", response_model=OutputData)
def predict(data: InputData):
    model_input = np.array([v for v in data.dict().values()]).reshape(1, -1)
    result = model.predict(model_input)
    return {"prediction": result[0]}