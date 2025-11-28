from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

# Load model and encoders
model = pickle.load(open("final_model.pkl", "rb"))
le_sex = pickle.load(open("sex_encoder.pkl", "rb"))
le_island = pickle.load(open("island_encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

app = FastAPI(title="Penguin Species Prediction API")

# Request schema
class InputData(BaseModel):
    bill_length: float
    bill_depth: float
    flipper_len: float
    body_mass: float
    sex: str
    island: str


@app.post("/predict")
def predict(data: InputData):

    # Convert to DataFrame
    df = pd.DataFrame([[
        data.island,
        data.bill_length,
        data.bill_depth,
        data.flipper_len,
        data.body_mass,
        data.sex
    ]],
    columns=[
        'island',
        'bill_length_mm',
        'bill_depth_mm',
        'flipper_length_mm',
        'body_mass_g',
        'sex'
    ])
    
    # Encode
    df["sex"] = le_sex.transform(df["sex"])
    df["island"] = le_island.transform(df["island"])

    # Scale
    df = scaler.transform(df)

    # Predict
    prediction = model.predict(df)

    return {"prediction": prediction[0]}
