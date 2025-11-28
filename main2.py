from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import pickle

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Load model and encoders
model = pickle.load(open("final_model.pkl", "rb"))
le_sex = pickle.load(open("sex_encoder.pkl", "rb"))
le_island = pickle.load(open("island_encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": ""})


@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    bill_length: float = Form(...),
    bill_depth: float = Form(...),
    flipper_len: float = Form(...),
    body_mass: float = Form(...),
    sex: str = Form(...),
    island: str = Form(...)
):

    # DataFrame
    df = pd.DataFrame([[
        island,
        bill_length,
        bill_depth,
        flipper_len,
        body_mass,
        sex
    ]],
    columns=[
        'island', 'bill_length_mm', 'bill_depth_mm',
        'flipper_length_mm', 'body_mass_g', 'sex'
    ])

    # Encode
    df["sex"] = le_sex.transform(df["sex"])
    df["island"] = le_island.transform(df["island"])

    # Scale
    df = scaler.transform(df)

    # Predict
    result = model.predict(df)[0]

    return templates.TemplateResponse("index.html",{"request": request, "prediction": result})


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", reload=True)
