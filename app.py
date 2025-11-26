from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and encoders
model = pickle.load(open("final_model.pkl", "rb"))
le_sex = pickle.load(open("sex_encoder.pkl", "rb"))
le_island = pickle.load(open("island_encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl",'rb'))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        bill_length = float(request.form["bill_length"])
        bill_depth = float(request.form["bill_depth"])
        flipper_len = float(request.form["flipper_len"])
        body_mass = float(request.form["body_mass"])
        sex_input = request.form["sex"]
        island_input = request.form["island"]

        # Create dataframe for prediction
        input_data = pd.DataFrame([[
        island_input,bill_length, bill_depth, flipper_len, body_mass,sex_input]],
        columns=['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
       'body_mass_g', 'sex'])

        # Encode user inputs
        input_data["sex"] = le_sex.transform(input_data["sex"])
        input_data["island"] = le_island.transform(input_data["island"])

        #scaling
        input_data=scaler.transform(input_data)
        
        # Predict
        result = model.predict(input_data)
        prediction = result[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
