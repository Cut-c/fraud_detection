from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("best_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form data
        features = [
            float(request.form.get("V1")),
            float(request.form.get("V2")),
            float(request.form.get("V3")),
            float(request.form.get("V4")),
            float(request.form.get("Amount")),
            float(request.form.get("Time"))
        ]
        
        input_df = pd.DataFrame([features], columns=["V1", "V2", "V3", "V4", "Amount", "Time"])
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1] * 100

        if pred == 1:
            label = f"⚠ Fraudulent Transaction (Risk: {prob:.2f}%)"
            color = "red"
        else:
            label = f"✅ Legitimate Transaction (Risk: {prob:.2f}%)"
            color = "green"

        return render_template("index.html", prediction_text=label, color=color)
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}", color="black")

if _name_ == "_main_":
    app.run(host="0.0.0.0", port=8080)
