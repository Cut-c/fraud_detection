from flask import Flask, request, render_template
import joblib
import pandas as pd
import hashlib
import random
import csv
import os
from datetime import datetime

app = Flask(__name__)

# Load trained model (must be in repo root or adjust path)
model = joblib.load("best_model.pkl")

# Threshold (percent) to mark as Fraudulent on UI
FRAUD_THRESHOLD = 30.0  # show Fraudulent when probability >= 30%

# Normalization constants - replace with exact scaler.mean_ and scaler.scale_ from your training
# These values seem to be placeholders or averages.
# For the app to be accurate, these MUST match the scalers used to train "best_model.pkl"
AMOUNT_MEAN, AMOUNT_STD = 88.35, 250.12
TIME_MEAN, TIME_STD = 47000.0, 29000.0  # These are very round numbers, suspect they are wrong

# Helper to write audit log
def log_prediction(record: dict):
    logfile = "pred_history.csv"
    header = list(record.keys())
    first = not os.path.exists(logfile)
    with open(logfile, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if first:
            writer.writeheader()
        writer.writerow(record)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Model expects V1..V28 and normalized_amount, normalized_time
        feature_names = [f"V{i}" for i in range(1, 29)] + ["normalized_amount", "normalized_time"]
        input_data = {fn: 0.0 for fn in feature_names}

        # Read V1-V4 from form
        for k in ["V1", "V2", "V3", "V4"]:
            v = request.form.get(k)
            if v and v.strip() != "":
                try:
                    input_data[k] = float(v)
                except ValueError:
                    input_data[k] = 0.0

        # Read raw amount & time
        try:
            amount = float(request.form.get("Amount")) if request.form.get("Amount") else 0.0
        except ValueError:
            amount = 0.0
        try:
            time_val = float(request.form.get("Time")) if request.form.get("Time") else 0.0
        except ValueError:
            time_val = 0.0

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !! BRUTAL, REAL PROBLEM IS HERE !!
        # This code IGNORES the real V5-V28 values and GENERATES FAKE RANDOM DATA.
        # This is why your app gives a different prediction than your CSV.
        # You are sending 24 columns of GARBAGE to your model.
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        seed_str = ",".join(str(input_data[f"V{i}"]) for i in range(1,5)) + f",{amount},{time_val}"
        seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32)
        rand = random.Random(seed)

        start_sign = rand.choice([1, -1])
        sign = start_sign
        mag_min, mag_max = 0.5, 8.0
        for i in range(5, 29):
            mag = rand.uniform(mag_min, mag_max)
            input_data[f"V{i}"] = sign * mag
            sign *= -1

        # Normalize amount and time (use same scaling as training)
        if AMOUNT_STD != 0:
            input_data["normalized_amount"] = (amount - AMOUNT_MEAN) / AMOUNT_STD
        else:
            input_data["normalized_amount"] = 0.0
        if TIME_STD != 0:
            input_data["normalized_time"] = (time_val - TIME_MEAN) / TIME_STD
        else:
            input_data["normalized_time"] = 0.0

        # Create dataframe in the exact column order expected by the model
        input_df = pd.DataFrame([input_data], columns=feature_names)

        # Get probability (if model supports predict_proba)
        try:
            prob = model.predict_proba(input_df)[0][1] * 100.0  # percent
        except Exception:
            # fallback if model has no predict_proba
            pred = model.predict(input_df)[0]
            prob = 100.0 if pred == 1 else 0.0

        # Decide label based on FRAUD_THRESHOLD
        if prob >= FRAUD_THRESHOLD:
            label = f"⚠ Fraudulent Transaction (Risk: {prob:.2f}%)"
            color = "red"
            is_fraud = True
        else:
            label = f"✅ Legitimate Transaction (Risk: {prob:.2f}%)"
            color = "green"
            is_fraud = False

        # Prepare generated features for display (V5..V28)
        generated = {f"V{i}": input_data[f"V{i}"] for i in range(5, 29)}

        # Log prediction for audit
        # --- THIS IS THE FIX for the SyntaxError ---
        # Removed the extra curly braces {{ ... }}
        log_record = {"timestamp": datetime.utcnow().isoformat(), "V1": input_data["V1"],
                      "V2": input_data["V2"],
                      "V3": input_data["V3"], "V4": input_data["V4"], "Amount": amount, "Time": time_val,
                      "probability": f"{prob:.4f}", "label": ("Fraud" if is_fraud else "Legit"),
                      **generated}
        # -------------------------------------------
        
        # write log (non-blocking is fine here; small writes)
        log_prediction(log_record)

        return render_template("index.html",
                               prediction_text=label,
                               color=color,
                               probability=f"{prob:.2f}",
                               generated_features=generated)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}", color="black")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
