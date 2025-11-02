from flask import Flask, request, render_template
import joblib
import pandas as pd
import random

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("best_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # All features the model expects (V1..V28 + normalized_amount + normalized_time)
        all_features = [f"V{i}" for i in range(1, 29)] + ["normalized_amount", "normalized_time"]

        # Initialize all features as 0.0
        input_data = {col: 0.0 for col in all_features}

        # --- 1) Fill V1-V4 from user input (if provided) ---
        for key in ["V1", "V2", "V3", "V4"]:
            value = request.form.get(key)
            if value and value.strip() != "":
                try:
                    input_data[key] = float(value)
                except ValueError:
                    # if parse fails, keep default 0.0
                    input_data[key] = 0.0

        # --- 2) Generate V5-V28 automatically with alternating signs ---
        # Choose starting sign randomly: +1 or -1
        start_sign = random.choice([1, -1])
        sign = start_sign

        # magnitude range for generated PCA-like values (adjustable)
        mag_min, mag_max = 0.5, 8.0

        # fill V5..V28
        for i in range(5, 29):  # 5..28 inclusive
            magnitude = random.uniform(mag_min, mag_max)
            input_data[f"V{i}"] = sign * magnitude
            sign *= -1  # alternate sign for next feature

        # --- 3) Handle Amount and Time from user form (default 0.0 if missing) ---
        try:
            amount = float(request.form.get("Amount")) if request.form.get("Amount") else 0.0
        except ValueError:
            amount = 0.0

        try:
            time = float(request.form.get("Time")) if request.form.get("Time") else 0.0
        except ValueError:
            time = 0.0

        # --- 4) Normalize Amount and Time to match training ---
        # Replace these constants with the exact scaler.mean_ and scaler.scale_ from your training for best results
        amount_mean, amount_std = 88.35, 250.12   # <-- placeholder values; replace if you have real ones
        time_mean, time_std = 47000.0, 29000.0    # <-- placeholder values; replace if you have real ones

        # avoid division by zero
        if amount_std == 0:
            input_data["normalized_amount"] = 0.0
        else:
            input_data["normalized_amount"] = (amount - amount_mean) / amount_std

        if time_std == 0:
            input_data["normalized_time"] = 0.0
        else:
            input_data["normalized_time"] = (time - time_mean) / time_std

        # --- 5) Build dataframe in correct order ---
        input_df = pd.DataFrame([input_data], columns=all_features)

        # --- 6) Predict ---
        prob = None
        try:
            prob = model.predict_proba(input_df)[0][1] * 100  # fraud probability %
        except AttributeError:
            # model may not implement predict_proba (rare); fall back to predict
            pred_only = model.predict(input_df)[0]
            prob = 100.0 if pred_only == 1 else 0.0

        # Use probability threshold to decide label (you can adjust threshold if needed)
        threshold = 50.0  # percent
        if prob >= threshold:
            label = f"⚠ Fraudulent Transaction (Risk: {prob:.2f}%)"
            color = "red"
        else:
            label = f"✅ Legitimate Transaction (Risk: {prob:.2f}%)"
            color = "green"

        # Optionally include generated features in the UI (not implemented in template)
        return render_template("index.html", prediction_text=label, color=color)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}", color="black")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
