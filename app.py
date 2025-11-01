from flask import Flask, request, render_template
import joblib
import pandas as pd

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
        # All features the model expects (V1–V28 + normalized_amount + normalized_time)
        all_features = [f"V{i}" for i in range(1, 29)] + ["normalized_amount", "normalized_time"]

        # Initialize all features as 0.0
        input_data = {col: 0.0 for col in all_features}

        # Fill in only the provided ones (V1–V4)
        for key in ["V1", "V2", "V3", "V4"]:
            value = request.form.get(key)
            if value and value.strip() != "":
                input_data[key] = float(value)

        # Get raw amount and time from form
        amount = float(request.form.get("Amount"))
        time = float(request.form.get("Time"))

        # Normalize them to match model’s expected input format
        # (Replace mean/std with exact ones used during training if available)
        amount_mean, amount_std = 88.35, 250.12
        time_mean, time_std = 47000, 29000

        input_data["normalized_amount"] = (amount - amount_mean) / amount_std
        input_data["normalized_time"] = (time - time_mean) / time_std

        # Convert to DataFrame in the correct feature order
        input_df = pd.DataFrame([input_data])

        # Predict using model
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1] * 100

        # Generate readable output
        if pred == 1:
            label = f"⚠ Fraudulent Transaction (Risk: {prob:.2f}%)"
            color = "red"
        else:
            label = f"✅ Legitimate Transaction (Risk: {prob:.2f}%)"
            color = "green"

        # Render result on webpage
        return render_template("index.html", prediction_text=label, color=color)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}", color="black")

# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
