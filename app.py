from flask import Flask, request, render_template
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
model = joblib.load("best_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # List of all features used during training (V1–V28 + Amount + Time)
        all_features = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]

        # Initialize all features as 0.0
        input_data = {col: 0.0 for col in all_features}

        # Fill in only the features the user provided from the HTML form
        for key in ["V1", "V2", "V3", "V4", "Amount", "Time"]:
            value = request.form.get(key)
            if value is not None and value.strip() != "":
                input_data[key] = float(value)

        # Convert into DataFrame in correct column order
        input_df = pd.DataFrame([input_data])

        # Predict using model
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1] * 100

        # Prepare result message
        if pred == 1:
            label = f"⚠ Fraudulent Transaction (Risk: {prob:.2f}%)"
            color = "red"
        else:
            label = f"✅ Legitimate Transaction (Risk: {prob:.2f}%)"
            color = "green"

        return render_template("index.html", prediction_text=label, color=color)

    except Exception as e:
        # Catch and show errors gracefully
        return render_template("index.html", prediction_text=f"Error: {e}", color="black")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
