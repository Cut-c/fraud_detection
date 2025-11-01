from flask import Flask, request, jsonify
import joblib
import pandas as pd
import json

app = Flask(__name__)

# Load your pre-trained model
model = joblib.load("best_model.pkl")

@app.route('/')
def home():
    return "<h2>Fraud Detection API is Running 🚀</h2>"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame(data['data'])
        preds = model.predict(df)
        return jsonify({'predictions': preds.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    
@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        vals = [float(x) for x in request.form['features'].split(',')]
        df = pd.DataFrame([vals])
        pred = model.predict(df)[0]
        label = 'FRAUD' if pred == 1 else 'LEGITIMATE'
        return f"<h2>Prediction: {label}</h2>"
    return '''
      <form method="post">
        Enter comma-separated features: <input name="features" size="80">
        <input type="submit" value="Predict">
      </form>
    '''
