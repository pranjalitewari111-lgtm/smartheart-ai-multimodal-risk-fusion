from flask import Flask, render_template, request
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler
model = joblib.load("smartheart_model.pkl")
scaler = joblib.load("scaler.pkl")

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    # Get values from form
    features = [
        float(request.form['age']),
        float(request.form['sex']),
        float(request.form['cp']),
        float(request.form['trestbps']),
        float(request.form['chol']),
        float(request.form['fbs']),
        float(request.form['restecg']),
        float(request.form['thalach']),
        float(request.form['exang']),
        float(request.form['oldpeak']),
        float(request.form['slope']),
        float(request.form['ca']),
        float(request.form['thal'])
    ]

    # Convert to array
    final_features = np.array(features).reshape(1, -1)

    # Scale features
    scaled_features = scaler.transform(final_features)

    # Predict probability
    probability = model.predict_proba(scaled_features)[0][1]

    # Risk category
    if probability < 0.3:
        risk = "Low Risk"
    elif probability < 0.7:
        risk = "Moderate Risk"
    else:
        risk = "High Risk"

    return render_template(
        'index.html',
        prediction_text=f"Heart Disease Probability: {round(probability*100,2)}%",
        risk=risk
    )

# Run app
if __name__ == "__main__":
    app.run(debug=True)