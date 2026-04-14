from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# Home Route
@app.route('/')
def home():
    return render_template("index.html")


# Load Model
model = pickle.load(open("credit_model.pkl", "rb"))

# Load Scaler
scaler = pickle.load(open("scaler.pkl", "rb"))


# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():

    try:
        data = request.get_json()

        # Get Inputs
        age = float(data['age'])
        income = float(data['income'])
        loan = float(data['loan'])
        credit = float(data['credit'])
        employment = float(data['employment'])
        education = float(data['education'])
        housing = float(data['housing'])

        # Convert to DataFrame (IMPORTANT)
        features = pd.DataFrame([[
            age,
            income,
            loan,
            credit,
            employment,
            education,
            housing
        ]], columns=[
            'Age',
            'Income',
            'Loan_Amount',
            'Credit_Score',
            'Employment_Years',
            'Education_Level',
            'Housing_Status'
        ])

        # Apply Scaling
        features = scaler.transform(features)

        # Prediction
        prediction = model.predict(features)
        predicted_value = int(prediction[0])

        # Result Message
        if predicted_value == 1:
            status = "Successful ✅"
        else:
            status = "Fail ❌"

        print("Prediction:", predicted_value)
        print("Status:", status)

        return jsonify({
            "prediction": predicted_value,
            "status": status
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({
            "prediction": None,
            "status": "Error",
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)
