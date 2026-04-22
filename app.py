from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input data
        data = pd.DataFrame([{
            'LIMIT_BAL': float(request.form.get('LIMIT_BAL', 0)),
            'AGE': float(request.form['AGE']),
            'PAY_0': float(request.form['PAY_0']),
            'BILL_AMT1': float(request.form['BILL_AMT1']),
            'PAY_AMT1': float(request.form['PAY_AMT1']),
            'TOTAL_PAY': float(request.form['TOTAL_PAY'])
        }])

        # Scale data
        data_scaled = scaler.transform(data)

        # Predict
        result = model.predict(data_scaled)[0]

        # Output + Color class
        if result == 1:
            output = "High Risk of Default"
            color_class = "high"
        else:
            output = "Low Risk of Default"
            color_class = "low"

    except Exception as e:
        output = f"Error: {str(e)}"
        color_class = "high"   # optional fallback color

    return render_template(
        'index.html',
        prediction_text=output,
        color_class=color_class
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)