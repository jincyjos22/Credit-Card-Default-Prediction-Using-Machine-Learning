from flask import Flask, render_template, request
import joblib
import pandas as pd

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
        # Get values from form
        limit_bal = float(request.form['LIMIT_BAL'])
        age = float(request.form['AGE'])
        pay_0 = float(request.form['PAY_0'])
        bill_amt1 = float(request.form['BILL_AMT1'])
        pay_amt1 = float(request.form['PAY_AMT1'])
        total_pay = float(request.form['TOTAL_PAY'])

        # ✅ Create DataFrame (no order issue)
        data = pd.DataFrame([{
            'LIMIT_BAL': limit_bal,
            'AGE': age,
            'PAY_0': pay_0,
            'BILL_AMT1': bill_amt1,
            'PAY_AMT1': pay_amt1,
            'TOTAL_PAY': total_pay
        }])

        # Scale
        data_scaled = scaler.transform(data)

        # Predict
        result = model.predict(data_scaled)[0]

        # Output
        if result == 1:
            output = "High Risk of Default"
        else:
            output = "Low Risk of Default"

    except Exception as e:
        output = f"Error: {str(e)}"

    return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)