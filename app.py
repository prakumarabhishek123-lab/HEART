from flask import Flask, request, render_template
import joblib
import numpy as np

# ✅ FIRST create app
app = Flask(__name__)

# ✅ Load model
model = joblib.load("C:\\Users\\praku\\OneDrive\\Desktop\\Heart\\heart_disease_model.pkl")

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [
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

        final_input = np.array([data])
        prediction = model.predict(final_input)

        result = "Heart Disease Detected ❤️" if prediction[0] == 1 else "No Heart Disease ✅"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return f"Error: {str(e)}"

# Run app
if __name__ == "__main__":
    app.run(debug=True)