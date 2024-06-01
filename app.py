from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('stress_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    respiration_rate = float(data['respiration_rate'])
    body_temperature = float(data['body_temperature'])
    blood_oxygen = float(data['blood_oxygen'])
    sleeping_hours = float(data['sleeping_hours'])
    heart_rate = float(data['heart_rate'])

    # Prepare the input data for prediction
    input_data = np.array([[respiration_rate, body_temperature, blood_oxygen, sleeping_hours, heart_rate]])
    scaled_data = scaler.transform(input_data)

    # Perform the prediction
    prediction = model.predict(scaled_data)[0]

    # Generate the result message based on the prediction
    if prediction == 0:
        message = f"Your stress level is {prediction}, and you are in a great condition."
    elif prediction == 1:
        message = f"Your stress level is {prediction}, and we'll concentrate on your condition for some more time."
    elif prediction == 2:
        message = f"Your stress level is {prediction}, and you are in a real danger, sir."
    elif prediction == 3:
        message = f"Your stress level is {prediction}, and you should reply to us now, we'll call 911."
    else:
        message = "You have entered invalid values"

    return jsonify({'prediction': int(prediction), 'message': message})

if __name__ == '__main__':
    app.run(debug=True)
