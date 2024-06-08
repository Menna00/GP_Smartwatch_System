from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('final_best_random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get the data from the POST request
    features = pd.DataFrame(data)  # Convert the JSON data to a DataFrame

    # Perform the same preprocessing as done during training
    features_scaled = scaler.transform(features)

    # Make predictions
    predictions = model.predict(features_scaled)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
