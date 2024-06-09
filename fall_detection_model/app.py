from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
import sys

# Ensure default encoding is set to UTF-8
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

# Set up logging to handle Unicode characters
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s', encoding='utf-8')

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('fall_detection_model.keras')

# Load the dataset for scaling and label encoding
data = pd.read_excel(r'acc_gyr.xlsx')

# Initialize the scaler and label encoder
scaler = StandardScaler()
scaler.fit(data[['xAcc', 'yAcc', 'zAcc', 'xGyro', 'yGyro', 'zGyro']])
label_encoder = LabelEncoder()
label_encoder.fit(data['label'])

def predict_sensor_data(readings):
    try:
        # Convert readings to a numpy array and scale
        readings_scaled = scaler.transform(np.array(readings).reshape(1, -1))

        # Create a sequence of length 10
        readings_seq = np.repeat(readings_scaled, 10, axis=0).reshape(1, 10, 6)

        # Predict using the model
        prediction = model.predict(readings_seq)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

        return predicted_label[0]
    except Exception as e:
        logging.error(f"Error in predict_sensor_data: {str(e)}")
        raise

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Get the data from the POST request
        readings = data['readings']  # Extract the readings

        # Check if readings are in the correct format
        if len(readings) != 6:
            return jsonify({'error': 'Each reading must contain exactly 6 values.'}), 400

        # Get the prediction
        predicted_label = predict_sensor_data(readings)
        return jsonify({'predicted_label': predicted_label})
    except Exception as e:
        logging.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Suppress specific warnings
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
    
    app.run(debug=True)
