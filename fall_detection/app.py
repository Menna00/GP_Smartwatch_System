from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('fall_detection_model.h5')

# Define the labels
LABELS = [
    "WALKING", 
    "WALKING_UPSTAIRS", 
    "WALKING_DOWNSTAIRS", 
    "SITTING", 
    "STANDING", 
    "LAYING"
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}, ensure_ascii=False), 400
    
    try:
        # Extract features from the JSON request
        features = np.array(data['features']).astype(np.float32)
        features = features.reshape(1, features.shape[0], features.shape[1])
        
        # Make a prediction
        predictions = model.predict(features)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = LABELS[predicted_class]
        
        return jsonify({
            'predicted_label': predicted_label,
            'predicted_class': int(predicted_class),
            'predictions': predictions.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}, ensure_ascii=False), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
