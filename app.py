# from flask import Flask, request, jsonify
# import tensorflow as tf 
# import os# or whatever you're using to load your model

# app = Flask(__name__)

# # Load your model (make sure it's in your project directory)
# model = tf.keras.models.load_model('hand_model5.h5')


# # Define a route for prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the input data (e.g., from JSON)
#     data = request.get_json(force=True)
    
#     # Make a prediction using the model
#     prediction = model.predict([data['input']])
    
#     # Return the prediction as a JSON response
#     return jsonify({'prediction': prediction.tolist()})

# # Start the Flask app
# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))
#     app.run(host='0.0.0.0', port=port)

from flask import Flask, request, jsonify
import tensorflow as tf
import os

app = Flask(__name__)

# Load your model
model = tf.keras.models.load_model('hand_model5.h5')  # Ensure this file exists

@app.route('/')
def index():
    return "Welcome to the Sign Language Prediction API"
# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        prediction = model.predict([data['input']])  # Ensure input matches model expectations
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400  # Return error message if something goes wrong

# Start the Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
