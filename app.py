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

# from flask import Flask, request, jsonify
# import tensorflow as tf
# import os

# app = Flask(__name__)

# # Load your model
# model = tf.keras.models.load_model('hand_model5.h5')  # Ensure this file exists

# @app.route('/')
# def index():
#     return "Welcome to the Sign Language Prediction API"

# # Define a route for prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json(force=True)
#         prediction = model.predict([data['input']])  # Ensure input matches model expectations
#         return jsonify({'prediction': prediction.tolist()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400  # Return error message if something goes wrong

# # Start the Flask app
# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))
#     app.run(host='0.0.0.0', port=port)




from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Initialize the Flask app
app = Flask(__name__)

# Load your trained model (ensure 'hand_model5.h5' exists in the same directory)
model = tf.keras.models.load_model('hand_model5.h5')

# Set the target image size (based on what the model expects, e.g., 224x224)
IMG_SIZE = (224, 224)

# Preprocess the uploaded image
def preprocess_image(image, target_size):
    if image.size != target_size:
        image = image.resize(target_size)  # Resize image to match model input size
    image = np.array(image)  # Convert image to numpy array
    image = image / 255.0    # Normalize pixel values to [0,1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, IMG_SIZE, IMG_SIZE, 3)
    return image

@app.route('/')
def index():
    return "Welcome to the Sign Language Prediction API"

# Define a route for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Ensure that the request contains an image file
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']

    try:
        # Open the image file and preprocess it
        image = Image.open(io.BytesIO(file.read()))
        image = preprocess_image(image, IMG_SIZE)

        # Perform prediction using the loaded model
        prediction = model.predict(image)
        
        # Convert prediction to a list and return as JSON
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Start the Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)