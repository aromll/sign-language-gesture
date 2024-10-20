from flask import Flask, request, jsonify
import joblib 
import tensorflow as tf # or whatever you're using to load your model

app = Flask(__name__)

# Load your model (make sure it's in your project directory)
model = tf.keras.models.load_model('hand_model5.h5')


# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data (e.g., from JSON)
    data = request.get_json(force=True)
    
    # Make a prediction using the model
    prediction = model.predict([data['input']])
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction.tolist()})

# Start the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

