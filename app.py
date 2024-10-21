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




# from flask import Flask, request, jsonify
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import io
# import os

# # Initialize the Flask app
# app = Flask(__name__)

# # Load your trained model (ensure 'hand_model5.h5' exists in the same directory)
# model = tf.keras.models.load_model('hand_model5.h5')

# # Set the target image size (based on what the model expects, e.g., 224x224)
# IMG_SIZE = (224, 224)

# # Preprocess the uploaded image
# def preprocess_image(image, target_size):
#     if image.size != target_size:
#         image = image.resize(target_size)  # Resize image to match model input size
#     image = np.array(image)  # Convert image to numpy array
#     image = image / 255.0    # Normalize pixel values to [0,1]
#     image = np.expand_dims(image, axis=0)  # Add batch dimension (1, IMG_SIZE, IMG_SIZE, 3)
#     return image

# # @app.route('/')
# # def index():
# #     return "Welcome to the Sign Language Prediction API"

# # Define a route for image prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Ensure that the request contains an image file
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file provided'}), 400
    
#     file = request.files['file']

#     try:
#         # Open the image file and preprocess it
#         image = Image.open(io.BytesIO(file.read()))
#         image = preprocess_image(image, IMG_SIZE)

#         # Perform prediction using the loaded model
#         prediction = model.predict(image)
        
#         # Convert prediction to a list and return as JSON
#         return jsonify({'prediction': prediction.tolist()})
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# # Start the Flask app
# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))
#     app.run(host='0.0.0.0', port=port)


# from flask import Flask, render_template, Response
# import cv2
# from utils import GestureRecognition, initialize_camera

# app = Flask(__name__)

# # Load class labels
# class_labels = ['okay', 'peace', 'thumbs up', 'thumbs down', 'call me', 'live long', 'fist', 'smile']

# # Initialize Gesture Recognition
# gesture_recognition = GestureRecognition('mp_hand_gesture.h5', class_labels)

# def real_time_detection():
#     capture = initialize_camera()

#     while True:
#         ret, frame = capture.read()
#         if not ret:
#             print("Failed to grab frame.")
#             break

#         gestures = gesture_recognition.process_frame(frame)

#         for gesture, hand_landmarks in gestures:
#             gesture_recognition.draw_landmarks(frame, hand_landmarks, gesture)

#         # Encode frame to JPEG for the video stream
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     capture.release()
#     cv2.destroyAllWindows()

# @app.route('/')
# def index():
#     """Video streaming home page."""
#     return render_template('app.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(real_time_detection(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, Response
import cv2
from utils import GestureRecognition, initialize_camera

app = Flask(__name__)

# Load the trained model and class labels
class_labels = ['okay', 'peace', 'thumbs up', 'thumbs down', 'call me','stop'
                , 'rock', 'live long', 'fist', 'smile']  # Add your actual class labels
model_path = 'mp_hand_gesture.h5'  # Update this path as necessary
gesture_recognition = GestureRecognition(model_path, class_labels)

def generate_frames():
    capture = initialize_camera()
    while True:
        # Read each frame from the webcam
        success, frame = capture.read()
        if not success:
            break

        # Process the frame for gesture recognition
        gestures = gesture_recognition.process_frame(frame)

        # If gestures were detected, draw landmarks and show the predicted gesture
        if gestures:
            for gesture, hand_landmarks in gestures:
                gesture_recognition.draw_landmarks(frame, hand_landmarks, gesture)

        # Encode the frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('app.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

