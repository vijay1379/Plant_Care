import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from flask import Flask, request, jsonify, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
import uuid
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = load_model('Custom_layer.keras', custom_objects={'Precision': tf.keras.metrics.Precision, 'Recall': tf.keras.metrics.Recall})

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def prepare_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize the image
    return image

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({'filename': filename}), 200
    
    except Exception as e:
        print(f"Error uploading file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/classify', methods=['POST'])
def classify():
    try:
        filename = request.json['filename']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        image = Image.open(filepath)
        processed_image = prepare_image(image, target_size=(224, 224))  # Adjust target size to the correct dimensions
  # Adjusted target size
        
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = round(100 * (np.max(predictions[0])), 2)
        
        class_labels = ['Healthy', 'Leaf Rust', 'Leaf Spot']  # Adjust based on your model
        predicted_label = class_labels[predicted_class]
        
        return jsonify({
            "predicted_class": predicted_label,
            "confidence": confidence
        }), 200
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
