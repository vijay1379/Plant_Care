import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import requests
from ultralytics import YOLO
import tempfile
import cv2

app = Flask(__name__)
FASTAPI_URL = "https://chris2002-ml-app.hf.space/predict"
model = YOLO("yolo_test.pt") 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/classify', methods=['POST'])
def classify():
    try:
        # ✅ Check for file upload
        if 'file' not in request.files:
            return jsonify({"error": "No file part in request"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # ✅ Save uploaded image to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name

        # ✅ Run YOLO prediction
        results = model.predict(source=temp_file_path, conf=0.4, verbose=False)
        boxes = results[0].boxes

        # ✅ Check if any detection is "mulberry_leaf" (class 0)
        mulberry_detected = False
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:  # Assuming class 0 is 'mulberry_leaf'
                    mulberry_detected = True
                    break

        if not mulberry_detected:
            os.remove(temp_file_path)
            return jsonify({"error": "Upload only mulberry leaf!"}), 400

        # ✅ Forward image to external FastAPI service
        with open(temp_file_path, 'rb') as f:
            files = {'file': (file.filename, f, file.content_type)}
            response = requests.post(FASTAPI_URL, files=files)

        os.remove(temp_file_path)

        if response.status_code == 200:
            result = response.json()
            return jsonify({
                "predicted_class": result.get("predicted_class", "Unknown"),
                "confidence": result.get("confidence", "N/A")
            }), 200
        else:
            return jsonify({"error": "FastAPI prediction failed"}), response.status_code

    except Exception as e:
        return jsonify({"error": f"Internal error: {str(e)}"}), 500
    
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
