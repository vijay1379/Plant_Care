import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import requests
import tempfile

app = Flask(__name__)
FASTAPI_URL = "https://chris2002-ml-app.hf.space/predict"

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

        # ✅ Forward image to external FastAPI service
        with open(temp_file_path, 'rb') as f:
            files = {'file': (file.filename, f, file.content_type)}
            response = requests.post(FASTAPI_URL, files=files)

        os.remove(temp_file_path)

        print("FastAPI status:", response.status_code)
        print("FastAPI response:", response.text)

        if response.status_code == 200:
            result = response.json()
            if "error" in result:
                return jsonify({"error": result["error"]}), 400
            return jsonify({
                "predicted_class": result.get("predicted_class", "Unknown"),
                "confidence": result.get("confidence", "N/A")
            }), 200
        else:
            try:
                error_msg = response.json().get("error", "FastAPI prediction failed")
            except Exception:
                error_msg = "FastAPI prediction failed"
            return jsonify({"error": error_msg}), response.status_code

    except Exception as e:
        return jsonify({"error": f"Internal error: {str(e)}"}), 500
    

if __name__ == '__main__':
    app.run(debug=True)
