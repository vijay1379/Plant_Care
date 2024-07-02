import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import requests

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
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Make API request to your FastAPI endpoint
        files = {'file': (file.filename, file.stream, file.content_type)}
        response = requests.post(FASTAPI_URL, files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("API Response:", result)  # Log the entire response
            
            predicted_class = result.get('predicted_class', 'Unknown')
            confidence = result.get('confidence', 'N/A')
            
            return jsonify({
                "predicted_class": predicted_class,
                "confidence": confidence
            }), 200
        else:
            print(f"API request failed with status code: {response.status_code}")
            print("Response content:", response.text)
            return jsonify({"error": "API request failed"}), response.status_code
    
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to FastAPI: {str(e)}")
        return jsonify({"error": "Failed to connect to classification service"}), 503
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
