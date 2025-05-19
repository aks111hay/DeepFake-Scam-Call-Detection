from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the pre-trained model and scaler
model = joblib.load('deepfake_scam_call_model.pkl')
scaler = joblib.load('scaler.pkl')

def extract_mfcc(file_path, n_mfcc=13):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f'Error extracting MFCC: {e}')
        return None

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Deepfake Scam Call Detection</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #ffb6c1, #ff69b4);
                margin: 0;
                padding: 50px;
                color: #444;
            }
            h2 {
                color: #fff;
                text-align: center;
                font-size: 2.5em;
                margin-bottom: 20px;
            }
            h3 {
                color: #fff;
                text-align: center;
                font-size: 1.8em;
                margin-bottom: 40px;
            }
            div {
                text-align: center;
                margin: auto;
                max-width: 400px;
                background-color: #fff;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            input[type="file"] {
                margin: 10px;
                padding: 8px;
                background-color: #f8f8f8;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            button {
                background-color: #ff6347;
                color: white;
                padding: 12px 20px;
                margin: 10px;
                border: none;
                border-radius: 5px;
                font-size: 1em;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            button:hover {
                background-color: #e53e3e;
                transform: scale(1.05);
            }
            #result {
                margin-top: 20px;
                font-size: 1.2em;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <h2>Deepfake Scam Call Detection</h2>
        <h3>Akshay Kumar Roll No: 2022/B/18 and Ritesh Kumar Roll No: 2022/B/21</h3>
        <div>
            <input type="file" id="audioFile" accept="audio/wav"><br>
            <button onclick="uploadAndPredict()">Predict</button>
            <p id="result"></p>
        </div>
        <script>
            function uploadAndPredict() {
                const fileInput = document.getElementById("audioFile").files[0];
                if (!fileInput) {
                    alert("Please select a .wav file.");
                    return;
                }
                const formData = new FormData();
                formData.append("file", fileInput);
                fetch("/predict", { method: "POST", body: formData })
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById("result").textContent = data.prediction ? "Prediction: " + data.prediction : "Error: " + data.error;
                    })
                    .catch(error => alert("Error: " + error));
            }
        </script>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        file_path = './uploads/uploaded.wav'
        if not os.path.exists('./uploads'):
            os.makedirs('./uploads')
        file.save(file_path)

        features = extract_mfcc(file_path)
        if features is None:
            return jsonify({'error': 'Feature extraction failed'}), 400

        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)
        result = 'Real' if prediction[0] == 0 else 'Fake'
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
