from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model & scaler & encoder & thresholds
with open('model/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model/encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

with open('model/thresholds.pkl', 'rb') as f:
    thresholds = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    nama_obat = encoders['nama_obat'].transform([data['nama_obat']])[0]
    gender = encoders['gender'].transform([data['gender']])[0]
    allergy = encoders['allergy_history'].transform([data['allergy_history']])[0]

    X = np.array([[nama_obat, data['age'], gender, data['dosage_mg'], allergy]])
    X_scaled = scaler.transform(X)

    probas = model.predict_proba(X_scaled)
    
    hasil = {}
    for i, label in enumerate(thresholds.keys()):
        proba = probas[i][:,1][0]
        hasil[label] = int(proba >= thresholds[label])

    return jsonify(hasil)

if __name__ == '__main__':
    app.run(debug=True)
