import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model & scaler & encoder & thresholds
with open('model/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model/encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

with open('model/thresholds.pkl', 'rb') as f:
    thresholds = pickle.load(f)

st.title("SAFEDOSE")

nama_obat = st.selectbox("Nama Obat", encoders['nama_obat'].classes_)
age = st.slider("Umur", 18, 80, 30)
gender = st.radio("Gender", encoders['gender'].classes_)
dosage_mg = st.selectbox("Dosis (mg)", [250, 500, 750])
allergy = st.radio("Riwayat Alergi", encoders['allergy_history'].classes_)

if st.button("Prediksi Efek Samping"):
    nama_obat_encoded = encoders['nama_obat'].transform([nama_obat])[0]
    gender_encoded = encoders['gender'].transform([gender])[0]
    allergy_encoded = encoders['allergy_history'].transform([allergy])[0]

    X = np.array([[nama_obat_encoded, age, gender_encoded, dosage_mg, allergy_encoded]])
    X_scaled = scaler.transform(X)

    probas = model.predict_proba(X_scaled)

    hasil = {}
    for i, label in enumerate(thresholds.keys()):
        proba = probas[i][:,1][0]
        hasil[label] = f"{proba:.2f} (Efek: {'Ya' if proba >= thresholds[label] else 'Tidak'})"

    st.write(pd.DataFrame.from_dict(hasil, orient='index', columns=['Probabilitas (Efek)']))

