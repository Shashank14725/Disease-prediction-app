import streamlit as st
import numpy as np
import pickle

# Load Model and Scaler
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Diabetes Prediction System")

st.write("Enter Patient Details")

preg = st.number_input("Pregnancies", min_value=0)

glucose = st.number_input("Glucose", min_value=0)

bp = st.number_input("Blood Pressure", min_value=0)

skin = st.number_input("Skin Thickness", min_value=0)

insulin = st.number_input("Insulin", min_value=0)

bmi = st.number_input("BMI", min_value=0.0)

dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)

age = st.number_input("Age", min_value=1)

if st.button("Predict"):

    data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Patient is Diabetic")
    else:
        st.success("Patient is Not Diabetic")