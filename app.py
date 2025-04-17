import streamlit as st
import openai
import pandas as pd
from predict_readmit import predict_readmission
from predict_icd import predict_icd

# Set your API key
openai.api_key = ""

# Streamlit Input Form

st.title("🩺 Clinical Risk & ICD Predictor")

with st.form("patient_form"):
    st.header("📋 Enter Patient Info")

    age = st.number_input("Age", min_value=0, max_value=120, value=65)
    los = st.number_input("Length of Stay (days)", min_value=0, value=4)
    n_diagnoses = st.number_input("Number of Diagnoses", min_value=0, value=3)
    n_procedures = st.number_input("Number of Procedures", min_value=0, value=2)
    icu_flag = st.radio("ICU Admission", ["Yes", "No"]) == "Yes"
    num_prev_admissions = st.number_input("Number of Previous Admissions", min_value=0, value=1)
    days_since_last_discharge = st.number_input("Days Since Last Discharge", value=10)
    
    admission_type = st.selectbox("Admission Type", [
        "ELECTIVE", "EMERGENCY", "URGENT", "OBSERVATION", "SURGICAL SAME DAY ADMISSION"
    ])
    
    insurance = st.selectbox("Insurance", [
        "Medicare", "Medicaid", "Private", "Other"
    ])
    
    gender = st.radio("Gender", ["Male", "Female"])

    submitted = st.form_submit_button("🔍 Predict Risk & Diagnoses")

# Prediction Logic

if submitted:
    # Prepare input dict
    input_dict = {
        "age": age,
        "los": los,
        "n_diagnoses": n_diagnoses,
        "n_procedures": n_procedures,
        "icu_flag": int(icu_flag),
        "num_prev_admissions": num_prev_admissions,
        "days_since_last_discharge": days_since_last_discharge,
        f"admission_type_{admission_type}": 1,
        f"insurance_{insurance}": 1,
        "gender_M": 1 if gender == "Male" else 0
    }

    # Fill other one-hot keys with 0
    expected_cats = [
        "admission_type_ELECTIVE", "admission_type_EMERGENCY", "admission_type_URGENT",
        "admission_type_OBSERVATION", "admission_type_SURGICAL SAME DAY ADMISSION",
        "insurance_Medicare", "insurance_Medicaid", "insurance_Private", "insurance_Other"
    ]
    for cat in expected_cats:
        if cat not in input_dict:
            input_dict[cat] = 0

    # Run both predictions
    readmit_result = predict_readmission(input_dict)
    icd_result = predict_icd(input_dict)

    # Show raw outputs
    st.subheader("📊 Readmission Prediction")
    st.write(readmit_result)

    st.subheader("🧠 ICD Chapter + Block Predictions")
    for chapter in icd_result['chapters'][:3]:  # show top 3
        st.markdown(f"**{chapter['chapter']}** — Confidence: {chapter['confidence']*100:.1f}%")
        for block in chapter['top_blocks']:
            st.markdown(f"&nbsp;&nbsp;&nbsp;• {block['block']} — {block['prob']*100:.1f}%")

    # Worded OpenAI Summary
    
    st.subheader("📝 Clinical Summary")

    prompt = f"""
A patient presents with the following details:
- Age: {age}, Gender: {gender}
- Length of Stay: {los} days
- ICU Admission: {"Yes" if icu_flag else "No"}
- Previous Admissions: {num_prev_admissions}
- Days Since Last Discharge: {days_since_last_discharge}
- Admission Type: {admission_type}
- Insurance: {insurance}

The 30-day readmission risk is **{readmit_result['risk_tier']}** ({readmit_result['readmit_prob']*100:.1f}% probability).

Most likely ICD-10 diagnosis chapters and blocks include:
"""

    for ch in icd_result['chapters'][:3]:
        block_lines = ", ".join([f"{b['block']} ({b['prob']*100:.1f}%)" for b in ch['top_blocks']])
        prompt += f"\n- {ch['chapter']}: {block_lines}"

    prompt += "\n\nSummarize the patient’s risk profile and suggest any possible clinical implications."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.5
        )
        summary = response['choices'][0]['message']['content']
        st.markdown(summary)
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
