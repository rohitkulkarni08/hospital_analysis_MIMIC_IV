# Clinical Readmission & ICD Prediction System

This project is a production-ready **clinical decision support tool** built using MIMIC-IV EHR data. It predicts:
- **30-day hospital readmission risk**
- **Likely ICD-10 diagnosis blocks** (multi-label, hierarchical)
- **Natural Language clinical summaries**

---

## Features

- Random Forest–based models trained on structured hospital admission data
- Multi-label hierarchical ICD-10 block and chapter prediction
- Probabilistic output with top-N diagnosis suggestions
- Unified **Streamlit dashboard** for real-time input and prediction
- Integration with **OpenAI GPT** for generating human-readable summaries
- Automatic model downloading from Google Drive (no LFS required)

---

## How to Run Locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/rohitkulkarni08/hospital_analysis_MIMIC_IV.git
   cd hospital_analysis_MIMIC_IV

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   
3. **Set your OpenAI API key**
   ```bash
   export OPENAI_API_KEY="your-openai-key"

4. **Run the Streamlit app**
   ```bash
   export OPENAI_API_KEY="your-openai-key"

---

## Model Files

Model files are downloaded automatically from Google Drive when you run the app or import predict_readmit.py or predict_icd.py.

---

## Notebooks

- 1_Readmissions.ipynb: Readmission model training + feature engineering
- 2_ICD_Prediction.ipynb: Multi-label ICD prediction pipeline

---

## Project Highlights

- Used MIMIC-IV v2.1 EHR dataset (structured only)
- Built hierarchical diagnosis classifier (ICD chapters → blocks)
- Integrated clinical history, demographics, and ICU data
- Deployed in Streamlit with OpenAI GPT-4 based summarization
