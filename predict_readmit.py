import pandas as pd
import joblib
import os
import gdown

# Google Drive file IDs
READMIT_MODEL_ID = "1TX9CHTSo9bQDR61lMqZx7C-LLPHBRvP-"  # rf_readmit_model.pkl
FEATURES_ID = "1tN5pbw9EjVbORckIlI36XFTcscE0UIhM"  # You'll replace this too

# Paths
MODEL_PATH = os.path.join("models", "rf_readmit_model.pkl")
FEATURES_PATH = os.path.join("models", "readmit_model_features.pkl")

def download_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        print(f"⬇️ Downloading {output_path} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
        
os.makedirs("models", exist_ok=True)

# Download models if missing
download_from_drive(READMIT_MODEL_ID, MODEL_PATH)
download_from_drive(FEATURES_ID, FEATURES_PATH)


def predict_readmission(input_data: dict) -> dict:
    """
    Predict readmission risk from a single patient input dictionary.

    Args:
        input_data (dict): A dictionary of patient input features

    Returns:
        dict: {
            'readmit_prob': float (0–1),
            'readmit_pred': 0 or 1,
            'risk_tier': "Low" / "Medium" / "High"
        }
    """
    # Build dataframe with one row
    df = pd.DataFrame([input_data])

    missing_cols = set(feature_list) - set(df.columns)
    for col in missing_cols:
        df[col] = 0

    # Reorder
    df = df[feature_list]

    # Predict
    prob = model.predict_proba(df)[0][1]
    pred = model.predict(df)[0]

    # Risk Tier
    if prob >= 0.75:
        tier = "High"
    elif prob >= 0.50:
        tier = "Medium"
    else:
        tier = "Low"

    return {
        "readmit_prob": round(prob, 4),
        "readmit_pred": int(pred),
        "risk_tier": tier
    }