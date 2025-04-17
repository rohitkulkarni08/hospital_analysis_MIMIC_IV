# predict_icd.py

import pandas as pd
import joblib
import numpy as np
import os
import gdown

# Google Drive file IDs (replace with your actual links)
CHAPTER_MODEL_ID = "1RKc4FyTyEvLDK_CU49SSdeWNIBd0MoeT"
BLOCK_MODEL_ID = "1wFUsVc-OHqnFRyEAXG0EboYqU47-HGlF"
CHAPTER_LABELS_ID = "1KHDnu8D2FxShwRbG4FB9sHoeYEiyAuyb"
BLOCK_LABELS_ID = "1x6t89yVh3fR-ZOXzAZ_j4BeYbPd5vDZW"
FEATURES_ID = "1VdbtBjBSnrZJBrQg7pIf7dcbgwKsZCqM"

os.makedirs("models", exist_ok=True)

CHAPTER_MODEL_PATH = "models/icd_chapter_model.pkl"
BLOCK_MODEL_PATH = "models/icd_block_model.pkl"
CHAPTER_LABELS_PATH = "models/icd_chapter_labels.pkl"
BLOCK_LABELS_PATH = "models/icd_block_labels.pkl"
FEATURES_PATH = "models/icd_model_features.pkl"

# Download helper
def download_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading {output_path} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

# Download all model files if needed
download_from_drive(CHAPTER_MODEL_ID, CHAPTER_MODEL_PATH)
download_from_drive(BLOCK_MODEL_ID, BLOCK_MODEL_PATH)
download_from_drive(CHAPTER_LABELS_ID, CHAPTER_LABELS_PATH)
download_from_drive(BLOCK_LABELS_ID, BLOCK_LABELS_PATH)
download_from_drive(FEATURES_ID, FEATURES_PATH)

# Load models
clf_chapter = joblib.load(CHAPTER_MODEL_PATH)
clf_block = joblib.load(BLOCK_MODEL_PATH)
chapter_labels = joblib.load(CHAPTER_LABELS_PATH)
block_labels = joblib.load(BLOCK_LABELS_PATH)
features = joblib.load(FEATURES_PATH)

def predict_icd(input_dict: dict, top_n_blocks: int = 3) -> dict:
    """
    Predict ICD chapter(s) and most likely ICD blocks with confidence.
    
    Args:
        input_dict (dict): Patient data
        top_n_blocks (int): Number of top block predictions per chapter

    Returns:
        dict: {
            'chapters': [
                {
                    'chapter': 'E',
                    'confidence': 0.76,
                    'top_blocks': [
                        {'block': 'E78', 'prob': 0.56},
                        {'block': 'E11', 'prob': 0.41}
                    ]
                }, ...
            ]
        }
    """
    # Create input DataFrame with correct feature order
    df = pd.DataFrame([input_dict])
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df[features]

    # Chapter prediction
    chapter_probs = clf_chapter.predict_proba(df)
    chapter_confidences = {chapter_labels[i]: prob[0][1] for i, prob in enumerate(chapter_probs)}

    # Block prediction
    block_probs = clf_block.predict_proba(df)
    block_confidences = {block_labels[i]: prob[0][1] for i, prob in enumerate(block_probs)}

    # Group blocks by chapter prefix
    final_output = {'chapters': []}
    for chapter, chap_prob in sorted(chapter_confidences.items(), key=lambda x: x[1], reverse=True):
        # Find blocks that belong to this chapter
        matching_blocks = {k: v for k, v in block_confidences.items() if k.startswith(chapter)}
        sorted_blocks = sorted(matching_blocks.items(), key=lambda x: x[1], reverse=True)[:top_n_blocks]
        final_output['chapters'].append({
            'chapter': chapter,
            'confidence': round(chap_prob, 4),
            'top_blocks': [{'block': b, 'prob': round(p, 4)} for b, p in sorted_blocks]
        })

    return final_output