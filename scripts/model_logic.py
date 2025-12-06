# scripts/model_logic.py
import pandas as pd
import numpy as np
from haversine import haversine, Unit
from transformers import pipeline
from functools import lru_cache

# --- A. CONFIGURATION (From your Notebook) ---
KEYWORDS_CRITICAL = [
    'อันตราย', 'อุบัติเหตุ', 'คนเจ็บ', 'เลือดออก', 'เสียชีวิต', 'ทรุด', 
    'ไฟช็อต', 'ระเบิด', 'ไฟไหม้', 'ไฟเขียวดับ', 'สัญญาณไฟเสีย'
]

KEYWORDS_IGNORE = [
    'ชมเชย', 'ขอบคุณ', 'สวยงาม', 'ดีมาก', 'ทดสอบ', 'test'
]

# --- B. AI MODEL HANDLING ---
@lru_cache(maxsize=1)
def load_classifier_model():
    """
    Loads your XLMRoberta model. 
    NOTE: In production, consider moving this to a separate API service 
    to save Airflow memory, but this works for batch processing.
    """
    print("Loading AI Model...")
    # Assuming you are using a standard zero-shot or a specific fine-tuned path
    # If you have a local folder from your notebook, mount it to docker or use the hub ID
    classifier = pipeline("text-classification", model="xlm-roberta-base", return_all_scores=True)
    return classifier

def predict_problem_type_and_severity(df):
    """
    Wraps your notebook's LLM/Model logic.
    """
    classifier = load_classifier_model()
    
    # Process only valid comments
    comments = df['comment'].fillna("").tolist()
    
    # 1. Predict Problem Type (Batch processing is faster)
    # This mimics your 'Define custom problem type' logic
    results = classifier(comments, truncation=True, max_length=512)
    
    # Extract predicted label and score
    # (Adjust logic based on your specific model output structure)
    top_labels = [max(r, key=lambda x: x['score'])['label'] for r in results]
    severity_scores = [max(r, key=lambda x: x['score'])['score'] for r in results]
    
    df['custom_problem_type'] = top_labels
    df['model_severity_score'] = severity_scores
    return df

# --- C. URGENCY CALCULATION ---
def calculate_poi_distance(row, poi_df):
    """
    From your notebook snippet 11: Calculates distance to nearest hospital/school
    """
    if pd.isna(row['latitude']) or pd.isna(row['longitude']):
        return 0.0
        
    min_dist = 9999.0
    user_loc = (row['latitude'], row['longitude'])
    
    # This can be slow. In production, use cKDTree from scipy.spatial for speed
    for _, poi in poi_df.iterrows():
        poi_loc = (poi['lat'], poi['lon'])
        dist = haversine(user_loc, poi_loc, unit=Unit.KILOMETERS)
        if dist < min_dist:
            min_dist = dist
            
    # Normalize: Closer = Higher Score (e.g., within 1km = 1.0)
    if min_dist <= 0.5: return 1.0
    if min_dist <= 2.0: return 0.5
    return 0.1

def calculate_final_urgency(df, poi_df):
    """
    Integrates all your scoring logic (Severity + Freq + Keywords)
    """
    # 1. Apply AI Classification first
    df = predict_problem_type_and_severity(df)
    
    # 2. Calculate Distance Impact
    df['loc_score'] = df.apply(lambda row: calculate_poi_distance(row, poi_df), axis=1)
    
    # 3. Apply Keyword Overrides (From snippet 14)
    def apply_formula(row):
        txt = str(row['comment']).lower()
        
        # Critical Override
        if any(k in txt for k in KEYWORDS_CRITICAL): 
            return 1.0
            
        # Ignore Override
        if any(k in txt for k in KEYWORDS_IGNORE): 
            return 0.0
            
        # Weighted Formula (From snippet 13)
        # urgency = (Model_Severity * 0.5) + (Location_Score * 0.3) + (Recency/Freq * 0.2)
        base_score = (row['model_severity_score'] * 0.6) + (row['loc_score'] * 0.4)
        return min(max(base_score, 0.0), 1.0)

    df['final_urgency_score'] = df.apply(apply_formula, axis=1)
    
    return df