import pandas as pd
import numpy as np
from haversine import haversine, Unit
from transformers import pipeline
from functools import lru_cache

# --- A. CONFIGURATION ---
KEYWORDS_CRITICAL = [
    'อันตราย', 'อุบัติเหตุ', 'คนเจ็บ', 'เลือดออก', 'เสียชีวิต', 'ทรุด', 
    'ไฟช็อต', 'ระเบิด', 'ไฟไหม้', 'ไฟเขียวดับ', 'สัญญาณไฟเสีย'
]

KEYWORDS_IGNORE = [
    'ชมเชย', 'ขอบคุณ', 'สวยงาม', 'ดีมาก', 'ทดสอบ', 'test'
]

# The EXACT labels you want the model to choose from
DEFINED_CATEGORIES = [
    "โครงสร้างพื้นฐานและสาธารณูปโภค",
    "การจราจรและสิ่งกีดขวาง",
    "ความปลอดภัยสาธารณะ",
    "ความสะอาดและสุขอนามัย",
    "ปัญหาน้ำท่วมและการระบายน้ำ",
    "ต้นไม้และสัตว์",
    "ป้ายและข้อมูลสาธารณะ",
    "มลภาวะและเสียงรบกวน",
    "เรื่องทั่วไปและการบริการ",
    "ข้อเสนอแนะเรื่องอื่น ๆ"
]

# --- B. AI MODEL HANDLING ---
@lru_cache(maxsize=1)
def load_classifier_model():
    """
    Loads the Zero-Shot Classification model.
    Using 'joeddav/xlm-roberta-large-xnli' as per your notebook.
    """
    print("Loading Zero-Shot Classification Model (joeddav/xlm-roberta-large-xnli)...")
    try:
        classifier = pipeline(
            "zero-shot-classification",
            model="joeddav/xlm-roberta-large-xnli",
            device=-1 # Use -1 for CPU, 0 for GPU if available in Docker
        )
        return classifier
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_problem_type_and_severity(df):
    """
    Applies Zero-Shot Classification to map comments to DEFINED_CATEGORIES.
    """
    classifier = load_classifier_model()
    
    if classifier is None:
        print("Model failed to load. Skipping classification.")
        df['predicted_category'] = "Unspecified"
        df['model_severity_score'] = 0.5
        return df
    
    # Process only valid comments
    # Convert to list for batch processing (faster)
    comments = df['comment'].fillna("").astype(str).tolist()
    
    print(f"Classifying {len(comments)} comments...")
    
    try:
        # Run Batch Classification
        results = classifier(
            comments,
            candidate_labels=DEFINED_CATEGORIES,
            multi_label=False # Force it to pick the ONE best category
        )
        
        # Extract results
        # Result format: {'labels': ['Top Cat', 'Next Cat'...], 'scores': [0.9, 0.1...]}
        top_labels = [r['labels'][0] for r in results]
        confidence_scores = [r['scores'][0] for r in results]
        
        df['predicted_category'] = top_labels
        df['model_severity_score'] = confidence_scores # We use confidence as a proxy for severity/relevance
        
    except Exception as e:
        print(f"Classification failed: {e}")
        df['predicted_category'] = "Error"
        df['model_severity_score'] = 0.0

    return df

# --- C. URGENCY CALCULATION ---
def calculate_poi_distance(row, poi_df):
    """
    Calculates distance to nearest hospital/school.
    """
    # NOTE: row['latitude'] and row['longitude'] must exist.
    # The ETL script ensures these columns are present before calling this.
    if pd.isna(row['latitude']) or pd.isna(row['longitude']):
        return 0.0
        
    min_dist = 9999.0
    user_loc = (row['latitude'], row['longitude'])
    
    # Simple loop (Optimization: Use Scipy KDTree in future for >10k POIs)
    for _, poi in poi_df.iterrows():
        poi_loc = (poi['lat'], poi['lon'])
        dist = haversine(user_loc, poi_loc, unit=Unit.KILOMETERS)
        if dist < min_dist:
            min_dist = dist
            
    # Normalize: Closer = Higher Score (e.g., within 0.5km = 1.0)
    if min_dist <= 0.5: return 1.0
    if min_dist <= 2.0: return 0.5
    return 0.1

def calculate_final_urgency(df, poi_df):
    """
    Integrates Classification + Distance + Keywords
    """
    # 1. Apply AI Classification (Zero-Shot)
    df = predict_problem_type_and_severity(df)
    
    # 2. Calculate Distance Impact
    df['loc_score'] = df.apply(lambda row: calculate_poi_distance(row, poi_df), axis=1)
    
    # 3. Apply Urgency Formula
    def apply_formula(row):
        txt = str(row['comment']).lower()
        
        # Critical Override
        if any(k in txt for k in KEYWORDS_CRITICAL): 
            return 1.0
            
        # Ignore Override
        if any(k in txt for k in KEYWORDS_IGNORE): 
            return 0.0
            
        # Weighted Formula
        # model_severity_score here is actually the Confidence of the classification.
        # You might want to adjust this weight if confidence isn't equal to severity.
        base_score = (row['model_severity_score'] * 0.5) + (row['loc_score'] * 0.5)
        return min(max(base_score, 0.0), 1.0)

    df['urgency_score'] = df.apply(apply_formula, axis=1)
    
    return df