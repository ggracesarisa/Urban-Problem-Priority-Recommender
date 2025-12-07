import pandas as pd
import numpy as np
from haversine import haversine, Unit
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
import joblib
import os

# --- CONFIGURATION ---
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

KEYWORDS_CRITICAL = ['อันตราย', 'อุบัติเหตุ', 'คนเจ็บ', 'เลือดออก', 'เสียชีวิต', 'ไฟไหม้']
KEYWORDS_IGNORE = ['ชมเชย', 'ขอบคุณ', 'ทดสอบ', 'test']

# --- 1. THE TEACHER (LLM) ---
def get_teacher_labels(texts, labels):
    """
    Uses the Slow LLM to label a small chunk of data accurately.
    """
    print(f"--- [Teacher] LLM labeling {len(texts)} rows (Grab a coffee, this takes ~5-10 mins)...")
    
    # Use the smaller, faster Zero-Shot model
    classifier = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        device=-1  # CPU
    )
    
    predictions = []
    # Batch size 16 for speed
    results = classifier(texts, candidate_labels=labels, multi_label=False, batch_size=16)
    
    for r in results:
        predictions.append(r['labels'][0])
        
    return predictions

# --- 2. THE STUDENT (Scikit-Learn) ---
def train_student_model(training_texts, training_labels):
    """
    Trains a super-fast statistical model using the LLM's answers.
    """
    print(f"--- [Student] Training fast model on {len(training_texts)} examples...")
    
    # TF-IDF + SGD (Support Vector Machine / Logistic Regression equivalent)
    # This processes text 10,000x faster than a Transformer
    model = make_pipeline(
        TfidfVectorizer(tokenizer=None, preprocessor=None, ngram_range=(1, 2), max_features=5000),
        SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, n_jobs=-1)
    )
    
    model.fit(training_texts, training_labels)
    print("--- [Student] Training Complete.")
    return model

# --- 3. MAIN PREDICTION LOGIC ---
def predict_problem_type_and_severity(df):
    
    # Clean data first
    df['comment'] = df['comment'].fillna("").astype(str)
    
    # Step A: Check if we already have a trained Student model
    model_path = '/opt/airflow/data/student_model.pkl'
    student_model = None
    
    # If we processed data before, load the existing model to save time
    if os.path.exists(model_path):
        print("Loading existing Student model...")
        student_model = joblib.load(model_path)
    
    # Step B: If no model, we must train one (Teacher-Student phase)
    if student_model is None:
        # 1. Take a sample (e.g., 2,000 rows) to teach the student
        # If df is smaller than 2000, use all of it.
        sample_size = min(len(df), 500)
        
        print(f"Sampling {sample_size} rows to train the model...")
        
        # Shuffle and pick sample
        train_df = df.sample(n=sample_size, random_state=42)
        train_texts = train_df['comment'].tolist()
        
        # 2. Teacher (LLM) labels the sample
        train_labels = get_teacher_labels(train_texts, DEFINED_CATEGORIES)
        
        # 3. Student learns from Teacher
        student_model = train_student_model(train_texts, train_labels)
        
        # Save the student so we don't have to use the LLM again next time
        joblib.dump(student_model, model_path)
        
    # Step C: Mass Inference (The fast part)
    print(f"--- [Speed] Predicting {len(df)} rows using Student model...")
    
    # This runs instantly (seconds for 500k rows)
    predictions = student_model.predict(df['comment'])
    
    # Get confidence scores (probability of the winning class)
    probs = student_model.predict_proba(df['comment'])
    confidence = np.max(probs, axis=1)
    
    df['predicted_category'] = predictions
    df['model_severity_score'] = confidence
    
    return df

# --- URGENCY CALCULATION (Same as before) ---
def calculate_poi_distance(row, poi_df):
    if pd.isna(row['latitude']) or pd.isna(row['longitude']): return 0.0
    
    # Quick optimization: check bounding box before haversine
    # (Optional but good for 500k rows)
    
    min_dist = 9999.0
    user_loc = (row['latitude'], row['longitude'])
    
    for _, poi in poi_df.iterrows():
        # Quick check: if lat diff > 0.1 degree (~11km), skip haversine
        if abs(user_loc[0] - poi['lat']) > 0.1 or abs(user_loc[1] - poi['lon']) > 0.1:
            continue
            
        poi_loc = (poi['lat'], poi['lon'])
        dist = haversine(user_loc, poi_loc, unit=Unit.KILOMETERS)
        if dist < min_dist:
            min_dist = dist
            
    if min_dist <= 0.5: return 1.0
    if min_dist <= 2.0: return 0.5
    return 0.1

def calculate_final_urgency(df, poi_df):
    # 1. Fast AI Prediction
    df = predict_problem_type_and_severity(df)
    
    # 2. Distance Calculation
    print("Calculating distances...")
    # Vectorized application is still fastest for custom logic in Pandas
    df['loc_score'] = df.apply(lambda row: calculate_poi_distance(row, poi_df), axis=1)
    
    # 3. Final Formula
    print("Calculating final scores...")
    def apply_formula(row):
        txt = str(row['comment']).lower()
        if any(k in txt for k in KEYWORDS_CRITICAL): return 1.0
        if any(k in txt for k in KEYWORDS_IGNORE): return 0.0
        
        base_score = (float(row['model_severity_score']) * 0.5) + (float(row['loc_score']) * 0.5)
        return min(max(base_score, 0.0), 1.0)

    df['urgency_score'] = df.apply(apply_formula, axis=1)
    
    return df