import pandas as pd
import numpy as np
import re
import os
from rapidfuzz import process, fuzz
from pythainlp.tokenize import word_tokenize
from scripts.model_logic import calculate_final_urgency

# --- A. CONFIGURATION & MAPPINGS ---
def get_district_mapping():
    # A sample of the mapping from your notebook (Expand this with your full list)
    return {
        "พระนคร": "Phra Nakhon",
        "ดุสิต": "Dusit",
        "หนองจอก": "Nong Chok",
        "บางรัก": "Bang Rak",
        "บางเขน": "Bang Khen",
        "บางกะปิ": "Bang Kapi",
        "ปทุมวัน": "Pathum Wan",
        "ป้อมปราบศัตรูพ่าย": "Pom Prap Sattru Phai",
        "พระโขนง": "Phra Khanong",
        "มีนบุรี": "Min Buri",
        "ลาดกระบัง": "Lat Krabang",
        "ยานนาวา": "Yan Nawa",
        "สัมพันธวงศ์": "Samphanthawong",
        "พญาไท": "Phaya Thai",
        "ธนบุรี": "Thon Buri",
        "บางกอกใหญ่": "Bangkok Yai",
        "ห้วยขวาง": "Huai Khwang",
        "คลองสาน": "Khlong San",
        "ตลิ่งชัน": "Taling Chan",
        "บางกอกน้อย": "Bangkok Noi",
        "บางขุนเทียน": "Bang Khun Thian",
        "ภาษีเจริญ": "Phasi Charoen",
        "หนองแขม": "Nong Khaem",
        "ราษฎร์บูรณะ": "Rat Burana",
        "บางพลัด": "Bang Phlat",
        "ดินแดง": "Din Daeng",
        "บึงกุ่ม": "Bueng Kum",
        "สาทร": "Sathon",
        "บางซื่อ": "Bang Sue",
        "จตุจักร": "Chatuchak",
        "บางคอแหลม": "Bang Kho Laem",
        "ประเวศ": "Prawet",
        "คลองเตย": "Khlong Toei",
        "สวนหลวง": "Suan Luang",
        "จอมทอง": "Chom Thong",
        "ดอนเมือง": "Don Mueang",
        "ราชเทวี": "Ratchathewi",
        "ลาดพร้าว": "Lat Phrao",
        "วัฒนา": "Watthana",
        "บางแค": "Bang Khae",
        "หลักสี่": "Lak Si",
        "สายไหม": "Sai Mai",
        "คันนายาว": "Khan Na Yao",
        "สะพานสูง": "Saphan Sung",
        "วังทองหลาง": "Wang Thonglang",
        "คลองสามวา": "Khlong Sam Wa",
        "บางนา": "Bang Na",
        "ทวีวัฒนา": "Thawi Watthana",
        "ทุ่งครุ": "Thung Khru",
        "บางบอน": "Bang Bon"
    }

# --- B. CLEANING HELPERS (From your Notebook) ---

def parse_coords(val):
    """
    Robust coordinate parser from your notebook.
    """
    if not isinstance(val, str) or val.strip() == "":
        return np.nan, np.nan
    
    s = val.strip()
    try:
        # Case 1: "100.5, 13.7"
        if "," in s:
            parts = s.split(",")
            if len(parts) == 2:
                # Usually Traffy data is Lon, Lat
                v1, v2 = float(parts[0]), float(parts[1])
                # Thai Lat is approx 5-20, Lon is 97-105
                if 5 < v2 < 21 and 97 < v1 < 106:
                    return v2, v1 # Lat, Lon
                elif 5 < v1 < 21 and 97 < v2 < 106:
                    return v1, v2 # Lat, Lon
        return np.nan, np.nan
    except:
        return np.nan, np.nan

def clean_text(text):
    """
    Standardizes text by removing 'Bangkok', 'District' prefixes.
    """
    if not isinstance(text, str): return ""
    text = text.strip()
    text = re.sub(r"แขวง|เขต|กรุงเทพมหานคร|กรุงเทพฯ|กรุงเทพ|จังหวัด", "", text)
    text = re.sub(r"[()\-_,]", " ", text)
    return text.strip()

def standardize_district(district_name, mapping):
    """
    Uses RapidFuzz to find the best English match for a Thai district name.
    """
    if not isinstance(district_name, str): return None
    clean_name = clean_text(district_name)
    
    # 1. Exact Match
    if clean_name in mapping:
        return mapping[clean_name]
    
    # 2. Fuzzy Match (Threshold 80)
    match = process.extractOne(clean_name, mapping.keys(), scorer=fuzz.WRatio)
    if match and match[1] >= 80:
        return mapping[match[0]]
    
    return None

def is_nonsense_comment(text):
    """
    Filters out spam or empty comments (e.g., 'test', '...', short text).
    """
    if not isinstance(text, str): return True
    if len(text) < 3: return True
    
    tokens = word_tokenize(text, engine="newmm")
    # Filter out whitespace/junk tokens
    valid_tokens = [t for t in tokens if t.strip() and len(t) > 1]
    
    if len(valid_tokens) == 0: return True
    
    # Check for repetitive spam (e.g., "aaaaa")
    if len(set(valid_tokens)) == 1 and len(valid_tokens) > 3:
        return True
        
    return False

# --- C. MAIN PRE-PROCESSING ---

def preprocess_data(df):
    print("Starts Pre-processing & Cleaning...")
    
    # 1. Parse Coordinates
    coords = df['coords'].apply(parse_coords)
    df['latitude'] = coords.apply(lambda x: x[0])
    df['longitude'] = coords.apply(lambda x: x[1])
    
    # 2. Drop invalid locations
    df = df.dropna(subset=['latitude', 'longitude'])
    
    # 3. Clean Text & District
    mapping = get_district_mapping()
    df['district_clean'] = df['district'].apply(lambda x: standardize_district(x, mapping))
    
    # 4. Filter Nonsense Comments
    # We keep the row but flag it, or drop it. Here we drop for cleaner analytics.
    df['is_spam'] = df['comment'].apply(is_nonsense_comment)
    df_clean = df[~df['is_spam']].copy()
    
    # 5. Drop Unused Columns (as per notebook)
    cols_to_drop = ['star', 'photo', 'photo_after', 'is_spam']
    df_clean = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns])
    
    print(f"Data cleaned. Rows remaining: {len(df_clean)} (dropped {len(df) - len(df_clean)})")
    return df_clean

# --- D. PIPELINE ENTRY POINT ---

def process_new_data(new_csv_path, master_parquet_path, poi_csv_path):
    # 1. Load
    try:
        new_df = pd.read_csv(new_csv_path)
    except FileNotFoundError:
        return None

    # Load POI
    try:
        poi_df = pd.read_csv(poi_csv_path)
    except FileNotFoundError:
        poi_df = pd.DataFrame(columns=['lat', 'lon'])

    # 2. PRE-PROCESS (The new logic)
    clean_df = preprocess_data(new_df)
    
    if clean_df.empty:
        print("No valid data after cleaning.")
        return None

    # 3. AI SCORING (Uses the clean data)
    final_df = calculate_final_urgency(clean_df, poi_df)

    # 4. UPSERT / SAVE
    try:
        master_df = pd.read_parquet(master_parquet_path)
        new_ids = final_df['ticket_id'].unique()
        master_df = master_df[~master_df['ticket_id'].isin(new_ids)]
        updated_master = pd.concat([master_df, final_df], ignore_index=True)
    except (FileNotFoundError, OSError):
        updated_master = final_df

    os.makedirs(os.path.dirname(master_parquet_path), exist_ok=True)
    updated_master.to_parquet(master_parquet_path)
    
    return updated_master