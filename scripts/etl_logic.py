import os
import sys
import pandas as pd
import numpy as np

# --- HELPER FUNCTIONS ---

def get_district_mapping():
    return {
        "พระนคร": "Phra Nakhon", "ดุสิต": "Dusit", "หนองจอก": "Nong Chok",
        "บางรัก": "Bang Rak", "บางเขน": "Bang Khen", "บางกะปิ": "Bang Kapi",
        "ปทุมวัน": "Pathum Wan", "ป้อมปราบศัตรูพ่าย": "Pom Prap Sattru Phai",
        "พระโขนง": "Phra Khanong", "มีนบุรี": "Min Buri", "ลาดกระบัง": "Lat Krabang",
        "ยานนาวา": "Yan Nawa", "สัมพันธวงศ์": "Samphanthawong", "พญาไท": "Phaya Thai",
        "ธนบุรี": "Thon Buri", "บางกอกใหญ่": "Bangkok Yai", "ห้วยขวาง": "Huai Khwang",
        "คลองสาน": "Khlong San", "ตลิ่งชัน": "Taling Chan", "บางกอกน้อย": "Bangkok Noi",
        "บางขุนเทียน": "Bang Khun Thian", "ภาษีเจริญ": "Phasi Charoen", "หนองแขม": "Nong Khaem",
        "ราษฎร์บูรณะ": "Rat Burana", "บางพลัด": "Bang Phlat", "ดินแดง": "Din Daeng",
        "บึงกุ่ม": "Bueng Kum", "สาทร": "Sathon", "บางซื่อ": "Bang Sue",
        "จตุจักร": "Chatuchak", "บางคอแหลม": "Bang Kho Laem", "ประเวศ": "Prawet",
        "คลองเตย": "Khlong Toei", "สวนหลวง": "Suan Luang", "จอมทอง": "Chom Thong",
        "ดอนเมือง": "Don Mueang", "ราชเทวี": "Ratchathewi", "ลาดพร้าว": "Lat Phrao",
        "วัฒนา": "Watthana", "บางแค": "Bang Khae", "หลักสี่": "Lak Si",
        "สายไหม": "Sai Mai", "คันนายาว": "Khan Na Yao", "สะพานสูง": "Saphan Sung",
        "วังทองหลาง": "Wang Thonglang", "คลองสามวา": "Khlong Sam Wa", "บางนา": "Bang Na",
        "ทวีวัฒนา": "Thawi Watthana", "ทุ่งครุ": "Thung Khru", "บางบอน": "Bang Bon"
    }

def assign_predicted_category(row):
    """
    [NEW] Define your logic here to fix the 'Wrong' type data.
    """
    original_type = str(row.get('type', '')).strip()
    comment = str(row.get('comment', '')).lower()

    # Example 1: Map specific Thai types to English Categories
    if "ความสะอาด" in original_type: return "Cleanliness"
    if "ไฟฟ้า" in original_type or "แสงสว่าง" in original_type: return "Lighting"
    if "ถนน" in original_type or "ทางเท้า" in original_type: return "Roads"
    if "น้ำท่วม" in original_type or "ระบายน้ำ" in original_type: return "Flooding"
    if "จราจร" in original_type: return "Traffic"
    if "สายไฟ" in original_type: return "Cables"
    if "เสียงรบกวน" in original_type: return "Noise"
    
    # Example 2: Fallback to using the 'type' if no match found
    # (Or change this to return 'Other' if you want to be strict)
    return original_type if original_type else "Unspecified"

def parse_coords(val):
    if not isinstance(val, str) or val.strip() == "":
        return np.nan, np.nan
    try:
        if "," in val:
            parts = val.split(",")
            if len(parts) == 2:
                v1, v2 = float(parts[0]), float(parts[1])
                if 5 < v2 < 21 and 97 < v1 < 106: return v2, v1
                elif 5 < v1 < 21 and 97 < v2 < 106: return v1, v2
        return np.nan, np.nan
    except:
        return np.nan, np.nan

def clean_text_helper(text):
    import re
    if not isinstance(text, str): return ""
    text = text.strip()
    text = re.sub(r"แขวง|เขต|กรุงเทพมหานคร|กรุงเทพฯ|กรุงเทพ|จังหวัด", "", text)
    text = re.sub(r"[()\-_,]", " ", text)
    return text.strip()

def standardize_district_single(district_name, mapping):
    from rapidfuzz import process, fuzz
    if not isinstance(district_name, str): return None
    clean_name = clean_text_helper(district_name)
    if clean_name in mapping: return mapping[clean_name]
    match = process.extractOne(clean_name, mapping.keys(), scorer=fuzz.WRatio)
    if match and match[1] >= 80: return mapping[match[0]]
    return None

def is_nonsense_comment(text):
    from pythainlp.tokenize import word_tokenize
    if not isinstance(text, str): return True
    if len(text) < 3: return True
    try:
        tokens = word_tokenize(text, engine="newmm")
        valid_tokens = [t for t in tokens if t.strip() and len(t) > 1]
        if len(valid_tokens) == 0: return True
        if len(set(valid_tokens)) == 1 and len(valid_tokens) > 3: return True
    except Exception:
        return False
    return False

# --- MAIN LOGIC ---

def process_new_data(new_csv_path, master_parquet_path, poi_csv_path):
    from scripts.model_logic import calculate_final_urgency

    print("--- [ETL] Starting Data Processing ---")

    # 1. Load Data
    if not os.path.exists(new_csv_path):
        print(f"File not found: {new_csv_path}")
        return None
    try:
        new_df = pd.read_csv(new_csv_path)
        print(f"Loaded {len(new_df)} rows.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    try:
        poi_df = pd.read_csv(poi_csv_path)
    except FileNotFoundError:
        poi_df = pd.DataFrame(columns=['lat', 'lon'])

    # 2. Coordinates
    print("Parsing Coordinates...")
    coords = new_df['coords'].apply(parse_coords)
    new_df['latitude'] = coords.apply(lambda x: x[0])
    new_df['longitude'] = coords.apply(lambda x: x[1])
    new_df['lat'] = new_df['latitude']
    new_df['lon'] = new_df['longitude']
    new_df = new_df.dropna(subset=['latitude', 'longitude'])

    # 3. Districts
    print("Standardizing Districts...")
    mapping = get_district_mapping()
    unique_districts = new_df['district'].dropna().unique()
    district_cache = {d: standardize_district_single(d, mapping) for d in unique_districts}
    new_df['district_clean'] = new_df['district'].map(district_cache)

    # 4. Spam
    print("Filtering Spam...")
    new_df['is_spam'] = new_df['comment'].apply(is_nonsense_comment)
    clean_df = new_df[~new_df['is_spam']].copy()

    # 5. [CORRECTED] ASSIGN PREDICTED CATEGORY
    # We now apply YOUR logic instead of just copying 'type'
    print("Assigning Predicted Categories...")
    clean_df['predicted_category'] = clean_df.apply(assign_predicted_category, axis=1)

    # Backup the category map using ticket_id
    category_map = clean_df.set_index('ticket_id')['predicted_category'].to_dict()

    # Drop unused
    cols_to_drop = ['star', 'photo', 'photo_after', 'is_spam']
    clean_df = clean_df.drop(columns=[c for c in cols_to_drop if c in clean_df.columns])

    if clean_df.empty:
        print("No valid data remaining.")
        return None

    # 6. AI Scoring
    print("Running AI Scoring Model...")
    final_df = calculate_final_urgency(clean_df, poi_df)

    # Restore Category via Map (in case model dropped it)
    if 'predicted_category' not in final_df.columns:
        print("Restoring 'predicted_category' via Ticket ID mapping...")
        final_df['predicted_category'] = final_df['ticket_id'].map(category_map)
    
    final_df['predicted_category'] = final_df['predicted_category'].fillna('Unspecified')

    # 7. Save
    print(f"Saving {len(final_df)} rows...")
    try:
        if os.path.exists(master_parquet_path):
            master_df = pd.read_parquet(master_parquet_path)
            
            # Clean duplicate columns in Master
            master_df = master_df.loc[:, ~master_df.columns.duplicated()]
            
            if 'latitude' in master_df.columns and 'lat' not in master_df.columns:
                master_df['lat'] = master_df['latitude']
                master_df['lon'] = master_df['longitude']

            new_ids = final_df['ticket_id'].unique()
            master_df = master_df[~master_df['ticket_id'].isin(new_ids)]
            updated_master = pd.concat([master_df, final_df], ignore_index=True)
        else:
            updated_master = final_df
            
        # Deduplicate columns one last time before saving
        updated_master = updated_master.loc[:, ~updated_master.columns.duplicated()]
        
        os.makedirs(os.path.dirname(master_parquet_path), exist_ok=True)
        updated_master.to_parquet(master_parquet_path)
        print("--- [ETL] Success ---")
        return updated_master
        
    except Exception as e:
        print(f"Error saving Parquet: {e}")
        return None