import pandas as pd
import os
from scripts.model_logic import calculate_final_urgency

def process_new_data(new_csv_path, master_parquet_path, poi_csv_path):
    """
    Reads new CSV data, runs AI enrichment, and merges it into the Master Parquet file.
    """
    # --- 1. Load Data ---
    print(f"Loading new data from {new_csv_path}...")
    try:
        new_df = pd.read_csv(new_csv_path)
    except FileNotFoundError:
        print("New data file not found. Skipping.")
        return None

    # Load POI data (Schools/Hospitals)
    try:
        poi_df = pd.read_csv(poi_csv_path)
    except FileNotFoundError:
        # Fallback if POI file is missing, create empty structure to prevent crash
        print("Warning: POI data not found. Distance calculations will be 0.")
        poi_df = pd.DataFrame(columns=['lat', 'lon', 'name'])

    # --- 2. Preprocessing ---
    # Convert 'coords' string "13.75,100.50" to float columns
    # We use errors='coerce' to turn bad data into NaN
    new_df['latitude'] = pd.to_numeric(new_df['coords'].str.split(',').str[1], errors='coerce')
    new_df['longitude'] = pd.to_numeric(new_df['coords'].str.split(',').str[0], errors='coerce')
    
    # Ensure timestamp is datetime
    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], errors='coerce')

    # --- 3. Run AI & Urgency Logic ---
    print("Running AI Classification & Urgency Scoring...")
    # This calls the function we defined in scripts/model_logic.py
    processed_df = calculate_final_urgency(new_df, poi_df)

    # --- 4. Save/Merge Logic (The part you requested) ---
    print(f"Merging {len(processed_df)} new records into master dataset...")
    
    try:
        # Try to read the existing master file
        master_df = pd.read_parquet(master_parquet_path)
        
        # UPSERT STRATEGY:
        # 1. Identify IDs in the new batch
        new_ids = processed_df['ticket_id'].unique()
        
        # 2. Remove rows from Master that match these IDs (to avoid duplicates)
        # This effectively "updates" old tickets with the fresh version
        master_df = master_df[~master_df['ticket_id'].isin(new_ids)]
        
        # 3. Concatenate the filtered Master with the Fresh Data
        updated_master_df = pd.concat([master_df, processed_df], ignore_index=True)
        
    except (FileNotFoundError, OSError):
        # If master file doesn't exist (first run), the new data IS the master
        print("Master dataset not found or empty. Creating a new one.")
        updated_master_df = processed_df

    # Save back to Parquet
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(master_parquet_path), exist_ok=True)
    
    updated_master_df.to_parquet(master_parquet_path)
    
    print(f"Merge complete. Master dataset now has {len(updated_master_df)} records.")
    return updated_master_df