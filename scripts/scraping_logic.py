import os
import time

def query_overpass(query):
    # Import requests here to avoid top-level load
    import requests
    
    url = "https://overpass-api.de/api/interpreter"
    
    # Header is CRITICAL. Without it, Overpass might silently ignore/block you.
    headers = {
        'User-Agent': 'TraffyFonduePipeline/1.0 (contact@example.com)' 
    }
    
    try:
        response = requests.post(url, data={'data': query}, headers=headers, timeout=120)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Overpass API Error: {e}")
        return {}

def get_bangkok_districts():
    bankok_osm_id = 92277
    return 3600000000 + bankok_osm_id

def scrape_poi_category(category_key, category_value, area_id):
    print(f"Scraping {category_value}...")
    
    # Wait 2 seconds to be polite to the API (prevents rate limiting)
    time.sleep(2) 
    
    query = f"""
    [out:json][timeout:120];
    area({area_id})->.searchArea;
    (
        node["{category_key}"="{category_value}"](area.searchArea);
        way["{category_key}"="{category_value}"](area.searchArea);
        relation["{category_key}"="{category_value}"](area.searchArea);
    );
    out center;
    """
    data = query_overpass(query)
    results = []
    
    for elem in data.get("elements", []):
        name = elem.get("tags", {}).get("name:en") or elem.get("tags", {}).get("name")
        lat, lon = None, None
        
        if elem["type"] == "node":
            lat, lon = elem.get("lat"), elem.get("lon")
        elif "center" in elem:
            lat, lon = elem["center"].get("lat"), elem["center"].get("lon")
            
        if lat and lon:
            results.append({
                "name": name or f"Unknown {category_value}",
                "lat": lat,
                "lon": lon,
                "type": category_value
            })
    return results

def update_poi_data(output_path):
    import pandas as pd
    
    print("--- [Scraper] Starting POI Update ---")
    
    # 1. Freshness Check
    if os.path.exists(output_path):
        file_age_days = (time.time() - os.path.getmtime(output_path)) / (60 * 60 * 24)
        if file_age_days < 30:
            print(f"POI Data is fresh ({file_age_days:.1f} days old). Skipping scrape.")
            return

    # 2. Scrape
    area_id = get_bangkok_districts()
    all_pois = []
    
    categories = [
        ("amenity", "hospital"),
        ("amenity", "school"),
        ("amenity", "university"),
        ("amenity", "place_of_worship")
    ]
    
    for key, val in categories:
        all_pois.extend(scrape_poi_category(key, val, area_id))

    # 3. Save
    if all_pois:
        df = pd.DataFrame(all_pois)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Successfully scraped {len(df)} POIs to {output_path}")
    else:
        print("Warning: No POI data was scraped (API might be down).")