# scripts/scraping_logic.py
import requests
import pandas as pd
import os
import time

def query_overpass(query):
    url = "https://overpass-api.de/api/interpreter"
    try:
        response = requests.post(url, data={'data': query}, timeout=120)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Overpass API Error: {e}")
        return {}

def get_bangkok_districts():
    # Helper to get the area ID for Bangkok
    bankok_osm_id = 92277
    return 3600000000 + bankok_osm_id

def scrape_poi_category(category_key, category_value, area_id):
    """
    Generic function to scrape any category (hospital, school, etc.)
    """
    print(f"Scraping {category_value}...")
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
        
        # Handle nodes vs ways (buildings)
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
    """
    Main function to run the scraping and save the CSV
    """
    # Check if file exists and is recent (e.g., < 30 days old)
    # We don't want to spam the API every day.
    if os.path.exists(output_path):
        file_age_days = (time.time() - os.path.getmtime(output_path)) / (60 * 60 * 24)
        if file_age_days < 30:
            print(f"POI Data is fresh ({file_age_days:.1f} days old). Skipping scrape.")
            return

    area_id = get_bangkok_districts()
    
    all_pois = []
    # Scrape Hospitals
    all_pois.extend(scrape_poi_category("amenity", "hospital", area_id))
    # Scrape Schools
    all_pois.extend(scrape_poi_category("amenity", "school", area_id))
    # Scrape Universities
    all_pois.extend(scrape_poi_category("amenity", "university", area_id))
    # Scrape Temples (optional, based on your notebook)
    all_pois.extend(scrape_poi_category("amenity", "place_of_worship", area_id))

    df = pd.DataFrame(all_pois)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully scraped {len(df)} POIs to {output_path}")