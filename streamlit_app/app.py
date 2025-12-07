import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import os
from datetime import datetime, timedelta
from sklearn.neighbors import KernelDensity

st.set_page_config(page_title="Bangkok Traffy Fondue", layout="wide")
st.title("Bangkok Traffy Fondue Data Visualization")

# Load and prepare data
@st.cache_data
def load_data():
    parquet_path = 'data/master_dataset.parquet'
    
    if os.path.exists(parquet_path):
        data = pd.read_parquet(parquet_path)
    else:
        st.warning(f"Data file not found at {parquet_path}. Please ensure the Airflow DAG has run.")
        data = pd.DataFrame()

    # --- FIX 1: Remove Duplicate Columns ---
    # This is the magic line that fixes your current error.
    # It keeps the first occurrence of a column name and drops the rest.
    data = data.loc[:, ~data.columns.duplicated()]

    # --- FIX 2: Ensure Lat/Lon are Numeric ---
    # Sometimes they load as text strings, which breaks the math.
    # We force them to be numbers (coercing errors to NaN).
    if 'lat' in data.columns:
        data['lat'] = pd.to_numeric(data['lat'], errors='coerce')
    if 'lon' in data.columns:
        data['lon'] = pd.to_numeric(data['lon'], errors='coerce')

    # --- FIX 3: Add Missing Columns (Safety Defaults) ---
    required_defaults = {
        'lat': np.nan, 
        'lon': np.nan, 
        'predicted_category': 'Unspecified',
        'district': 'Unknown',
        'state': 'New',
        'urgency_score': 0.0,
        'ticket_id': 'Unknown',
        'organization': '-',
        'comment': '-',
        'address': '-',
        'timestamp': pd.Timestamp.now()
    }

    for col, default_val in required_defaults.items():
        if col not in data.columns:
            data[col] = default_val

    # Clean up coordinates
    data.dropna(subset=['lat', 'lon'], inplace=True)
    
    return data

# Load data
data = load_data()

## Sidebar code
st.sidebar.header("Filters")

# Multiselect for districts
districts = data['district'].unique().tolist()
selected_districts = st.sidebar.multiselect(
    "Districts",
    options=districts,
    default=[]
)

#multiselect for incident types
predicted_category = data['predicted_category'].unique().tolist()
selected_predicted_category = st.sidebar.multiselect(
    "Incident Types",
    options=predicted_category,
    default=[]
)

# Urgency Score filter
default_urgency_score = 1.0
urgency_score_slider = st.sidebar.slider(
    "Urgency Score",
    min_value=0.0001,
    max_value=1.0000,
    value=default_urgency_score,
    step=0.0001
)

# Date range filter
data['timestamp'] = pd.to_datetime(data['timestamp'], format='mixed', errors='coerce')

# Filter out NaT values for date range calculation
valid_dates = data['timestamp'].dropna()

if len(valid_dates) > 0:
    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()
    selected_date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
else:
    # Default date range if no valid dates found
    default_date = datetime.now().date()
    selected_date_range = st.sidebar.date_input(
        "Date Range",
        value=(default_date - timedelta(days=30), default_date),
        min_value=default_date - timedelta(days=365),
        max_value=default_date
    )

### Main panel code
st.header("Traffic Incident Map")

# Filter data based on selection
filtered_data = data.copy()
if selected_districts:
    filtered_data = filtered_data[filtered_data['district'].isin(selected_districts)]

if selected_predicted_category:
    filtered_data = filtered_data[filtered_data['predicted_category'].isin(selected_predicted_category)]

filtered_data = filtered_data[
    (filtered_data['urgency_score'] <= urgency_score_slider) &
    (filtered_data['timestamp'].dt.date >= selected_date_range[0]) &
    (filtered_data['timestamp'].dt.date <= selected_date_range[1])
]

# Main content
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Incidents", len(filtered_data))
with col2:
    st.metric("Waiting", len(filtered_data[filtered_data['state'] == 'รอดำเนินการ']))
with col3:
    st.metric("On going", len(filtered_data[filtered_data['state'] == 'กำลังดำเนินการ']))
with col4:
    st.metric("Done", len(filtered_data[filtered_data['state'] == 'เสร็จสิ้น']))
    
# Urgency Score Map
    
# Create color mapping based on urgency score (green = low, red = high)
# Use absolute scale: 0.0 = green, 1.0 = red
def urgency_to_color(urgency_score):
    # Normalize to 0-1 using fixed bounds (0.0 to 1.0)
    normalized = min(max(urgency_score, 0.0), 1.0)  # Clamp to [0, 1]
    r = int(normalized * 255)
    g = int((1 - normalized) * 255)
    b = 0
    return [r, g, b, 160]  # Added alpha channel
    
# Apply color mapping using absolute urgency score values
if len(filtered_data) > 0:
    filtered_data['urgency_color'] = filtered_data['urgency_score'].apply(urgency_to_color)
    filtered_data['timestamp_formatted'] = filtered_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    filtered_data['urgency_score_formatted'] = filtered_data['urgency_score'].apply(lambda x: f"{x:.4f}")

    # Create scatter layer for urgency score
    urgency_layer = pdk.Layer(
        "ScatterplotLayer",
        filtered_data,
        get_position=['lon', 'lat'],
        get_color='urgency_color',
        get_radius=100,
        pickable=True,
    )
    st.pydeck_chart(
        pdk.Deck(
            layers=[urgency_layer],
            initial_view_state=pdk.ViewState(
                latitude=float(filtered_data['lat'].mean()),
                longitude=float(filtered_data['lon'].mean()),
                zoom=12,
                pitch=0
            ),
            map_style='light',
            tooltip={
                "html":
                    "<b>Ticket ID:</b> {ticket_id}<br/>"
                    "<b>Organization:</b> {organization}<br/>"
                    "<b>Comment:</b> {comment}<br/>"
                    "<b>Address:</b> {address}<br/>"
                    "<b>Timestamp:</b> {timestamp_formatted}<br/>"
                    "<b>Predicted Category:</b> {predicted_category}<br/>"
                    "<b>Urgency Score:</b> {urgency_score_formatted}<br/>"
                    "<b>State:</b> {state}"
            }
        ),
        height=600
    )

    # Display urgency statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Min Urgency Score", f"{filtered_data['urgency_score'].min():.4f}")
    with col2:
        st.metric("Mean Urgency Score", f"{filtered_data['urgency_score'].mean():.4f}")
    with col3:
        st.metric("Max Urgency Score", f"{filtered_data['urgency_score'].max():.4f}")
    
    # View all highest urgency score locations with pagination
    st.subheader('All Incidents Sorted by Urgency Score')
    
    # Prepare all urgency data sorted by urgency score
    all_urgency = filtered_data[
        ['ticket_id', 'organization', 'comment', 'address', 'timestamp_formatted', 'state', 'predicted_category', 'urgency_score']
    ].sort_values('urgency_score', ascending=False).reset_index(drop=True)
    all_urgency['urgency_score'] = all_urgency['urgency_score'].apply(lambda x: f"{x:.4f}")
    
    # Pagination setup
    items_per_page = 10
    total_items = len(all_urgency)
    total_pages = (total_items + items_per_page - 1) // items_per_page
    
    # Initialize session state for page number
    if 'urgency_page' not in st.session_state:
        st.session_state.urgency_page = 0
    
    # Display current page
    start_idx = st.session_state.urgency_page * items_per_page
    end_idx = start_idx + items_per_page
    page_data = all_urgency.iloc[start_idx:end_idx]
    
    st.dataframe(page_data, width='stretch', height=400)
    
    # Pagination controls
    col1, col2, col3 = st.columns([2, 7, 1])
    with col1:
        if st.button("← Previous", key="prev_urgency"):
            if st.session_state.urgency_page > 0:
                st.session_state.urgency_page -= 1
    with col2:
        st.markdown(f"<div style='text-align: center'>Page {st.session_state.urgency_page + 1} of {total_pages} (Total: {total_items} incidents)</div>", unsafe_allow_html=True)
    with col3:
        if st.button("Next →", key="next_urgency"):
            if st.session_state.urgency_page < total_pages - 1:
                st.session_state.urgency_page += 1

# Kernel Density Estimation Analysis
st.subheader("Kernel Density Estimation (KDE) Statistics")

try:
    # Prepare coordinates
    coords = filtered_data[['lat', 'lon']].values

    # Fit KDE model
    kde = KernelDensity(bandwidth=0.005, kernel='gaussian')
    kde.fit(coords)
    
    # Calculate density score for each point
    log_density = kde.score_samples(coords)
    density = np.exp(log_density)
    
    # Add density scores to dataframe
    filtered_data['density'] = density
    filtered_data['density_normalized'] = (density - density.min()) / (density.max() - density.min())
    
    # Format density and timestamp for tooltip
    filtered_data['density_formatted'] = filtered_data['density'].apply(lambda x: f"{x:.4f}")
    
    # Create color mapping based on density (blue = low, red = high)
    def density_to_color(density_normalized):
        r = int(density_normalized * 255)
        g = int((1 - abs(2 * density_normalized - 1)) * 255)
        b = int((1 - density_normalized) * 255)
        return [r, g, b, 160]  # Added alpha channel
    
    # Apply color mapping
    filtered_data['density_color'] = filtered_data['density_normalized'].apply(density_to_color)
 
    # Create scatter layer for KDE
    kde_layer = pdk.Layer(
        "ScatterplotLayer",
        filtered_data,
        get_position=['lon', 'lat'],
        get_color='density_color',
        get_radius=100,
        pickable=True,
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[kde_layer],
            initial_view_state=pdk.ViewState(
                latitude=float(filtered_data['lat'].mean()),
                longitude=float(filtered_data['lon'].mean()),
                zoom=12,
                pitch=0
            ),
            map_style='light',
            tooltip={
                "html":
                    "<b>Ticket ID:</b> {ticket_id}<br/>"
                    "<b>Organization:</b> {organization}<br/>"
                    "<b>Comment:</b> {comment}<br/>"
                    "<b>Address:</b> {address}<br/>"
                    "<b>Timestamp:</b> {timestamp_formatted}<br/>"
                    "<b>Predicted Category:</b> {predicted_category}<br/>"
                    "<b>Urgency Score:</b> {urgency_score_formatted}<br/>"
                    "<b>State:</b> {state}"
            }
        ),
        height=600
    )

    # Display statistics for KDE
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Min Density", f"{filtered_data['density'].min():.4f}")
    with col2:
        st.metric("Mean Density", f"{filtered_data['density'].mean():.4f}")
    with col3:
        st.metric("Max Density", f"{filtered_data['density'].max():.4f}")

except Exception as e:
    st.error(f"Error in KDE analysis: {e}")