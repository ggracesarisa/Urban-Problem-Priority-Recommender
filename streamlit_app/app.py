import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import os
from datetime import datetime, timedelta
from sklearn.neighbors import KernelDensity

st.set_page_config(page_title="Bangkok Traffy Fondue", layout="wide")
st.title("Bangkok Traffy Fondue Data Visualization")

MAX_POINTS_MAP = 50000
MAX_POINTS_KDE = 50000  
ITEMS_PER_PAGE = 10

@st.cache_data
def load_data():
    parquet_path = 'data/master_dataset.parquet'
    
    if os.path.exists(parquet_path):
        data = pd.read_parquet(parquet_path)
    else:
        st.warning(f"Data file not found at {parquet_path}. Please ensure the Airflow DAG has run.")
        data = pd.DataFrame()

    data = data.loc[:, ~data.columns.duplicated()]

    if 'lat' in data.columns:
        data['lat'] = pd.to_numeric(data['lat'], errors='coerce')
    if 'lon' in data.columns:
        data['lon'] = pd.to_numeric(data['lon'], errors='coerce')

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

    # limit comment length globally
    data['comment'] = data['comment'].astype(str).str.slice(0, 300)

    data.dropna(subset=['lat', 'lon'], inplace=True)
    
    return data

data = load_data()

st.sidebar.header("Filters")

districts = data['district'].unique().tolist()
selected_districts = st.sidebar.multiselect(
    "Districts",
    options=districts,
    default=[]
)

predicted_category = data['predicted_category'].unique().tolist()
selected_predicted_category = st.sidebar.multiselect(
    "Incident Types",
    options=predicted_category,
    default=[]
)

default_urgency_score = 1.0
urgency_score_slider = st.sidebar.slider(
    "Urgency Score (max)",
    min_value=0.0001,
    max_value=1.0000,
    value=default_urgency_score,
    step=0.0001
)

# zoom control for map
zoom_level = st.sidebar.slider(
    "Map zoom level",
    min_value=8,
    max_value=15,
    value=12
)

data['timestamp'] = pd.to_datetime(data['timestamp'], format='mixed', errors='coerce')
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
    default_date = datetime.now().date()
    selected_date_range = st.sidebar.date_input(
        "Date Range",
        value=(default_date - timedelta(days=30), default_date),
        min_value=default_date - timedelta(days=365),
        max_value=default_date
    )

st.header("Traffic Incident Map")

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

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Incidents", len(filtered_data))
with col2:
    st.metric("Waiting", len(filtered_data[filtered_data['state'] == 'รอดำเนินการ']))
with col3:
    st.metric("On going", len(filtered_data[filtered_data['state'] == 'กำลังดำเนินการ']))
with col4:
    st.metric("Done", len(filtered_data[filtered_data['state'] == 'เสร็จสิ้น']))

def urgency_to_color(urgency_score: float):
    normalized = min(max(float(urgency_score), 0.0), 1.0)
    r = int(normalized * 255)
    g = int((1 - normalized) * 255)
    b = 0
    return [r, g, b, 160]

if len(filtered_data) > 0:
    n = len(filtered_data)

    # adaptive, but always capped
    if zoom_level <= 9:
        max_points = min(2000, n)
    elif zoom_level <= 11:
        max_points = min(5000, n)
    elif zoom_level <= 13:
        max_points = min(10000, n)
    else:
        max_points = min(MAX_POINTS_MAP, n)

    # always sample, never send full 98k
    map_data = filtered_data.sample(max_points, random_state=42).copy()
    st.caption(f"Showing {max_points} incidents at this zoom level out of {n} filtered incidents.")

    map_data['urgency_color'] = map_data['urgency_score'].apply(urgency_to_color)
    map_data['timestamp_formatted'] = map_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    map_data['urgency_score_formatted'] = map_data['urgency_score'].apply(lambda x: f"{x:.4f}")
    map_data['comment_short'] = map_data['comment'].astype(str).str.slice(0, 120)

    # only keep columns needed by the map + tooltip
    map_data = map_data[[
        'lat', 'lon',
        'urgency_color',
        'ticket_id', 'organization',
        'comment_short', 'address',
        'timestamp_formatted',
        'predicted_category',
        'urgency_score_formatted',
        'state'
    ]]

    urgency_layer = pdk.Layer(
        "ScatterplotLayer",
        map_data,
        get_position=['lon', 'lat'],
        get_color='urgency_color',
        get_radius=100,
        pickable=True,
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[urgency_layer],
            initial_view_state=pdk.ViewState(
                latitude=float(map_data['lat'].mean()),
                longitude=float(map_data['lon'].mean()),
                zoom=zoom_level,
                pitch=0
            ),
            map_style='light',
            tooltip={
                "html":
                    "<b>Ticket ID:</b> {ticket_id}<br/>"
                    "<b>Organization:</b> {organization}<br/>"
                    "<b>Comment:</b> {comment_short}<br/>"
                    "<b>Address:</b> {address}<br/>"
                    "<b>Timestamp:</b> {timestamp_formatted}<br/>"
                    "<b>Predicted Category:</b> {predicted_category}<br/>"
                    "<b>Urgency Score:</b> {urgency_score_formatted}<br/>"
                    "<b>State:</b> {state}"
            }
        ),
        height=600
    )


    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Min Urgency Score", f"{filtered_data['urgency_score'].min():.4f}")
    with col2:
        st.metric("Mean Urgency Score", f"{filtered_data['urgency_score'].mean():.4f}")
    with col3:
        st.metric("Max Urgency Score", f"{filtered_data['urgency_score'].max():.4f}")

    st.subheader('All Incidents Sorted by Urgency Score')

    all_urgency = filtered_data[
        ['ticket_id', 'organization', 'comment', 'address', 'timestamp', 'state', 'predicted_category', 'urgency_score']
    ].copy()

    all_urgency['timestamp_formatted'] = all_urgency['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    all_urgency['urgency_score'] = all_urgency['urgency_score'].apply(lambda x: f"{x:.4f}")
    all_urgency['comment_short'] = all_urgency['comment'].astype(str).str.slice(0, 120)

    all_urgency = all_urgency[
        ['ticket_id', 'organization', 'comment_short', 'address', 'timestamp_formatted', 'state', 'predicted_category', 'urgency_score']
    ].sort_values('urgency_score', ascending=False).reset_index(drop=True)

    total_items = len(all_urgency)
    total_pages = (total_items + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE

    if 'urgency_page' not in st.session_state:
        st.session_state.urgency_page = 0

    start_idx = st.session_state.urgency_page * ITEMS_PER_PAGE
    end_idx = start_idx + ITEMS_PER_PAGE
    page_data = all_urgency.iloc[start_idx:end_idx]

    st.dataframe(page_data, use_container_width=True, height=400)

    col1, col2, col3 = st.columns([2, 7, 1])
    with col1:
        if st.button("← Previous", key="prev_urgency"):
            if st.session_state.urgency_page > 0:
                st.session_state.urgency_page -= 1
    with col2:
        st.markdown(
            f"<div style='text-align: center'>Page {st.session_state.urgency_page + 1} of {total_pages} "
            f"(Total: {total_items} incidents)</div>",
            unsafe_allow_html=True
        )
    with col3:
        if st.button("Next →", key="next_urgency"):
            if st.session_state.urgency_page < total_pages - 1:
                st.session_state.urgency_page += 1

st.subheader("Kernel Density Estimation (KDE) Statistics")

if len(filtered_data) < 2:
    st.info("KDE Analysis requires at least 2 data points. Please select a larger date range or more districts.")
else:
    try:
        # decide sample size based on zoom, also capped by MAX_POINTS_KDE
        n_kde = len(filtered_data)
        if zoom_level <= 9:
            max_kde_points = min(2000, n_kde, MAX_POINTS_KDE)
        elif zoom_level <= 11:
            max_kde_points = min(5000, n_kde, MAX_POINTS_KDE)
        elif zoom_level <= 13:
            max_kde_points = min(10000, n_kde, MAX_POINTS_KDE)
        else:
            max_kde_points = min(MAX_POINTS_KDE, n_kde)

        if n_kde > max_kde_points:
            kde_source = filtered_data.sample(max_kde_points, random_state=42).copy()
            st.caption(
                f"KDE computed on a sample of {max_kde_points} incidents "
                f"out of {n_kde} filtered incidents."
            )
        else:
            kde_source = filtered_data.copy()
            st.caption(f"KDE computed on all {n_kde} filtered incidents.")

        coords = kde_source[['lat', 'lon']].values

        kde = KernelDensity(bandwidth=0.005, kernel='gaussian')
        kde.fit(coords)

        log_density = kde.score_samples(coords)
        density = np.exp(log_density)

        d_min = float(density.min())
        d_max = float(density.max())

        if d_max == d_min:
            density_normalized = np.full_like(density, 0.5)
        else:
            density_normalized = (density - d_min) / (d_max - d_min)

        kde_source['density'] = density
        kde_source['density_normalized'] = density_normalized
        kde_source['density_formatted'] = kde_source['density'].apply(lambda x: f"{x:.4f}")

        def density_to_color(dn):
            if pd.isna(dn):
                return [0, 0, 255, 160]
            dn = float(dn)
            r = int(dn * 255)
            g = int((1 - abs(2 * dn - 1)) * 255)
            b = int((1 - dn) * 255)
            return [r, g, b, 160]

        kde_source['density_color'] = kde_source['density_normalized'].apply(density_to_color)

        kde_layer = pdk.Layer(
            "ScatterplotLayer",
            kde_source,
            get_position=['lon', 'lat'],
            get_color='density_color',
            get_radius=100,
            pickable=True,
        )

        st.pydeck_chart(
            pdk.Deck(
                layers=[kde_layer],
                initial_view_state=pdk.ViewState(
                    latitude=float(kde_source['lat'].mean()),
                    longitude=float(kde_source['lon'].mean()),
                    zoom=zoom_level,   # same zoom slider
                    pitch=0
                ),
                map_style='light',
                tooltip={
                    "html":
                        "<b>Ticket ID:</b> {ticket_id}<br/>"
                        "<b>Category:</b> {predicted_category}<br/>"
                        "<b>Density Score:</b> {density_formatted}"
                }
            ),
            height=600
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Min Density", f"{d_min:.4f}")
        with col2:
            st.metric("Mean Density", f"{float(density.mean()):.4f}")
        with col3:
            st.metric("Max Density", f"{d_max:.4f}")

    except Exception as e:
        st.error(f"Error in KDE analysis: {e}")
