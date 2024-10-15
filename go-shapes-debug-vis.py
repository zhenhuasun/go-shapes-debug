import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import Search
import plotly.graph_objects as go

# Constants for Earth's radius (in meters)
EARTH_RADIUS = 6371000  # meters

# Approximate distance calculation using lat/lon differences (ignores Earth's curvature)
def fast_distance_approx(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    dx = dlon * np.cos((lat1 + lat2) / 2)
    dy = dlat
    return np.sqrt(dx * dx + dy * dy) * EARTH_RADIUS

# Streamlit app layout
st.title("Shape File Outlier Detection Tool")

# File uploader
uploaded_file = st.file_uploader("Please upload shapes.txt from GO Transit GTFS file", type="txt")

if uploaded_file:
    # Load the shapes.txt file into a DataFrame
    shapes = pd.read_csv(uploaded_file)

    # Sort by shape_id and shape_pt_sequence
    shapes = shapes.sort_values(by=['shape_id', 'shape_pt_sequence'])
    
    # Shift lat/lon columns to compare distances between consecutive points
    shapes['next_lat'] = shapes.groupby('shape_id')['shape_pt_lat'].shift(-1)
    shapes['next_lon'] = shapes.groupby('shape_id')['shape_pt_lon'].shift(-1)

    # Approximate distances between consecutive points
    shapes['distance_to_next'] = fast_distance_approx(
        shapes['shape_pt_lat'], shapes['shape_pt_lon'],
        shapes['next_lat'], shapes['next_lon']
    )

    # Step 1: Sigma-level analysis and Plotly chart
    st.subheader("Sigma-level Analysis")

    # Add a description of the plotly chart
    st.markdown("""
    **Description:**
    The chart below shows the number of shape IDs identified as potential outliers at each sigma level. 
    Sigma levels represent thresholds based on the mean and standard deviation of the distance between consecutive points in each shape. 
    As the sigma level increases, more leniency is applied, leading to fewer identified outliers. 
    The hover text includes the count of problematic shape IDs and their corresponding values at each sigma level.
    """)

    # Dictionary to store the number of problematic shape IDs per sigma level
    shape_id_problem_counts = {}
    sigma = 1

    while True:
        # Calculate mean and standard deviation of distances
        mean_distance = shapes['distance_to_next'].mean()
        std_distance = shapes['distance_to_next'].std()

        # Threshold for outliers
        threshold = mean_distance + sigma * std_distance
        shapes['is_outlier'] = shapes['distance_to_next'] > threshold

        # Identify problematic shape IDs
        potential_outliers = shapes[shapes['is_outlier']]
        problematic_shape_ids = potential_outliers['shape_id'].unique()

        # Store the results for the current sigma level
        shape_id_problem_counts[sigma] = {
            'count': len(problematic_shape_ids),
            'problematic_ids': problematic_shape_ids.tolist()
        }

        if len(problematic_shape_ids) == 0:
            break
        sigma += 1

    # Convert to DataFrame for Plotly
    df_shape_id_problems = pd.DataFrame.from_dict(
        shape_id_problem_counts, orient='index'
    ).reset_index().rename(columns={'index': 'Sigma'})

    sigma_values = df_shape_id_problems['Sigma']
    counts = df_shape_id_problems['count']
    shape_ids_list = df_shape_id_problems['problematic_ids']

    # Plotly chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sigma_values,
        y=counts,
        mode='lines',
        name='Shape IDs with Problems',
        text=[f"Count: {count}<br>Shape IDs: {', '.join(map(str, shape_ids))}"
              for count, shape_ids in zip(counts, shape_ids_list)],
        hoverinfo='text'
    ))

    # Update layout
    fig.update_layout(
        title='Number of Shape IDs with Potential Problems by Sigma Level',
        xaxis_title='Sigma Level',
        yaxis_title='Number of Shape IDs with Potential Problems',
        hovermode='x unified'
    )

    # Display the interactive chart
    st.plotly_chart(fig)

    # Step 2: Map Visualization with Folium
    st.subheader("Map Visualization of Shape IDs")

    # Create a Folium map centered on the average lat/lon of the shapes
    avg_lat = shapes['shape_pt_lat'].mean()
    avg_lon = shapes['shape_pt_lon'].mean()
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)

    # Create GeoJSON features for shape_id polylines
    features = []
    for shape_id, group in shapes.groupby('shape_id'):
        coordinates = [[lon, lat] for lat, lon in zip(group['shape_pt_lat'], group['shape_pt_lon'])]
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': coordinates,
            },
            'properties': {
                'shape_id': shape_id
            }
        }
        features.append(feature)

    # Create a GeoJson layer from the features
    geojson_layer = folium.GeoJson(
        {'type': 'FeatureCollection', 'features': features},
        name="Shapes",
        tooltip=folium.GeoJsonTooltip(fields=["shape_id"], aliases=["Shape ID"], localize=True)
    ).add_to(m)

    # Add search functionality based on shape_id
    search = Search(
        layer=geojson_layer,
        geom_type="LineString",
        placeholder="Search for Shape ID",
        search_label="shape_id",
        collapsed=False,
        weight=5,
        color='red', 
        opacity=0.8,
        fill=False
    ).add_to(m)

    # Display the Folium map in Streamlit
    st_folium(m, width=700, height=500)
