import streamlit as st
import pandas as pd
import pydeck as pdk
from datetime import datetime
import numpy as np
import os
from functools import partial
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

# Set page configuration
st.set_page_config(
    page_title="NYC Crime Analysis",
    page_icon="🔍",
    layout="wide"
)

# Create the main layout columns
left_col, right_col = st.columns([7, 1])
col1, col2 = st.columns([.6, .4])

def get_snowflake_connection():
    """Create a connection to Snowflake or return existing connection from session state"""
    if 'snowflake_conn' not in st.session_state:
        try:
            # Create new connection
            conn = snowflake.connector.connect(
                user='',
                password='',
                account='',
                warehouse='COMPUTE_WH',
                database='NYC_DATA',
                schema='SOURCE_DATA'
            )
            # Store in session state
            st.session_state['snowflake_conn'] = conn
            st.info("Created new Snowflake connection")
        except Exception as e:
            st.error(f"Failed to create Snowflake connection: {str(e)}")
            return None
    else:
        # Check if existing connection is still valid
        try:
            conn = st.session_state['snowflake_conn']
            # Test the connection
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            st.info("Using existing Snowflake connection")
        except Exception as e:
            # If connection is invalid, create a new one
            try:
                conn = snowflake.connector.connect(
                    user='',
                    password='',
                    account='',
                    warehouse='COMPUTE_WH',
                    database='NYC_DATA',
                    schema='SOURCE_DATA'
                )
                st.session_state['snowflake_conn'] = conn
                st.info("Recreated Snowflake connection")
            except Exception as e:
                st.error(f"Failed to recreate Snowflake connection: {str(e)}")
                return None
    
    return st.session_state['snowflake_conn']

def init_data():
    """Initialize data from Snowflake and create geospatial analysis table if it doesn't exist"""
    # Check if we already have the data in session state
    if 'crime_df' in st.session_state:
        st.info("Using cached data")
        return st.session_state['crime_df']
        
    try:
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Connect to Snowflake
        status_text.text("Connecting to Snowflake...")
        try:
            conn = get_snowflake_connection()
            if conn is None:
                st.error("Failed to establish Snowflake connection")
                return None
        except Exception as e:
            st.error(f"Failed to connect to Snowflake: {str(e)}")
            return None
        progress_bar.progress(20)
        
        # Check if the geospatial analysis table exists and has data
        cursor = conn.cursor()
        if cursor is None:
            st.error("Failed to create cursor")
            return None
            
        # Check if table exists and has data
        cursor.execute("""
            SELECT COUNT(*) 
            FROM NYC_DATA.SOURCE_DATA.NYC_YTD_COMPLAINTS_20250206_GEO
        """)
        count = cursor.fetchone()[0]
        
        if count > 0:
            st.info("Using existing geospatial analysis table")
            progress_bar.progress(50)
        else:
            # Create the geospatial analysis table if it doesn't exist
            status_text.text("Creating geospatial analysis table...")
            try:
                create_table_query = """
                CREATE OR REPLACE TABLE NYC_DATA.SOURCE_DATA.NYC_YTD_COMPLAINTS_20250206_GEO AS
                WITH base_data AS (
                    SELECT 
                        *,
                        ST_POINT(LONGITUDE, LATITUDE) as GEO_POINT
                    FROM NYC_DATA.SOURCE_DATA.NYC_YTD_COMPLAINTS_20250206
                    WHERE LATITUDE IS NOT NULL 
                    AND LONGITUDE IS NOT NULL
                    AND OFNS_DESC IS NOT NULL
                    AND OFNS_DESC != 'NULL'
                    AND OFNS_DESC != '(null)'
                ),
                crime_counts AS (
                    SELECT 
                        OFNS_DESC,
                        COUNT(*) as crime_count
                    FROM base_data
                    GROUP BY OFNS_DESC
                ),
                filtered_crimes AS (
                    SELECT OFNS_DESC
                    FROM crime_counts
                    WHERE crime_count < 4000
                ),
                nearby_crimes AS (
                    SELECT 
                        a.CMPLNT_NUM,
                        a.OFNS_DESC,
                        a.LATITUDE,
                        a.LONGITUDE,
                        COUNT(b.CMPLNT_NUM) as nearby_crimes,
                        AVG(ST_DISTANCE(a.GEO_POINT, b.GEO_POINT) * 0.000621371) as avg_distance
                    FROM base_data a
                    JOIN base_data b 
                        ON a.OFNS_DESC = b.OFNS_DESC
                        AND ST_DWITHIN(a.GEO_POINT, b.GEO_POINT, 1609.34) -- 1 mile in meters
                        AND a.CMPLNT_NUM != b.CMPLNT_NUM
                    WHERE a.OFNS_DESC IN (SELECT OFNS_DESC FROM filtered_crimes)
                    GROUP BY a.CMPLNT_NUM, a.OFNS_DESC, a.LATITUDE, a.LONGITUDE
                )
                SELECT 
                    n.*,
                    TO_CHAR(TO_DATE(b.CMPLNT_FR_DT), 'YYYY-MM-DD') as CMPLNT_FR_DT,
                    b.CMPLNT_FR_TM,
                    b.BORO_NM
                FROM nearby_crimes n
                JOIN base_data b ON n.CMPLNT_NUM = b.CMPLNT_NUM
                """
                
                cursor.execute(create_table_query)
                st.info("Successfully created geospatial analysis table")
            except Exception as e:
                st.error(f"Failed to create geospatial analysis table: {str(e)}")
                return None
            progress_bar.progress(50)
        
        # Read the data into a pandas DataFrame using read_snowflake().to_pandas()
        status_text.text("Loading data from Snowflake...")
        try:
            # First, let's check the column names in the table
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = 'SOURCE_DATA' 
                AND table_name = 'NYC_YTD_COMPLAINTS_20250206_GEO'
                ORDER BY ordinal_position
            """)
            columns = [col[0] for col in cursor.fetchall()]
            if not columns:
                st.error("No columns found in the table")
                return None
            st.info(f"Available columns: {', '.join(columns)}")
            
            # Use read_snowflake().to_pandas() to load the data
            query = """
            SELECT *
            FROM NYC_DATA.SOURCE_DATA.NYC_YTD_COMPLAINTS_20250206_GEO
            """
            
            # Execute query and convert to pandas DataFrame
            cursor.execute(query)
            df = cursor.fetch_pandas_all()
            
            if df is None:
                st.error("DataFrame is None after reading from Snowflake")
                return None
                
            if len(df) == 0:
                st.error("No data returned from Snowflake query")
                return None
                
            # Convert column names to uppercase to ensure consistency
            df.columns = [col.upper() for col in df.columns]
            
            # Verify required columns exist
            required_columns = ['LATITUDE', 'LONGITUDE', 'OFNS_DESC', 'CMPLNT_NUM']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                return None
            
            # Debug information about the DataFrame
            st.info(f"DataFrame shape: {df.shape}")
            st.info(f"DataFrame columns: {df.columns.tolist()}")
            st.info(f"First few rows of data:\n{df.head()}")
                
            st.info(f"Successfully loaded {len(df):,} records from Snowflake")
            
            # Cache the DataFrame in session state
            st.session_state['crime_df'] = df
            
        except Exception as e:
            st.error(f"Failed to load data from Snowflake: {str(e)}")
            return None
        progress_bar.progress(100)
        
        # Clear progress indicators
        status_text.empty()
        progress_bar.empty()
        
        return df
        
    except Exception as e:
        st.error(f"Error in init_data: {str(e)}")
        return None

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in miles"""
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 3959  # Radius of earth in miles
    return c * r

def calculate_nearby_crimes_stats(df):
    """Calculate nearby crimes statistics for each crime type"""
    # Group by crime type
    crime_types = df['OFNS_DESC'].unique()
    
    # Create a copy of the dataframe to store results
    result_df = df.copy()
    result_df['nearby_crimes'] = 0
    result_df['nearby_crimes_percentage'] = 0.0
    result_df['avg_nearby_crimes'] = 0.0
    
    # Create a progress bar and counter for record processing
    progress_bar = st.progress(0)
    status_text = st.empty()
    counter_text = st.empty()
    total_types = len(crime_types)
    
    # Initialize counters
    total_records_processed = 0
    records_with_distance = 0
    
    for i, crime_type in enumerate(crime_types):
        # Filter for current crime type
        crime_mask = df['OFNS_DESC'] == crime_type
        crime_df = df[crime_mask]
        
        if len(crime_df) == 0:
            continue
            
        status_text.text(f"Processing {crime_type} ({len(crime_df):,} records)...")
        
        # Calculate nearby crimes for each point
        for idx, row in crime_df.iterrows():
            # Calculate distances to all other points of the same crime type
            distances = calculate_distance(
                row['LATITUDE'], row['LONGITUDE'],
                crime_df['LATITUDE'], crime_df['LONGITUDE']
            )
            
            # Count crimes within 1 mile (excluding self)
            nearby_mask = (distances <= 1) & (distances > 0)
            nearby_count = nearby_mask.sum()
            
            # Calculate average distance to nearby crimes
            avg_distance = distances[nearby_mask].mean() if nearby_count > 0 else 0
            
            # Update the result dataframe
            result_df.loc[idx, 'nearby_crimes'] = nearby_count
            result_df.loc[idx, 'avg_distance'] = avg_distance
        
        # Calculate average nearby crimes for this crime type
        avg_nearby = result_df.loc[crime_mask, 'nearby_crimes'].mean()
        result_df.loc[crime_mask, 'avg_nearby_crimes'] = avg_nearby
        result_df.loc[crime_mask, 'nearby_crimes_percentage'] = (
            result_df.loc[crime_mask, 'nearby_crimes'] / avg_nearby * 100
        ).round(1)
        
        # Update counters
        total_records_processed += len(crime_df)
        records_with_distance += len(crime_df[result_df.loc[crime_mask, 'nearby_crimes'] > 0])
        
        # Update the counter display
        counter_text.text(f"Total records processed: {total_records_processed:,}")
        
        # Update progress
        progress_bar.progress((i + 1) / total_types)
    
    # Clear progress indicators
    status_text.empty()
    progress_bar.empty()
    counter_text.empty()
    
    # Store the statistics in the session state
    st.session_state['distance_stats'] = {
        'total_records': total_records_processed,
        'records_with_distance': records_with_distance,
        'percentage': (records_with_distance / total_records_processed * 100) if total_records_processed > 0 else 0
    }
    
    return result_df

def calculate_nearby_crimes(df, crime_type):
    """Calculate the number of crimes of the selected type within 1 mile of each crime location"""
    # Filter for selected crime type
    crimes_df = df[df['OFNS_DESC'] == crime_type].copy()
    
    if len(crimes_df) == 0:
        return pd.DataFrame()
    
    # The nearby_crimes and avg_distance are already calculated in Snowflake
    return crimes_df

def create_crime_map(crimes_df, crime_type):
    """Create a map visualization of crime locations"""
    if crimes_df.empty:
        st.warning(f"No {crime_type} data found in the dataset.")
        return None
    
    try:
        # Clean the data for visualization
        crimes_df = crimes_df.copy()
        
        # Ensure all required columns are present and have the correct data types
        required_columns = ['LATITUDE', 'LONGITUDE', 'NEARBY_CRIMES', 'AVG_DISTANCE', 'CMPLNT_NUM', 'CMPLNT_FR_DT', 'CMPLNT_FR_TM', 'BORO_NM', 'OFNS_DESC']
        for col in required_columns:
            if col not in crimes_df.columns:
                st.error(f"Missing required column: {col}")
                return None
        
        # Convert numeric columns to float
        numeric_columns = ['LATITUDE', 'LONGITUDE', 'NEARBY_CRIMES', 'AVG_DISTANCE']
        for col in numeric_columns:
            crimes_df[col] = pd.to_numeric(crimes_df[col], errors='coerce')
        
        # Fill NA values in numeric columns with 0
        for col in numeric_columns:
            crimes_df[col] = crimes_df[col].fillna(0)
        
        # Round average distance to 2 decimal places
        crimes_df['AVG_DISTANCE'] = crimes_df['AVG_DISTANCE'].round(2)
        
        # Calculate average and thresholds
        avg_nearby = crimes_df['NEARBY_CRIMES'].mean()
        high_threshold = avg_nearby * 1.25
        low_threshold = avg_nearby * 0.75
        
        # Create color function based on nearby crimes
        def get_color(nearby_count):
            if pd.isna(nearby_count):
                return [128, 128, 128, 160]  # Gray for NA values
            if nearby_count > high_threshold:
                return [255, 0, 0, 160]  # Red
            elif nearby_count >= low_threshold:
                return [255, 165, 0, 160]  # Orange
            else:
                return [0, 255, 0, 160]  # Green
        
        # Add color column
        crimes_df['color'] = crimes_df['NEARBY_CRIMES'].apply(get_color)
        
        # If there's a selected crime in session state, update colors for selected and nearby points
        if ('crime_map' in st.session_state and 
            'selection' in st.session_state['crime_map'] and 
            'objects' in st.session_state['crime_map']['selection'] and 
            'crimes' in st.session_state['crime_map']['selection']['objects'] and 
            st.session_state['crime_map']['selection']['objects']['crimes']):
            
            selected_crime = st.session_state['crime_map']['selection']['objects']['crimes'][0]
            selected_lat = float(selected_crime.get('LATITUDE', 0))
            selected_lon = float(selected_crime.get('LONGITUDE', 0))
            
            # Calculate distances to all points
            distances = calculate_distance(
                selected_lat, selected_lon,
                crimes_df['LATITUDE'], crimes_df['LONGITUDE']
            )
            
            # Find points within 1 mile
            nearby_mask = distances <= 1
            # Set blue color for nearby points
            for idx in crimes_df[nearby_mask].index:
                crimes_df.at[idx, 'color'] = [0, 0, 255, 200]  # Blue for nearby points
        
        # Convert DataFrame to records and ensure all values are JSON serializable
        data = []
        for _, row in crimes_df.iterrows():
            data.append({
                'LATITUDE': float(row['LATITUDE']),
                'LONGITUDE': float(row['LONGITUDE']),
                'NEARBY_CRIMES': int(row['NEARBY_CRIMES']),
                'AVG_DISTANCE': float(row['AVG_DISTANCE']),
                'CMPLNT_NUM': str(row['CMPLNT_NUM']),
                'CMPLNT_FR_DT': str(row['CMPLNT_FR_DT']),
                'CMPLNT_FR_TM': str(row['CMPLNT_FR_TM']),
                'BORO_NM': str(row['BORO_NM']),
                'OFNS_DESC': str(row['OFNS_DESC']),
                'color': row['color']
            })
        
        # Create the map layer
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=data,
            get_position='[LONGITUDE, LATITUDE]',
            get_radius=50,
            get_fill_color='color',
            pickable=True,
            auto_highlight=True,
            opacity=0.8,
            stroked=False,
            filled=True,
            radius_scale=3,
            radius_min_pixels=1,
            radius_max_pixels=50,
            id='crimes'
        )
        
        # Set the initial view state (centered on NYC with higher zoom)
        view_state = pdk.ViewState(
            latitude=40.7128,
            longitude=-74.0060,
            zoom=12,
            pitch=0,
        )
        
        # Create the deck.gl map
        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={
                "html": f"""
                    <b>Case Number:</b> {{CMPLNT_NUM}}<br/>
                    <b>Date:</b> {{CMPLNT_FR_DT}}<br/>
                    <b>Time:</b> {{CMPLNT_FR_TM}}<br/>
                    <b>Borough:</b> {{BORO_NM}}<br/>
                    <b>Coordinates:</b> {{LATITUDE}}, {{LONGITUDE}}<br/>
                    <b>{crime_type} - within 1 mile:</b> {{NEARBY_CRIMES}}<br/>
                    <b>Avg Distance:</b> {{AVG_DISTANCE}} miles
                """,
                "style": {"color": "white"}
            },
            map_style='mapbox://styles/mapbox/light-v10',
            height=800,
            width=600
        )
        
        return r
        
    except Exception as e:
        st.error(f"Error creating crime map: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def get_nearby_crimes_for_case(df, case_number):
    """Get all crimes of the same type within 1 mile of a specific case number"""
    # Get the target crime
    target_crime = df[df['CMPLNT_NUM'] == case_number]
    if len(target_crime) == 0:
        return pd.DataFrame()
    
    # Get all crimes of the same type
    same_type_crimes = df[df['OFNS_DESC'] == target_crime['OFNS_DESC'].iloc[0]]
    
    # Calculate distances
    distances = calculate_distance(
        target_crime['LATITUDE'].iloc[0], target_crime['LONGITUDE'].iloc[0],
        same_type_crimes['LATITUDE'], same_type_crimes['LONGITUDE']
    )
    
    # Filter for crimes within 1 mile (excluding self)
    nearby_mask = (distances <= 1) & (distances > 0)
    nearby_crimes = same_type_crimes[nearby_mask].copy()
    nearby_crimes['Distance (miles)'] = distances[nearby_mask].round(4)
    
    # Select and rename columns
    result = nearby_crimes[['CMPLNT_NUM', 'LATITUDE', 'LONGITUDE', 'Distance (miles)']]
    result.columns = ['Case Number', 'Latitude', 'Longitude', 'Distance (miles)']
    
    return result.sort_values('Distance (miles)')

def map_click():
    """Handle map click events and display nearby crimes of the same type"""
    with col2:
        if 'crimes' in st.session_state['crime_map']['selection']['objects']:
            for crime in st.session_state['crime_map']['selection']['objects']['crimes']:
                # Get the selected crime's coordinates and type
                selected_lat = float(crime.get('LATITUDE', 0))
                selected_lon = float(crime.get('LONGITUDE', 0))
                crime_type = crime.get('OFNS_DESC', '')
                
                # Get all crimes of the same type from session state
                same_type_crimes = st.session_state['crime_data'][
                    st.session_state['crime_data']['OFNS_DESC'] == crime_type
                ].copy()
                
                # Calculate distances to all other crimes of the same type
                distances = calculate_distance(
                    selected_lat, selected_lon,
                    same_type_crimes['LATITUDE'], same_type_crimes['LONGITUDE']
                )
                
                # Filter for crimes within 1 mile (excluding self)
                nearby_mask = (distances <= 1) & (distances > 0)
                nearby_crimes = same_type_crimes[nearby_mask].copy()
                nearby_crimes['Distance (miles)'] = distances[nearby_mask].round(4)
                
                # Create a table with the nearby crimes
                if not nearby_crimes.empty:
                    st.subheader(f"Nearby {crime_type} Crimes")
                    display_data = nearby_crimes[['CMPLNT_NUM', 'CMPLNT_FR_DT', 'LATITUDE', 'LONGITUDE', 'Distance (miles)']].copy()
                    display_data.columns = ['Case Number', 'Date', 'Latitude', 'Longitude', 'Distance (miles)']
                    display_data['Location'] = display_data.apply(lambda x: f"{x['Latitude']:.6f}, {x['Longitude']:.6f}", axis=1)
                    display_data = display_data[['Case Number', 'Date', 'Location', 'Distance (miles)']]
                    st.dataframe(display_data, hide_index=True, use_container_width=True)
                else:
                    st.write(f"No other {crime_type} crimes found within 1 mile")
        else:
            st.write("No crimes selected")

# Main code
with left_col:
    st.subheader("Welcome to the NYC Crime Analysis Dashboard!")
    st.subheader("This dashboard will help you explore crime patterns across New York City boroughs.")

# Load the data
try:
    st.info("Loading data...")
    df = init_data()
    if df is None:
        st.error("Failed to load data")
        st.stop()
        
    # Store the DataFrame in session state (this is now redundant but kept for compatibility)
    st.session_state['crime_data'] = df
    st.success(f"Data loaded successfully! ({len(df):,} records)")
    
    # Get list of available crime types
    crime_types = df['OFNS_DESC'].unique()
    crime_types = sorted(crime_types)
    
    # Create the crime type selector
    with left_col:
        selector_col, _ = st.columns([1, 4])
        with selector_col:
            selected_crime = st.selectbox("Select Crime Type", crime_types, key="crime_selector")
    
    # Create the map and statistics side by side
    with col1:
        # Create and display the crime map
        st.subheader(f"{selected_crime} Locations in NYC")
        crimes_df = calculate_nearby_crimes(df, selected_crime)
        if crimes_df is None:
            st.error("Failed to calculate nearby crimes")
            st.stop()
            
        crime_map = create_crime_map(crimes_df, selected_crime)
        if crime_map is None:
            st.error("Failed to create crime map")
            st.stop()
            
        st.pydeck_chart(crime_map, width=1200, height=800, key='crime_map', selection_mode='single-object', on_select=map_click)
    
    with col2:
        st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
        st.subheader(f"{selected_crime} Statistics")
        
        # Get total complaints for selected crime type
        total_complaints = len(df[df['OFNS_DESC'] == selected_crime])
        st.metric(f"Total {selected_crime}", f"{total_complaints:,}")
        
        # Calculate and display average nearby crimes
        if not crimes_df.empty:
            avg_nearby = crimes_df['NEARBY_CRIMES'].mean()
            st.metric(f"Avg. {selected_crime} within 1 Mile", f"{avg_nearby:.1f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add a separator
        st.markdown("---")
        
        # Display nearby crimes list below statistics
        if ('crime_map' in st.session_state and 
            'selection' in st.session_state['crime_map'] and 
            'objects' in st.session_state['crime_map']['selection'] and 
            'crimes' in st.session_state['crime_map']['selection']['objects']):
            
            selected_crimes = st.session_state['crime_map']['selection']['objects']['crimes']
            if selected_crimes:
                selected_crime = selected_crimes[0]
                selected_lat = float(selected_crime.get('LATITUDE', 0))
                selected_lon = float(selected_crime.get('LONGITUDE', 0))
                crime_type = selected_crime.get('OFNS_DESC', '')
                
                same_type_crimes = st.session_state['crime_data'][
                    st.session_state['crime_data']['OFNS_DESC'] == crime_type
                ].copy()
                
                if not same_type_crimes.empty:
                    distances = calculate_distance(
                        selected_lat, selected_lon,
                        same_type_crimes['LATITUDE'], same_type_crimes['LONGITUDE']
                    )
                    
                    nearby_mask = (distances <= 1) & (distances > 0)
                    nearby_crimes = same_type_crimes[nearby_mask].copy()
                    nearby_crimes['Distance (miles)'] = distances[nearby_mask].round(4)
                    
                    if not nearby_crimes.empty:
                        st.subheader(f"Nearby {crime_type} Crimes")
                        display_data = nearby_crimes[['CMPLNT_NUM', 'CMPLNT_FR_DT', 'LATITUDE', 'LONGITUDE', 'Distance (miles)']].copy()
                        display_data.columns = ['Case Number', 'Date', 'Latitude', 'Longitude', 'Distance (miles)']
                        display_data['Location'] = display_data.apply(lambda x: f"{x['Latitude']:.6f}, {x['Longitude']:.6f}", axis=1)
                        display_data = display_data[['Case Number', 'Date', 'Location', 'Distance (miles)']]
                        st.dataframe(display_data, hide_index=True, use_container_width=True)
                    else:
                        st.write(f"No other {crime_type} crimes found within 1 mile")

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    import traceback
    st.error(f"Traceback: {traceback.format_exc()}") 