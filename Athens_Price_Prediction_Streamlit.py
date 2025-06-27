import streamlit as st
from model.airbnb_price_predictor import AirbnbPredictor
import os
import base64
import folium
from streamlit_folium import st_folium

def get_base64_image(image_path):
    """Convert image to base64 string"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Page configuration
st.set_page_config(
    page_title="Athens Airbnb Price Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
# Get the background image
bg_image = get_base64_image("data/city-5761429_1920.jpg")
bg_css = f"background-image: linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.4)), url('data:image/jpeg;base64,{bg_image}');" if bg_image else "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);"

st.markdown(f"""
<style>
    /* Hide Streamlit deploy bar */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .stDeployButton {{display: none;}}
    
    /* Import the background image  */
    .stApp {{
        {bg_css}
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    /* Main content container - remove background */
    .main .block-container {{
        padding: 0.2rem;
        margin: 0;
        max-width: 1200px;
        padding-top: 0;
    }}
    
    .main-header {{
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
        text-align: center;
        margin-bottom: 0.1rem;
        margin-top: -1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
    }}
    
    .app-description {{
        text-align: center;
        margin: 2rem 0;
    }}
    
    .app-description p {{
        font-size: 1.3rem;
        color: white;
        line-height: 1.6;
        margin: 0;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.7);
        font-weight: 500;
    }}
    
    .input-section-header {{
        font-size: 1.5rem;
        font-weight: 600;
        color: white;
        text-align: center;
        margin: 0.2rem 0 0.2rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
    }}
    
    .input-group h3 {{
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.1rem;
        margin-top: 0.1rem;
        text-align: center;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.7);
    }}
    
    .prediction-box {{
        color: white;
        padding: 1rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }}
    
    .prediction-box h2 {{
        color: white;
        font-size: 2rem;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.7);
    }}
    
    .prediction-box h1 {{
        text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
    }}
    
    .sub-header {{
        font-size: 1.1rem;
        color: white;
        margin-bottom: 0.2rem;
        margin-top: 0.2rem;
        font-weight: 600;
        text-align: center;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.7);
    }}
    
    /* Input styling - keep backgrounds for readability and make them narrower */
    .stSelectbox > div > div {{
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(2px) !important;
        color: #000000 !important;
        max-width: 250px !important;
    }}
    .stSelectbox > div > div > div {{
        color: #000000 !important;
    }}
    .stSelectbox option {{
        color: #000000 !important;
        background: rgba(255, 255, 255, 0.9) !important;
    }}
    .stNumberInput > div > div {{
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(2px) !important;
        color: #000000 !important;
        max-width: 250px !important;
        width: 250px !important;
    }}
    .stNumberInput input {{
        color: #000000 !important;
        background: rgba(255, 255, 255, 0.95) !important;
        max-width: 250px !important;
        width: 250px !important;
    }}
    .stNumberInput {{
        max-width: 250px !important;
        width: 250px !important;
    }}
    .stTextInput > div > div {{
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(2px) !important;
        color: #000000 !important;
        max-width: 200px !important;
    }}
    .stTextInput input {{
        color: #000000 !important;
        background: rgba(255, 255, 255, 0.95) !important;
    }}
    .stCheckbox > label {{
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(2px) !important;
        border-radius: 0.25rem !important;
        padding: 0.25rem !important;
        color: #000000 !important;
    }}
    .stCheckbox input + div {{
        color: #000000 !important;
    }}
    
    /* Make buttons less wide */
    .stButton > button {{
        max-width: 200px !important;
        margin: 0 auto !important;
        display: block !important;
    }}
    
    /* Input labels */
    .stSelectbox label, .stNumberInput label, .stTextInput label, .stCheckbox label {{
        color: white !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7) !important;
    }}
    
    /* Input label text - more specific selectors */
    .stSelectbox > label {{
        color: white !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7) !important;
    }}
    .stNumberInput > label {{
        color: white !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7) !important;
    }}
    .stTextInput > label {{
        color: white !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7) !important;
    }}
    .stCheckbox > label {{
        color: white !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7) !important;
    }}
    
    /* All label elements */
    label {{
        color: white !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7) !important;
    }}
    
    /* Streamlit label elements */
    .stSelectbox label[data-testid], .stNumberInput label[data-testid], .stTextInput label[data-testid] {{
        color: white !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7) !important;
    }}
    
    /* Map container styling */
    iframe[title="streamlit_folium.st_folium"] {{
        width: 100% !important;
        height: 250px !important;
        border-radius: 0.5rem !important;
        border: 2px solid rgba(255,255,255,0.3) !important;
    }}
    
    /* Fix map container to prevent black box */
    div[data-testid="stElementContainer"] > div > iframe[title="streamlit_folium.st_folium"] {{
        height: 250px !important;
    }}
    
    /* Map wrapper styling */
    .stElementContainer > div > div > iframe[title="streamlit_folium.st_folium"] {{
        height: 250px !important;
        max-height: 250px !important;
    }}
    
    /* Widget labels */
    div[data-testid="stMarkdownContainer"] p {{
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7) !important;
    }}
    
    /* Override checkbox text to be white outside the box */
    .stCheckbox div[data-testid="stMarkdownContainer"] p {{
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7) !important;
    }}
    
    /* Additional selectors for input text */
    .stSelectbox div[data-baseweb="select"] {{
        color: #000000 !important;
    }}
    .stSelectbox div[data-baseweb="select"] > div {{
        color: #000000 !important;
    }}
    .stSelectbox div[data-baseweb="select"] span {{
        color: #000000 !important;
    }}
    
    /* Input placeholders */
    .stNumberInput input::placeholder {{
        color: #666666 !important;
    }}
    .stTextInput input::placeholder {{
        color: #666666 !important;
    }}
    
    /* Selected option text */
    [data-baseweb="select"] [data-baseweb="popover"] li {{
        color: #000000 !important;
    }}
    
    /* Input value display */
    .stSelectbox > div > div[data-baseweb="select"] > div > div {{
        color: #000000 !important;
    }}
    
    /* Override any white text in inputs */
    .stSelectbox *, .stNumberInput *, .stTextInput * {{
        color: #000000 !important;
    }}
    .stSelectbox label, .stNumberInput label, .stTextInput label {{
        color: white !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7) !important;
    }}
    
    /* Fix number input +/- buttons */
    .stNumberInput button {{
        background: rgba(255, 255, 255, 0.8) !important;
        color: #000000 !important;
        border: 1px solid rgba(0,0,0,0.2) !important;
    }}
    .stNumberInput button:hover {{
        background: rgba(255, 255, 255, 0.9) !important;
        color: #000000 !important;
    }}
    
    /* Dropdown menu styling */
    [data-baseweb="menu"] {{
        background: rgba(255, 255, 255, 0.95) !important;
    }}
    [data-baseweb="menu"] li {{
        color: #000000 !important;
        background: rgba(255, 255, 255, 0.95) !important;
    }}
    [data-baseweb="menu"] li:hover {{
        background: rgba(240, 240, 240, 0.95) !important;
    }}
    
    /* Button styling */
    .stButton > button {{
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 50%, #eab308 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(217, 119, 6, 0.4) !important;
        backdrop-filter: blur(5px) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        padding: 0.75rem 1.5rem !important;
        font-size: 1.1rem !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(217, 119, 6, 0.5) !important;
        background: linear-gradient(135deg, #c2710c 0%, #d97706 50%, #f59e0b 100%) !important;
    }}
    
    /* Metric styling - keep some background for readability */
    .stMetric {{
        background: rgba(255, 255, 255, 0.15) !important;
        padding: 1rem !important;
        border-radius: 1rem !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05) !important;
        backdrop-filter: blur(3px) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
    }}
    .stMetric [data-testid="metric-container"] {{
        background: rgba(255, 255, 255, 0.15) !important;
        border-radius: 0.5rem !important;
        padding: 0.5rem !important;
        backdrop-filter: blur(3px) !important;
    }}
    .stMetric label {{
        color: white !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7) !important;
    }}
    .stMetric div[data-testid="metric-value"] {{
        color: white !important;
        font-weight: 700 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7) !important;
    }}
    
    /* Pricing analysis boxes - keep some background for readability */
    .pricing-analysis-box {{
        background: rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(5px) !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
    }}
    
    /* Text content styling */
    p, li, div {{
        color: white !important;
    }}
    
    strong {{
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7) !important;
    }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model (cached for performance)"""
    if not os.path.exists('model.pkl'):
        return None
    
    predictor = AirbnbPredictor()
    try:
        predictor.load('model.pkl')
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">Athens Airbnb Price Prediction</h1>', unsafe_allow_html=True)
    
    # Subtitle
    st.markdown("""
    <div style="text-align: center; margin-bottom: 0.1rem;">
        <h2 style="color: white; font-size: 1rem; font-weight: 400; text-shadow: 1px 1px 2px rgba(0,0,0,0.7); margin: 0;">
            ML-powered Airbnb price prediction for Athens properties
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    predictor = load_model()
    
    if predictor is None:
        st.error("Model not found! Please run `python model/airbnb_price_predictor.py` first to train the model.")
        st.info("Make sure you have the `athens_listings.csv` file in the data folder and run the training script.")
        return
    
    # Initialize session state for selected location
    if "selected_lat" not in st.session_state:
        st.session_state.selected_lat = 37.9755
        st.session_state.selected_lng = 23.7348
        st.session_state.selected_neighborhood = "Athens Center"
    
    # Initialize session state for showing results
    if "show_results" not in st.session_state:
        st.session_state.show_results = False
    
    # Set default booking settings values
    min_nights = 2
    max_nights = 30
    availability = 300
    instant_bookable = True
    
    # Show input sections or results based on state
    if not st.session_state.show_results:
        # Create columns with spacers for better centering
        spacer_left, left_col, right_col, spacer_right = st.columns([0.15, 0.4, 0.35, 0.1])
        
        with left_col:
            # Property Details Section
            st.markdown('<h2 class="input-section-header">Property Details</h2>', unsafe_allow_html=True)
            
            # Property and Room Type
            prop_col1, prop_col2 = st.columns([1, 1], gap="small")
            with prop_col1:
                # Create mapping for user-friendly names to actual data values
                property_type_mapping = {
                    "Entire rental unit": "Entire rental unit",
                    "Condo": "Entire condo", 
                    "Private room in rental unit": "Private room in rental unit",
                    "Hotel room": "Room in hotel",
                    "Entire home": "Entire home",
                    "Loft": "Entire loft"
                }
                
                selected_property_display = st.selectbox(
                    "Property Type",
                    list(property_type_mapping.keys()),
                    help="What type of property are you listing?"
                )
                
                # Get the actual property type value for the model
                property_type = property_type_mapping[selected_property_display]
            
            with prop_col2:
                # Dynamic room type options based on property type
                if selected_property_display in ["Entire rental unit", "Condo", "Entire home", "Loft"]:
                    room_options = ["Entire home/apt"]
                elif selected_property_display == "Private room in rental unit":
                    room_options = ["Private room"]
                elif selected_property_display == "Hotel room":
                    room_options = ["Hotel room"]
                else:
                    room_options = ["Entire home/apt", "Private room", "Hotel room", "Shared room"]
                
                room_type = st.selectbox(
                    "Room Type",
                    room_options,
                    help="What are you offering to guests?"
                )
            
            # Capacity and Rooms in 2x2 grid
            st.markdown('<h2 class="input-section-header">Capacity & Rooms</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1], gap="small")
            with col1:
                accommodates = st.number_input("Guests", min_value=1, max_value=16, value=4, help="Maximum number of guests")
                beds = st.number_input("Beds", min_value=1, max_value=20, value=2, help="Total number of beds")
            
            with col2:
                bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=2, help="Number of bedrooms (0 for studio)")
                bathrooms_num = st.number_input("Bathrooms", min_value=1, max_value=10, value=1, step=1, help="Number of bathrooms")
            
            # Convert bathrooms to text format
            bathrooms_text = f"{int(bathrooms_num)} bath{'s' if bathrooms_num != 1 else ''}"
            
            # Store values in session state
            st.session_state.property_type = property_type
            st.session_state.room_type = room_type
            st.session_state.accommodates = accommodates
            st.session_state.bedrooms = bedrooms
            st.session_state.beds = beds
            st.session_state.bathrooms_text = bathrooms_text
        
        with right_col:
            # Location Section
            st.markdown('<h2 class="input-section-header">Location</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            <p style="color: white; font-size: 0.9rem; margin-bottom: 0.1rem; margin-top: 0.1rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.7);">
                Click on the map to select your property location:
            </p>
            """, unsafe_allow_html=True)
            
            # Create a map centered on Athens with controlled zoom
            m = folium.Map(
                location=[37.9755, 23.7348],  # Center of Athens
                zoom_start=13,
                min_zoom=13,  # Prevent zooming out further than this
                max_zoom=18,  # Allow zooming in for detail
                width="100%",
                height=250,  # Smaller map for better balance
                zoom_control=True,  # Enable zoom controls
                scrollWheelZoom=True,  # Enable scroll wheel zoom
                dragging=True,
                doubleClickZoom=False,
                boxZoom=False,
                keyboard=False,
                tiles='OpenStreetMap'
            )
            
            # Add some popular neighborhoods as markers for reference
            popular_spots = {
                "Acropolis": (37.9715, 23.7267),
                "Plaka": (37.9715, 23.7300),
                "Koukaki": (37.963807, 23.722057),
                "Monastiraki": (37.9755, 23.7255),
                "Psyrri": (37.9775, 23.7255),
                "Exarchia": (37.9855, 23.7355),
                "Kolonaki": (37.9795, 23.7445),
                "Syntagma": (37.9755, 23.7348)
            }
            
            # Add markers for popular spots
            for name, (lat, lon) in popular_spots.items():
                folium.Marker(
                    [lat, lon],
                    popup=f"<b>{name}</b>",
                    tooltip=name,
                    icon=folium.Icon(color='blue', icon='info-sign')
                ).add_to(m)
            
            # Add red marker for selected location
            if st.session_state.selected_lat != 37.9755 or st.session_state.selected_lng != 23.7348:
                folium.Marker(
                    [st.session_state.selected_lat, st.session_state.selected_lng],
                    popup=f"<b>Your Property</b><br>Lat: {st.session_state.selected_lat:.4f}<br>Lng: {st.session_state.selected_lng:.4f}",
                    tooltip="Selected Property Location",
                    icon=folium.Icon(color='red', icon='home')
                ).add_to(m)
            
            # Display the map and get click data
            map_data = st_folium(m, width="100%", height=250, returned_objects=["last_clicked"])
            
            # Update session state when map is clicked
            if map_data["last_clicked"] is not None:
                st.session_state.selected_lat = map_data["last_clicked"]["lat"]
                st.session_state.selected_lng = map_data["last_clicked"]["lng"]
                st.session_state.selected_neighborhood = "Selected Location"
                st.rerun()  # Rerun to update the map with the new marker
            
            # Display selected coordinates
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.1); padding: 0.3rem; border-radius: 0.3rem; margin-top: 0.1rem;">
                <p style="color: white; margin: 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.7); font-size: 0.8rem;">
                    <strong>Selected Location:</strong><br>
                    Latitude: {st.session_state.selected_lat:.6f}<br>
                    Longitude: {st.session_state.selected_lng:.6f}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Centered predict button below the inputs
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Predict Price", type="primary"):
                st.session_state.show_results = True
                st.rerun()
        

    
    else:
        # Show results page
        # Add back button
        if st.button("← Back to Inputs", type="secondary"):
            st.session_state.show_results = False
            st.rerun()
        
        # Get values from session state
        property_type = st.session_state.get('property_type', 'Entire rental unit')
        room_type = st.session_state.get('room_type', 'Entire home/apt')
        accommodates = st.session_state.get('accommodates', 4)
        bedrooms = st.session_state.get('bedrooms', 2)
        beds = st.session_state.get('beds', 2)
        bathrooms_text = st.session_state.get('bathrooms_text', '1 bath')
        
        # Use coordinates from session state
        latitude = st.session_state.selected_lat
        longitude = st.session_state.selected_lng
        neighbourhood = st.session_state.selected_neighborhood
    
    # Results section - show in center when predict is clicked
    if st.session_state.show_results:
        # Create listing dictionary with enhanced features
        listing = {
            'property_type': property_type,
            'room_type': room_type,
            'accommodates': accommodates,
            'bedrooms': bedrooms,
            'beds': beds,
            'bathrooms_text': bathrooms_text,
            'latitude': latitude,
            'longitude': longitude,
            'neighbourhood_cleansed': neighbourhood,
            'minimum_nights': min_nights,
            'maximum_nights': max_nights,
            'availability_365': availability,
            'instant_bookable': 't' if instant_bookable else 'f',
            'host_since': '2024-01-01',  # New host
            'host_is_superhost': 'f',  # Removed host details
            'host_identity_verified': 'f',  # New hosts typically not verified yet
            'calculated_host_listings_count': 0,  # Removed host details
            'number_of_reviews': 0,  # New listing has no reviews
            'review_scores_rating': 4.5  # Default neutral rating for new listings
        }
        
        try:
            # Make prediction
            predicted_price = predictor.predict(listing)
            
            # Show results in centered columns
            result_col1, result_col2, result_col3 = st.columns([0.2, 0.6, 0.2])
            
            with result_col2:
                # Display results
                st.markdown(f"""
                <div class="prediction-box">
                    <h2 style="color: white; font-size: 1.8rem; margin-bottom: 0.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.9); text-align: center;">Predicted Price</h2>
                    <h1 style="color: #fbbf24; font-size: 3.5rem; font-weight: bold; margin: 0.5rem 0; text-shadow: 3px 3px 6px rgba(0,0,0,0.9); text-align: center;">€{predicted_price:.2f}</h1>
                    <p style="color: white; font-size: 1.2rem; margin-top: 0.2rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.9); text-align: center;">per night</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Model performance in two columns
                perf_col1, perf_col2 = st.columns(2)
                
                # Get performance metrics from the model (with fallback for older models)
                if hasattr(predictor, 'get_performance_metrics'):
                    metrics = predictor.get_performance_metrics()
                    r2_score = metrics.get('r2_score', 0.76) * 100  # Convert to percentage
                    mae = metrics.get('mae', 24.32)
                    rmse = metrics.get('rmse', 43.38)
                else:
                    # Fallback values for older model versions
                    r2_score = 76
                    mae = 24.32
                    rmse = 43.38
                
                with perf_col1:
                    st.markdown('<h3 class="sub-header">Model Performance</h3>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="text-align: center;">
                        <p style="color: white; font-size: 0.9rem; margin: 0.1rem 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.7);">
                            <strong>Model:</strong> XGBoost
                        </p>
                        <p style="color: white; font-size: 0.9rem; margin: 0.1rem 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.7);">
                            <strong>R² Score:</strong> {r2_score:.1f}%
                        </p>
                        <p style="color: white; font-size: 0.9rem; margin: 0.1rem 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.7);">
                            <strong>MAE:</strong> €{mae:.2f}
                        </p>
                        <p style="color: white; font-size: 0.9rem; margin: 0.1rem 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.7);">
                            <strong>RMSE:</strong> €{rmse:.2f}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with perf_col2:
                    st.markdown('<h3 class="sub-header">Property Summary</h3>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="text-align: center; color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.7); font-size: 0.9rem; line-height: 1.3;">
                    <strong>Details:</strong><br>
                    {property_type}, {room_type}<br>
                    {accommodates} guests, {bedrooms} bedrooms<br>
                    {beds} beds, {bathrooms_text}<br><br>
                    <strong>Location:</strong><br>
                    {neighbourhood}<br>
                    {latitude:.4f}, {longitude:.4f}
                    </div>
                    """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please check your inputs and try again.")

    
    # Image attribution 
    st.markdown("""
    <div style="text-align: center; margin-top: 0.5rem; padding: 0.2rem;">
        <small style="color: white; background: rgba(0,0,0,0.4); padding: 0.1rem 0.2rem; border-radius: 0.1rem; font-size: 0.6rem;">
            Image by <a href="https://pixabay.com/el/users/viarami-13458823/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=5761429" target="_blank" style="color: white; text-decoration: none;">Markus Winkler</a> from <a href="https://pixabay.com/el//?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=5761429" target="_blank" style="color: white; text-decoration: none;">Pixabay</a>
        </small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 