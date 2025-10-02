import streamlit as st
import pandas as pd
import numpy as np
import joblib
from math import radians, sin, cos, sqrt, atan2
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Delivery Time Predictor",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .feature-section {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_encoders():
    """Load the trained model and label encoders"""
    try:
        model = joblib.load('/Users/nehadhananju/Desktop/AmazonDeliveryTimePrediction/models/best_amazon_delivery_time_model.pkl')
        label_encoders = joblib.load('/Users/nehadhananju/Desktop/AmazonDeliveryTimePrediction/models/label_encoders.pkl')
        return model, label_encoders
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two geographic points"""
    R = 6371  # Earth radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

def create_features_from_input(user_input, label_encoders):
    """Create features from user input matching training data format"""
    
    # Calculate distance
    distance = calculate_distance(
        user_input['store_lat'], user_input['store_lon'],
        user_input['drop_lat'], user_input['drop_lon']
    )
    
    # Create feature dictionary - EXACTLY matching your training features
    features = {
        'Agent_Age': user_input['agent_age'],
        'Agent_Rating': user_input['agent_rating'],
        'delivery_distance_km': distance,
        'weather_severity': user_input['weather_severity'],
        'traffic_level': user_input['traffic_level'],
        'vehicle_speed_capability': user_input['vehicle_speed'],
        'area_density': user_input['area_density'],
        'order_hour': user_input['order_hour'],
        'order_day_of_week': user_input['day_of_week'],
        'order_month_num': user_input['month'],
        'pickup_delay_hours': user_input['pickup_delay'],
        'agent_performance_score': (user_input['agent_rating'] * 0.7 + (user_input['agent_age'] / 50) * 0.3),
        'weather_traffic_impact': user_input['weather_severity'] * user_input['traffic_level'],
        'vehicle_area_score': user_input['vehicle_speed'] / user_input['area_density']
    }
    
    # Add categorical features (will be encoded)
    categorical_values = {
        'distance_category': pd.cut([distance], bins=[0, 5, 15, 50, 100, float('inf')],
                                  labels=['Very_Near', 'Near', 'Medium', 'Far', 'Very_Far'])[0],
        'agent_experience': pd.cut([user_input['agent_rating']], bins=[0, 3, 4, 4.7, 5.1],
                                 labels=['Beginner', 'Intermediate', 'Experienced', 'Expert'])[0],
        'agent_age_group': pd.cut([user_input['agent_age']], bins=[15, 25, 35, 50],
                                labels=['Young', 'Middle', 'Senior'])[0],
        'Weather': user_input['weather'],
        'Traffic': user_input['traffic'],
        'Vehicle': user_input['vehicle'],
        'Area': user_input['area'],
        'Category': user_input['category']
    }
    
    # Encode categorical features using your label encoders
    for feature, value in categorical_values.items():
        if feature in label_encoders:
            try:
                # Convert to string and encode
                encoded_value = label_encoders[feature].transform([str(value)])[0]
                features[feature] = encoded_value
            except ValueError as e:
                # If value not seen during training, use most common class
                st.warning(f"Using default value for {feature} (unseen category: {value})")
                features[feature] = 0  # Default to first category
    
    return features, distance, categorical_values

def main():
    # Header
    st.markdown('<h1 class="main-header">🚚 Delivery Time Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### AI-powered delivery time estimation based on multiple factors")
    
    # Load model
    with st.spinner("Loading prediction model..."):
        model, label_encoders = load_model_and_encoders()
    
    if model is None or label_encoders is None:
        st.error(" Could not load the prediction model. Please check if the model files exist in the models/ folder.")
        return
    
    st.success(" Model loaded successfully!")
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs([" Order Details", " Prediction Results"])
    
    with tab1:
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)
        st.subheader("Enter Order Details")
        
        # Create three columns for input form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("** Location Details**")
            
            # Store coordinates
            store_lat = st.number_input("Store Latitude", value=12.9716, format="%.6f", key="store_lat")
            store_lon = st.number_input("Store Longitude", value=77.5946, format="%.6f", key="store_lon")
            
            # Drop coordinates
            drop_lat = st.number_input("Drop Latitude", value=12.9352, format="%.6f", key="drop_lat")
            drop_lon = st.number_input("Drop Longitude", value=77.6245, format="%.6f", key="drop_lon")
            
            # Auto-calculate distance
            distance = calculate_distance(store_lat, store_lon, drop_lat, drop_lon)
            st.metric(" Delivery Distance", f"{distance:.2f} km")
            
            st.markdown("** Agent Details**")
            agent_age = st.slider("Agent Age", 18, 50, 30)
            agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.5, 0.1)
        
        with col2:
            st.markdown("** Time & Date**")
            order_hour = st.slider("Order Hour", 0, 23, 12)
            day_of_week = st.selectbox("Day of Week", 
                                     ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            month = st.selectbox("Month", 
                               ["January", "February", "March", "April", "May", "June",
                                "July", "August", "September", "October", "November", "December"])
            pickup_delay = st.slider("Pickup Delay (hours)", 0.0, 5.0, 0.5, 0.1)
            
            st.markdown("** Vehicle & Area**")
            vehicle = st.selectbox("Vehicle Type", ["motorcycle", "scooter", "van", "bicycle"])
            area = st.selectbox("Area Type", ["Urban", "Metropolitian", "Semi-Urban", "Other"])
        
        with col3:
            st.markdown("** Weather & Traffic**")
            weather = st.selectbox("Weather Condition", 
                                 ["Sunny", "Cloudy", "Windy", "Fog", "Sandstorms", "Stormy"])
            traffic = st.selectbox("Traffic Condition", ["Low", "Medium", "High", "Jam"])
            
            st.markdown("** Product Details**")
            category = st.selectbox("Product Category", 
                                  ["Electronics", "Clothing", "Food", "Books", "Furniture", 
                                   "Medicines", "Sports", "Toys", "Jewelry", "Cosmetics"])
            
            # Derived features (calculated automatically)
            st.markdown("** Derived Features**")
            weather_severity_map = {"Sunny": 1, "Cloudy": 2, "Windy": 3, "Fog": 4, "Sandstorms": 5, "Stormy": 6}
            traffic_level_map = {"Low": 1, "Medium": 2, "High": 3, "Jam": 4}
            vehicle_speed_map = {"bicycle": 1, "scooter": 2, "motorcycle": 3, "van": 2}
            area_density_map = {"Semi-Urban": 1, "Urban": 2, "Metropolitian": 3, "Other": 2}
            day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
            month_map = {month: i+1 for i, month in enumerate(["January", "February", "March", "April", "May", "June",
                                                             "July", "August", "September", "October", "November", "December"])}
            
            weather_severity = weather_severity_map[weather]
            traffic_level = traffic_level_map[traffic]
            vehicle_speed = vehicle_speed_map[vehicle]
            area_density = area_density_map[area]
            day_of_week_num = day_map[day_of_week]
            month_num = month_map[month]
            
            # Display derived values
            st.metric("Weather Severity", weather_severity)
            st.metric("Traffic Level", traffic_level)
            st.metric("Vehicle Speed", vehicle_speed)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Prediction button
        predict_button = st.button(" Predict Delivery Time", type="primary", use_container_width=True)
    
    with tab2:
        st.markdown("### Prediction Results")
        
        if predict_button:
            with st.spinner("Calculating delivery time..."):
                # Prepare user input
                user_input = {
                    'store_lat': store_lat,
                    'store_lon': store_lon,
                    'drop_lat': drop_lat,
                    'drop_lon': drop_lon,
                    'agent_age': agent_age,
                    'agent_rating': agent_rating,
                    'weather': weather,
                    'traffic': traffic,
                    'vehicle': vehicle,
                    'area': area,
                    'category': category,
                    'order_hour': order_hour,
                    'day_of_week': day_of_week_num,
                    'month': month_num,
                    'pickup_delay': pickup_delay,
                    'weather_severity': weather_severity,
                    'traffic_level': traffic_level,
                    'vehicle_speed': vehicle_speed,
                    'area_density': area_density
                }
                
                # Create features
                features, distance, categorical_values = create_features_from_input(user_input, label_encoders)
                
                # Convert to DataFrame
                features_df = pd.DataFrame([features])
                
                # Make prediction
                try:
                    prediction = model.predict(features_df)[0]
                    
                    # Display results
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(" Product Category", category)
                        st.metric(" Delivery Distance", f"{distance:.2f} km")
                        st.metric(" Agent Experience", categorical_values['agent_experience'])
                    
                    with col2:
                        st.metric(" Weather", weather)
                        st.metric(" Traffic", traffic)
                        st.metric(" Vehicle", vehicle)
                    
                    with col3:
                        st.metric(" Area Type", area)
                        st.metric(" Order Time", f"{order_hour:02d}:00")
                        st.metric(" Pickup Delay", f"{pickup_delay:.1f} hours")
                    
                    st.markdown("---")
                    
                    # Main prediction
                    st.markdown(f"##  Predicted Delivery Time: **{prediction:.1f} hours**")
                    
                    # Interpretation
                    if prediction < 2:
                        st.success(" Express Delivery - Very fast service expected!")
                    elif prediction < 6:
                        st.info(" Standard Delivery - Good delivery time")
                    elif prediction < 12:
                        st.warning(" Extended Delivery - Longer than average time")
                    else:
                        st.error(" Delayed Delivery - Significant delays expected")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Feature importance explanation
                    st.subheader("📊 Key Factors Affecting Delivery Time")
                    
                    factors = [
                        f"**Distance**: {distance:.1f} km ({'Short' if distance < 10 else 'Medium' if distance < 30 else 'Long'} distance)",
                        f"**Weather**: {weather} ({'Good' if weather_severity <= 2 else 'Moderate' if weather_severity <= 4 else 'Poor'} conditions)",
                        f"**Traffic**: {traffic} ({'Light' if traffic_level <= 2 else 'Heavy'} traffic)",
                        f"**Time of Day**: {order_hour}:00 ({'Peak' if (7 <= order_hour <= 9) or (17 <= order_hour <= 19) else 'Off-peak'} hours)",
                        f"**Agent Rating**: {agent_rating}/5.0 ({'Expert' if agent_rating >= 4.7 else 'Experienced' if agent_rating >= 4.0 else 'Intermediate'})"
                    ]
                    
                    for factor in factors:
                        st.write(f"• {factor}")
                        
                except Exception as e:
                    st.error(f" Prediction failed: {e}")
        
        else:
            st.info("👆 Click the 'Predict Delivery Time' button in the Order Details tab to see predictions")
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with using Streamlit and Machine Learning*")

if __name__ == "__main__":
    main()