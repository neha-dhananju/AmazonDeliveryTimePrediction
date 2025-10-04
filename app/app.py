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

# Custom CSS for better styling with insights highlighting
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
    .insight-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #f8d7da;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d1edff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #0d6efd;
        margin: 1rem 0;
    }
    .factor-impact {
        background-color: #e9ecef;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .impact-high {
        color: #dc3545;
        font-weight: bold;
    }
    .impact-medium {
        color: #fd7e14;
        font-weight: bold;
    }
    .impact-low {
        color: #198754;
        font-weight: bold;
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

def generate_insights(prediction, user_input, distance, categorical_values):
    """Generate business insights based on the prediction and input factors"""
    
    insights = []
    warnings = []
    recommendations = []
    
    # 🎯 DELIVERY TIME INSIGHTS
    if prediction < 2:
        insights.append("⚡ **Express Delivery Performance**: This delivery meets premium service standards")
        recommendations.append("Consider promoting express delivery options for similar orders")
    elif prediction < 4:
        insights.append("🚀 **Standard Delivery Performance**: Delivery within acceptable timeframe")
    elif prediction < 6:
        insights.append("🐢 **Extended Delivery**: Longer than average delivery time expected")
        warnings.append("Customer communication recommended for delay management")
    else:
        insights.append("🚨 **Significant Delay Expected**: Delivery exceeds 6 hours")
        warnings.append("Urgent review needed for route optimization and resource allocation")
    
    # 📍 DISTANCE INSIGHTS
    if distance > 50:
        insights.append("🌍 **Long-Haul Delivery**: Distance exceeds 50km - consider specialized logistics")
        warnings.append("High fuel costs and vehicle wear expected")
        recommendations.append("Optimize route with GPS navigation and consider break points")
    elif distance < 5:
        insights.append("📍 **Hyper-Local Delivery**: Ideal for quick turnaround and multiple deliveries")
        recommendations.append("Perfect for batch deliveries in the same area")
    
    # 🌦️ WEATHER INSIGHTS
    weather_impact = {
        "Stormy": "Severe weather will significantly impact delivery speed",
        "Sandstorms": "Poor visibility and road conditions affect safety and speed", 
        "Fog": "Reduced visibility may slow down delivery by 20-30%",
        "Windy": "Moderate impact on vehicle stability and speed",
        "Cloudy": "Minimal impact on delivery operations",
        "Sunny": "Optimal conditions for fastest delivery"
    }
    if user_input['weather'] in weather_impact:
        insights.append(f"🌤️ **Weather Impact**: {weather_impact[user_input['weather']]}")
    
    # 🚦 TRAFFIC INSIGHTS
    traffic_impact = {
        "Jam": "Heavy congestion will increase delivery time by 40-60%",
        "High": "Busy traffic conditions will slow down delivery by 20-40%",
        "Medium": "Moderate traffic may cause 10-20% delays",
        "Low": "Light traffic enables optimal delivery speed"
    }
    if user_input['traffic'] in traffic_impact:
        insights.append(f"🚗 **Traffic Analysis**: {traffic_impact[user_input['traffic']]}")
    
    # ⏰ TIME-BASED INSIGHTS
    if 7 <= user_input['order_hour'] <= 9:
        insights.append("🌅 **Morning Rush Hour**: Peak traffic conditions expected")
        recommendations.append("Consider scheduling deliveries before 7 AM or after 9 AM")
    elif 17 <= user_input['order_hour'] <= 19:
        insights.append("🌇 **Evening Rush Hour**: Heavy traffic and delivery congestion")
        recommendations.append("Evening deliveries may benefit from extended time windows")
    else:
        insights.append("🕒 **Off-Peak Timing**: Optimal delivery window with less congestion")
    
    # 👤 AGENT PERFORMANCE INSIGHTS
    if user_input['agent_rating'] >= 4.7:
        insights.append("⭐ **Expert Agent Assignment**: Highly rated agent ensures reliable delivery")
        recommendations.append("Consider assigning complex or high-value deliveries to this agent")
    elif user_input['agent_rating'] <= 3.5:
        warnings.append("Agent performance below average - may affect delivery reliability")
        recommendations.append("Provide additional training or assign with experienced partner")
    
    # 🚗 VEHICLE EFFICIENCY INSIGHTS
    vehicle_efficiency = {
        "motorcycle": "Best for urban areas, quick maneuvers, and parking",
        "scooter": "Good for medium distances with better fuel efficiency", 
        "van": "Necessary for large items but slower in dense traffic",
        "bicycle": "Eco-friendly but limited to short distances and good weather"
    }
    if user_input['vehicle'] in vehicle_efficiency:
        insights.append(f"🛵 **Vehicle Selection**: {vehicle_efficiency[user_input['vehicle']]}")
    
    # 🏙️ AREA-SPECIFIC INSIGHTS
    area_characteristics = {
        "Metropolitian": "High density with potential parking challenges and traffic",
        "Urban": "Moderate density with established delivery routes", 
        "Semi-Urban": "Better traffic flow with possible longer access routes",
        "Other": "Variable conditions requiring local knowledge"
    }
    if user_input['area'] in area_characteristics:
        insights.append(f"🏢 **Area Analysis**: {area_characteristics[user_input['area']]}")
    
    return insights, warnings, recommendations

def calculate_factor_impacts(user_input, distance):
    """Calculate the impact of each factor on delivery time"""
    impacts = {}
    
    # Distance impact (0.1 hours per km beyond 5km)
    impacts['Distance'] = max(0, (distance - 5) * 0.1)
    
    # Weather impact
    weather_impact_map = {"Sunny": 0, "Cloudy": 0.2, "Windy": 0.5, "Fog": 0.8, "Sandstorms": 1.2, "Stormy": 1.5}
    impacts['Weather'] = weather_impact_map.get(user_input['weather'], 0.5)
    
    # Traffic impact  
    traffic_impact_map = {"Low": 0, "Medium": 0.3, "High": 0.8, "Jam": 1.5}
    impacts['Traffic'] = traffic_impact_map.get(user_input['traffic'], 0.5)
    
    # Time of day impact
    if 7 <= user_input['order_hour'] <= 9 or 17 <= user_input['order_hour'] <= 19:
        impacts['Peak Hours'] = 0.6
    else:
        impacts['Peak Hours'] = -0.2  # Benefit for off-peak
    
    # Agent performance impact
    impacts['Agent Rating'] = (5 - user_input['agent_rating']) * 0.3
    
    # Vehicle efficiency impact
    vehicle_impact_map = {"motorcycle": -0.3, "scooter": 0, "van": 0.4, "bicycle": 0.8}
    impacts['Vehicle Type'] = vehicle_impact_map.get(user_input['vehicle'], 0)
    
    return impacts

def main():
    # Header
    st.markdown('<h1 class="main-header">🚚 Delivery Time Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### AI-powered delivery insights and time estimation")
    
    # Load model
    with st.spinner("Loading prediction model..."):
        model, label_encoders = load_model_and_encoders()
    
    if model is None or label_encoders is None:
        st.error("❌ Could not load the prediction model. Please check if the model files exist in the models/ folder.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📋 Order Details", "🎯 Prediction & Insights", "📊 Factor Analysis"])
    
    with tab1:
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)
        st.subheader("Enter Order Details")
        
        # Create three columns for input form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📍 Location Details**")
            
            # Store coordinates
            store_lat = st.number_input("Store Latitude", value=12.9716, format="%.6f", key="store_lat")
            store_lon = st.number_input("Store Longitude", value=77.5946, format="%.6f", key="store_lon")
            
            # Drop coordinates
            drop_lat = st.number_input("Drop Latitude", value=12.9352, format="%.6f", key="drop_lat")
            drop_lon = st.number_input("Drop Longitude", value=77.6245, format="%.6f", key="drop_lon")
            
            # Auto-calculate distance
            distance = calculate_distance(store_lat, store_lon, drop_lat, drop_lon)
            st.metric("📏 Delivery Distance", f"{distance:.2f} km")
            
            st.markdown("**👤 Agent Details**")
            agent_age = st.slider("Agent Age", 18, 50, 30)
            agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.5, 0.1)
        
        with col2:
            st.markdown("**🕒 Time & Date**")
            order_hour = st.slider("Order Hour", 0, 23, 12)
            day_of_week = st.selectbox("Day of Week", 
                                     ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            month = st.selectbox("Month", 
                               ["January", "February", "March", "April", "May", "June",
                                "July", "August", "September", "October", "November", "December"])
            pickup_delay = st.slider("Pickup Delay (hours)", 0.0, 5.0, 0.5, 0.1)
            
            st.markdown("**🚗 Vehicle & Area**")
            vehicle = st.selectbox("Vehicle Type", ["motorcycle", "scooter", "van", "bicycle"])
            area = st.selectbox("Area Type", ["Urban", "Metropolitian", "Semi-Urban", "Other"])
        
        with col3:
            st.markdown("**🌦️ Weather & Traffic**")
            weather = st.selectbox("Weather Condition", 
                                 ["Sunny", "Cloudy", "Windy", "Fog", "Sandstorms", "Stormy"])
            traffic = st.selectbox("Traffic Condition", ["Low", "Medium", "High", "Jam"])
            
            st.markdown("**📦 Product Details**")
            category = st.selectbox("Product Category", 
                                  ["Electronics", "Clothing", "Food", "Books", "Furniture", 
                                   "Medicines", "Sports", "Toys", "Jewelry", "Cosmetics"])
            
            # Derived features (calculated automatically)
            st.markdown("**📊 Derived Features**")
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
        predict_button = st.button("🚀 Predict Delivery Time", type="primary", use_container_width=True)
    
    with tab2:
        st.markdown("### Prediction Results & Business Insights")
        
        if predict_button:
            with st.spinner("Analyzing delivery factors and generating insights..."):
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
                    
                    # Generate insights
                    insights, warnings, recommendations = generate_insights(prediction, user_input, distance, categorical_values)
                    
                    # Display results
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("📦 Product Category", category)
                        st.metric("📍 Delivery Distance", f"{distance:.2f} km")
                        st.metric("👤 Agent Experience", categorical_values['agent_experience'])
                    
                    with col2:
                        st.metric("🌦️ Weather", weather)
                        st.metric("🚦 Traffic", traffic)
                        st.metric("🚗 Vehicle", vehicle)
                    
                    with col3:
                        st.metric("🏙️ Area Type", area)
                        st.metric("🕒 Order Time", f"{order_hour:02d}:00")
                        st.metric("⏱️ Pickup Delay", f"{pickup_delay:.1f} hours")
                    
                    st.markdown("---")
                    
                    # Main prediction
                    st.markdown(f"## 🎯 Predicted Delivery Time: **{prediction:.1f} hours**")
                    
                    # Performance categorization
                    if prediction < 2:
                        st.success("⚡ **EXPRESS DELIVERY** - Premium service performance")
                    elif prediction < 4:
                        st.info("🚀 **STANDARD DELIVERY** - Meets service level expectations")
                    elif prediction < 6:
                        st.warning("🐢 **EXTENDED DELIVERY** - Moderate delays expected")
                    else:
                        st.error("🚨 **DELAYED DELIVERY** - Significant delays requiring attention")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 💡 BUSINESS INSIGHTS SECTION
                    st.subheader("💡 Business Insights & Analysis")
                    
                    for insight in insights:
                        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
                    
                    # ⚠️ WARNINGS SECTION
                    if warnings:
                        st.subheader("⚠️ Potential Issues & Risks")
                        for warning in warnings:
                            st.markdown(f'<div class="warning-box">{warning}</div>', unsafe_allow_html=True)
                    
                    # 🎯 RECOMMENDATIONS SECTION
                    if recommendations:
                        st.subheader("🎯 Optimization Recommendations")
                        for recommendation in recommendations:
                            st.markdown(f'<div class="success-box">{recommendation}</div>', unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
        
        else:
            st.info("👆 Go to 'Order Details' tab and click 'Predict Delivery Time' to see detailed insights")
    
    with tab3:
        st.markdown("### 📊 Factor Impact Analysis")
        
        if predict_button:
            # Calculate factor impacts
            factor_impacts = calculate_factor_impacts(user_input, distance)
            
            st.subheader("Factor Impact on Delivery Time")
            
            for factor, impact in factor_impacts.items():
                if impact > 0.5:
                    impact_class = "impact-high"
                    impact_text = "High Impact"
                elif impact > 0.2:
                    impact_class = "impact-medium" 
                    impact_text = "Medium Impact"
                else:
                    impact_class = "impact-low"
                    impact_text = "Low Impact"
                
                st.markdown(f"""
                <div class="factor-impact">
                    <strong>{factor}:</strong> 
                    <span class="{impact_class}">{impact:.2f} hours ({impact_text})</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Summary of key factors
            st.subheader("🔍 Key Performance Drivers")
            
            high_impact_factors = {k: v for k, v in factor_impacts.items() if v > 0.3}
            if high_impact_factors:
                st.write("**Primary factors affecting delivery time:**")
                for factor, impact in sorted(high_impact_factors.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"• **{factor}**: {impact:.2f} hours additional time")
            
            # Optimization opportunities
            st.subheader("💡 Quick Optimization Opportunities")
            optimization_tips = []
            
            if factor_impacts.get('Traffic', 0) > 0.5:
                optimization_tips.append("**Traffic Management**: Consider alternative routes or off-peak delivery")
            if factor_impacts.get('Weather', 0) > 0.5:
                optimization_tips.append("**Weather Planning**: Monitor weather updates and plan accordingly")
            if factor_impacts.get('Distance', 0) > 1.0:
                optimization_tips.append("**Distance Optimization**: Consider local distribution centers")
            if factor_impacts.get('Peak Hours', 0) > 0.3:
                optimization_tips.append("**Timing Adjustment**: Schedule deliveries during off-peak hours")
            
            for tip in optimization_tips:
                st.write(f"✅ {tip}")
        
        else:
            st.info("👆 Generate a prediction first to see factor impact analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with ❤️ using Streamlit and Machine Learning*")

if __name__ == "__main__":
    main()