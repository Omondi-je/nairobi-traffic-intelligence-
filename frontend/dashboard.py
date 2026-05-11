import streamlit as st
import requests
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import tempfile
import os

# Page config
st.set_page_config(
    page_title="Nairobi Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f1f1f;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-severe { background-color: #dc3545; color: white; padding: 0.3rem 0.8rem; border-radius: 1rem; font-weight: bold; }
    .risk-high { background-color: #fd7e14; color: white; padding: 0.3rem 0.8rem; border-radius: 1rem; font-weight: bold; }
    .risk-moderate { background-color: #ffc107; color: black; padding: 0.3rem 0.8rem; border-radius: 1rem; font-weight: bold; }
    .risk-low { background-color: #28a745; color: white; padding: 0.3rem 0.8rem; border-radius: 1rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE = "https://nairobi-traffic-api.onrender.com"

# Sidebar
st.sidebar.markdown("## 🚦 Navigation")
page = st.sidebar.radio(
    "Select Dashboard",
    ["🏠 Live Overview", "📊 Location Analytics", "🔮 Forecasting", "🗺️ Route Planner", "📈 Model Performance"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "Real-time traffic intelligence for Nairobi. "
    "Predicting congestion, optimizing routes, and saving commuter time."
)

# Helper functions
@st.cache_data(ttl=60)
def fetch_locations():
    try:
        response = requests.get(f"{API_BASE}/locations")
        return response.json()
    except:
        return []

@st.cache_data(ttl=60)
def fetch_current(location):
    try:
        response = requests.get(f"{API_BASE}/current/{location}")
        return response.json()
    except:
        return None

@st.cache_data(ttl=300)
def fetch_forecast(location, hours=24):
    try:
        response = requests.post(
            f"{API_BASE}/forecast",
            json={"location": location, "hours_ahead": hours, "include_weather": False}
        )
        return response.json()
    except:
        return None

@st.cache_data(ttl=60)
def fetch_route_risk(origin, destination):
    try:
        response = requests.post(
            f"{API_BASE}/route-risk",
            json={"origin": origin, "destination": destination}
        )
        return response.json()
    except:
        return None

def get_risk_badge(risk):
    risk_class = f"risk-{risk.lower()}"
    return f'<span class="{risk_class}">{risk}</span>'

def render_folium_map(m):
    """Render Folium map as HTML iframe without st_folium"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
        m.save(tmpfile.name)
        with open(tmpfile.name, 'r') as f:
            html = f.read()
        os.unlink(tmpfile.name)
    return html

# ==================== PAGE 1: LIVE OVERVIEW ====================
if page == "🏠 Live Overview":
    st.markdown('<p class="main-header">Nairobi Traffic Intelligence</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time Congestion Monitoring & Forecasting System</p>', unsafe_allow_html=True)
    
    locations = fetch_locations()
    
    if not locations:
        st.error("⚠️ Cannot connect to API. Make sure backend is running on port 8000.")
        st.stop()
    
    # Current status cards
    st.markdown("### 🌍 Current Traffic Status")
    
    cols = st.columns(4)
    for idx, loc in enumerate(locations[:4]):
        data = fetch_current(loc['name'])
        if data:
            with cols[idx]:
                risk_html = get_risk_badge(data['risk_score'])
                st.markdown(f"""
                <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
                    <h4 style="margin: 0;">{loc['name'].replace('_', ' ')}</h4>
                    <p style="font-size: 2rem; margin: 5px 0;">{data['congestion_score']:.0%}</p>
                    <p>Congestion Score</p>
                    {risk_html}
                    <p style="font-size: 0.9rem; color: #666; margin-top: 8px;">
                        🚗 {data['speed_kmh']} km/h | ⏱️ {data['delay_minutes']} min delay
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    cols = st.columns(4)
    for idx, loc in enumerate(locations[4:]):
        data = fetch_current(loc['name'])
        if data:
            with cols[idx]:
                risk_html = get_risk_badge(data['risk_score'])
                st.markdown(f"""
                <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
                    <h4 style="margin: 0;">{loc['name'].replace('_', ' ')}</h4>
                    <p style="font-size: 2rem; margin: 5px 0;">{data['congestion_score']:.0%}</p>
                    <p>Congestion Score</p>
                    {risk_html}
                    <p style="font-size: 0.9rem; color: #666; margin-top: 8px;">
                        🚗 {data['speed_kmh']} km/h | ⏱️ {data['delay_minutes']} min delay
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    # Live Map - Using HTML iframe instead of st_folium
    st.markdown("---")
    st.markdown("### 🗺️ Live Congestion Heatmap")
    
    m = folium.Map(location=[-1.2921, 36.8219], zoom_start=12, tiles='CartoDB dark_matter')
    
    heat_data = []
    for loc in locations:
        data = fetch_current(loc['name'])
        if data:
            heat_data.append([
                loc['coordinates']['lat'],
                loc['coordinates']['lon'],
                data['congestion_score'] * 100
            ])
    
    if heat_data:
        HeatMap(heat_data, radius=25, blur=15, max_zoom=13).add_to(m)
    
    # Add markers
    for loc in locations:
        data = fetch_current(loc['name'])
        if data:
            color = 'red' if data['congestion_score'] > 0.7 else 'orange' if data['congestion_score'] > 0.4 else 'green'
            folium.CircleMarker(
                location=[loc['coordinates']['lat'], loc['coordinates']['lon']],
                radius=8,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
    
    # Render as HTML iframe
    map_html = render_folium_map(m)
    st.components.v1.html(map_html, width=700, height=500)

# ==================== PAGE 2: LOCATION ANALYTICS ====================
elif page == "📊 Location Analytics":
    st.markdown('<p class="main-header">Location Analytics</p>', unsafe_allow_html=True)
    
    locations = fetch_locations()
    loc_names = [l['name'] for l in locations]
    
    selected_loc = st.selectbox("Select Location", loc_names, format_func=lambda x: x.replace('_', ' '))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📍 Current Status")
        current = fetch_current(selected_loc)
        if current:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = current['congestion_score'] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Congestion %"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred" if current['congestion_score'] > 0.7 else "orange" if current['congestion_score'] > 0.4 else "green"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 60], 'color': "yellow"},
                        {'range': [60, 80], 'color': "orange"},
                        {'range': [80, 100], 'color': "red"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            - **Speed**: {current['speed_kmh']} km/h
            - **Delay**: {current['delay_minutes']} minutes
            - **Fuel Cost**: KES {current['fuel_cost_kes']}
            - **Matatu Fare**: KES {current['matatu_fare_kes']}
            - **Weather**: {'🌧️ Raining' if current['is_raining'] else '☀️ Dry'}
            """)
    
    with col2:
        st.markdown("### 📈 24-Hour Forecast")
        forecast = fetch_forecast(selected_loc, 24)
        if forecast and forecast.get('forecast'):
            df = pd.DataFrame(forecast['forecast'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            fig = px.area(df, x='timestamp', y='congestion_score',
                         title=f"{selected_loc.replace('_', ' ')} - Next 24 Hours",
                         labels={'congestion_score': 'Congestion', 'timestamp': 'Time'})
            fig.update_traces(fillcolor='rgba(255,0,0,0.2)', line_color='red')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 📋 Hourly Breakdown")
            display_df = df[['hour', 'congestion_score', 'risk_score', 'speed_kmh', 'delay_minutes']].copy()
            display_df.columns = ['Hour', 'Congestion', 'Risk', 'Speed (km/h)', 'Delay (min)']
            st.dataframe(display_df.head(12), use_container_width=True)

# ==================== PAGE 3: FORECASTING ====================
elif page == "🔮 Forecasting":
    st.markdown('<p class="main-header">Advanced Forecasting</p>', unsafe_allow_html=True)
    
    locations = fetch_locations()
    loc_names = [l['name'] for l in locations]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_loc = st.selectbox("Location", loc_names, format_func=lambda x: x.replace('_', ' '))
    with col2:
        hours = st.slider("Forecast Hours", 1, 168, 24)
    with col3:
        include_weather = st.checkbox("Simulate Rain", value=False)
    
    if st.button("🔮 Generate Forecast", type="primary"):
        with st.spinner("Running XGBoost forecast model..."):
            forecast = fetch_forecast(selected_loc, hours)
            
            if forecast:
                df = pd.DataFrame(forecast['forecast'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['timestamp'], y=df['congestion_score'],
                    mode='lines', name='Congestion Forecast',
                    line=dict(color='red', width=2),
                    fill='tozeroy', fillcolor='rgba(255,0,0,0.1)'
                ))
                
                fig.add_hrect(y0=0.8, y1=1.0, line_width=0, fillcolor="red", opacity=0.1, annotation_text="SEVERE")
                fig.add_hrect(y0=0.6, y1=0.8, line_width=0, fillcolor="orange", opacity=0.1, annotation_text="HIGH")
                fig.add_hrect(y0=0.3, y1=0.6, line_width=0, fillcolor="yellow", opacity=0.1, annotation_text="MODERATE")
                
                fig.update_layout(
                    title=f"{selected_loc.replace('_', ' ')} - {hours} Hour Forecast",
                    xaxis_title="Time",
                    yaxis_title="Congestion Score",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                avg_congestion = df['congestion_score'].mean()
                max_congestion = df['congestion_score'].max()
                peak_hour = df.loc[df['congestion_score'].idxmax(), 'hour']
                
                cols = st.columns(4)
                cols[0].metric("Average Congestion", f"{avg_congestion:.1%}")
                cols[1].metric("Peak Congestion", f"{max_congestion:.1%}")
                cols[2].metric("Peak Hour", f"+{peak_hour}h")
                cols[3].metric("Route Risk", forecast['risk_score'])
                
                st.markdown("### 💰 Commute Cost Estimation")
                cost_df = df[['hour', 'congestion_score', 'fuel_cost_kes', 'matatu_fare_kes', 'delay_minutes']].copy()
                cost_df['hour'] = cost_df['hour'].apply(lambda x: f"+{x}h")
                
                fig_cost = go.Figure()
                fig_cost.add_trace(go.Bar(x=cost_df['hour'], y=cost_df['fuel_cost_kes'], name='Fuel Cost (KES)'))
                fig_cost.add_trace(go.Scatter(x=cost_df['hour'], y=cost_df['delay_minutes'], name='Delay (min)', yaxis='y2'))
                fig_cost.update_layout(
                    yaxis2=dict(title='Delay (minutes)', overlaying='y', side='right'),
                    height=300,
                    title="Cost vs Delay Over Forecast Period"
                )
                st.plotly_chart(fig_cost, use_container_width=True)

# ==================== PAGE 4: ROUTE PLANNER ====================
elif page == "🗺️ Route Planner":
    st.markdown('<p class="main-header">Smart Route Planner</p>', unsafe_allow_html=True)
    
    locations = fetch_locations()
    loc_names = [l['name'] for l in locations]
    
    col1, col2 = st.columns(2)
    with col1:
        origin = st.selectbox("From", loc_names, format_func=lambda x: x.replace('_', ' '))
    with col2:
        destination = st.selectbox("To", [l for l in loc_names if l != origin], format_func=lambda x: x.replace('_', ' '))
    
    if st.button("🚀 Calculate Route Risk", type="primary"):
        with st.spinner("Analyzing route conditions..."):
            route_data = fetch_route_risk(origin, destination)
            
            if route_data:
                st.markdown("### 🛣️ Route Analysis")
                
                cols = st.columns(3)
                cols[0].markdown(f"""
                <div style="text-align: center; padding: 20px; border: 2px solid #ddd; border-radius: 10px;">
                    <h2>{route_data['route_risk']}</h2>
                    <p>Overall Risk</p>
                </div>
                """, unsafe_allow_html=True)
                
                cols[1].metric("Avg Congestion", f"{route_data['average_congestion']:.1%}")
                cols[2].metric("Total Delay", f"{route_data['estimated_total_delay_minutes']:.0f} min")
                
                st.markdown("### 📍 Endpoint Comparison")
                col_o, col_d = st.columns(2)
                
                with col_o:
                    st.markdown(f"**{origin.replace('_', ' ')}**")
                    st.markdown(f"- Congestion: {route_data['origin_status']['congestion']:.1%}")
                    st.markdown(f"- Risk: {route_data['origin_status']['risk']}")
                
                with col_d:
                    st.markdown(f"**{destination.replace('_', ' ')}**")
                    st.markdown(f"- Congestion: {route_data['destination_status']['congestion']:.1%}")
                    st.markdown(f"- Risk: {route_data['destination_status']['risk']}")
                
                st.markdown("### 💡 Recommendations")
                for rec in route_data['recommendations']:
                    st.info(rec)
                
                st.markdown("### 💵 Cost Breakdown")
                st.markdown(f"**Estimated Fuel Cost**: KES {route_data['estimated_fuel_cost_kes']:.2f}")
                
                # Route map using HTML iframe
                m = folium.Map(location=[-1.2921, 36.8219], zoom_start=12)
                origin_coords = next(l['coordinates'] for l in locations if l['name'] == origin)
                dest_coords = next(l['coordinates'] for l in locations if l['name'] == destination)
                
                folium.Marker(
                    [origin_coords['lat'], origin_coords['lon']],
                    icon=folium.Icon(color='green')
                ).add_to(m)
                
                folium.Marker(
                    [dest_coords['lat'], dest_coords['lon']],
                    icon=folium.Icon(color='red')
                ).add_to(m)
                
                folium.PolyLine(
                    locations=[[origin_coords['lat'], origin_coords['lon']], 
                              [dest_coords['lat'], dest_coords['lon']]],
                    color='blue', weight=4, opacity=0.7
                ).add_to(m)
                
                map_html = render_folium_map(m)
                st.components.v1.html(map_html, width=700, height=400)

# ==================== PAGE 5: MODEL PERFORMANCE ====================
elif page == "📈 Model Performance":
    st.markdown('<p class="main-header">Model Performance</p>', unsafe_allow_html=True)
    
    try:
        summary = pd.read_csv('reports/forecast_model_summary.csv')
        
        st.markdown("### 🎯 XGBoost Forecasting Accuracy")
        
        fig = px.bar(summary, x='location', y='mae', color='rmse',
                    title='Mean Absolute Error by Location',
                    labels={'mae': 'MAE (lower is better)', 'location': 'Location'},
                    color_continuous_scale='RdYlGn_r')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📊 Detailed Metrics")
        display_summary = summary.copy()
        display_summary.columns = ['Location', 'MAE', 'RMSE', 'Avg Congestion']
        display_summary['MAE'] = display_summary['MAE'].apply(lambda x: f"{x:.4f}")
        display_summary['RMSE'] = display_summary['RMSE'].apply(lambda x: f"{x:.4f}")
        display_summary['Avg Congestion'] = display_summary['Avg Congestion'].apply(lambda x: f"{x:.1%}")
        st.dataframe(display_summary, use_container_width=True)
        
        st.markdown("### 🧠 Model Architecture")
        st.markdown("""
        - **Algorithm**: XGBoost Regressor
        - **Features**: 13 (temporal cyclical encoding + weather + lag features)
        - **Training Data**: 6 months hourly data (35,144 records)
        - **Top Predictors**: lag_24h, lag_168h, is_raining
        - **Validation**: 80/20 chronological split
        """)
        
    except FileNotFoundError:
        st.warning("Model summary not found. Run the forecasting notebook first.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Nairobi Traffic Intelligence System | Built with FastAPI + XGBoost + Streamlit</p>
</div>
""", unsafe_allow_html=True)