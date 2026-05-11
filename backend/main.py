from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from typing import List, Optional
import json
import os

app = FastAPI(
    title="Nairobi Traffic Intelligence API",
    description="Real-time congestion forecasting for Nairobi traffic hotspots",
    version="1.0.0"
)

# CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
MODELS = {}
LOCATIONS = ['CBD', 'Westlands', 'Thika_Road', 'Waiyaki_Way', 
             'Mombasa_Road', 'Ngong_Road', 'Jogoo_Road', 'Langata_Road']

def load_models():
    for loc in LOCATIONS:
        model_path = f'models/xgboost_{loc.lower()}.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                MODELS[loc] = pickle.load(f)
            print(f"✅ Loaded model for {loc}")
        else:
            print(f"⚠️ Model not found: {model_path}")

load_models()

# Load historical data for lag features
HISTORICAL_DATA = None
def load_historical():
    global HISTORICAL_DATA
    try:
        HISTORICAL_DATA = pd.read_csv('data/raw/nairobi_traffic_raw.csv', parse_dates=['timestamp'])
        print(f"✅ Loaded historical data: {len(HISTORICAL_DATA)} records")
    except Exception as e:
        print(f"⚠️ Could not load historical data: {e}")

load_historical()

# Pydantic models
class ForecastRequest(BaseModel):
    location: str
    hours_ahead: int = 24
    include_weather: bool = False
    rain_probability: float = 0.0

class CongestionResponse(BaseModel):
    location: str
    current_congestion: float
    forecast: List[dict]
    risk_score: str
    estimated_delay_minutes: float
    fuel_cost_kes: float
    matatu_fare_kes: float

class RouteRequest(BaseModel):
    origin: str
    destination: str
    departure_time: Optional[datetime] = None

# Helper functions
def get_location_coords(location: str) -> dict:
    coords = {
        'CBD': {'lat': -1.2921, 'lon': 36.8219},
        'Westlands': {'lat': -1.2680, 'lon': 36.8110},
        'Thika_Road': {'lat': -1.2100, 'lon': 36.8900},
        'Waiyaki_Way': {'lat': -1.2580, 'lon': 36.7850},
        'Mombasa_Road': {'lat': -1.3230, 'lon': 36.8500},
        'Ngong_Road': {'lat': -1.3000, 'lon': 36.7600},
        'Jogoo_Road': {'lat': -1.2900, 'lon': 36.8700},
        'Langata_Road': {'lat': -1.3400, 'lon': 36.7900}
    }
    return coords.get(location, {'lat': -1.2921, 'lon': 36.8219})

def calculate_risk_score(congestion: float) -> str:
    if congestion < 0.3:
        return "LOW"
    elif congestion < 0.6:
        return "MODERATE"
    elif congestion < 0.8:
        return "HIGH"
    else:
        return "SEVERE"

def derive_metrics(congestion: float) -> dict:
    speed = max(5, 60 * (1 - congestion))
    delay = max(0, (congestion - 0.3) * 40)
    fuel_cost = 15.5 * (1 + congestion * 0.6) * 10
    matatu_fare = 50 * (1 + congestion * 0.4)
    return {
        'speed_kmh': round(speed, 1),
        'delay_minutes': round(delay, 1),
        'fuel_cost_kes': round(fuel_cost, 2),
        'matatu_fare_kes': round(matatu_fare, 2)
    }

def create_forecast_features(location: str, target_time: datetime, 
                           last_known_congestion: float, 
                           is_raining: bool = False,
                           rainfall_mm: float = 0) -> pd.DataFrame:
    """Create feature vector for XGBoost prediction"""
    hour = target_time.hour
    dow = target_time.weekday()
    month = target_time.month
    
    # Get historical lags for this location
    loc_hist = HISTORICAL_DATA[HISTORICAL_DATA['location'] == location] if HISTORICAL_DATA is not None else None
    
    if loc_hist is not None and len(loc_hist) > 0:
        last_vals = loc_hist['congestion_score'].tail(200).values
        lag_1h = last_vals[-1] if len(last_vals) >= 1 else last_known_congestion
        lag_24h = last_vals[-24] if len(last_vals) >= 24 else last_known_congestion
        lag_168h = last_vals[-168] if len(last_vals) >= 168 else last_known_congestion
        rolling_3h = np.mean(last_vals[-3:]) if len(last_vals) >= 3 else last_known_congestion
        rolling_24h = np.mean(last_vals[-24:]) if len(last_vals) >= 24 else last_known_congestion
    else:
        lag_1h = lag_24h = lag_168h = rolling_3h = rolling_24h = last_known_congestion
    
    features = pd.DataFrame([{
        'hour_sin': np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np.pi * hour / 24),
        'dow_sin': np.sin(2 * np.pi * dow / 7),
        'dow_cos': np.cos(2 * np.pi * dow / 7),
        'month': month,
        'is_raining': 1 if is_raining else 0,
        'rainfall_mm': rainfall_mm,
        'temperature_c': 22,
        'lag_1h': lag_1h,
        'lag_24h': lag_24h,
        'lag_168h': lag_168h,
        'rolling_3h': rolling_3h,
        'rolling_24h': rolling_24h
    }])
    
    return features

# API Endpoints
@app.get("/")
def root():
    return {
        "message": "Nairobi Traffic Intelligence API",
        "version": "1.0.0",
        "locations": LOCATIONS,
        "endpoints": [
            "/current/{location}",
            "/forecast/{location}",
            "/route-risk",
            "/locations"
        ]
    }

@app.get("/locations")
def get_locations():
    """Get all available locations with coordinates"""
    return [
        {
            "name": loc,
            "coordinates": get_location_coords(loc),
            "current_status": "active"
        }
        for loc in LOCATIONS
    ]

@app.get("/current/{location}")
def get_current_status(location: str):
    """Get current congestion status for a location"""
    if location not in LOCATIONS:
        raise HTTPException(status_code=404, detail=f"Location '{location}' not found")
    
    # Get latest historical value or simulate
    if HISTORICAL_DATA is not None:
        loc_data = HISTORICAL_DATA[HISTORICAL_DATA['location'] == location]
        if len(loc_data) > 0:
            latest = loc_data.iloc[-1]
            congestion = latest['congestion_score']
            is_raining = bool(latest['is_raining'])
        else:
            congestion = 0.5
            is_raining = False
    else:
        congestion = 0.5
        is_raining = False
    
    metrics = derive_metrics(congestion)
    
    return {
        "location": location,
        "timestamp": datetime.now().isoformat(),
        "congestion_score": round(congestion, 3),
        "risk_score": calculate_risk_score(congestion),
        "is_raining": is_raining,
        **metrics
    }

@app.post("/forecast", response_model=CongestionResponse)
def get_forecast(request: ForecastRequest):
    """Get congestion forecast for a location"""
    location = request.location
    
    if location not in LOCATIONS:
        raise HTTPException(status_code=404, detail=f"Location '{location}' not found")
    
    if location not in MODELS:
        raise HTTPException(status_code=503, detail=f"Model for '{location}' not loaded")
    
    # Get current status
    current = get_current_status(location)
    current_congestion = current['congestion_score']
    
    # Generate forecasts
    forecasts = []
    now = datetime.now()
    
    for i in range(request.hours_ahead):
        forecast_time = now + timedelta(hours=i+1)
        
        # Determine rain probability
        is_raining = np.random.random() < request.rain_probability if request.include_weather else False
        rainfall = np.random.exponential(5) if is_raining else 0
        
        # Create features and predict
        features = create_forecast_features(
            location, forecast_time, current_congestion, 
            is_raining, rainfall
        )
        
        pred = MODELS[location].predict(features)[0]
        pred = float(np.clip(pred, 0.05, 1.0))
        
        metrics = derive_metrics(pred)
        
        forecasts.append({
            "hour": i + 1,
            "timestamp": forecast_time.isoformat(),
            "congestion_score": round(pred, 3),
            "risk_score": calculate_risk_score(pred),
            "is_raining": is_raining,
            **metrics
        })
    
    # Calculate overall route risk
    avg_congestion = np.mean([f['congestion_score'] for f in forecasts])
    
    return CongestionResponse(
        location=location,
        current_congestion=round(current_congestion, 3),
        forecast=forecasts,
        risk_score=calculate_risk_score(avg_congestion),
        estimated_delay_minutes=forecasts[0]['delay_minutes'],
        fuel_cost_kes=forecasts[0]['fuel_cost_kes'],
        matatu_fare_kes=forecasts[0]['matatu_fare_kes']
    )

@app.post("/route-risk")
def calculate_route_risk(request: RouteRequest):
    """Calculate risk score for a route between two locations"""
    if request.origin not in LOCATIONS or request.destination not in LOCATIONS:
        raise HTTPException(status_code=404, detail="Invalid origin or destination")
    
    departure = request.departure_time or datetime.now()
    
    # Get forecasts for both endpoints
    origin_forecast = get_forecast(ForecastRequest(
        location=request.origin, 
        hours_ahead=1,
        include_weather=False
    ))
    
    dest_forecast = get_forecast(ForecastRequest(
        location=request.destination,
        hours_ahead=1,
        include_weather=False
    ))
    
    # Calculate route metrics
    avg_congestion = (origin_forecast.current_congestion + dest_forecast.current_congestion) / 2
    total_delay = origin_forecast.estimated_delay_minutes + dest_forecast.estimated_delay_minutes
    total_fuel = origin_forecast.fuel_cost_kes + dest_forecast.fuel_cost_kes
    
    # Risk assessment
    risk = calculate_risk_score(avg_congestion)
    
    # Recommendations
    recommendations = []
    if risk == "SEVERE":
        recommendations.append("Avoid this route. Consider alternative transport.")
        recommendations.append("Expected delays exceed 45 minutes.")
    elif risk == "HIGH":
        recommendations.append("Leave 30 minutes earlier than usual.")
        recommendations.append("Monitor traffic updates before departure.")
    elif risk == "MODERATE":
        recommendations.append("Standard commute time with minor delays expected.")
    else:
        recommendations.append("Clear route. Normal commute time.")
    
    return {
        "origin": request.origin,
        "destination": request.destination,
        "departure_time": departure.isoformat(),
        "route_risk": risk,
        "average_congestion": round(avg_congestion, 3),
        "estimated_total_delay_minutes": round(total_delay, 1),
        "estimated_fuel_cost_kes": round(total_fuel, 2),
        "origin_status": {
            "congestion": origin_forecast.current_congestion,
            "risk": origin_forecast.risk_score
        },
        "destination_status": {
            "congestion": dest_forecast.current_congestion,
            "risk": dest_forecast.risk_score
        },
        "recommendations": recommendations
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(MODELS),
        "locations": len(LOCATIONS),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)