import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from geopy.distance import geodesic
import json

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Nairobi traffic hotspots with coordinates
HOTSPOTS = {
    'CBD': {'lat': -1.2921, 'lon': 36.8219, 'base_congestion': 0.85},
    'Westlands': {'lat': -1.2680, 'lon': 36.8110, 'base_congestion': 0.75},
    'Thika_Road': {'lat': -1.2100, 'lon': 36.8900, 'base_congestion': 0.90},
    'Waiyaki_Way': {'lat': -1.2580, 'lon': 36.7850, 'base_congestion': 0.80},
    'Mombasa_Road': {'lat': -1.3230, 'lon': 36.8500, 'base_congestion': 0.88},
    'Ngong_Road': {'lat': -1.3000, 'lon': 36.7600, 'base_congestion': 0.72},
    'Jogoo_Road': {'lat': -1.2900, 'lon': 36.8700, 'base_congestion': 0.78},
    'Langata_Road': {'lat': -1.3400, 'lon': 36.7900, 'base_congestion': 0.70}
}

# Generate 6 months of hourly data (March 2025 - August 2025)
start_date = datetime(2025, 3, 1)
end_date = datetime(2025, 8, 31)
hours = int((end_date - start_date).total_seconds() / 3600) + 1

records = []

for i in range(hours):
    current_time = start_date + timedelta(hours=i)
    
    # Time features
    hour = current_time.hour
    day_of_week = current_time.weekday()  # 0=Monday, 6=Sunday
    is_weekend = day_of_week >= 5
    month = current_time.month
    
    # Weather simulation (rainy seasons: March-May, Oct-Dec in Nairobi)
    is_rainy_season = month in [3, 4, 5, 10, 11, 12]
    base_rain_prob = 0.35 if is_rainy_season else 0.10
    is_raining = random.random() < base_rain_prob
    rainfall_mm = np.random.exponential(8) if is_raining else 0
    temperature = np.random.normal(22, 3)  # Nairobi avg ~22°C
    
    # Rush hour multipliers
    morning_rush = 1.0
    evening_rush = 1.0
    
    if not is_weekend:
        if 7 <= hour <= 9:
            morning_rush = 1.4 + np.random.normal(0, 0.1)
        elif 17 <= hour <= 19:
            evening_rush = 1.5 + np.random.normal(0, 0.1)
        elif 12 <= hour <= 13:
            # Lunch hour slight bump
            morning_rush = 1.1
    
    # Weekend pattern (less congested, different peak times)
    if is_weekend:
        if 10 <= hour <= 14:
            morning_rush = 1.2  # Late morning shopping/leisure
        elif 19 <= hour <= 22:
            evening_rush = 1.15  # Nightlife
    
    # Night time reduction
    night_factor = 0.3 if (hour >= 22 or hour <= 5) else 1.0
    
    for location, coords in HOTSPOTS.items():
        base = coords['base_congestion']
        
        # Calculate congestion score (0-1)
        congestion = base * morning_rush * evening_rush * night_factor
        
        # Rain impact (varies by location - Thika Road floods more)
        rain_multiplier = 1.0
        if is_raining:
            if location in ['Thika_Road', 'Mombasa_Road']:
                rain_multiplier = 1.3 + (rainfall_mm / 50)  # Major roads flood
            else:
                rain_multiplier = 1.15 + (rainfall_mm / 80)
        
        congestion *= rain_multiplier
        
        # Add noise
        congestion += np.random.normal(0, 0.05)
        congestion = np.clip(congestion, 0.05, 1.0)
        
        # Derived metrics
        speed_kmh = max(5, 60 * (1 - congestion))  # Speed drops with congestion
        delay_minutes = max(0, (congestion - 0.3) * 40)  # Delay estimate
        
        # Fuel cost estimation (KES per km, assuming 10km distance)
        base_fuel_rate = 15.5  # KES per km at normal speed
        fuel_multiplier = 1 + (congestion * 0.6)  # Idling burns more fuel
        fuel_cost_kes = base_fuel_rate * fuel_multiplier * 10  # 10km trip
        
        # Matatu fare inflation (correlates with congestion)
        base_fare = 50  # KES base
        fare_inflation = 1 + (congestion * 0.4)
        matatu_fare = base_fare * fare_inflation
        
        records.append({
            'timestamp': current_time,
            'location': location,
            'latitude': coords['lat'],
            'longitude': coords['lon'],
            'congestion_score': round(congestion, 3),
            'speed_kmh': round(speed_kmh, 1),
            'delay_minutes': round(delay_minutes, 1),
            'is_raining': is_raining,
            'rainfall_mm': round(rainfall_mm, 1),
            'temperature_c': round(temperature, 1),
            'fuel_cost_kes': round(fuel_cost_kes, 2),
            'matatu_fare_kes': round(matatu_fare, 2),
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'month': month
        })

# Create DataFrame
df = pd.DataFrame(records)
print(f"Generated {len(df):,} records")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"\nLocations: {df['location'].nunique()}")
print(f"Records per location: {len(df) // df['location'].nunique():,}")

# Save raw data
df.to_csv('data/raw/nairobi_traffic_raw.csv', index=False)
print("\n✅ Saved to data/raw/nairobi_traffic_raw.csv")

# Quick validation
print("\n--- Sample Data ---")
print(df[df['location'] == 'CBD'].tail(3)[['timestamp', 'congestion_score', 'speed_kmh', 'is_raining']])

print("\n--- Congestion by Location (Mean) ---")
print(df.groupby('location')['congestion_score'].mean().sort_values(ascending=False))

print("\n--- Rush Hour Pattern (Weekday CBD) ---")
weekday_cbd = df[(df['location'] == 'CBD') & (~df['is_weekend'])]
rush_pattern = weekday_cbd.groupby('hour')['congestion_score'].mean()
print(rush_pattern.head(10))