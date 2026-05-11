# 🚦 Nairobi Traffic Intelligence System

&gt; **Real-time congestion forecasting + route optimization for Africa's most dynamic city.**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-red)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange)](https://xgboost.ai)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)

---

## 🎯 The Problem

Nairobi loses **KES 50M+ daily** to traffic congestion. Commuters face:
- Unpredictable rush-hour delays (20-60 minutes)
- Fuel waste from idling in jams
- Matatu fare inflation during peak congestion
- No data-driven route planning

## 💡 The Solution

A full-stack **traffic intelligence platform** that:
- **Predicts** congestion 7 days ahead using XGBoost + temporal features
- **Visualizes** live heatmaps across 8 major Nairobi corridors
- **Optimizes** routes with risk scores + cost estimates
- **Correlates** weather (rain) with traffic impact (+10% congestion)

---

## 🏗 Architecture
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Streamlit     │────▶│    FastAPI      │────▶│   XGBoost       │
│   Dashboard     │     │    Backend      │     │   Models (8)    │
│  (Port 8501)    │     │  (Port 8000)    │     │  MAE: 0.03-0.04 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
│                       │                       │
▼                       ▼                       ▼
Folium Heatmaps         5 REST Endpoints      35K Training Records
Route Planner           Health/Current/      Weather Correlation
Cost Estimator          Forecast/Route-Risk   Lag Features (1h/24h/168h)

---

## 📊 Model Performance

| Location | MAE | RMSE | Avg Congestion | Top Predictor |
|----------|-----|------|----------------|---------------|
| **Thika Road** | 0.031 | 0.043 | 73.9% | lag_24h |
| **Mombasa Road** | 0.033 | 0.046 | 73.0% | lag_168h |
| **CBD** | 0.034 | 0.046 | 71.3% | lag_24h |
| **Waiyaki Way** | 0.037 | 0.049 | 68.9% | lag_24h |
| **Jogoo Road** | 0.038 | 0.050 | 67.7% | lag_24h |
| **Ngong Road** | 0.039 | 0.051 | 63.9% | lag_24h |
| **Westlands** | 0.039 | 0.050 | 65.7% | lag_168h |
| **Langata Road** | 0.042 | 0.054 | 62.5% | lag_168h |

**Key Insight**: Rain increases congestion by **10% on average**, with Thika Road and Mombasa Road most affected (+13-14%).

---

## 🚀 Quick Start

### Local (Docker)
```bash
# Clone
git clone https://github.com/Omondi-je/nairobi-traffic-intelligence-.git
cd nairobi-traffic-intelligence-

# Build & run
docker build -t nairobi-traffic .
docker run -p 8000:8000 -p 8501:8501 nairobi-traffic

# API: http://localhost:8000
# Dashboard: http://localhost:8501
# Health check
curl http://localhost:8000/health

# Current status
curl http://localhost:8000/current/Thika_Road

# 24-hour forecast
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"location": "CBD", "hours_ahead": 24}'

# Route risk assessment
curl -X POST http://localhost:8000/route-risk \
  -H "Content-Type: application/json" \
  -d '{"origin": "CBD", "destination": "Westlands"}'
📁 Project Structure
nairobi-traffic-intelligence/
├── backend/
│   └── main.py              # FastAPI application
├── frontend/
│   └── dashboard.py         # Streamlit dashboard
├── notebooks/
│   ├── 01_data_generation.py   # Synthetic data engine
│   └── 02_eda_forecasting.py # XGBoost model training
├── models/
│   └── xgboost_*.pkl       # 8 location-specific models
├── data/
│   └── raw/
│       └── nairobi_traffic_raw.csv  # 35K records
├── reports/
│   ├── heatmaps/eda_overview.png
│   └── forecasts/xgboost_forecasts.png
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
🧠 Technical Highlights
Table
Skill	Implementation
Time-Series Forecasting	XGBoost with cyclical encoding (sin/cos for hour/day)
Feature Engineering	Lag features (1h, 24h, 168h), rolling averages, weather
Geospatial Viz	Folium heatmaps + coordinate-based risk mapping
API Design	RESTful FastAPI with Pydantic validation
Containerization	Multi-service Docker with health checks
Data Storytelling	Streamlit dashboard with Plotly interactive charts
🌍 Real-World Impact
Commuters: Save 15-30 minutes with predictive routing
Matatu Operators: Optimize pricing based on congestion forecasts
Urban Planners: Identify chronic bottleneck corridors
Insurance: Risk-adjusted premiums for high-congestion routes
🔮 Future Roadmap
[ ] Live TomTom/Google Maps API integration
[ ] SMS/WhatsApp alerts for severe congestion
[ ] Matatu fleet optimization (multi-stop routing)
[ ] Historical comparison: "Today vs Last Week"
[ ] Mobile app with push notifications
📬 Contact
Built by Jeff Omondi Ooko — Data Engineer & Urban Intelligence Developer
The best time to predict traffic was yesterday. The second best time is now.
