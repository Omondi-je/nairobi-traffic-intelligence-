# 🚦 Nairobi Traffic Intelligence System

> **Real-time congestion forecasting + route optimization for Africa's most dynamic city.**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-red)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange)](https://xgboost.ai)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)

---

## 🎯 The Problem

Nairobi loses **KES 50M+ daily** to traffic congestion. Commuters face:
- Unpredictable rush-hour delays (20–60 minutes)
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

## ⚠️ Data Transparency

> **This project currently uses synthetic traffic simulations calibrated using publicly observable Nairobi congestion patterns** (rush-hour behavior, rain impact, known bottleneck corridors like Thika Road and Mombasa Road).  
>  
> The data generation engine mirrors real-world dynamics without relying on proprietary traffic APIs. This ensures the system is fully reproducible and deployable without API costs.

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT DASHBOARD                        │
│              (Port 8501 — Interactive UI)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Live Map    │  │ Forecasts   │  │ Route Risk Planner  │  │
│  │ Heatmap     │  │ Charts      │  │ Cost Estimator      │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
└─────────┼────────────────┼────────────────────┼─────────────┘
          │                │                    │
          ▼                ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND                              │
│              (Port 8000 — REST API)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐    │
│  │ /health     │  │ /forecast   │  │ /route-risk       │    │
│  │ /current/*  │  │ POST JSON   │  │ POST JSON         │    │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘    │
└─────────┼────────────────┼────────────────────┼───────────────┘
          │                │                    │
          ▼                ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                 XGBOOST FORECASTING ENGINE                    │
│                                                               │
│  • 8 location-specific models (MAE: 0.03–0.04)              │
│  • Cyclical temporal encoding (hour/day sin-cos)              │
│  • Lag features: 1h, 24h, 168h (1 week)                       │
│  • Weather regressors (rain, temperature)                     │
│  • 35,000 training records (6 months hourly)                │
└─────────────────────────────────────────────────────────────┘
```

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

**Key Insight**: Rain increases congestion by **10% on average**, with Thika Road and Mombasa Road most affected (+13–14%).


## 🚀 Quick Start

### Prerequisites

```bash
Python 3.11+
Docker (optional)
```

### Local Setup

```bash
# Clone
git clone https://github.com/Omondi-je/nairobi-traffic-intelligence-.git
cd nairobi-traffic-intelligence-

# Install dependencies
pip install -r requirements.txt

# Generate data (if missing)
python notebooks/01_data_generation.py

# Train models (if missing)
python notebooks/02_eda_forecasting.py

# Start API (Terminal 1)
python backend/main.py

# Start Dashboard (Terminal 2)
streamlit run frontend/dashboard.py
```

### Docker

```bash
docker build -t nairobi-traffic .
docker run -p 8000:8000 -p 8501:8501 nairobi-traffic
```

**Access:**
- API: `http://localhost:8000`
- Dashboard: `http://localhost:8501`

---

## 📡 API Endpoints

| Endpoint | Method | Purpose | Example |
|----------|--------|---------|---------|
| `/health` | GET | System health check | `curl /health` |
| `/locations` | GET | List all traffic hotspots | `curl /locations` |
| `/current/{location}` | GET | Current congestion status | `curl /current/Thika_Road` |
| `/forecast` | POST | Predict congestion N hours ahead | `POST /forecast {"location":"CBD","hours_ahead":24}` |
| `/route-risk` | POST | Assess route risk + cost | `POST /route-risk {"origin":"CBD","destination":"Westlands"}` |

### Example Requests

```bash
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
```

---

## 📁 Project Structure

```
nairobi-traffic-intelligence/
├── backend/
│   └── main.py                    # FastAPI application (5 endpoints)
├── frontend/
│   └── dashboard.py               # Streamlit interactive dashboard
├── notebooks/
│   ├── 01_data_generation.py      # Synthetic data engine (35K records)
│   └── 02_eda_forecasting.py      # XGBoost model training + evaluation
├── models/
│   └── xgboost_*.pkl              # 8 location-specific trained models
├── data/
│   └── raw/
│       └── nairobi_traffic_raw.csv    # 6 months hourly traffic data
├── reports/
│   ├── heatmaps/
│   │   └── eda_overview.png       # EDA visualizations
│   ├── forecasts/
│   │   └── xgboost_forecasts.png  # Model performance charts
│   └── screenshots/               # Dashboard screenshots (add these)
├── Dockerfile                     # Container definition
├── docker-compose.yml             # Multi-service orchestration
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🧠 Technical Highlights

| Skill | Implementation |
|-------|----------------|
| **Time-Series Forecasting** | XGBoost with cyclical encoding (sin/cos for hour/day) |
| **Feature Engineering** | Lag features (1h, 24h, 168h), rolling averages, weather regressors |
| **Geospatial Visualization** | Folium heatmaps with coordinate-based risk mapping |
| **API Design** | RESTful FastAPI with Pydantic validation + CORS |
| **Containerization** | Docker with multi-service orchestration |
| **Data Storytelling** | Streamlit dashboard with Plotly interactive charts |

---

## 🌍 Real-World Impact

| Stakeholder | Benefit |
|-------------|---------|
| **Commuters** | Save 15–30 minutes with predictive routing |
| **Matatu Operators** | Optimize pricing based on congestion forecasts |
| **Urban Planners** | Identify chronic bottleneck corridors |
| **Insurance** | Risk-adjusted premiums for high-congestion routes |

---

## 🔮 Future Roadmap

- [ ] Live TomTom / Google Maps API integration
- [ ] SMS / WhatsApp alerts for severe congestion
- [ ] Matatu fleet optimization (multi-stop routing)
- [ ] Historical comparison: "Today vs Last Week"
- [ ] Mobile app with push notifications

---

## 📬 Contact

Built by **Jeff Omondi Ooko** — Data Engineer & Urban Intelligence Developer
 | 📧 Email @ojeff1211.gmail.com
---

> *"The best time to predict traffic was yesterday. The second best time is now."*
EOF
```

## Step 2: Commit & Push

```bash
git add README.md
git commit -m "docs: polish README with transparency, screenshots section, API table, clean architecture diagram"
git push origin main
```

---
