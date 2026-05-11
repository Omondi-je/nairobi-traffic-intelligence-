import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('data/raw/nairobi_traffic_raw.csv', parse_dates=['timestamp'])
print(f"Loaded {len(df):,} records")

# ============================================
# 1. EXPLORATORY DATA ANALYSIS
# ============================================

print("\n" + "="*50)
print("1. EXPLORATORY DATA ANALYSIS")
print("="*50)

print("\n--- Numeric Summary ---")
print(df[['congestion_score', 'speed_kmh', 'delay_minutes', 'fuel_cost_kes']].describe())

# Weather impact
print("\n--- Weather Impact on Congestion ---")
weather_impact = df.groupby(['location', 'is_raining'])['congestion_score'].mean().unstack()
print(weather_impact)
print(f"\nRain increases congestion by: {(weather_impact[True] - weather_impact[False]).mean():.1%} on average")

# Create EDA visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Hourly heatmap (weekdays)
ax1 = axes[0, 0]
weekday_data = df[~df['is_weekend']]
pivot_hourly = weekday_data.groupby(['location', 'hour'])['congestion_score'].mean().unstack()
sns.heatmap(pivot_hourly, cmap='YlOrRd', ax=ax1, cbar_kws={'label': 'Congestion'})
ax1.set_title('Weekday Congestion by Hour & Location', fontweight='bold')

# Plot 2: Rain impact
ax2 = axes[0, 1]
rain_effect = df.groupby('location').apply(
    lambda x: x[x['is_raining']]['congestion_score'].mean() - x[~x['is_raining']]['congestion_score'].mean()
).sort_values(ascending=True)
colors = ['#d62728' if v > 0 else '#2ca02c' for v in rain_effect.values]
rain_effect.plot(kind='barh', ax=ax2, color=colors)
ax2.set_title('Rain Impact by Location', fontweight='bold')
ax2.set_xlabel('Congestion Increase')

# Plot 3: Speed vs Congestion
ax3 = axes[1, 0]
sample = df.sample(5000)
scatter = ax3.scatter(sample['congestion_score'], sample['speed_kmh'], 
                     c=sample['delay_minutes'], cmap='plasma', alpha=0.6, s=20)
ax3.set_xlabel('Congestion Score')
ax3.set_ylabel('Speed (km/h)')
ax3.set_title('Speed vs Congestion')
plt.colorbar(scatter, ax=ax3, label='Delay (min)')

# Plot 4: Monthly trend
ax4 = axes[1, 1]
monthly = df.groupby([df['timestamp'].dt.month, 'location'])['congestion_score'].mean().unstack()
monthly.plot(ax=ax4, marker='o')
ax4.set_title('Monthly Trends')
ax4.set_xlabel('Month')

plt.tight_layout()
plt.savefig('reports/heatmaps/eda_overview.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved EDA to reports/heatmaps/eda_overview.png")
plt.close()

# ============================================
# 2. XGBOOST FORECASTING
# ============================================

print("\n" + "="*50)
print("2. XGBOOST FORECASTING MODEL")
print("="*50)

def create_features(data_df):
    """Create time-series features for XGBoost"""
    df_feat = data_df.copy()
    df_feat['hour'] = df_feat['timestamp'].dt.hour
    df_feat['day_of_week'] = df_feat['timestamp'].dt.dayofweek
    df_feat['month'] = df_feat['timestamp'].dt.month
    df_feat['day_of_year'] = df_feat['timestamp'].dt.dayofyear
    
    # Cyclical encoding for hour
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)
    
    # Cyclical encoding for day of week
    df_feat['dow_sin'] = np.sin(2 * np.pi * df_feat['day_of_week'] / 7)
    df_feat['dow_cos'] = np.cos(2 * np.pi * df_feat['day_of_week'] / 7)
    
    # Lag features (previous hours)
    df_feat = df_feat.sort_values('timestamp')
    df_feat['lag_1h'] = df_feat['congestion_score'].shift(1)
    df_feat['lag_24h'] = df_feat['congestion_score'].shift(24)
    df_feat['lag_168h'] = df_feat['congestion_score'].shift(168)  # 1 week
    
    # Rolling averages
    df_feat['rolling_3h'] = df_feat['congestion_score'].shift(1).rolling(3).mean()
    df_feat['rolling_24h'] = df_feat['congestion_score'].shift(1).rolling(24).mean()
    
    return df_feat

def train_xgboost_forecaster(location_name, df_full):
    """Train XGBoost model for a specific location"""
    
    loc_df = df_full[df_full['location'] == location_name].copy()
    loc_df = create_features(loc_df)
    
    # Drop rows with NaN from lag features
    loc_df = loc_df.dropna()
    
    # Features
    feature_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month',
                  'is_raining', 'rainfall_mm', 'temperature_c',
                  'lag_1h', 'lag_24h', 'lag_168h', 'rolling_3h', 'rolling_24h']
    
    X = loc_df[feature_cols]
    y = loc_df['congestion_score']
    
    # Split: 80% train, 20% test (chronological)
    split_idx = int(len(loc_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    test_dates = loc_df['timestamp'].iloc[split_idx:]
    
    # Train XGBoost
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)],
              early_stopping_rounds=20,
              verbose=False)
    
    # Predict
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n{location_name}:")
    print(f"  MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(f"  Top features: {', '.join(importance.head(3)['feature'].values)}")
    
    # Forecast next 168 hours (7 days)
    last_row = loc_df.iloc[-1:].copy()
    future_predictions = []
    future_dates = []
    
    # Start from last known values
    current_lags = {
        'lag_1h': y.iloc[-1],
        'lag_24h': y.iloc[-24] if len(y) >= 24 else y.mean(),
        'lag_168h': y.iloc[-168] if len(y) >= 168 else y.mean(),
        'rolling_3h': y.iloc[-3:].mean(),
        'rolling_24h': y.iloc[-24:].mean()
    }
    
    from datetime import timedelta
    last_date = loc_df['timestamp'].max()
    
    for i in range(168):
        future_date = last_date + timedelta(hours=i+1)
        future_dates.append(future_date)
        
        # Create feature row
        hour = future_date.hour
        dow = future_date.weekday()
        month = future_date.month
        
        row = pd.DataFrame([{
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'dow_sin': np.sin(2 * np.pi * dow / 7),
            'dow_cos': np.cos(2 * np.pi * dow / 7),
            'month': month,
            'is_raining': 0,  # Assume dry for forecast
            'rainfall_mm': 0,
            'temperature_c': 22,
            'lag_1h': current_lags['lag_1h'],
            'lag_24h': current_lags['lag_24h'],
            'lag_168h': current_lags['lag_168h'],
            'rolling_3h': current_lags['rolling_3h'],
            'rolling_24h': current_lags['rolling_24h']
        }])
        
        pred = model.predict(row)[0]
        pred = np.clip(pred, 0.05, 1.0)
        future_predictions.append(pred)
        
        # Update lags for next iteration
        current_lags['lag_1h'] = pred
        if i >= 23:
            current_lags['lag_24h'] = future_predictions[i-23]
        if i >= 167:
            current_lags['lag_168h'] = future_predictions[i-167]
        current_lags['rolling_3h'] = np.mean(future_predictions[max(0, i-2):i+1])
        current_lags['rolling_24h'] = np.mean(future_predictions[max(0, i-23):i+1])
    
    return {
        'model': model,
        'metrics': {'mae': mae, 'rmse': rmse},
        'test_actual': y_test.values,
        'test_predicted': y_pred,
        'test_dates': test_dates.values,
        'future_dates': np.array(future_dates),
        'future_predictions': np.array(future_predictions),
        'feature_importance': importance
    }

# Train models for all locations
results = {}
for location in df['location'].unique():
    results[location] = train_xgboost_forecaster(location, df)

# ============================================
# 3. VISUALIZE FORECASTS
# ============================================

fig, axes = plt.subplots(4, 2, figsize=(16, 20))
axes = axes.flatten()

for idx, (location, result) in enumerate(results.items()):
    ax = axes[idx]
    
    # Plot test period
    ax.plot(result['test_dates'], result['test_actual'], label='Actual', alpha=0.7, color='black')
    ax.plot(result['test_dates'], result['test_predicted'], label='Predicted', alpha=0.7, color='red')
    
    # Plot future forecast
    ax.plot(result['future_dates'], result['future_predictions'], 
            label='7-Day Forecast', color='blue', linestyle='--', alpha=0.8)
    
    ax.set_title(f'{location}\nMAE: {result["metrics"]["mae"]:.3f}', 
                fontsize=10, fontweight='bold')
    ax.set_ylabel('Congestion Score')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('XGBoost Forecasting: Actual vs Predicted + 7-Day Forecast', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('reports/forecasts/xgboost_forecasts.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved forecasts to reports/forecasts/xgboost_forecasts.png")

# ============================================
# 4. SAVE MODELS & SUMMARY
# ============================================

# Save models
for location, result in results.items():
    model_path = f'models/xgboost_{location.lower()}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(result['model'], f)
    print(f"✅ Saved: {model_path}")

# Summary
summary = []
for location, result in results.items():
    summary.append({
        'location': location,
        'mae': result['metrics']['mae'],
        'rmse': result['metrics']['rmse'],
        'avg_congestion': df[df['location'] == location]['congestion_score'].mean()
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv('reports/forecast_model_summary.csv', index=False)
print(f"\n✅ Saved summary to reports/forecast_model_summary.csv")
print("\n--- Model Performance ---")
print(summary_df.to_string(index=False))

print("\n" + "="*50)
print("PHASE 1 COMPLETE")
print("="*50)