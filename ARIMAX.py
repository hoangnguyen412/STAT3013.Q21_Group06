import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# 1. Define city files and exogenous variables
city_files = {
    'Delhi': 'delhi_project_final_data.csv',
    'London': 'london_project_final_data.csv',
    'Mexico City': 'mexico_project_final_data.csv'
}
exog_cols = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']

# Create subplots for final predictions
fig, axes = plt.subplots(3, 1, figsize=(12, 18)) 

for i, (city, file_path) in enumerate(city_files.items()):
    print(f"\n{'='*70}")
    print(f"STARTING COMPREHENSIVE ANALYSIS FOR: {city.upper()} ")
    
    # 2. Load and Prepare Data
    df_city = pd.read_csv(file_path)
    df_city['date'] = pd.to_datetime(df_city['date'], format='mixed')
    df_city = df_city.sort_values('date').set_index('date')
    df_city = df_city.dropna(subset=['hospital_admissions'] + exog_cols)
    
    y = df_city['hospital_admissions']
    X = df_city[exog_cols].astype(float)
    
    # ---> SỬA ĐỔI: Chuẩn hóa dữ liệu ngoại sinh (Scale Data)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    # 3. ADF Test for Stationarity
    adf_result = adfuller(y)
    print(f"[ADF Test] Statistic: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}")
    
    # 4. Fit ARIMAX Model
    print(f"Finding optimal ARIMAX order for {city}...")
    try:
        # ---> SỬA ĐỔI: Bật seasonal, m=7 (chu kỳ tuần), tắt stepwise để tìm kiếm chi tiết
        model = pm.auto_arima(y, X=X_scaled, 
                              seasonal=True, m=7, 
                              stepwise=False, max_p=3, max_q=3,
                              trace=False)
    except TypeError:
        model = pm.auto_arima(y, exogenous=X_scaled, 
                              seasonal=True, m=7, 
                              stepwise=False, max_p=3, max_q=3,
                              trace=False)
    
    print(f"Best Model: ARIMAX{model.order} (Seasonal: {model.seasonal_order})")
    print(f"AIC Score: {model.aic():.2f}")
    
    # 5. Model Diagnostics (Health Check)
    print(f"Generating diagnostic plots for {city}...")
    model.plot_diagnostics(figsize=(10, 8))
    plt.suptitle(f"Diagnostics: {city}", fontsize=16)
    plt.savefig(f'diagnostics_{city.lower().replace(" ", "_")}.png')
    plt.close() # Đóng plot diagnostic để không dính vào plot tổng ở cuối
    
    # 6. Calculate Accuracy Metrics 
    try:
        # ---> SỬA ĐỔI: Chạy phân tích trên dữ liệu X đã chuẩn hóa
        fitted_values = model.predict_in_sample(X=X_scaled)
    except TypeError:
        fitted_values = model.predict_in_sample(exogenous=X_scaled)
    
    rmse = np.sqrt(mean_squared_error(y, fitted_values))
    mae = mean_absolute_error(y, fitted_values)
    r2 = r2_score(y, fitted_values)
    mape = mean_absolute_percentage_error(y, fitted_values) * 100 
    
    print(f"Accuracy Metrics for {city}:")
    print(f"   - RMSE: {rmse:.2f} (Avg. error in cases)")
    print(f"   - MAE:  {mae:.2f} (Median error in cases)")
    print(f"   - MAPE: {mape:.2f}% (Percentage error)")
    print(f"   - R²:   {r2:.2f} (Explanatory power)")

    # 7. Final Comparison Plot
    ax = axes[i]
    ax.plot(y.index[-100:], y[-100:], label='Actual Admissions', color='#1f77b4', alpha=0.7, linewidth=2)
    ax.plot(y.index[-100:], fitted_values[-100:], label='ARIMAX Fit', color='#d62728', linestyle='--', linewidth=2)
    # ---> SỬA ĐỔI: Đưa R2 lên tiêu đề để dễ dàng phân tích
    ax.set_title(f'{city} (Order: {model.order}, R²: {r2:.2f}, MAPE: {mape:.2f}%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Admissions')
    ax.legend()
    ax.grid(True, alpha=0.3)

# 8. Save and Show Results
plt.tight_layout()
fig.subplots_adjust(hspace=0.4) 
plt.savefig('arimax_full_report_plot.png', dpi=300)
print(f"\n{'='*70}")
print("ALL CITIES PROCESSED SUCCESSFULLY!")
print("Files saved: 'arimax_full_report_plot.png' and individual diagnostics plots.")
print(f"{'='*70}")
plt.show()