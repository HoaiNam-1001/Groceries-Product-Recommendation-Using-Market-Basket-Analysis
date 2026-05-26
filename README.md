# 🌫️ Air Quality Analysis \& PM2.5 Forecasting

## Overview

Linear Regression model trained on 385,182 air quality records from 90 countries and 579 cities worldwide (2024–2025), forecasting PM2.5 concentration with R² ≈ 0.54 and RMSE ≈ 23.33 µg/m³. Predictions are converted to AQI using EPA standard breakpoints.

## Tech Stack

Python | Scikit-learn | Pandas | NumPy | Matplotlib | Seaborn | Statsmodels | Joblib

Dataset
📥 **Tải tại:** [air\_quality\_worldwide\_2025.csv](https://drive.google.com/file/d/1sm--adJFJDP3XM3WINjXwbJHlqGN0vC9/view?usp=drive_link)

Sau khi tải, đặt file vào thư mục gốc của project.

## Results

|Chỉ số|Giá trị|
|-|-|
|R²|\~0.54|
|RMSE|\~23.33 µg/m³|

## How to Run

```
pip install -r requirements.txt
python KhaiPha2.py
```

