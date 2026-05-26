import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import *
import joblib as jl

# Đọc file dữ liệu wide (file bạn đã upload)
df = pd.read_csv('air_quality_worldwide_2025.csv')
# Chuyển cột Date sang dạng datetime
df['Date'] = pd.to_datetime(df['Date'])
print(df)

#Chọn các cột cần thiết
columns_to_keep = ['Date', 'Country', 'City', 'pm25', 'humidity', 'temperature', 'wind-speed', 'pressure']
# Lọc DataFrame
df1 = df[columns_to_keep]
# Kiểm tra kết quả
print(df1.head())

# Thống kê mô tả biến số
print("\n Thống kê mô tả biến số ")
print(df1[['pm25', 'humidity', 'temperature', 'wind-speed', 'pressure']].describe())

# Biểu đồ mối qua hệ giữa các biến độc lập với PM2.5
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Phân tích mối quan hệ giữa các biến độc lập và nồng độ PM2.5', fontsize=20)
country_means = df1.groupby('Country')['pm25'].mean()
df1['Country_Encoded'] = df1['Country'].map(country_means)

# Mã hoá City
city_means = df1.groupby('City')['pm25'].mean()
df1['City_Encoded'] = df1['City'].map(city_means)

# Danh sách các đặc trưng cần vẽ
features = ['humidity', 'temperature', 'wind-speed', 'pressure', 'Country_Encoded', 'City_Encoded']
titles = ['Độ ẩm vs PM2.5', 'Nhiệt độ vs PM2.5', 'Tốc độ gió vs PM2.5',
          'Áp suất vs PM2.5', 'Mã hoá Quốc gia vs PM2.5', 'Mã hoá Thành phố vs PM2.5']

# Lấy mẫu 2000 dòng để tránh chồng chéo dữ liệu, giúp dễ nhìn thấy xu hướng
df_sample = df1.sample(2000, random_state=42)

for i, col in enumerate(features):
    row = i // 3
    ax_col = i % 3
    sns.scatterplot(data=df_sample, x=col, y='pm25', ax=axes[row, ax_col], alpha=0.6, color='teal')
    axes[row, ax_col].set_title(titles[i], fontsize=14)
    axes[row, ax_col].set_xlabel(col)
    axes[row, ax_col].set_ylabel('PM2.5')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('scatter_plots_grid.png') # Lưu file để chèn vào báo cáo

#Kiểm tra khuyết
num_cols = df1.select_dtypes(include=[np.number]).columns
print("Số lượng giá trị thiếu trước xử lý:")
df1[num_cols].isna().sum()

# Xử lý khuyết
for c in num_cols:
    df1[c] = df1[c].fillna(df1[c].median())
print("\nSố lượng giá trị thiếu sau khi điền median:")
print(df1[num_cols].isnull().sum())

#Kiểm tra trùng lặp
print("\n KIỂM TRA TRÙNG LẶP ")
n_duplicates = df1.duplicated().sum()
print(f"Số hàng trùng lặp hoàn toàn: {n_duplicates}")

#Vẽ biểu đồ boxplot trước khi xử lý ngoại lai
num_cols = ['pm25', 'humidity', 'temperature', 'wind-speed', 'pressure']

# Thiết lập biểu đồ 1 hàng 5 cột
fig, axes = plt.subplots(1, 5, figsize=(22, 6))
fig.suptitle('Biểu đồ Boxplot của các biến số trước xử lý ngoại lai', fontsize=20)

colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']

for i, col in enumerate(num_cols):
    sns.boxplot(y=df[col], ax=axes[i], color=colors[i])
    axes[i].set_title(f'{col}', fontsize=14)
    axes[i].set_ylabel('')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('boxplot_raw_data.png')

#Kiểm tra ngoại lai
print(" Kiểm tra ngoại lai bằng IQR (1.5 * IQR) ")
def count_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return len(outliers), lower_bound, upper_bound
for col in num_cols:
    n_outliers, lower, upper = count_outliers_iqr(df1[col])
    print(f"{col}: {n_outliers}")

# Giới hạn riêng cho pm25 trước khi xử lý IQR
pm25_out_range = ((df1['pm25'] < 0) | (df1['pm25'] > 230)).sum()
df1['pm25'] = df1['pm25'].clip(0, 230)
print(f"pm25 - Trước xử lý (ngoài [0,230]): {pm25_out_range} outliers")

# Cập nhật lại danh sách biến số sau khi đã giới hạn pm25
num_cols = df1.select_dtypes(include=[np.number]).columns
# Loại pm25 khỏi vòng lặp IQR vì đã xử lý riêng theo domain knowledge
num_cols = num_cols.drop('pm25', errors='ignore')

# Xử lý ngoại lai các biến còn lại bằng IQR clipping
for col in num_cols:
    n_outliers, lower, upper = count_outliers_iqr(df1[col])
    df1[col] = df1[col].clip(lower=lower, upper=upper)
    print(f"{col} - Trước xử lý (IQR): {n_outliers} outliers")

# Kiểm tra lại outlier sau xử lý
print("\nSau xử lý ngoại lai:")
for col in num_cols:
    n_outliers, _, _ = count_outliers_iqr(df1[col])
    print(f"{col}: {n_outliers} outliers còn lại")

# Kiểm tra lại pm25 lần cuối
print(f"pm25: {((df1['pm25'] < 0) | (df1['pm25'] > 230)).sum()} outliers còn lại")

num_cols = ['pm25', 'humidity', 'temperature', 'wind-speed', 'pressure']

# Biểu đồ Boxplot sau khi xử lý ngoại lai
fig, axes = plt.subplots(1, 5, figsize=(22, 6))
fig.suptitle('Biểu đồ Boxplot của các biến số sau khi xử lý ngoại lai', fontsize=20)

# Màu sắc phân biệt cho từng biến
colors = ['#ff6666', '#3399ff', '#66ff66', '#ff9933', '#9999ff']

for i, col in enumerate(num_cols):
    sns.boxplot(y=df1[col], ax=axes[i], color=colors[i])
    axes[i].set_title(f'{col}', fontsize=14)
    axes[i].set_ylabel('')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('boxplot_after_outliers.png')

# Mã hóa biến phân loại và tạo dữ liệu cuối cùng cho mô hình
country_means = df1.groupby('Country')['pm25'].mean()
df1['Country_Encoded'] = df['Country'].map(country_means)

#Mã hóa biến phân loại
# Mã hoá City
city_means = df1.groupby('City')['pm25'].mean()
df1['City_Encoded'] = df1['City'].map(city_means)

# XỬ LÝ BIẾN THỜI GIAN (DATE)
# Chuyển Date thành các con số mà máy có thể hiểu (Tháng, Thứ)
df1['Month'] = df1['Date'].dt.month
df1['DayOfWeek'] = df1['Date'].dt.dayofweek

# CHỌN CÁC CỘT CUỐI CÙNG ĐỂ ĐƯA VÀO MÔ HÌNH HỒI QUY
# Loại bỏ các cột dạng chữ (Country, City) và cột Date gốc
features = ['Country_Encoded', 'City_Encoded', 'humidity', 'temperature',
            'wind-speed', 'pressure', 'Month', 'DayOfWeek']
target = 'pm25'

df_final = df1[features + [target]]

# Kiểm tra kết quả cuối cùng
print("\n--- Dữ liệu đã sẵn sàng cho Hồi quy tuyến tính ---")
print(df_final.head())

#Chuẩn hóa biến độc lập
features = ['Country_Encoded', 'City_Encoded', 'humidity', 'temperature',
            'wind-speed', 'pressure', 'Month', 'DayOfWeek']

scaler = StandardScaler()
# Thực hiện chuẩn hoá (biến đổi về mean=0, std=1)
X_scaled = scaler.fit_transform(df1[features])

# Tạo DataFrame kết quả
df_final = pd.DataFrame(X_scaled, columns=features)
df_final['pm25'] = df1['pm25'].values # Giữ nguyên biến mục tiêu không chuẩn hoá (tùy chọn)

# Kiểm tra kết quả
print("Cấu trúc dữ liệu sau khi chuẩn hoá:")
print(df_final.head())

# Thống kê mô tả các biến sau xử lý ngoại lai
print("\n THỐNG KÊ MÔ TẢ CÁC BIẾN")
print(df_final.describe())

# Vẽ Histogram phân phối các biến
hist_features = [
    'Country_Encoded', 'City_Encoded',
    'humidity', 'temperature', 'wind-speed', 'pressure',
    'Month', 'DayOfWeek', 'pm25'
]

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("Histogram phân phối các biến", fontsize=16)

for i, col in enumerate(hist_features):
    r = i // 3
    c = i % 3
    axes[r, c].hist(df_final[col], bins=40, alpha=0.7)
    axes[r, c].set_title(col)
    axes[r, c].set_xlabel(col)
    axes[r, c].set_ylabel("Tần suất")

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig("histogram_after_preprocessing.png", dpi=300, bbox_inches='tight')
plt.show()

#Phân tích tương quan biến độc lập và biến mục tiêu
corr_matrix = df_final[num_cols].corr()
corr_matrix

#vẽ heatmap
num_cols = df_final.select_dtypes(include=[np.number]).columns
corr_matrix = df_final[num_cols].corr()

plt.figure(figsize=(10, 8))
plt.title("Correlation Heatmap (Mức độ ảnh hưởng giữa các biến)")

# 🌞 colormap sáng
plt.imshow(corr_matrix, cmap='coolwarm')
plt.colorbar()

plt.xticks(range(len(num_cols)), num_cols, rotation=90)
plt.yticks(range(len(num_cols)), num_cols)

for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        value = corr_matrix.iloc[i, j]
        plt.text(j, i, f"{value:.2f}", ha='center', va='center', fontsize=10)

plt.tight_layout()
plt.show()

#Vẽ biểu đồ phân tích tương quan PM2.5 và các biến độc lập
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Phân tích mối quan hệ giữa các biến độc lập và nồng độ PM2.5', fontsize=20)

# Danh sách các đặc trưng cần vẽ
features = ['humidity', 'temperature', 'wind-speed', 'pressure', 'Country_Encoded', 'City_Encoded']
titles = ['Độ ẩm vs PM2.5', 'Nhiệt độ vs PM2.5', 'Tốc độ gió vs PM2.5',
          'Áp suất vs PM2.5', 'Mã hoá Quốc gia vs PM2.5', 'Mã hoá Thành phố vs PM2.5']

# Lấy mẫu 2000 dòng để tránh chồng chéo dữ liệu, giúp dễ nhìn thấy xu hướng
df_sample = df_final.sample(2000, random_state=42)

for i, col in enumerate(features):
    row = i // 3
    ax_col = i % 3
    sns.scatterplot(data=df_sample, x=col, y='pm25', ax=axes[row, ax_col], alpha=0.6, color='teal')
    axes[row, ax_col].set_title(titles[i], fontsize=14)
    axes[row, ax_col].set_xlabel(col)
    axes[row, ax_col].set_ylabel('PM2.5')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('scatter_plots_grid.png') # Lưu file để chèn vào báo cáo

#vẽ bảng VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
X = df1[['humidity', 'temperature', 'wind-speed', 'pressure', 'Country_Encoded', 'City_Encoded', 'Month', 'DayOfWeek']].copy()
X['intercept'] = 1  # thêm hệ số chặn cho VIF

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print("\nChỉ số VIF (đánh giá đa cộng tuyến):")
print(vif_data)

# lưu bảng VIF thành hình nếu cần chèn báo cáo
plt.figure(figsize=(6,3))
plt.table(cellText=vif_data.values, colLabels=vif_data.columns, loc='center')
plt.axis('off')
plt.tight_layout()
plt.savefig("vif_table.png", dpi=300, bbox_inches='tight')
plt.show()

# Biểu đồ phân phối PM2.5
plt.figure(figsize=(10, 6))
sns.histplot(df_final['pm25'], kde=True, color='skyblue')
plt.title('Phân phối của nồng độ PM2.5')
plt.xlabel('PM2.5')
plt.ylabel('Tần suất')
plt.savefig('pm25_distribution.png')

#Tạo biến độc lâp và biến mục tiêu
X = df_final.drop(columns='pm25')
y = df_final['pm25']
#Log transform biến mục tiêu
y_log = np.log1p(y)

# Xây dựng , huấn luyện và dự báo
all_preds = []
for seed in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.15, random_state=seed)
    # Khởi tạo mô hình
    model = LinearRegression()
    # Huấn luyện
    model.fit(X_train, y_train)
    # Dự báo
    y_pred = model.predict(X_test)

    all_preds.append(y_pred)
print("Kết quả dự báo trung bình :", np.mean(all_preds, axis=0))

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# chuyển đổi kết quả dự báo
y_pred_real = np.expm1(y_pred)
y_test_real = np.expm1(y_test)
y_train_real = np.expm1(y_train)

# Tính sai số dữ liệu trên tập train
residuals = y_train - model.predict(X_train)
# Tính Smearing Factor
smearing = np.mean(np.exp(residuals))
# Tính kết quả dự bao thực tế
y_pred_pm25_corrected = smearing * y_pred_real

# Hiển thị kết quả dự báo
df_pred = pd.DataFrame({
    "STT": range(1, len(y_pred_pm25_corrected) + 1),
    "Giá trị dự báo": y_pred_pm25_corrected
})

print(df_pred.head(10))

# Tính chỉ số hồi quy của mô hình
print("Chỉ số coefficents của mô hình : ", model.coef_)
print("Chỉ số intercept của mô hình : ", model.intercept_)

#Phương pháp K-fold tính mean các chỉ số
# Tính chỉ số
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Chạy cross-validation
mae= -cross_val_score(model, X_train, y_train_real,cv=kf,scoring="neg_mean_absolute_error")

mse = -cross_val_score(model,X_train, y_train_real,cv=kf,scoring="neg_mean_squared_error")

r2 = cross_val_score(model, X_train, y_train_real,cv=kf,scoring="r2")

# Tính RMSE
rmse = np.sqrt(mse)

results = {
    "R2": r2,
    "RMSE": rmse,
    "MSE" : mse,
    "MAE": mae
}

df_results = pd.DataFrame(results)

# In kết quả
df_results.loc["Average"] = [
    r2.mean(),
    rmse.mean(),
    mse.mean(),
    mae.mean()
]
print("Kết quả trung bình chỉ số")
df_results

# Hàm quy đổi pm25 qua AQI
def pm25_to_aqi(pm25):
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500)
    ]

    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= pm25 <= bp_hi:
            return ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (pm25 - bp_lo) + aqi_lo

    return None

# Kết quả dự báo AQI
aqi_pred = np.array([pm25_to_aqi(x) for x in y_pred_pm25_corrected])
aqi_pred

# Tính các chỉ số
mae= mean_absolute_error(y_test_real,y_pred_pm25_corrected)

mse = mean_squared_error(y_test_real,y_pred_pm25_corrected)

r2 = r2_score(y_test_real,y_pred_pm25_corrected)

# Tính RMSE
rmse = np.sqrt(mse)

results_1 = {
    "R2": r2,
    "RMSE": rmse,
    "MSE" : mse,
    "MAE": mae
}
df_results_1 = pd.DataFrame(
    list(results_1.items()),
    columns=["Chỉ số đánh giá", "Giá trị"]
)
# In kết quả
print("Kết quả chỉ số")
df_results_1

# Tạo đường dự báo hoàn hảo y = x
min_val = min(y_test_real.min(), y_pred_pm25_corrected.min())
max_val = max(y_test_real.max(), y_pred_pm25_corrected.max())
perfect_line = np.linspace(min_val, max_val, 100)

plt.figure(figsize=(10, 6))

# Scatter plot: giá trị thực tế vs dự báo
plt.scatter(
    y_test_real,
    y_pred_pm25_corrected,
    alpha=0.6,
    edgecolors='white',
    label='Data Points'
)

# Đường dự báo hoàn hảo
plt.plot(
    perfect_line,
    perfect_line,
    linestyle='--',
    label='Perfect Prediction Line'
)

# Nhãn và tiêu đề
plt.xlabel('Giá trị Thực tế')
plt.ylabel('Giá trị Dự đoán')
plt.title('So sánh Giá trị Thực tế và Dự đoán')

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Lưu mô hình
joblib.dump(model, "linear_regression_model.pkl")