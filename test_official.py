import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from datetime import datetime, timedelta

# Tham số thị trường theo công thức Bertrand
PARAMS = {
    'xp': 100,  # Giữ nguyên
    'xa': 100,  # Giữ nguyên
    'yp': 2.2,  # Giữ nguyên
    'ya': 2.2,  # Giữ nguyên
    'zp': 0.8,  # Tăng từ 0.8 lên 1.8
    'za': 0.8,  # Tăng từ 0.8 lên 1.8
    'cp': 20,   # Giữ nguyên
    'ca': 20,   # Tăng từ 18 lên 19
    'fp': 800, # Giữ nguyên  
    'fa': 1000   # Tăng từ 900 lên 950
}
def calculate_theoretical_price(prev_pima, prev_amip, params):
    """Tính giá lý thuyết theo công thức Bertrand"""
    pima = (params['xp'] + params['zp'] * prev_amip + params['cp'] * params['yp']) / (2 * params['yp'])
    amip = (params['xa'] + params['za'] * prev_pima + params['ca'] * params['ya']) / (2 * params['ya'])
    return pima, amip

def calculate_demand(pima_price, amip_price, params):
    """Tính lượng cầu cho mỗi cửa hàng"""
    pima_demand = params['xp'] - params['yp'] * pima_price + params['zp'] * amip_price
    amip_demand = params['xa'] - params['ya'] * amip_price + params['za'] * pima_price
    return max(0, pima_demand), max(0, amip_demand)

def calculate_profit(price, demand, cost, fixed_cost):
    """Tính lợi nhuận"""
    return price * demand - cost * demand - fixed_cost

def generate_training_data(n_days=30):
    """Tạo dữ liệu training với nhiều features hơn"""
    dates = [datetime.now() - timedelta(days=x) for x in range(n_days, 0, -1)]
    
    # Khởi tạo giá ban đầu
    pima_prices = [50]
    amip_prices = [48]
    
    data = []
    
    np.random.seed(0)
    for i in range(n_days - 1):
        # Tính giá lý thuyết
        next_pima_theo, next_amip_theo = calculate_theoretical_price(pima_prices[-1], amip_prices[-1], PARAMS)
        
        # Thêm nhiễu để tạo biến động thị trường thực tế
        next_pima = next_pima_theo + np.random.normal(0, 0.5)
        next_amip = next_amip_theo + np.random.normal(0, 0.5)
        
        # Tính lượng cầu
        pima_demand, amip_demand = calculate_demand(next_pima, next_amip, PARAMS)
        
        # Tính lợi nhuận
        pima_profit = calculate_profit(next_pima, pima_demand, PARAMS['cp'], PARAMS['fp'])
        amip_profit = calculate_profit(next_amip, amip_demand, PARAMS['ca'], PARAMS['fa'])
        
        # Thêm yếu tố mùa vụ
        seasonal_factor = np.sin(2 * np.pi * i / 30) * 0.3
        
        data.append({
            'date': dates[i],
            'pima_price': next_pima,
            'amip_price': next_amip,
            'pima_demand': pima_demand,
            'amip_demand': amip_demand,
            'pima_profit': pima_profit,
            'amip_profit': amip_profit,
            'seasonal_factor': seasonal_factor,
            'pima_theoretical': next_pima_theo,
            'amip_theoretical': next_amip_theo
        })
        
        pima_prices.append(next_pima)
        amip_prices.append(next_amip)
    
    return pd.DataFrame(data)

def prepare_features(df, lookback=5):
    """Chuẩn bị features cho mô hình"""
    features = []
    targets_pima = []
    targets_amip = []
    
    for i in range(lookback, len(df)):
        # Features cho mỗi cửa hàng
        row_features = []
        
        # Thêm giá quá khứ
        for j in range(lookback):
            row_features.extend([
                df['pima_price'].iloc[i-j-1],
                df['amip_price'].iloc[i-j-1]
            ])
        
        # Thêm các features khác
        row_features.extend([
            df['pima_demand'].iloc[i-1],
            df['amip_demand'].iloc[i-1],
            df['pima_profit'].iloc[i-1],
            df['amip_profit'].iloc[i-1],
            df['seasonal_factor'].iloc[i-1]
        ])
        
        features.append(row_features)
        targets_pima.append(df['pima_price'].iloc[i])
        targets_amip.append(df['amip_price'].iloc[i])
    
    return np.array(features), np.array(targets_pima), np.array(targets_amip)

def train_and_predict(df, future_days=90):
    """Huấn luyện mô hình và dự đoán"""
    # Chuẩn bị dữ liệu
    X, y_pima, y_amip = prepare_features(df)
    
    # Chuẩn hóa dữ liệu
    scaler_X = StandardScaler()
    scaler_y_pima = StandardScaler()
    scaler_y_amip = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_pima_scaled = scaler_y_pima.fit_transform(y_pima.reshape(-1, 1))
    y_amip_scaled = scaler_y_amip.fit_transform(y_amip.reshape(-1, 1))
    
    # Xây dựng và huấn luyện mô hình
    model_pima = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        alpha=0.1,
        learning_rate_init=0.001,
        max_iter=2000,
        random_state=42
    )
    
    model_amip = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        alpha=0.1,
        learning_rate_init=0.001,
        max_iter=2000,
        random_state=42
    )
    
    model_pima.fit(X_scaled, y_pima_scaled.ravel())
    model_amip.fit(X_scaled, y_amip_scaled.ravel())
    
    # Dự đoán tương lai
    future_dates = [df['date'].iloc[-1] + timedelta(days=x) for x in range(1, future_days + 1)]
    future_pima = []
    future_amip = []
    future_pima_theo = []
    future_amip_theo = []
    
    # Lấy dữ liệu cuối cùng để dự đoán
    last_features = X[-1].copy()
    last_pima = df['pima_price'].iloc[-1]
    last_amip = df['amip_price'].iloc[-1]
    
    for i in range(future_days):
        # Dự đoán giá từ mô hình ML
        features_scaled = scaler_X.transform(last_features.reshape(1, -1))
        pima_pred_scaled = model_pima.predict(features_scaled)
        amip_pred_scaled = model_amip.predict(features_scaled)
        
        pima_pred = scaler_y_pima.inverse_transform(pima_pred_scaled.reshape(-1, 1))[0][0]
        amip_pred = scaler_y_amip.inverse_transform(amip_pred_scaled.reshape(-1, 1))[0][0]
        
        # Tính giá lý thuyết
        pima_theo, amip_theo = calculate_theoretical_price(last_pima, last_amip, PARAMS)
        #Lưu dự đoán
        future_pima.append(pima_pred)
        future_amip.append(amip_pred)
        future_pima_theo.append(pima_theo)
        future_amip_theo.append(amip_theo)
        
        # tinh cac gia trị phụ thuộc
        pima_demand, amip_demand = calculate_demand(pima_pred, amip_pred, PARAMS)
        pima_profit = calculate_profit(pima_pred, pima_demand, PARAMS['cp'], PARAMS['fp'])
        amip_profit = calculate_profit(amip_pred, amip_demand, PARAMS['ca'], PARAMS['fa'])
        seasonal_factor = np.sin(2 * np.pi * i / 30) * 0.3  # Thêm seasonal factor
        # Cập nhật last_features
        price_features = last_features[:10]
        price_features = np.roll(price_features,-2)  # Cập nhật giá
        price_features[-2:] = [pima_pred, amip_pred] 
        
        last_features= np.concatenate([
            price_features,
            [pima_demand, amip_demand, pima_profit, amip_profit, seasonal_factor]
        ])
        
        last_pima, last_amip = pima_pred, amip_pred
    return pd.DataFrame({
        'date': future_dates,
        'pima_predicted': future_pima,
        'amip_predicted': future_amip,
        'pima_theoretical': future_pima_theo,
        'amip_theoretical': future_amip_theo
    })

# Thực thi mô hình
df = generate_training_data()
future_predictions = train_and_predict(df)

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))

# Dữ liệu quá khứ
plt.plot(df['date'], df['pima_price'], 
         label='PiMA - Quá khứ', color='blue', marker='o', markersize=2)
plt.plot(df['date'], df['amip_price'], 
         label='AMIP - Quá khứ', color='green', marker='o', markersize=2)

"""# Dự đoán từ ML
plt.plot(future_predictions['date'], future_predictions['pima_predicted'], 
         label='PiMA - Dự đoán', color='blue', linestyle='--')
plt.plot(future_predictions['date'], future_predictions['amip_predicted'], 
         label='AMIP - Dự đoán', color='green', linestyle='--')"""

# Giá lý thuyết
plt.plot(future_predictions['date'], future_predictions['pima_theoretical'], 
         label='PiMA - Lý thuyết', color='purple', linestyle=':')
plt.plot(future_predictions['date'], future_predictions['amip_theoretical'], 
         label='AMIP - Lý thuyết', color='orange', linestyle=':')

plt.title('Biến động giá bán sổ tay')
plt.xlabel('Thời gian')
plt.ylabel('Giá (nghìn đồng)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# In kết quả so sánh
"""print("\nSo sánh giá dự đoán và lý thuyết (10 ngày đầu):")
comparison = future_predictions.head(10)[['date', 'pima_predicted', 'pima_theoretical', 
                                       'amip_predicted', 'amip_theoretical']]
print(comparison.to_string(index=False))"""