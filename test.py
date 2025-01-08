import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from datetime import datetime, timedelta

# Tham số thị trường theo mô hình Bertrand
PARAMS = {
    'xp': 110,  # Lượng cầu tối đa PiMA
    'xa': 100,  # Lượng cầu tối đa AMIP 
    'yp': 2.2,  # Hệ số cầu biên PiMA
    'ya': 2.0,  # Hệ số cầu biên AMIP
    'zp': 0.8,  # Hệ số cầu chéo PiMA
    'za': 0.8,  # Hệ số cầu chéo AMIP
    'cp': 18,   # Chi phí sản xuất PiMA
    'ca': 20,   # Chi phí sản xuất AMIP
    'fp': 800,  # Chi phí vận hành PiMA
    'fa': 1000  # Chi phí vận hành AMIP
}

def calculate_demand(price, competitor_price, params, is_pima=True):
    """Tính lượng cầu theo công thức Qpt+1 và Qat+1"""
    if is_pima:
        return params['xp'] - params['yp'] * price + params['zp'] * competitor_price
    return params['xa'] - params['ya'] * price + params['za'] * competitor_price

def calculate_cost(quantity, params, is_pima=True):
    """Tính chi phí theo công thức Cpt và Cat"""
    if is_pima:
        return params['cp'] * quantity + params['fp']
    return params['ca'] * quantity + params['fa']

def calculate_profit(price, quantity, cost):
    """Tính lợi nhuận theo công thức π(p,at) và π(a,pt)"""
    return price * quantity - cost

def find_optimal_price(competitor_price, params, is_pima=True):
    """Tìm giá tối ưu theo công thức argmax[π(p,at)] và argmax[π(a,pt)]"""
    if is_pima:
        return (params['xp'] + params['zp'] * competitor_price + params['cp'] * params['yp']) / (2 * params['yp'])
    return (params['xa'] + params['za'] * competitor_price + params['ca'] * params['ya']) / (2 * params['ya'])

def generate_training_data(n_days=30):
    """Tạo dữ liệu training với yếu tố mùa vụ và nhiễu ngẫu nhiên"""
    dates = [datetime.now() - timedelta(days=x) for x in range(n_days, 0, -1)]
    
    # Khởi tạo giá ban đầu
    pima_price = 42
    amip_price = 40
    
    data = []
    
    for i in range(n_days):
        # Thêm yếu tố mùa vụ và nhiễu
        seasonal_factor = np.sin(2 * np.pi * i / 30) * 0.3
        market_noise = np.random.normal(0, 0.2)
        
        # Tính giá tối ưu
        new_pima_price = find_optimal_price(amip_price, PARAMS, True) + seasonal_factor + market_noise
        new_amip_price = find_optimal_price(pima_price, PARAMS, False) + seasonal_factor + market_noise
        
        # Tính các chỉ số kinh doanh
        pima_demand = calculate_demand(new_pima_price, amip_price, PARAMS, True)
        amip_demand = calculate_demand(new_amip_price, pima_price, PARAMS, False)
        
        pima_cost = calculate_cost(pima_demand, PARAMS, True)
        amip_cost = calculate_cost(amip_demand, PARAMS, False)
        
        pima_profit = calculate_profit(new_pima_price, pima_demand, pima_cost)
        amip_profit = calculate_profit(new_amip_price, amip_demand, amip_cost)
        
        data.append({
            'date': dates[i],
            'pima_price': new_pima_price,
            'amip_price': new_amip_price,
            'pima_demand': pima_demand,
            'amip_demand': amip_demand,
            'pima_profit': pima_profit,
            'amip_profit': amip_profit,
            'seasonal_factor': seasonal_factor
        })
        
        pima_price, amip_price = new_pima_price, new_amip_price
    
    return pd.DataFrame(data)

def train_price_predictor(historical_data, future_days=90):
    """Huấn luyện mô hình dự đoán giá và tạo dự báo"""
    # Chuẩn bị features
    X = np.column_stack([
        historical_data['pima_price'],
        historical_data['amip_price'],
        historical_data['pima_demand'],
        historical_data['amip_demand'],
        historical_data['seasonal_factor']
    ])
    
    y_pima = historical_data['pima_price'].values
    y_amip = historical_data['amip_price'].values
    
    # Chuẩn hóa dữ liệu
    scaler_X = StandardScaler()
    scaler_y_pima = StandardScaler()
    scaler_y_amip = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_pima_scaled = scaler_y_pima.fit_transform(y_pima.reshape(-1, 1))
    y_amip_scaled = scaler_y_amip.fit_transform(y_amip.reshape(-1, 1))
    
    # Xây dựng và huấn luyện mô hình neural network
    model_pima = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=2000,
        random_state=42
    )
    
    model_amip = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=2000,
        random_state=42
    )
    
    model_pima.fit(X_scaled, y_pima_scaled.ravel())
    model_amip.fit(X_scaled, y_amip_scaled.ravel())
    
    # Dự đoán giá tương lai
    future_dates = [historical_data['date'].iloc[-1] + timedelta(days=x) for x in range(1, future_days + 1)]
    future_predictions = []
    
    last_prices = X[-1].copy()
    
    for i in range(future_days):
        # Thêm yếu tố mùa vụ
        seasonal_factor = np.sin(2 * np.pi * i / 30) * 0.3
        features = np.array([last_prices[0], last_prices[1], last_prices[2], last_prices[3], seasonal_factor]).reshape(1, -1)
        
        # Dự đoán giá
        pima_pred = scaler_y_pima.inverse_transform(
            model_pima.predict(scaler_X.transform(features)).reshape(-1, 1)
        )[0][0]
        
        amip_pred = scaler_y_amip.inverse_transform(
            model_amip.predict(scaler_X.transform(features)).reshape(-1, 1)
        )[0][0]
        
        # Tính giá lý thuyết
        pima_theo = find_optimal_price(last_prices[1], PARAMS, True)
        amip_theo = find_optimal_price(last_prices[0], PARAMS, False)
        
        future_predictions.append({
            'date': future_dates[i],
            'pima_predicted': pima_pred,
            'amip_predicted': amip_pred,
            'pima_theoretical': pima_theo,
            'amip_theoretical': amip_theo
        })
        
        # Cập nhật giá cho lần dự đoán tiếp theo
        last_prices = np.array([
            pima_pred, 
            amip_pred,
            calculate_demand(pima_pred, amip_pred, PARAMS, True),
            calculate_demand(amip_pred, pima_pred, PARAMS, False),
            seasonal_factor
        ])
    
    return pd.DataFrame(future_predictions)

def plot_results(historical_data, future_predictions):
    """Vẽ biểu đồ kết quả"""
    # Plot 1: Dữ liệu quá khứ
    plt.figure(figsize=(12, 5))
    plt.plot(historical_data['date'], historical_data['pima_price'], 'b-', label='PiMA - Quá khứ')
    plt.plot(historical_data['date'], historical_data['amip_price'], 'g-', label='AMIP - Quá khứ')
    plt.title('Biến động giá bán sổ tay - Dữ liệu quá khứ')
    plt.xlabel('Thời gian')
    plt.ylabel('Giá (nghìn đồng)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot 2: Dự đoán tương lai và giá lý thuyết
    plt.figure(figsize=(12, 5))
    plt.plot(future_predictions['date'], future_predictions['pima_theoretical'], 
             'r:', label='PiMA - Lý thuyết')
    plt.plot(future_predictions['date'], future_predictions['amip_theoretical'], 
             'orange:', label='AMIP - Lý thuyết')
    plt.plot(future_predictions['date'], future_predictions['pima_predicted'], 
             'b--', label='PiMA - Dự đoán')
    plt.plot(future_predictions['date'], future_predictions['amip_predicted'], 
             'g--', label='AMIP - Dự đoán')
    plt.title('Dự đoán giá bán sổ tay - So sánh với lý thuyết')
    plt.xlabel('Thời gian')
    plt.ylabel('Giá (nghìn đồng)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Thực thi mô hình
np.random.seed(42)
historical_data = generate_training_data(30)  # Tạo dữ liệu 30 ngày quá khứ
future_predictions = train_price_predictor(historical_data, 90)  # Dự đoán 90 ngày tương lai
plot_results(historical_data, future_predictions)