
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from datetime import datetime, timedelta

# Tham số thị trường
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
        return max(0, params['xp'] - params['yp'] * price + params['zp'] * competitor_price)
    return max(0, params['xa'] - params['ya'] * price + params['za'] * competitor_price)

def calculate_cost(quantity, params, is_pima=True):
    """Tính chi phí theo công thức Cpt và Cat"""
    if is_pima:
        return params['cp'] * quantity + params['fp']
    return params['ca'] * quantity + params['fa']

def find_optimal_price(competitor_price, params, is_pima=True, noise_factor=0.2):
    """Tìm giá tối ưu theo argmax với các điều chỉnh để phản ánh thực tế thị trường"""
    # Tính giá cơ bản theo công thức Nash
    if is_pima:
        base_price = (params['xp'] + params['zp'] * competitor_price + params['cp'] * params['yp']) / (2 * params['yp'])
    else:
        base_price = (params['xa'] + params['za'] * competitor_price + params['ca'] * params['ya']) / (2 * params['ya'])
    
    # 1. Thêm yếu tố phản ứng với giá đối thủ
    price_difference = 0.3 * (base_price - competitor_price)
    adjusted_price = base_price - price_difference
    
    # 2. Thêm yếu tố thị trường theo mùa
    seasonal_adjustment = np.sin(2 * np.pi * datetime.now().month / 12) * 0.5
    adjusted_price += seasonal_adjustment
    
    # 3. Thêm yếu tố chi phí vận hành
    if is_pima:
        operating_cost_factor = params['fp'] / (params['yp'] * 1000)  # Chuẩn hóa chi phí
        adjusted_price += operating_cost_factor
    else:
        operating_cost_factor = params['fa'] / (params['ya'] * 1000)
        adjusted_price += operating_cost_factor
    
    # 4. Điều chỉnh nhiễu ngẫu nhiên
    noise = np.random.normal(0, noise_factor * 0.5)  # Giảm mức độ nhiễu
    adjusted_price += noise
    
    # 5. Thêm giới hạn động
    min_price = max(35, competitor_price * 0.9)  # Không thấp hơn 90% giá đối thủ
    max_price = min(50, competitor_price * 1.1)  # Không cao hơn 110% giá đối thủ
    
    return max(min_price, min(max_price, adjusted_price))

def generate_competitive_data(n_days=30):
    """Tạo dữ liệu quá khứ với tính cạnh tranh cao"""
    dates = [datetime.now() - timedelta(days=x) for x in range(n_days, 0, -1)]
    
    # Khởi tạo giá ban đầu
    pima_price = 42
    amip_price = 40
    
    data = []
    
    for i in range(n_days):
        # Tính toán yếu tố mùa vụ
        seasonal_factor = np.sin(2 * np.pi * i / 30) * 0.3
        reaction_strength= np.random.uniform(0.3,0.7)
        competitive_factor= np.random.normal(0,2)
        # Cập nhật giá theo chiến lược cạnh tranh
    
        # PIMA điều chỉnh giá dựa trên giá AMIP
        pima_optimal = find_optimal_price(amip_price, PARAMS, True, 0.3)

        if pima_price > amip_price: 
            pima_price = pima_optimal + seasonal_factor- reaction_strength*(pima_price-amip_price)
        else: 
            pima_price = pima_optimal + competitive_factor
        
        # AMIP phản ứng với giá mới của PIMA
        amip_optimal = find_optimal_price(pima_price, PARAMS, False, 0.3)
        if amip_price > pima_price: 
            amip_price = amip_optimal + seasonal_factor- reaction_strength*(amip_price-pima_price)
        else: 
            amip_price = amip_optimal + competitive_factor
        
        pima_price = max(35, min(50, pima_price))
        amip_price = max(35, min(50, amip_price))
        # Tính các chỉ số kinh doanh
        pima_demand = calculate_demand(pima_price, amip_price, PARAMS, True)
        amip_demand = calculate_demand(amip_price, pima_price, PARAMS, False)
        
        pima_cost = calculate_cost(pima_demand, PARAMS, True)
        amip_cost = calculate_cost(amip_demand, PARAMS, False)
        
        pima_profit = pima_price * pima_demand - pima_cost
        amip_profit = amip_price * amip_demand - amip_cost
        
        # Thêm nhiễu thị trường
        market_noise = np.random.normal(0, 0.5)
        pima_price += market_noise
        amip_price += market_noise
        
        data.append({
            'date': dates[i],
            'pima_price': pima_price,
            'amip_price': amip_price,
            'pima_demand': pima_demand,
            'amip_demand': amip_demand,
            'pima_profit': pima_profit,
            'amip_profit': amip_profit,
            'seasonal_factor': seasonal_factor
        })
    
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
        hidden_layer_sizes=(128,64, 32),
        activation='relu',
        solver='adam',
        learning_rate= 'adaptive',
        max_iter=3000,
        random_state=42,
        early_stopping= True
    )
    
    model_amip = MLPRegressor(
        hidden_layer_sizes=(128,64, 32),
        activation='relu',
        solver='adam',
        max_iter=3000,
        learning_rate= 'adaptive',
        random_state=42,
        early_stopping= True
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
        
        #Thêm nhiễu ngẫu nhiên nhỏ để tạo biến động tự nhiên 
        noise = np.random.normal(0,0.2)
        pima_pred= max(35,min(50,pima_pred+noise))
        amip_pred = max(35, min(50, amip_pred+noise))
        
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
def plot_competitive_results(historical_data, future_predictions):
    """Vẽ biểu đồ với dữ liệu cạnh tranh"""
    plt.figure(figsize=(15, 10))
    
    # Plot dữ liệu quá khứ
    plt.plot(historical_data['date'], historical_data['pima_price'], 
             'b-', label='PiMA - Quá khứ', linewidth=2)
    plt.plot(historical_data['date'], historical_data['amip_price'], 
             'g-', label='AMIP - Quá khứ', linewidth=2)
    
    # Plot giá lý thuyết
    plt.plot(future_predictions['date'], future_predictions['pima_theoretical'], 
             'b:', label='PiMA - Lý thuyết')
    plt.plot(future_predictions['date'], future_predictions['amip_theoretical'], 
             'g:', label='AMIP - Lý thuyết')
    """#Plot giá thực tế 
    plt.plot(future_predictions['date'], future_predictions['pima_predicted'],
             'b--', label= 'PiMA- Thực tế')
    plt.plot(future_predictions['date'], future_predictions['amip_predicted'],
             'g--', label= 'AMIP- Thực tế')"""
    
    plt.title('Biến động giá bán sổ tay')
    plt.xlabel('Thời gian')
    plt.ylabel('Giá (nghìn đồng)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Thực thi mô hình
np.random.seed(42)
historical_data = generate_competitive_data(60)
future_predictions = train_price_predictor(historical_data, 180)  # Giả sử hàm này không thay đổi
plot_competitive_results(historical_data, future_predictions)