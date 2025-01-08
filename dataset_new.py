import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class BertrandParameters:
    def __init__(self, name, x, y, z, c, F):
        """
        Khởi tạo tham số cho mô hình Bertrand
        name: Tên cửa hàng
        x: Lượng cầu tối đa
        y: Hệ số cầu biên
        z: Hệ số cầu chéo
        c: Giá sản xuất mỗi đơn vị
        F: Chi phí vận hành
        """
        self.name = name
        self.x = x  # Lượng cầu tối đa
        self.y = y  # Hệ số cầu biên
        self.z = z  # Hệ số cầu chéo
        self.c = c  # Giá sản xuất
        self.F = F  # Chi phí vận hành

def calculate_demand(p, a, params):
    """Tính lượng cầu Q(t+1) theo công thức đã cho"""
    demand = params.x - params.y * p + params.z * a
    return max(0, demand)  # Đảm bảo lượng cầu không âm

def calculate_cost(Q, params):
    """Tính giá vốn C(t) theo công thức đã cho"""
    return params.c * Q + params.F

def calculate_optimal_price(competitor_price, params):
    """Tính giá tối ưu cho ngày tiếp theo"""
    return (params.x + params.z * competitor_price + params.c * params.y) / (2 * params.y)

def generate_dataset(days=60, tol=1e-6):
    # Khởi tạo tham số cho hai cửa hàng
    pima = BertrandParameters(
        name="PIMA",
        x=110,   # Lượng cầu tối đa
        y=2.0,     # Hệ số cầu biên
        z=0.8,    # Hệ số cầu chéo
        c=20,   # Giá sản xuất mỗi đơn vị
        F=1000  # Chi phí vận hành
    )

    amip = BertrandParameters(
        name="AMIP",
        x=100,   # Lượng cầu tối đa
        y=2.2,    # Hệ số cầu biên
        z=0.8,    # Hệ số cầu chéo
        c=18,   # Giá sản xuất mỗi đơn vị
        F=900  # Chi phí vận hành
    )

    # Khởi tạo giá ban đầu
    p_current = 50  # Giá PIMA
    a_current = 48  # Giá AMIP

    data = []
    start_date = datetime(2024, 1, 1)

    for day in range(days):
        current_date = start_date + timedelta(days=day)

        # Tính lượng cầu
        Q_pima = calculate_demand(p_current, a_current, pima)
        Q_amip = calculate_demand(a_current, p_current, amip)

        # Tính giá vốn
        C_pima = calculate_cost(Q_pima, pima)
        C_amip = calculate_cost(Q_amip, amip)

        # Tính lợi nhuận
        profit_pima = p_current * Q_pima - C_pima
        profit_amip = a_current * Q_amip - C_amip

        # Lưu dữ liệu
        data.append({
            'Date': current_date,
            'Day': day + 1,
            'PIMA_Price': round(p_current),
            'AMIP_Price': round(a_current),
            'PIMA_Demand': round(Q_pima),
            'AMIP_Demand': round(Q_amip),
            'PIMA_Cost': round(C_pima),
            'AMIP_Cost': round(C_amip),
            'PIMA_Profit': round(profit_pima),
            'AMIP_Profit': round(profit_amip)
        })

        # Tính giá mới cho ngày tiếp theo
        p_next = calculate_optimal_price(a_current, pima)
        a_next = calculate_optimal_price(p_current, amip)

        # Kiểm tra hàm hội tụ
        if abs(p_next - p_current) < tol and abs(a_next - a_current) < tol:
            print(f"Giá cả không đổi vào ngày: {day + 1}.")
            break

        # Cập nhật giá
        p_current = p_next
        a_current = a_next

    return pd.DataFrame(data)

def plot_price_trends(df):
    """Vẽ biểu đồ xu hướng giá và lợi nhuận"""
    plt.figure(figsize=(15, 10))

    # Subplot 1: Giá bán
    plt.subplot(2, 1, 1)
    plt.plot(df['Day'], df['PIMA_Price'], 'b-', label='PIMA')
    plt.plot(df['Day'], df['AMIP_Price'], 'r-', label='AMIP')
    plt.title('Xu hướng giá bán theo thời gian')
    plt.xlabel('Ngày')
    plt.ylabel('Giá (VNĐ)')
    plt.grid(True)
    plt.legend()

    # Subplot 2: Lợi nhuận
    plt.subplot(2, 1, 2)
    plt.plot(df['Day'], df['PIMA_Profit'], 'b-', label='PIMA')
    plt.plot(df['Day'], df['AMIP_Profit'], 'r-', label='AMIP')
    plt.title('Xu hướng lợi nhuận theo thời gian')
    plt.xlabel('Ngày')
    plt.ylabel('Lợi nhuận (VNĐ)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def equilibrium_analysis(df):
    """Tính giá bình ổn của hai cửa hàng"""
    last_row = df.iloc[-1]
    pima_equilibrium = last_row['PIMA_Price']
    amip_equilibrium = last_row['AMIP_Price']
    print(f"Giá ở PIMA: {pima_equilibrium:.2f} VND")
    print(f"Giá ở AMIP: {amip_equilibrium:.2f} VND")

def main():
    # Tạo dataset
    df = generate_dataset(1000)

    # Lưu vào file CSV
    df.to_csv('price_analysis.csv', index=False)
    print("Đã lưu dữ liệu vào file 'price_analysis.csv'")

    # In thống kê mô tảs
    print("\nThống kê về giá:")
    print(df[['PIMA_Price', 'AMIP_Price']].describe())

    print("\nThống kê về lợi nhuận:")
    print(df[['PIMA_Profit', 'AMIP_Profit']].describe())

    # Vẽ biểu đồ
    plot_price_trends(df)

    # Giá bình ổn
    equilibrium_analysis(df)

if __name__ == "__main__":
    main()