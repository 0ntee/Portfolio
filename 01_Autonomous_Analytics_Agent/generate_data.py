import os
import pandas as pd
import numpy as np

def create_synthetic_data():
    os.makedirs("data", exist_ok=True)
    
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", end="2026-03-01", freq="h")
    n_rows = len(dates)
    
    amounts = np.random.exponential(scale=500, size=n_rows) + 100
    
    df = pd.DataFrame({
        "timestamp": dates,
        "transaction_id": [f"TX_{i:06d}" for i in range(n_rows)],
        "amount": np.round(amounts, 2),
        "category": np.random.choice(["Супермаркеты", "Кафе", "Транспорт", "Аптеки", "АЗС"], size=n_rows)
    })
    
    friday_mask = df["timestamp"].dt.day_name() == "Friday"
    friday_indices = df[friday_mask].index
    
    anomaly_indices = np.random.choice(friday_indices, size=15, replace=False)
    df.loc[anomaly_indices, "amount"] = np.round(np.random.uniform(50000, 150000, size=15), 2)
    
    df.to_csv("data/ecommerce_transactions.csv", index=False)
    print("Файл data/ecommerce_transactions.csv успешно сгенерирован! Найдено пятничных аномалий:", len(anomaly_indices))

if __name__ == "__main__":
    create_synthetic_data()
