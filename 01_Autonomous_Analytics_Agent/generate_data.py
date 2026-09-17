import os
import pandas as pd
import numpy as np

def create_synthetic_data():
    os.makedirs("data", exist_ok=True)
    
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", end="2026-03-01", freq="h")
    n_rows = len(dates)
    
    purchase_amounts = np.random.exponential(scale=500, size=n_rows) + 100
    ages = np.random.randint(18, 75, size=n_rows)
    
    df = pd.DataFrame({
        "timestamp": dates,
        "transaction_id": np.char.add("TX_", np.char.zfill(np.arange(n_rows).astype(str), 6)),
        "Age": ages,
        "Purchase_Amount": np.round(purchase_amounts, 2),
        "category": np.random.choice(["Супермаркеты", "Кафе", "Транспорт", "Аптеки", "АЗС"], size=n_rows)
    })
    
    output_path = "data/ecommerce_transactions.csv"
    df.to_csv(output_path, index=False)
    print(f"Файл {output_path} успешно сгенерирован для анализа агентом!")

if __name__ == "__main__":
    create_synthetic_data()
