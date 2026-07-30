import os
import pandas as pd

# -------------------------------
# Project Paths
# -------------------------------

RAW_DATA_PATH = os.path.join("data", "raw")

# -------------------------------
# Dataset Files
# -------------------------------

dataset_files = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

# -------------------------------
# Load Datasets
# -------------------------------

datasets = {}

for name, file_name in dataset_files.items():
    file_path = os.path.join(RAW_DATA_PATH, file_name)

    try:
        datasets[name] = pd.read_csv(file_path)
        print(f"✅ Loaded {name}")

    except FileNotFoundError:
        print(f"❌ File not found: {file_name}")

# -------------------------------
# Dataset Summary
# -------------------------------

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

for name, df in datasets.items():
    print(f"\n{name.upper()}")
    print("-" * 40)
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")