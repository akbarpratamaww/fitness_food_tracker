import sqlite3
import pandas as pd
import re
import os

csv_path = "data/Food and Calories - Sheet1.csv"
db_path = "data/fitnessfoodtracker.db"

if not os.path.exists(csv_path):
    print(f"❌ File {csv_path} tidak ditemukan!")
    exit(1)

# Read the CSV
df = pd.read_csv(csv_path)

processed_rows = []

for idx, row in df.iterrows():
    food_name = str(row['Food']).strip()
    serving = str(row['Serving']).strip()
    calories_str = str(row['Calories']).strip()
    
    # 1. Parse calories
    cal_match = re.search(r'(\d+(?:\.\d+)?)', calories_str)
    if not cal_match:
        continue
    calories = float(cal_match.group(1))
    
    # 2. Parse weight in grams (g) or ml
    # Look for (X g) or (X ml) or (X grams)
    weight_match = re.search(r'\((\d+(?:\.\d+)?)\s*(?:g|ml|grams)\)', serving)
    if not weight_match:
        # Fallback: look for just X g or X ml in the serving text
        weight_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:g|ml|grams)', serving)
        
    if weight_match:
        weight = float(weight_match.group(1))
    else:
        # Default fallback weight is 100g if we can't find it
        weight = 100.0
        
    # Prevent division by zero
    if weight == 0:
        weight = 100.0
        
    # Calculate calories per 100g
    calories_per_100g = (calories / weight) * 100.0
    
    # We set default macros to 0 since the CSV doesn't provide them
    processed_rows.append({
        'Food': food_name,
        'Calories_per_100g': round(calories_per_100g, 1),
        'Protein_g': 0.0,
        'Carbs_g': 0.0,
        'Fat_g': 0.0
    })

processed_df = pd.DataFrame(processed_rows)

# Connect to database and replace the table
os.makedirs("data", exist_ok=True)
conn = sqlite3.connect(db_path)
processed_df.to_sql("food_dataset", conn, if_exists="replace", index=False)
conn.close()

# Also save to data/food_dataset.csv as backup / source
processed_df.to_csv("data/food_dataset.csv", index=False)

print(f"✅ Berhasil memproses {len(processed_df)} baris makanan dan menyimpannya ke tabel 'food_dataset' di {db_path}!")
