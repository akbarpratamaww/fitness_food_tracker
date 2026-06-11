import sqlite3
import pandas as pd
import subprocess
import re
import os

# 1. Read the original dataset from Git
try:
    git_data = subprocess.check_output(["git", "show", "HEAD:data/food_dataset.csv"])
    original_text = git_data.decode("utf-8", errors="ignore")
    with open("scratch/clean_original.csv", "w", encoding="utf-8") as f:
        f.write(original_text)
    original_df = pd.read_csv("scratch/clean_original.csv")
    print(f"Original dataset loaded: {len(original_df)} rows")
    
    # Rename columns to match what the app expects
    original_df = original_df.rename(columns={
        'Food Name': 'Food',
        'Calories': 'Calories_per_100g',
        'Protein': 'Protein_g',
        'Carbs': 'Carbs_g',
        'Fats': 'Fat_g'
    })
    
    # Keep only the columns we need
    original_df = original_df[['Food', 'Calories_per_100g', 'Protein_g', 'Carbs_g', 'Fat_g']]
    print("Mapped original columns successfully.")
except Exception as e:
    print(f"Failed to load original dataset from Git: {e}")
    original_df = pd.DataFrame(columns=['Food', 'Calories_per_100g', 'Protein_g', 'Carbs_g', 'Fat_g'])

# 2. Read and parse the new custom dataset
csv_path = "data/Food and Calories - Sheet1.csv"
if os.path.exists(csv_path):
    df_new = pd.read_csv(csv_path)
    new_rows = []
    for idx, row in df_new.iterrows():
        food_name = str(row['Food']).strip()
        serving = str(row['Serving']).strip()
        calories_str = str(row['Calories']).strip()
        
        cal_match = re.search(r'(\d+(?:\.\d+)?)', calories_str)
        if not cal_match:
            continue
        calories = float(cal_match.group(1))
        
        weight_match = re.search(r'\((\d+(?:\.\d+)?)\s*(?:g|ml|grams)\)', serving)
        if not weight_match:
            weight_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:g|ml|grams)', serving)
            
        weight = float(weight_match.group(1)) if weight_match else 100.0
        if weight == 0:
            weight = 100.0
            
        calories_per_100g = (calories / weight) * 100.0
        
        new_rows.append({
            'Food': food_name,
            'Calories_per_100g': round(calories_per_100g, 1),
            'Protein_g': 0.0,
            'Carbs_g': 0.0,
            'Fat_g': 0.0
        })
    new_df = pd.DataFrame(new_rows)
    print(f"New custom dataset parsed: {len(new_df)} rows")
else:
    new_df = pd.DataFrame(columns=['Food', 'Calories_per_100g', 'Protein_g', 'Carbs_g', 'Fat_g'])

# 3. Merge the datasets
# Place original_df first so that duplicate rows (by lowercase name) keep the original item (which has protein/carbs/fat macros)
combined_df = pd.concat([original_df, new_df], ignore_index=True)

# Clean food names for deduplication
combined_df['Food_clean'] = combined_df['Food'].str.strip().str.lower()

# Drop duplicates, keeping the first (original item with macro data)
combined_df = combined_df.drop_duplicates(subset=['Food_clean'], keep='first')

# Drop the temp clean column
combined_df = combined_df.drop(columns=['Food_clean'])

print(f"Combined dataset: {len(combined_df)} rows")

# 4. Save to CSV and SQLite
os.makedirs("data", exist_ok=True)
combined_df.to_csv("data/food_dataset.csv", index=False)

db_path = "data/fitnessfoodtracker.db"
conn = sqlite3.connect(db_path)
combined_df.to_sql("food_dataset", conn, if_exists="replace", index=False)
conn.close()

print("Merged database successfully written.")
