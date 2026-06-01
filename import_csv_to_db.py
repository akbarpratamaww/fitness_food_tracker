import sqlite3
import pandas as pd
import os

db_path = "data/fitnessfoodtracker.db"
os.makedirs("data", exist_ok=True)

# Baca CSV (sesuaikan nama file dan path)
exercise_df = pd.read_csv("data/exercise_dataset.csv")
body_df = pd.read_csv("data/body_performance.csv")
food_df = pd.read_csv("data/food_dataset.csv")

conn = sqlite3.connect(db_path)
exercise_df.to_sql("exercise_dataset", conn, if_exists="replace", index=False)
body_df.to_sql("body_performance", conn, if_exists="replace", index=False)
food_df.to_sql("food_dataset", conn, if_exists="replace", index=False)
conn.close()
print("✅ Tabel berhasil diimpor.")