import sqlite3
import pandas as pd

# Path database
DB_PATH = "data/fitnessfoodtracker.db"

# Baca file CSV yang sudah kamu rename jadi bodyPerformance.csv
df = pd.read_csv("data/bodyPerformance.csv")

# Koneksi ke database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Hapus tabel jika sudah ada (opsional)
cursor.execute("DROP TABLE IF EXISTS body_performance")
print("✅ Tabel lama dihapus (jika ada)")

# Simpan dataframe ke SQLite sebagai tabel body_performance
df.to_sql("body_performance", conn, if_exists="replace", index=False)
print(f"✅ Tabel 'body_performance' dibuat dengan {len(df)} baris")

# Lihat kolom-kolom yang ada
print("📋 Kolom yang tersedia:", df.columns.tolist())

conn.commit()
conn.close()
print("✅ Selesai!")