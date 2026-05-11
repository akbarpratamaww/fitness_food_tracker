import sqlite3
import os

# Ganti dengan path absolut atau relatif yang benar
db_path = "data/fitnessfoodtracker.db"  # pastikan file ini ada

if not os.path.exists(db_path):
    print(f"❌ File tidak ditemukan: {db_path}")
    # Coba cari file .db lain di folder data
    print("Mencari file .db lain di folder data...")
    for f in os.listdir("data"):
        if f.endswith(".db"):
            print(f"   - {f}")
else:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"✅ Tabel dalam {db_path}:")
    if tables:
        for t in tables:
            print(f"   - {t[0]}")
    else:
        print("   (Tidak ada tabel)")
    conn.close()