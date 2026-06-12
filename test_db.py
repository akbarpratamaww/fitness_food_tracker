from database import get_connection
import os
from dotenv import load_dotenv

load_dotenv()

try:
    conn = get_connection()
    db_type = os.getenv("DB_TYPE", "sqlite").lower()
    
    if db_type == "mysql":
        print("\n" + "="*50)
        print("[BERHASIL] Aplikasi Anda saat ini menggunakan Cloud MySQL!")
        print(f"Host: {os.getenv('DB_HOST')}")
        print("="*50 + "\n")
    else:
        print("\n" + "="*50)
        print("[PERHATIAN] Aplikasi Anda MASIH menggunakan SQLite lokal.")
        print("Cek kembali file .env Anda, pastikan DB_TYPE=mysql sudah disetel.")
        print("="*50 + "\n")
        
    conn.close()
except Exception as e:
    print(f"\n[GAGAL] Tidak dapat terhubung ke database. Error: {e}\n")
