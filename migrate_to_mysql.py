import sqlite3
import mysql.connector
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Path to local SQLite database
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQLITE_DB_PATH = os.path.join(BASE_DIR, "data", "fitness_tracker.db")

def migrate_data():
    print("=" * 50)
    print("MIGRASI DATA: SQLite -> MySQL Cloud")
    print("=" * 50)

    # 1. Pastikan variabel MySQL sudah di-set di .env
    db_host = os.getenv("DB_HOST")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_name = os.getenv("DB_NAME", "fitness_tracker")

    if not db_host or not db_user:
        print("ERROR: Informasi MySQL (DB_HOST, DB_USER) tidak ditemukan di file .env!")
        print("Pastikan Anda sudah mengisi kredensial database Cloud Anda di .env")
        return

    print(f"Mengkoneksikan ke MySQL di {db_host}...")
    
    try:
        # Koneksi MySQL
        mysql_conn = mysql.connector.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name
        )
        mysql_cursor = mysql_conn.cursor()
        print("✅ Terhubung ke MySQL Cloud!")
        
        # Koneksi SQLite
        if not os.path.exists(SQLITE_DB_PATH):
            print(f"❌ File SQLite {SQLITE_DB_PATH} tidak ditemukan.")
            return
            
        sqlite_conn = sqlite3.connect(SQLITE_DB_PATH)
        sqlite_conn.row_factory = sqlite3.Row
        sqlite_cursor = sqlite_conn.cursor()
        print("✅ Terhubung ke SQLite Lokal!")

        # Daftar tabel yang akan dimigrasi
        tables = [
            "users", 
            "food_log", 
            "activity_log", 
            "weight_progress", 
            "chat_history"
        ]

        # Untuk menonaktifkan foreign key checks sementara agar insert tidak error karena urutan
        mysql_cursor.execute("SET FOREIGN_KEY_CHECKS=0;")

        for table in tables:
            print(f"\nMengambil data dari tabel: {table}...")
            
            # Ambil struktur kolom dari SQLite
            try:
                sqlite_cursor.execute(f"SELECT * FROM {table}")
                rows = sqlite_cursor.fetchall()
            except sqlite3.OperationalError:
                print(f"⚠️ Tabel {table} tidak ada di SQLite. Dilewati.")
                continue

            if not rows:
                print(f"Tabel {table} kosong. Dilewati.")
                continue

            # Ambil nama kolom
            columns = rows[0].keys()
            cols_str = ", ".join(columns)
            placeholders = ", ".join(["%s"] * len(columns))

            # Siapkan query insert MySQL
            insert_query = f"INSERT IGNORE INTO {table} ({cols_str}) VALUES ({placeholders})"

            # Konversi baris menjadi tuple
            data_to_insert = [tuple(row) for row in rows]

            print(f"Memigrasi {len(data_to_insert)} baris ke MySQL...")
            try:
                mysql_cursor.executemany(insert_query, data_to_insert)
                mysql_conn.commit()
                print(f"✅ Tabel {table} berhasil dimigrasi!")
            except Exception as e:
                print(f"❌ Gagal memigrasi tabel {table}: {e}")

        # Aktifkan kembali foreign key checks
        mysql_cursor.execute("SET FOREIGN_KEY_CHECKS=1;")
        print("\n🎉 MIGRASI SELESAI!")

    except Exception as e:
        print(f"\n❌ Terjadi kesalahan: {e}")
    finally:
        if 'mysql_cursor' in locals(): mysql_cursor.close()
        if 'mysql_conn' in locals() and mysql_conn.is_connected(): mysql_conn.close()
        if 'sqlite_cursor' in locals(): sqlite_cursor.close()
        if 'sqlite_conn' in locals(): sqlite_conn.close()

if __name__ == "__main__":
    # Karena kita butuh inisialisasi tabel di MySQL sebelum dipindah datanya
    try:
        from database import init_database
        print("Mempersiapkan tabel di database Cloud...")
        # Simulasikan environment database type ke mysql
        os.environ["DB_TYPE"] = "mysql"
        init_database()
    except Exception as e:
        print(f"Gagal inisialisasi tabel (abaikan jika tabel sudah ada): {e}")

    migrate_data()
