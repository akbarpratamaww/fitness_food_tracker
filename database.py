import sqlite3
import pandas as pd
from datetime import datetime, date
import hashlib

# Path database utama (untuk data transaksional)
DB_PATH = "data/fitness_tracker.db"

# Path database untuk dataset (bisa sama atau berbeda. Sesuaikan dengan file .db Anda)
DATASET_DB_PATH = "data/fitnessfoodtracker.db"   # ganti dengan nama file database Anda

def init_database():
    """Initialize all application tables (users, food_log, activity_log, weight_progress, chat_history)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            gender TEXT,
            height_cm REAL,
            weight_kg REAL,
            activity_level TEXT,
            fitness_goal TEXT,
            bmr REAL,
            tdee REAL,
            daily_target_calories REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Food log table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS food_log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            food_name TEXT,
            calories REAL,
            protein REAL,
            carbs REAL,
            fat REAL,
            meal_type TEXT,
            log_date DATE,
            logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    # Activity log table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS activity_log (
            activity_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            activity_type TEXT,
            duration_minutes REAL,
            calories_burned REAL,
            intensity TEXT,
            log_date DATE,
            logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    # Weight progress table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weight_progress (
            progress_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            weight_kg REAL,
            record_date DATE DEFAULT CURRENT_DATE,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    # Chat history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            chat_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            user_message TEXT,
            bot_response TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    conn.commit()
    conn.close()

def read_table_to_df(table_name, db_path=None):
    """
    Membaca seluruh isi tabel dari database SQLite ke pandas DataFrame.
    Berguna untuk membaca dataset seperti exercise_dataset, body_performance, food_dataset.
    
    Parameters:
    - table_name: nama tabel (contoh: 'exercise_dataset', 'body_performance', 'food_dataset')
    - db_path: path database, jika None menggunakan DATASET_DB_PATH
    """
    if db_path is None:
        db_path = DATASET_DB_PATH
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        return df
    except Exception as e:
        print(f"Error membaca tabel {table_name}: {e}")
        return None
    finally:
        conn.close()
def table_exists(table_name, db_path=None):
    """
    Mengecek apakah suatu tabel ada di database SQLite.
    
    Parameters:
    - table_name: nama tabel yang ingin dicek
    - db_path: path database, jika None menggunakan DATASET_DB_PATH
    
    Returns:
    - True jika tabel ditemukan, False jika tidak
    """
    if db_path is None:
        db_path = DATASET_DB_PATH
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def get_table_columns(table_name, db_path=None):
    """Mendapatkan daftar kolom dari suatu tabel (utilitas)."""
    if db_path is None:
        db_path = DATASET_DB_PATH
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    conn.close()
    return columns

# ==================== FUNGSI-FUNGSI ASLI (TIDAK BERUBAH) ====================
def get_user(user_id=None):
    """Get user data. If user_id is None, get the most recent user."""
    conn = sqlite3.connect(DB_PATH)
    if user_id:
        df = pd.read_sql_query("SELECT * FROM users WHERE user_id = ?", conn, params=(user_id,))
    else:
        df = pd.read_sql_query("SELECT * FROM users ORDER BY user_id DESC LIMIT 1", conn)
    conn.close()
    return df.iloc[0] if len(df) > 0 else None

def save_user(user_data):
    """Save or update user data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if 'user_id' in user_data and user_data['user_id']:
        cursor.execute('''
            UPDATE users SET name=?, age=?, gender=?, height_cm=?, weight_kg=?, 
            activity_level=?, fitness_goal=?, bmr=?, tdee=?, daily_target_calories=?
            WHERE user_id=?
        ''', (
            user_data['name'], user_data['age'], user_data['gender'],
            user_data['height_cm'], user_data['weight_kg'], user_data['activity_level'],
            user_data['fitness_goal'], user_data['bmr'], user_data['tdee'],
            user_data['daily_target_calories'], user_data['user_id']
        ))
        user_id = user_data['user_id']
    else:
        cursor.execute('''
            INSERT INTO users (name, age, gender, height_cm, weight_kg, activity_level,
            fitness_goal, bmr, tdee, daily_target_calories)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_data['name'], user_data['age'], user_data['gender'],
            user_data['height_cm'], user_data['weight_kg'], user_data['activity_level'],
            user_data['fitness_goal'], user_data['bmr'], user_data['tdee'],
            user_data['daily_target_calories']
        ))
        user_id = cursor.lastrowid
    
    conn.commit()
    conn.close()
    return user_id

def add_food_log(user_id, food_name, calories, protein, carbs, fat, meal_type, log_date):
    """Add a food entry to the log."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO food_log (user_id, food_name, calories, protein, carbs, fat, meal_type, log_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, food_name, calories, protein, carbs, fat, meal_type, log_date))
        conn.commit()
        print(f"DEBUG: Food logged - {food_name}, {calories} kcal for user {user_id}")
    except Exception as e:
        print(f"ERROR in add_food_log: {e}")
        raise
    finally:
        if conn:
            conn.close()

def add_activity_log(user_id, activity_type, duration_minutes, calories_burned, intensity, log_date):
    """Add an activity entry to the log."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO activity_log (user_id, activity_type, duration_minutes, calories_burned, intensity, log_date)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (user_id, activity_type, duration_minutes, calories_burned, intensity, log_date))
    conn.commit()
    conn.close()

def update_weight(user_id, weight_kg, record_date=None):
    """Record weight progress."""
    if record_date is None:
        record_date = date.today()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO weight_progress (user_id, weight_kg, record_date)
        VALUES (?, ?, ?)
    ''', (user_id, weight_kg, record_date))
    conn.commit()
    conn.close()

def get_food_logs(user_id, days=7):
    """Get food logs for the last N days."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('''
        SELECT * FROM food_log 
        WHERE user_id = ? AND log_date >= DATE('now', ?)
        ORDER BY log_date DESC, logged_at DESC
    ''', conn, params=(user_id, f'-{days} days'))
    conn.close()
    return df

def get_activity_logs(user_id, days=7):
    """Get activity logs for the last N days."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('''
        SELECT * FROM activity_log 
        WHERE user_id = ? AND log_date >= DATE('now', ?)
        ORDER BY log_date DESC, logged_at DESC
    ''', conn, params=(user_id, f'-{days} days'))
    conn.close()
    return df

def get_weight_progress(user_id):
    """Get weight history."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('''
        SELECT * FROM weight_progress 
        WHERE user_id = ? 
        ORDER BY record_date ASC
    ''', conn, params=(user_id,))
    conn.close()
    return df

def save_chat_message(user_id, user_message, bot_response):
    """Save chat interaction to database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO chat_history (user_id, user_message, bot_response)
        VALUES (?, ?, ?)
    ''', (user_id, user_message, bot_response))
    conn.commit()
    conn.close()

def get_today_summary(user_id):
    """Get today's calorie summary."""
    today = date.today().isoformat()
    conn = sqlite3.connect(DB_PATH)
    
    food_df = pd.read_sql_query('''
        SELECT SUM(calories) as total_calories FROM food_log 
        WHERE user_id = ? AND log_date = ?
    ''', conn, params=(user_id, today))
    calories_in = food_df['total_calories'].iloc[0] if not food_df['total_calories'].isna().iloc[0] else 0
    
    activity_df = pd.read_sql_query('''
        SELECT SUM(calories_burned) as total_burned FROM activity_log 
        WHERE user_id = ? AND log_date = ?
    ''', conn, params=(user_id, today))
    calories_out = activity_df['total_burned'].iloc[0] if not activity_df['total_burned'].isna().iloc[0] else 0
    
    conn.close()
    return calories_in, calories_out