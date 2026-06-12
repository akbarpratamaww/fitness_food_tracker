import sqlite3
import pandas as pd
from datetime import datetime, date, timedelta
import hashlib
import hmac
import os
import secrets
from dotenv import load_dotenv

load_dotenv()

# Path database utama (untuk data transaksional)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "fitness_tracker.db")

# Path database untuk dataset (bisa sama atau berbeda. Sesuaikan dengan file .db Anda)
DATASET_DB_PATH = os.path.join(BASE_DIR, "data", "fitnessfoodtracker.db")   # ganti dengan nama file database Anda

def get_connection():
    """Get database connection based on environment configurations."""
    db_type = os.getenv("DB_TYPE", "sqlite").lower()
    if db_type == "mysql":
        import mysql.connector
        return mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", "fitness_tracker")
        )
    else:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        return sqlite3.connect(DB_PATH)

def get_placeholder():
    """Get dynamic parameter placeholder for queries (MySQL uses %s, SQLite uses ?)."""
    db_type = os.getenv("DB_TYPE", "sqlite").lower()
    return "%s" if db_type == "mysql" else "?"

def init_database():
    """Initialize database tables for SQLite or MySQL."""
    db_type = os.getenv("DB_TYPE", "sqlite").lower()
    
    if db_type == "mysql":
        import mysql.connector
        # Connect to MySQL server first to create the database if not exists
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", "")
        )
        cursor = conn.cursor()
        db_name = os.getenv("DB_NAME", "fitness_tracker")
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        conn.commit()
        cursor.close()
        conn.close()
        
        # Re-connect to specify database
        conn = get_connection()
        cursor = conn.cursor()
        
        # Users table (MySQL)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(100) UNIQUE,
                password_hash VARCHAR(255),
                name VARCHAR(255),
                age INT,
                gender VARCHAR(50),
                height_cm DOUBLE,
                weight_kg DOUBLE,
                activity_level VARCHAR(255),
                fitness_goal VARCHAR(255),
                bmr DOUBLE,
                tdee DOUBLE,
                daily_target_calories DOUBLE,
                session_token VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Self-healing migration: add columns if they don't exist (MySQL)
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN username VARCHAR(100) UNIQUE")
            conn.commit()
        except Exception:
            pass
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN password_hash VARCHAR(255)")
            conn.commit()
        except Exception:
            pass
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN session_token VARCHAR(255)")
            conn.commit()
        except Exception:
            pass
        
        # Food log table (MySQL)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS food_log (
                log_id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                food_name VARCHAR(255),
                calories DOUBLE,
                protein DOUBLE,
                carbs DOUBLE,
                fat DOUBLE,
                meal_type VARCHAR(100),
                log_date DATE,
                logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Activity log table (MySQL)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_log (
                activity_id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                activity_type VARCHAR(255),
                duration_minutes DOUBLE,
                calories_burned DOUBLE,
                intensity VARCHAR(100),
                log_date DATE,
                logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Weight progress table (MySQL)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weight_progress (
                progress_id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                weight_kg DOUBLE,
                record_date DATE,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Chat history table (MySQL)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                chat_id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                user_message TEXT,
                bot_response TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()
    else:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Users table (SQLite)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password_hash TEXT,
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
        # Self-healing migration: add columns if they don't exist (SQLite)
        cursor.execute("PRAGMA table_info(users)")
        existing_cols = [row[1] for row in cursor.fetchall()]
        if 'username' not in existing_cols:
            cursor.execute("ALTER TABLE users ADD COLUMN username TEXT")
            conn.commit()
        if 'password_hash' not in existing_cols:
            cursor.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
            conn.commit()
        if 'session_token' not in existing_cols:
            cursor.execute("ALTER TABLE users ADD COLUMN session_token TEXT")
            conn.commit()
        
        # Food log table (SQLite)
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
        
        # Activity log table (SQLite)
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
        
        # Weight progress table (SQLite)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weight_progress (
                progress_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                weight_kg REAL,
                record_date DATE DEFAULT CURRENT_DATE,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Chat history table (SQLite)
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

# ==================== SESSION TOKEN FUNCTIONS ====================

def create_session_token(user_id: int) -> str:
    """
    Generate a cryptographically secure random session token, store it
    in the database, and return it.  The token is stored server-side so
    it can be invalidated on logout regardless of what is in the cookie.
    """
    token = secrets.token_hex(32)   # 64-char hex string
    conn = get_connection()
    cursor = conn.cursor()
    p = get_placeholder()
    try:
        cursor.execute(
            f"UPDATE users SET session_token = {p} WHERE user_id = {p}",
            (token, user_id)
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()
    return token


def validate_session_token(user_id: int, token: str) -> bool:
    """
    Return True only when the supplied token matches the one stored in
    the database for user_id.  Returns False if the token has been
    invalidated (logout) or if the user does not exist.
    """
    if not user_id or not token:
        return False
    conn = get_connection()
    p = get_placeholder()
    try:
        df = pd.read_sql_query(
            f"SELECT session_token FROM users WHERE user_id = {p}",
            conn, params=(user_id,)
        )
        if df.empty:
            return False
        stored = df['session_token'].iloc[0]
        if not stored:
            return False
        # Constant-time comparison to prevent timing attacks
        return hmac.compare_digest(str(stored), str(token))
    finally:
        conn.close()


def invalidate_session_token(user_id: int) -> None:
    """Clear the session token for user_id (called on logout)."""
    conn = get_connection()
    cursor = conn.cursor()
    p = get_placeholder()
    try:
        cursor.execute(
            f"UPDATE users SET session_token = NULL WHERE user_id = {p}",
            (user_id,)
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()


# ==================== AUTH FUNCTIONS ====================

def hash_password(password: str) -> str:
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def username_exists(username: str) -> bool:
    """Check if a username is already taken."""
    conn = get_connection()
    p = get_placeholder()
    try:
        df = pd.read_sql_query(
            f"SELECT user_id FROM users WHERE username = {p}",
            conn, params=(username,)
        )
        return len(df) > 0
    finally:
        conn.close()

def register_user(username: str, password: str, profile_data: dict) -> dict:
    """
    Register a new user. Returns dict with 'success' (bool), 'user_id' or 'error' (str).
    profile_data keys: name, age, gender, height_cm, weight_kg, activity_level,
                       fitness_goal, bmr, tdee, daily_target_calories
    """
    if username_exists(username):
        return {'success': False, 'error': 'Username sudah digunakan, silakan pilih username lain.'}
    
    pw_hash = hash_password(password)
    conn = get_connection()
    cursor = conn.cursor()
    p = get_placeholder()
    try:
        cursor.execute(f'''
            INSERT INTO users (username, password_hash, name, age, gender, height_cm,
                weight_kg, activity_level, fitness_goal, bmr, tdee, daily_target_calories)
            VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p})
        ''', (
            username, pw_hash,
            profile_data['name'], profile_data['age'], profile_data['gender'],
            profile_data['height_cm'], profile_data['weight_kg'],
            profile_data['activity_level'], profile_data['fitness_goal'],
            profile_data['bmr'], profile_data['tdee'], profile_data['daily_target_calories']
        ))
        conn.commit()
        user_id = cursor.lastrowid
        cursor.close()
        conn.close()
        return {'success': True, 'user_id': user_id}
    except Exception as e:
        conn.close()
        return {'success': False, 'error': str(e)}

def authenticate_user(username: str, password: str):
    """
    Authenticate a user. Returns the user row (Pandas Series) on success, or None on failure.
    """
    pw_hash = hash_password(password)
    conn = get_connection()
    p = get_placeholder()
    try:
        df = pd.read_sql_query(
            f"SELECT * FROM users WHERE username = {p} AND password_hash = {p}",
            conn, params=(username, pw_hash)
        )
        return df.iloc[0] if len(df) > 0 else None
    finally:
        conn.close()

def read_table_to_df(table_name, db_path=None):
    """
    Membaca seluruh isi tabel dari database SQLite ke pandas DataFrame.
    Berguna untuk membaca dataset seperti exercise_dataset, body_performance, food_dataset.
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
    """Mengecek apakah suatu tabel ada di database SQLite."""
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

def get_user(user_id=None):
    """Get user data. If user_id is None, get the most recent user."""
    conn = get_connection()
    placeholder = get_placeholder()
    if user_id:
        df = pd.read_sql_query(f"SELECT * FROM users WHERE user_id = {placeholder}", conn, params=(user_id,))
    else:
        df = pd.read_sql_query("SELECT * FROM users ORDER BY user_id DESC LIMIT 1", conn)
    conn.close()
    return df.iloc[0] if len(df) > 0 else None

def save_user(user_data):
    """Save or update user data."""
    conn = get_connection()
    cursor = conn.cursor()
    p = get_placeholder()
    
    if 'user_id' in user_data and user_data['user_id']:
        # Update existing user
        cursor.execute(f'''
            UPDATE users SET name={p}, age={p}, gender={p}, height_cm={p}, weight_kg={p}, 
            activity_level={p}, fitness_goal={p}, bmr={p}, tdee={p}, daily_target_calories={p}
            WHERE user_id={p}
        ''', (
            user_data['name'], user_data['age'], user_data['gender'],
            user_data['height_cm'], user_data['weight_kg'], user_data['activity_level'],
            user_data['fitness_goal'], user_data['bmr'], user_data['tdee'],
            user_data['daily_target_calories'], user_data['user_id']
        ))
        user_id = user_data['user_id']
    else:
        # Insert new user
        cursor.execute(f'''
            INSERT INTO users (name, age, gender, height_cm, weight_kg, activity_level,
            fitness_goal, bmr, tdee, daily_target_calories)
            VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p})
        ''', (
            user_data['name'], user_data['age'], user_data['gender'],
            user_data['height_cm'], user_data['weight_kg'], user_data['activity_level'],
            user_data['fitness_goal'], user_data['bmr'], user_data['tdee'],
            user_data['daily_target_calories']
        ))
        user_id = cursor.lastrowid
    
    conn.commit()
    cursor.close()
    conn.close()
    return user_id

def add_food_log(user_id, food_name, calories, protein, carbs, fat, meal_type, log_date):
    """Add a food entry to the log."""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        p = get_placeholder()
        cursor.execute(f'''
            INSERT INTO food_log (user_id, food_name, calories, protein, carbs, fat, meal_type, log_date)
            VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p}, {p})
        ''', (user_id, food_name, calories, protein, carbs, fat, meal_type, log_date))
        conn.commit()
        cursor.close()
    except Exception as e:
        print(f"ERROR in add_food_log: {e}")
        raise
    finally:
        if conn:
            conn.close()

def add_activity_log(user_id, activity_type, duration_minutes, calories_burned, intensity, log_date):
    """Add an activity entry to the log."""
    conn = get_connection()
    cursor = conn.cursor()
    p = get_placeholder()
    cursor.execute(f'''
        INSERT INTO activity_log (user_id, activity_type, duration_minutes, calories_burned, intensity, log_date)
        VALUES ({p}, {p}, {p}, {p}, {p}, {p})
    ''', (user_id, activity_type, duration_minutes, calories_burned, intensity, log_date))
    conn.commit()
    cursor.close()
    conn.close()

def update_weight(user_id, weight_kg, record_date=None):
    """Record weight progress."""
    if record_date is None:
        record_date = date.today().isoformat()
    elif isinstance(record_date, date):
        record_date = record_date.isoformat()
        
    conn = get_connection()
    cursor = conn.cursor()
    p = get_placeholder()
    cursor.execute(f'''
        INSERT INTO weight_progress (user_id, weight_kg, record_date)
        VALUES ({p}, {p}, {p})
    ''', (user_id, weight_kg, record_date))
    conn.commit()
    cursor.close()
    conn.close()

def get_food_logs(user_id, days=7):
    """Get food logs for the last N days."""
    conn = get_connection()
    p = get_placeholder()
    start_date = (date.today() - timedelta(days=days)).isoformat()
    df = pd.read_sql_query(f'''
        SELECT * FROM food_log 
        WHERE user_id = {p} AND log_date >= {p}
        ORDER BY log_date DESC, logged_at DESC
    ''', conn, params=(user_id, start_date))
    conn.close()
    return df

def get_activity_logs(user_id, days=7):
    """Get activity logs for the last N days."""
    conn = get_connection()
    p = get_placeholder()
    start_date = (date.today() - timedelta(days=days)).isoformat()
    df = pd.read_sql_query(f'''
        SELECT * FROM activity_log 
        WHERE user_id = {p} AND log_date >= {p}
        ORDER BY log_date DESC, logged_at DESC
    ''', conn, params=(user_id, start_date))
    conn.close()
    return df

def get_weight_progress(user_id):
    """Get weight history."""
    conn = get_connection()
    p = get_placeholder()
    df = pd.read_sql_query(f'''
        SELECT * FROM weight_progress 
        WHERE user_id = {p} 
        ORDER BY record_date ASC
    ''', conn, params=(user_id,))
    conn.close()
    return df

def save_chat_message(user_id, user_message, bot_response):
    """Save chat interaction to database."""
    conn = get_connection()
    cursor = conn.cursor()
    p = get_placeholder()
    cursor.execute(f'''
        INSERT INTO chat_history (user_id, user_message, bot_response)
        VALUES ({p}, {p}, {p})
    ''', (user_id, user_message, bot_response))
    conn.commit()
    cursor.close()
    conn.close()

def get_today_summary(user_id):
    """Get today's calorie summary."""
    today = date.today().isoformat()
    conn = get_connection()
    p = get_placeholder()
    
    # Total calories consumed today
    food_df = pd.read_sql_query(f'''
        SELECT SUM(calories) as total_calories FROM food_log 
        WHERE user_id = {p} AND log_date = {p}
    ''', conn, params=(user_id, today))
    calories_in = food_df['total_calories'].iloc[0] if not food_df['total_calories'].isna().iloc[0] else 0
    
    # Total calories burned today
    activity_df = pd.read_sql_query(f'''
        SELECT SUM(calories_burned) as total_burned FROM activity_log 
        WHERE user_id = {p} AND log_date = {p}
    ''', conn, params=(user_id, today))
    calories_out = activity_df['total_burned'].iloc[0] if not activity_df['total_burned'].isna().iloc[0] else 0
    
    conn.close()
    return float(calories_in), float(calories_out)

def reset_today_logs(user_id):
    """Delete all food and activity logs for today for the given user."""
    today = date.today().isoformat()
    conn = get_connection()
    cursor = conn.cursor()
    p = get_placeholder()
    try:
        cursor.execute(f"DELETE FROM food_log WHERE user_id = {p} AND log_date = {p}", (user_id, today))
        cursor.execute(f"DELETE FROM activity_log WHERE user_id = {p} AND log_date = {p}", (user_id, today))
        conn.commit()
    finally:
        cursor.close()
        conn.close()

def add_custom_food_to_dataset(food_name, calories_per_100g, protein_g, carbs_g, fat_g):
    """Add a custom food item to the food_dataset table in the SQLite database and append to CSV."""
    conn = sqlite3.connect(DATASET_DB_PATH)
    cursor = conn.cursor()
    csv_path = os.path.join(BASE_DIR, "data", "food_dataset.csv")
    try:
        # Self-healing: create the table if it does not exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS food_dataset (
                Food TEXT,
                Calories_per_100g REAL,
                Protein_g REAL,
                Carbs_g REAL,
                Fat_g REAL
            )
        ''')
        
        # Check if the table is empty. If empty, populate it.
        cursor.execute("SELECT COUNT(*) FROM food_dataset")
        count = cursor.fetchone()[0]
        if count == 0:
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    df.to_sql("food_dataset", conn, if_exists="append", index=False)
                except Exception as e:
                    print(f"Error seeding food_dataset table from CSV: {e}")
            else:
                # Seeding with basic sample foods
                sample_foods = pd.DataFrame({
                    'Food': ['Apple', 'Banana', 'Orange', 'Chicken Breast', 'Salmon', 'Rice', 'Pasta', 
                             'Bread', 'Egg', 'Milk', 'Yogurt', 'Cheese', 'Broccoli', 'Spinach', 'Carrot',
                             'Pizza', 'Burger', 'French Fries', 'Ice Cream', 'Chocolate Cake'],
                    'Calories_per_100g': [52, 89, 47, 165, 208, 130, 131, 265, 155, 42, 59, 402, 34, 23, 41,
                                           285, 354, 312, 207, 424],
                    'Protein_g': [0.3, 1.1, 0.9, 31, 20, 2.7, 5, 9, 13, 3.4, 10, 25, 2.8, 2.9, 0.9,
                                  12, 17, 3.4, 3.5, 5.3],
                    'Carbs_g': [14, 23, 12, 0, 0, 28, 25, 49, 1.1, 5, 3.6, 1.3, 7, 3.6, 10,
                                30, 30, 41, 24, 58],
                    'Fat_g': [0.2, 0.3, 0.1, 3.6, 13, 0.3, 1.1, 3.2, 11, 1, 0.4, 33, 0.4, 0.4, 0.2,
                              10, 20, 15, 11, 20]
                })
                sample_foods.to_sql("food_dataset", conn, if_exists="append", index=False)
                try:
                    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                    sample_foods.to_csv(csv_path, index=False)
                except:
                    pass

        cursor.execute('''
            INSERT INTO food_dataset (Food, Calories_per_100g, Protein_g, Carbs_g, Fat_g)
            VALUES (?, ?, ?, ?, ?)
        ''', (food_name, calories_per_100g, protein_g, carbs_g, fat_g))
        conn.commit()
    finally:
        cursor.close()
        conn.close()
        
    # Also append to the CSV file to keep them in sync
    if os.path.exists(csv_path):
        try:
            df_new = pd.DataFrame([{
                'Food': food_name,
                'Calories_per_100g': calories_per_100g,
                'Protein_g': protein_g,
                'Carbs_g': carbs_g,
                'Fat_g': fat_g
            }])
            df_new.to_csv(csv_path, mode='a', header=False, index=False)
        except Exception as e:
            print(f"Error appending to food_dataset.csv: {e}")

def delete_food_log(log_id):
    """Delete a food log entry."""
    conn = get_connection()
    cursor = conn.cursor()
    p = get_placeholder()
    try:
        cursor.execute(f"DELETE FROM food_log WHERE log_id = {p}", (log_id,))
        conn.commit()
    finally:
        cursor.close()
        conn.close()

def delete_activity_log(activity_id):
    """Delete an activity log entry."""
    conn = get_connection()
    cursor = conn.cursor()
    p = get_placeholder()
    try:
        cursor.execute(f"DELETE FROM activity_log WHERE activity_id = {p}", (activity_id,))
        conn.commit()
    finally:
        cursor.close()
        conn.close()
