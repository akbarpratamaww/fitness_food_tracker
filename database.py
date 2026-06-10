import sqlite3
import pandas as pd
from datetime import datetime, date
import bcrypt

DB_PATH = "data/fitness_tracker.db"
DATASET_DB_PATH = "data/fitnessfoodtracker.db"

def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
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
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weight_progress (
            progress_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            weight_kg REAL,
            record_date DATE DEFAULT CURRENT_DATE,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
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

def register_user(email, password, name=""):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    try:
        cursor.execute("INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?)", 
                       (email, password_hash, name))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return user_id, True
    except sqlite3.IntegrityError:
        conn.close()
        return None, False

def login_user(email, password):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash']):
        return dict(user)
    return None

def get_user(user_id=None):
    conn = sqlite3.connect(DB_PATH)
    if user_id:
        df = pd.read_sql_query("SELECT * FROM users WHERE user_id = ?", conn, params=(user_id,))
    else:
        df = pd.read_sql_query("SELECT * FROM users ORDER BY user_id DESC LIMIT 1", conn)
    conn.close()
    return df.iloc[0] if len(df) > 0 else None

def save_user(user_data):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if 'user_id' in user_data and user_data['user_id']:
        cursor.execute('''
            UPDATE users SET name=?, age=?, gender=?, height_cm=?, weight_kg=?, 
            activity_level=?, fitness_goal=?, bmr=?, tdee=?, daily_target_calories=?
            WHERE user_id=?
        ''', (user_data['name'], user_data['age'], user_data['gender'],
              user_data['height_cm'], user_data['weight_kg'], user_data['activity_level'],
              user_data['fitness_goal'], user_data['bmr'], user_data['tdee'],
              user_data['daily_target_calories'], user_data['user_id']))
        user_id = user_data['user_id']
    else:
        cursor.execute('''
            INSERT INTO users (name, age, gender, height_cm, weight_kg, activity_level,
            fitness_goal, bmr, tdee, daily_target_calories)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        ''', (user_data['name'], user_data['age'], user_data['gender'],
              user_data['height_cm'], user_data['weight_kg'], user_data['activity_level'],
              user_data['fitness_goal'], user_data['bmr'], user_data['tdee'],
              user_data['daily_target_calories']))
        user_id = cursor.lastrowid
    
    conn.commit()
    conn.close()
    return user_id

def add_food_log(user_id, food_name, calories, protein, carbs, fat, meal_type, log_date):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO food_log (user_id, food_name, calories, protein, carbs, fat, meal_type, log_date)
        VALUES (?,?,?,?,?,?,?,?)
    ''', (user_id, food_name, calories, protein, carbs, fat, meal_type, log_date))
    conn.commit()
    conn.close()

def get_food_logs(user_id, days=7):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('''
        SELECT * FROM food_log WHERE user_id=? AND log_date>=DATE('now',?)
        ORDER BY log_date DESC
    ''', conn, params=(user_id, f'-{days} days'))
    conn.close()
    return df

def add_activity_log(user_id, activity_type, duration_minutes, calories_burned, intensity, log_date):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO activity_log (user_id, activity_type, duration_minutes, calories_burned, intensity, log_date)
        VALUES (?,?,?,?,?,?)
    ''', (user_id, activity_type, duration_minutes, calories_burned, intensity, log_date))
    conn.commit()
    conn.close()

def get_activity_logs(user_id, days=7):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('''
        SELECT * FROM activity_log WHERE user_id=? AND log_date>=DATE('now',?)
        ORDER BY log_date DESC
    ''', conn, params=(user_id, f'-{days} days'))
    conn.close()
    return df

def update_weight(user_id, weight_kg, record_date=None):
    if record_date is None:
        record_date = date.today()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO weight_progress (user_id, weight_kg, record_date) VALUES (?,?,?)',
                   (user_id, weight_kg, record_date))
    conn.commit()
    conn.close()
    
    # Update also current weight in users table
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET weight_kg = ? WHERE user_id = ?", (weight_kg, user_id))
    conn.commit()
    conn.close()

def get_weight_progress(user_id):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT * FROM weight_progress WHERE user_id=? ORDER BY record_date ASC',
                           conn, params=(user_id,))
    conn.close()
    return df

def get_today_summary(user_id):
    today = date.today().isoformat()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COALESCE(SUM(calories),0) FROM food_log WHERE user_id=? AND log_date=?", 
                   (user_id, today))
    calories_in = cursor.fetchone()[0]
    cursor.execute("SELECT COALESCE(SUM(calories_burned),0) FROM activity_log WHERE user_id=? AND log_date=?", 
                   (user_id, today))
    calories_out = cursor.fetchone()[0]
    conn.close()
    return calories_in, calories_out

def save_chat_message(user_id, user_message, bot_response):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO chat_history (user_id, user_message, bot_response) VALUES (?,?,?)',
                   (user_id, user_message, bot_response))
    conn.commit()
    conn.close()

def read_table_to_df(table_name, db_path=None):
    if db_path is None:
        db_path = DATASET_DB_PATH
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        conn.close()

def table_exists(table_name, db_path=None):
    if db_path is None:
        db_path = DATASET_DB_PATH
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def get_table_columns(table_name, db_path=None):
    if db_path is None:
        db_path = DATASET_DB_PATH
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    conn.close()
    return columns