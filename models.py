# models.py - Lengkap dengan Model 1 (Regresi) dan Model 2 (Klasifikasi)
# Dataset dibaca dari SQLite, bukan CSV
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import xgboost as xgb
from sklearn.svm import SVC
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Import fungsi dari database.py
from database import read_table_to_df, table_exists

# ==================== MODEL 1: PREDIKSI KALORI (REGRESI) ====================
CAL_MODEL_PATH = 'models/calorie_predictor.pkl'
CAL_SCALER_PATH = 'models/calorie_scaler.pkl'
CAL_FEATURE_ORDER_PATH = 'models/calorie_feature_order.pkl'

def train_calorie_prediction_model():
    print("🔄 Training calorie prediction model (reading from SQLite)...")
    
    # Pastikan tabel exercise_dataset ada di database
    if not table_exists('exercise_dataset'):
        raise FileNotFoundError("❌ Tabel 'exercise_dataset' tidak ditemukan dalam database. Pastikan file database sudah benar.")
    
    # Baca data dari SQLite
    df = read_table_to_df('exercise_dataset')
    if df is None or df.empty:
        raise ValueError("❌ Data dari tabel 'exercise_dataset' kosong.")
    
    print(f"✅ Dataset asli: {len(df)} baris")
    
    # Sampling: gunakan 60.000 baris (sesuai kode Anda)
    n_samples = 60000
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42)
        print(f"✅ Menggunakan {n_samples} baris untuk training (sampling acak)")
    
    # Kolom yang diperlukan (pastikan nama kolom sesuai dengan di database)
    required_cols = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
    # Cek ketersediaan kolom (case insensitive)
    for col in required_cols:
        if col not in df.columns:
            # Coba cari versi lowercase
            alt = col.lower()
            if alt in df.columns:
                df.rename(columns={alt: col}, inplace=True)
            else:
                raise KeyError(f"❌ Kolom '{col}' tidak ditemukan. Kolom yang ada: {df.columns.tolist()}")
    
    X = df[['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']].copy()
    y = df['Calories']
    
    # Mapping Sex ke numerik
    sex_map = {'male': 1, 'Male': 1, 'M': 1, 'female': 0, 'Female': 0, 'F': 0}
    X['Sex'] = X['Sex'].map(sex_map).fillna(0).astype(int)
    
    # Standardisasi fitur numerik
    scaler = StandardScaler()
    numeric_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    feature_order = X.columns.tolist()
    print(f"📌 Feature order: {feature_order}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Random Forest
    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluasi
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"✅ Model terlatih! MAE: {mae:.2f}, R²: {r2:.3f}")
    
    # Simpan model dan komponen dengan kompresi
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, CAL_MODEL_PATH, compress=3)
    joblib.dump(scaler, CAL_SCALER_PATH, compress=3)
    joblib.dump(feature_order, CAL_FEATURE_ORDER_PATH, compress=3)
    
    return model, None, mae, r2

def load_calorie_model():
    try:
        model = joblib.load(CAL_MODEL_PATH)
        scaler = joblib.load(CAL_SCALER_PATH)
        feature_order = joblib.load(CAL_FEATURE_ORDER_PATH)
        return model, scaler, feature_order
    except Exception as e:
        print(f"⚠️ Model tidak ditemukan atau error: {e}. Melatih model baru...")
        train_calorie_prediction_model()
        model = joblib.load(CAL_MODEL_PATH)
        scaler = joblib.load(CAL_SCALER_PATH)
        feature_order = joblib.load(CAL_FEATURE_ORDER_PATH)
        return model, scaler, feature_order

def predict_calories_burned(gender, age, height_cm, weight_kg, duration_min, heart_rate_bpm, body_temp_c):
    model, scaler, feature_order = load_calorie_model()
    gender_val = 1 if gender == 'Male' else 0
    input_dict = {
        'Sex': gender_val,
        'Age': age,
        'Height': height_cm,
        'Weight': weight_kg,
        'Duration': duration_min,
        'Heart_Rate': heart_rate_bpm,
        'Body_Temp': body_temp_c
    }
    input_df = pd.DataFrame([input_dict])[feature_order]
    numeric_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    prediction = model.predict(input_df)[0]
    return round(prediction, 0)


# ==================== MODEL 2: KLASIFIKASI TINGKAT KEBUGARAN ====================
BODY_MODEL_PATH = 'models/body_performance_classifier.pkl'
BODY_SCALER_PATH = 'models/body_scaler.pkl'
BODY_TARGET_PATH = 'models/body_target_encoder.pkl'
BODY_GENDER_MAP_PATH = 'models/body_gender_map.pkl'
BODY_FEATURE_ORDER_PATH = 'models/body_feature_order.pkl'

def train_body_performance_model():
    print("🔄 Training fitness classification model (reading from SQLite)...")
    
    # Cek apakah tabel body_performance ada
    if not table_exists('body_performance'):
        raise FileNotFoundError("❌ Tabel 'body_performance' tidak ditemukan dalam database.")
    
    # Baca data dari SQLite
    df = read_table_to_df('body_performance')
    if df is None or df.empty:
        raise ValueError("❌ Data dari tabel 'body_performance' kosong.")
    
    print(f"✅ Dataset dimuat: {len(df)} baris")
    print(f"📋 Kolom asli di database: {df.columns.tolist()}")
    
    # Bersihkan nama kolom (lowercase, ganti spasi dengan underscore)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Mapping nama kolom yang mungkin berbeda
    rename_map = {
        'body_fat_%': 'body_fat_percent',
        'sit_and_bend_forward_cm': 'sit_bend',
        'sit-ups_counts': 'situps',
        'gripforce': 'gripforce',
        'broad_jump_cm': 'broad_jump'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Kolom yang diperlukan
    required_cols = ['age', 'gender', 'height_cm', 'weight_kg', 'body_fat_percent',
                     'diastolic', 'systolic', 'gripforce', 'sit_bend', 'situps', 'broad_jump', 'class']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"❌ Kolom tidak ditemukan: {missing}. Kolom yang ada: {list(df.columns)}")
    
    # Encode target class (A,B,C,D)
    target_encoder = LabelEncoder()
    df['class'] = target_encoder.fit_transform(df['class'])  # A=0, B=1, C=2, D=3
    
    # Hitung BMI
    df['bmi'] = df['weight_kg'] / ((df['height_cm']/100) ** 2)
    
    feature_cols = required_cols[:-1] + ['bmi']  # semua kolom kecuali 'class'
    X = df[feature_cols].copy()
    y = df['class']
    
    # Mapping gender: database biasanya berisi 'F' dan 'M'
    gender_map = {'F': 0, 'M': 1}
    X['gender'] = X['gender'].str.strip().str.upper().map(gender_map).fillna(0).astype(int)
    
    # Simpan urutan fitur
    feature_order = X.columns.tolist()
    print(f"📌 Feature order (model 2): {feature_order}")
    
    # Standardisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42,
                                  use_label_encoder=False, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)
    
    # SVM
    svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    svm.fit(X_train, y_train)
    
    # Evaluasi
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    acc_xgb = accuracy_score(y_test, xgb_model.predict(X_test))
    acc_svm = accuracy_score(y_test, svm.predict(X_test))
    print(f"✅ Fitness models trained! RF: {acc_rf:.2%}, XGB: {acc_xgb:.2%}, SVM: {acc_svm:.2%}")
    
    models = {'random_forest': rf, 'xgboost': xgb_model, 'svm': svm}
    os.makedirs('models', exist_ok=True)
    # Simpan dengan kompresi
    joblib.dump(models, BODY_MODEL_PATH, compress=3)
    joblib.dump(scaler, BODY_SCALER_PATH, compress=3)
    joblib.dump(target_encoder, BODY_TARGET_PATH, compress=3)
    joblib.dump(gender_map, BODY_GENDER_MAP_PATH, compress=3)
    joblib.dump(feature_order, BODY_FEATURE_ORDER_PATH, compress=3)
    
    return models, scaler, target_encoder, gender_map, (acc_rf, acc_xgb, acc_svm)

def load_body_performance_model():
    try:
        models = joblib.load(BODY_MODEL_PATH)
        scaler = joblib.load(BODY_SCALER_PATH)
        target_encoder = joblib.load(BODY_TARGET_PATH)
        gender_map = joblib.load(BODY_GENDER_MAP_PATH)
        feature_order = joblib.load(BODY_FEATURE_ORDER_PATH)
        return models, scaler, target_encoder, gender_map, feature_order
    except Exception as e:
        print(f"⚠️ Model kebugaran belum ada atau error: {e}. Melatih dari awal...")
        models, scaler, target_encoder, gender_map, _ = train_body_performance_model()
        feature_order = joblib.load(BODY_FEATURE_ORDER_PATH)
        return models, scaler, target_encoder, gender_map, feature_order

def predict_fitness_level(model_name, input_data):
    models, scaler, target_encoder, gender_map, feature_order = load_body_performance_model()
    
    # Hitung BMI
    height_m = input_data['height_cm'] / 100
    bmi = input_data['weight_kg'] / (height_m ** 2)
    
    # Konversi gender input user ke kode dataset
    user_gender = input_data['gender']
    gender_code = 'F' if user_gender == 'Female' else 'M'
    gender_val = gender_map.get(gender_code, 0)
    
    # Siapkan dictionary input
    data_dict = {
        'gender': gender_val,
        'age': input_data['age'],
        'height_cm': input_data['height_cm'],
        'weight_kg': input_data['weight_kg'],
        'body_fat_percent': input_data['body_fat'],
        'diastolic': input_data['diastolic'],
        'systolic': input_data['systolic'],
        'gripforce': input_data['gripForce'],
        'sit_bend': input_data['sit_bend'],
        'situps': input_data['situps'],
        'broad_jump': input_data['broad_jump'],
        'bmi': bmi
    }
    # Urutkan sesuai feature_order
    input_df = pd.DataFrame([data_dict])[feature_order]
    
    # Scaling
    input_scaled = scaler.transform(input_df)
    
    # Prediksi
    model = models[model_name]
    pred = model.predict(input_scaled)[0]
    
    # Confidence
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(input_scaled)[0]
        confidence = probs[pred] * 100
    else:
        confidence = 100.0
    
    # Inverse transform target ke label asli
    class_label = target_encoder.inverse_transform([pred])[0]
    display_map = {'A': 'A (Excellent)', 'B': 'B (Good)', 'C': 'C (Average)', 'D': 'D (Poor)'}
    display = display_map.get(class_label, class_label)
    
    return display, confidence, pred