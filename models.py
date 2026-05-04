# models.py - Lengkap dengan Model 1 (Regresi) dan Model 2 (Klasifikasi)
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

# ==================== MODEL 1: PREDIKSI KALORI (REGRESI) ====================
# Menggunakan dataset Predict Calorie Expenditure (Kaggle)
# Kolom: Sex, Age, Height, Weight, Duration, Heart_Rate, Body_Temp, Calories
CAL_MODEL_PATH = 'models/calorie_predictor.pkl'
CAL_SCALER_PATH = 'models/calorie_scaler.pkl'
CAL_FEATURE_ORDER_PATH = 'models/calorie_feature_order.pkl'

def train_calorie_prediction_model():
    print("🔄 Training calorie prediction model (with compression)...")
    file_path = 'data/exercise_dataset.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError("Dataset exercise_dataset.csv tidak ditemukan.")
    
    df = pd.read_csv(file_path)
    print(f"Dataset asli: {len(df)} baris")
    
    # Sampling: gunakan 8.000 baris (cukup untuk akurasi baik)
    n_samples = 60000
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42)
        print(f"✅ Menggunakan {n_samples} baris untuk training")
    
    # Pastikan kolom yang diperlukan ada
    required_cols = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Kolom '{col}' tidak ditemukan.")
    
    X = df[['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']].copy()
    y = df['Calories']
    
    # Mapping Sex ke numerik
    sex_map = {'male': 1, 'Male': 1, 'M': 1, 'female': 0, 'Female': 0, 'F': 0}
    X['Sex'] = X['Sex'].map(sex_map).fillna(0).astype(int)
    
    # Standardisasi
    scaler = StandardScaler()
    numeric_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    feature_order = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"✅ Model terlatih! MAE: {mae:.2f}, R²: {r2:.3f}")
    
    os.makedirs('models', exist_ok=True)
    # Simpan dengan kompresi
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
# Dataset: body_performance.csv
BODY_MODEL_PATH = 'models/body_performance_classifier.pkl'
BODY_SCALER_PATH = 'models/body_scaler.pkl'
BODY_TARGET_PATH = 'models/body_target_encoder.pkl'
BODY_GENDER_MAP_PATH = 'models/body_gender_map.pkl'
BODY_FEATURE_ORDER_PATH = 'models/body_feature_order.pkl'

def train_body_performance_model():
    possible_names = ['data/body_performance.csv', 'data/bodyPerformance.csv', 'data/bodyperformance.csv']
    df = None
    for name in possible_names:
        try:
            df = pd.read_csv(name)
            print(f"✅ Dataset ditemukan: {name}")
            break
        except FileNotFoundError:
            continue
    
    if df is None:
        raise FileNotFoundError("❌ Dataset tidak ditemukan. Pastikan file 'body_performance.csv' di folder 'data/'")
    
    # Bersihkan nama kolom
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    print("Kolom asli setelah cleaning:", list(df.columns))
    
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
        raise KeyError(f"Kolom tidak ditemukan: {missing}. Kolom yang ada: {list(df.columns)}")
    
    # Encode target
    target_encoder = LabelEncoder()
    df['class'] = target_encoder.fit_transform(df['class'])
    
    # Hitung BMI
    df['bmi'] = df['weight_kg'] / ((df['height_cm']/100) ** 2)
    
    feature_cols = required_cols[:-1] + ['bmi']
    X = df[feature_cols].copy()
    y = df['class']
    
    # Mapping gender: dataset menggunakan 'F' dan 'M'
    gender_map = {'F': 0, 'M': 1}
    # Bersihkan kolom gender (hapus spasi, uppercase)
    X['gender'] = X['gender'].str.strip().str.upper().map(gender_map).fillna(0).astype(int)
    
    # Simpan urutan fitur
    feature_order = X.columns.tolist()
    print("Feature order (model 2):", feature_order)
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
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
    
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    acc_xgb = accuracy_score(y_test, xgb_model.predict(X_test))
    acc_svm = accuracy_score(y_test, svm.predict(X_test))
    
    print(f"✅ Fitness model trained! RF: {acc_rf:.2%}, XGB: {acc_xgb:.2%}, SVM: {acc_svm:.2%}")
    
    models = {'random_forest': rf, 'xgboost': xgb_model, 'svm': svm}
    os.makedirs('models', exist_ok=True)
    # Di dalam models.py, pada fungsi train_body_performance_model
    # Saat menyimpan models, tambahkan parameter compress=3
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
        print(f"⚠️ Model kebugaran belum ada atau error: {e}, melatih dari awal...")
        models, scaler, target_encoder, gender_map, _ = train_body_performance_model()
        feature_order = joblib.load(BODY_FEATURE_ORDER_PATH)
        return models, scaler, target_encoder, gender_map, feature_order

def predict_fitness_level(model_name, input_data):
    models, scaler, target_encoder, gender_map, feature_order = load_body_performance_model()
    
    # Hitung BMI
    height_m = input_data['height_cm'] / 100
    bmi = input_data['weight_kg'] / (height_m ** 2)
    
    # Konversi gender input user ('Female'/'Male') ke kode dataset ('F'/'M')
    user_gender = input_data['gender']
    if user_gender == 'Female':
        gender_code = 'F'
    elif user_gender == 'Male':
        gender_code = 'M'
    else:
        gender_code = 'F'
    gender_val = gender_map.get(gender_code, 0)
    
    # Buat dictionary dengan urutan sesuai feature_order
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
    # Reorder sesuai feature_order
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