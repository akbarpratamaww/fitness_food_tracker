# models.py - Final with feature order fix (tanpa merusak yang lain)
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

# ==================== MODEL 1: PREDIKSI KALORI (REGRESI) ====================
CAL_MODEL_PATH = 'models/calorie_predictor.pkl'
CAL_ENCODER_PATH = 'models/label_encoder.pkl'
CAL_SCALER_PATH = 'models/scaler.pkl'

def train_calorie_prediction_model():
    try:
        df = pd.read_csv('data/exercise_dataset.csv')
    except FileNotFoundError:
        np.random.seed(42)
        n_samples = 2000
        df = pd.DataFrame({
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Age': np.random.randint(18, 70, n_samples),
            'Height_cm': np.random.normal(170, 10, n_samples).clip(140, 220),
            'Weight_kg': np.random.normal(75, 15, n_samples).clip(40, 150),
            'Duration_min': np.random.randint(10, 120, n_samples),
            'Heart_Rate_bpm': np.random.randint(80, 180, n_samples),
            'Body_Temp_C': np.random.normal(37.5, 0.5, n_samples).clip(36.0, 39.5),
        })
        df['Calories_Burned'] = (
            0.5 * df['Duration_min'] + 
            0.8 * df['Weight_kg'] + 
            0.3 * df['Heart_Rate_bpm'] - 
            0.2 * df['Age'] +
            np.where(df['Gender'] == 'Male', 50, 0) +
            np.random.normal(0, 20, n_samples)
        ).clip(50, 1200).round()
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/exercise_dataset.csv', index=False)
    
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    X = df.drop('Calories_Burned', axis=1)
    y = df['Calories_Burned']
    
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"✅ Calorie model trained! MAE: {mae:.2f}, R²: {r2:.3f}")
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, CAL_MODEL_PATH)
    joblib.dump(label_encoders, CAL_ENCODER_PATH)
    joblib.dump(scaler, CAL_SCALER_PATH)
    return model, label_encoders, mae, r2

def load_calorie_model():
    try:
        model = joblib.load(CAL_MODEL_PATH)
        label_encoders = joblib.load(CAL_ENCODER_PATH)
        scaler = joblib.load(CAL_SCALER_PATH)
        return model, label_encoders, scaler
    except:
        return train_calorie_prediction_model()

def predict_calories_burned(gender, age, height_cm, weight_kg, duration_min, heart_rate_bpm, body_temp_c):
    model, label_encoders, scaler = load_calorie_model()
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Height_cm': [height_cm],
        'Weight_kg': [weight_kg],
        'Duration_min': [duration_min],
        'Heart_Rate_bpm': [heart_rate_bpm],
        'Body_Temp_C': [body_temp_c]
    })
    if 'Gender' in label_encoders:
        input_data['Gender'] = label_encoders['Gender'].transform([gender])[0]
    numeric_cols = ['Age', 'Height_cm', 'Weight_kg', 'Duration_min', 'Heart_Rate_bpm', 'Body_Temp_C']
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
    prediction = model.predict(input_data)[0]
    return round(prediction, 0)

# ==================== MODEL 2: KLASIFIKASI TINGKAT KEBUGARAN ====================
BODY_MODEL_PATH = 'models/body_performance_classifier.pkl'
BODY_SCALER_PATH = 'models/body_scaler.pkl'
BODY_TARGET_PATH = 'models/body_target_encoder.pkl'
BODY_FEATURE_ORDER_PATH = 'models/feature_order.pkl'   # untuk menyimpan urutan kolom

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
    
    # Mapping nama kolom sesuai dengan dataset Anda
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
    
    # Encode target (A,B,C,D)
    target_encoder = LabelEncoder()
    df['class'] = target_encoder.fit_transform(df['class'])
    
    # Hitung BMI
    df['bmi'] = df['weight_kg'] / ((df['height_cm']/100) ** 2)
    
    feature_cols = required_cols[:-1] + ['bmi']
    X = df[feature_cols].copy()
    y = df['class']
    
    # Mapping gender berdasarkan nilai di dataset: 'F' dan 'M'
    gender_map = {'F': 0, 'M': 1}
    X['gender'] = X['gender'].str.upper().map(gender_map).fillna(0).astype(int)
    
    # Simpan urutan fitur (feature order) - PENTING untuk prediksi
    feature_order = X.columns.tolist()
    print("Feature order:", feature_order)
    
    # Scaling semua fitur numerik
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
    joblib.dump(models, BODY_MODEL_PATH)
    joblib.dump(scaler, BODY_SCALER_PATH)
    joblib.dump(target_encoder, BODY_TARGET_PATH)
    joblib.dump(gender_map, 'models/gender_map.pkl')
    joblib.dump(feature_order, BODY_FEATURE_ORDER_PATH)   # simpan feature order
    
    return models, scaler, target_encoder, gender_map, (acc_rf, acc_xgb, acc_svm)

def load_body_performance_model():
    """Load model dan komponen, tetap mengembalikan 4 nilai (tanpa feature_order)"""
    try:
        models = joblib.load(BODY_MODEL_PATH)
        scaler = joblib.load(BODY_SCALER_PATH)
        target_encoder = joblib.load(BODY_TARGET_PATH)
        gender_map = joblib.load('models/gender_map.pkl')
        # feature_order tidak dikembalikan, akan dimuat langsung di predict_fitness_level
        return models, scaler, target_encoder, gender_map
    except Exception as e:
        print(f"⚠️ Model kebugaran belum ada atau error: {e}, melatih dari awal...")
        models, scaler, target_encoder, gender_map, _ = train_body_performance_model()
        return models, scaler, target_encoder, gender_map

def predict_fitness_level(model_name, input_data):
    """Prediksi tingkat kebugaran dengan memastikan urutan fitur sama seperti training"""
    # Load semua komponen
    models, scaler, target_encoder, gender_map = load_body_performance_model()
    # Load feature order dari file terpisah
    feature_order = joblib.load(BODY_FEATURE_ORDER_PATH)
    
    # Hitung BMI
    height_m = input_data['height_cm'] / 100
    bmi = input_data['weight_kg'] / (height_m ** 2)
    
    # Konversi gender input user ('Female'/'Male') ke format dataset ('F'/'M')
    user_gender = input_data['gender']
    if user_gender == 'Female':
        gender_code = 'F'
    elif user_gender == 'Male':
        gender_code = 'M'
    else:
        gender_code = 'F'
    gender_val = gender_map.get(gender_code, 0)
    
    # Buat dictionary dengan semua fitur
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
    
    # Buat DataFrame dengan urutan kolom sesuai feature_order
    input_df = pd.DataFrame([data_dict])[feature_order]
    
    # Scaling
    input_scaled = scaler.transform(input_df)
    
    # Prediksi
    model = models[model_name]
    pred = model.predict(input_scaled)[0]
    
    # Confidence score
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(input_scaled)[0]
        confidence = probs[pred] * 100
    else:
        confidence = 100.0
    
    # Inverse transform target ke label asli (A, B, C, D)
    class_label = target_encoder.inverse_transform([pred])[0]
    display_map = {'A': 'A (Excellent)', 'B': 'B (Good)', 'C': 'C (Average)', 'D': 'D (Poor)'}
    display = display_map.get(class_label, class_label)
    
    return display, confidence, pred