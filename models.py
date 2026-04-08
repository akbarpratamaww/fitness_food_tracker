import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

MODEL_PATH = 'models/calorie_predictor.pkl'
ENCODER_PATH = 'models/label_encoder.pkl'
SCALER_PATH = 'models/scaler.pkl'

def train_calorie_prediction_model():
    """
    Train a machine learning model to predict calories burned during exercise.
    Returns: model, label_encoders, mae, r2
    """
    try:
        df = pd.read_csv('data/exercise_dataset.csv')
    except FileNotFoundError:
        # Create sample dataset with safe ranges
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
        )
        df['Calories_Burned'] = df['Calories_Burned'].clip(lower=50, upper=1200).round()
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/exercise_dataset.csv', index=False)
    
    # Clean data
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    X = df.drop('Calories_Burned', axis=1)
    y = df['Calories_Burned']
    
    # Encode categorical
    label_encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Scale numeric features
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model trained! MAE: {mae:.2f} calories, R²: {r2:.3f}")
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoders, ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    # Return only 4 values (scaler is saved but not returned)
    return model, label_encoders, mae, r2

def load_model():
    """Load model, encoders, and scaler."""
    model = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, label_encoders, scaler

def predict_calories_burned(gender, age, height_cm, weight_kg, duration_min, heart_rate_bpm, body_temp_c):
    """Predict calories burned using the ML model."""
    model, label_encoders, scaler = load_model()
    
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