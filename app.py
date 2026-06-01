import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
import time

# Import modules
from database import *
from utils import *
from models import predict_calories_burned, train_calorie_prediction_model
from chatbot import FitnessChatbot

# Page configuration
st.set_page_config(
    page_title="Smart Fitness Tracker",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1rem;
        color: white;
    }
    .info-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        transition: 0.3s;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
init_database()

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar Navigation
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 10px 0;">
        <img src="https://img.icons8.com/?size=100&id=Gfu2ShSgiWKq&format=png&color=000000" width="70">
        <h2 style="margin-top: 10px; margin-bottom: 5px;"> Smart Fitness & Food Tracker</h2>
        <hr style="margin-top: 10px; margin-bottom: 10px;">
    </div>
    """,
    unsafe_allow_html=True
)

# Navigation menu
menu = st.sidebar.radio(
    "📋 Menu",
    ["🏠 Dashboard", "👤 Profile", "🍎 Food Log", "🏃 Activity Log", "🏋️ Fitness Level Classifier",
     "📈 Progress", "🤖 AI Chatbot", "📊 ML Predictor", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Tips:**
    - Track your meals daily
    - Log your workouts
    - Check progress weekly
    - Ask AI for advice
    """
)

# ==================== MAIN APPLICATION ====================

if menu == "🏠 Dashboard":
    st.markdown('<div class="main-header">🏠 Dashboard</div>', unsafe_allow_html=True)
    
    # Check if user exists
    user = get_user()
    if user is None:
        st.warning("⚠️ Please complete your profile first!")
        if st.button("Go to Profile"):
            st.session_state.menu = "👤 Profile"
            st.rerun()
    else:
        st.session_state.user_id = user['user_id']
        
        # Display user greeting
        st.markdown(f"### Welcome back, {user['name']}! 👋")
        
        # Today's summary
        col1, col2, col3, col4 = st.columns(4)
        
        calories_in, calories_out = get_today_summary(user['user_id'])
        net_calories = calories_in - calories_out
        target = user['daily_target_calories']
        remaining = target - net_calories
        
        with col1:
            st.metric("🔥 Calories In", f"{calories_in:.0f} kcal", 
                     delta=f"vs {target:.0f} target")
        with col2:
            st.metric("⚡ Calories Out", f"{calories_out:.0f} kcal")
        with col3:
            st.metric("📊 Net Balance", f"{net_calories:.0f} kcal",
                     delta=f"{remaining:.0f} remaining")
        with col4:
            st.metric("💪 BMI", f"{calculate_bmi(user['weight_kg'], user['height_cm']):.1f}",
                     delta=get_bmi_category(calculate_bmi(user['weight_kg'], user['height_cm'])))
        
        # Quick actions
        # st.markdown("---")
        # st.subheader("📝 Quick Actions")
        # col1, col2, col3 = st.columns(3)
        # with col1:
        #     if st.button("🍎 Quick Food Log", use_container_width=True):
        #         st.switch_page("app.py")  # Navigate to food log
        # with col2:
        #     if st.button("🏃 Quick Activity Log", use_container_width=True):
        #         st.switch_page("app.py")
        # with col3:
        #     if st.button("🤖 Ask AI Coach", use_container_width=True):
        #         st.switch_page("app.py")
        
        # Weekly calorie chart
        st.markdown("---")
        st.subheader("📊 Weekly Calorie Summary")
        
        # Get last 7 days data
        food_logs = get_food_logs(user['user_id'], 7)
        activity_logs = get_activity_logs(user['user_id'], 7)
        
        # Prepare data for chart
        dates = [(date.today() - timedelta(days=i)).isoformat() for i in range(6, -1, -1)]
        daily_in = []
        daily_out = []
        
        for d in dates:
            day_in = food_logs[food_logs['log_date'] == d]['calories'].sum() if len(food_logs) > 0 else 0
            day_out = activity_logs[activity_logs['log_date'] == d]['calories_burned'].sum() if len(activity_logs) > 0 else 0
            daily_in.append(day_in)
            daily_out.append(day_out)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Calories In', x=dates, y=daily_in, marker_color='#FF6B6B'))
        fig.add_trace(go.Bar(name='Calories Out', x=dates, y=daily_out, marker_color='#4ECDC4'))
        fig.update_layout(barmode='group', title='Daily Calorie Comparison', height=400)
        fig.update_xaxes(title='Date')
        fig.update_yaxes(title='Calories (kcal)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent activities
        st.subheader("📋 Recent Activities")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🍎 Recent Meals**")
            if len(food_logs) > 0:
                recent_meals = food_logs.head(5)
                for _, meal in recent_meals.iterrows():
                    st.markdown(f"- {meal['food_name']}: {meal['calories']:.0f} kcal")
            else:
                st.info("No meals logged today")
        
        with col2:
            st.markdown("**🏃 Recent Workouts**")
            if len(activity_logs) > 0:
                recent_activities = activity_logs.head(5)
                for _, activity in recent_activities.iterrows():
                    st.markdown(f"- {activity['activity_type']}: {activity['duration_minutes']:.0f} min, {activity['calories_burned']:.0f} kcal")
            else:
                st.info("No activities logged today")

elif menu == "👤 Profile":
    st.markdown('<div class="main-header">👤 Profile Setup</div>', unsafe_allow_html=True)
    
    user = get_user()
    
    with st.form("user_profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Name", value=user['name'] if user is not None else "")
            age = st.number_input("Age", min_value=15, max_value=100, value=int(user['age']) if user is not None else 25)
            gender = st.selectbox("Gender", ["Male", "Female"], index=0 if user is None or user['gender'] == "Male" else 1)
            height_cm = st.number_input("Height (cm)", min_value=100, max_value=250, value=int(user['height_cm']) if user is not None else 170)
        
        with col2:
            weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=float(user['weight_kg']) if user is not None else 70.0)
            activity_level = st.selectbox("Activity Level", list(ACTIVITY_MULTIPLIERS.keys()), 
                                         index=2 if user is None else list(ACTIVITY_MULTIPLIERS.keys()).index(user['activity_level']))
            fitness_goal = st.selectbox("Fitness Goal", list(FITNESS_GOALS.keys()),
                                       index=1 if user is None else list(FITNESS_GOALS.keys()).index(user['fitness_goal']))
        
        st.markdown("---")
        st.subheader("📊 Your Health Metrics")
        
        # Calculate metrics
        bmr = calculate_bmr(weight_kg, height_cm, age, gender)
        tdee = calculate_tdee(bmr, activity_level)
        daily_target = calculate_daily_target(tdee, fitness_goal)
        bmi = calculate_bmi(weight_kg, height_cm)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("BMR", f"{bmr:.0f} kcal/day", help="Basal Metabolic Rate - calories at rest")
        with col2:
            st.metric("TDEE", f"{tdee:.0f} kcal/day", help="Total Daily Energy Expenditure")
        with col3:
            st.metric("Daily Target", f"{daily_target:.0f} kcal/day")
        with col4:
            st.metric("BMI", f"{bmi:.1f}", delta=get_bmi_category(bmi))
        
        submitted = st.form_submit_button("💾 Save Profile")
        
        if submitted:
            user_data = {
                'name': name,
                'age': age,
                'gender': gender,
                'height_cm': height_cm,
                'weight_kg': weight_kg,
                'activity_level': activity_level,
                'fitness_goal': fitness_goal,
                'bmr': bmr,
                'tdee': tdee,
                'daily_target_calories': daily_target
            }
            
            if user is not None:
                user_data['user_id'] = user['user_id']
            
            user_id = save_user(user_data)
            st.session_state.user_id = user_id
            st.session_state.chatbot = FitnessChatbot(user_data)
            
            # Record initial weight
            update_weight(user_id, weight_kg)
            
            st.success("✅ Profile saved successfully!")
            st.balloons()
            time.sleep(1)
            st.rerun()

elif menu == "🍎 Food Log":
    st.markdown('<div class="main-header">🍎 Food Log</div>', unsafe_allow_html=True)
    
    user = get_user()
    if user is None:
        st.warning("⚠️ Please complete your profile first!")
        st.stop()
    
    st.session_state.user_id = user['user_id']
    
    tab1, tab2, tab3 = st.tabs(["📝 Log Food", "🍽️ Meal Suggestions", "📋 Food History"])
    
    with tab1:
        st.subheader("Add Food Entry")
        
        # Gunakan form agar submit bisa langsung memproses
        with st.form(key="food_form"):
            food_input = st.text_area(
                "Describe what you ate",
                placeholder="Examples: '2 slices of pizza', '200g chicken breast', 'banana', 'rice'",
                height=100
            )
            meal_type = st.selectbox("Meal Type", MEAL_TYPES)
            submitted = st.form_submit_button("📝 Log Food")
            
            if submitted:
                if not food_input.strip():
                    st.error("Please enter a food description.")
                else:
                    # Parsing input
                    parsed = parse_food_input(food_input)
                    if parsed['detected']:
                        try:
                            # Simpan ke database
                            add_food_log(
                                user['user_id'],
                                parsed['food_name'],
                                parsed['calories'],
                                parsed['protein'],
                                parsed['carbs'],
                                parsed['fat'],
                                meal_type,
                                date.today().isoformat()
                            )
                            st.success(f"✅ {parsed['food_name']} ({parsed['calories']:.0f} kcal) logged successfully!")
                            st.balloons()
                            # Tunggu sebentar lalu refresh halaman agar history terlihat
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to save: {str(e)}")
                    else:
                        st.warning("Could not detect food. Please specify a common food name (e.g., 'rice', 'chicken', 'apple').")
    
    with tab2:
        st.subheader("🍽️ Meal Suggestions")
        today_in, _ = get_today_summary(user['user_id'])
        remaining = user['daily_target_calories'] - today_in
        st.metric("Calories remaining today", f"{remaining:.0f} kcal")
        if remaining > 0:
            recommendations = generate_meal_recommendation(remaining)
            for rec in recommendations:
                st.markdown(f"**{rec['type']}**")
                for option in rec['options']:
                    st.markdown(f"- {option}")
        else:
            st.warning("⚠️ You've reached your daily calorie target!")
    
    with tab3:
        st.subheader("📋 Food History")
        food_logs = get_food_logs(user['user_id'], 30)
        if not food_logs.empty:
            total_calories = food_logs['calories'].sum()
            st.metric("Total calories last 30 days", f"{total_calories:.0f} kcal")
            st.dataframe(
                food_logs[['log_date', 'meal_type', 'food_name', 'calories', 'protein', 'carbs', 'fat']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No food logs yet. Start tracking your meals!")

elif menu == "🏃 Activity Log":
    st.markdown('<div class="main-header">🏃 Activity Log</div>', unsafe_allow_html=True)
    
    user = get_user()
    if user is None:
        st.warning("⚠️ Please complete your profile first!")
        st.stop()
    
    st.session_state.user_id = user['user_id']
    
    tab1, tab2 = st.tabs(["📝 Log Activity", "📋 Activity History"])
    
    with tab1:
        st.subheader("Log Your Workout")
        
        col1, col2 = st.columns(2)
        
        with col1:
            activity_type = st.selectbox("Activity Type", list(ACTIVITY_MET.keys()))
            duration_min = st.number_input("Duration (minutes)", min_value=1, max_value=300, value=30)
            intensity = st.select_slider("Intensity", options=["Low", "Medium", "High"], value="Medium")
        
        with col2:
            calories_burned = calculate_calories_burned_met(activity_type, duration_min, user['weight_kg'])
            st.metric("Estimated Calories Burned", f"{calories_burned:.0f} kcal")
            st.caption(f"Based on MET value: {ACTIVITY_MET[activity_type]} MET")
        
        if st.button("✅ Log Activity", use_container_width=True):
            add_activity_log(
                user['user_id'],
                activity_type,
                duration_min,
                calories_burned,
                intensity,
                date.today().isoformat()
            )
            st.success(f"✅ {activity_type} logged! You burned {calories_burned:.0f} calories.")
            st.balloons()
    
    with tab2:
        st.subheader("📋 Activity History")
        
        activity_logs = get_activity_logs(user['user_id'], 30)
        
        if len(activity_logs) > 0:
            total_burned = activity_logs['calories_burned'].sum()
            st.metric("Total calories burned last 30 days", f"{total_burned:.0f} kcal")
            
            st.dataframe(
                activity_logs[['log_date', 'activity_type', 'duration_minutes', 'calories_burned', 'intensity']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No activities logged yet. Start moving!")

elif menu == "🏋️ Fitness Level Classifier":
    st.markdown('<div class="main-header">🏋️ Klasifikasi Tingkat Kebugaran</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: #f0f9ff; border-left: 4px solid #4f46e5; border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem;">
        Masukkan data fisik dan hasil tes kebugaran Anda. Sistem akan memprediksi tingkat kebugaran 
        menggunakan 3 algoritma Machine Learning: <strong>Random Forest, XGBoost, dan SVM</strong>.
    </div>
    """, unsafe_allow_html=True)

    # Definisi fungsi dengan caching (indentasi rata kiri dalam blok ini)
    @st.cache_resource
    def load_fitness_models():
        from models import load_body_performance_model
        try:
            models, scaler, target_encoder, gender_map, feature_order = load_body_performance_model()
            return models, scaler, target_encoder, gender_map, feature_order
        except Exception as e:
            st.error(f"Gagal memuat model: {str(e)}")
            return None, None, None, None, None

    # Panggil fungsi
    models, scaler, target_encoder, gender_map, feature_order = load_fitness_models()

    if models is not None:
        # ... sisanya (form input, prediksi)
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("👤 Jenis Kelamin", ["Female", "Male"])
            age = st.number_input("📅 Usia (tahun)", min_value=15, max_value=100, value=25)
            height = st.number_input("📏 Tinggi Badan (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.5)
            weight = st.number_input("⚖️ Berat Badan (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
            body_fat = st.slider("🧬 Persentase Lemak Tubuh (%)", min_value=5.0, max_value=50.0, value=20.0, step=0.5)
        
        with col2:
            diastolic = st.number_input("❤️ Tekanan Darah Diastolik (mmHg)", min_value=40, max_value=120, value=80)
            systolic = st.number_input("❤️ Tekanan Darah Sistolik (mmHg)", min_value=80, max_value=200, value=120)
            gripForce = st.number_input("💪 Kekuatan Genggaman (kg)", min_value=10.0, max_value=100.0, value=40.0, step=0.5)
            sit_bend = st.number_input("🧘 Kelenturan (sit and bend) cm", min_value=-20.0, max_value=50.0, value=15.0, step=0.5)
            situps = st.number_input("🏃‍♂️ Sit-up per Menit", min_value=0, max_value=100, value=30)
            broad_jump = st.number_input("🦵 Lompatan Lebar (cm)", min_value=50, max_value=300, value=150)
        
        st.markdown("---")
        
        # Pilihan algoritma
        st.subheader("🤖 Pilih Algoritma")
        algorithm = st.selectbox(
            "Model Machine Learning",
            ["random_forest", "xgboost", "svm"],
            format_func=lambda x: {
                "random_forest": "🌲 Random Forest (Akurasi ~75%)",
                "xgboost": "⚡ XGBoost (Akurasi ~75%)",
                "svm": "🎯 SVM - Support Vector Machine (Akurasi ~70%)"
            }[x]
        )
        
        if st.button("🔍 Prediksi Tingkat Kebugaran", use_container_width=True):
            with st.spinner("Menganalisis data dengan model Machine Learning..."):
                from models import predict_fitness_level
                
                input_data = {
                    'gender': gender,
                    'age': age,
                    'height_cm': height,
                    'weight_kg': weight,
                    'body_fat': body_fat,
                    'diastolic': diastolic,
                    'systolic': systolic,
                    'gripForce': gripForce,
                    'sit_bend': sit_bend,
                    'situps': situps,
                    'broad_jump': broad_jump
                }
                
                try:
                    result, confidence, _ = predict_fitness_level(algorithm, input_data)
                    
                    # Tampilkan hasil prediksi dalam bentuk card
                    st.markdown("---")
                    st.subheader("📊 Hasil Klasifikasi")
                    
                    # Warna berdasarkan tingkat kebugaran
                    if "Excellent" in result:
                        bg_color = "#d4edda"
                        border_color = "#28a745"
                        icon = "🏆"
                    elif "Good" in result:
                        bg_color = "#d1ecf1"
                        border_color = "#17a2b8"
                        icon = "💪"
                    elif "Average" in result:
                        bg_color = "#fff3cd"
                        border_color = "#ffc107"
                        icon = "👍"
                    else:
                        bg_color = "#f8d7da"
                        border_color = "#dc3545"
                        icon = "⚠️"
                    
                    st.markdown(f"""
                    <div style="background: {bg_color}; border-left: 6px solid {border_color}; border-radius: 16px; padding: 1.5rem; margin: 1rem 0;">
                        <h3 style="margin: 0 0 0.5rem 0;">{icon} Tingkat Kebugaran: <strong>{result}</strong></h3>
                        <p style="margin: 0;">Model {algorithm.replace('_', ' ').title()} memprediksi dengan tingkat keyakinan <strong>{confidence:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Rekomendasi berdasarkan hasil
                    st.subheader("📋 Rekomendasi")
                    if "Excellent" in result:
                        st.success("🎉 **Pertahankan!** Anda dalam kondisi bugar prima. Lanjutkan rutinitas olahraga dan pola makan sehat Anda.")
                    elif "Good" in result:
                        st.info("💪 **Bagus!** Tingkat kebugaran Anda sudah baik. Tingkatkan intensitas latihan untuk mencapai level Excellent.")
                    elif "Average" in result:
                        st.warning("📈 **Mulai tingkatkan!** Coba tambahkan latihan kardio 3x seminggu dan perbaiki pola makan.")
                    else:
                        st.error("🏃‍♂️ **Ayo mulai bergerak!** Konsultasikan dengan pelatih kebugaran untuk program latihan yang sesuai dengan kondisi Anda.")
                    
                    # Tambahkan informasi tentang model
                    with st.expander("📚 Tentang Model Machine Learning"):
                        st.markdown("""
                        **Random Forest**
                        - Ensemble learning menggunakan banyak pohon keputusan
                        - Akurasi pada data test: ~75%
                        - Keunggulan: Tidak mudah overfitting, bisa handle data non-linear
                        
                        **XGBoost (Extreme Gradient Boosting)**
                        - Algoritma boosting yang sangat populer
                        - Akurasi pada data test: ~75%
                        - Keunggulan: Cepat, akurat, dan memiliki regularisasi
                        
                        **SVM (Support Vector Machine)**
                        - Mencari hyperplane terbaik untuk memisahkan kelas
                        - Akurasi pada data test: ~70%
                        - Keunggulan: Efektif untuk data berdimensi tinggi
                        """)
                        
                except Exception as e:
                    st.error(f"Error saat prediksi: {str(e)}")
    
    else:
        st.error("Gagal memuat model. Pastikan file dataset 'bodyPerformance.csv' tersedia di folder 'data/'.")

elif menu == "📈 Progress":
    st.markdown('<div class="main-header">📈 Progress Tracking</div>', unsafe_allow_html=True)
    
    user = get_user()
    if user is None:
        st.warning("⚠️ Please complete your profile first!")
        st.stop()
    
    # Weight progress
    st.subheader("⚖️ Weight Progress")
    
    weight_history = get_weight_progress(user['user_id'])
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if len(weight_history) > 0:
            fig = px.line(
                weight_history, 
                x='record_date', 
                y='weight_kg',
                title='Weight Over Time',
                labels={'record_date': 'Date', 'weight_kg': 'Weight (kg)'}
            )
            fig.update_traces(line=dict(color='#4ECDC4', width=3))
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate progress
            start_weight = weight_history.iloc[0]['weight_kg']
            latest_weight = weight_history.iloc[-1]['weight_kg']
            change = latest_weight - start_weight
            
            if change < 0:
                st.success(f"✅ You've lost {abs(change):.1f} kg since starting!")
            elif change > 0:
                st.info(f"📈 You've gained {change:.1f} kg since starting")
            else:
                st.info("Weight has remained stable")
    
    with col2:
        new_weight = st.number_input("Update weight (kg)", min_value=30.0, max_value=200.0, value=float(user['weight_kg']))
        if st.button("📝 Record Weight"):
            update_weight(user['user_id'], new_weight)
            st.success("Weight recorded!")
            st.rerun()
    
    # Calorie trend
    st.subheader("🔥 Calorie Trend Analysis")
    
    food_logs = get_food_logs(user['user_id'], 30)
    activity_logs = get_activity_logs(user['user_id'], 30)
    
    if len(food_logs) > 0 or len(activity_logs) > 0:
        # Aggregate by date
        daily_summary = {}
        for _, row in food_logs.iterrows():
            date_str = row['log_date']
            if date_str not in daily_summary:
                daily_summary[date_str] = {'in': 0, 'out': 0}
            daily_summary[date_str]['in'] += row['calories']
        
        for _, row in activity_logs.iterrows():
            date_str = row['log_date']
            if date_str not in daily_summary:
                daily_summary[date_str] = {'in': 0, 'out': 0}
            daily_summary[date_str]['out'] += row['calories_burned']
        
        # Create DataFrame for chart
        trend_data = pd.DataFrame([
            {'Date': d, 'Calories In': v['in'], 'Calories Out': v['out'], 'Net': v['in'] - v['out']}
            for d, v in sorted(daily_summary.items())
        ])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(name='Calories In', x=trend_data['Date'], y=trend_data['Calories In'], 
                                 mode='lines+markers', line=dict(color='#FF6B6B', width=2)))
        fig.add_trace(go.Scatter(name='Calories Out', x=trend_data['Date'], y=trend_data['Calories Out'], 
                                 mode='lines+markers', line=dict(color='#4ECDC4', width=2)))
        fig.add_trace(go.Scatter(name='Net', x=trend_data['Date'], y=trend_data['Net'], 
                                 mode='lines+markers', line=dict(color='#FFE66D', width=2, dash='dash')))
        fig.update_layout(title='Daily Calorie Trends', height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Log your meals and activities to see progress charts!")

elif menu == "🤖 AI Chatbot":
    st.markdown('<div class="main-header">🤖 AI Fitness Coach</div>', unsafe_allow_html=True)
    
    user = get_user()
    if user is None:
        st.warning("⚠️ Please complete your profile first!")
        st.stop()
    
    # Initialize chatbot with user data
    if st.session_state.chatbot is None:
        user_data = {
            'name': user['name'],
            'age': user['age'],
            'gender': user['gender'],
            'height_cm': user['height_cm'],
            'weight_kg': user['weight_kg'],
            'bmr': user['bmr'],
            'tdee': user['tdee'],
            'daily_target_calories': user['daily_target_calories'],
            'fitness_goal': user['fitness_goal']
        }
        st.session_state.chatbot = FitnessChatbot(user_data)
    
    # --- Greeting in Indonesian (only once) ---
    if 'greeting_sent' not in st.session_state:
        st.session_state.greeting_sent = False
    
    if not st.session_state.greeting_sent and len(st.session_state.messages) == 0:
        from datetime import datetime
        hour = datetime.now().hour
        if hour < 12:
            sapaan = "Selamat pagi 🌞"
        elif hour < 18:
            sapaan = "Selamat siang 🌤️"
        else:
            sapaan = "Selamat malam 🌙"
        
        nama = user['name'] if user['name'] else "Teman"
        tujuan = user['fitness_goal']
        if tujuan == "Weight Loss":
            pesan_tujuan = "🎯 Ayo capai target penurunan berat badanmu! Aku akan membantu dengan defisit kalori yang sehat."
        elif tujuan == "Muscle Gain":
            pesan_tujuan = "💪 Siap membentuk otot? Aku akan memandumu dengan pola makan dan latihan yang tepat."
        else:
            pesan_tujuan = "🌟 Aku di sini untuk membantumu menjaga gaya hidup sehat dan seimbang."
        
        greeting = f"""{sapaan} **{nama}**! Senang berkenalan denganmu 👋

Aku **FitBot**, asisten kebugaran dan nutrisi pribadimu.

{pesan_tujuan}

**Apa yang bisa aku bantu?**
• 🍎 Menganalisis makanan & menghitung kalori
• 🏋️ Menyusun rencana olahraga
• 📊 Memantau progres harian
• 💡 Memberi motivasi dan tips kesehatan

**Coba tanyakan ini (dalam bahasa Inggris atau Indonesia):**
- "How many calories should I eat today?"
- "Give me a quick home workout"
- "What's a healthy breakfast idea?"
- "Aku butuh motivasi!"

Aku akan merespon dalam bahasa yang kamu gunakan. Yuk mulai dengan mengetik pertanyaanmu di bawah! 😊"""
        
        st.session_state.messages.append({"role": "assistant", "content": greeting})
        st.session_state.greeting_sent = True
        try:
            save_chat_message(user['user_id'], "[System Greeting]", greeting)
        except:
            pass
        st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about fitness, nutrition, or workouts..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get today's context
        calories_in, calories_out = get_today_summary(user['user_id'])
        context = {
            'calories_in': calories_in,
            'calories_out': calories_out
        }
        
        # Get history (all messages except the current one)
        history = st.session_state.messages[:-1]
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.get_response(
                    user_message=prompt,
                    context=context,
                    history=history
                )
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        save_chat_message(user['user_id'], prompt, response)
        st.rerun()
    
    # Quick questions sidebar (remain in English as before)
    with st.sidebar:
        st.markdown("### 💡 Quick Questions")
        st.markdown("Ask me anything:")
        
        quick_questions = [
            "How do I lose weight effectively?",
            "Best foods for muscle gain?",
            "How many calories should I eat today?",
            "Give me a home workout plan",
            "How to stay motivated?",
            "Tips for better sleep",
            "What should I eat after workout?"
        ]
        
        for q in quick_questions:
            if st.button(q, key=f"quick_{q[:20]}"):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": q})
                with st.chat_message("user"):
                    st.markdown(q)
                
                # Define context inside the button action
                calories_in, calories_out = get_today_summary(user['user_id'])
                context = {
                    'calories_in': calories_in,
                    'calories_out': calories_out
                }
                history = st.session_state.messages[:-1]
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = st.session_state.chatbot.get_response(
                            user_message=q,
                            context=context,
                            history=history
                        )
                        st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                save_chat_message(user['user_id'], q, response)
                st.rerun()
                
elif menu == "📊 ML Predictor":
    st.markdown('<div class="main-header">📊 ML Calorie Burn Predictor</div>', unsafe_allow_html=True)
    st.info("This ML model predicts calories burned during exercise.")

    # Caching model agar hanya di-load sekali
    @st.cache_resource
    def load_cached_calorie_model():
        from models import load_calorie_model, train_calorie_prediction_model
        try:
            model, encoders, scaler = load_calorie_model()
            return model, encoders, scaler, None
        except:
            # Jika belum ada, latih dan simpan
            model, encoders, mae, r2 = train_calorie_prediction_model()
            return model, encoders, None, (mae, r2)

    model, encoders, scaler, training_info = load_cached_calorie_model()
    if training_info:
        st.success(f"✅ Model trained! MAE: {training_info[0]:.2f} cal, R²: {training_info[1]:.3f}")
    else:
        st.success("✅ Model loaded successfully!")

    st.markdown("---")
    st.subheader("Enter Workout Details")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 15, 100, 30)
        height = st.number_input("Height (cm)", 100, 250, 170)
        weight = st.number_input("Weight (kg)", 30, 200, 70)
    with col2:
        duration = st.number_input("Duration (min)", 5, 180, 45)
        hr = st.number_input("Heart Rate (bpm)", 60, 200, 140)
        temp = st.number_input("Body Temp (°C)", 35.0, 40.0, 37.0, 0.1)

    if st.button("🔮 Predict Calories Burned", use_container_width=True):
        with st.spinner("Predicting..."):
            from models import predict_calories_burned
            pred = predict_calories_burned(gender, age, height, weight, duration, hr, temp)
            st.markdown(f"""
            <div style="text-align:center; padding:2rem; background:linear-gradient(135deg,#4f46e5,#c084fc); border-radius:40px; color:white;">
                <h2 style="color:white;">🔥 Predicted Calories Burned</h2>
                <h1 style="font-size:4rem; margin:0;">{pred:.0f} kcal</h1>
                <p>for {duration} minutes of exercise</p>
            </div>
            """, unsafe_allow_html=True)

elif menu == "ℹ️ About":
    st.markdown('<div class="main-header">ℹ️ About This App</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 💪 Smart Fitness & Calorie Tracker Chatbot
    
    **Version:** 2.0.0
    **Tech Stack:** Python, Streamlit, SQLite, Scikit-learn, Plotly, Groq API
    
    ---
    
    ### 🎯 Key Features
    
    - **Personalized Calorie Tracking:** Calculates BMR, TDEE, and daily targets based on your profile
    - **Natural Language Food Logging:** Just describe what you ate
    - **Activity Tracking:** Log workouts with automatic calorie burn calculation (MET formula)
    - **AI-Powered Chatbot:** Get personalized fitness advice (Groq Llama 3.3 70B or rule-based fallback)
    - **Machine Learning Predictor:** Predicts calorie burn during exercise (Random Forest Regressor)
    - **Fitness Level Classifier:** Classify fitness level (A/B/C/D) using 3 ML algorithms:
        - Random Forest (74.5% accuracy)
        - XGBoost (76.4% accuracy)
        - SVM (70.1% accuracy)
    - **Progress Dashboard:** Visual charts for weight and calorie trends
    
    ---
    
    ### 📊 How It Works
    
    1. **Setup Profile:** Enter your age, gender, height, weight, activity level, and fitness goal
    2. **Track Daily:** Log your meals and activities throughout the day
    3. **Monitor Progress:** Check your dashboard to see if you're on track
    4. **Get AI Advice:** Ask the chatbot any fitness-related questions
    5. **Check Fitness Level:** Input your physical test results to get fitness classification
    
    ---
    
    ### 🔬 Formulas Used
    
    - **BMR (Mifflin-St Jeor):**
        - Male: (10 × weight) + (6.25 × height) - (5 × age) + 5
        - Female: (10 × weight) + (6.25 × height) - (5 × age) - 161
    
    - **TDEE:** BMR × Activity Multiplier
    
    - **Calories Burned (MET):** (MET × 3.5 × weight) / 200 × minutes
    
    - **Machine Learning for Calorie Prediction:** Random Forest Regressor
    
    - **Machine Learning for Fitness Classification:** Random Forest, XGBoost, SVM with StandardScaler
    
    ---
    
    ### 📊 Dataset Information
    
    - **Food Dataset:** Auto-generated with common foods and calories
    - **Exercise Dataset:** 2,000 synthetic samples for calorie prediction training
    - **Fitness Classification Dataset:** Body Performance Data (13,393 samples) from Kaggle
        - Features: age, gender, height, weight, body fat %, blood pressure, grip strength, flexibility, sit-ups, broad jump
        - Target: 4 fitness classes (A, B, C, D)
    
    ---
    
    ### 💡 Tips for Best Results
    
    - Log everything you eat and drink
    - Be consistent with your tracking
    - Update your weight weekly
    - Use the AI chatbot for personalized advice
    - Check your progress regularly to stay motivated
    - For accurate fitness classification, input honest physical test results
    
    ---
    
    Made with ❤️ for Final Project
    """)
    
    # Display dataset information (optional)
    st.subheader("📁 Dataset Status")
    col1, col2 = st.columns(2)
    with col1:
        try:
            import pandas as pd
            food_df = pd.read_csv('data/food_dataset.csv')
            st.success(f"✅ Food dataset: {len(food_df)} items")
        except:
            st.info("📝 Food dataset will be created automatically")
    with col2:
        try:
            exercise_df = pd.read_csv('data/exercise_dataset.csv')
            st.success(f"✅ Exercise dataset: {len(exercise_df)} samples")
        except:
            st.info("📝 Exercise dataset will be created automatically")
    
    # Cek dataset fitness
    try:
        body_df = pd.read_csv('data/body_performance.csv')
        st.success(f"✅ Body performance dataset: {len(body_df)} samples")
    except:
        st.warning("⚠️ Body performance dataset not found. Fitness classifier may not work.")
# Run the app
if __name__ == "__main__":
    pass