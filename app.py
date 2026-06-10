import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
import time

# Import modules
from database import *
from utils import *
from models import predict_calories_burned, predict_fitness_level
from chatbot import FitnessChatbot

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Smart Fitness Tracker",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main .block-container { padding: 1.5rem 2rem; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%); border-right: none; }
    [data-testid="stSidebar"] *:not(button) { color: #E2E8F0 !important; }
    [data-testid="metric-container"] { background: white; border: none; border-radius: 20px; padding: 1.25rem; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }
    .welcome-card { background: linear-gradient(135deg, #6366F1, #8B5CF6, #D946EF); padding: 2rem; border-radius: 28px; margin-bottom: 2rem; color: white; }
    .stButton button { background: linear-gradient(90deg, #6366F1, #8B5CF6); color: white; border: none; border-radius: 14px; height: 48px; font-weight: 600; }
    .main-header { font-size: 2rem; font-weight: 700; background: linear-gradient(135deg, #1E293B, #334155); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 1.5rem; border-bottom: 3px solid #6366F1; display: inline-block; }
</style>
""", unsafe_allow_html=True)

# ==================== INITIALIZATION ====================
init_database()

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None

# ==================== LOGIN PAGE ====================
if not st.session_state.logged_in:
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <div style="font-size: 4rem;">💪</div>
        <h1 style="background: linear-gradient(135deg, #6366F1, #8B5CF6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Smart Fitness Tracker</h1>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
    
    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
            if submitted:
                user = login_user(email, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user_id = user['user_id']
                    st.session_state.user_name = user.get('name', 'User')
                    st.rerun()
                else:
                    st.error("Email atau password salah!")
    
    with tab2:
        with st.form("register_form"):
            name = st.text_input("Nama")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm = st.text_input("Konfirmasi Password", type="password")
            submitted = st.form_submit_button("Register", use_container_width=True)
            if submitted:
                if password != confirm:
                    st.error("Password tidak cocok!")
                else:
                    user_id, success = register_user(email, password, name)
                    if success:
                        st.success("Registrasi berhasil! Silakan login.")
                    else:
                        st.error("Email sudah terdaftar!")
    st.stop()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem;">
        <div style="font-size: 2.5rem;">💪</div>
        <h3>Smart Fitness</h3>
        <p>Halo, {st.session_state.user_name}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    menu = st.radio(
        "MENU",
        ["📊 Dashboard", "🍎 Manajemen Kalori", "💪 Kebugaran", "🤖 AI Coach", "ℹ️ Tentang"],
        index=0
    )
    
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        for key in ['logged_in', 'user_id', 'user_name', 'messages', 'chatbot']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ==================== DASHBOARD ====================
if menu == "📊 Dashboard":
    st.markdown('<div class="main-header">📊 Dashboard</div>', unsafe_allow_html=True)
    
    user_data = get_user(st.session_state.user_id)
    
    # Konversi ke dictionary jika perlu
    if user_data is not None and hasattr(user_data, 'to_dict'):
        user = user_data.to_dict()
    else:
        user = user_data
    
    # Cek apakah user sudah punya data lengkap
    if user is None:
        st.warning("⚠️ Silakan lengkapi profil Anda terlebih dahulu!")
        st.stop()
    
    # Cek berat dan tinggi
    weight_val = user.get('weight_kg') if isinstance(user, dict) else getattr(user, 'weight_kg', None)
    height_val = user.get('height_cm') if isinstance(user, dict) else getattr(user, 'height_cm', None)
    
    if weight_val is None or height_val is None:
        st.warning("⚠️ Silakan lengkapi profil Anda terlebih dahulu!")
        
        with st.form("profile_form"):
            col1, col2 = st.columns(2)
            with col1:
                name_input = st.text_input("Nama", value=user.get('name', '') if isinstance(user, dict) else getattr(user, 'name', ''))
                age_input = st.number_input("Usia", min_value=15, max_value=100, value=25)
                gender_input = st.selectbox("Jenis Kelamin", ["Male", "Female"])
            with col2:
                height_input = st.number_input("Tinggi Badan (cm)", min_value=100.0, max_value=250.0, value=165.0)
                weight_input = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0, value=60.0)
                activity_input = st.selectbox("Level Aktivitas", ["Sedentary", "Light", "Moderate", "Active", "Very Active"])
            
            submitted = st.form_submit_button("Simpan Profile")
            if submitted:
                bmr = calculate_bmr(weight_input, height_input, age_input, gender_input)
                tdee = calculate_tdee(bmr, activity_input)
                target = calculate_daily_target(tdee, "Maintain")
                
                user_dict = {
                    'user_id': st.session_state.user_id,
                    'name': name_input,
                    'age': age_input,
                    'gender': gender_input,
                    'height_cm': height_input,
                    'weight_kg': weight_input,
                    'activity_level': activity_input,
                    'fitness_goal': "Maintain",
                    'bmr': bmr,
                    'tdee': tdee,
                    'daily_target_calories': target
                }
                save_user(user_dict)
                update_weight(st.session_state.user_id, weight_input)
                st.success("Profile berhasil disimpan!")
                st.rerun()
        st.stop()
    
    # Welcome Card
    name_display = user.get('name') if isinstance(user, dict) else getattr(user, 'name', 'User')
    st.markdown(f"""
    <div class="welcome-card">
        <div style="display: flex; justify-content: space-between;">
            <div>
                <div style="font-size: 1.75rem; font-weight: 700;">👋 Halo, {name_display}!</div>
                <p>Terus konsisten. Setiap pilihan sehat itu berarti.</p>
            </div>
            <div>{date.today().strftime('%A, %d %B %Y')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    calories_in, calories_out = get_today_summary(st.session_state.user_id)
    net = calories_in - calories_out
    target = user.get('daily_target_calories') if isinstance(user, dict) else getattr(user, 'daily_target_calories', 2000)
    weight = user.get('weight_kg') if isinstance(user, dict) else getattr(user, 'weight_kg', 60)
    height = user.get('height_cm') if isinstance(user, dict) else getattr(user, 'height_cm', 165)
    bmi = calculate_bmi(weight, height)
    
    with col1:
        st.metric("🍽️ Kalori Masuk", f"{calories_in:.0f} kcal")
    with col2:
        st.metric("🏃 Kalori Keluar", f"{calories_out:.0f} kcal")
    with col3:
        st.metric("⚖️ Net Balance", f"{net:.0f} kcal")
    with col4:
        st.metric("💪 BMI", f"{bmi:.1f}", delta=get_bmi_category(bmi))
    
    # Warning
    remaining = target - net
    if net > target:
        st.error(f"⚠️ Kelebihan {net - target:.0f} kcal! Kurangi porsi makan atau tambah olahraga.")
    elif net < 0:
        st.warning(f"⚠️ Defisit {abs(net):.0f} kcal! Jangan lupa makan.")
    else:
        st.success(f"✅ Sisa {remaining:.0f} kcal untuk hari ini. Pertahankan!")
    
    # Weight Progress
    st.markdown("---")
    st.subheader("📈 Progress Berat Badan")
    weight_history = get_weight_progress(st.session_state.user_id)
    if len(weight_history) > 1:
        fig = px.line(weight_history, x='record_date', y='weight_kg', markers=True)
        fig.update_traces(line=dict(color='#6366F1', width=3))
        fig.update_layout(plot_bgcolor='white', height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Belum ada data berat badan. Update berat badan Anda secara rutin.")

# ==================== MANAJEMEN KALORI ====================
elif menu == "🍎 Manajemen Kalori":
    st.markdown('<div class="main-header">🍎 Manajemen Kalori</div>', unsafe_allow_html=True)
    
    user_data = get_user(st.session_state.user_id)
    if user_data is not None and hasattr(user_data, 'to_dict'):
        user = user_data.to_dict()
    else:
        user = user_data
    
    if user is None:
        st.warning("⚠️ Silakan lengkapi profil terlebih dahulu di Dashboard!")
        st.stop()
    
    weight = user.get('weight_kg') if isinstance(user, dict) else getattr(user, 'weight_kg', None)
    if weight is None:
        st.warning("⚠️ Silakan lengkapi profil terlebih dahulu di Dashboard!")
        st.stop()
    
    tab1, tab2, tab3 = st.tabs(["📝 Catat Makanan", "🏃 Catat Aktivitas", "📊 Riwayat"])
    
    with tab1:
        st.subheader("Tambahkan Makanan")
        with st.form("food_form"):
            food = st.text_area("Apa yang kamu makan?", placeholder="Contoh: nasi goreng, ayam bakar, pisang")
            meal = st.selectbox("Waktu Makan", ["Sarapan", "Makan Siang", "Makan Malam", "Camilan"])
            submitted = st.form_submit_button("Simpan")
            if submitted and food:
                parsed = parse_food_input(food)
                if parsed['detected']:
                    add_food_log(st.session_state.user_id, parsed['food_name'], parsed['calories'],
                                parsed['protein'], parsed['carbs'], parsed['fat'],
                                meal, date.today().isoformat())
                    st.success(f"✅ {parsed['food_name']} ({parsed['calories']:.0f} kcal) tersimpan!")
                    st.rerun()
                else:
                    st.warning("Tidak dapat mendeteksi makanan. Coba sebutkan nama makanan yang umum.")
    
    with tab2:
        st.subheader("Tambahkan Aktivitas")
        with st.form("activity_form"):
            activity = st.selectbox("Jenis Olahraga", list(ACTIVITY_MET.keys()))
            duration = st.number_input("Durasi (menit)", min_value=1, max_value=300, value=30)
            submitted = st.form_submit_button("Simpan")
            if submitted:
                calories = calculate_calories_burned_met(activity, duration, weight)
                add_activity_log(st.session_state.user_id, activity, duration, calories, "Sedang", date.today().isoformat())
                st.success(f"✅ {activity} selama {duration} menit! Terbakar {calories:.0f} kcal")
                st.rerun()
    
    with tab3:
        st.subheader("Riwayat 7 Hari Terakhir")
        food_logs = get_food_logs(st.session_state.user_id, 7)
        activity_logs = get_activity_logs(st.session_state.user_id, 7)
        
        dates = [(date.today() - timedelta(days=i)).isoformat() for i in range(6, -1, -1)]
        daily_in = []
        daily_out = []
        for d in dates:
            day_in = food_logs[food_logs['log_date'] == d]['calories'].sum() if len(food_logs) > 0 else 0
            day_out = activity_logs[activity_logs['log_date'] == d]['calories_burned'].sum() if len(activity_logs) > 0 else 0
            daily_in.append(day_in)
            daily_out.append(day_out)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Kalori Masuk', x=dates, y=daily_in, marker_color='#FF6B6B'))
        fig.add_trace(go.Bar(name='Kalori Keluar', x=dates, y=daily_out, marker_color='#4ECDC4'))
        fig.update_layout(barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)

# ==================== KEBUGARAN ====================
elif menu == "💪 Kebugaran":
    st.markdown('<div class="main-header">💪 Kebugaran</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: #F0F9FF; padding: 1rem; border-radius: 16px; margin-bottom: 1rem;">
        Masukkan data tes kebugaran untuk mendapatkan klasifikasi level fitness Anda.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])
        age = st.number_input("Usia", 15, 100, 25)
        height = st.number_input("Tinggi (cm)", 100.0, 250.0, 165.0)
        weight = st.number_input("Berat (kg)", 30.0, 200.0, 60.0)
    with col2:
        situp = st.number_input("Sit-up per menit", 0, 100, 30)
        jump = st.number_input("Lompatan Lebar (cm)", 50, 300, 150)
        body_fat = st.slider("Persentase Lemak Tubuh (%)", 5.0, 50.0, 20.0)
    
    if st.button("🔍 Prediksi Level Kebugaran", use_container_width=True):
        with st.spinner("Menganalisis..."):
            try:
                input_data = {
                    'gender': gender, 'age': age, 'height_cm': height, 'weight_kg': weight,
                    'body_fat': body_fat, 'diastolic': 80, 'systolic': 120,
                    'gripForce': 40, 'sit_bend': 15, 'situps': situp, 'broad_jump': jump
                }
                result, confidence, _ = predict_fitness_level("random_forest", input_data)
                
                if "Excellent" in result:
                    st.success(f"🏆 Hasil: {result} (Keyakinan: {confidence:.1f}%) - Pertahankan!")
                elif "Good" in result:
                    st.success(f"💪 Hasil: {result} (Keyakinan: {confidence:.1f}%) - Bagus, tingkatkan!")
                elif "Average" in result:
                    st.warning(f"👍 Hasil: {result} (Keyakinan: {confidence:.1f}%) - Perbanyak olahraga!")
                else:
                    st.error(f"⚠️ Hasil: {result} - Ayo mulai bergerak lebih aktif!")
            except Exception as e:
                st.error(f"Error: {e}")

# ==================== AI COACH ====================
elif menu == "🤖 AI Coach":
    st.markdown('<div class="main-header">🤖 AI Coach</div>', unsafe_allow_html=True)
    
    user_data = get_user(st.session_state.user_id)
    if user_data is not None and hasattr(user_data, 'to_dict'):
        user = user_data.to_dict()
    else:
        user = user_data
    
    if user is None:
        st.warning("⚠️ Silakan lengkapi profil Anda terlebih dahulu di Dashboard!")
        st.stop()
    
    # Inisialisasi chatbot
    if st.session_state.chatbot is None:
        try:
            from chatbot import FitnessChatbot
            user_dict = {
                'name': user.get('name', 'User'),
                'daily_target_calories': user.get('daily_target_calories', 2000)
            }
            st.session_state.chatbot = FitnessChatbot(user_dict)
        except Exception as e:
            st.session_state.chatbot = None
    
    # Tampilkan chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Tanyakan tentang fitness, nutrisi, atau olahraga..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Sedang berpikir..."):
                try:
                    if st.session_state.chatbot:
                        response = st.session_state.chatbot.get_response(prompt, {}, [])
                    else:
                        response = "Maaf, AI Coach sedang tidak tersedia. Silakan coba lagi nanti."
                    st.markdown(response)
                except Exception as e:
                    response = f"Maaf, terjadi kesalahan. Silakan coba lagi. Detail: {e}"
                    st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# ==================== TENTANG ====================
elif menu == "ℹ️ Tentang":
    st.markdown('<div class="main-header">ℹ️ Tentang Aplikasi</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 💪 Smart Fitness Tracker
    
    Aplikasi ini membantu Anda:
    - ✅ Melacak asupan kalori harian
    - ✅ Mencatat aktivitas fisik
    - ✅ Mendapatkan klasifikasi level kebugaran
    - ✅ Konsultasi dengan AI Coach
    - ✅ Memantau progress berat badan
    
    **Teknologi:** Python, Streamlit, SQLite, Scikit-learn, Plotly
    
    ---
    
    ### Cara Penggunaan
    
    1. **Lengkapi profil** (berat, tinggi, usia)
    2. **Catat makanan** di Manajemen Kalori
    3. **Catat aktivitas** olahraga
    4. **Pantau progress** di Dashboard
    5. **Tanyakan AI Coach** untuk tips fitness
    """)

if __name__ == "__main__":
    pass