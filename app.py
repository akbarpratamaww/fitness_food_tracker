import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
import time

# Import modules
from database import *
from utils import *
from streamlit_cookies_controller import CookieController
from models import predict_calories_burned, train_calorie_prediction_model
from chatbot import FitnessChatbot
from mining import run_apriori, get_sample_transactions

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
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"], .main, h1, h2, h3, h4, h5, h6, .stMarkdown p {
        font-family: 'Poppins', sans-serif;
        background-color: transparent !important;
        color: var(--text-color) !important;
    }
    .main {
        background-color: var(--background-color) !important;
    }
    
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #FF5722 !important;
        -webkit-text-fill-color: #FF5722 !important;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: -0.5px;
    }
    
    /* Modern Glassmorphic Metric Cards */
    div[data-testid="metric-container"] {
        background: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.1) !important;
        border-radius: 18px !important;
        padding: 1.2rem 1.5rem !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px) !important;
        border-color: rgba(255, 87, 34, 0.4) !important;
        box-shadow: 0 12px 40px rgba(255, 87, 34, 0.15) !important;
    }
 
    div[data-testid="stMetricLabel"] {
        color: var(--text-color) !important;
        opacity: 0.7;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
 
    div[data-testid="stMetricValue"] {
        color: var(--text-color) !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }
 
    div[data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }
    
    /* Custom Info / Alert Boxes */
    .info-box, .success-box, .warning-box, .danger-box {
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid;
        box-shadow: 0 8px 24px rgba(0,0,0,0.05);
        background-color: var(--secondary-background-color);
    }
    
    .info-box {
        border-left-color: #06B6D4;
        color: #06B6D4;
        background: linear-gradient(90deg, rgba(6, 182, 212, 0.05) 0%, rgba(6, 182, 212, 0.01) 100%);
    }
    
    .success-box {
        border-left-color: #10B981;
        color: #10B981;
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.05) 0%, rgba(16, 185, 129, 0.01) 100%);
    }
    
    .warning-box {
        border-left-color: #F59E0B;
        color: #F59E0B;
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.05) 0%, rgba(245, 158, 11, 0.01) 100%);
    }
    
    .danger-box {
        border-left-color: #EF4444;
        color: #EF4444;
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.05) 0%, rgba(239, 68, 68, 0.01) 100%);
    }
    
    .main div.stButton > button {
        background: linear-gradient(135deg, #FF5722 0%, #FF2E93 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.8rem 2.2rem !important;
        font-weight: 800 !important;
        box-shadow: 0 6px 20px rgba(255, 46, 147, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        width: 100%;
        letter-spacing: 0.5px;
    }
    
    .main div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(255, 46, 147, 0.45) !important;
        filter: brightness(1.1) !important;
    }
 
    .main div.stButton > button:active {
        transform: translateY(1px) !important;
    }

    /* Force bold text for all buttons and their inner text elements */
    button[data-testid^="stBaseButton"], 
    button[data-testid^="stBaseButton"] *,
    div.stButton > button,
    div.stButton > button * {
        font-weight: 800 !important;
    }
    
    /* Sidebar styling and custom navigation menu */
    section[data-testid="stSidebar"] {
        background-color: var(--secondary-background-color) !important;
        border-right: 1px solid rgba(128, 128, 128, 0.1) !important;
    }
 
    section[data-testid="stSidebar"] div.stButton > button {
        background: transparent !important;
        color: var(--text-color) !important;
        opacity: 0.8;
        border: none !important;
        border-radius: 12px !important;
        text-align: left !important;
        padding: 0.7rem 1.2rem !important;
        font-weight: 500 !important;
        box-shadow: none !important;
        transition: all 0.2s ease !important;
        display: flex !important;
        justify-content: flex-start !important;
        width: 100% !important;
        margin-bottom: 0.3rem !important;
    }
    
    section[data-testid="stSidebar"] div.stButton > button:hover {
        background: rgba(255, 87, 34, 0.08) !important;
        color: #FF5722 !important;
        transform: translateX(4px) !important;
        opacity: 1;
    }
    
    /* Input & Control overrides */
    div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea, div[data-testid="stNumberInput"] input {
        background-color: var(--secondary-background-color) !important;
        color: var(--text-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.15) !important;
        border-radius: 12px !important;
        padding: 0.7rem 1rem !important;
        transition: all 0.2s ease !important;
    }
    
    div[data-testid="stTextInput"] input:focus, div[data-testid="stTextArea"] textarea:focus, div[data-testid="stNumberInput"] input:focus {
        border-color: #FF5722 !important;
        box-shadow: 0 0 0 3px rgba(255, 87, 34, 0.2) !important;
    }
 
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.15) !important;
        border-radius: 12px !important;
    }
    
    div[data-testid="stSelectbox"] div[role="button"] {
        color: var(--text-color) !important;
    }
 
    div[data-testid="stSlider"] [role="slider"] {
        background-color: #FF5722 !important;
    }
    
    div[data-testid="stSlider"] div[data-testid="stSliderTrack"] > div {
        background-color: #FF5722 !important;
    }
 
    /* Custom styled Premium Tabs */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        color: var(--text-color) !important;
        opacity: 0.7;
        border-bottom: 2px solid transparent !important;
        font-weight: 600 !important;
        padding: 0.8rem 1.8rem !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        font-size: 1rem !important;
    }
 
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #FF5722 !important;
        border-bottom-color: #FF5722 !important;
        opacity: 1;
    }
 
    /* Premium Expander Design */
    div[data-testid="stExpander"] {
        background: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.1) !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05) !important;
    }
    
    div[data-testid="stExpander"] details {
        border: none !important;
    }
 
    /* Hide Streamlit default styling elements & Sidebar completely */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    .main .block-container, [data-testid="stAppViewBlockContainer"] {
        background: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.12) !important;
        border-radius: 28px !important;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.12) !important;
        padding: 2.5rem 3rem !important;
        max-width: 1200px !important;
        margin: 90px auto 30px auto !important;
    }
   /* div[data-testid="stToolbar"] {
        display: none !important;
    } */
    footer {
        display: none !important;
    }
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Ensure parent container is on top and has no stacking issues */
    div[data-testid="element-container"]:has(div[data-testid="stRadio"]),
    div.element-container:has(div[data-testid="stRadio"]) {
        position: fixed !important;
        z-index: 999999999 !important;
        top: 0px !important;
        left: 0px !important;
        width: 100% !important;
        height: 0px !important;
        overflow: visible !important;
    }
    
    /* ── Navigation Radio Navbar ── */
    div[data-testid="stRadio"] {
        position: fixed !important;
        top: 15px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        z-index: 999999999 !important;
        background: color-mix(in srgb, var(--background-color) 70%, transparent) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        opacity: 1 !important;
        border: 1px solid rgba(128, 128, 128, 0.18) !important;
        border-radius: 16px !important;
        padding: 0.5rem 0.8rem !important;
        max-width: fit-content !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15) !important;
    }
    /* Make options sit in a horizontal flex row */
    div[data-testid="stRadio"] > div[role="radiogroup"],
    div[data-testid="stRadio"] > div:last-child {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: 6px !important;
        align-items: center !important;
        justify-content: center !important;
    }
    /* Each option wrapper */
    div[data-testid="stRadio"] label[data-baseweb="radio"] {
        padding: 0.42rem 0.85rem !important;
        border-radius: 10px !important;
        border: 1px solid rgba(128, 128, 128, 0.25) !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        white-space: nowrap !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
        background: rgba(255, 87, 34, 0.08) !important;
        border-color: rgba(255, 87, 34, 0.2) !important;
    }
    /* Hide the radio circle dot */
    div[data-testid="stRadio"] [data-testid="stRadioOption"] > div:first-child,
    div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
        display: none !important;
    }
    /* Option text colour */
    div[data-testid="stRadio"] label[data-baseweb="radio"] span,
    div[data-testid="stRadio"] label[data-baseweb="radio"] p,
    div[data-testid="stRadio"] label[data-baseweb="radio"] div,
    div[data-testid="stRadio"] label[data-baseweb="radio"] * {
        color: var(--text-color) !important;
        opacity: 1 !important;
        font-size: 0.9rem !important;
        font-weight: 800 !important;
    }
    /* Active / selected option */
    div[data-testid="stRadio"] label[data-baseweb="radio"][aria-checked="true"],
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
        background: rgba(255, 87, 34, 0.15) !important;
        border-color: rgba(255, 87, 34, 0.35) !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"][aria-checked="true"] span,
    div[data-testid="stRadio"] label[data-baseweb="radio"][aria-checked="true"] p,
    div[data-testid="stRadio"] label[data-baseweb="radio"][aria-checked="true"] div,
    div[data-testid="stRadio"] label[data-baseweb="radio"][aria-checked="true"] *,
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) span,
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p,
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) div,
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) * {
        color: #FF5722 !important;
        font-weight: 900 !important;
        opacity: 1 !important;
    }
    @media (max-width: 768px) {
        div[data-testid="stRadio"] label[data-baseweb="radio"] {
            font-size: 0.78rem !important;
            padding: 0.32rem 0.55rem !important;
        }
    }
 
 
    /* Premium dataframes styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(128, 128, 128, 0.1) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    
    /* Center horizontal radio buttons (navigation menu) */
    div[role="radiogroup"] {
        justify-content: center !important;
    }
    
    /* Responsive Styling for Mobile */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
            margin-bottom: 1.2rem;
        }
        div[data-testid="metric-container"] {
            padding: 0.8rem 1rem !important;
            margin-bottom: 0.8rem !important;
        }
        .main div.stButton > button {
            padding: 0.6rem 1.8rem !important;
            font-size: 0.95rem !important;
        }
    }
    /* Auth Page Styles */
    .auth-container {
        max-width: 480px;
        margin: 3rem auto;
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 24px;
        padding: 2.5rem 2.5rem;
        box-shadow: 0 24px 80px rgba(0,0,0,0.15);
    }
    .auth-logo {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .auth-logo h1 {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF5722 0%, #FF2E93 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .auth-logo p {
        color: var(--text-color);
        opacity: 0.7;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    .auth-divider {
        border: none;
        border-top: 1px solid rgba(128, 128, 128, 0.15);
        margin: 1.2rem 0;
    }
    
    /* Onboarding wizard custom styling via native border container */
    div[data-baseweb="tab-panel"] div[data-testid="stVerticalBlockBorderWrapper"] {
        max-width: 480px;
        margin: 1rem auto !important;
        background: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.15) !important;
        border-radius: 24px !important;
        padding: 2.2rem !important;
        box-shadow: 0 24px 80px rgba(0,0,0,0.15) !important;
    }
    
    .progress-bar-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 1.5rem;
        gap: 6px;
    }
    
    .progress-segment {
        flex: 1;
        height: 6px;
        background: rgba(128, 128, 128, 0.2);
        border-radius: 3px;
        transition: all 0.3s ease;
    }
    
    .progress-segment.active {
        background: linear-gradient(90deg, #FF5722 0%, #FF2E93 100%);
        box-shadow: 0 0 8px rgba(255, 87, 34, 0.5);
    }
    
    .wizard-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 0.5rem;
        line-height: 1.3;
    }
    
    .wizard-subtitle {
        color: var(--text-color);
        opacity: 0.7;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    
    div[data-testid="stForm"], div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.15) !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        margin-bottom: 2rem !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        overflow: visible !important;
    }
    
    div[data-testid="stVerticalBlock"] {
        overflow: visible !important;
    }
    
    div[data-testid="stForm"]:hover, div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2) !important;
        border-color: rgba(255, 87, 34, 0.3) !important;
        transform: translateY(-2px) !important;
    }
    
    .wizard-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 2rem;
    }
    
    /* Highlight the sidebar toggle button to be highly visible and interactive */
    button[data-testid="stHeaderSidebarToggle"] {
        background-color: #FF5722 !important;
        border-radius: 50% !important;
        padding: 8px !important;
        box-shadow: 0 4px 15px rgba(255, 87, 34, 0.4) !important;
        transition: all 0.3s ease !important;
        border: 1px solid rgba(128,128,128,0.2) !important;
        width: 42px !important;
        height: 42px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    button[data-testid="stHeaderSidebarToggle"]:hover {
        transform: scale(1.1) !important;
        background-color: #FF2E93 !important;
        box-shadow: 0 6px 20px rgba(255, 46, 147, 0.5) !important;
    }
    button[data-testid="stHeaderSidebarToggle"] svg {
        fill: #FFFFFF !important;
        color: #FFFFFF !important;
    }
    
    /* ── Chat Bubble Styling ── */
    [data-testid="stChatMessage"] {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.1) !important;
        border-radius: 16px !important;
        padding: 1.2rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] > p:last-child {
        margin-bottom: 0 !important;
    }

</style>
""", unsafe_allow_html=True)

# Initialize database
init_database()

# Initialize Cookie Controller
controller = CookieController()

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

# Recover session from cookie if available and user not explicitly logged out
if st.session_state.user_id is None and not st.session_state.get('logged_out', False):
    try:
        token = controller.get('user_session')
        if token:
            parts = token.split(":")
            if len(parts) == 2:
                uid, sig = parts[0], parts[1]
                if verify_user_id(uid, sig):
                    st.session_state.user_id = int(uid)
                    if st.query_params:
                        st.query_params.clear()
                    st.rerun()
    except Exception:
        pass

if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'user' not in st.session_state:
    if st.session_state.user_id is not None:
        user_row = get_user(st.session_state.user_id)
        st.session_state.user = user_row.to_dict() if user_row is not None else None
    else:
        st.session_state.user = None
if 'profile_updated' not in st.session_state:
    st.session_state.profile_updated = False
if 'greeting_sent' not in st.session_state:
    st.session_state.greeting_sent = False
if 'active_menu' not in st.session_state:
    st.session_state.active_menu = "Dashboard"

# Multi-step registration state
if 'reg_step' not in st.session_state:
    st.session_state.reg_step = 0
if 'reg_username' not in st.session_state:
    st.session_state.reg_username = ""
if 'reg_password' not in st.session_state:
    st.session_state.reg_password = ""
if 'reg_name' not in st.session_state:
    st.session_state.reg_name = ""
if 'reg_age' not in st.session_state:
    st.session_state.reg_age = 25
if 'reg_gender' not in st.session_state:
    st.session_state.reg_gender = "Male"
if 'reg_height' not in st.session_state:
    st.session_state.reg_height = 170
if 'reg_weight' not in st.session_state:
    st.session_state.reg_weight = 65.0
if 'reg_activity' not in st.session_state:
    st.session_state.reg_activity = list(ACTIVITY_MULTIPLIERS.keys())[0]
if 'reg_goal' not in st.session_state:
    st.session_state.reg_goal = list(FITNESS_GOALS.keys())[0]

# Custom food expander states
if 'expander_expanded' not in st.session_state:
    st.session_state.expander_expanded = False
if 'scroll_to_custom' not in st.session_state:
    st.session_state.scroll_to_custom = False
if 'prefilled_custom_name' not in st.session_state:
    st.session_state.prefilled_custom_name = ""

# ==================== AUTH GATE ====================
if st.session_state.user_id is None:

    st.markdown('<div class="auth-logo"><h1>💪 FitTrack AI</h1><p>Smart Fitness & Food Tracker</p></div>', unsafe_allow_html=True)

    auth_tab1, auth_tab2 = st.tabs(["🔑 Login", "📝 Register"])

    # -------- LOGIN TAB --------
    with auth_tab1:
        st.markdown('<br>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div style="text-align: center; margin-bottom: 1.5rem;"><h3 style="margin:0; color:var(--text-color);">🔑 Welcome Back</h3><p style="color:#8E8E93; font-size:0.9rem; margin:0.3rem 0 0 0;">Login to your fitness tracker</p></div>', unsafe_allow_html=True)
            with st.form("login_form"):
                login_username = st.text_input("Username", placeholder="Masukkan username kamu")
                login_password = st.text_input("Password", type="password", placeholder="Masukkan password kamu")
                login_btn = st.form_submit_button("Login", use_container_width=True)

            if login_btn:
                if not login_username or not login_password:
                    st.markdown('<div class="warning-box">⚠️ Username dan password tidak boleh kosong.</div>', unsafe_allow_html=True)
                else:
                    if not username_exists(login_username):
                        st.markdown('<div class="danger-box">❌ Username tidak ditemukan. Silakan ke tab Register untuk membuat akun.</div>', unsafe_allow_html=True)
                    else:
                        user_row = authenticate_user(login_username, login_password)
                        if user_row is not None:
                            st.session_state.user_id = int(user_row['user_id'])
                            st.session_state.user = user_row.to_dict()
                            st.session_state.logged_out = False
                            token = f"{st.session_state.user_id}:{sign_user_id(st.session_state.user_id)}"
                            controller.set('user_session', token)
                            if st.query_params:
                                st.query_params.clear()
                            st.session_state.active_menu = "Dashboard"
                            st.session_state.chatbot = None
                            st.session_state.messages = []
                            st.session_state.greeting_sent = False
                            st.success(f"✅ Selamat datang kembali, **{user_row['name']}**! 🎉")
                            time.sleep(0.8)
                            st.rerun()
                        else:
                            st.markdown('<div class="danger-box">❌ Password salah. Silakan coba lagi.</div>', unsafe_allow_html=True)

    # -------- REGISTER TAB --------
    with auth_tab2:
        st.markdown('<br>', unsafe_allow_html=True)
        with st.container(border=True):
            if st.session_state.reg_step == 0:
                st.markdown('<div class="wizard-title">💪 Welcome to FitTrack AI</div>', unsafe_allow_html=True)
                st.markdown('<div class="wizard-subtitle">Ready for some wins? Start tracking, it\'s easy! Let\'s customize FitTrack AI for your goals.</div>', unsafe_allow_html=True)
                
                st.markdown("""
                <div style="background-color: rgba(255, 87, 34, 0.05); padding: 1.2rem; border-radius: 16px; border-left: 4px solid #FF5722; font-size: 0.9rem; line-height: 1.5; margin-bottom: 2rem;">
                    Menghitung TDEE, BMR, BMI, serta rekomendasi nutrisi harian kamu berdasarkan metode kebugaran yang terbukti secara ilmiah.
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Get Started", use_container_width=True, key="start_btn"):
                    st.session_state.reg_step = 1
                    st.rerun()

            elif st.session_state.reg_step == 1:
                st.markdown('<div class="progress-bar-container"><div class="progress-segment active"></div><div class="progress-segment"></div><div class="progress-segment"></div><div class="progress-segment"></div><div class="progress-segment"></div></div>', unsafe_allow_html=True)
                st.markdown('<div class="wizard-title">👤 Buat Akun Baru</div>', unsafe_allow_html=True)
                st.markdown('<div class="wizard-subtitle">Masukkan username dan password unik Anda.</div>', unsafe_allow_html=True)
                
                reg_username = st.text_input("Username*", value=st.session_state.reg_username, placeholder="Buat username unikmu")
                reg_password = st.text_input("Password*", type="password", value=st.session_state.reg_password, placeholder="Min. 6 karakter")
                reg_password2 = st.text_input("Konfirmasi Password*", type="password", placeholder="Ulangi password")
                
                st.markdown('<div class="wizard-nav">', unsafe_allow_html=True)
                col1, col2 = st.columns([1, 2])
                with col1:
                    if st.button("Back", key="back_1", use_container_width=True):
                        st.session_state.reg_step = 0
                        st.rerun()
                with col2:
                    if st.button("Next", key="next_1", use_container_width=True):
                        if not reg_username or len(reg_username) < 3:
                            st.error("❌ Username minimal 3 karakter.")
                        elif username_exists(reg_username):
                            st.error("❌ Username sudah digunakan, silakan pilih username lain.")
                        elif not reg_password or len(reg_password) < 6:
                            st.error("❌ Password minimal 6 karakter.")
                        elif reg_password != reg_password2:
                            st.error("❌ Konfirmasi password tidak cocok.")
                        else:
                            st.session_state.reg_username = reg_username
                            st.session_state.reg_password = reg_password
                            st.session_state.reg_step = 2
                            st.rerun()

            elif st.session_state.reg_step == 2:
                st.markdown('<div class="progress-bar-container"><div class="progress-segment active"></div><div class="progress-segment active"></div><div class="progress-segment"></div><div class="progress-segment"></div><div class="progress-segment"></div></div>', unsafe_allow_html=True)
                st.markdown('<div class="wizard-title">🙋‍♂️ Siapa Nama Anda?</div>', unsafe_allow_html=True)
                st.markdown('<div class="wizard-subtitle">Beritahu kami sedikit informasi dasar tentang diri Anda.</div>', unsafe_allow_html=True)
                
                reg_name = st.text_input("Nama Lengkap*", value=st.session_state.reg_name, placeholder="Nama kamu")
                reg_age = st.number_input("Usia (tahun)*", min_value=15, max_value=100, value=int(st.session_state.reg_age))
                reg_gender = st.selectbox("Jenis Kelamin*", ["Male", "Female"], index=0 if st.session_state.reg_gender == "Male" else 1)
                
                st.markdown('<div class="wizard-nav">', unsafe_allow_html=True)
                col1, col2 = st.columns([1, 2])
                with col1:
                    if st.button("Back", key="back_2", use_container_width=True):
                        st.session_state.reg_step = 1
                        st.rerun()
                with col2:
                    if st.button("Next", key="next_2", use_container_width=True):
                        if not reg_name.strip():
                            st.error("❌ Nama lengkap wajib diisi.")
                        else:
                            st.session_state.reg_name = reg_name
                            st.session_state.reg_age = reg_age
                            st.session_state.reg_gender = reg_gender
                            st.session_state.reg_step = 3
                            st.rerun()

            elif st.session_state.reg_step == 3:
                st.markdown('<div class="progress-bar-container"><div class="progress-segment active"></div><div class="progress-segment active"></div><div class="progress-segment active"></div><div class="progress-segment"></div><div class="progress-segment"></div></div>', unsafe_allow_html=True)
                st.markdown('<div class="wizard-title">📏 Ukuran Tubuh Anda</div>', unsafe_allow_html=True)
                st.markdown('<div class="wizard-subtitle">Masukkan tinggi dan berat badan terbaru Anda untuk menghitung BMI secara akurat.</div>', unsafe_allow_html=True)
                
                reg_height = st.number_input("Tinggi Badan (cm)*", min_value=100, max_value=250, value=int(st.session_state.reg_height))
                reg_weight = st.number_input("Berat Badan (kg)*", min_value=30.0, max_value=200.0, value=float(st.session_state.reg_weight), step=0.5)
                
                st.markdown('<div class="wizard-nav">', unsafe_allow_html=True)
                col1, col2 = st.columns([1, 2])
                with col1:
                    if st.button("Back", key="back_3", use_container_width=True):
                        st.session_state.reg_step = 2
                        st.rerun()
                with col2:
                    if st.button("Next", key="next_3", use_container_width=True):
                        st.session_state.reg_height = reg_height
                        st.session_state.reg_weight = reg_weight
                        st.session_state.reg_step = 4
                        st.rerun()

            elif st.session_state.reg_step == 4:
                st.markdown('<div class="progress-bar-container"><div class="progress-segment active"></div><div class="progress-segment active"></div><div class="progress-segment active"></div><div class="progress-segment active"></div><div class="progress-segment"></div></div>', unsafe_allow_html=True)
                st.markdown('<div class="wizard-title">🏃‍♂️ Tingkat Aktivitas & Tujuan</div>', unsafe_allow_html=True)
                st.markdown('<div class="wizard-subtitle">Sesuaikan target energi berdasarkan tingkat aktivitas fisik dan goal Anda.</div>', unsafe_allow_html=True)
                
                reg_activity = st.selectbox("Tingkat Aktivitas*", list(ACTIVITY_MULTIPLIERS.keys()), index=list(ACTIVITY_MULTIPLIERS.keys()).index(st.session_state.reg_activity))
                reg_goal = st.selectbox("Tujuan Kebugaran*", list(FITNESS_GOALS.keys()), index=list(FITNESS_GOALS.keys()).index(st.session_state.reg_goal))
                
                st.markdown('<div class="wizard-nav">', unsafe_allow_html=True)
                col1, col2 = st.columns([1, 2])
                with col1:
                    if st.button("Back", key="back_4", use_container_width=True):
                        st.session_state.reg_step = 3
                        st.rerun()
                with col2:
                    if st.button("Next", key="next_4", use_container_width=True):
                        st.session_state.reg_activity = reg_activity
                        st.session_state.reg_goal = reg_goal
                        st.session_state.reg_step = 5
                        st.rerun()

            elif st.session_state.reg_step == 5:
                st.markdown('<div class="progress-bar-container"><div class="progress-segment active"></div><div class="progress-segment active"></div><div class="progress-segment active"></div><div class="progress-segment active"></div><div class="progress-segment active"></div></div>', unsafe_allow_html=True)
                st.markdown('<div class="wizard-title">🎉 Analisis & Ringkasan Profil</div>', unsafe_allow_html=True)
                st.markdown('<div class="wizard-subtitle">Berikut estimasi perhitungan kesehatan Anda. Klik tombol di bawah untuk membuat akun!</div>', unsafe_allow_html=True)
                
                bmr = calculate_bmr(st.session_state.reg_weight, st.session_state.reg_height, st.session_state.reg_age, st.session_state.reg_gender)
                tdee = calculate_tdee(bmr, st.session_state.reg_activity)
                daily_target = calculate_daily_target(tdee, st.session_state.reg_goal)
                bmi = calculate_bmi(st.session_state.reg_weight, st.session_state.reg_height)
                bmi_cat = get_bmi_category(bmi)
                
                st.markdown(f"""
                <div style="background: var(--secondary-background-color); border: 1px solid rgba(128,128,128,0.15); padding: 1.2rem; border-radius: 16px; margin-bottom: 1.5rem;">
                    <h5 style="margin-top:0; color:#FF5722;">📊 Hasil Analisis Tubuh</h5>
                    <table style="width:100%; font-size:0.9rem; border-collapse: collapse;">
                        <tr style="border-bottom: 1px solid rgba(128,128,128,0.15);"><td style="padding:6px 0; color:#8E8E93;">BMR</td><td style="padding:6px 0; text-align:right; font-weight:600; color:var(--text-color);">{bmr:.0f} kcal/day</td></tr>
                        <tr style="border-bottom: 1px solid rgba(128,128,128,0.15);"><td style="padding:6px 0; color:#8E8E93;">TDEE</td><td style="padding:6px 0; text-align:right; font-weight:600; color:var(--text-color);">{tdee:.0f} kcal/day</td></tr>
                        <tr style="border-bottom: 1px solid rgba(128,128,128,0.15);"><td style="padding:6px 0; color:#8E8E93;">Target Kalori</td><td style="padding:6px 0; text-align:right; font-weight:600; color:#FF2E93;">{daily_target:.0f} kcal/day</td></tr>
                        <tr><td style="padding:6px 0; color:#8E8E93;">BMI (IMT)</td><td style="padding:6px 0; text-align:right; font-weight:600; color:#06B6D4;">{bmi:.1f} ({bmi_cat})</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="wizard-nav">', unsafe_allow_html=True)
                col1, col2 = st.columns([1, 2])
                with col1:
                    if st.button("Back", key="back_5", use_container_width=True):
                        st.session_state.reg_step = 4
                        st.rerun()
                with col2:
                    if st.button("Buat Akun & Mulai!", key="submit_btn", use_container_width=True):
                        result = register_user(st.session_state.reg_username, st.session_state.reg_password, {
                            'name': st.session_state.reg_name, 
                            'age': st.session_state.reg_age, 
                            'gender': st.session_state.reg_gender,
                            'height_cm': st.session_state.reg_height, 
                            'weight_kg': st.session_state.reg_weight,
                            'activity_level': st.session_state.reg_activity, 
                            'fitness_goal': st.session_state.reg_goal,
                            'bmr': bmr, 
                            'tdee': tdee, 
                            'daily_target_calories': daily_target
                        })
                        
                        if result['success']:
                            new_uid = result['user_id']
                            update_weight(new_uid, st.session_state.reg_weight)
                            user_row = get_user(new_uid)
                            st.session_state.user_id = new_uid
                            st.session_state.user = user_row.to_dict()
                            st.session_state.logged_out = False
                            token = f"{new_uid}:{sign_user_id(new_uid)}"
                            controller.set('user_session', token)
                            if st.query_params:
                                st.query_params.clear()
                            st.session_state.active_menu = "Dashboard"
                            st.session_state.chatbot = None
                            st.session_state.messages = []
                            st.session_state.greeting_sent = False
                            st.session_state.reg_step = 0
                            st.success(f"🎉 Akun berhasil dibuat! Selamat datang, **{st.session_state.reg_name}**!")
                            st.balloons()
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ {result['error']}")

    st.stop()  # Hentikan render konten di bawah jika belum login

# ==================== NAVIGATION & TOP NAVBAR ====================

# Ambil data user dari session state
logged_user = st.session_state.user
if logged_user is None:
    logged_user_row = get_user(st.session_state.user_id)
    if logged_user_row is not None:
        logged_user = logged_user_row.to_dict()
        st.session_state.user = logged_user

menu_options = [
    "Dashboard", "Profile", "Food Log", "Activity Log", "Fitness Level Classifier",
    "AI Chatbot", "About"
]

# ── Single safe navigation: st.radio() reads value directly, no on_change callbacks ──
current_idx = menu_options.index(st.session_state.active_menu) if st.session_state.active_menu in menu_options else 0
selected_menu = st.radio(
    "Navigation",
    menu_options,
    index=current_idx,
    horizontal=True,
    key="main_nav_radio",
    label_visibility="collapsed"
)
# Only update active_menu if user explicitly changed the radio (not during form submits)
if selected_menu != st.session_state.active_menu:
    st.session_state.active_menu = selected_menu
    st.rerun()

menu = st.session_state.active_menu

# ==================== MAIN APPLICATION ====================

if menu == "Dashboard":
    st.markdown('<div class="main-header">Dashboard</div>', unsafe_allow_html=True)
    
    # Use stored user object if available, otherwise fetch from DB
    if st.session_state.user is not None:
        user = st.session_state.user
    elif st.session_state.user_id is not None:
        user = get_user(st.session_state.user_id)
        if user is not None:
            st.session_state.user = user
    else:
        user = None
    
    if user is None:
        st.markdown('''
        <div class="warning-box">
            <strong>⚠️ Profil Belum Lengkap!</strong><br>Silakan lengkapi profil Anda terlebih dahulu di menu 👤 Profile untuk mengaktifkan seluruh fitur tracker.
        </div>
        ''', unsafe_allow_html=True)
        if st.button("Go to Profile", key="dashboard_go_profile"):
            st.session_state.active_menu = "Profile"
            st.rerun()
    else:
        # Ensure session_state is synced
        st.session_state.user_id = user['user_id']
        st.session_state.user = user
        
        dash_col1, dash_col2 = st.columns([5, 1])
        with dash_col1:
            st.markdown(f"### Welcome back, {user['name']}! 👋")
        with dash_col2:
            if st.button("Logout", key="dash_logout_btn", use_container_width=True):
                controller.remove('user_session')
                st.session_state.logged_out = True
                st.session_state.user_id = None
                st.query_params.clear()
                for key in list(st.session_state.keys()):
                    if key != 'logged_out':
                        del st.session_state[key]
                time.sleep(0.5)
                st.rerun()
        
        # Quick terminology guide for laypeople
        with st.expander("ℹ️ Panduan Singkat Istilah Kesehatan (BMR, TDEE, BMI, Defisit/Surplus)"):
            st.markdown("""
            Jika kamu baru di dunia kebugaran, berikut adalah penjelasan singkat istilah yang digunakan di dashboard ini:
            *   **BMR (Basal Metabolic Rate):** Jumlah kalori minimum yang dibakar tubuhmu hanya untuk bernapas dan bertahan hidup saat rebahan seharian.
            *   **TDEE (Total Daily Energy Expenditure):** Total kalori yang kamu bakar dalam sehari setelah ditambah aktivitas fisik (jalan, kerja, olahraga).
            *   **BMI (Body Mass Index):** Angka untuk melihat apakah tubuhmu kurus, ideal, gemuk, atau obesitas berdasarkan tinggi & berat badan.
            *   **Defisit Kalori (Net Balance Negatif):** Makan lebih sedikit dari yang kamu bakar. Tubuh akan membakar lemak untuk energi (menurunkan berat badan).
            *   **Surplus Kalori (Net Balance Positif):** Makan lebih banyak dari yang kamu bakar. Berguna untuk menaikkan berat badan atau membangun otot.
            """)
        
        # Today's summary & Alerts container card
        with st.container(border=True):
            st.markdown("### 📊 Ringkasan Aktivitas & Nutrisi Hari Ini")
            col1, col2, col3, col4 = st.columns(4)
            
            calories_in, calories_out = get_today_summary(user['user_id'])
            net_calories = calories_in - calories_out
            target = user['daily_target_calories']
            remaining = target - net_calories
            
            with col1:
                st.metric(
                    "🔥 Calories In", 
                    f"{calories_in:.0f} kcal", 
                    delta=f"vs {target:.0f} target",
                    help="Total kalori dari makanan dan minuman yang kamu konsumsi hari ini."
                )
            with col2:
                st.metric(
                    "⚡ Calories Out", 
                    f"{calories_out:.0f} kcal",
                    help="Total kalori yang dibakar hari ini melalui metabolisme tubuh (TDEE) ditambah aktivitas olahraga."
                )
            with col3:
                st.metric(
                    "📊 Net Balance", 
                    f"{net_calories:.0f} kcal",
                    delta=f"{remaining:.0f} remaining",
                    help="Kalori Masuk dikurangi Kalori Keluar. Nilai negatif (-) berarti kamu berada dalam kondisi defisit kalori (bagus untuk menurunkan berat badan)."
                )
            with col4:
                st.metric(
                    "💪 BMI", 
                    f"{calculate_bmi(user['weight_kg'], user['height_cm']):.1f}",
                    delta=get_bmi_category(calculate_bmi(user['weight_kg'], user['height_cm'])),
                    help="Body Mass Index (Indeks Massa Tubuh) adalah pengukuran lemak tubuh berdasarkan tinggi dan berat badan untuk menentukan kategori berat badan ideal."
                )
            
            # Daily Calorie Warnings/Alerts (Dynamic based on Net Calories)
            if net_calories > target:
                exceeded_by = net_calories - target
                st.markdown(f'''
                <div class="danger-box">
                    <strong>⚠️ Batas Kalori Bersih Terlampaui!</strong><br>
                    Kalori bersih Anda hari ini ({net_calories:.0f} kcal) telah melebihi target harian Anda ({target:.0f} kcal) sebanyak <strong>{exceeded_by:.0f} kcal</strong>. 
                    Pertimbangkan untuk berolahraga lebih banyak atau mengurangi porsi makan.
                </div>
                ''', unsafe_allow_html=True)
            elif net_calories <= 0 and calories_in > 0:
                st.markdown(f'''
                <div class="info-box">
                    <strong>⚡ Kalori Terbakar Lebih Banyak!</strong><br>
                    Hebat! Total kalori yang Anda bakar hari ini lebih besar daripada kalori masuk (Net Balance: <strong>{net_calories:.0f} kcal</strong>). Ini sangat baik untuk pembakaran lemak!
                </div>
                ''', unsafe_allow_html=True)
            elif net_calories > 0 and net_calories < (target * 0.7):
                remaining_percentage = ((target - net_calories) / target) * 100
                st.markdown(f'''
                <div class="warning-box">
                    <strong>⚠️ Net Kalori Masih Kurang!</strong><br>
                    Net kalori Anda saat ini adalah {net_calories:.0f} kcal, masih kurang <strong>{target - net_calories:.0f} kcal</strong> ({remaining_percentage:.0f}% lagi) untuk memenuhi target harian Anda ({target:.0f} kcal). 
                    Pastikan Anda makan dengan cukup agar tubuh memiliki energi yang cukup.
                </div>
                ''', unsafe_allow_html=True)
            elif net_calories >= (target * 0.95) and net_calories <= target:
                st.markdown(f'''
                <div class="success-box">
                    <strong>✅ Target Net Kalori Hampir Tercapai!</strong><br>
                    Kerja bagus! Net kalori Anda ({net_calories:.0f} kcal) sudah sangat mendekati target harian Anda ({target:.0f} kcal). Pertahankan!
                </div>
                ''', unsafe_allow_html=True)

        # Button to reset daily calories
        col_btn1, col_btn2 = st.columns([5, 1.2])
        with col_btn2:
            if st.button("Reset Hari Ini", key="reset_daily_btn", use_container_width=True):
                reset_today_logs(user['user_id'])
                st.toast("✅ Log makanan & aktivitas hari ini berhasil di-reset!")
                time.sleep(0.6)
                st.rerun()

        # ==================== ROW 1: CALORIE SUMMARY & TRENDS ====================
        with st.container(border=True):
            st.markdown("### 📈 Grafik Analisis Kalori (7 Hari & 30 Hari)")
            col_r1a, col_r1b = st.columns(2)
            
            with col_r1a:
                st.subheader("📊 Weekly Calorie Summary")
                food_logs = get_food_logs(user['user_id'], 7)
                activity_logs = get_activity_logs(user['user_id'], 7)
                
                dates = [(date.today() - timedelta(days=i)).isoformat() for i in range(6, -1, -1)]
                daily_in = []
                daily_out = []
                
                for d in dates:
                    day_in = food_logs[food_logs['log_date'] == d]['calories'].sum() if len(food_logs) > 0 else 0
                    day_out = activity_logs[activity_logs['log_date'] == d]['calories_burned'].sum() if len(activity_logs) > 0 else 0
                    daily_in.append(day_in)
                    daily_out.append(day_out)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Calories In', x=dates, y=daily_in, marker_color='#FF5722'))
                fig.add_trace(go.Bar(name='Calories Out', x=dates, y=daily_out, marker_color='#06B6D4'))
                fig.update_layout(
                    barmode='group', 
                    title='Daily Calorie Comparison (7 Days)', 
                    height=350,
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Poppins, sans-serif', color='#E0E0E0'),
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                fig.update_xaxes(title='Date', gridcolor='rgba(255,255,255,0.05)')
                fig.update_yaxes(title='Calories (kcal)', gridcolor='rgba(255,255,255,0.05)')
                st.plotly_chart(fig, use_container_width=True)
                
            with col_r1b:
                st.subheader("🔥 Calorie Trend Analysis")
                food_logs_30 = get_food_logs(user['user_id'], 30)
                activity_logs_30 = get_activity_logs(user['user_id'], 30)
                
                if len(food_logs_30) > 0 or len(activity_logs_30) > 0:
                    daily_summary = {}
                    for _, row in food_logs_30.iterrows():
                        date_str = row['log_date']
                        if date_str not in daily_summary:
                            daily_summary[date_str] = {'in': 0, 'out': 0}
                        daily_summary[date_str]['in'] += row['calories']
                    
                    for _, row in activity_logs_30.iterrows():
                        date_str = row['log_date']
                        if date_str not in daily_summary:
                            daily_summary[date_str] = {'in': 0, 'out': 0}
                        daily_summary[date_str]['out'] += row['calories_burned']
                    
                    trend_data = pd.DataFrame([
                        {'Date': d, 'Calories In': v['in'], 'Calories Out': v['out'], 'Net': v['in'] - v['out']}
                        for d, v in sorted(daily_summary.items())
                    ])
                    
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(name='Calories In', x=trend_data['Date'], y=trend_data['Calories In'], 
                                             mode='lines+markers', line=dict(color='#FF5722', width=2.5)))
                    fig_trend.add_trace(go.Scatter(name='Calories Out', x=trend_data['Date'], y=trend_data['Calories Out'], 
                                             mode='lines+markers', line=dict(color='#06B6D4', width=2.5)))
                    fig_trend.add_trace(go.Scatter(name='Net', x=trend_data['Date'], y=trend_data['Net'], 
                                             mode='lines+markers', line=dict(color='#8B5CF6', width=2, dash='dash')))
                    fig_trend.update_layout(
                        title='Daily Calorie Trends (30 Days)', 
                        height=350, 
                        hovermode='x unified',
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Poppins, sans-serif', color='#E0E0E0'),
                        margin=dict(l=20, r=20, t=50, b=20)
                    )
                    fig_trend.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
                    fig_trend.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.info("Log your meals and activities to see 30-day calorie trend charts!")
        
        # ==================== ROW 2: WEIGHT PROGRESS & FORECASTING ====================
        with st.container(border=True):
            st.markdown("### ⚖️ Progress & Prediksi Berat Badan")
            weight_history = get_weight_progress(user['user_id'])
            if len(weight_history) > 0:
                weight_history['record_date'] = pd.to_datetime(weight_history['record_date']).dt.strftime('%Y-%m-%d')
            
            # Sub-columns inside weight progress to fit Update Weight form neatly
            col_inner1, col_inner2 = st.columns([2, 1])
            with col_inner1:
                if len(weight_history) > 0:
                    fig_w = px.line(
                        weight_history, 
                        x='record_date', 
                        y='weight_kg',
                        title='Weight Over Time',
                        labels={'record_date': 'Date', 'weight_kg': 'Weight (kg)'}
                    )
                    fig_w.update_traces(line=dict(color='#FF5722', width=3))
                    fig_w.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Poppins, sans-serif', color='#E0E0E0'),
                        margin=dict(l=20, r=20, t=40, b=20),
                        height=280
                    )
                    fig_w.update_xaxes(type='category', gridcolor='rgba(255,255,255,0.05)')
                    fig_w.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
                    st.plotly_chart(fig_w, use_container_width=True)
                    
                    start_weight = weight_history.iloc[0]['weight_kg']
                    latest_weight = weight_history.iloc[-1]['weight_kg']
                    change = latest_weight - start_weight
                    
                    if change < 0:
                        st.success(f"✅ Lost {abs(change):.1f} kg since start!")
                    elif change > 0:
                        st.info(f"📈 Gained {change:.1f} kg since start")
                    else:
                        st.info("Weight is stable")
                else:
                    st.info("No weight history found.")
                    
            with col_inner2:
                st.markdown("**Update Weight**")
                new_weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=float(user['weight_kg']), key="dash_update_weight_input")
                if st.button("Record Weight", key="dash_record_weight_btn"):
                    update_weight(user['user_id'], new_weight)
                    st.success("Recorded!")
                    st.rerun()
            
            # ==================== ROW 2.5: WEIGHT FORECASTING ====================
            st.markdown("---")
            st.markdown("#### 🔮 Weight Forecasting (ML)")
            forecast_col1, forecast_col2 = st.columns([1, 2])
            with forecast_col1:
                forecast_days = st.selectbox(
                    "Forecast Range",
                    options=[7, 14, 30],
                    format_func=lambda x: f"{x} days",
                    index=2,
                    key="dash_forecast_horizon",
                )
            
            result_fc = forecast_weight(user['user_id'], days=forecast_days)
            if result_fc['enough_data']:
                forecast_df = result_fc['forecast']
                hist_df = result_fc['history'].copy()
                coef = result_fc['model_coef']
                r2 = result_fc['model_r2']
                
                # Forecasting Metrics
                fm1, fm2, fm3 = st.columns(3)
                with fm1:
                    direction = "📉 Down" if coef < 0 else ("📈 Up" if coef > 0 else "➡️ Stable")
                    st.metric("Trend", f"{abs(coef):.3f} kg/d", delta=direction)
                with fm2:
                    st.metric("R² Score", f"{r2:.2f}")
                with fm3:
                    pred_end = forecast_df.iloc[-1]['predicted_weight_kg']
                    st.metric("Forecast", f"{pred_end:.1f} kg")
                
                fig_fc = go.Figure()
                hist_df['record_date'] = pd.to_datetime(hist_df['record_date'])
                fig_fc.add_trace(go.Scatter(
                    x=hist_df['record_date'],
                    y=hist_df['weight_kg'],
                    mode='lines+markers',
                    name='History',
                    line=dict(color='#4ECDC4', width=2.5),
                    marker=dict(size=6),
                ))
                bridge_dates = [
                    hist_df['record_date'].iloc[-1],
                    forecast_df['date'].iloc[0]
                ]
                bridge_weights = [
                    hist_df['weight_kg'].iloc[-1],
                    forecast_df['predicted_weight_kg'].iloc[0]
                ]
                fig_fc.add_trace(go.Scatter(
                    x=bridge_dates,
                    y=bridge_weights,
                    mode='lines',
                    line=dict(color='#FF6B6B', width=1.5, dash='dot'),
                    showlegend=False,
                ))
                fig_fc.add_trace(go.Scatter(
                    x=forecast_df['date'],
                    y=forecast_df['predicted_weight_kg'],
                    mode='lines+markers',
                    name=f'Forecast',
                    line=dict(color='#FF6B6B', width=2.5, dash='dot'),
                    marker=dict(size=5, symbol='diamond'),
                ))
                fig_fc.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Poppins, sans-serif', color='#E0E0E0'),
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=200,
                    hovermode='x unified',
                    showlegend=False
                )
                st.plotly_chart(fig_fc, use_container_width=True)
            else:
                st.info(
                    "⚠️ Minimum 2 weight logs required to predict trend."
                )
                
        # ==================== ROW 3: RECENT ACTIVITIES ====================
        with st.container(border=True):
            st.markdown("### 📋 Riwayat Aktivitas & Makanan Hari Ini")
            col_r3a, col_r3b = st.columns(2)
            
            with col_r3a:
                st.markdown("**🍎 Recent Meals**")
                if len(food_logs) > 0:
                    recent_meals = food_logs.head(5)
                    for _, meal in recent_meals.iterrows():
                        st.markdown(f"- {meal['food_name']}: {meal['calories']:.0f} kcal")
                else:
                    st.info("No meals logged today")
            
            with col_r3b:
                st.markdown("**🏃 Recent Workouts**")
                if len(activity_logs) > 0:
                    recent_activities = activity_logs.head(5)
                    for _, activity in recent_activities.iterrows():
                        st.markdown(f"- {activity['activity_type']}: {activity['duration_minutes']:.0f} min, {activity['calories_burned']:.0f} kcal")
                else:
                    st.info("No activities logged today")

elif menu == "Profile":
    st.markdown('<div class="main-header">Profile Setup</div>', unsafe_allow_html=True)
    
    # Use session state user if available, otherwise fetch from DB
    if st.session_state.user is not None:
        user = st.session_state.user
    elif st.session_state.user_id is not None:
        user = get_user(st.session_state.user_id)
        if user is not None:
            st.session_state.user = user
    else:
        user = None
    
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
        st.subheader("📊 Your Health Metrics (Hasil Analisis Tubuhmu)")
        st.markdown("""
        <div style="background-color: rgba(102, 126, 234, 0.05); padding: 10px; border-radius: 8px; border-left: 4px solid #667eea; margin-bottom: 15px; font-size: 0.9rem;">
            Analisis di bawah dihitung otomatis berdasarkan berat, tinggi, usia, gender, dan tingkat aktivitasmu. 
            Gunakan angka-angka ini sebagai acuan porsi makan harianmu!
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate metrics based on current form inputs
        bmr = calculate_bmr(weight_kg, height_cm, age, gender)
        tdee = calculate_tdee(bmr, activity_level)
        daily_target = calculate_daily_target(tdee, fitness_goal)
        bmi = calculate_bmi(weight_kg, height_cm)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "BMR", 
                f"{bmr:.0f} kcal/day", 
                help="Basal Metabolic Rate: Kalori yang dibakar organ tubuh secara otomatis (jantung, paru-paru) saat kamu tidak melakukan apa pun."
            )
        with col2:
            st.metric(
                "TDEE", 
                f"{tdee:.0f} kcal/day", 
                help="Total Daily Energy Expenditure: Total kebutuhan kalori harianmu setelah memperhitungkan tingkat aktivitas fisik pilihanmu."
            )
        with col3:
            st.metric(
                "Daily Target", 
                f"{daily_target:.0f} kcal/day",
                help="Target Kalori Harian: Jumlah kalori yang harus dikonsumsi untuk mencapai tujuanmu (defisit 500 kalori untuk menurunkan berat badan, surplus 300 kalori untuk menambah otot, atau sama dengan TDEE untuk mempertahankan berat)."
            )
        with col4:
            st.metric(
                "BMI", 
                f"{bmi:.1f}", 
                delta=get_bmi_category(bmi),
                help="Body Mass Index: Perbandingan tinggi dan berat badanmu. Kategori: Underweight (<18.5), Normal (18.5-24.9), Overweight (25-29.9), Obese (>=30)."
            )
        
        submitted = st.form_submit_button("Save Profile", use_container_width=True)

    # ── Process profile save OUTSIDE the form block ──
    if submitted:
        user_data = {
            'name': name,
            'age': int(age),
            'gender': gender,
            'height_cm': int(height_cm),
            'weight_kg': float(weight_kg),
            'activity_level': activity_level,
            'fitness_goal': fitness_goal,
            'bmr': float(bmr),
            'tdee': float(tdee),
            'daily_target_calories': float(daily_target)
        }
        
        if user is not None:
            user_data['user_id'] = int(user['user_id'])
        
        try:
            user_id = save_user(user_data)
            user_data['user_id'] = int(user_id)
            
            # Update session state
            st.session_state.user = user_data
            st.session_state.user_id = int(user_id)
            # Explicitly stay on Profile after save
            st.session_state.active_menu = "Profile"
            
            # Reset chatbot with new profile data (lazy — don't initialize now)
            st.session_state.chatbot = None
            st.session_state.profile_updated = True
            st.session_state.greeting_sent = False
            
            # Record weight
            update_weight(int(user_id), float(weight_kg))
            
            st.success("✅ Profile berhasil disimpan!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Gagal menyimpan profil: {str(e)}")



elif menu == "Food Log":
    st.markdown('<div class="main-header">Food Log</div>', unsafe_allow_html=True)
    
    # Gunakan session state user
    if st.session_state.user is None:
        st.markdown('''
        <div class="warning-box">
            <strong>⚠️ Profil Belum Lengkap!</strong><br>Silakan lengkapi profil Anda terlebih dahulu di menu 👤 Profile untuk mengaktifkan fitur pencatatan makanan.
        </div>
        ''', unsafe_allow_html=True)
        st.stop()
    user = st.session_state.user
    
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Log Food", "🍽️ Meal Suggestions", "📋 Food History", "🛒 Food Association Rules"])
    
    with tab1:
        with st.container(border=True):
            st.subheader("Add Food Entry")
            
            # Macronutrients & Calorie explanation glossary
            with st.expander("📚 Mengenal Kalori & Makronutrisi (Karbohidrat, Protein, Lemak)"):
                st.markdown("""
                Saat mencatat makanan, penting untuk memahami nutrisi yang masuk ke tubuhmu:
                *   **Kalori (kcal):** Satuan energi yang didapat dari makanan. Untuk turun berat badan, konsumsi kalori harus lebih kecil dari pembakaran (TDEE).
                *   **Protein:** Makronutrisi penting untuk membangun dan memperbaiki jaringan otot. Sangat direkomendasikan saat diet/olahraga (1g protein = 4 kalori).
                *   **Karbohidrat (Carbs):** Sumber energi utama tubuh untuk beraktivitas dan olahraga berat (1g karbohidrat = 4 kalori).
                *   **Lemak (Fat):** Penting untuk kesehatan hormon, sendi, dan penyerapan vitamin (1g lemak = 9 kalori).
                """)
            
            # Load and sort dataset alphabetically for better browsing
            food_dataset = load_food_dataset().sort_values('Food').reset_index(drop=True)
            
            search_query = st.text_input("🔍 Cari Nama Makanan...", placeholder="Ketik kata kunci untuk menyaring makanan (contoh: Apple, Rice, Chicken)...")
            
            # Filter matches (flexible multi-word matching)
            if search_query.strip():
                keywords = search_query.lower().split()
                mask = pd.Series(True, index=food_dataset.index)
                for kw in keywords:
                    mask = mask & food_dataset['Food'].str.lower().str.contains(kw, na=False)
                matches = food_dataset[mask]
            else:
                matches = food_dataset
                
            if not matches.empty:
                # Format options list: "Food Name (Calories kcal/100g)"
                options = []
                for _, row in matches.iterrows():
                    options.append(f"{row['Food']} ({row['Calories_per_100g']:.1f} kcal/100g)")
                
                selected_opt = st.selectbox("Pilih Makanan yang Cocok", options)
                selected_idx = options.index(selected_opt)
                chosen_food = matches.iloc[selected_idx]
                
                col_portion, col_meal = st.columns(2)
                with col_portion:
                    portion_g = st.number_input("Porsi (Gram)", min_value=1.0, max_value=2000.0, value=100.0, step=10.0)
                with col_meal:
                    meal_type = st.selectbox("Meal Type", MEAL_TYPES, key="meal_db")
                    
                # Calculate estimated calories and macros based on portion
                factor = portion_g / 100.0
                est_cal = chosen_food['Calories_per_100g'] * factor
                est_protein = chosen_food['Protein_g'] * factor
                est_carbs = chosen_food['Carbs_g'] * factor
                est_fat = chosen_food['Fat_g'] * factor
                
                # Show estimation
                st.info(f"Estimasi Nutrisi Porsi: **{est_cal:.0f} kcal** | Protein: {est_protein:.1f}g | Karbohidrat: {est_carbs:.1f}g | Lemak: {est_fat:.1f}g")
                
                if st.button("Catat Makanan", use_container_width=True):
                    try:
                        add_food_log(
                            user['user_id'],
                            chosen_food['Food'],
                            est_cal,
                            est_protein,
                            est_carbs,
                            est_fat,
                            meal_type,
                            date.today().isoformat()
                        )
                        st.success(f"✅ {chosen_food['Food']} ({est_cal:.0f} kcal) berhasil dicatat!")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Gagal mencatat: {str(e)}")
            else:
                st.warning("⚠️ Tidak ada makanan yang cocok di database.")
                if st.button("Daftarkan Makanan Custom Baru", use_container_width=True):
                    st.session_state.expander_expanded = True
                    st.session_state.scroll_to_custom = True
                    st.session_state.prefilled_custom_name = search_query.strip().title()
                    st.rerun()
        
        # HTML Anchor for scrolling
        st.markdown('<div id="custom-food-anchor"></div>', unsafe_allow_html=True)
        
        # JavaScript for smooth scrolling
        if st.session_state.scroll_to_custom:
            import streamlit.components.v1 as components
            components.html(
                """
                <script>
                    window.parent.document.getElementById("custom-food-anchor").scrollIntoView({behavior: "smooth"});
                </script>
                """,
                height=0
            )
            st.session_state.scroll_to_custom = False
        
        st.markdown("---")
        with st.expander("➕ Daftarkan Makanan Custom Baru ke Database", expanded=st.session_state.expander_expanded):
            st.markdown("Jika makanan tidak ada dalam daftar pencarian di atas, Anda bisa mendaftarkannya di sini agar bisa dicari di kemudian hari.")
            with st.form("custom_food_form"):
                custom_name = st.text_input("Nama Makanan*", value=st.session_state.prefilled_custom_name, placeholder="Contoh: Nasi Goreng Spesial")
                custom_cals = st.number_input("Kalori per 100g (kcal)*", min_value=0.0, max_value=1000.0, value=150.0, step=10.0)
                
                col_c1, col_c2, col_c3 = st.columns(3)
                with col_c1:
                    custom_protein = st.number_input("Protein per 100g (g)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
                with col_c2:
                    custom_carbs = st.number_input("Karbohidrat per 100g (g)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
                with col_c3:
                    custom_fat = st.number_input("Lemak per 100g (g)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
                    
                register_submitted = st.form_submit_button("Simpan ke Database")
                
                if register_submitted:
                    if not custom_name.strip():
                        st.error("Nama makanan tidak boleh kosong.")
                    else:
                        existing_match = food_dataset[food_dataset['Food'].str.strip().str.lower() == custom_name.strip().lower()]
                        if not existing_match.empty:
                            st.warning(f"⚠️ Makanan dengan nama '{custom_name}' sudah terdaftar dalam database!")
                        else:
                            try:
                                add_custom_food_to_dataset(
                                    custom_name.strip(),
                                    custom_cals,
                                    custom_protein,
                                    custom_carbs,
                                    custom_fat
                                )
                                st.success(f"✅ Makanan '{custom_name}' berhasil didaftarkan! Sekarang Anda bisa memilihnya pada menu di atas.")
                                st.session_state.expander_expanded = False
                                st.session_state.prefilled_custom_name = ""
                                time.sleep(1.5)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Gagal mendaftarkan makanan: {str(e)}")
    
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
            
    with tab4:
        st.subheader("🛒 Food Combination Pattern Analysis (Association Rule Mining)")
        st.markdown("""
        This feature applies the **Apriori Algorithm** to analyze which foods are frequently consumed together.
        It is useful for understanding your eating habits and generating food recommendations automatically.
        """)
        
        # Controls
        col_src, col_group = st.columns(2)
        with col_src:
            data_source = st.selectbox(
                "📁 Data Source",
                ["Sample Food History Dataset (Demo)", "My Personal Food Logs"],
                help="Choose the demo dataset to try the feature instantly or use your own food logging history."
            )
        with col_group:
            grouping = st.selectbox(
                "🔄 Transaction Grouping",
                ["By Day", "By Meal Type"],
                help="Determine whether a transaction represents all foods consumed in a day or foods consumed during a specific meal."
            )
            
        col_sup, col_conf = st.columns(2)
        with col_sup:
            min_support = st.slider(
                "📈 Minimum Support",
                min_value=0.01, max_value=0.50, value=0.05, step=0.01,
                format="%.2f",
                help="Support measures how often a food combination appears across all transactions. A value of 0.05 means it must appear in at least 5% of transactions."
            )
        with col_conf:
            min_confidence = st.slider(
                "🎯 Minimum Confidence",
                min_value=0.10, max_value=1.00, value=0.30, step=0.05,
                format="%.2f",
                help="Confidence measures how often the rule is true. A value of 0.50 means that if someone eats A, there is a 50% chance they also eat B."
            )
            
        if st.button("🚀 Analyze Patterns", use_container_width=True):
            transactions = []
            
            if data_source == "My Personal Log History":
                # Ambil semua data log makanan pengguna (tidak dibatasi 30 hari agar datanya lebih banyak)
                personal_logs = get_food_logs(user['user_id'], 365)
                if personal_logs.empty:
                    st.warning("⚠️ Your food history is empty. Please log some foods first in the 'Log Food' tab or use the demo dataset.")
                else:
                    # Bersihkan nama makanan agar seragam
                    personal_logs['food_name'] = personal_logs['food_name'].str.strip().str.title()
                    
                    if grouping == "By Day":
                        # Group by log_date
                        grouped = personal_logs.groupby('log_date')['food_name'].apply(list)
                        transactions = grouped.tolist()
                    else:
                        # Group by log_date and meal_type
                        grouped = personal_logs.groupby(['log_date', 'meal_type'])['food_name'].apply(list)
                        transactions = grouped.tolist()
            else:
                transactions = get_sample_transactions()
                
            if len(transactions) < 3:
                if data_source == "My Personal Log History":
                    st.error(f"❌ Not enough transactions (only {len(transactions)} found). Please add at least 3 different days or meal records.")
            else:
                with st.spinner("Mining food combination patterns using the Apriori Algorithm..."):
                    rules, freq_1, freq_2 = run_apriori(transactions, min_support, min_confidence)
                    
                    # Display Overview Metrics
                    st.markdown("---")
                    st.subheader("📊 Association Rule Mining Results")
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Total Transactions", len(transactions))
                    with col_m2:
                        # count unique foods in transactions
                        unique_foods = len(set([item for tx in transactions for item in tx]))
                        st.metric("Unique Foods", unique_foods)
                    with col_m3:
                        st.metric("Generated Rules", len(rules))
                        
                    # Show rules if any
                    if len(rules) > 0:
                        # Build DataFrame for display
                        rules_display = []
                        for r_idx, r in enumerate(rules, 1):
                            rules_display.append({
                                "No": r_idx,
                                "If Eats (Antecedent)": f"{r['antecedent']}",
                                "Then Likely Also Eats (Consequent)": f"{r['consequent']}",
                                "Support": f"{r['support']:.1%}",
                                "Confidence": f"{r['confidence']:.1%}",
                                "Lift": f"{r['lift']:.2f}",
                                "Interpretasi": f"If a user eats {r['antecedent']}, there is a{r['confidence']:.0%} chance they also eat {r['consequent']}"
                            })
                        
                        df_rules = pd.DataFrame(rules_display)
                        st.markdown("#### 🛒 Aturan Asosiasi Makanan yang Ditemukan")
                        st.dataframe(df_rules, use_container_width=True, hide_index=True)
                        
                        # Interpretasi Lift
                        st.info("""
                        💡 **How to Interpret Lift Values:**
                        - **Lift > 1**: Strong positive relationship. Foods A and B are frequently consumed together.
                        - **Lift = 1**: Independent relationship. Eating A does not affect the likelihood of eating B.
                        - **Lift < 1**: Negative relationship. Foods A and B tend not to be consumed together.
                        """)
                    else:
                        st.warning("⚠️ No association rules meet the selected Minimum Support or Minimum Confidence thresholds. Try lowering the slider values.")
                        
                    # Visualisasi Item Terbanyak (Frequent Items)
                    if freq_1:
                        st.markdown("---")
                        st.markdown("#### 🌟 Most Popular Foods (Frequent 1-Itemsets)")
                        # Convert to dataframe for charting
                        freq_items_df = pd.DataFrame([
                            {"Makanan": item, "Popularitas (Support)": sup} for item, sup in freq_1.items()
                        ]).sort_values(by="Popularitas (Support)", ascending=False)
                        
                        fig = px.bar(
                            freq_items_df.head(10), 
                            x="Popularitas (Support)", 
                            y="Makanan", 
                            orientation='h',
                            title="Top 10 Most Frequently Consumed Foods",
                            color="Popularitas (Support)",
                            color_continuous_scale="Viridis"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
        # Educational Section
        with st.expander("📚 Learn About Apriori Algorithm & Data Mining"):
            st.markdown("""
            ### What is Association Rule Mining & the Apriori Algorithm?

            **Association Rule Mining** is a data mining technique used to discover interesting relationships or patterns among items in large transactional datasets. One of its most common applications is *Market Basket Analysis* in supermarkets and retail stores.

            **The Apriori Algorithm** is a classic algorithm that operates based on the following principle:

            > If an item combination (*itemset*) occurs infrequently, then all of its supersets will also occur infrequently.

            #### 3 Main Metrics Used:

            1. **Support**: Measures how frequently an item combination appears in the database.
            $$\\text{Support}(A \\rightarrow B) = \\frac{\\text{Number of transactions containing A and B}}{\\text{Total number of transactions}}$$

            2. **Confidence**: Measures the reliability or certainty of a rule. It indicates how often B appears in transactions that already contain A.
            $$\\text{Confidence}(A \\rightarrow B) = \\frac{\\text{Number of transactions containing A and B}}{\\text{Number of transactions containing A}}$$

            3. **Lift**: Measures the strength of a rule compared to the expected occurrence if A and B were independent.
            $$\\text{Lift}(A \\rightarrow B) = \\frac{\\text{Confidence}(A \\rightarrow B)}{\\text{Support}(B)}$$
            """)

elif menu == "Activity Log":
    st.markdown('<div class="main-header">Activity Log</div>', unsafe_allow_html=True)
    
    if st.session_state.user is None:
        st.markdown('''
        <div class="warning-box">
            <strong>⚠️ Profil Belum Lengkap!</strong><br>Silakan lengkapi profil Anda terlebih dahulu di menu 👤 Profile untuk mengaktifkan fitur pencatatan aktivitas.
        </div>
        ''', unsafe_allow_html=True)
        st.stop()
    user = st.session_state.user
    
    tab1, tab2 = st.tabs(["📝 Log Activity", "📋 Activity History"])
    
    with tab1:
        st.subheader("Log Your Workout")
        
        # MET & Calorie explanation for laypeople
        with st.expander("🔍 Bagaimana Kalori Olahraga Dihitung? (Mengenal MET)"):
            st.markdown("""
            Aplikasi ini menghitung kalori olahraga menggunakan nilai **MET (Metabolic Equivalent of Task)**:
            *   **MET** menunjukkan seberapa banyak energi yang kamu gunakan untuk suatu aktivitas dibandingkan dengan saat duduk diam (1 MET).
            *   Misalnya, **HIIT (8.5 MET)** membakar energi 8.5 kali lebih banyak daripada diam/istirahat.
            
            **Rumus Perhitungan Kalori:**
            $$\\text{Kalori Dibakar} = \\frac{\\text{MET} \\times 3.5 \\times \\text{Berat Badan (kg)}}{200} \\times \\text{Durasi (menit)}$$
            *Jadi, berat badan yang lebih besar atau intensitas yang lebih tinggi akan membakar kalori lebih cepat!*
            """)
        
        with st.form("activity_log_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                activity_type = st.selectbox("Activity Type", list(ACTIVITY_MET.keys()))
                duration_min = st.number_input("Duration (minutes)", min_value=1, max_value=300, value=30)
                intensity = st.select_slider("Intensity", options=["Low", "Medium", "High"], value="Medium")
            
            with col2:
                st.markdown("**Estimated Calories Burned**")
                st.markdown("*Will be calculated after you click Log Activity*")
            
            log_activity_btn = st.form_submit_button("Log Activity", use_container_width=True)
        
        if log_activity_btn:
            calories_burned = calculate_calories_burned_met(activity_type, duration_min, user['weight_kg'])
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

elif menu == "Fitness Level Classifier":
    st.markdown('<div class="main-header">Klasifikasi Tingkat Kebugaran</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: #f0f9ff; border-left: 4px solid #4f46e5; border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem;">
        Masukkan data fisik dan hasil tes kebugaran Anda. Sistem akan memprediksi tingkat kebugaran 
        menggunakan algoritma Machine Learning: <strong>Random Forest</strong>.
    </div>
    """, unsafe_allow_html=True)

    @st.cache_resource
    def load_fitness_models():
        from models import load_body_performance_model
        try:
            models, scaler, target_encoder, gender_map, feature_order = load_body_performance_model()
            return models, scaler, target_encoder, gender_map, feature_order
        except Exception as e:
            st.error(f"Gagal memuat model: {str(e)}")
            return None, None, None, None, None

    models, scaler, target_encoder, gender_map, feature_order = load_fitness_models()

    if models is not None:
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
        algorithm = "random_forest"
        
        if st.button("Prediksi Tingkat Kebugaran", use_container_width=True):
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
                    
                    st.markdown("---")
                    st.subheader("📊 Hasil Klasifikasi")
                    
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
                        <p style="margin: 0;">Sistem memprediksi dengan tingkat keyakinan <strong>{confidence:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.subheader("📋 Rekomendasi")
                    if "Excellent" in result:
                        st.success("🎉 **Pertahankan!** Anda dalam kondisi bugar prima. Lanjutkan rutinitas olahraga dan pola makan sehat Anda.")
                    elif "Good" in result:
                        st.info("💪 **Bagus!** Tingkat kebugaran Anda sudah baik. Tingkatkan intensitas latihan untuk mencapai level Excellent.")
                    elif "Average" in result:
                        st.warning("📈 **Mulai tingkatkan!** Coba tambahkan latihan kardio 3x seminggu dan perbaiki pola makan.")
                    else:
                        st.error("🏃‍♂️ **Ayo mulai bergerak!** Konsultasikan dengan pelatih kebugaran untuk program latihan yang sesuai dengan kondisi Anda.")
                    
                    with st.expander("📚 Tentang Model Machine Learning"):
                        st.markdown("""
                        **Random Forest**
                        - Ensemble learning menggunakan banyak pohon keputusan
                        - Akurasi pada data test: ~75%
                        - Keunggulan: Tidak mudah overfitting, bisa handle data non-linear
                        """)
                        
                except Exception as e:
                    st.error(f"Error saat prediksi: {str(e)}")
    
    else:
        st.error("Gagal memuat model. Silakan hubungi administrator aplikasi.")

elif menu == "AI Chatbot":
    st.markdown('<div class="main-header">AI Fitness Coach</div>', unsafe_allow_html=True)
    
    # Gunakan session state user
    if st.session_state.user is None:
        st.markdown('''
        <div class="warning-box">
            <strong>⚠️ Profil Belum Lengkap!</strong><br>Silakan lengkapi profil Anda terlebih dahulu di menu 👤 Profile untuk dapat berkonsultasi dengan AI Fitness Coach.
        </div>
        ''', unsafe_allow_html=True)
        st.stop()
    user = st.session_state.user
    
    # Inisialisasi chatbot jika belum ada atau jika profil baru diupdate
    if st.session_state.chatbot is None or st.session_state.profile_updated:
        user_data = {
            'user_id': user['user_id'],
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
        
        # Jika pesan sambutan sudah ada di history, ganti dengan nama baru
        if len(st.session_state.messages) > 0 and st.session_state.messages[0]['role'] == 'assistant':
            from datetime import datetime
            hour = datetime.now().hour
            if hour < 12:
                sapaan = "Selamat pagi"
            elif hour < 18:
                sapaan = "Selamat siang"
            else:
                sapaan = "Selamat malam"
            
            nama = user['name'] if user['name'] else "Teman"
            tujuan = user['fitness_goal']
            if tujuan == "Weight Loss":
                pesan_tujuan = "Ayo capai target penurunan berat badanmu! Aku akan membantu dengan defisit kalori yang sehat."
            elif tujuan == "Muscle Gain":
                pesan_tujuan = "Siap membentuk otot? Aku akan memandumu dengan pola makan dan latihan yang tepat."
            else:
                pesan_tujuan = "Aku di sini untuk membantumu menjaga gaya hidup sehat dan seimbang."
            
            new_greeting = f"""{sapaan} {nama}! Senang berkenalan denganmu.

Aku FitBot, asisten kebugaran dan nutrisi pribadimu.

{pesan_tujuan}

Apa yang bisa aku bantu?
- Menganalisis makanan dan menghitung kalori
- Menyusun rencana olahraga
- Memantau progres harian
- Memberi motivasi dan tips kesehatan

Coba tanyakan ini (dalam bahasa Inggris atau Indonesia):
- "How many calories should I eat today?"
- "Give me a quick home workout"
- "What's a healthy breakfast idea?"
- "Aku butuh motivasi!"

Aku akan merespon dalam bahasa yang kamu gunakan. Yuk mulai dengan mengetik pertanyaanmu di bawah!"""
            st.session_state.messages[0]['content'] = new_greeting
        
        st.session_state.profile_updated = False
        st.rerun()
    
    # Tampilkan sapaan pertama kali jika belum ada pesan
    if not st.session_state.greeting_sent and len(st.session_state.messages) == 0:
        from datetime import datetime
        hour = datetime.now().hour
        if hour < 12:
            sapaan = "Selamat pagi"
        elif hour < 18:
            sapaan = "Selamat siang"
        else:
            sapaan = "Selamat malam"
        
        nama = user['name'] if user['name'] else "Teman"
        tujuan = user['fitness_goal']
        if tujuan == "Weight Loss":
            pesan_tujuan = "Ayo capai target penurunan berat badanmu! Aku akan membantu dengan defisit kalori yang sehat."
        elif tujuan == "Muscle Gain":
            pesan_tujuan = "Siap membentuk otot? Aku akan memandumu dengan pola makan dan latihan yang tepat."
        else:
            pesan_tujuan = "Aku di sini untuk membantumu menjaga gaya hidup sehat dan seimbang."
        
        greeting = f"""{sapaan} {nama}! Senang berkenalan denganmu.

Aku FitBot, asisten kebugaran dan nutrisi pribadimu.

{pesan_tujuan}

Apa yang bisa aku bantu?
- Menganalisis makanan dan menghitung kalori
- Menyusun rencana olahraga
- Memantau progres harian
- Memberi motivasi dan tips kesehatan

Coba tanyakan ini (dalam bahasa Inggris atau Indonesia):
- "How many calories should I eat today?"
- "Give me a quick home workout"
- "What's a healthy breakfast idea?"
- "Aku butuh motivasi!"

Aku akan merespon dalam bahasa yang kamu gunakan. Yuk mulai dengan mengetik pertanyaanmu di bawah!"""
        
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
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        calories_in, calories_out = get_today_summary(user['user_id'])
        context = {
            'calories_in': calories_in,
            'calories_out': calories_out
        }
        history = st.session_state.messages[:-1]
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.get_response(
                    user_message=prompt,
                    context=context,
                    history=history
                )
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        try:
            save_chat_message(user['user_id'], prompt, response)
        except Exception:
            pass
    
    # Quick questions sidebar
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
                st.session_state.messages.append({"role": "user", "content": q})
                with st.chat_message("user"):
                    st.markdown(q)
                
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
                try:
                    save_chat_message(user['user_id'], q, response)
                except Exception:
                    pass

elif menu == "📊 ML Predictor":
    st.markdown('<div class="main-header">📊 ML Calorie Burn Predictor</div>', unsafe_allow_html=True)
    st.info("This machine learning model predicts calories burned during exercise based on your workout parameters.")
    
    # Workout parameter glossary for laypeople
    with st.expander("📚 Penjelasan Parameter Latihan (Mengapa ini penting?)"):
        st.markdown("""
        Model Machine Learning (Random Forest) memprediksi kalori terbakar berdasarkan faktor berikut:
        *   **Durasi (Workout Duration):** Semakin lama latihan, semakin banyak energi total yang dikeluarkan.
        *   **Detak Jantung (Heart Rate):** Mengukur intensitas latihan secara langsung. Detak jantung lebih tinggi menunjukkan beban kerja jantung & otot yang lebih berat.
        *   **Suhu Tubuh (Body Temperature):** Selama olahraga, metabolisme tubuh menghasilkan panas. Suhu tubuh meningkat seiring dengan tingginya pembakaran kalori.
        *   **Berat Badan (Weight):** Memindahkan tubuh yang lebih berat membutuhkan usaha (energi) yang lebih besar, sehingga membakar kalori lebih banyak pada aktivitas yang sama.
        """)

    @st.cache_resource
    def load_cached_calorie_model():
        from models import load_calorie_model, train_calorie_prediction_model
        try:
            model, encoders, scaler = load_calorie_model()
            return model, encoders, scaler, None
        except:
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

elif menu == "About":
    st.markdown('<div class="main-header">About This App</div>', unsafe_allow_html=True)

    # ── Hero card ──────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("""
        <div style="text-align:center; padding:0.5rem 0 1rem 0;">
            <span style="font-size:3.5rem;">💪</span>
            <h2 style="margin:0.4rem 0 0.2rem 0;">Smart Fitness &amp; Calorie Tracker</h2>
            <p style="color:#8E8E93; margin:0;">Version 2.0.0 &nbsp;|&nbsp; Python · Streamlit · SQLite · Scikit-learn · Plotly · Groq API</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Key Features ───────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("### 🎯 Key Features")
        feat_col1, feat_col2 = st.columns(2)
        features = [
            ("📊", "Personalized Calorie Tracking", "Menghitung BMR, TDEE, dan target harian berdasarkan profil kamu"),
            ("🍽️", "Food Log", "Catat makanan yang dimakan setiap hari"),
            ("🏃", "Activity Tracking", "Catat olahraga dengan kalkulasi pembakaran kalori otomatis (rumus MET)"),
            ("🤖", "AI Fitness Chatbot", "Saran kebugaran personal (Groq Llama 3.3 70B atau rule-based fallback)"),
            ("🔮", "ML Calorie Predictor", "Memprediksi kalori terbakar saat olahraga (Random Forest Regressor)"),
            ("🏅", "Fitness Level Classifier", "Klasifikasi level kebugaran A/B/C/D (Random Forest, akurasi ~74.5%)"),
            ("📈", "Progress Dashboard", "Grafik interaktif berat badan dan tren kalori harian"),
            ("🔒", "Session Isolation", "Setiap user punya sesi mandiri berbasis browser cookies"),
        ]
        for i, (icon, title, desc) in enumerate(features):
            col = feat_col1 if i % 2 == 0 else feat_col2
            with col:
                st.markdown(f"""
                <div style="display:flex; align-items:flex-start; gap:0.75rem; margin-bottom:0.9rem;">
                    <span style="font-size:1.6rem; line-height:1.2;">{icon}</span>
                    <div>
                        <strong>{title}</strong><br>
                        <span style="font-size:0.85rem; color:#8E8E93;">{desc}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── How It Works ───────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("### 📋 Cara Penggunaan")
        steps = [
            ("1", "#4f46e5", "Setup Profile", "Masukkan usia, jenis kelamin, tinggi, berat, level aktivitas, dan tujuan kebugaran"),
            ("2", "#0ea5e9", "Catat Harian", "Log makanan dan aktivitas sepanjang hari"),
            ("3", "#10b981", "Pantau Progress", "Cek dashboard untuk melihat apakah kamu on-track"),
            ("4", "#f59e0b", "Tanya AI Coach", "Gunakan chatbot untuk pertanyaan seputar kebugaran"),
            ("5", "#ef4444", "Cek Fitness Level", "Input hasil tes fisik untuk mendapat klasifikasi kebugaran kamu"),
        ]
        for num, color, title, desc in steps:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:1rem; margin-bottom:0.85rem;">
                <div style="min-width:2.2rem; height:2.2rem; border-radius:50%; background:{color};
                            display:flex; align-items:center; justify-content:center;
                            color:white; font-weight:700; font-size:1rem;">{num}</div>
                <div><strong>{title}</strong> — <span style="font-size:0.9rem; color:#8E8E93;">{desc}</span></div>
            </div>
            """, unsafe_allow_html=True)

    # ── Formulas ───────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("### 🔬 Formula yang Digunakan")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.markdown("""
            **BMR (Mifflin-St Jeor)**
            - Pria: `(10×BB) + (6.25×TB) - (5×Usia) + 5`
            - Wanita: `(10×BB) + (6.25×TB) - (5×Usia) - 161`

            **TDEE**
            - `BMR × Activity Multiplier`
            """)
        with f_col2:
            st.markdown("""
            **Kalori Terbakar (MET)**
            - `(MET × 3.5 × BB) / 200 × menit`

            **ML Calorie Prediction**
            - Random Forest Regressor

            **ML Fitness Classification**
            - Random Forest + StandardScaler
            """)

    # ── Dataset Info ───────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("### 📁 Informasi Dataset")
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            try:
                food_df = pd.read_csv('data/food_dataset.csv')
                st.success(f"✅ Food dataset: **{len(food_df):,}** items")
            except:
                st.info("📝 Food dataset akan dibuat otomatis")
            try:
                exercise_df = pd.read_csv('data/exercise_dataset.csv')
                st.success(f"✅ Exercise dataset: **{len(exercise_df):,}** samples")
            except:
                st.info("📝 Exercise dataset akan dibuat otomatis")
        with d_col2:
            try:
                body_df = pd.read_csv('data/body_performance.csv')
                st.success(f"✅ Body performance dataset: **{len(body_df):,}** samples")
                st.caption("Features: usia, gender, tinggi, berat, body fat %, tekanan darah, grip strength, fleksibilitas, sit-ups, broad jump → Target: kelas A/B/C/D")
            except:
                st.warning("⚠️ Body performance dataset tidak ditemukan. Fitness classifier mungkin tidak berfungsi.")

    # ── Tips ───────────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("### 💡 Tips untuk Hasil Terbaik")
        tips_col1, tips_col2 = st.columns(2)
        tips = [
            "Catat semua yang kamu makan dan minum",
            "Konsisten dalam tracking setiap hari",
            "Update berat badan setiap minggu",
            "Gunakan AI chatbot untuk saran personal",
            "Cek progress secara rutin agar tetap termotivasi",
            "Input hasil tes fisik yang jujur untuk klasifikasi akurat",
        ]
        for i, tip in enumerate(tips):
            col = tips_col1 if i % 2 == 0 else tips_col2
            with col:
                st.markdown(f"✔️ {tip}")

    # ── Footer ─────────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("""
        <div style="text-align:center; padding:0.5rem 0;">
            <p style="margin:0; color:#8E8E93; font-size:0.9rem;">
                Made with ❤️ for Final Project &nbsp;·&nbsp; Smart Fitness &amp; Food Tracker
            </p>
        </div>
        """, unsafe_allow_html=True)
        
if __name__ == "__main__":
    pass