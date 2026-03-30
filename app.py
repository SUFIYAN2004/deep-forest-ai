import streamlit as st
import pandas as pd
import pickle
import string
import re
import time
import random

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Deep Forest | Gamified AI",
    page_icon="🌲",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. EXTREME CSS OVERHAUL (SLEEK TYPOGRAPHY) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    
    /* Global Base Font Size Reduction */
    html, body, [class*="css"] { 
        font-family: 'Space Grotesk', sans-serif !important; 
        font-size: 14px !important; 
    }
    
    /* Hide Streamlit branding and sidebar toggle */
    #MainMenu {visibility: hidden;} 
    header {visibility: hidden;} 
    footer {visibility: hidden;}
    [data-testid="collapsedControl"] {display: none;}
    
    /* App Background */
    .stApp {
        background: radial-gradient(circle at top right, #0a192f 0%, #020c1b 100%);
        color: #e6f1ff;
    }

    /* Adjust container padding */
    .block-container { 
        padding-top: 2rem !important; 
        padding-bottom: 5rem !important; 
        max-width: 800px !important; 
    }

    /* Glassmorphism Buttons - COMPACT VERSION */
    div[data-testid="stButton"] button {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(100, 255, 218, 0.2) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1) !important;
        color: #e6f1ff !important;
        white-space: normal !important; 
        text-align: left !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        padding: 12px 15px !important; /* Tighter padding */
        min-height: 50px; /* Shorter buttons */
        margin-bottom: 8px !important;
        font-size: 0.85rem !important; /* Smaller button text */
    }
    
    /* Glowing Hover Effect */
    div[data-testid="stButton"] button:hover { 
        border-color: #64ffda !important; 
        background: rgba(100, 255, 218, 0.08) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px -10px rgba(100, 255, 218, 0.3) !important;
        color: #64ffda !important;
    }

    /* Custom Chat Bubbles - Tighter Text */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 12px 15px;
        margin-bottom: 10px;
        backdrop-filter: blur(5px);
        font-size: 0.9rem !important; /* Smaller chat text */
    }
    
    /* Avatars */
    [data-testid="stChatMessageAvatarUser"] { background-color: #112240; border: 1px solid #233554;}
    [data-testid="stChatMessageAvatarAssistant"] { background-color: #020c1b; border: 1px solid #64ffda; }
    
    /* Header Scaledowns */
    h1 { color: #ccd6f6 !important; text-align: center; font-size: 1.8rem !important; }
    h2 { color: #ccd6f6 !important; text-align: center; font-size: 1.3rem !important; margin-bottom: 1.5rem !important;}
    h3 { color: #ccd6f6 !important; text-align: center; font-size: 1.1rem !important; }
    p { color: #8892b0 !important; font-size: 0.9rem !important; }
    
    </style>
    """, unsafe_allow_html=True)

# --- 3. DATA & MODEL LOADING ---
@st.cache_data
def load_and_categorize_questions():
    try:
        df = pd.read_csv('Question_2k.csv')
        all_q = df['Questions'].dropna().tolist()
        categorized = {"Easy": [], "Intermediate": [], "Hard": []}
        for q in all_q:
            if len(q) < 80: categorized["Easy"].append(q)
            elif len(q) > 200: categorized["Hard"].append(q)
            else: categorized["Intermediate"].append(q)
        for k in categorized:
            if not categorized[k]: categorized[k] = all_q[:50]
        return categorized
    except FileNotFoundError:
        return {"Easy": ["What is a loop?"], "Intermediate": ["Explain matrices."], "Hard": ["Write a complex ML script."]}

@st.cache_resource
def load_models():
    with open('rf_model.pkl', 'rb') as f: model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f: vec = pickle.load(f)
    return model, vec

q_bank = load_and_categorize_questions()
model, vec = load_models()

def clean_dl_text(text):
    text = str(text).lower()
    text = re.sub(r'```[a-z]*', '', text) 
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(text.split())

# --- 4. STATE MACHINE ---
if "stage" not in st.session_state: st.session_state.stage = "intro"
if "messages" not in st.session_state: st.session_state.messages = []
if "difficulty" not in st.session_state: st.session_state.difficulty = None
if "options" not in st.session_state: st.session_state.options = []

# --- 5. MOBILE-FRIENDLY TOP NAVIGATION ---
if st.session_state.stage != "intro":
    st.markdown(f"<div style='text-align: center; color: #64ffda; font-weight: bold; margin-bottom: 15px; font-size: 0.9rem;'>🌲 DEEP FOREST // {str(st.session_state.difficulty).upper()}</div>", unsafe_allow_html=True)
    
    if st.button("🎚️ Change Level", use_container_width=True):
        st.session_state.stage = "level_select"
        st.rerun()
    if st.button("🔄 Reboot System", use_container_width=True):
        st.session_state.stage = "intro"
        st.session_state.messages = []
        st.session_state.options = []
        st.rerun()
    
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin-top: 10px;'>", unsafe_allow_html=True)

# --- 6. UI FLOW RENDERING ---

# STAGE 1: THE INTRO
if st.session_state.stage == "intro":
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1>DEEP FOREST_</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64ffda; font-family: monospace; font-size: 0.85rem; margin-bottom: 2rem;'>// RULE-BASED NEURAL ENGINE ALIGNED //</p>", unsafe_allow_html=True)
    
    if st.button("INITIALIZE SEQUENCE", use_container_width=True):
        st.session_state.stage = "level_select"
        st.rerun()

# STAGE 2: LEVEL SELECT
elif st.session_state.stage == "level_select":
    st.markdown("<h2>Select Engine Complexity</h2>", unsafe_allow_html=True)
    
    if st.button("🟢 EASY - Short & Sweet", use_container_width=True):
        st.session_state.difficulty = "Easy"
        st.session_state.options = random.sample(q_bank["Easy"], 5)
        st.session_state.stage = "chatting"
        st.rerun()
        
    if st.button("🟡 INTERMEDIATE - Standard Logic", use_container_width=True):
        st.session_state.difficulty = "Intermediate"
        st.session_state.options = random.sample(q_bank["Intermediate"], 5)
        st.session_state.stage = "chatting"
        st.rerun()
        
    if st.button("🔴 HARD - Complex Algorithms", use_container_width=True):
        st.session_state.difficulty = "Hard"
        st.session_state.options = random.sample(q_bank["Hard"], 5)
        st.session_state.stage = "chatting"
        st.rerun()

# STAGE 3: CHATTING
elif st.session_state.stage == "chatting":
    
    USER_AVATAR = "🧑‍💻"
    BOT_AVATAR = "🌲"
    
    for message in st.session_state.messages:
        avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(f"AWAITING INPUT // SELECT QUERY:")
    
    selected_query = None
    
    for i in range(5):
        opt = st.session_state.options[i]
        if st.button(opt, key=f"opt_{i}", use_container_width=True):
            selected_query = opt

    if selected_query:
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(selected_query)
        st.session_state.messages.append({"role": "user", "content": selected_query})
        
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            try:
                cleaned = clean_dl_text(selected_query)
                vectorized = vec.transform([cleaned])
                response = model.predict(vectorized)[0]
                
                def stream_data(text, delay=0.015):
                    for word in text.split(" "):
                        yield word + " "
                        time.sleep(delay)
                        
                st.write_stream(stream_data(response))
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                st.session_state.options = random.sample(q_bank[st.session_state.difficulty], 5)
                st.rerun()
                
            except Exception as e:
                st.error(f"SYSTEM FAULT: {e}")