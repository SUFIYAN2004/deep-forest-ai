import streamlit as st
import pandas as pd
import pickle
import string
import re
import time
import random

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Deep Forest | Coding Assistant",
    page_icon="🌲",
    layout="centered"
)

# --- 2. PRODUCTION CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    #MainMenu {visibility: hidden;} header {visibility: hidden;} footer {visibility: hidden;}

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 6rem !important;
        max-width: 800px !important; 
    }

    /* Chat Input Styling */
    [data-testid="stChatInput"] {
        border-radius: 20px !important;
        border: 1px solid #333 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    }
    [data-testid="stChatInput"]:focus-within { border: 1px solid #4CAF50 !important; }

    /* Avatars */
    [data-testid="stChatMessageAvatarUser"] { background-color: #2b313e; }
    [data-testid="stChatMessageAvatarAssistant"] { background-color: #0e1117; border: 1px solid #4CAF50; }
    
    /* Suggestion Cards/Chips Styling */
    div[data-testid="stButton"] button {
        height: auto;
        min-height: 60px;
        white-space: normal;
        text-align: left;
        border-radius: 12px;
        border: 1px solid #444;
        background-color: transparent;
        transition: all 0.2s ease-in-out;
        font-size: 0.85rem;
        padding: 10px 15px;
    }
    div[data-testid="stButton"] button:hover {
        border-color: #4CAF50;
        background-color: rgba(76, 175, 80, 0.05);
        color: #fff;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD DATA & ASSETS ---
@st.cache_data
def load_questions():
    """Loads the CSV once and caches it so it doesn't slow down the app."""
    try:
        df = pd.read_csv('Question_2k.csv')
        return df['Questions'].dropna().tolist()
    except FileNotFoundError:
        return ["What is a loop?", "Explain matrices.", "How to reverse a string?", "What is a prime number?"]

@st.cache_resource
def load_models():
    """Loads the ML models."""
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vec = pickle.load(f)
    return model, vec

question_bank = load_questions()
model, vec = load_models()

def clean_dl_text(text):
    text = str(text).lower()
    text = re.sub(r'```[a-z]*', '', text) 
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(text.split())

# --- 4. STATE MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the first 4 specific suggestions if the chat is empty
if "current_suggestions" not in st.session_state:
    st.session_state.current_suggestions = [
        "Create a nested loop to print every combination of numbers between 0-9...",
        "Write a function to find the number of distinct states in a given matrix.",
        "Write code that removes spaces and punctuation marks from a given string.",
        "Write a function that checks if a given number is prime or not."
    ]

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Settings")
    st.warning("⚠️ This is a rule-based AI. Click the suggestions below the chat to test it!")
    if st.button("🗑️ Clear Conversation & Restart", use_container_width=True):
        st.session_state.messages = []
        # Reset back to the first 4 questions
        del st.session_state.current_suggestions 
        st.rerun()

# --- 6. RENDER CHAT HISTORY ---
USER_AVATAR = "🧑‍💻"
BOT_AVATAR = "🤖"

# If zero state, show big header
if len(st.session_state.messages) == 0:
    st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>🌲 Deep Forest Bot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888; margin-bottom: 2rem;'>Trained on 2,000 coding QA pairs. Get started below.</p>", unsafe_allow_html=True)

# Draw the messages
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# --- 7. DYNAMIC SUGGESTIONS UI ---
suggestion_clicked = None
st.write("") # Add a little visual padding

if len(st.session_state.messages) == 0:
    # FIRST VIEW: 4 Questions in a 2x2 grid
    col1, col2 = st.columns(2)
    sugs = st.session_state.current_suggestions
    
    if col1.button(f"🔄 {sugs[0][:60]}...", help=sugs[0], use_container_width=True): suggestion_clicked = sugs[0]
    if col1.button(f"🔢 {sugs[1][:60]}...", help=sugs[1], use_container_width=True): suggestion_clicked = sugs[1]
    if col2.button(f"✂️ {sugs[2][:60]}...", help=sugs[2], use_container_width=True): suggestion_clicked = sugs[2]
    if col2.button(f"⏱️ {sugs[3][:60]}...", help=sugs[3], use_container_width=True): suggestion_clicked = sugs[3]

else:
    # SUBSEQUENT VIEWS: 3 Random questions in a 3-column grid at the bottom
    st.caption("✨ Try asking one of these:")
    cols = st.columns(3)
    sugs = st.session_state.current_suggestions
    
    for i in range(3):
        # We truncate the button text so they don't get too massive, but keep the full text for the AI logic
        btn_label = sugs[i][:55] + "..." if len(sugs[i]) > 55 else sugs[i]
        if cols[i].button(btn_label, key=f"rnd_{i}", help=sugs[i], use_container_width=True):
            suggestion_clicked = sugs[i]

# --- 8. CHAT INPUT & PROCESSING LOGIC ---
chat_input_value = st.chat_input("Ask a coding question...")

# Determine if the user clicked a button OR typed something
final_prompt = suggestion_clicked or chat_input_value

if final_prompt:
    # 1. Show user message
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(final_prompt)
    st.session_state.messages.append({"role": "user", "content": final_prompt})

    # 2. Show bot response
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        try:
            cleaned_prompt = clean_dl_text(final_prompt)
            vectorized_prompt = vec.transform([cleaned_prompt])
            full_response = model.predict(vectorized_prompt)[0]
            
            def stream_data(text, delay=0.02):
                for word in text.split(" "):
                    yield word + " "
                    time.sleep(delay)
            
            st.write_stream(stream_data(full_response))
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # 3. GENERATE 3 NEW RANDOM QUESTIONS
            # This triggers right after the bot finishes typing!
            if len(question_bank) >= 3:
                st.session_state.current_suggestions = random.sample(question_bank, 3)
            
            # 4. Refresh the UI to show the new buttons
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {e}")
